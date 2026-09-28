#!/usr/bin/env python3
"""
Goal-to-Action Converter for Manual Control - Multi-Waypoint Support

Converts high-level navigation waypoints to low-level action sequences.
Each waypoint specifies (forward_distance, yaw_change) relative to the previous pose.

Usage:
    # Single waypoint: move forward 1m
    python scripts/utils/goal_to_actions.py --waypoints "[[1.0, 0]]"
    
    # Multiple waypoints (relative): forward 1m, turn 45°, forward 0.5m, turn -30°
    python scripts/utils/goal_to_actions.py --waypoints "[[1.0, 0], [0, 45], [0.5, 0], [0, -30]]"
    
    # From JSON file
    python scripts/utils/goal_to_actions.py --waypoints-file waypoints.json
    
    # Execute immediately
    python scripts/utils/goal_to_actions.py --waypoints "[[1.0, 45], [0.5, -30]]" --execute
    
    # Tune parameters
    python scripts/utils/goal_to_actions.py --waypoints "[[1.0, 0]]" --forward-speed 0.6 --rotation-speed 0.4
    
Waypoint Format:
    [forward_distance, yaw_change]
    - forward_distance: meters (positive=forward, negative=backward)
    - yaw_change: degrees (positive=turn left, negative=turn right)
    - Both are RELATIVE to the previous waypoint
    
Example waypoints.json:
    {
        "waypoints": [
            [1.0, 0],      # Move forward 1m
            [0, 45],       # Turn left 45°
            [0.5, 0],      # Move forward 0.5m
            [0, -30]       # Turn right 30°
        ]
    }
"""

import argparse
import json
import time
import math
from pathlib import Path
from typing import List, Tuple, Dict


class GoalToActionConverter:
    """Converts navigation goals to action sequences."""
    
    def __init__(
        self,
        forward_speed: float = 0.5,
        lateral_speed: float = 0.25,
        rotation_speed: float = 0.3,
        steps_per_action: int = 50,
        physics_hz: int = 100,
    ):
        """
        Initialize converter with tunable parameters.
        
        Args:
            forward_speed: Forward velocity (0.0-1.0)
            lateral_speed: Lateral velocity (0.0-0.5)
            rotation_speed: Rotation velocity (0.0-0.5 rad/s equivalent)
            steps_per_action: Number of physics steps per action
            physics_hz: Physics simulation frequency (Hz)
        """
        self.forward_speed = forward_speed
        self.lateral_speed = lateral_speed
        self.rotation_speed = rotation_speed
        self.steps_per_action = steps_per_action
        self.physics_hz = physics_hz
        
        # Calculate actual velocities (approximate)
        # These are rough estimates - actual depends on RL policy training
        self.dt = steps_per_action / physics_hz  # Time per action (seconds)
        self.forward_distance_per_action = forward_speed * 0.4 * self.dt  # ~0.1m per action
        self.rotation_per_action = rotation_speed * 30 * self.dt  # ~4.5 degrees per action
        
    def forward_goal(self, distance: float) -> List[Tuple[float, float, float]]:
        """
        Convert forward distance goal to actions.
        
        Args:
            distance: Distance to move forward (meters, positive=forward)
            
        Returns:
            List of (forward, lateral, yaw) actions
        """
        if abs(distance) < 0.01:
            return []
        
        # Calculate number of actions needed
        num_actions = int(abs(distance) / self.forward_distance_per_action)
        num_actions = max(1, num_actions)  # At least 1 action
        
        # Direction
        direction = 1.0 if distance > 0 else -1.0
        
        actions = []
        for _ in range(num_actions):
            actions.append((direction * self.forward_speed, 0.0, 0.0))
        
        return actions
    
    def yaw_goal(self, target_yaw: float, current_yaw: float = 0.0) -> List[Tuple[float, float, float]]:
        """
        Convert target yaw to rotation actions.
        
        Args:
            target_yaw: Target yaw angle in degrees (-180 to 180)
            current_yaw: Current yaw angle in degrees (default: 0)
            
        Returns:
            List of (forward, lateral, yaw) actions
        """
        # Calculate delta yaw (shortest path)
        delta_yaw = target_yaw - current_yaw
        
        # Normalize to -180 to 180
        while delta_yaw > 180:
            delta_yaw -= 360
        while delta_yaw < -180:
            delta_yaw += 360
        
        if abs(delta_yaw) < 2:  # Threshold: 2 degrees
            return []
        
        # Calculate number of actions needed
        num_actions = int(abs(delta_yaw) / self.rotation_per_action)
        num_actions = max(1, num_actions)
        
        # Direction (positive = left, negative = right)
        direction = 1.0 if delta_yaw > 0 else -1.0
        
        actions = []
        for _ in range(num_actions):
            actions.append((0.0, 0.0, direction * self.rotation_speed))
        
        return actions
    
    def strafe_goal(self, distance: float) -> List[Tuple[float, float, float]]:
        """
        Convert lateral distance goal to strafe actions.
        
        Args:
            distance: Distance to strafe (meters, positive=left, negative=right)
            
        Returns:
            List of (forward, lateral, yaw) actions
        """
        if abs(distance) < 0.01:
            return []
        
        # Lateral movement is less stable, use conservative estimate
        lateral_distance_per_action = self.lateral_speed * 0.2 * self.dt
        
        num_actions = int(abs(distance) / lateral_distance_per_action)
        num_actions = max(1, num_actions)
        
        direction = 1.0 if distance > 0 else -1.0
        
        actions = []
        for _ in range(num_actions):
            actions.append((0.0, direction * self.lateral_speed, 0.0))
        
        return actions
    
    def waypoints_to_actions(
        self,
        waypoints: List[Tuple[float, float]],
        rotate_first: bool = True,
        use_arcs: bool = True,
    ) -> List[Tuple[float, float, float]]:
        """
        Convert a sequence of relative waypoints to actions.
        
        Each waypoint is [forward_distance, yaw_change] relative to previous pose.
        
        Args:
            waypoints: List of [forward_distance, yaw_change] pairs
                - forward_distance: meters to move forward (positive) or backward (negative)
                - yaw_change: degrees to turn left (positive) or right (negative)
            rotate_first: If True, rotate then move for each waypoint
            use_arcs: If True, combine rotation and forward into arc movements
            
        Returns:
            List of (forward, lateral, yaw) actions
        """
        all_actions = []
        current_yaw = 0.0  # Track absolute yaw orientation
        
        for i, (forward_dist, yaw_change) in enumerate(waypoints):
            waypoint_actions = []
            print("use_arcs:", use_arcs)
            if use_arcs and forward_dist != 0 and yaw_change != 0:
                # Arc movement: turn while moving
                waypoint_actions = self.arc_goal(forward_dist, yaw_change)
                current_yaw += yaw_change
            elif rotate_first:
                # Rotate first, then move
                if yaw_change != 0:
                    target_yaw = current_yaw + yaw_change
                    waypoint_actions.extend(self.yaw_goal(target_yaw, current_yaw))
                    current_yaw = target_yaw
                
                if forward_dist != 0:
                    waypoint_actions.extend(self.forward_goal(forward_dist))
            else:
                # Move first, then rotate
                if forward_dist != 0:
                    waypoint_actions.extend(self.forward_goal(forward_dist))
                
                if yaw_change != 0:
                    target_yaw = current_yaw + yaw_change
                    waypoint_actions.extend(self.yaw_goal(target_yaw, current_yaw))
                    current_yaw = target_yaw
            
            # Add waypoint marker comment
            if waypoint_actions:
                # Store waypoint info for printing
                for action in waypoint_actions:
                    all_actions.append(action)
        
        return all_actions
    
    def print_waypoint_plan(
        self,
        waypoints: List[Tuple[float, float]],
        actions: List[Tuple[float, float, float]],
    ):
        """Print waypoint sequence and resulting actions."""
        print(f"\n{'='*70}")
        print(f"Waypoint Plan")
        print(f"{'='*70}")
        print(f"Number of waypoints: {len(waypoints)}")
        print(f"Total actions: {len(actions)}")
        print(f"Estimated duration: {len(actions) * self.dt:.2f}s")
        print(f"{'='*70}\n")
        
        # Print waypoints
        print("Waypoints (relative):")
        cumulative_yaw = 0.0
        cumulative_forward = 0.0
        
        for i, (fwd, yaw) in enumerate(waypoints, 1):
            cumulative_forward += fwd
            cumulative_yaw += yaw
            
            parts = []
            if fwd != 0:
                direction = "forward" if fwd > 0 else "backward"
                parts.append(f"{direction} {abs(fwd):.2f}m")
            if yaw != 0:
                direction = "left" if yaw > 0 else "right"
                parts.append(f"turn {direction} {abs(yaw):.1f}°")
            
            desc = ", ".join(parts) if parts else "no change"
            print(f"  {i:2d}. [{fwd:6.2f}m, {yaw:6.1f}°]  →  {desc}")
            print(f"      Cumulative: forward={cumulative_forward:6.2f}m, yaw={cumulative_yaw:6.1f}°")
        
        print(f"\n{'='*70}\n")
    
    def arc_goal(
        self,
        forward: float,
        yaw_delta: float,
    ) -> List[Tuple[float, float, float]]:
        """
        Create arc trajectory (move forward while turning).
        
        Args:
            forward: Forward distance (meters)
            yaw_delta: Change in yaw while moving (degrees)
            
        Returns:
            List of (forward, lateral, yaw) actions
        """
        if abs(forward) < 0.01:
            return []
        
        num_actions = int(abs(forward) / self.forward_distance_per_action)
        num_actions = max(1, num_actions)
        
        # Distribute rotation across forward movement
        rotation_per_step = (yaw_delta / self.rotation_per_action) / num_actions
        rotation_speed_scaled = rotation_per_step * self.rotation_speed
        
        # Clamp rotation speed
        rotation_speed_scaled = max(-self.rotation_speed, min(self.rotation_speed, rotation_speed_scaled))
        
        forward_direction = 1.0 if forward > 0 else -1.0
        
        actions = []
        for _ in range(num_actions):
            actions.append((
                forward_direction * self.forward_speed,
                0.0,
                rotation_speed_scaled
            ))
        
        return actions
    
    def print_action_plan(self, actions: List[Tuple[float, float, float]]):
        """Print the action sequence in human-readable format."""
        if not actions:
            print("No actions needed (already at goal)")
            return
        
        print(f"\n{'='*70}")
        print(f"Action Plan ({len(actions)} actions)")
        print(f"{'='*70}")
        print(f"Estimated duration: {len(actions) * self.dt:.2f}s")
        print(f"{'='*70}\n")
        
        for i, (fwd, lat, yaw) in enumerate(actions, 1):
            action_type = []
            if abs(fwd) > 0.01:
                direction = "forward" if fwd > 0 else "backward"
                action_type.append(f"{direction} {abs(fwd):.2f}")
            if abs(lat) > 0.01:
                direction = "left" if lat > 0 else "right"
                action_type.append(f"strafe {direction} {abs(lat):.2f}")
            if abs(yaw) > 0.01:
                direction = "left" if yaw > 0 else "right"
                action_type.append(f"turn {direction} {abs(yaw):.2f}")
            
            action_desc = ", ".join(action_type) if action_type else "stand still"
            print(f"  {i:3d}. [{fwd:5.2f}, {lat:5.2f}, {yaw:5.2f}]  # {action_desc}")
        
        print(f"\n{'='*70}\n")


def execute_actions(
    actions: List[Tuple[float, float, float]],
    action_file: Path,
    delay: float = 0.3,
):
    """
    Execute action sequence by sending to manual control.
    Waits for each action to be consumed before sending next.
    
    Args:
        actions: List of (forward, lateral, yaw) actions
        action_file: Path to action command file
        delay: Additional delay between actions (seconds) - added AFTER action is consumed
    """
    print(f"\n🎮 Executing {len(actions)} actions...")
    print(f"   Action file: {action_file}")
    print(f"   Delay after consumption: {delay}s")
    print(f"   Estimated total time: ~{len(actions) * (delay + 0.5):.1f}s")
    print(f"   (0.5s/action execution time)\n")
    
    for i, (fwd, lat, yaw) in enumerate(actions, 1):
        # Wait for previous action to be consumed (file deleted)
        if action_file.exists():
            print(f"  [{i:3d}/{len(actions)}] ⏳ Waiting for previous action to be consumed...")
            wait_start = time.time()
            timeout = 15.0  # 10 second timeout
            
            while action_file.exists():
                if time.time() - wait_start > timeout:
                    print(f"  ⚠️  Timeout waiting for action consumption! Deleting stale file...")
                    action_file.unlink()
                    break
                time.sleep(0.1)  # Check every 100ms
            
            # Additional delay for action execution
            time.sleep(delay)
        
        # Write new action
        action_data = {
            "action": [fwd, lat, yaw],
            "timestamp": time.time(),
            "sequence": i,
            "total": len(actions),
        }
        
        with open(action_file, 'w') as f:
            json.dump(action_data, f, indent=2)
        
        print(f"  [{i:3d}/{len(actions)}] ✅ Sent: [{fwd:5.2f}, {lat:5.2f}, {yaw:5.2f}]")
    
    # Wait for last action to be consumed
    print(f"\n⏳ Waiting for final action to complete...")
    wait_start = time.time()
    while action_file.exists():
        if time.time() - wait_start > 10.0:
            print(f"⚠️  Timeout, but action may still complete")
            break
        time.sleep(0.1)
    
    print(f"\n✅ All {len(actions)} actions sent and consumed!")
    print(f"   Check saved frames in: manual_logs/manual_control_eval/video/")



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert navigation waypoints to action sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single waypoint: move forward 1m
  python scripts/utils/goal_to_actions.py --waypoints "[[1.0, 0]]"
  
  # Multiple waypoints: forward 1m, turn 45°, forward 0.5m, turn -30°
  python scripts/utils/goal_to_actions.py --waypoints "[[1.0, 0], [0, 45], [0.5, 0], [0, -30]]"
  
  # Square path: forward, turn 90°, repeat 4 times
  python scripts/utils/goal_to_actions.py --waypoints "[[1, 0], [0, 90], [1, 0], [0, 90], [1, 0], [0, 90], [1, 0], [0, 90]]"
  
  # From JSON file
  python scripts/utils/goal_to_actions.py --waypoints-file path/to/waypoints.json
  
  # Disable arc movements
  python scripts/utils/goal_to_actions.py --waypoints "[[1.0, 45], [0.5, -30]]" --no-arcs
  
  # Execute immediately
  python scripts/utils/goal_to_actions.py --waypoints "[[1.0, 0], [0, 90]]" --execute
  
  # Custom speeds
  python scripts/utils/goal_to_actions.py --waypoints "[[1.0, 0]]" --forward-speed 0.7 --rotation-speed 0.4
        """
    )
    
    # Waypoints input
    waypoint_group = parser.add_mutually_exclusive_group(required=True)
    waypoint_group.add_argument('--waypoints', type=str,
                       help='Waypoints as JSON array: [[forward1, yaw1], [forward2, yaw2], ...]')
    waypoint_group.add_argument('--waypoints-file', type=str,
                       help='Load waypoints from JSON file')
    
    # Movement strategy
    parser.add_argument('--rotate-first', action='store_true', default=True,
                       help='Rotate before moving at each waypoint (default: True)')
    parser.add_argument('--move-first', dest='rotate_first', action='store_false',
                       help='Move before rotating at each waypoint')
    parser.add_argument('--no-arcs', action='store_false', dest='use_arcs', default=True,
                       help='Disable arc movements (turn while moving) instead of separate rotate+move')
    
    # Execution
    parser.add_argument('--execute', action='store_true',
                       help='Execute actions immediately')
    parser.add_argument('--delay', type=float, default=0.3,
                       help='Delay between actions when executing (seconds)')
    
    # Tunable parameters
    parser.add_argument('--forward-speed', type=float, default=0.5,
                       help='Forward speed (0.0-1.0, default: 0.5)')
    parser.add_argument('--lateral-speed', type=float, default=0.25,
                       help='Lateral speed (0.0-0.5, default: 0.25)')
    parser.add_argument('--rotation-speed', type=float, default=0.3,
                       help='Rotation speed (0.0-0.5, default: 0.3)')
    parser.add_argument('--steps-per-action', type=int, default=50,
                       help='Physics steps per action (default: 50)')
    parser.add_argument('--physics-hz', type=int, default=200,
                       help='Physics simulation frequency (default: 200Hz)')
    
    # Output
    parser.add_argument('--action-file', type=str,
                       default='manual_logs/manual_control/action_command.json',
                       help='Path to action command file')
    parser.add_argument('--save-plan', type=str, default=None,
                       help='Save action plan to JSON file')
    parser.add_argument('--save-actions', type=str, default=None,
                       help='Save raw actions to JSON file')
    
    return parser.parse_args()


def load_waypoints_from_file(filepath: str) -> List[Tuple[float, float]]:
    """Load waypoints from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'waypoints' in data:
        waypoints = data['waypoints']
    elif isinstance(data, list):
        waypoints = data
    else:
        raise ValueError("JSON must contain 'waypoints' key or be a list")
    
    return [(float(w[0]), float(w[1])) for w in waypoints]


def main():
    args = parse_args()
    
    # Load waypoints
    if args.waypoints:
        waypoints = json.loads(args.waypoints)
        waypoints = [(float(w[0]), float(w[1])) for w in waypoints]
    else:
        waypoints = load_waypoints_from_file(args.waypoints_file)
    
    # Validate waypoints
    if not waypoints:
        print("❌ Error: No waypoints provided!")
        return
    
    # Create converter with tuned parameters
    converter = GoalToActionConverter(
        forward_speed=args.forward_speed,
        lateral_speed=args.lateral_speed,
        rotation_speed=args.rotation_speed,
        steps_per_action=args.steps_per_action,
        physics_hz=args.physics_hz,
    )
    
    # Print parameters
    print("\n" + "="*70)
    print("Goal-to-Action Converter - Multi-Waypoint")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Forward speed:      {args.forward_speed:.2f}")
    print(f"  Lateral speed:      {args.lateral_speed:.2f}")
    print(f"  Rotation speed:     {args.rotation_speed:.2f}")
    print(f"  Steps per action:   {args.steps_per_action}")
    print(f"  Physics frequency:  {args.physics_hz} Hz")
    print(f"  Time per action:    {converter.dt:.3f}s")
    print(f"\nStrategy:")
    print(f"  Rotate first:       {args.rotate_first}")
    print(f"  Use arc movements:  {args.use_arcs}")
    
    # Generate actions from waypoints
    actions = converter.waypoints_to_actions(
        waypoints=waypoints,
        rotate_first=args.rotate_first,
        use_arcs=args.use_arcs,
    )
    
    # Print plan
    converter.print_waypoint_plan(waypoints, actions)
    
    if actions:
        converter.print_action_plan(actions)
    
    # Save plan
    if args.save_plan:
        plan_data = {
            "waypoints": waypoints,
            "parameters": {
                "forward_speed": args.forward_speed,
                "lateral_speed": args.lateral_speed,
                "rotation_speed": args.rotation_speed,
                "steps_per_action": args.steps_per_action,
                "physics_hz": args.physics_hz,
                "rotate_first": args.rotate_first,
                "use_arcs": args.use_arcs,
            },
            "num_actions": len(actions),
            "estimated_duration": len(actions) * converter.dt,
        }
        
        with open(args.save_plan, 'w') as f:
            json.dump(plan_data, f, indent=2)
        print(f"💾 Saved plan to: {args.save_plan}")
    
    # Save raw actions
    if args.save_actions:
        actions_data = {
            "actions": [[fwd, lat, yaw] for fwd, lat, yaw in actions],
            "format": "[forward, lateral, yaw]",
        }
        
        with open(args.save_actions, 'w') as f:
            json.dump(actions_data, f, indent=2)
        print(f"💾 Saved actions to: {args.save_actions}")
    
    # Execute if requested
    if args.execute:
        if not actions:
            print("⚠️  No actions to execute!")
            return
        
        action_file = Path(args.action_file)
        action_file.parent.mkdir(parents=True, exist_ok=True)
        
        print("\n⏳ Starting execution in 3 seconds...")
        print("   Press Ctrl+C to cancel")
        try:
            time.sleep(3)
            execute_actions(actions, action_file, args.delay)
        except KeyboardInterrupt:
            print("\n\n⛔ Execution cancelled by user")
    else:
        print("\n💡 Tip: Add --execute to run these actions immediately")
        print(f"   Or save and execute later with --save-actions output.json")


if __name__ == '__main__':
    main()
