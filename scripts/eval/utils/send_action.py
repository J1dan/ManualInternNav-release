#!/usr/bin/env python3
"""
Helper script to send action commands to the manual control agent.

Usage:
    # Move forward
    python scripts/utils/send_action.py --forward 0.5
    
    # Turn left
    python scripts/utils/send_action.py --yaw 0.3
    
    # Turn right
    python scripts/utils/send_action.py --yaw -0.3
    
    # Move forward and turn left simultaneously
    python scripts/utils/send_action.py --forward 0.5 --yaw 0.3
    
    # Stop
    python scripts/utils/send_action.py --stop
    
    # Custom action
    python scripts/utils/send_action.py --action 0.5 0.0 0.2
"""

import argparse
import json
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Send action commands to manual control agent')
    parser.add_argument(
        '--action-file',
        type=str,
        default='manual_logs/manual_control/action_command.json',
        help='Path to action command file',
    )
    parser.add_argument(
        '--forward',
        type=float,
        default=None,
        help='Forward speed (positive=forward, negative=backward)',
    )
    parser.add_argument(
        '--lateral',
        type=float,
        default=None,
        help='Lateral speed (positive=left, negative=right)',
    )
    parser.add_argument(
        '--yaw',
        type=float,
        default=None,
        help='Yaw rate (positive=turn left, negative=turn right)',
    )
    parser.add_argument(
        '--stop',
        action='store_true',
        help='Send stop command (zeros all velocities)',
    )
    parser.add_argument(
        '--action',
        type=float,
        nargs=3,
        metavar=('FORWARD', 'LATERAL', 'YAW'),
        help='Direct action command [forward, lateral, yaw]',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Duration in seconds to keep sending the action (default: send once)',
    )
    return parser.parse_args()


def send_action(action_file: Path, forward: float, lateral: float, yaw: float):
    """Write action command to JSON file."""
    action_data = {
        "action": [forward, lateral, yaw],
        "timestamp": time.time(),
        "description": "Control format: [forward_speed, lateral_speed, yaw_rate]"
    }
    
    with open(action_file, 'w') as f:
        json.dump(action_data, f, indent=2)
    
    print(f"✅ Action sent: forward={forward:.2f}, lateral={lateral:.2f}, yaw={yaw:.2f}")
    print(f"   File: {action_file}")


def main():
    args = parse_args()
    action_file = Path(args.action_file)
    
    # Ensure directory exists
    action_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine action
    if args.stop:
        forward, lateral, yaw = 0.0, 0.0, 0.0
        print("🛑 Sending STOP command")
    elif args.action:
        forward, lateral, yaw = args.action
        print(f"📤 Sending custom action")
    else:
        # Build action from individual components
        # Read current action if it exists
        current_action = [0.0, 0.0, 0.0]
        if action_file.exists():
            try:
                with open(action_file, 'r') as f:
                    data = json.load(f)
                    current_action = data.get('action', [0.0, 0.0, 0.0])
            except (json.JSONDecodeError, KeyError):
                pass
        
        forward = args.forward if args.forward is not None else current_action[0]
        lateral = args.lateral if args.lateral is not None else current_action[1]
        yaw = args.yaw if args.yaw is not None else current_action[2]
        
        print(f"📤 Sending modified action")
    
    # Send action with optional duration
    if args.duration:
        print(f"⏱️  Duration: {args.duration}s (will keep resending)")
        start_time = time.time()
        count = 0
        while time.time() - start_time < args.duration:
            send_action(action_file, forward, lateral, yaw)
            count += 1
            time.sleep(0.1)  # Resend every 100ms
        print(f"✅ Sent {count} times over {args.duration}s")
    else:
        # Send once
        send_action(action_file, forward, lateral, yaw)
    
    # Print helpful info
    print("\n💡 Tips:")
    print("   - Forward speed: 0.0 to 1.0 (typical: 0.3-0.7)")
    print("   - Yaw rate: -0.5 to 0.5 (typical: 0.2-0.4)")
    print("   - Lateral speed: -0.3 to 0.3 (less commonly used)")
    print("   - Add --duration 1.0 to keep sending for 1 second")


if __name__ == '__main__':
    main()
