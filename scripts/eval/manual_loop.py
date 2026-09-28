#!/usr/bin/env python3
"""
Manual Control Loop for VLN-PE Zero-Shot Evaluation

This script runs the VLN-PE evaluation loop with manual control.
Control the robot by editing: logs/manual_control/action_command.json

Usage:
    python scripts/eval/manual_loop.py
    python scripts/eval/manual_loop.py --config scripts/eval/configs/manual_control_cfg.py
    python scripts/eval/manual_loop.py --max-steps 500
"""

import sys
sys.path.append('.')

import argparse
import importlib.util
import time
from pathlib import Path

from internnav.configs.evaluator.vln_default_config import get_config
from internnav.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Manual control loop for VLN-PE')
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/eval/configs/manual_control_cfg.py',
        help='Config file path for manual control',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000,
        help='Maximum number of steps to run (default: 1000)',
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Evaluation split to use (e.g., val_seen, val_unseen)',
    )
    parser.add_argument(
        '--scene',
        type=str,
        default=None,
        help='Specific scene ID to explore (e.g., 17DRP5sb8fy)',
    )
    parser.add_argument(
        '--episode',
        type=str,
        default=None,
        help='Specific episode/path_key to explore (e.g., 5593_571)',
    )
    parser.add_argument(
        '--list-episodes',
        action='store_true',
        help='List all available episodes and exit',
    )
    # Allow unknown args for compatibility
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Ignoring unknown arguments: {unknown}")
    return args


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    """Load evaluation config from Python file."""
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


def list_available_episodes(cfg):
    """List all available episodes for manual exploration."""
    from internnav.env.utils.episode_loader.resumable import ResumablePathKeyEpisodeloader
    from internnav.evaluator.utils.config import get_lmdb_path
    from pathlib import Path
    
    # Initialize dataloader
    task_name = cfg.task.task_name
    
    # Create dataloader with all required settings from config
    dataloader = ResumablePathKeyEpisodeloader(
        cfg.dataset.dataset_type, 
        **cfg.dataset.dataset_settings
    )
    
    print("\n" + "="*70)
    print("📋 AVAILABLE EPISODES FOR MANUAL EXPLORATION")
    print("="*70)
    print(f"Total episodes: {dataloader.size}\n")
    
    # Group by scene
    scene_episodes = {}
    for path_key in dataloader.resumed_path_key_list:
        scene_id = dataloader.path_key_scan[path_key]
        if scene_id not in scene_episodes:
            scene_episodes[scene_id] = []
        scene_episodes[scene_id].append(path_key)
    
    # Print organized by scene
    for scene_id in sorted(scene_episodes.keys()):
        episodes = scene_episodes[scene_id]
        print(f"🏠 Scene: {scene_id} ({len(episodes)} episodes)")
        for ep in episodes[:5]:  # Show first 5 episodes per scene
            print(f"   - {ep}")
        if len(episodes) > 5:
            print(f"   ... and {len(episodes) - 5} more episodes")
        print()
    
    print("="*70)
    print("\nUsage examples:")
    print(f"  # Explore specific episode:")
    first_episode = dataloader.resumed_path_key_list[0]
    print(f"  python scripts/eval/manual_loop.py --episode {first_episode}")
    print(f"\n  # Explore all episodes in a specific scene:")
    first_scene = list(scene_episodes.keys())[0]
    print(f"  python scripts/eval/manual_loop.py --scene {first_scene}")
    print("="*70 + "\n")


def filter_episodes(evaluator_cfg, scene_filter=None, episode_filter=None):
    """
    Filter episodes based on scene or episode ID.
    This modifies the dataloader to only include selected episodes.
    """
    if scene_filter is None and episode_filter is None:
        return  # No filtering needed
    
    # We'll use a custom filter by modifying the dataset settings
    print(f"\n🔍 Filtering episodes...")
    if episode_filter:
        print(f"   Episode filter: {episode_filter}")
        # Set filter to only include this specific episode
        evaluator_cfg.dataset.dataset_settings['episode_filter'] = episode_filter
    elif scene_filter:
        print(f"   Scene filter: {scene_filter}")
        # Set filter to only include episodes from this scene
        evaluator_cfg.dataset.dataset_settings['scene_filter'] = scene_filter
    print()


def print_instructions(action_file: Path):
    """Print user instructions."""
    print("\n" + "="*70)
    print("🎮 MANUAL CONTROL MODE")
    print("="*70)
    print(f"\n📝 Control file: {action_file}")
    print("\nTo control the robot, edit the action_command.json file:")
    print("  - [forward, lateral, yaw_rate]")
    print("  - forward: positive = forward, negative = backward")
    print("  - lateral: positive = left, negative = right")
    print("  - yaw_rate: positive = turn left, negative = turn right")
    print("\nExample commands:")
    print('  {"action": [0.5, 0.0, 0.0]}   # Move forward')
    print('  {"action": [0.0, 0.0, 0.3]}   # Turn left')
    print('  {"action": [0.0, 0.0, -0.3]}  # Turn right')
    print('  {"action": [0.0, 0.0, 0.0]}   # Stop')
    print("\n📸 RGB frames are saved to: manual_logs/manual_control/frames/")
    print("\nPress Ctrl+C to stop the simulation.")
    print("="*70 + "\n")


def main():
    args = parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    evaluator_cfg = load_eval_cfg(args.config, attr_name='eval_cfg')
    
    # Override split if specified
    if args.split:
        print(f"🎯 Setting evaluation split to: {args.split}")
        evaluator_cfg.dataset.dataset_settings['split_data_types'] = [args.split]
    
    # Filter by scene or episode if specified (before get_config)
    filter_episodes(evaluator_cfg, scene_filter=args.scene, episode_filter=args.episode)
    
    # Get final config (this adds missing dataset_settings fields)
    cfg = get_config(evaluator_cfg)
    
    # List episodes mode (after get_config so all fields are populated)
    if args.list_episodes:
        list_available_episodes(cfg)
        return
    
    # Print action file location
    action_file = Path(cfg.agent.model_settings.get('output_dir', 'logs/manual_control')) / 'action_command.json'
    print_instructions(action_file)
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = Evaluator.init(cfg)
    
    # Give user time to see the instructions
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    try:
        # Run evaluation loop
        # The ManualAgent will handle reading actions and saving frames
        evaluator.eval()
    except KeyboardInterrupt:
        print("\n\n⛔ Interrupted by user. Shutting down...")
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✅ Manual control session ended.")
        print(f"   Check frames in: manual_logs/manual_control/frames/")


if __name__ == '__main__':
    main()
