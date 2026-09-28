#!/usr/bin/env python3
"""
ImagiNav Evaluation Loop for VLN-PE

This script runs the VLN-PE evaluation loop with ImagiNav agent.
The agent uses: Gemini (Reasoning) → LTX-Video (Imagination) → VGGT (Navigation)

Usage:
    python scripts/eval/imaginav_loop.py
    python scripts/eval/imaginav_loop.py --config scripts/eval/configs/imaginav_eval_cfg.py
    python scripts/eval/imaginav_loop.py --episode 5593_571
    python scripts/eval/imaginav_loop.py --scene 17DRP5sb8fy
    python scripts/eval/imaginav_loop.py --list-episodes
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
    parser = argparse.ArgumentParser(description='ImagiNav evaluation loop for VLN-PE')
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/eval/configs/imaginav_eval_cfg.py',
        help='Config file path for ImagiNav evaluation',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000,
        help='Maximum number of steps per episode (default: 1000)',
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
        help='Specific scene ID to evaluate (e.g., 17DRP5sb8fy)',
    )
    parser.add_argument(
        '--episode',
        type=str,
        default=None,
        help='Specific episode/path_key to evaluate (e.g., 5593_571)',
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
    """List all available episodes for evaluation."""
    from internnav.env.utils.episode_loader.resumable import ResumablePathKeyEpisodeloader
    
    try:
        # Initialize dataloader
        dataloader = ResumablePathKeyEpisodeloader(
            cfg.dataset.dataset_type, 
            **cfg.dataset.dataset_settings
        )
    except FileNotFoundError as e:
        print("\n" + "="*70)
        print("❌ Dataset not found")
        print("="*70)
        print(f"\nError: {e}")
        print("\nPlease ensure the VLN-PE dataset is downloaded and available at:")
        print(f"  {cfg.dataset.dataset_settings.get('base_data_dir', 'data/vln_pe/raw_data/r2r')}")
        print("\nRefer to InternNav documentation for dataset setup instructions.")
        print("="*70 + "\n")
        return
    
    print("\n" + "="*70)
    print("📋 AVAILABLE EPISODES FOR IMAGINAV EVALUATION")
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
    print(f"  # Evaluate specific episode:")
    first_episode = dataloader.resumed_path_key_list[0]
    print(f"  python scripts/eval/imaginav_loop.py --episode {first_episode}")
    print(f"\n  # Evaluate all episodes in a specific scene:")
    first_scene = list(scene_episodes.keys())[0]
    print(f"  python scripts/eval/imaginav_loop.py --scene {first_scene}")
    print("="*70 + "\n")


def filter_episodes(evaluator_cfg, scene_filter=None, episode_filter=None):
    """
    Filter episodes based on scene or episode ID.
    This modifies the dataloader to only include selected episodes.
    """
    if scene_filter is None and episode_filter is None:
        return  # No filtering needed
    
    print(f"\n🔍 Filtering episodes...")
    if episode_filter:
        # Special handling for InteriorNav/Kujiale where trajectory_id == episode_id
        # If user passes single ID "179", convert to "179_179"
        dataset_type = getattr(evaluator_cfg.dataset, 'dataset_type', '')
        if dataset_type == 'kujiale' and '_' not in str(episode_filter):
             print(f"   ℹ️  InteriorNav detected: Converting episode '{episode_filter}' to '{episode_filter}_{episode_filter}'")
             episode_filter = f"{episode_filter}_{episode_filter}"

        print(f"   Episode filter: {episode_filter}")
        evaluator_cfg.dataset.dataset_settings['episode_filter'] = episode_filter
    elif scene_filter:
        print(f"   Scene filter: {scene_filter}")
        evaluator_cfg.dataset.dataset_settings['scene_filter'] = scene_filter
    print()


def print_instructions(task_name: str):
    """Print ImagiNav pipeline information."""
    print("\n" + "="*70)
    print("🎬 IMAGINAV EVALUATION MODE")
    print("="*70)
    print("\n📋 Pipeline stages:")
    print("   1️⃣  Gemini Reasoning: Generate video prompt from instruction")
    print("   2️⃣  LTX-Video Generation: Create imagined first-person video")
    print("   3️⃣  VGGT Navigation: Extract camera poses → velocity commands")
    print("\n⏱️  Expected timing per navigation decision:")
    print("   - Gemini reasoning: ~1-2 seconds")
    print("   - LTX-Video generation: ~30-60 seconds")
    print("   - VGGT trajectory extraction: ~2-4 minutes")
    print("   - Total per step: ~3-5 minutes")
    print("\n📂 Outputs saved to:")
    print(f"   - Generated videos: logs/{task_name}/<timestamp>/videos/")
    print(f"   - Trajectories: logs/{task_name}/<timestamp>/trajectories/")
    print(f"   - Evaluation metrics: logs/{task_name}/results/")
    print("\nPress Ctrl+C to stop the evaluation.")
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
    
    # Print pipeline instructions
    print_instructions(cfg.task.task_name)
    
    # Initialize evaluator
    print("🚀 Initializing ImagiNav evaluator...")
    print("   This will load:")
    print("   - Gemini model")
    print("   - LTX-Video pipeline")
    print("   - VGGT navigation model")
    print()
    
    evaluator = Evaluator.init(cfg)
    
    # Give user time to see the instructions
    print("Starting evaluation in 3 seconds...")
    time.sleep(3)
    
    try:
        # Run evaluation loop
        # The ImagiNavAgent will handle the full pipeline
        evaluator.eval()
    except KeyboardInterrupt:
        print("\n\n⛔ Interrupted by user. Shutting down...")
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✅ ImagiNav evaluation session ended.")
        print(f"   Check outputs in: logs/{cfg.task.task_name}/")


if __name__ == '__main__':
    main()
