"""
Manual Zero-Shot Agent for VLN-PE
Enables manual control of the H1 robot with RGB frame capture.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image
from gym import spaces

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg
from internnav.evaluator.utils.common import obs_to_image


@Agent.register('manual_zero_shot')
class ManualAgent(Agent):
    """
    Manual control agent that:
    1. Captures and saves RGB frames
    2. Reads action commands from a JSON file
    3. Sends move_by_speed commands to the robot
    """
    
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 3),
        dtype=np.float32,
    )

    def __init__(self, agent_config: AgentCfg):
        super().__init__(agent_config)
        
        # Extract settings
        self.env_num = agent_config.model_settings.get('env_num', 1)
        self.proc_num = agent_config.model_settings.get('proc_num', 1)
        self.total_envs = self.env_num * self.proc_num
        
        # Setup directories
        self.output_dir = Path(agent_config.model_settings.get('output_dir', 'logs/manual_control'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Episode tracking for directory organization
        self.current_episode_ids = ['unknown'] * self.total_envs
        self.episode_frames_dirs = {}  # Maps env_idx -> episode-specific frames dir
        
        # Action control file (where you'll write commands)
        self.action_file = self.output_dir / 'action_command.json'
        
        # Do NOT initialize action file - wait for user to send first action
        # This prevents automatic execution of default [0.0, 0.0, 0.0]
        
        # State tracking
        self.step_count = 0
        self.action_sent = False  # Track if user has sent any action
        self.waiting_for_action = True  # Currently waiting for action
        self.last_action_step = 0  # Step when last action was received
        
        print(f"✅ ManualAgent initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Action command file: {self.action_file}")
        print(f"   Edit {self.action_file} to control the robot")
        print(f"   RGB frames will be saved to: Episode_<id>/frames/")

    def _write_default_action(self):
        """Write default action file with instructions."""
        default_action = {
            "action": [0.0, 0.0, 0.0],
            "description": "Control format: [forward_speed, lateral_speed, yaw_rate]",
            "example_forward": [0.5, 0.0, 0.0],
            "example_turn_left": [0.0, 0.0, 0.3],
            "example_turn_right": [0.0, 0.0, -0.3],
            "timestamp": time.time()
        }
        with open(self.action_file, 'w') as f:
            json.dump(default_action, f, indent=2)

    def _read_action_command(self) -> List[float]:
        """
        Read action command from JSON file and DELETE it after reading.
        Each action is consumed once - no caching!
        Returns None if no action file exists.
        """
        # If action file doesn't exist, wait for user to create it
        if not self.action_file.exists():
            return None
        
        try:
            # Read the action
            with open(self.action_file, 'r') as f:
                data = json.load(f)
                action = data.get('action', None)
            
            # Validate action
            if action is not None and isinstance(action, list) and len(action) == 3:
                # DELETE the file immediately after reading
                # This ensures action is only executed once
                self.action_file.unlink()
                self.action_sent = True
                print(f"📝 Action consumed: forward={action[0]:.2f}, lateral={action[1]:.2f}, yaw={action[2]:.2f}")
                print(f"   🗑️  Action file deleted - send next action when ready")
                return action
            else:
                print(f"⚠️  Invalid action format in {self.action_file}, deleting file")
                self.action_file.unlink()
                return None
                
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Error reading action file: {e}")
            # Try to delete corrupted file
            if self.action_file.exists():
                self.action_file.unlink()
            return None

    def _get_episode_frames_dir(self, env_idx: int) -> Path:
        """Get or create episode-specific frames directory for an environment."""
        if env_idx not in self.episode_frames_dirs:
            episode_id = self.current_episode_ids[env_idx]
            episode_dir = self.output_dir / episode_id / 'frames'
            episode_dir.mkdir(parents=True, exist_ok=True)
            self.episode_frames_dirs[env_idx] = episode_dir
        return self.episode_frames_dirs[env_idx]
    
    def _save_rgb_frame(self, obs: Dict[str, Any], env_idx: int = 0, prefix: str = "step"):
        """Save RGB frame from observation to episode-specific directory."""
        try:
            # Extract RGB from observation
            if 'rgb' in obs:
                rgb = obs['rgb']
                
                # Convert to PIL Image
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                
                # Ensure correct format
                if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                    rgb = (rgb * 255).astype(np.uint8)
                
                img = Image.fromarray(rgb)
                
                # Get episode-specific frames directory
                frames_dir = self._get_episode_frames_dir(env_idx)
                
                # Save with step counter
                filename = f"{prefix}_env{env_idx:02d}_step{self.step_count:06d}.png"
                filepath = frames_dir / filename
                img.save(filepath)
                
                return filepath
        except Exception as e:
            print(f"⚠️  Error saving RGB frame: {e}")
            return None

    def reset(self, reset_ls=None, episode_ids=None):
        """Reset agent state for specified environments.
        
        Args:
            reset_ls: List of environment indices to reset
            episode_ids: List of episode IDs corresponding to reset environments
        """
        if reset_ls is not None and len(reset_ls) > 0:
            print(f"🔄 ManualAgent reset for envs: {reset_ls}")
            # Update episode IDs for reset environments
            if episode_ids is not None:
                for env_idx, episode_id in zip(reset_ls, episode_ids):
                    if env_idx < len(self.current_episode_ids):
                        self.current_episode_ids[env_idx] = episode_id
                        # Clear cached frames directory for this env
                        if env_idx in self.episode_frames_dirs:
                            del self.episode_frames_dirs[env_idx]
                        print(f"   📂 Episode {episode_id} assigned to env {env_idx}")
        else:
            print(f"🔄 ManualAgent reset all")
            self.step_count = 0
            # Clear action state on episode reset
            self.action_sent = False
            self.waiting_for_action = True
            self.last_action_step = 0
            # Clear episode tracking
            self.episode_frames_dirs.clear()
            # Delete action file so agent will wait for new action
            if self.action_file.exists():
                self.action_file.unlink()
                print(f"   🗑️  Deleted pending action file")
            print(f"   ⏸️  Ready for new episode - send first action")

    def step(self, obs_batch: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Process observations and return actions.
        NON-BLOCKING: Checks for action file and returns immediately.
        
        Args:
            obs_batch: List of observations, one per environment
            
        Returns:
            List of actions in format [[forward, lateral, yaw], ...]
            Returns [-1] when waiting for action (stand_still)
        """
        self.step_count += 1
        
        # Save RGB frame ONLY on first step or when new action is received
        if self.step_count == 1:
            print(f"\n📸 Saving initial RGB frame (step 1)...")
            for env_idx, obs in enumerate(obs_batch):
                filepath = self._save_rgb_frame(obs, env_idx, prefix="initial")
                if filepath and env_idx == 0:
                    print(f"   Saved: {filepath}")
            
            print(f"\n⏸️  Waiting for first action...")
            print(f"   Send action to: {self.action_file}")
            print(f"   Or run: python scripts/utils/send_action.py --forward 0.5")
            while not self.action_file.exists():
                time.sleep(2)
        # NON-BLOCKING: Try to read new action from file (if exists)
        action = self._read_action_command()
        
        if action is not None:
            # NEW ACTION RECEIVED
            self.waiting_for_action = False
            self.last_action_step = self.step_count
            
            # Save RGB frame when action is received
            print(f"📸 Saving RGB frame at action step {self.step_count}...")
            for env_idx, obs in enumerate(obs_batch):
                filepath = self._save_rgb_frame(obs, env_idx, prefix=f"action_{self.step_count}")
                if filepath and env_idx == 0:
                    print(f"   Saved: {filepath}")
            
            print(f"🎯 Step {self.step_count}: Executing action: {action}")
            
            # Return action for this step only
            return [action for _ in range(len(obs_batch))]
        
        else:
            # NO NEW ACTION - BLOCK and wait for action file (freeze robot)
            if not self.waiting_for_action:
                # Just finished executing previous action
                self.waiting_for_action = True
                print(f"⏸️  Step {self.step_count}: Action completed, waiting for next action...")
            
            # BLOCK until new action arrives (this freezes the robot)
            print(f"   Checking for action file: {self.action_file}")
            while not self.action_file.exists():
                time.sleep(0.5)
            
            # Action file found, read it
            action = self._read_action_command()
            if action is not None:
                self.waiting_for_action = False
                self.last_action_step = self.step_count
                
                # Save RGB frame when action is received
                print(f"📸 Saving RGB frame at action step {self.step_count}...")
                for env_idx, obs in enumerate(obs_batch):
                    filepath = self._save_rgb_frame(obs, env_idx, prefix=f"action_{self.step_count}")
                    if filepath and env_idx == 0:
                        print(f"   Saved: {filepath}")
                
                print(f"🎯 Step {self.step_count}: Executing action: {action}")
                return [action for _ in range(len(obs_batch))]
            else:
                # Fallback: return stand_still if file exists but couldn't be read
                return [[-1] for _ in range(len(obs_batch))]

    def save_frame_on_demand(self, obs_batch: List[Dict[str, Any]], filename_prefix: str = "manual"):
        """
        Manually save RGB frames from current observations.
        Call this method when you want to capture frames outside the normal flow.
        
        Args:
            obs_batch: List of observations
            filename_prefix: Prefix for saved filenames
        """
        print(f"\n📸 Saving frames on demand (prefix: {filename_prefix})...")
        saved_files = []
        for env_idx, obs in enumerate(obs_batch):
            filepath = self._save_rgb_frame(obs, env_idx, prefix=filename_prefix)
            if filepath:
                saved_files.append(filepath)
                print(f"   Saved: {filepath}")
        return saved_files
