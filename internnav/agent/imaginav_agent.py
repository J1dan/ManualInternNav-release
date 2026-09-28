"""
ImagiNav Agent for InternNav
Integrates the ImagiNav pipeline (Gemini + LTX-Video + VGGT) with InternNav simulation.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from gym import spaces

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg

# Add ImagiNav to path
IMAGINAV_PATH = Path(__file__).parent.parent.parent / 'ImagiNav'
IMAGINAV_PARENT = IMAGINAV_PATH.parent
if str(IMAGINAV_PARENT) not in sys.path:
    sys.path.insert(0, str(IMAGINAV_PARENT))

# Import ImagiNav components
try:
    from ImagiNav.core.agent import ImagiNavAgent
    from ImagiNav.core.config import ImagiNavConfig
    from ImagiNav.core.types import NavigationCommand
    IMAGINAV_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import ImagiNav: {e}")
    print(f"   Make sure ImagiNav is installed or accessible at: {IMAGINAV_PATH}")
    IMAGINAV_AVAILABLE = False


@Agent.register('imaginav')
class ImagiNavInternNavAgent(Agent):
    """
    ImagiNav agent that integrates with InternNav simulation.
    
    Pipeline:
    1. Receives observation (RGB image + instruction) from InternNav
    2. On first step, calls ImagiNav to generate full trajectory
    3. Returns actions sequentially from the generated trajectory
    4. When trajectory is exhausted, generates a new one
    """
    
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 3),
        dtype=np.float32,
    )

    def __init__(self, agent_config: AgentCfg):
        super().__init__(agent_config)
        
        if not IMAGINAV_AVAILABLE:
            raise ImportError(
                "ImagiNav is not available. Please install ImagiNav or check the path: "
                f"{IMAGINAV_PATH}"
            )
        
        # Extract settings
        self.env_num = agent_config.model_settings.get('env_num', 1)
        self.proc_num = agent_config.model_settings.get('proc_num', 1)
        self.total_envs = self.env_num * self.proc_num
        
        # Setup directories
        self.output_dir = Path(agent_config.model_settings.get('output_dir', 'logs/imaginav'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ImagiNav config path
        config_path = agent_config.model_settings.get(
            'imaginav_config',
            str(IMAGINAV_PATH / 'configs' / 'default_config.yaml')
        )
        
        # Load ImagiNav agent
        print(f"🚀 Loading ImagiNav agent...")
        print(f"   Config: {config_path}")
        self.imaginav_config = ImagiNavConfig.from_yaml(config_path)
        
        # Override logging directory so ImagiNav logs are stored under the
        # task output directory (e.g., logs/TASK_NAME/ImagiNav-logs)
        imaginav_logs_dir = self.output_dir / 'ImagiNav-logs'
        imaginav_logs_dir.mkdir(parents=True, exist_ok=True)
        # The ImagiNav config uses the `logs_dir` key for logging location
        self.imaginav_config.logging.logs_dir = str(imaginav_logs_dir)
        print(f"   ImagiNav logs: {imaginav_logs_dir}")
        
        self.imaginav_agent = ImagiNavAgent(self.imaginav_config)
        print(f"✅ ImagiNav agent loaded successfully")
        
        # State tracking per environment
        self.current_instructions = [None] * self.total_envs
        self.trajectory_commands = [None] * self.total_envs  # List of NavigationCommand
        self.command_indices = [0] * self.total_envs
        self.step_counts = [0] * self.total_envs
        self.episode_ids = ['unknown'] * self.total_envs
        
        # Frame saving
        self.save_frames = agent_config.model_settings.get('save_frames', True)
        self.frames_dir = self.output_dir / 'frames'
        self.frames_dir.mkdir(exist_ok=True)
        
        print(f"   Output directory: {self.output_dir}")
        print(f"   Frames directory: {self.frames_dir}")

    def reset(self, reset_ls: Optional[List[int]] = None, episode_ids: Optional[List[str]] = None):
        """
        Reset agent state for specified environments.
        
        Args:
            reset_ls: List of environment indices to reset
            episode_ids: List of episode IDs corresponding to reset environments
        """
        if reset_ls is not None and len(reset_ls) > 0:
            print(f"🔄 ImagiNavAgent reset for envs: {reset_ls}")
            for i, env_idx in enumerate(reset_ls):
                if env_idx < self.total_envs:
                    self.current_instructions[env_idx] = None
                    self.trajectory_commands[env_idx] = None
                    self.command_indices[env_idx] = 0
                    self.step_counts[env_idx] = 0
                    if episode_ids and i < len(episode_ids):
                        self.episode_ids[env_idx] = episode_ids[i]
                        print(f"   📂 Episode {episode_ids[i]} assigned to env {env_idx}")
        else:
            print(f"🔄 ImagiNavAgent reset all")
            self.current_instructions = [None] * self.total_envs
            self.trajectory_commands = [None] * self.total_envs
            self.command_indices = [0] * self.total_envs
            self.step_counts = [0] * self.total_envs

    def step(self, obs_batch: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Process observations and return actions.
        
        Args:
            obs_batch: List of observations, one per environment
                Each obs should contain:
                - 'rgb': RGB image (H, W, 3) as numpy array or torch tensor
                - 'instruction': Navigation instruction text
            
        Returns:
            List of actions in format [[forward, lateral, yaw], ...]
        """
        actions = []
        
        for env_idx, obs in enumerate(obs_batch):
            self.step_counts[env_idx] += 1
            
            # Extract instruction
            instruction = obs.get('instruction', {})
            if isinstance(instruction, dict):
                instruction_text = instruction.get('instruction', '')
            else:
                instruction_text = str(instruction)
            
            # Check if we need to generate a new trajectory
            need_new_trajectory = (
                self.trajectory_commands[env_idx] is None or
                self.command_indices[env_idx] >= len(self.trajectory_commands[env_idx]) or
                self.current_instructions[env_idx] != instruction_text
            )
            
            if need_new_trajectory:
                print(f"\n🎬 Env {env_idx}: Generating new trajectory (step {self.step_counts[env_idx]})")
                print(f"   Instruction: {instruction_text[:100]}...")
                
                # Save current RGB frame
                image_path = self._save_observation_image(obs, env_idx)
                
                if image_path is None:
                    print(f"⚠️  Failed to save observation image, using stop action")
                    raise RuntimeError("Failed to save observation image")
                
                # Generate trajectory using ImagiNav
                try:
                    trajectory_output = self.imaginav_agent.navigate(
                        image_path=str(image_path),
                        instruction=instruction_text,
                        episode_id=self.episode_ids[env_idx],
                        step_id=self.step_counts[env_idx]
                    )
                    
                    # Extract navigation commands
                    self.trajectory_commands[env_idx] = trajectory_output.navigation_commands
                    self.command_indices[env_idx] = 0
                    self.current_instructions[env_idx] = instruction_text
                    
                    print(f"✅ Generated trajectory with {len(self.trajectory_commands[env_idx])} commands")
                    
                except Exception as e:
                    print(f"❌ Error generating trajectory: {e}")
                    import traceback
                    traceback.print_exc()
                    error_text = str(e)
                    # Abort evaluation for any Gemini/GenAI-related failure (quota, server errors, etc.)
                    if any(s in error_text for s in ["RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE", "Gemini"]):
                        raise RuntimeError(
                            "Gemini error encountered. Aborting evaluation immediately."
                        ) from e
                    raise RuntimeError("Failed to generate trajectory") from e
            
            # Get next command from trajectory
            if (self.trajectory_commands[env_idx] is not None and
                self.command_indices[env_idx] < len(self.trajectory_commands[env_idx])):
                
                cmd = self.trajectory_commands[env_idx][self.command_indices[env_idx]]
                self.command_indices[env_idx] += 1
                
                # Convert NavigationCommand to InternNav action format
                if hasattr(cmd, 'linear_velocity'):
                     action = [
                        cmd.linear_velocity,   # forward speed
                        cmd.lateral_velocity,  # lateral speed
                        cmd.angular_velocity   # yaw rate
                    ]
                     cmd_type = getattr(cmd, 'command_type', 'unknown')
                else:
                    # Assume list [v, lat, w]
                    action = [
                        cmd[0],
                        cmd[1],
                        cmd[2]
                    ]
                    cmd_type = "velocity"
                
                if self.command_indices[env_idx] % 10 == 0:
                    print(f"   Step {self.step_counts[env_idx]}: "
                          f"Command {self.command_indices[env_idx]}/{len(self.trajectory_commands[env_idx])} "
                          f"({cmd_type})")
                
                actions.append(action)
            else:
                # Trajectory exhausted, stop
                print(f"⏹️  Env {env_idx}: Trajectory complete")
                actions.append([-1])  # Stop action
        
        return actions

    def _save_observation_image(
        self,
        obs: Dict[str, Any],
        env_idx: int
    ) -> Optional[Path]:
        """
        Save RGB observation to file.
        
        Args:
            obs: Observation dictionary containing 'rgb'
            env_idx: Environment index
            
        Returns:
            Path to saved image, or None if failed
        """
        try:
            # Extract RGB from observation
            if 'rgb' not in obs:
                print(f"⚠️  No RGB in observation")
                return None
            
            rgb = obs['rgb']
            
            # Convert to numpy if tensor
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            
            # Ensure correct format (H, W, 3) uint8
            if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                rgb = (rgb * 255).astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(rgb)
            
            # Create episode-specific directory
            episode_id = self.episode_ids[env_idx]
            episode_frames_dir = self.frames_dir / episode_id
            episode_frames_dir.mkdir(exist_ok=True)
            
            # Save with timestamp
            filename = f"obs_env{env_idx:02d}_step{self.step_counts[env_idx]:06d}.jpg"
            filepath = episode_frames_dir / filename
            img.save(filepath, quality=95)
            
            return filepath
            
        except Exception as e:
            print(f"⚠️  Error saving observation image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'imaginav_agent'):
            self.imaginav_agent.cleanup()
            print("🧹 ImagiNav agent cleaned up")
