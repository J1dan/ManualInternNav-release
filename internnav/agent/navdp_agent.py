import sys
import time
import json
import yaml
from datetime import datetime
import requests
import io
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from gym import spaces
from PIL import Image

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg

# Add ImagiNav to path
IMAGINAV_PATH = Path(__file__).parent.parent.parent / 'ImagiNav'
IMAGINAV_PARENT = IMAGINAV_PATH.parent
if str(IMAGINAV_PARENT) not in sys.path:
    sys.path.insert(0, str(IMAGINAV_PARENT))

try:
    from ImagiNav.models.reasoner import GeminiReasoner
    from ImagiNav.core.config import ReasonerConfig
    from ImagiNav.models.controller import PurePursuitController, ProportionalController
except ImportError as e:
    print(f"⚠️  Warning: Could not import ImagiNav components: {e}")

@Agent.register('navdp')
class NavDPInternNavAgent(Agent):
    """
    NavDP Agent: Gemini -> NavDP -> Controller -> Actions
    """
    
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 3),
        dtype=np.float32,
    )

    def __init__(self, agent_config: AgentCfg):
        super().__init__(agent_config)
        
        self.env_num = agent_config.model_settings.get('env_num', 1)
        self.proc_num = agent_config.model_settings.get('proc_num', 1)
        self.total_envs = self.env_num * self.proc_num
        
        self.output_dir = Path(agent_config.model_settings.get('output_dir', 'logs/navdp'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store task_name for use in logging and output paths
        self.task_name = agent_config.model_settings.get('task_name', None)
        
        # Setup Logging (place agent logs under output_dir/task_name if provided)
        if self.task_name:
            # place logs under output_dir/<task_name>/navdp_agent_logs for consistency with eval logs
            self.log_dir = (self.output_dir / str(self.task_name) / "navdp_agent_logs").resolve()
        else:
            self.log_dir = Path("logs/navdp_agent_logs").resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a custom logger
        self.logger = logging.getLogger("NavDPAgent")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_dir / "navdp_agent.log")
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handler
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            
        # Per-environment episode log handlers (indexed by env)
        self._episode_log_handlers = [None] * self.total_envs

        self.logger.info("NavDP Agent Initialized")
        
        # Load ImagiNav Config
        config_path = IMAGINAV_PATH / "configs/navdp_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.nav_config = yaml.safe_load(f)
            self.logger.info(f"Loaded config from {config_path}")
        else:
            self.logger.warning(f"Config not found at {config_path}")
            self.nav_config = {}

        # Initialize Reasoner
        reasoner_cfg = self.nav_config.get('reasoner', {})
        reasoner_config = ReasonerConfig(
            model_name=reasoner_cfg.get('model_name', "gemini-pro"),
            api_key_env=reasoner_cfg.get('api_key_env', "GOOGLE_API_KEY"),
            system_prompt=reasoner_cfg.get('system_prompt', ""),
            prompt_template=reasoner_cfg.get('prompt_template', "")
        )
        # self.logger.info(f"DEBUGGING: Reasoner System Prompt:\n{reasoner_config.system_prompt}")
        # self.logger.info(f"DEBUGGING: Reasoner Prompt Template:\n{reasoner_config.prompt_template}")
        self.reasoners = []
        for env_idx in range(self.total_envs):
            env_out_dir = self.log_dir / f"env_{env_idx}"
            env_out_dir.mkdir(parents=True, exist_ok=True)
            self.reasoners.append(
                GeminiReasoner(
                    reasoner_config,
                    logger=self.logger,
                    agent_name="navdp",
                    output_dir=str(env_out_dir)
                )
            )

        # Prompt limiting: track per-env prompts sent and apply limit if configured
        self.max_prompts_per_episode = reasoner_cfg.get('max_prompts_per_episode', None)
        # Initialize counters per environment
        self.reasoner_prompts_sent = [0] * self.total_envs
        
        # Initialize Controller
        ctrl_cfg = self.nav_config.get('controller', {})
        ctrl_type = ctrl_cfg.get('type', 'pure_pursuit')

        self.controller_dt = ctrl_cfg.get('dt', 0.25)
        self.controller_max_steps = ctrl_cfg.get('max_steps', 50)
        self.controller_goal_threshold = ctrl_cfg.get('goal_threshold', 0.2)

        if ctrl_type == 'proportional':
            p_cfg = ctrl_cfg.get('proportional_control', {})
            k_v = p_cfg.get('k_v', 1.0)
            k_theta = p_cfg.get('k_theta', 2.0)

            self.controller = ProportionalController(
                kv=k_v,
                ktheta=k_theta,
                max_linear_speed=ctrl_cfg.get('max_linear_speed', 0.6),
                max_angular_speed=ctrl_cfg.get('max_angular_speed', 1.5),
                max_steps=self.controller_max_steps
            )
            self.logger.info(f"Initialized ProportionalController (kv={k_v}, ktheta={k_theta})")

        elif ctrl_type == 'pure_pursuit':
            self.controller = PurePursuitController(
                lookahead_distance=ctrl_cfg.get('lookahead_distance', 0.4), 
                max_linear_speed=ctrl_cfg.get('max_linear_speed', 0.6),   
                max_angular_speed=ctrl_cfg.get('max_angular_speed', 1.5),
                smooth_trajectory=ctrl_cfg.get('smooth_trajectory', True),
                dt=self.controller_dt,
                max_steps=self.controller_max_steps,
                goal_threshold=self.controller_goal_threshold
            )
            self.logger.info("Initialized PurePursuitController")

        else:
            self.logger.warning(f"Unknown controller type '{ctrl_type}', falling back to PurePursuitController")
            self.controller = PurePursuitController(
                lookahead_distance=ctrl_cfg.get('lookahead_distance', 0.4), 
                max_linear_speed=ctrl_cfg.get('max_linear_speed', 0.6),   
                max_angular_speed=ctrl_cfg.get('max_angular_speed', 1.5)
            )
        
        # NavDP Server Config
        self.navdp_port = agent_config.model_settings.get('navdp_port', 1234)
        self.navdp_url = f"http://localhost:{self.navdp_port}"

        # Manual waypoint control
        self.manual_control = agent_config.model_settings.get('manual_control', False)
        manual_waypoint_file = agent_config.model_settings.get('manual_waypoint_file', None)
        if manual_waypoint_file is None:
            self.manual_waypoint_file = (self.output_dir / "manual_waypoint.json").resolve()
        else:
            self.manual_waypoint_file = Path(manual_waypoint_file).expanduser().resolve()
        self.manual_waypoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        # State
        self.step_counts = [0] * self.total_envs
        self.episode_ids = ['unknown'] * self.total_envs
        self.trajectory_commands = [None] * self.total_envs
        self.command_indices = [0] * self.total_envs
        self.waiting_for_waypoint = [True] * self.total_envs
        # Track whether we've saved the first pre-action frame for each episode
        self.preaction_saved = [False] * self.total_envs
        # Flag to mark that the next executed action is the first after a decision/trajectory generation
        self.mark_next_action_decision = [False] * self.total_envs
        
        # Reset NavDP Server
        self._reset_navdp_server()

    def _reset_navdp_server(self):
        """Reset the NavDP server with intrinsic parameters."""
        # Assuming 480x256 input for NavDP
        width = 480
        height = 256
        fov = 90
        fx = width / 2.0  # Approx for 90 deg FOV
        fy = width / 2.0
        cx = width / 2.0
        cy = height / 2.0
        
        intrinsic = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])
        
        try:
            response = requests.post(f"{self.navdp_url}/navigator_reset", json={
                'intrinsic': intrinsic.tolist(),
                'stop_threshold': -4.5,
                'batch_size': 1 # We query one by one for now
            })
            if response.status_code == 200:
                print("✅ NavDP Server Reset Successful")
                self.logger.info("NavDP Server Reset Successful")
            else:
                print(f"❌ NavDP Server Reset Failed: {response.text}")
                self.logger.error(f"NavDP Server Reset Failed: {response.text}")
        except Exception as e:
            print(f"❌ Could not connect to NavDP Server: {e}")
            self.logger.error(f"Could not connect to NavDP Server: {e}")

    def reset(self, reset_ls: Optional[List[int]] = None, episode_ids: Optional[List[str]] = None):
        if reset_ls is not None:
            for i, env_idx in enumerate(reset_ls):
                if env_idx < self.total_envs:
                    self.step_counts[env_idx] = 0
                    self.trajectory_commands[env_idx] = None
                    self.command_indices[env_idx] = 0
                    self.waiting_for_waypoint[env_idx] = True
                    self.preaction_saved[env_idx] = False
                    self.mark_next_action_decision[env_idx] = False
                    if episode_ids:
                            self.episode_ids[env_idx] = episode_ids[i]
                            # reset per-episode prompt counter for this env when episode changes
                            try:
                                self.reasoner_prompts_sent[env_idx] = 0
                            except Exception:
                                # ensure list is long enough
                                if env_idx >= len(self.reasoner_prompts_sent):
                                    self.reasoner_prompts_sent = self.reasoner_prompts_sent + [0] * (env_idx - len(self.reasoner_prompts_sent) + 1)
                                self.reasoner_prompts_sent[env_idx] = 0
                            # Attach a per-episode file handler to capture logs for this episode
                            try:
                                # Remove existing handler for this env if present
                                if env_idx < len(self._episode_log_handlers) and self._episode_log_handlers[env_idx] is not None:
                                    old_h = self._episode_log_handlers[env_idx]
                                    if old_h in self.logger.handlers:
                                        self.logger.removeHandler(old_h)
                                        try:
                                            old_h.close()
                                        except Exception:
                                            pass
                                # Build per-episode log directory
                                ep = str(episode_ids[i]).replace('/', '_').replace(' ', '_')
                                if self.task_name:
                                    per_dir = (self.output_dir / str(self.task_name) / "navdp_agent_logs" / ep).resolve()
                                else:
                                    per_dir = (self.output_dir / "navdp_agent_logs" / ep).resolve()
                                per_dir.mkdir(parents=True, exist_ok=True)
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                per_file = per_dir / f"navdp_agent_{ts}.log"
                                per_fh = logging.FileHandler(per_file)
                                per_fh.setLevel(logging.INFO)
                                # reuse existing formatter if available
                                fmt = None
                                for h in self.logger.handlers:
                                    if getattr(h, 'formatter', None):
                                        fmt = h.formatter
                                        break
                                if fmt is None:
                                    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                                per_fh.setFormatter(fmt)
                                self.logger.addHandler(per_fh)
                                self._episode_log_handlers[env_idx] = per_fh
                                self.logger.info(f"Attached per-episode log: {per_file}")
                            except Exception as e:
                                self.logger.error("Could not attach per-episode log handler", exc_info=True)
                                raise RuntimeError("Failed to attach per-episode log handler") from e

                            # reset per-env reasoner chat when episode changes
                            try:
                                if env_idx < len(self.reasoners):
                                    self.reasoners[env_idx].reset()
                                    # reasoner.reset() logs should now go into the per-episode handler
                            except Exception as e:
                                self.logger.error(f"Failed to reset reasoner for env {env_idx}", exc_info=True)
                                raise RuntimeError(f"Reasoner reset failed for env {env_idx}: {e}") from e

                            self.logger.info(f"Reset Env {env_idx} Episode {episode_ids[i]}")
        else:
            self.step_counts = [0] * self.total_envs
            self.trajectory_commands = [None] * self.total_envs
            self.command_indices = [0] * self.total_envs
            self.waiting_for_waypoint = [True] * self.total_envs
            self.mark_next_action_decision = [False] * self.total_envs
            # reset per-episode prompt counters
            self.reasoner_prompts_sent = [0] * self.total_envs
            # reset all reasoner chats
            try:
                for r in self.reasoners:
                    r.reset()
            except Exception as e:
                self.logger.warning("Failed to reset one or more reasoner chats", exc_info=True)
                raise e

            # Remove any per-episode handlers
            try:
                for i, h in enumerate(self._episode_log_handlers):
                    if h is not None and h in self.logger.handlers:
                        try:
                            self.logger.removeHandler(h)
                        except Exception:
                            pass
                        try:
                            h.close()
                        except Exception:
                            pass
                        self._episode_log_handlers[i] = None
            except Exception:
                self.logger.warning("Failed to remove per-episode log handlers cleanly", exc_info=True)

            self.logger.info("Reset All Envs")

    def step(self, obs_batch: List[Dict[str, Any]]) -> List[List[float]]:
        actions = []
        
        for env_idx, obs in enumerate(obs_batch):
            self.step_counts[env_idx] += 1
            
            # 1. Get Inputs
            rgb = obs.get('rgb') # (H, W, 3)
            depth = obs.get('depth') # (H, W, 1) or (H, W)
            instruction = obs.get('instruction')

            # Log depth diagnostics (shape, dtype, range, NaN/Inf counts) for determinism
            try:
                if depth is None:
                    self.logger.debug(f"Env {env_idx} depth: None")
                else:
                    d_arr = np.asarray(depth)
                    # Guard against empty arrays
                    if d_arr.size:
                        self.logger.debug(
                            f"Env {env_idx} depth dtype={d_arr.dtype}, shape={d_arr.shape}, "
                            f"min={float(np.nanmin(d_arr))}, max={float(np.nanmax(d_arr))}, "
                            f"mean={float(np.nanmean(d_arr))}, n_nan={int(np.isnan(d_arr).sum())}, n_inf={int(np.isinf(d_arr).sum())}"
                        )
                    else:
                        self.logger.debug(f"Env {env_idx} depth array is empty, shape={d_arr.shape}")
            except Exception as e:
                self.logger.warning(f"Could not log depth diagnostics for env {env_idx}: {e}")
            
            self.logger.info(f"Step {self.step_counts[env_idx]} Env {env_idx}: Instruction: {instruction}")

            # 2. Check if we need new trajectory
            need_new_trajectory = (
                self.trajectory_commands[env_idx] is None or
                self.command_indices[env_idx] >= len(self.trajectory_commands[env_idx])
            )
            
            if need_new_trajectory:
                self.logger.info(f"  Env {env_idx} Step {self.step_counts[env_idx]}: Generating new trajectory")

                # Save the first pre-action RGB frame of the episode only
                if not self.preaction_saved[env_idx]:
                    try:
                        # Determine episode id for this env to create per-episode folder
                        episode_id = self.episode_ids[env_idx] if env_idx < len(self.episode_ids) else None
                        if episode_id is None:
                            ep = 'unknown'
                        else:
                            ep = str(episode_id)
                        # sanitize folder name
                        ep = ep.replace('/', '_').replace(' ', '_')

                        # Use task_name in path if available
                        if self.task_name:
                            frames_dir = Path(self.output_dir) / str(self.task_name) / "video" / "navdp_decision_making" / ep
                        else:
                            frames_dir = Path(self.output_dir) / "video" / "navdp_decision_making" / ep
                        frames_dir.mkdir(parents=True, exist_ok=True)

                        if rgb is None:
                            self.logger.warning(f"  No RGB available to save pre-action frame (env {env_idx}, episode {ep})")
                        else:
                            rgb_img = rgb
                            # Handle float images in [0,1]
                            if getattr(rgb_img, 'dtype', None) is not None and rgb_img.dtype != np.uint8:
                                rgb_u8 = np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)
                            else:
                                rgb_u8 = rgb_img.astype(np.uint8)

                            ts = int(time.time() * 1000000)
                            filename = frames_dir / f"env{env_idx}_ep{ep}_step{self.step_counts[env_idx]:06d}_preaction_{ts}.jpg"
                            Image.fromarray(rgb_u8).save(filename)
                            self.logger.info(f"  Saved pre-action frame: {filename}")

                        # Mark as saved so we do not save again for this episode
                        self.preaction_saved[env_idx] = True
                    except Exception as e:
                        self.logger.warning(f"  Could not save pre-action frame: {e}")

                if self.manual_control:
                    waypoint = self._read_manual_waypoint(env_idx)
                    if waypoint is None:
                        if not self.waiting_for_waypoint[env_idx]:
                            self.waiting_for_waypoint[env_idx] = True
                            self.logger.info("  Action completed, waiting for next manual waypoint...")

                        self.logger.info(f"  Checking for manual waypoint file: {self.manual_waypoint_file}")
                        while waypoint is None:
                            time.sleep(0.5)
                            waypoint = self._read_manual_waypoint(env_idx)

                    self.waiting_for_waypoint[env_idx] = False
                else:
                    # Gemini Query -> Waypoint
                    # Enforce per-episode prompt limit similar to ImagiNav
                    try:
                        if (self.max_prompts_per_episode is not None and
                            self.reasoner_prompts_sent[env_idx] >= int(self.max_prompts_per_episode)):
                            # Max prompts reached -> signal full stop via [0,0,0]
                            self.logger.warning(
                                f"Max reasoner prompts reached for env {env_idx} ({self.reasoner_prompts_sent[env_idx]} >= {self.max_prompts_per_episode}). Signaling STOP.")
                            waypoint = [0.0, 0.0, 0.0]
                        else:
                            reasoner = self.reasoners[env_idx] if env_idx < len(self.reasoners) else self.reasoners[0]
                            waypoint = reasoner.generate_waypoint(
                                image_input=rgb, 
                                instruction=instruction,
                                step_count=self.step_counts[env_idx]
                            )
                            # Count this prompt for the current environment/episode
                            try:
                                self.reasoner_prompts_sent[env_idx] += 1
                            except Exception:
                                # ensure robustness: initialize counter if missing
                                if env_idx >= len(self.reasoner_prompts_sent):
                                    # expand list
                                    self.reasoner_prompts_sent = self.reasoner_prompts_sent + [0] * (env_idx - len(self.reasoner_prompts_sent) + 1)
                                self.reasoner_prompts_sent[env_idx] += 1

                            # To avoid hitting Gemini/Reasoner quota limits, pause briefly after a successful query
                            self.logger.info("Gemini queried successfully; sleeping 15s to avoid quota exceeding")
                            time.sleep(15)
                    except RuntimeError as e:
                        # Gemini/Reasoner failed; abort evaluation immediately for all episodes
                        self.logger.critical(f"Gemini/Reasoner error: {e}. Aborting evaluation.")
                        # Raise to bubble up to the main evaluation loop which will terminate
                        raise

                # Convert LLM convention (positive = right) to controller/NavDP convention (positive = left)
                # Note: change applied only within NavDP agent to avoid affecting other agents
                try:
                    waypoint = list(waypoint)
                    if len(waypoint) >= 2:
                        waypoint[1] = -float(waypoint[1])
                    if len(waypoint) >= 3:
                        waypoint[2] = -float(waypoint[2])
                    self.logger.debug(f"Converted waypoint LLM->controller: {waypoint}")
                except Exception as e:
                    self.logger.warning(f"Could not convert waypoint signs: {e}")

                self.logger.info(f"  Waypoint: {waypoint}")

                # If the reasoner signaled a full STOP (e.g., due to prompt limit), enqueue a stop command
                try:
                    if isinstance(waypoint, (list, tuple)) and len(waypoint) >= 3 and abs(waypoint[0]) < 1e-6 and abs(waypoint[1]) < 1e-6 and abs(waypoint[2]) < 1e-6:
                        self.logger.info(f"Reasoner signaled STOP for env {env_idx}; enqueuing stop command.")
                        self.trajectory_commands[env_idx] = [[0.0, 0.0, 0.0]]
                        # mark that we will execute the stop immediately
                        self.command_indices[env_idx] = 1  # consume the single stop command
                        self.mark_next_action_decision[env_idx] = True
                        # Immediately append stop action for this step to ensure caller receives an action
                        action = [0.0, 0.0, 0.0]
                        actions.append(action)
                        self.logger.info(f"  Enqueued and executed STOP for env {env_idx}")
                        # Move on to next environment (we already appended action for this env)
                        continue
                except Exception as e:
                    ep = self.episode_ids[env_idx] if env_idx < len(self.episode_ids) else 'unknown'
                    wp = locals().get('waypoint', None)
                    msg = (f"Error processing waypoint for env {env_idx} (episode={ep}) at step "
                           f"{self.step_counts[env_idx]}: waypoint={repr(wp)}, instruction={repr(instruction)}; "
                           f"original error: {e}")
                    self.logger.error(msg)
                    raise RuntimeError(msg) from e

                # Check for explicit rotation command from Reasoner
                # Waypoint format is [x, y, theta]
                # If x and y are close to 0, and theta is non-zero, it's a turn-in-place
                if len(waypoint) >= 3 and abs(waypoint[0]) < 0.3 and abs(waypoint[1]) < 0.3 and abs(waypoint[2]) > 1.0:
                     theta_deg = waypoint[2]
                     self.logger.info(f"  Reasoner requests turn: {theta_deg} degrees")
                     
                     # Simpler: split total yaw into N equal steps of ~5° each (no remainder handling)
                     dt = self.controller_dt
                     abs_deg = abs(float(theta_deg))

                     # Minimum per-step degree (can be tuned)
                     per_step_deg = 10.0
                     num_steps = max(1, int(np.ceil(abs_deg / per_step_deg)))

                     # Actual per-step degrees so steps evenly cover requested yaw
                     step_deg = abs_deg / num_steps

                     # Direction: +1 for CCW, -1 for CW (based on sign of theta_deg)
                     direction = np.sign(float(theta_deg)) if float(theta_deg) != 0 else 1.0

                     commands = []

                     # Helper to cap angular velocity by controller's max angular speed
                     max_ang = getattr(self.controller, 'max_angular_speed', 1.5)

                     # Compute angular velocity required to perform `step_deg` in one dt
                     step_ang_vel = direction * (np.deg2rad(step_deg) / dt)
                     # Cap if necessary
                     if abs(step_ang_vel) > max_ang:
                         step_ang_vel = direction * max_ang

                     # Generate identical commands for each step
                     for _ in range(num_steps):
                         commands.append([0.0, 0.0, float(step_ang_vel)])

                     # Log how many steps and per-step degrees
                     self.logger.info(f"  Reasoner turn: {theta_deg}° -> {num_steps} steps, {step_deg:.2f}° per step, ang_vel={step_ang_vel:.3f} rad/s")
                     self.trajectory_commands[env_idx] = commands
                     self.command_indices[env_idx] = 0
                     self.mark_next_action_decision[env_idx] = True
                     self.logger.info(f"  Generated {len(commands)} turning commands from Reasoner (~5° per step)")
                     
                else: 
                    # NavDP Query -> Trajectory
                    episode_id = self.episode_ids[env_idx] if env_idx < len(self.episode_ids) else None
                    trajectory = self._query_navdp(rgb, depth, waypoint[:2], episode_id=episode_id)
                    self.logger.info(f"  Received trajectory from NavDP: {trajectory}")
                    if trajectory is not None:
                        # Check for "turn in place" condition from NavDP (x components are ~0)
                        # NavDP sets x=0, y=sign(turn) when confidence is low / stopping
                        if np.all(np.abs(trajectory[:, 0]) < 1e-4):
                            self.logger.info("  NavDP signal: Stop/Turn in place")
                            # Determine turn direction from y values (should be +1 or -1)
                            turn_sign = np.sign(np.mean(trajectory[:, 1]))
                            
                            # Generate manual commands for turning
                            # Fixed number of steps (e.g. 5) to execute the turn
                            commands = []
                            for _ in range(5): 
                                cmd = [0.0, 0.0, 0.5 * float(turn_sign)]
                                commands.append(cmd)
                                
                            self.trajectory_commands[env_idx] = commands
                            self.command_indices[env_idx] = 0
                            self.mark_next_action_decision[env_idx] = True
                            self.logger.info(f"  Generated {len(commands)} turning commands")
                            
                        else:
                            # Pure Pursuit -> Commands
                            commands = self.controller.calculate_trajectory_commands(
                                waypoints=trajectory,
                            )
                            
                            if len(commands) > 0:
                                self.trajectory_commands[env_idx] = commands
                                self.command_indices[env_idx] = 0
                                self.mark_next_action_decision[env_idx] = True
                                self.logger.info(f"  Generated {len(commands)} commands")
                            else:
                                self.logger.warning("  Controller generated 0 commands")
                                self.trajectory_commands[env_idx] = None
                    else:
                        self.logger.warning("  Trajectory generation failed")
                        self.trajectory_commands[env_idx] = None

            # 3. Execute next command
            if (self.trajectory_commands[env_idx] is not None and
                self.command_indices[env_idx] < len(self.trajectory_commands[env_idx])):
                
                is_first_cmd = (self.command_indices[env_idx] == 0)
                cmd = self.trajectory_commands[env_idx][self.command_indices[env_idx]]
                self.command_indices[env_idx] += 1
                
                action = cmd

                # If this is the first executed command after a new decision/trajectory, save a decision-making frame
                if is_first_cmd and self.mark_next_action_decision[env_idx]:
                    try:
                        episode_id = self.episode_ids[env_idx] if env_idx < len(self.episode_ids) else None
                        ep = 'unknown' if episode_id is None else str(episode_id).replace('/', '_').replace(' ', '_')

                        # Use task_name in path if available
                        if self.task_name:
                            frames_dir = Path(self.output_dir) / str(self.task_name) / "video" / "navdp_decision_making" / ep
                        else:
                            frames_dir = Path(self.output_dir) / "video" / "navdp_decision_making" / ep
                        frames_dir.mkdir(parents=True, exist_ok=True)

                        if rgb is None:
                            self.logger.warning(f"  No RGB available to save decision-making frame (env {env_idx}, episode {ep})")
                        else:
                            rgb_img = rgb
                            # Handle float images in [0,1]
                            if getattr(rgb_img, 'dtype', None) is not None and rgb_img.dtype != np.uint8:
                                rgb_u8 = np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)
                            else:
                                rgb_u8 = rgb_img.astype(np.uint8)

                            ts = int(time.time() * 1000000)
                            filename = frames_dir / f"env{env_idx}_ep{ep}_step{self.step_counts[env_idx]:06d}_decision_making_{ts}.jpg"
                            Image.fromarray(rgb_u8).save(filename)
                            self.logger.info(f"  Saved decision-making frame: {filename}")
                    except Exception as e:
                        self.logger.warning(f"  Could not save decision-making frame: {e}")
                    finally:
                        self.mark_next_action_decision[env_idx] = False

                self.logger.info(f"  Env {env_idx} Step {self.step_counts[env_idx]}: Action ({self.command_indices[env_idx]}/{len(self.trajectory_commands[env_idx])}): {action}")
            else:
                # Fallback if trajectory failed or empty
                action = [0.0, 0.0, 0.0]
                self.logger.warning("  No valid command available, stopping")
            
            actions.append(action)
        
        # print("actions:", actions)
        return actions

    def _query_navdp(self, rgb: np.ndarray, depth: np.ndarray, goal: List[float], episode_id: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Query NavDP server for trajectory.
        Passes the episode_id in the form data so the NavDP server can save per-episode overlays.
        """
        # try:
        #     # Resize to 224x224
        #     rgb_resized = cv2.resize(rgb, (224, 224))
            
        #     # Handle Depth
        #     if depth is None:
        #         # Create dummy depth if missing (dangerous but prevents crash)
        #         depth_resized = np.zeros((224, 224), dtype=np.uint32)
        #     else:
        #         # Resize depth
        #         # Depth is usually float meters. NavDP expects uint32 (meters * 10000)
        #         if depth.dtype == np.float32 or depth.dtype == np.float64:
        #             depth_uint = (depth * 10000).astype(np.uint32)
        #         else:
        #             depth_uint = depth.astype(np.uint32)
                
        #         # OpenCV resize does not support uint32 directly in some versions or specific contexts
        #         # We can convert to float32 for resize, then back to uint32
        #         depth_float = depth_uint.astype(np.float32)
        #         depth_resized_float = cv2.resize(depth_float, (224, 224), interpolation=cv2.INTER_NEAREST)
        #         depth_resized = depth_resized_float.astype(np.uint32)

        #     # Encode images
        #     _, rgb_encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))
            
        #     # Depth needs to be saved as PNG to preserve values
        #     # Image.fromarray(depth_resized, mode='I')
        #     depth_pil = Image.fromarray(depth_resized, mode='I')
        #     depth_bytes = io.BytesIO()
        #     depth_pil.save(depth_bytes, format='PNG')
        #     depth_bytes.seek(0)
            

        try:
            # Resize to 224x224
            # rgb_resized = cv2.resize(rgb, (224, 224))
            
            # Resize depth
            # Depth is usually float meters. NavDP expects depth in PNG uint16 with values = depth*10000
            # Convert to float and scale accordingly to uint16 to match client_utils/server expectations
            depth_float = depth.astype(np.float32)
            # Ensure depth is 2D (H, W) not (H, W, 1)
            if depth_float.ndim == 3 and depth_float.shape[2] == 1:
                depth_float = depth_float.squeeze(2)

            depth_uint16 = np.clip(depth_float * 10000.0, 0.0, 65535.0).astype(np.uint16)

            # Optionally resize here if needed (commented out for now)
            # depth_resized = cv2.resize(depth_uint16, (224, 224), interpolation=cv2.INTER_NEAREST)
            depth_resized = depth_uint16

            # Encode images
            _, rgb_encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # Encode depth as PNG (uint16) to match NavDP server's expected format
            ok, depth_png = cv2.imencode('.png', depth_resized)
            if not ok:
                self.logger.warning("Failed to encode depth image as PNG; falling back to PIL")
                depth_pil = Image.fromarray(depth_resized, mode='I;16')
                depth_bytes = io.BytesIO()
                depth_pil.save(depth_bytes, format='PNG')
                depth_bytes.seek(0)
                depth_bytes_val = depth_bytes.getvalue()
            else:
                depth_bytes_val = depth_png.tobytes()

            # DEBUG: Save depth image for inspection
            try:
                debug_sub = self.log_dir / "debug_decision_depths"
                debug_sub.mkdir(exist_ok=True)
                ts = int(time.time() * 1000000)
                with open(debug_sub / f"depth_{ts}.png", 'wb') as df:
                    df.write(depth_bytes_val)
            except Exception as e:
                self.logger.warning(f"Could not save debug depth: {e}")


            files = {
                'image': ('image.jpg', rgb_encoded.tobytes(), 'image/jpeg'),
                'depth': ('depth.png', depth_bytes_val, 'image/png'),
            }
            print(f"  Querying NavDP with goal: {goal}")
            goal_data = {
                'goal_x': [goal[0]],
                'goal_y': [goal[1]]
            }
            
            data = {
                'goal_data': json.dumps(goal_data),
                'depth_time': 0,
                'rgb_time': 0,
            }

            # Include episode id if available so the NavDP server can save overlays per-episode
            if episode_id is not None:
                data['episode_id'] = str(episode_id)
            
            response = requests.post(f"{self.navdp_url}/pointgoal_step", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                trajectory = np.array(result['trajectory']) # (1, N, 3) or (N, 3)?
                # all_trajectory = np.array(result['all_trajectory'])
                all_values = np.array(result['all_values'])

                # Log receiving result and stats
                self.logger.info("[NavDP] Received trajectory response")
                self.logger.info(f"[NavDP] Waypoint queried: {goal}")

                # Safely compute stats and log them
                try:
                    self.logger.info(f"[NavDP] all_values max: {float(np.max(all_values))}, min: {float(np.min(all_values))}")
                except Exception as e:
                    self.logger.warning(f"[NavDP] Could not compute all_values stats: {e}")

                try:
                    # Flatten trajectory across batch & timesteps, keep last dim for x,y,yaw
                    traj_arr = trajectory
                    if traj_arr.ndim == 3:
                        flat = traj_arr.reshape(-1, traj_arr.shape[-1])
                    else:
                        flat = traj_arr.reshape(-1, traj_arr.shape[-1])

                except Exception as e:
                    self.logger.warning(f"[NavDP] Could not compute trajectory stats: {e}")

                self.logger.info(f"[NavDP] Trajectory shape: {trajectory.shape}")
                try:
                    if trajectory.ndim == 3:
                        self.logger.info(f"[NavDP] Trajectory (first seq, first 24 points): {trajectory[0, :24].tolist()}")
                    else:
                        self.logger.info(f"[NavDP] Trajectory (first 24 points): {trajectory[:24].tolist()}")
                except Exception:
                    # Fallback if conversion to list fails
                    self.logger.debug("[NavDP] Could not log trajectory sample due to unexpected shape or content")

                # Server returns list.
                # navdp_server.py: return jsonify({'trajectory': execute_trajectory.tolist(), ...})
                # execute_trajectory is usually (N, 3) or (Batch, N, 3)
                # In server: execute_trajectory ... = navdp_navigator.step_pointgoal(...)
                # If batch_size=1, it might be (1, N, 3)

                if trajectory.ndim == 3:
                    return trajectory[0]
                return trajectory
            else:
                print(f"❌ NavDP Query Failed: {response.status_code} {response.text}")
                self.logger.error(f"NavDP Query Failed: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ NavDP Request Error: {e}")
            self.logger.error(f"NavDP Request Error: {e}")
            return None

    def _read_manual_waypoint(self, env_idx: int) -> Optional[List[float]]:
        """
        Read a manual waypoint from JSON file and delete it after reading.
        Expected format: {"waypoint": [x, y]} or {"waypoint": [x, y, theta]}
        """
        if not self.manual_waypoint_file.exists():
            return None

        try:
            with open(self.manual_waypoint_file, 'r') as f:
                data = json.load(f)

            waypoint = data.get('waypoint', None)
            if waypoint is None:
                waypoint = data.get('goal', None)

            if isinstance(waypoint, list) and len(waypoint) in (2, 3):
                if len(waypoint) == 2:
                    waypoint = [float(waypoint[0]), float(waypoint[1]), 0.0]
                else:
                    waypoint = [float(waypoint[0]), float(waypoint[1]), float(waypoint[2])]

                self.manual_waypoint_file.unlink()
                self.logger.info(
                    f"  Manual waypoint consumed (env {env_idx}): {waypoint}. File deleted."
                )
                return waypoint

            self.logger.warning("  Invalid manual waypoint format, deleting file")
            self.manual_waypoint_file.unlink()
            return None
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"  Error reading manual waypoint file: {e}")
            if self.manual_waypoint_file.exists():
                self.manual_waypoint_file.unlink()
            return None
