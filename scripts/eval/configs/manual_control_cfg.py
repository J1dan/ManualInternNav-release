"""
Manual Control Configuration for VLN-PE with Zero-Shot Agent
This config enables visible simulation with manual robot control.
"""

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8066,
        model_name='manual_zero_shot',  # Use manual agent
        ckpt_path='',  # No checkpoint needed for manual control
        model_settings={
            'output_dir': 'manual_logs/manual_control',  # Where to save frames and action file
            'auto_save_frames': True,  # Automatically save frames periodically
            'frame_save_interval': 10,  # Save every N steps
        },
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,
            'headless': True,  # ← ENABLE GUI MODE so you can see the robot
        },
    ),
    task=TaskCfg(
        task_name='manual_control_eval',
        task_settings={
            'env_num': 1,  # Single environment for easier manual control
            'use_distributed': False,
            'proc_num': 1,
        },
        scene=SceneCfg(
            scene_type='mp3d',
            scene_data_dir='data/scene_data/mp3d_pe',
        ),
        robot_name='h1',
        robot_usd_path='data/Embodiments/vln-pe/h1/h1_vln_pointcloud.usd',
        camera_resolution=[480, 256],  # (W,H)
        camera_prim_path='torso_link/h1_pano_camera_0',
        vlnce=False,   # vlnpe by default
        obstacle_detection=False,   # whether allow flash across obstacle
        robot_flash=False,  # Disable flash mode for manual control
    ),
    dataset=EvalDatasetCfg(
        dataset_type="mp3d",
        dataset_settings={
            'base_data_dir': 'data/vln_pe/raw_data/r2r',
            'split_data_types': ['val_seen'],  # Just use val_seen for testing
            'filter_stairs': False,
        },
    ),
    eval_settings={
        'save_to_json': False,  # Don't need JSON results for manual control
        'vis_output': True,  # Enable visualization output
    },
    eval_type='vln_distributed'
)
