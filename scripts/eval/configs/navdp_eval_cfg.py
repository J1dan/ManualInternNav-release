"""
NavDP Evaluation Configuration
Uses NavDP agent (Gemini + NavDP + Controller) for navigation.
"""

# =============================================================================
# User Configuration
# =============================================================================
# Select Dataset: 'r2r' or 'interiornav'
DATASET = 'interiornav'
# Name of the evaluation task; used for output directories
TASK_NAME = 'navdp_eval_manualtest'
# =============================================================================

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)
output_dir = f'logs/{TASK_NAME}'
# Configuration Logic
if DATASET == 'interiornav':
    base_data_dir = 'data/vln_pe/raw_data/interiornav'
    split_data_types = ['val_imaginav']
    dataset_type = 'kujiale'
    scene_type = 'kujiale'
    scene_data_dir = 'interiornav_data/scene_data'
else:  # r2r (default)
    base_data_dir = 'data/vln_pe/raw_data/r2r'
    split_data_types = ['val_seen','val_unseen']
    dataset_type = 'mp3d'
    scene_type = 'mp3d'
    scene_data_dir = 'data/scene_data/mp3d_pe'

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8123,
        model_name='navdp',  # Use NavDP agent
        ckpt_path='',  # No checkpoint needed
        model_settings={
            'output_dir': output_dir,  # Where to save outputs
            'navdp_port': 1234,
            'env_num': 1,
            'proc_num': 1,
            # Manual waypoint control
            'manual_control': True,
            'manual_waypoint_file': f'{output_dir}/manual_waypoint.json',
        },
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,
            'headless': True,  # Set to False to see visualization
        },
    ),
    task=TaskCfg(
        task_name=TASK_NAME,
        task_settings={
            'env_num': 1,  # Single environment
            'use_distributed': False,
            'proc_num': 1,
        },
        scene=SceneCfg(
            scene_type=scene_type,
            scene_data_dir=scene_data_dir,
        ),
        robot_name='h1',
        robot_usd_path='data/Embodiments/vln-pe/h1/h1_vln_pointcloud.usd',
        camera_resolution=[480, 256],  # (W,H)
        camera_prim_path='torso_link/h1_pano_camera_0',
        vlnce=False,   # vlnpe by default
        obstacle_detection=True,
        robot_flash=False,  # Use realistic robot movement
    ),
    dataset=EvalDatasetCfg(
        dataset_type=dataset_type,
        dataset_settings={
            'base_data_dir': base_data_dir,
            'split_data_types': split_data_types,
            'filter_stairs': False,
        },
    ),
    eval_settings={
        'save_to_json': True,  # Save evaluation results
        'vis_output': True,  # Enable visualization
        'output_dir': f'{output_dir}/results',
    },
    eval_type='vln_distributed'
)
