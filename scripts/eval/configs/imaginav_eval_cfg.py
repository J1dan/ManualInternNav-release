"""
ImagiNav Evaluation Configuration
Uses ImagiNav agent (Gemini + LTX-Video + VGGT) for navigation.
"""

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)

# =============================================================================
# User Configuration
# =============================================================================
DATASET = 'interiornav'
# Name of the evaluation task; used for output directories
TASK_NAME = 'imaginav_eval'
# =============================================================================

# Configuration Logic
if DATASET == 'interiornav':
    base_data_dir = 'data/vln_pe/raw_data/interiornav'
    split_data_types = ['val_imaginav_0_99']
    dataset_type = 'kujiale'
    scene_type = 'kujiale'
    scene_data_dir = 'interiornav_data/scene_data'
else:  # r2r
    base_data_dir = 'data/vln_pe/raw_data/r2r'
    split_data_types = ['val_seen']
    dataset_type = 'mp3d'
    scene_type = 'mp3d'
    scene_data_dir = 'data/scene_data/mp3d_pe'

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8000,
        model_name='imaginav',  # Use ImagiNav agent
        ckpt_path='',  # No checkpoint needed
        model_settings={
            'output_dir': f'logs/{TASK_NAME}',  # Where to save outputs
            'save_frames': True,  # Save observation frames
            'imaginav_config': 'ImagiNav/configs/default_config.yaml',  # Path to ImagiNav config
            'env_num': 1,
            'proc_num': 1,
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
        obstacle_detection=False,
        robot_flash=False,  # Use realistic robot movement
    ),
    dataset=EvalDatasetCfg(
        dataset_type=dataset_type,
        dataset_settings={
            'base_data_dir': base_data_dir,
            'split_data_types': split_data_types,  # Official ImagiNav validation split
            'filter_stairs': False,
        },
    ),
    eval_settings={
        'save_to_json': True,  # Save evaluation results
        'vis_output': True,  # Enable visualization
        'output_dir': f'logs/{TASK_NAME}/results',
    },
    eval_type='vln_distributed'
)
