import os
import glob

import lmdb
import msgpack_numpy

from internnav.evaluator.utils.config import get_lmdb_path

from .base import BasePathKeyEpisodeloader
from .dataset_utils import skip_list


class ResumablePathKeyEpisodeloader(BasePathKeyEpisodeloader):
    def __init__(
        self,
        dataset_type,
        base_data_dir,
        split_data_types,
        robot_offset,
        filter_same_trajectory,
        task_name,
        run_type,
        retry_list,
        filter_stairs,
        rank=0,
        world_size=1,
        episode_filter=None,  # NEW: Filter by specific episode/path_key
        scene_filter=None,    # NEW: Filter by specific scene ID
    ):
        # 加载所有数据
        super().__init__(
            dataset_type=dataset_type,
            base_data_dir=base_data_dir,
            split_data_types=split_data_types,
            robot_offset=robot_offset,
            filter_same_trajectory=filter_same_trajectory,
            revise_data=True,
            filter_stairs=filter_stairs,
        )
        self.task_name = task_name
        self.run_type = run_type
        self.lmdb_path = get_lmdb_path(task_name)
        self.retry_list = retry_list
        self.episode_filter = episode_filter
        self.scene_filter = scene_filter
        
        # Check if LMDB database exists
        lmdb_file = f'{self.lmdb_path}/sample_data.lmdb'
        database_exists = os.path.exists(lmdb_file)

        # Open main LMDB if present; otherwise fall back to any per-rank LMDBs
        databases = []
        if database_exists:
            databases.append(
                lmdb.open(
                    lmdb_file,
                    map_size=1 * 1024 * 1024 * 1024 * 1024,
                    readonly=True,
                    lock=False,
                )
            )
        else:
            # Create the directory structure if it doesn't exist
            os.makedirs(self.lmdb_path, exist_ok=True)
            # Fallback: use per-rank LMDBs if available (sample_data0.lmdb, sample_data1.lmdb, ...)
            for lmdb_path in glob.glob(os.path.join(self.lmdb_path, 'sample_data*.lmdb')):
                if os.path.basename(lmdb_path) == 'sample_data.lmdb':
                    continue
                try:
                    databases.append(
                        lmdb.open(
                            lmdb_path,
                            map_size=1 * 1024 * 1024 * 1024 * 1024,
                            readonly=True,
                            lock=False,
                        )
                    )
                except Exception:
                    continue
            database_exists = len(databases) > 0

        filtered_target_path_key_list = []
        for path_key in self.path_key_data.keys():
            trajectory_id = int(path_key.split('_')[0])
            if trajectory_id in skip_list:
                continue
            
            # NEW: Apply episode filter
            if self.episode_filter and path_key != self.episode_filter:
                continue
            
            # NEW: Apply scene filter
            if self.scene_filter:
                scene_id = self.path_key_scan.get(path_key, '')
                if scene_id != self.scene_filter:
                    continue
            
            if database_exists:
                value = None
                for db in databases:
                    with db.begin() as txn:
                        value = txn.get(path_key.encode())
                    if value is not None:
                        break
                if value is None:
                    filtered_target_path_key_list.append(path_key)
                else:
                    value = msgpack_numpy.unpackb(value)
                    if value['finish_status'] == 'success':
                        if 'success' in self.retry_list:
                            filtered_target_path_key_list.append(path_key)
                        else:
                            continue
                    else:
                        fail_reason = value['fail_reason']
                        if fail_reason in retry_list:
                            filtered_target_path_key_list.append(path_key)
            else:
                # If no database exists, treat all episodes as not yet attempted
                filtered_target_path_key_list.append(path_key)

        filtered_target_path_key_list.reverse()
        self.resumed_path_key_list = filtered_target_path_key_list
        
        if database_exists:
            for db in databases:
                try:
                    db.close()
                except Exception:
                    pass

    @property
    def size(self):
        return len(self.resumed_path_key_list)
