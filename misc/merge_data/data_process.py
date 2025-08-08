import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import tqdm
import yaml
from easydict import EasyDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskConfig:
    """Task Configuration Class"""
    task_name: str
    data_pattern: List[str]
    instructions: List[str]
    exclude_episodes: List[str] = field(default_factory=list)
    chunk_size: int = 20
    required_keys: List[str] = field(default_factory=lambda: ['action', 'timestamp', 'obs/state'])
    min_trajectory_length: int = 10

    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskConfig':
        # Filter parameters that TaskConfig actually needs
        from dataclasses import fields
        valid_fields = {field.name for field in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

@dataclass
class ProcessConfig:
    """Processing Configuration Class"""
    target_dir: str
    output_filename: str
    chunk_size: int = 20
    percentile_range: Tuple[int, int] = (1, 99)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessConfig':
        # Filter parameters that ProcessConfig actually needs
        from dataclasses import fields
        valid_fields = {field.name for field in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

class UniversalTaskProcessor:
    """Universal Task Processor - Supports all types of tasks"""

    def __init__(self, config: TaskConfig):
        self.config = config

    def get_episode_files(self) -> List[str]:
        """Get list of data files"""
        all_files = []
        for data_pattern in self.config.data_pattern:
            all_files += glob(data_pattern)
        return all_files

    def validate_episode(self, episode_file: str) -> bool:
        """Validate if data file is valid"""
        try:
            with h5py.File(episode_file, 'r') as root:
                # Check if required keys exist
                if not all(key in root for key in self.config.required_keys):
                    return False

                # Check trajectory length
                if 'action' in root:
                    actions = np.array(root['action'])
                    if len(actions) < self.config.min_trajectory_length:
                        return False

                return True
        except Exception as e:
            logger.warning(f"Failed to validate file {episode_file}: {e}")
            return False

    def filter_episodes(self, episode_files: List[str]) -> List[str]:
        """Filter data files"""
        filtered_files = []
        for episode_file in episode_files:
            file_name = Path(episode_file).stem
            if file_name not in self.config.exclude_episodes and self.validate_episode(episode_file):
                filtered_files.append(episode_file)
        return filtered_files

class DataProcessor:
    """Main Data Processor"""

    def __init__(self, config: ProcessConfig):
        self.config = config
        self.target_dir = Path(config.target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

        self.all_task_path_lang = {}
        self.all_episode_path = []

    def add_task(self, task_id: str, task_config: TaskConfig) -> None:
        """Add task"""
        processor = UniversalTaskProcessor(task_config)

        # Get and filter data files
        episode_files = processor.get_episode_files()
        filtered_files = processor.filter_episodes(episode_files)

        logger.info(f"Task {task_id}: Found {len(episode_files)} files, {len(filtered_files)} files after filtering")

        # Store task information
        self.all_task_path_lang[task_id] = {
            "instructions": task_config.instructions,
            "data_path": filtered_files
        }

        self.all_episode_path.extend(filtered_files)

    def process_episodes(self) -> Dict[str, Any]:
        """Process all data files"""
        rel_action_list = []
        state_list = []
        counter = 0

        logger.info(f"Starting to process {len(self.all_episode_path)} data files...")

        for episode_file in tqdm.tqdm(self.all_episode_path, desc="Processing data files"):
            try:
                episode_actions, episode_states = self._process_single_episode(episode_file)
                rel_action_list.extend(episode_actions)
                state_list.append(episode_states)
                counter += 1
            except Exception as e:
                logger.error(f"Failed to process file {episode_file}: {e}")
                continue

        logger.info(f"Successfully processed {counter} files")

        if not rel_action_list:
            raise ValueError("No data files were successfully processed")

        return self._compute_statistics(rel_action_list, state_list)

    def _process_single_episode(self, episode_file: str) -> List[np.ndarray]:
        """Process a single data file"""
        rel_action_list = []

        with h5py.File(episode_file, 'r') as root:
            actions = np.array(root['action'])
            state = np.array(root['obs/state'])

            # Calculate relative actions
            delta_actions = actions - np.append(actions[0][np.newaxis, :], actions[:-1], axis=0)
            delta_state_action = actions - state

            num_traj = actions.shape[0]

            for idx in range(num_traj):
                rel_action = np.concatenate([
                    np.concatenate([
                        delta_state_action[idx, :5][np.newaxis, :],
                        actions[idx, 5:][np.newaxis, :]
                    ], axis=1),
                    np.concatenate([
                        delta_actions[idx+1:idx + self.config.chunk_size, :5],
                        actions[idx+1:idx + self.config.chunk_size, 5:]
                    ], axis=1)
                ], axis=0)

                rel_action_list.append(rel_action)

        return rel_action_list, state

    def _compute_statistics(self, rel_action_list: List[np.ndarray], state_list: List[np.ndarray]) -> Dict[str, Any]:
        """Calculate statistics"""
        rel_action_array = np.concatenate(rel_action_list, axis=0)
        state_array = np.concatenate(state_list, axis=0)

        min_percentile, max_percentile = self.config.percentile_range

        statistics = {
            'rel_min_action': np.percentile(rel_action_array, min_percentile, axis=0).tolist(),
            'rel_max_action': np.percentile(rel_action_array, max_percentile, axis=0).tolist(),
            'rel_mean_action': np.mean(rel_action_array, axis=0).tolist(),
            'rel_std_action': np.std(rel_action_array, axis=0).tolist(),
            'mean_state': np.mean(state_array, axis=0).tolist(),
            'std_state': np.std(state_array, axis=0).tolist(),
            'task_data': self.all_task_path_lang
        }

        return statistics

    def save_results(self, statistics: Dict[str, Any]) -> None:
        """Save results"""
        output_path = self.target_dir / self.config.output_filename

        with open(output_path, 'w') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        logger.info(f"Results have been saved to: {output_path}")

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)