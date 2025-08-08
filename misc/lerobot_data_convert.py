import glob
import os
import sys
from typing import Any, Iterator, Tuple

import h5py
import numpy as np

sys.path.append('.')

import argparse
import gc
import logging
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import torch
import torch.utils.data
import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def collect_episode_data(dataset: LeRobotDataset, episode_idx: int):
    episode_sampler = EpisodeSampler(dataset, episode_idx)
    episode_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=episode_sampler,
        num_workers=0,
        pin_memory=False
    )

    frames = []
    task_index = None
    language_instruction = None

    for batch in episode_dataloader:
        if task_index is None:
            task_index = batch["task_index"].item()
            language_instruction = batch['task']

        current_frame = {
            'observation.images.front': to_hwc_uint8_numpy(batch['observation.images.front'].squeeze(0)),
            'observation.images.wrist': to_hwc_uint8_numpy(batch['observation.images.wrist'].squeeze(0)),
            'action': batch['action'].squeeze(0).numpy().astype(np.float32),
            'observation.state': batch['observation.state'].squeeze(0).numpy().astype(np.float32),
            'timestamp': batch['timestamp'].squeeze(0).numpy().astype(np.float32),
        }
        frames.append(current_frame)

    return episode_idx, frames, task_index, language_instruction


def save_episode_hdf5(episode_data, hdf5_output_dir):
    episode_idx, frames, task_index, language_instruction = episode_data

    hdf5_episode_id = f"episode_{episode_idx:06d}"
    hdf5_output_dir = f'{hdf5_output_dir}/{language_instruction[0].replace(" ", "_")}'
    if not os.path.exists(hdf5_output_dir):
        os.makedirs(hdf5_output_dir, exist_ok=True)

    hdf5_path = os.path.join(hdf5_output_dir, f"{hdf5_episode_id}.hdf5")

    try:
        with h5py.File(hdf5_path, 'w') as f:
            group = f.create_group("obs")
            group.create_dataset("front_image", data=np.stack([f['observation.images.front'] for f in frames]), dtype='uint8')
            group.create_dataset("wrist_image", data=np.stack([f['observation.images.wrist'] for f in frames]), dtype='uint8')
            group.create_dataset("state", data=np.stack([f['observation.state'] for f in frames]), dtype='float32')

            f.create_dataset("action", data=np.stack([f['action'] for f in frames]), dtype='float32')
            f.create_dataset("timestamp", data=np.stack([f['timestamp'] for f in frames]), dtype='float32')

            f.attrs["task_index"] = task_index
            f.attrs["language_instruction"] = language_instruction

        print(f'Saved episode {episode_idx} to {hdf5_path}')
        return True, episode_idx, None
    except Exception as e:
        print(f'Error saving episode {episode_idx}: {str(e)}')
        return False, episode_idx, str(e)


def process_episode_worker(args):
    dataset, episode_idx, hdf5_output_dir = args
    try:
        episode_data = collect_episode_data(dataset, episode_idx)

        success, ep_idx, error = save_episode_hdf5(episode_data, hdf5_output_dir)

        del episode_data
        gc.collect()

        return success, ep_idx, error
    except Exception as e:
        print(f'Error processing episode {episode_idx}: {str(e)}')
        return False, episode_idx, str(e)


def process_dataset_multithreaded(dataset, hdf5_output_dir, max_workers=None):
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)

    print(f"Using {max_workers} threads to process {dataset.num_episodes} episodes")

    tasks = [(dataset, episode_idx, hdf5_output_dir) for episode_idx in range(dataset.num_episodes)]

    success_count = 0
    error_count = 0
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_episode = {executor.submit(process_episode_worker, task): task[1] for task in tasks}

        with tqdm.tqdm(total=len(tasks), desc="Processing episodes") as pbar:
            for future in as_completed(future_to_episode):
                episode_idx = future_to_episode[future]
                try:
                    success, ep_idx, error = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        errors.append((ep_idx, error))
                except Exception as e:
                    error_count += 1
                    errors.append((episode_idx, str(e)))

                pbar.update(1)
                pbar.set_postfix({
                    'Success': success_count,
                    'Errors': error_count
                })

    print(f"\nProcessing completed:")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")

    if errors:
        print(f"\nErrors encountered:")
        for ep_idx, error in errors[:10]:
            print(f"  Episode {ep_idx}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--episode_start_idx", type=int, default=None)
    parser.add_argument("--episode_end_idx", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of worker threads")
    args = parser.parse_args()

    if args.episode_start_idx is not None and args.episode_end_idx is not None:
        dataset = LeRobotDataset(
            None,
            root=f'{args.dataset_dir}/{args.task_name}',
            tolerance_s=1e-4,
            episodes=(args.episode_start_idx, args.episode_end_idx)
        )
    else:
        dataset = LeRobotDataset(
            None,
            root=f'{args.dataset_dir}/{args.task_name}',
            tolerance_s=1e-4
        )

    hdf5_output_dir = f'{args.save_dir}/{args.task_name}'
    os.makedirs(hdf5_output_dir, exist_ok=True)

    start_time = time.time()
    process_dataset_multithreaded(dataset, hdf5_output_dir, args.max_workers)
    end_time = time.time()

    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
