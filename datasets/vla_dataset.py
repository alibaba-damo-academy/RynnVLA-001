import json
import random
import traceback
from dataclasses import asdict
from glob import glob
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TransformsF
from decord import VideoReader, cpu
from PIL import Image

from datasets.img_transform import (AugmentationCfg, PreprocessCfg,
                                    image_transform_v1, merge_preprocess_dict,
                                    merge_preprocess_kwargs)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def create_transforms(
    is_train,
    image_size,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    image_interpolation: Optional[str] = None,
    image_resize_mode: Optional[str] = None,  # only effective for inference
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
    ):

    force_preprocess_cfg = merge_preprocess_kwargs(
        {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode, size=image_size)

    preprocess_cfg = asdict(PreprocessCfg())

    pp_cfg = PreprocessCfg(**merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))


    if is_train:
        return image_transform_v1(
            pp_cfg,
            is_train=True,
            aug_cfg=aug_cfg,
        )
    else:
        return image_transform_v1(
            pp_cfg,
            is_train=False,
        )



def ConsistentVideoAug(frames, color_jitter_params, main_cam=True):  # list of PIL.Image
    orig_w, orig_h = frames[0].size
    if main_cam:
        crop_h, crop_w = int(orig_h * 0.95), int(orig_w * 0.95)
        i, j, h, w = T.RandomCrop.get_params(frames[0], output_size=(crop_h, crop_w))
        angle = random.uniform(-5, 5)

    brightness, contrast, saturation, hue, fn_idx = color_jitter_params

    augmented = []
    for img in frames:
        if main_cam:
            img = TransformsF.crop(img, i, j, h, w)
            img = TransformsF.resize(img, (orig_h, orig_w))
            # img = TransformsF.rotate(img, angle)
        for fn_id in fn_idx:
            if fn_id == 0:
                img = TransformsF.adjust_brightness(img, brightness)
            elif fn_id == 1:
                img = TransformsF.adjust_contrast(img, contrast)
            elif fn_id == 2:
                img = TransformsF.adjust_saturation(img, saturation)
            elif fn_id == 3:
                img = TransformsF.adjust_hue(img, hue)
        augmented.append(img)

    return augmented

class LeRobotVideoDataset(data.Dataset):

    def __init__(self, config):

        self.data_path = config['data_path']

        data = read_json(self.data_path)

        self.all_data = data

        self.use_rel_action = config['use_rel_action']

        self.min_max_norm = config['min_max_norm']
        self.mean_std_norm = config['mean_std_norm']

        assert (self.min_max_norm ^ self.mean_std_norm), "Only one of 'min_max_norm' or 'mean_std_norm' should be True"

        assert self.use_rel_action

        # if self.use_rel_action:
        self.rel_min_action = np.array(data['rel_min_action'])
        self.rel_max_action = np.array(data['rel_max_action'])

        # else:
        self.rel_mean_action = np.array(data['rel_mean_action'])
        self.rel_std_action = np.array(data['rel_std_action'])

        self.mean_state = np.array(data['mean_state'])
        self.std_state = np.array(data['std_state'])

        self.img_size = config['img_size']

        self.target_video_len = config['num_frames']

        self.chunk_size = config['chunk_size']

        self.is_train = config['is_train']

        self.num_chunks_forward = config.get('num_chunks_forward', 0)

        self.filter_data_threshold = config.get('filter_data_threshold', -1)

        # TODO: change img_size
        if config["is_train"]:
            self.transform = create_transforms(
                is_train=False,
                image_size=config["img_size"],
                image_mean=config.get("image_mean", None),
                image_std=config.get("image_std", None),
                aug_cfg=config['aug_cfg'])
        else:
            self.transform = create_transforms(
                is_train=False,
                image_mean=config.get("image_mean", None),
                image_std=config.get("image_std", None),
                image_size=config["img_size"],
                image_resize_mode=config.get("image_resize_mode", None))

        self.is_inference = config.get('is_inference', False)

        self.initialize_seq()

        assert len(self.traj_start_seq) == len(self.traj_total_end_seq)
        assert len(self.traj_start_seq) == len(self.traj_lang)

    def initialize_seq(self):
        self.traj_episode_path = []
        self.traj_start_seq = []
        self.traj_total_end_seq = []
        self.traj_lang = []

        for task_key in self.all_data['task_data'].keys():
            for episode_path in self.all_data['task_data'][task_key]['data_path']:
                with h5py.File(episode_path, 'r', locking=False) as root:
                    actions = np.array(root['action'])

                total_traj = actions.shape[0]

                start_idx = 0
                end_idx = total_traj - 1

                for sampled_start_idx in range(start_idx, end_idx - self.chunk_size -  self.target_video_len + 1):
                    self.traj_start_seq.append(sampled_start_idx)
                    self.traj_lang.append(self.all_data['task_data'][task_key]["instructions"])
                    self.traj_total_end_seq.append(end_idx)
                    self.traj_episode_path.append(episode_path)


    def __len__(self):
        return len(self.traj_start_seq)

    def __getitem__(self, idx):

        while True:
            try:
                start, end_idx, language_cond_list = self.traj_start_seq[idx], self.traj_total_end_seq[idx], self.traj_lang[idx]

                language_cond = random.choice(language_cond_list)

                episode_path = self.traj_episode_path[idx]

                episode_idx_list = list(
                    range(start,
                            min(end_idx + 1,
                                start + self.target_video_len + self.chunk_size * (1 + self.num_chunks_forward))
                            )
                    )

                frames = []
                wrist_frames = []
                actions_list = []
                state_list = []

                assert len(episode_idx_list) > 0

                with h5py.File(episode_path, 'r', locking=False) as root:
                    actions = np.array(root['action'])
                    state = np.array(root['obs/state'])

                    delta_actions = actions - np.append(actions[0][np.newaxis, :], actions[:-1], axis=0)

                    delta_state_action = actions - state

                    normed_state = (state - self.mean_state) / self.std_state

                    for episode_idx in episode_idx_list:
                        frames.append(torch.from_numpy(root['obs/front_image'][episode_idx]))
                        wrist_frames.append(torch.from_numpy(root['obs/wrist_image'][episode_idx]))
                        if self.use_rel_action:
                            rel_actions = np.concatenate([
                                np.concatenate([delta_state_action[episode_idx, :5][np.newaxis, :], actions[episode_idx, 5:][np.newaxis, :]], axis=1),
                                np.concatenate([delta_actions[episode_idx+1:episode_idx + self.chunk_size, :5], actions[episode_idx+1:episode_idx + self.chunk_size, 5:]], axis=1)
                            ], axis=0)

                        if (rel_actions.shape[0] < self.chunk_size) and len(actions_list) < self.target_video_len:
                            raise NotImplementedError

                        if self.mean_std_norm:
                            rel_actions = (rel_actions - self.rel_mean_action) / self.rel_std_action
                        else:
                            rel_actions = 2 * (rel_actions - self.rel_min_action) / (self.rel_max_action - self.rel_min_action + 1e-8) - 1

                        if len(actions_list) < self.target_video_len:
                            actions_list.append(rel_actions)
                            state_list.append(normed_state[episode_idx])
                        else:
                            break

                final_output_frames_num = min(self.target_video_len, len(frames))

                frames_final = []
                wrist_frames_final = []

                for frame_idx in range(final_output_frames_num):
                    frames_final.append(Image.fromarray(frames[frame_idx].numpy()).convert('RGB'))
                    wrist_frames_final.append(Image.fromarray(wrist_frames[frame_idx].numpy()).convert('RGB'))

                # Sample jitter parameters manually
                brightness = random.uniform(max(0, 1 - 0.3), 1 + 0.3)
                contrast = random.uniform(max(0, 1 - 0.4), 1 + 0.4)
                saturation = random.uniform(max(0, 1 - 0.5), 1 + 0.5)

                hue = random.uniform(-0.1, 0.1)  # default hue jitter
                fn_idx = torch.randperm(4)  # random order of color jitter ops

                colorjit_params = (brightness, contrast, saturation, hue, fn_idx)

                frames_final = ConsistentVideoAug(frames_final, colorjit_params, False)
                wrist_frames_final = ConsistentVideoAug(wrist_frames_final, colorjit_params, False)

                for frame_idx in range(final_output_frames_num):
                    frames_final[frame_idx] = self.transform(frames_final[frame_idx])
                    wrist_frames_final[frame_idx] = self.transform(wrist_frames_final[frame_idx])

                action_whole_episode = torch.from_numpy(np.array(actions_list).astype(np.float32))
                frames_final = torch.stack(frames_final, dim=0)
                wrist_frames_final = torch.stack(wrist_frames_final, dim=0)
                states_episode = torch.from_numpy(np.array(state_list).astype(np.float32))

                return {"video_frames": frames_final,
                        "wrist_frames": wrist_frames_final,
                        "action_whole_episode": action_whole_episode,
                        "text": language_cond,
                        "state": states_episode
                        }

            except Exception as e:
                print("An error occurred:", e)
                traceback.print_exc()

                idx = random.randint(0, len(self.traj_start_seq) - 1)
