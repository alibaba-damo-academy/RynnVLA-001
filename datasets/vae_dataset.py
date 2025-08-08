import json
import random
from glob import glob

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

class LeRobotVAEData(data.Dataset):
    def __init__(self, args):
        self.data_path = args.data_path

        data = read_json(self.data_path)

        self.all_data = data


        self.all_data_path = []

        for task_key in self.all_data['task_data'].keys():
            self.all_data_path += self.all_data['task_data'][task_key]['data_path']

        self.val = args.val

        if self.val:
            self.all_data_path = self.all_data_path[0:1]

        self.chunk_size = args.chunk_size

        self.rel_mean_action = np.array(data['rel_mean_action'])
        self.rel_std_action = np.array(data['rel_std_action'])

        self.initialize_seq()
        assert len(self.traj_start_seq) == len(self.traj_total_end_seq)
        assert len(self.traj_start_seq) == len(self.traj_episode_path)

    def initialize_seq(self):
        self.traj_episode_path = []
        self.traj_start_seq = []
        self.traj_total_end_seq = []

        for episode_path in self.all_data_path:
            with h5py.File(episode_path, 'r', locking=False) as root:
                actions = np.array(root['action'])

            total_traj = actions.shape[0]

            start_idx = 0
            end_idx = total_traj - 1

            for sampled_start_idx in range(start_idx, end_idx - self.chunk_size + 1):
                self.traj_start_seq.append(sampled_start_idx)
                self.traj_total_end_seq.append(end_idx)
                self.traj_episode_path.append(episode_path)

    def __len__(self):
        return len(self.traj_start_seq)

    def __getitem__(self, idx):
        while True:
            try:
                start_idx, end_idx = self.traj_start_seq[idx], self.traj_total_end_seq[idx],

                episode_path = self.traj_episode_path[idx]

                actions_list = []

                with h5py.File(episode_path, 'r', locking=False) as root:
                    actions = np.array(root['action'])
                    state = np.array(root['obs/state'])

                    delta_actions = actions - np.append(actions[0][np.newaxis, :], actions[:-1], axis=0)

                    delta_state_action = actions - state

                    rel_actions = np.concatenate([
                        np.concatenate([delta_state_action[start_idx, :5][np.newaxis, :], actions[start_idx, 5:][np.newaxis, :]], axis=1),
                        np.concatenate([delta_actions[start_idx+1:start_idx + self.chunk_size, :5], actions[start_idx+1:start_idx + self.chunk_size, 5:]], axis=1)
                    ], axis=0)

                    rel_actions = (rel_actions - self.rel_mean_action) / self.rel_std_action

                padded_action = np.zeros((self.chunk_size, rel_actions.shape[1]), dtype=np.float32)

                action_len = rel_actions.shape[0]

                rel_actions = np.array(rel_actions).astype(np.float32)

                padded_action[:action_len] = rel_actions
                padded_action[action_len:, -1] = rel_actions[-1][-1]

                is_pad = np.zeros(self.chunk_size)
                is_pad[action_len:] = 1

                action_data = torch.from_numpy(padded_action)
                is_pad = torch.from_numpy(is_pad).bool()

                return action_data, is_pad
            except Exception as e:
                print("An error occurred:", e)

                idx = random.randint(0, len(self.traj_start_seq) - 1)
