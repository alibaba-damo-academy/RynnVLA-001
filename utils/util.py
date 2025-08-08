import collections.abc
import logging
import os
import os.path as osp
import random
import sys
import time
from collections import OrderedDict
from datetime import datetime
from itertools import repeat
from shutil import get_terminal_size

import numpy as np
import torch
from safetensors.torch import load_file

logger = logging.getLogger('base')

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


def load_safetensor_model(model, model_path, strict=True):
    state_dict = load_file(model_path)

    retrain_params_prefix_list = ['vision_model', 'logit_scale', 'logit_bias']

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        for params_prefix in retrain_params_prefix_list:
            if params_prefix in key:
                new_state_dict[key] = value

    model.load_state_dict(new_state_dict, strict=strict)


def load_clip_model(model, model_path, strict=True):
    state_dict = torch.load(model_path)

    retrain_params_prefix_list = ['vision_model', 'logit_scale', 'visual_projection']

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        for params_prefix in retrain_params_prefix_list:
            if params_prefix in key:
                new_state_dict[key] = value

    model.load_state_dict(new_state_dict, strict=strict)


def load_pretrained_model(model, model_path, strict=True, params_prefix_list=None, replace_prefix_list=None):
    """Load model from a given path."""

    if osp.isfile(model_path):
        state_dict = torch.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if params_prefix_list is not None:
        new_state_dict = OrderedDict()

        for key, value in state_dict.items():
            for prefix_idx, params_prefix in enumerate(params_prefix_list):
                if key.startswith(f'{params_prefix}'):
                    if replace_prefix_list is not None:
                        new_key = key.replace(f'{params_prefix}', f'{replace_prefix_list[prefix_idx]}')
                    else:
                        new_key = key[len(f"{params_prefix}"):]
                    new_state_dict[new_key] = value

                    break

        model.load_state_dict(new_state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)


def get_time_str():
    # Get the current date and time
    now = datetime.now()

    # Format the current time
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    return formatted_time


def make_exp_dirs(path):
    """Make dirs for experiments."""
    
    os.makedirs(path, exist_ok=True if 'debug' in path else False)


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProgressBar(object):
    """A progress bar which can print the progress.

    Modified from:
    https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (
            bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(f'terminal width is too small ({terminal_width}), '
                  'please consider widen the terminal for better '
                  'progressbar visualization')
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write(f"[{' ' * self.bar_width}] 0/{self.task_num}, "
                             f'elapsed: 0s, ETA:\nStart...\n')
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write(
                '\033[J'
            )  # clean the output (remove extra chars since last display)
            sys.stdout.write(
                f'[{bar_chars}] {self.completed}/{self.task_num}, '
                f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, '
                f'ETA: {eta:5}s\n{msg}\n')
        else:
            sys.stdout.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s, '
                f'{fps:.1f} tasks/s')
        sys.stdout.flush()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # running average = running sum / running count
        self.sum = 0  # running sum
        self.count = 0  # running count

    def update(self, val, n=1):
        # n = batch_size

        # val = batch accuracy for an attribute
        # self.val = val

        # sum = 100 * accumulative correct predictions for this attribute
        self.sum += val * n

        # count = total samples so far
        self.count += n

        # avg = 100 * avg accuracy for this attribute
        # for all the batches so far
        self.avg = self.sum / self.count

