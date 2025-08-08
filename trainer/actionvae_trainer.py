import json
import logging
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (DistributedDataParallelKwargs,
                              ProjectConfiguration)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import create_dataset
from models.actionvae_model import ActionVAE
from utils.logger import Log
from utils.lr_scheduler import build_scheduler
from utils.util import make_exp_dirs, set_random_seed

logger = get_logger(__name__, log_level="INFO")


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


class ActionVAETrainer(nn.Module):

    def __init__(self, configs):
        super().__init__()

        accelerator_project_config = ProjectConfiguration(project_dir=configs['exp_dir'], logging_dir=f"{configs['exp_dir']}/log_dir")

        if torch.cuda.device_count() > 1:
            print('multi-gpu')
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

            self.accelerator = Accelerator(
                mixed_precision=configs['trainer_config']['mixed_precision'],
                gradient_accumulation_steps=configs['trainer_config']['grad_accum_steps'],
                log_with=configs['trainer_config']['log_with'],
                project_config=accelerator_project_config,
                kwargs_handlers=[ddp_kwargs])
        else:
            self.accelerator = Accelerator(
                mixed_precision=configs['trainer_config']['mixed_precision'],
                gradient_accumulation_steps=configs['trainer_config']['grad_accum_steps'],
                log_with=configs['trainer_config']['log_with'],
                project_config=accelerator_project_config)

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        logger.info(self.accelerator.state, main_process_only=False)

        # If passed along, set the training seed now.
        if configs['seed'] is not None:
            set_random_seed(configs['seed'] + self.accelerator.process_index)

        # Handle the repository creation
        if self.accelerator.is_main_process:
            make_exp_dirs(configs['exp_dir'])

            with open(f'{configs["exp_dir"]}/config.json', 'w') as json_file:
                json.dump(configs, json_file, indent=4)

        self.policy_model = ActionVAE(configs['actionvae_config'])

        # define dataloader
        self.train_ds = create_dataset(configs['train_dataset'])
        self.valid_ds = create_dataset(configs['val_dataset'])

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=configs['train_dataset']['batch_size'],
            shuffle=True, num_workers=configs['train_dataset']['num_workers'],
            pin_memory=configs['train_dataset']['pin_memory']
        )
        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size=configs['val_dataset']['batch_size'],
            shuffle=False, num_workers=configs['val_dataset']['num_workers'],
            pin_memory=configs['val_dataset']['pin_memory'])

        # define optimizer
        param_dicts = [
            {"params": [p for n, p in self.policy_model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.policy_model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": configs['trainer_config']['lr_backbone'],
            },
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=configs['trainer_config']['lr'],
                                           weight_decay=configs['trainer_config']['weight_decay'])

        # TODO: check scheduler
        self.sched = build_scheduler(self.optimizer,
            configs['trainer_config']['num_epoch'],
            len(self.train_dl),
            configs['trainer_config']['lr_min'],
            configs['trainer_config']['warmup_steps'],
            configs['trainer_config']['warmup_lr_init'],
            configs['trainer_config']['decay_steps']
        )

        self.kl_weight = configs['kl_weight']

        (
            self.policy_model,
            self.optimizer,
            self.sched,
            self.train_dl,
            self.valid_dl,
        ) = self.accelerator.prepare(
            self.policy_model,
            self.optimizer,
            self.sched,
            self.train_dl,
            self.valid_dl,
        )

        self.num_epoch = configs['trainer_config']['num_epoch']
        self.save_every = configs['trainer_config']['save_every']
        self.samp_every = configs['trainer_config']['sample_every']
        self.max_grad_norm = configs['trainer_config']['max_grad_norm']

        self.model_saved_dir = os.path.join(configs['exp_dir'], 'models')
        os.makedirs(self.model_saved_dir, exist_ok=True)

        self.image_saved_dir = os.path.join(configs['exp_dir'], 'images')
        os.makedirs(self.image_saved_dir, exist_ok=True)

        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'number of learnable parameters: {n_parameters/1e6}M')

        logger.info(f'Num of training data: {len(self.train_ds)}')
        logger.info(f'Num of validation data: {len(self.valid_ds)}')

        self.evaluate()

    @property
    def device(self):
        return self.accelerator.device

    def forward(self):
        pass

    def forward_pass(self, data):
        actions, is_pad = data

        if actions is not None: # training time
            if isinstance(self.policy_model, torch.nn.parallel.DistributedDataParallel):
                actions = actions[:, :self.policy_model.module.num_queries]
                is_pad = is_pad[:, :self.policy_model.module.num_queries]
            else:
                actions = actions[:, :self.policy_model.num_queries]
                is_pad = is_pad[:, :self.policy_model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.policy_model(actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            raise NotImplementedError

    def train(self):
        self.steps = 0
        self.accelerator.init_trackers("cvae")
        self.log = Log()
        self.policy_model.train()
        for epoch in range(self.num_epoch):
            with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
                for batch_idx, data in enumerate(train_dl):
                    with self.accelerator.accumulate(self.policy_model):
                        with self.accelerator.autocast():
                            loss_dict = self.forward_pass(data)
                            # backward
                            loss = loss_dict['loss']

                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients and self.max_grad_norm is not None:
                                    self.accelerator.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                            self.optimizer.step()
                            self.sched.step_update(self.steps)
                            self.optimizer.zero_grad()

                    self.steps += 1

                    self.log.update({'loss': loss.item() if torch.is_tensor(loss) else loss,
                                     'l1 loss': loss_dict['l1'].item() if torch.is_tensor(loss_dict['l1']) else loss_dict['l1'],
                                     'kl loss': loss_dict['kl'].item() if torch.is_tensor(loss_dict['kl']) else loss_dict['kl'],
                                     'lr': self.optimizer.param_groups[0]['lr']})

                    train_dl.set_postfix(
                        ordered_dict={
                                "epoch"               : epoch,
                                "loss"                : self.log['loss'],
                                "l1 loss"             : self.log['l1 loss'],
                                "kl loss"             : self.log['kl loss'],
                                "lr"                  : self.log['lr']
                            }
                    )
                    self.accelerator.log(
                            {
                                "loss"                : self.log['loss'],
                                "l1 loss"             : self.log['l1 loss'],
                                "kl loss"             : self.log['kl loss'],
                                "lr"                  : self.log['lr']
                            },
                            step=self.steps
                        )

                    if not (self.steps % self.save_every):
                        self.save()

                    # if (not (self.steps % self.samp_every)) and self.accelerator.is_main_process:
                    if not (self.steps % self.samp_every):
                        self.evaluate()

        self.accelerator.end_training()
        print("Train finished!")

    def save(self):
        self.accelerator.wait_for_everyone()
        state_dict = self.accelerator.unwrap_model(self.policy_model).state_dict()
        self.accelerator.save(state_dict, os.path.join(self.model_saved_dir, f'step_{self.steps}.pt'))

    @torch.no_grad()
    def evaluate(self):
        self.policy_model.eval()

        if not hasattr(self, 'steps'):
            self.steps = 0
            self.accelerator.init_trackers("cvae")


        with tqdm(self.valid_dl, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process) as valid_dl:
            epoch_dicts = []
            for batch_idx, data in enumerate(self.valid_dl):
                forward_dict = self.forward_pass(data)
                epoch_dicts.append(forward_dict)

            epoch_summary = compute_dict_mean(epoch_dicts)

            epoch_val_loss = epoch_summary['loss']

            if self.accelerator.is_main_process:
                for tracker in self.accelerator.trackers:
                    if tracker.name == 'tensorboard':
                        tracker.writer.add_scalar('validation/rec_loss', epoch_val_loss, self.steps)

        self.policy_model.train()

