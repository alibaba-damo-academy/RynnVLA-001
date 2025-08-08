import contextlib
import datetime
import functools
import gc
import json
import logging
import math
import os
import sys
import time
from abc import ABC
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import init_empty_weights
from einops import rearrange
from fairscale.nn.model_parallel import initialize as fs_init
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

import utils.ckpt as ckpt
import utils.lr_scheduler as lr_sched
import utils.misc as misc
from datasets import create_dataset
from datasets.sampler import NaiveDistSampler
from models.actionvae_model import ActionVAE
from models.chameleon_model import (RynnVLAConfig,
                                    RynnVLAXLLMXForActionPrediction,
                                    chameleon_vae_ori)
from models.chameleon_model.tokenizer import Tokenizer
from utils.dist_utils import (all_reduce_mean, init_distributed_mode,
                              promote_param_to_fp32)


class RynnVLAActionPredictionTrainer(ABC):

    def __init__(self, args):
        self.args = args

        self.load_pretrain = args.get('load_pretrain', True)

        init_distributed_mode(args)

        assert args.model_parallel_size == 1, (
            "Model parallelism currently not supported, ",
            "so please keep model_parallel_size to 1\n"
            "Note that model parallelism is different from and orthogonal to FSDP"
        )
        fs_init.initialize_model_parallel(args.model_parallel_size)

        self.global_rank = dist.get_rank()
        self.mp_rank = fs_init.get_model_parallel_rank()
        self.mp_world_size = fs_init.get_model_parallel_world_size()
        self.mp_group = fs_init.get_model_parallel_group()
        self.dp_rank = fs_init.get_data_parallel_rank()
        self.dp_world_size = fs_init.get_data_parallel_world_size()
        self.dp_group = fs_init.get_data_parallel_group()

        if args.exp_dir and self.global_rank == 0:
            Path(args.exp_dir).mkdir(parents=True, exist_ok=True)

        dist.barrier()

        self.logger = self.configure_logger()
        self.logger.info(args)

        if self.args.auto_resume and self.args.resume_path is None:
            existing_checkpoints = [_ for _ in os.listdir(self.args.exp_dir) if "epoch" in _]
            if len(existing_checkpoints) > 0:

                def ckpt_sort_key(s):
                    # divide ckpt directory names into epoch and iter parts
                    epoch, iteration = ckpt.split_ckpt_str_into_epoch_iter(s)
                    if iteration is None:
                        iteration = float("inf")
                    return epoch, iteration

                self.args.resume_path = os.path.join(
                    self.args.exp_dir, sorted(existing_checkpoints, key=ckpt_sort_key)[-1]
                )
                self.logger.info(f"auto resume from {self.args.resume_path}")



        if args.precision == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.logger.info("work dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        self.logger.info("{}".format(self.args).replace(", ", ",\n"))

        # define the model
        self.mixed_precision_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "tf32": torch.float32,
        }[self.args.precision]

        self.language_instruction_per_frame = args.get('language_instruction_per_frame', False)
        self.repeat_lang_tokens = args.get('repeat_lang_tokens', 1)
        self.language_first = args.get('language_first', True)
        self.predict_actions_forward = args.get('predict_actions_forward', False)
        self.action_chunk_size = args['train_dataset']['chunk_size']
        self.action_head_type = args.get('action_head_type', 'one_layer')
        self.replace_inputs_with_action = args.get('replace_inputs_with_action', True)
        self.is_rel_act = args.get('is_rel_act', False)

        self.state_dim = args.get('state_dim', 6)

        self.use_wrist_frame = args.get('use_wrist_frame', False)

        if self.use_wrist_frame:
            self.action_token_id = 65536
            self.wrist_start_token_id = 65537
            self.wrist_end_token_id = 65538

            self.state_token_id = 65539

            self.enlarged_vocab_size = 65536 + 4
        else:
            # define new_tokens, original token num 65536
            self.action_token_id = 65536
            # self.wrist_start_token_id = 65537
            # self.wrist_end_token_id = 65538

            self.state_token_id = 65539

            self.enlarged_vocab_size = 65536 + 4


        # TODO: update model
        self.model, self.tokenizer, self.optimizer = self.build_model()

        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()

        self.condition_frame_num = self.args.condition_frame_num

        # define image tokenizer
        self.image_start_token = "<racm3:break>"  # fixed tokens for start and end, so can hardcode
        self.image_end_token = "<eoss>"
        self.full_sub_sep_token = "<reserved08796>"
        self.sub_sub_sep_token = "<reserved08797>"
        self.sub_skip_token = "<reserved08798>"
        self.new_line_token = "<reserved08799>"

        self.chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
            json.load(open("./pretrained_models/Chameleon/original_tokenizers/text_tokenizer.json", encoding="utf8"))["model"]["vocab"]
        )
        self.chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(self.chameleon_ori_vocab, device=self.model.device)
        self.chameleon_ori_image_tokenizer = chameleon_vae_ori.ImageTokenizer(
            cfg_path="./pretrained_models/Chameleon/original_tokenizers/vqgan.yaml",
            ckpt_path="./pretrained_models/Chameleon/original_tokenizers/vqgan.ckpt",
            device=self.model.device,
        )

        # TODO: change to continuous
        # define action tokenizer
        self.action_tokenizer = ActionVAE(args['actionvae_config'])
        checkpoint = torch.load(args['actionvae_pretrained_path'])
        self.action_tokenizer.load_state_dict(checkpoint,strict=True)
        self.action_tokenizer = self.action_tokenizer.to(self.model.device)
        self.action_tokenizer.eval()

        self.dataset_train, self.sampler_train, self.dataloader_train = self.build_data()
        # self.dataset_val, self.sampler_val, self.dataloader_val = self.build_val_data()

        self.start_epoch = 0
        self.start_iter = 0
        self.metric_logger_to_resume = None

        if self.args.resume_path:
            self.resume(self.args.resume_path)

        if self.global_rank == 0:
            (Path(args.exp_dir) / "tensorboard").mkdir(parents=True, exist_ok=True)
            self.log_writer = SummaryWriter(log_dir=str(Path(args.exp_dir) / "tensorboard"))
        else:
            self.log_writer = None

        gc.collect()
        torch.cuda.empty_cache()

    def configure_logger(self):
        rank = dist.get_rank()

        logger = logging.getLogger()

        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()  # Console handler
        f_handler = logging.FileHandler(Path(self.args.exp_dir) / f"common.log")  # Rank-specific
        f_rank_handler = logging.FileHandler(
            Path(self.args.exp_dir) / f"rank-{dist.get_rank()}.log"
        )  # Rank-specific

        # Console and common file handler captures all INFO and above messages
        c_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        f_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        f_rank_handler.setLevel(logging.INFO)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter(f"[rank{rank}:%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        f_rank_handler.setFormatter(formatter)
        # Set the log level based on the rank argument

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        logger.addHandler(f_rank_handler)

        return logger

    def build_model(self):
        init_from = self.args.resume_path or self.args.init_from
        if init_from is None:
            starting_point_path = Path(self.args.exp_dir) / "starting_point"
            if dist.get_rank() == 0:
                if (starting_point_path / "config.json").exists():
                    self.logger.info(f"will use existing starting point at {starting_point_path}")
                    self.logger.info(
                        f"***********************************************************************\n"
                        f"********************************Caution********************************\n"
                        f"Caution: the starting point is created by some previous experiment run \n"
                        f"If the starting point saved by that run is broken, or if the expected  \n"
                        f"starting weights for the model has changed since that run, please manu-\n"
                        f"remove the saved path: \n"
                        f"{starting_point_path} \n"
                        f"and rerun the experiment.\n"
                        f"***********************************************************************\n"
                        f"***********************************************************************\n"
                    )
                else:
                    self.logger.info(f"creating starting-point weights at {starting_point_path}")
                    self._make_and_save_starting_point(save_path=str(starting_point_path))
            dist.barrier()
            init_from = str(starting_point_path)

        self.logger.info(f"Start instantiating unwrapped model from {init_from}")

        # only rank 0 instantiate, otherwise to meta
        unwrapped_model = self._model_func(init_from)

        tokenizer = Tokenizer(model_path='./pretrained_models/Chameleon')

        if hasattr(unwrapped_model, "get_trainable_params"):
            trainable_params = dict(unwrapped_model.get_trainable_params())
            for key, param in unwrapped_model.named_parameters():
                if key in trainable_params:
                    param.requires_grad = True
                    promote_param_to_fp32(param)
                else:
                    # TODO: maybe affect performance
                    param.requires_grad = False
                    promote_param_to_fp32(param)
                    # keep_fp32_keywords = ["norm", "lm_head", "embed_tokens"]
                    # if any([_ in key for _ in keep_fp32_keywords]):
                    #     promote_param_to_fp32(param)
                    # elif param.is_floating_point():
                    #     param.data = param.data.to(self.mixed_precision_dtype)
        else:
            self.logger.warning(
                f"model class {type(unwrapped_model)} does not have `get_trainable_params` method,"
                f"set all params to trainable"
            )
            for key, param in unwrapped_model.named_parameters():
                param.requires_grad = True
                param.requires_grad = True
                promote_param_to_fp32(param)

        self.logger.info("Finish instantiating unwrapped model.")
        self.logger.info(f"Unwrapped model: \n{str(unwrapped_model)}")
        self.logger.info(f"Model config: \n{unwrapped_model.config.to_dict()}")

        # ----------------
        self.is_peft = getattr(unwrapped_model, "is_peft", False)  # todo
        self.logger.info(f"Model is Peft: {self.is_peft}")
        # ----------------

        misc.mark_mp_params(unwrapped_model)

        # defer this after FSDP
        misc.print_param_status(unwrapped_model)

        train_param_count_local, train_param_count_all = 0, 0
        frozen_param_count_local, frozen_param_count_all = 0, 0
        for name, param in unwrapped_model.named_parameters():
            model_parallel = getattr(param, "model_parallel", False)
            if param.requires_grad:
                if model_parallel:
                    train_param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    train_param_count_all += param.numel()
                train_param_count_local += param.numel()
            else:
                if model_parallel:
                    frozen_param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    frozen_param_count_all += param.numel()
                frozen_param_count_local += param.numel()

        self.logger.info(
            f"Trainable parameter count : {train_param_count_local} (local rank), {train_param_count_all} (all).\n"
            f"Frozen parameter count : {frozen_param_count_local} (local rank), {frozen_param_count_all} (all)."
        )

        # checkpointing (part1, should be called before FSDP wrapping)
        if self.args.checkpointing:
            # todo more hints for not-implemented
            checkpointing_list = unwrapped_model.get_checkpointing_wrap_module_list()
        else:
            checkpointing_list = []

        # todo pre-sync ignored states
        model = self.setup_fsdp_sync(
            unwrapped_model, self.args.data_parallel, self.args.precision, self.args.grad_precision
        )

        # broadcast non-model-parallel parameters within model parallel group
        misc.broadcast_nonmp_parameters(model)

        # checkpointing (part2, after FSDP wrapping)
        if self.args.checkpointing:
            print("apply gradient checkpointing")
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: submodule in checkpointing_list,
            )

        self.logger.info(f"Wrapped model: \n{str(model)}")

        # Setup optimizer
        opt = AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.95))

        return model, tokenizer, opt

    def _model_func(
        self,
        init_from,
    ) -> (RynnVLAXLLMXForActionPrediction, None):

        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840

        if self.dp_rank == 0:
            if self.load_pretrain:
                model = RynnVLAXLLMXForActionPrediction.from_pretrained(
                        init_from,
                        max_position_embeddings=self.args.max_seq_len,
                        mask_image_logits=self.args.mask_image_logits,
                        dropout=self.args.dropout,
                        z_loss_weight=self.args.z_loss_weight,
                        action_head_type=self.action_head_type,
                        replace_inputs_with_action=self.replace_inputs_with_action,
                        torch_dtype=torch.bfloat16,
                        device_map="cpu",
                        state_dim=self.state_dim,
                    )
            else:
                self.logger.info(f"Randomly initialize the weights.")
                config = RynnVLAConfig.from_pretrained(
                    init_from,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    action_head_type=self.action_head_type,
                    replace_inputs_with_action=self.replace_inputs_with_action,
                    torch_dtype=torch.bfloat16,
                    state_dim=self.state_dim,
                )
                model = RynnVLAXLLMXForActionPrediction(config)

            if self.enlarged_vocab_size > model.model.embed_tokens.weight.size(0):
                model.resize_token_embeddings(self.enlarged_vocab_size)

        else:
            with init_empty_weights():
                config = RynnVLAConfig.from_pretrained(
                    init_from,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    action_head_type=self.action_head_type,
                    replace_inputs_with_action=self.replace_inputs_with_action,
                    torch_dtype=torch.bfloat16,
                    state_dim=self.state_dim,
                )
                model = RynnVLAXLLMXForActionPrediction(config)

                if self.enlarged_vocab_size > model.model.embed_tokens.weight.size(0):
                    model.resize_token_embeddings(self.enlarged_vocab_size)

        del model.model.vqmodel

        return model

    def _make_and_save_starting_point(self, save_path: str) -> None:

        raise NotImplementedError

    def setup_fsdp_sync(self, model: nn.Module, data_parallel: str, precision: str, grad_precision: Optional[str]) -> FSDP:

        if self.dp_rank == 0:
            param_init_fn = None
        else:
            param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)


        model = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
            ),
            process_group=fs_init.get_data_parallel_group(),
            sharding_strategy={
                "fsdp": ShardingStrategy.FULL_SHARD,
                "sdp": ShardingStrategy.SHARD_GRAD_OP,
            }[data_parallel],
            mixed_precision=MixedPrecision(
                param_dtype={
                    "fp32": torch.float,
                    "tf32": torch.float,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }[precision],
                reduce_dtype={
                    "fp32": torch.float,
                    "tf32": torch.float,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }[grad_precision or precision],
            ),
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            limit_all_gathers=True,
            use_orig_params=True,
            param_init_fn=param_init_fn

        )
        torch.cuda.synchronize()

        return model

    def build_data(self):

        dataset_train = create_dataset(self.args['train_dataset'])
        self.logger.info(dataset_train)

        eff_batch_size = self.args.batch_size * self.args.accum_iter * fs_init.get_data_parallel_world_size()
        self.logger.info("effective batch size: %d" % eff_batch_size)


        sampler_train = NaiveDistSampler(
            dataset_train,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True,
            batch_size=self.args.batch_size,
            acc_grad=self.args.accum_iter,
            seed=self.args.seed,
        )
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            sampler=sampler_train,
            collate_fn=lambda batch: {key: [item[key] for item in batch] for key in batch[0]},
            drop_last=True,
        )

        return dataset_train, sampler_train, dataloader_train


    def resume(self, resume_path: str):
        """
        Note: model ckpt is not loaded here because _model_func should already have met the resume path as init path
        """

        def _load_optimizer():
            opt_state_world_size = len(
                [x for x in os.listdir(resume_path) if x.startswith("optimizer.") and x.endswith(".pth")]
            )
            assert opt_state_world_size == dist.get_world_size(), (
                f"Resuming from a checkpoint with unmatched world size "
                f"({dist.get_world_size()} vs. {opt_state_world_size}) "
                f"is currently not supported."
            )
            self.logger.info(f"Resuming optimizer states from: {self.args.resume_path}")
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(
                        resume_path,
                        f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth",
                    ),
                    map_location="cpu",
                )
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.args.lr
                param_group["weight_decay"] = self.args.wd

        _load_optimizer()
        self.logger.info("Optimizer resume complete")

        resume_epoch, resume_iteration = ckpt.split_ckpt_str_into_epoch_iter(resume_path.split("/")[-1])

        if resume_iteration is None:
            self.start_epoch = resume_epoch + 1
            self.start_iter = 0
        else:
            self.start_epoch = resume_epoch
            self.start_iter = resume_iteration + 1

        self.logger.info(f"resume to epoch {self.start_epoch} iter {self.start_iter}")

        additional_rank_specific = os.path.join(
            resume_path, f"additional.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth"
        )
        if os.path.exists(additional_rank_specific):
            additional_rank_specific = torch.load(additional_rank_specific, map_location="cpu")
            if "metric_logger" in additional_rank_specific:
                self.metric_logger_to_resume = additional_rank_specific["metric_logger"]
                self.logger.info("metric logger resumed")

    def train(self):
        self.logger.info(f"Start training for {self.args.epochs} epochs")
        start_time = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            self.dataloader_train.sampler.set_epoch(epoch, self.start_iter)  # todo rename set_epoch

            train_stats = self.train_one_epoch(
                epoch,
                self.start_iter,
                log_writer=self.log_writer,
                metric_logger=self.metric_logger_to_resume,
            )

            if epoch + 1 == self.args.epochs:
                ckpt.save(
                    self.args.exp_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    max_keep=self.args.ckpt_max_keep,
                )

            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

            if self.global_rank == 0:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.exp_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

            self.start_iter = 0
            self.metric_logger_to_resume = None

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("Training time {}".format(total_time_str))

    def token2id(self, token):
        return self.tokenizer.tokenizer.vocab[token]

    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"


    @torch.no_grad()
    def quantize_batch_video(self, video_frames, start_token_id, end_token_id):
        videos = torch.stack(video_frames, dim=0)

        # quantize videos in a batch
        batch_size, num_frames, channel, image_height, image_width = videos.size()

        patch_size = 32
        h_grids, w_grids = image_height // patch_size, image_width // patch_size

        crop_h = h_grids * patch_size
        crop_w = w_grids * patch_size

        videos = rearrange(videos, 'b f c h w -> (b f) c h w')

        videos = videos[:, :, :crop_h, :crop_w]

        image_toks = self.chameleon_ori_translation.convert_img2bp2(self.chameleon_ori_image_tokenizer.img_tokens_from_tensor(videos.to(self.model.device))).view(-1)

        full_image_toks = rearrange(image_toks, '(b h w) -> b h w', b=batch_size * num_frames, h=image_height // 16, w=image_width // 16)

        new_line_id = self.token2id(self.new_line_token)

        full_image_toks = torch.cat((full_image_toks, torch.ones(batch_size * num_frames, image_height // 16, 1, device=full_image_toks.device, dtype=full_image_toks.dtype) * new_line_id,),dim=2,).flatten()

        full_image_toks = rearrange(full_image_toks, '(b f c) -> b f c', b=batch_size, f=num_frames)

        video_tokens_list, labels_list = [], []
        for video_idx in range(batch_size):
            video_tokens = []
            for frame_idx in range(num_frames):

                video_tokens += [
                    start_token_id,
                    self.token2id(self.get_n_grids_token(h_grids)),
                    self.token2id(self.get_n_grids_token(w_grids)),
                    *full_image_toks[video_idx][frame_idx].tolist(),
                    end_token_id,
                ]

            assert len(video_tokens) % num_frames == 0

            num_tokens_per_frame = len(video_tokens) // num_frames

            labels = video_tokens.copy()

            num_condition_tokens = self.condition_frame_num * num_tokens_per_frame

            labels[:num_condition_tokens] = [-100] * num_condition_tokens

            video_tokens_list.append(video_tokens)
            labels_list.append(labels)

        return video_tokens_list, labels_list
        return video_tokens_list, labels_list


    @torch.no_grad()
    def encode_batch_action_in_batch(self, actions, is_pad):
        num_videos = len(actions)
        actions = torch.stack(actions, dim=0)
        is_pad = torch.stack(is_pad, dim=0)

        num_videos, num_video_frames = actions.size(0), actions.size(1)

        actions = rearrange(actions, 'n f c d -> (n f) c d')
        is_pad = rearrange(is_pad, 'n f c-> (n f) c')

        action_embeds = self.action_tokenizer.encode_inputs(actions.to(self.model.device), is_pad.to(self.model.device), self.model.device)

        action_embeds = rearrange(action_embeds, '(n f) c -> n f c', n = num_videos, f = num_video_frames)

        return action_embeds

    @torch.no_grad()
    def encode_batch_action(self, actions, is_pad):

        action_embeds = self.encode_batch_action_in_batch(actions, is_pad)


        return action_embeds

    @torch.no_grad()
    def get_text_tokens(self, text):
        text_tokens_list = []
        text_labels_list = []
        for idx_in_batch, caption in enumerate(text):
            text_tokens = self.tokenizer.encode(caption, bos=True, eos=False)
            text_labels = [-100 for _ in text_tokens]

            text_tokens_list.append(text_tokens)
            text_labels_list.append(text_labels)

        return text_tokens_list, text_labels_list

    @torch.no_grad()
    def get_action_embeddings_from_whole_episode(self, action_whole_episode, video_frames):

        batch_size = len(video_frames)
        num_video_frames = video_frames[0].size(0)

        batch_actions = []
        batch_is_pad = []

        for batch_idx in range(batch_size):
            current_video_actions = []
            current_video_is_pad = []
            for frame_idx in range(num_video_frames):
                action_chunk = action_whole_episode[batch_idx][frame_idx]
                if action_chunk.size(0) == self.action_chunk_size:
                    current_video_actions.append(action_chunk)
                    is_pad = torch.zeros(self.action_chunk_size, dtype=torch.bool)
                    current_video_is_pad.append(is_pad)
                else:
                    raise NotImplementedError

            current_video_actions = torch.stack(current_video_actions, dim = 0)
            current_video_is_pad = torch.stack(current_video_is_pad, dim = 0)

            batch_actions.append(current_video_actions)
            batch_is_pad.append(current_video_is_pad)

        action_embeds = self.encode_batch_action(batch_actions, batch_is_pad)

        return action_embeds


    @torch.no_grad()
    def generate_inputs_labels(self, batch_data):
        video_examples, video_labels = self.quantize_batch_video(batch_data['video_frames'], self.token2id(self.image_start_token), self.token2id(self.image_end_token))

        if self.use_wrist_frame:
            wrist_video_examples, wrist_video_labels = self.quantize_batch_video(batch_data['wrist_frames'], self.wrist_start_token_id, self.wrist_end_token_id)

        action_whole_episode = batch_data["action_whole_episode"]
        action_embeds = self.get_action_embeddings_from_whole_episode(action_whole_episode, batch_data['video_frames'])
        if self.predict_actions_forward:
            forward_action_episode = [action_seq[self.action_chunk_size:] for action_seq in action_whole_episode]
            forward_action_embeds = self.get_action_embeddings_from_whole_episode(forward_action_episode, batch_data['video_frames'])
        text_examples, text_labels = self.get_text_tokens(batch_data['text'])
        num_videos = len(video_examples)

        state_episode = batch_data["state"]


        composed_examples_list, composed_labels_list = [], []
        action_embeds_list = []

        state_list = []

        for video_idx in range(num_videos):
            composed_examples, composed_labels = [], []

            num_frames = batch_data['video_frames'][video_idx].size(0)

            assert len(video_examples[video_idx]) % num_frames == 0
            # assert len(action_examples[video_idx]) % num_frames == 0

            video_examples_per_frame = len(video_examples[video_idx]) // num_frames

            if self.use_wrist_frame:
                wrist_video_examples_per_frame = len(wrist_video_examples[video_idx]) // num_frames

            # action_examples_per_frame = len(action_examples[video_idx]) // num_frames

            for frame_idx in range(num_frames):

                if self.language_first:
                    if frame_idx == 0:
                        composed_examples += text_examples[video_idx] * self.repeat_lang_tokens

                    if self.language_instruction_per_frame and frame_idx > 0:
                        composed_examples += text_examples[video_idx] * self.repeat_lang_tokens

                composed_examples += video_examples[video_idx][frame_idx * video_examples_per_frame: (frame_idx + 1) * video_examples_per_frame]

                if self.use_wrist_frame:
                    composed_examples += wrist_video_examples[video_idx][frame_idx * wrist_video_examples_per_frame: (frame_idx + 1) * wrist_video_examples_per_frame]


                composed_examples += [self.state_token_id]
                state_list += [state_episode[video_idx][frame_idx]]

                if not self.language_first:
                    if frame_idx == 0:
                        composed_examples += text_examples[video_idx] * self.repeat_lang_tokens

                    if self.language_instruction_per_frame and frame_idx > 0:
                        composed_examples += text_examples[video_idx] * self.repeat_lang_tokens

                composed_examples += [self.action_token_id]
                action_embeds_list += [action_embeds[video_idx][frame_idx]]

                if self.predict_actions_forward:
                    composed_examples += [self.action_token_id]
                    action_embeds_list += [forward_action_embeds[video_idx][frame_idx]]

                if self.language_first:
                    if frame_idx == 0:
                        composed_labels += text_labels[video_idx] * self.repeat_lang_tokens

                    if self.language_instruction_per_frame and frame_idx > 0:
                        composed_labels += text_examples[video_idx] * self.repeat_lang_tokens

                composed_labels += video_labels[video_idx][frame_idx * video_examples_per_frame: (frame_idx + 1) * video_examples_per_frame]

                if self.use_wrist_frame:
                    composed_labels += wrist_video_labels[video_idx][frame_idx * wrist_video_examples_per_frame: (frame_idx + 1) * wrist_video_examples_per_frame]


                composed_labels += [self.state_token_id]

                if not self.language_first:
                    if frame_idx == 0:
                        composed_labels += text_labels[video_idx] * self.repeat_lang_tokens

                    if self.language_instruction_per_frame and frame_idx > 0:
                        composed_labels += text_examples[video_idx] * self.repeat_lang_tokens

                composed_labels += [self.action_token_id]

                if self.predict_actions_forward:
                    composed_labels += [self.action_token_id]

            composed_examples_list.append(composed_examples)
            composed_labels_list.append(composed_labels)


        return composed_examples_list, composed_labels_list, action_embeds_list, state_list

    def train_one_epoch(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        self.model.train(True)
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        accum_iter = self.args.accum_iter
        accum_counter = 0

        self.optimizer.zero_grad()
        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_train,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):
            accum_counter = (accum_counter + 1) % accum_iter
            is_gradient_accumulation_boundary = accum_counter == 0



            with torch.no_grad():
                examples, labels, action_embeds, state_embeds = self.generate_inputs_labels(batch_data)


            if is_gradient_accumulation_boundary or data_iter_step == start_iter:
                lr_sched.adjust_learning_rate_epoch(
                    self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args
                )

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                # import pdb
                # pdb.set_trace()
                c_loss, action_loss, additional_loss_dict = self.model(examples, labels, action_embeds, state_embeds)

            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + action_loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            action_loss_value = action_loss.item()
            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            effective_loss = loss / accum_iter

            with (
                self.model.no_sync()
                if self.args.data_parallel in ["sdp", "hsdp"] and not is_gradient_accumulation_boundary
                else contextlib.nullcontext()
            ):
                effective_loss.backward()

            if is_gradient_accumulation_boundary:
                grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                metric_logger.update(grad_norm=grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(actionloss=action_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            for metric_name, metric in metric_logger.meters.items():
                metric_value = metric.value
                metric_value = all_reduce_mean(metric_value)
                if log_writer is not None:
                    log_writer.add_scalar(
                        metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch
                    )

            # save within epoch
            n_update_per_save = self.args.save_iteration_interval // accum_iter
            if (
                is_gradient_accumulation_boundary and ((data_iter_step + 1) // accum_iter) % n_update_per_save == 0
            ) or (data_iter_step + 1 == accum_iter and epoch == 0):
                ckpt.save(
                    self.args.exp_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    iteration=data_iter_step,
                    additional_rank_specific={
                        "metric_logger": metric_logger,
                    },
                    max_keep=self.args.ckpt_max_keep,
                )
        try:
            data_iter_step

            to_save=True
        except:
            to_save=False

        if to_save:
            ckpt.save(
                    self.args.exp_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    iteration=data_iter_step,
                    additional_rank_specific={
                        "metric_logger": metric_logger,
                    },
                    max_keep=self.args.ckpt_max_keep,
                    )

            torch.cuda.empty_cache()


        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
