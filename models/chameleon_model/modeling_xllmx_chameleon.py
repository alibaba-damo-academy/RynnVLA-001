import functools
import logging
import math
from typing import List

import torch
from torch import nn

from .chameleon import RynnVLAForActionPrediction
from .configuration_xllmx_chameleon import RynnVLAConfig

logger = logging.getLogger(__name__)

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))


class RynnVLAXLLMXForActionPrediction(RynnVLAForActionPrediction):
    config_class = RynnVLAConfig

    def __init__(self, config):
        super().__init__(config)

    def get_trainable_params(self):
        params_keys = []
        for key, param in self.named_parameters():
            if param.requires_grad:
                params_keys.append((key, param))

        return params_keys

    def forward(self, input_ids=None, labels=None, action_embeds=None, state_embeds=None, training=True, **kwargs):

        max_tokens = max([len(_) for _ in input_ids])
        max_tokens = min(max_tokens, self.config.max_position_embeddings)
        input_ids = [_[:max_tokens] for _ in input_ids]
        labels = [_[:max_tokens] for _ in labels]

        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)

        labels = [label + [-100] * (max_tokens - len(label)) for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)

        # explicit use_cache=False for the following
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19267
        result = RynnVLAForActionPrediction.forward(
            self, input_ids=input_ids, labels=labels, use_cache=False, action_embeds=action_embeds, state_embeds=state_embeds, **kwargs
        )

        c_loss = result[0]
        action_loss = result[1]

        additional_loss_dict = {}
        if self.config.z_loss_weight > 0:
            logits: torch.Tensor = result[2]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels >= 0
            z_loss = torch.logsumexp(shift_logits, dim=-1).pow(2)[valid_mask].mean()
            additional_loss_dict["z_loss"] = (z_loss, self.config.z_loss_weight)
        return c_loss, action_loss, additional_loss_dict

    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.layers), self.lm_head, self.model.embed_tokens]
        if hasattr(self.model, "vqmodel"):  # may be deleted
            modules.append(self.model.vqmodel)
        return modules

    def get_checkpointing_wrap_module_list(self) -> List:
        modules = [
            *list(self.model.layers),
        ]
        return modules

