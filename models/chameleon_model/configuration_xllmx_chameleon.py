import logging
from typing import List

from .chameleon import ChameleonConfig

logger = logging.getLogger(__name__)


class RynnVLAConfig(ChameleonConfig):

    def __init__(
        self,
        z_loss_weight: float = 0.0,
        action_head_type: str = 'small',
        replace_inputs_with_action: bool = True,
        action_chunk_size: int = 10,
        action_dim: int = 7,
        visual_head_type: str = 'one_layer',
        state_dim: int=6,
        **kwargs,
    ):
        self.z_loss_weight = z_loss_weight
        self.action_head_type = action_head_type
        self.replace_inputs_with_action = replace_inputs_with_action
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.visual_head_type = visual_head_type
        self.state_dim = state_dim
        super().__init__(
            **kwargs,
        )
