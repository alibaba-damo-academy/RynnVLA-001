import json

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (Compose, InterpolationMode, Normalize,
                                    Resize, ToTensor)
from transformers import GenerationConfig

from models.actionvae_model import ActionVAE
from models.chameleon_model import chameleon_vae_ori
from models.chameleon_model.chameleon import RynnVLAForActionPrediction
from models.chameleon_model.tokenizer import Tokenizer


def _convert_to_rgb(image):
    return image.convert('RGB')


class BaseModel:
    """
    Base class for all models.
    """

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        raise NotImplementedError

    def step(self, obs, goal):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """



class LeRobotInferencer(BaseModel):
    def __init__(self,
                 pretrained_policy_ckpt,
                 precision,
                 configs,
                 device,
                 condition_frame_num = 1):

        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        self.model = RynnVLAForActionPrediction.from_pretrained(
            pretrained_policy_ckpt,
            torch_dtype=self.dtype,
            device_map=device,
        )

        self.patch_size = 32

        self.condition_frame_num = condition_frame_num

        # define tokenizer
        self.image_start_token = "<racm3:break>"  # fixed tokens for start and end, so can hardcode
        self.image_end_token = "<eoss>"
        self.full_sub_sep_token = "<reserved08796>"
        self.sub_sub_sep_token = "<reserved08797>"
        self.sub_skip_token = "<reserved08798>"
        self.new_line_token = "<reserved08799>"

        self.action_token_id = 65536
        self.wrist_start_token_id = 65537
        self.wrist_end_token_id = 65538
        self.state_token_id = 65539

        self.chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
            json.load(open("./pretrained_models/Chameleon/original_tokenizers/text_tokenizer.json", encoding="utf8"))["model"]["vocab"]
        )
        self.chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(self.chameleon_ori_vocab, device=self.model.model.vqmodel.device)
        self.chameleon_ori_image_tokenizer = chameleon_vae_ori.ImageTokenizer(
            cfg_path="./pretrained_models/Chameleon/original_tokenizers/vqgan.yaml",
            ckpt_path="./pretrained_models/Chameleon/original_tokenizers/vqgan.ckpt",
            device=self.model.model.vqmodel.device,
        )

        self.tokenizer = Tokenizer(model_path='./pretrained_models/Chameleon')

        self.tokenizer.tokenizer.add_tokens('<|image|>')

        self.action_tokenizer = ActionVAE(configs['actionvae_config'])
        checkpoint = torch.load(configs['actionvae_pretrained_path'])
        self.action_tokenizer.load_state_dict(checkpoint,strict=True)
        self.action_tokenizer = self.action_tokenizer.to(self.model.device)
        self.action_tokenizer.eval()


        # image transformations
        normalize = Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])


        self.image_transforms = Compose([
            Resize(configs['train_dataset']['img_size'], interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])

        try:
            self.scale_p = configs['train_dataset']['scale_p']
        except:
            pass

        self.use_rel_action = configs['train_dataset']['use_rel_action']

        self.min_max_norm = configs['train_dataset']['min_max_norm']
        self.mean_std_norm = configs['train_dataset']['mean_std_norm']

        # action normalization params
        with open(configs['train_dataset']['data_path'], "r") as f:
            action_stats = json.load(f)

        if self.use_rel_action:
            self.min_action = np.array(action_stats['rel_min_action'])
            self.max_action = np.array(action_stats['rel_max_action'])
            self.mean_action = np.array(action_stats['rel_mean_action'])
            self.std_action = np.array(action_stats['rel_std_action'])
        else:
            self.min_action = np.array(action_stats['min_action'])
            self.max_action = np.array(action_stats['max_action'])
            self.mean_action = np.array(action_stats['mean_action'])
            self.std_action = np.array(action_stats['std_action'])

        self.mean_state = np.array(action_stats['mean_state'])
        self.std_state = np.array(action_stats['std_state'])

        self.repeat_lang_tokens = configs.get('repeat_lang_tokens', 1)
        self.language_first = configs.get('language_first', True)
        self.predict_actions_forward = configs.get('predict_actions_forward', False)

        self.action_chunk_size = configs.get('action_chunk_size', 20)
        self.action_dim = configs.get('action_dim', 6)

    def reset(self):
        """Reset function."""
        self.rgb_list = []
        self.hand_rgb_list = []
        self.state_list = []
        self.rollout_step_counter = 0

    def token2id(self, token):
        return self.tokenizer.tokenizer.vocab[token]

    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"

    @torch.no_grad()
    def quantize_images(self, image, start_token_id, end_token_id):
        patch_size = 32
        h_grids, w_grids = image.size()[1] // patch_size, image.size()[2] // patch_size

        crop_h = h_grids * patch_size
        crop_w = w_grids * patch_size

        image = image[:, :crop_h, :crop_w].unsqueeze(0)

        image_toks = self.chameleon_ori_translation.convert_img2bp2(
            self.chameleon_ori_image_tokenizer.img_tokens_from_tensor(image.to(self.model.device))
        ).view(-1)

        full_image_toks = image_toks.reshape(image.size(2) // 16, image.size(3) // 16)
        new_line_id = self.token2id(self.new_line_token)

        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.ones(image.size(2) // 16, 1, device=full_image_toks.device, dtype=full_image_toks.dtype)
                * new_line_id,
            ),
            dim=1,
        ).flatten()

        result_toks = [
            start_token_id,
            self.token2id(self.get_n_grids_token(h_grids)),
            self.token2id(self.get_n_grids_token(w_grids)),
            *full_image_toks.tolist(),
            end_token_id,
        ]

        return result_toks

    def get_condition_tokens(self, text, image, wrist_image=None):
        text_tokens = self.tokenizer.encode(text, bos=True, eos=False)

        image_tokens = self.quantize_images(image, start_token_id=self.token2id(self.image_start_token), end_token_id=self.token2id(self.image_end_token))

        if wrist_image is None:
            return text_tokens, image_tokens
        else:
            wrist_image_tokens = self.quantize_images(wrist_image, start_token_id=self.wrist_start_token_id, end_token_id=self.wrist_end_token_id)

            return text_tokens, image_tokens, wrist_image_tokens


    @torch.no_grad()
    def step(self, obs, text):

        # RGB observation
        image = Image.fromarray(obs['rgb_obs']['rgb_static'])

        image = self.image_transforms(image).to(self.model.device).to(torch.float32)


        wrist_image = Image.fromarray(obs['rgb_obs']['wrist_static'])

        wrist_image = self.image_transforms(wrist_image).to(self.model.device).to(torch.float32)

        text_tokens, image_tokens, wrist_image_tokens = self.get_condition_tokens(text, image, wrist_image)

        condition_tokens = text_tokens * self.repeat_lang_tokens + image_tokens + wrist_image_tokens + [self.state_token_id]

        ori_state = obs['state'].copy()

        ori_state = (ori_state - self.mean_state) / self.std_state

        state_embeds = [torch.tensor(ori_state, dtype=torch.bfloat16, device=self.model.device)]

        # define generation configs
        generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True
        )

        condition_tokens = torch.tensor(condition_tokens, dtype=torch.int64, device=self.model.device).unsqueeze(0)

        # inference forward pass
        with torch.cuda.amp.autocast(dtype=self.dtype):
            generation_result, action_pred = self.model.generate(
                condition_tokens, state_embeds, generation_config
            )

            generation_result = generation_result[0][1:]
            action_pred = action_pred[0]

        action_selected = (generation_result == self.action_token_id)
        action_pred = action_pred[action_selected]

        action_pred, _ = self.action_tokenizer.decode_inputs(action_pred)

        # decode actions
        try:
            decode_actions = action_pred.view(self.action_chunk_size, self.action_dim).to(torch.float32)
        except:
            print("no_pred_error")
            decode_actions = torch.zeros((self.action_chunk_size, self.action_dim)).to(torch.float32)

        decode_actions = decode_actions.cpu().numpy()

        # decode actions denormalization
        if self.min_max_norm:
            decode_actions = (decode_actions + 1) / 2
            decode_actions = (self.max_action - self.min_action + 1e-8) * decode_actions + self.min_action
        elif self.mean_std_norm:
            decode_actions = decode_actions * self.std_action + self.mean_action

        decode_actions[:, :5] = np.cumsum(decode_actions[:, :5], axis=0)

        decode_actions[:, :5]= decode_actions[:, :5] + obs['state'][:5]

        decode_actions = [decode_actions]

        torch.cuda.empty_cache()

        if len(decode_actions) == 0:
            output_actions = None
        elif len(decode_actions) > 0:
            output_actions = []
            last_action = None
            for decode_action in decode_actions:
                if last_action is not None:
                    if torch.sum(torch.abs(last_action - decode_action)) != 0:
                        output_actions.append(decode_action)

                if last_action is None:
                    output_actions.append(decode_action)

                last_action = decode_action

            output_actions = np.concatenate(output_actions, axis=0)

        return None, output_actions


