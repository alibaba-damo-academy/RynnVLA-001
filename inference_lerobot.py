import argparse

import torch
import yaml
from easydict import EasyDict

from inferencer.lerobot_inferencer import LeRobotInferencer


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument('--vla_ckpt_path', type=str, help="Path to the evaluating checkpoint")
    parser.add_argument('--action_vae_path', type=str)
    parser.add_argument('--configs_path', type=str, help="Path to the config json file")
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument('--device', default=0, type=int, help="CUDA device")
    args = parser.parse_args()


    # define configs
    with open(args.configs_path, encoding="utf-8") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    opt = EasyDict(opt)

    opt['actionvae_pretrained_path'] = args.action_vae_path

    device = torch.device('cuda:0')

    # define models
    vla_model = LeRobotInferencer(args.vla_ckpt_path,
                                  args.precision,
                                  opt,
                                  device)


    img = None
    wrist_img = None
    lang_annotation = None
    state = None

    obs = {
        'rgb_obs': {
            'rgb_static': img,
            'wrist_static': wrist_img,
        },
        'state': state
        }

    # example of one forward pass
    _, action = vla_model.step(obs, lang_annotation)


if __name__ == "__main__":
    main()

