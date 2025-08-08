import argparse
import os

import yaml
from easydict import EasyDict

from trainer import create_trainer


def main(configs):

    trainer = create_trainer(configs)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help='Path to config file.', type=str, required=True)
    parser.add_argument("--exp_dir", default='./experiments', type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    with open(args.config_file, encoding="utf-8") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    opt = EasyDict(opt)

    opt['exp_dir'] = f'{args.exp_dir}/{opt["exp_name"]}'

    main(opt)
