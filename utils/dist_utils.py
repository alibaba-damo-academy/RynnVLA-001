import datetime
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


def random_seed(seed=0):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def init_distributed_mode(args=SimpleNamespace()):
    random_seed(getattr(args, "seed", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.local_rank = args.gpu
        args.dist_url = "env://"
    elif "SLURM_PROCID" in os.environ:
        os.environ["MASTER_PORT"] = "8966"
        while "MASTER_ADDR" not in os.environ or len(os.environ["MASTER_ADDR"].strip()) == 0:
            os.environ["MASTER_ADDR"] = (
                subprocess.check_output(
                    "sinfo -Nh -n %s | head -n 1 | awk '{print $1}'" % os.environ["SLURM_NODELIST"],
                    shell=True,
                )
                .decode()
                .strip()
            )
            time.sleep(1)
        print(os.environ["MASTER_ADDR"])
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        args.local_rank = args.gpu
        args.dist_url = "env://"
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["RANK"] = str(args.rank)
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port(9000, 10000))
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        args.rank = 0
        args.gpu = args.local_rank = 0
        args.world_size = 1
        args.dist_url = "env://"

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}, gpu {}".format(args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(seconds=2 * 60 * 60),
    )
    torch.distributed.barrier()


def promote_param_to_fp32(param: nn.Parameter) -> None:
    if param.is_floating_point() and torch.finfo(param.dtype).bits < 32:
        param.data = param.data.float()
    if param.is_complex() and torch.finfo(param.dtype).bits < 32:
        param.data = param.data.to(torch.complex64)


def all_reduce_mean(x, group=None):
    world_size = dist.get_world_size(group=group)
    if world_size > 1:
        if isinstance(x, torch.Tensor):
            x_reduce = x.clone().cuda()
        else:
            x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce, group=group)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
