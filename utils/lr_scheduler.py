import math

from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(optimizer, n_epoch, n_iter_per_epoch, lr_min, warmup_steps, warmup_lr_init, decay_steps=None):
    if decay_steps is None:
        decay_steps = n_epoch * n_iter_per_epoch

    scheduler = CosineLRScheduler(optimizer, t_initial=decay_steps, lr_min=lr_min, warmup_t=warmup_steps, warmup_lr_init=warmup_lr_init,
                                  cycle_limit=1, t_in_epochs=False, warmup_prefix=True)

    return scheduler



def adjust_learning_rate(optimizer, it, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if it < args.warmup_iters:  # 1) linear warmup for warmup_iters steps
        lr = args.lr * it / args.warmup_iters
    elif it > args.lr_decay_iters:  # 2) if it > lr_decay_iters, return min learning rate
        lr = args.min_lr
    else:  # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        lr = args.min_lr + (args.lr - args.min_lr) * coeff

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate_epoch(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

