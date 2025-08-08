import logging
from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


# todo too slow to be used
def mild_shuffle(items: List, shuffle_factor, engine: np.random.Generator):
    """
    Perform a mild shuffle on the list of items.

    Args:
        engine: random engine
        items (list): The list of items to shuffle.
        shuffle_factor (float): max swap range is computed as len(item) * shuffle_factor.

    Returns:
        list: The mildly shuffled list.
    """

    n = len(items)
    swap_range = int(shuffle_factor * n)
    shuffled_items = [None for _ in items]
    cache = list(range(swap_range))
    for i in range(n):
        if i + swap_range < n:
            cache.append(i + swap_range)
        if len(cache) == 0 or cache[0] != i:  # already swapped
            assert shuffled_items[i] is not None
            continue
        else:
            cache = cache[1:]
            if len(cache) == 0:
                shuffled_items[i] = items[i]
            else:
                cache_idx = engine.integers(low=0, high=len(cache))
                j = cache[cache_idx]
                del cache[cache_idx]
                shuffled_items[i], shuffled_items[j] = items[j], items[i]

    return shuffled_items


class NaiveDistSampler(Sampler):
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        batch_size=None,
        acc_grad=1,
        allow_mixed_task_among_acc=False,
    ):
        """
        Distributed Sampler ensuring data in a batch are of the same type (e.g. text, image-text)
        :param dataset:
        :param num_replicas:
        :param rank:
        :param shuffle:
        :param seed:
        :param batch_size:
        :param acc_grad:
        :param allow_mixed_task_among_acc:
        """
        # super().__init__()

        if num_replicas is None or rank is None or rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid num_replicas ({num_replicas}) or rank ({rank})")
        assert batch_size is not None

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.acc_grad = acc_grad
        self.allow_mixed_task_among_acc = allow_mixed_task_among_acc

        self.epoch = 0
        self.start_iter = 0

        global_bsz_acc = batch_size * num_replicas * acc_grad

        # group_len = defaultdict(int)
        # for i, meta in enumerate(dataset.meta_collection):
        #     group_len[meta["type"]] += int(meta["len"] * meta.get("ratio", 1.0))

        # group_len = {key: val // global_bsz_acc * global_bsz_acc for key, val in group_len.items()}
        # self.total_size = sum(list(group_len.values()))

        self.total_size = len(dataset) // global_bsz_acc * global_bsz_acc
        assert self.total_size % num_replicas == 0
        self.num_samples = self.total_size // num_replicas

    def __iter__(self) -> Iterator:
        global_batch_size = self.batch_size * self.num_replicas
        global_bsz_acc = self.batch_size * self.num_replicas * self.acc_grad
        rng = np.random.default_rng(self.seed + self.epoch)

        indices = list(range(self.total_size))

        if self.shuffle:
            rng.shuffle(indices)
            global_batched_indices = [
                indices[i : i + global_batch_size]
                for i in range(0, len(indices), global_batch_size)
            ]

            indices = [_ for batch_indices in global_batched_indices for _ in batch_indices]
        else:
            raise NotImplementedError()

        assert len(indices) == self.total_size

        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), self.num_replicas * self.batch_size):
            own_indices += indices[start_pos : start_pos + self.batch_size]
        # subsample
        assert len(own_indices) == self.num_samples

        if self.start_iter * self.batch_size > len(own_indices):
            own_indices = []
        else:
            own_indices = own_indices[self.start_iter * self.batch_size :]

        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int, start_iter: int = 0) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            start_iter (int): start iter number.
        """
        self.epoch = epoch
        self.start_iter = start_iter

