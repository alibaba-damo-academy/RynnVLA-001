import glob
import importlib
import logging
import os.path as osp

import torch
import torch.utils.data

__all__ = ['create_dataset']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in glob.glob(f'{data_folder}/*_dataset.py')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'datasets.{file_name}')
    for file_name in dataset_filenames
]


def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt['type']

    # dynamically instantiation
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(dataset_opt)

    logger = logging.getLogger('base')
    logger.info(
        f"Dataset {dataset.__class__.__name__} - {dataset_opt['type']} "
        'is created.')
    return dataset

