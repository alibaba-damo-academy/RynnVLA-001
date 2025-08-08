import glob
import importlib
import logging
import os.path as osp

# automatically scan and import trainer modules
# scan all the files under the 'trainer' folder and collect files ending with
# '_trainer.py'
trainer_folder = osp.dirname(osp.abspath(__file__))
trainer_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in glob.glob(f'{trainer_folder}/*_trainer.py')
]
# import all the model modules
_trainer_modules = [
    importlib.import_module(f'trainer.{file_name}')
    for file_name in trainer_filenames
]

def create_trainer(opt):
    """Create trainer.
    """
    trainer_type = opt['trainer_type']

    # dynamically instantiation
    for module in _trainer_modules:
        trainer_cls = getattr(module, trainer_type, None)
        if trainer_cls is not None:
            break
    if trainer_cls is None:
        raise ValueError(f'Trainer {trainer_type} is not found.')

    trainer = trainer_cls(opt)

    logger = logging.getLogger('base')
    logger.info(f'Trainer [{trainer.__class__.__name__}] is created.')
    return trainer

