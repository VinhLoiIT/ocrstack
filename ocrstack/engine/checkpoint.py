import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch


class CkptSaver(object):

    '''
    Checkpointer class
    '''

    def __init__(self, checkpoint_dir: Optional[Path] = None, exist_ok: bool = False):
        if checkpoint_dir is None:
            folder_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            checkpoint_dir = Path('ckpts').joinpath(folder_name)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=exist_ok, parents=True)

    def save(self, trainer_state: Dict, train_metrics: Dict, eval_metrics: List[Dict]):
        '''
        '''
        pass

    def __call__(self, trainer_state: Dict, train_metrics: Dict, eval_metrics: List[Dict]):
        self.save(trainer_state, train_metrics, eval_metrics)


class LossCkptSaver(CkptSaver):
    def save(self, trainer_state: Dict, train_metrics: Dict, eval_metrics: List[Dict]):
        filename = f'loss={train_metrics["Loss"]:.04f}.pth'
        checkpoint_path = self.checkpoint_dir.joinpath(filename)
        torch.save(trainer_state, checkpoint_path)
