
__all__ = [
    'AverageMeter'
]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._num_samples = 0

    def compute(self) -> float:
        if self._num_samples == 0:
            return 0.
        return self._sum / self._num_samples

    def reset(self) -> None:
        self._sum = 0
        self._num_samples = 0

    def add(self, sum, n_samples) -> None:
        self._sum += sum
        self._num_samples += n_samples

    def update(self, *args, **kwargs):
        raise NotImplementedError()
