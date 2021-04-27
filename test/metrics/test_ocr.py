import numpy as np
from ocrstack.metrics.ocr import compute_global_cer, compute_norm_cer


def test_compute_norm_cer():
    cers = compute_norm_cer([['a', 'b', 'c']], [['c', 'b', 'c']])
    assert np.array(cers) == np.array([1 / 3])

    cers = compute_norm_cer([['a', 'b', 'c']], [['c', 'b', 'c', 'e']])
    assert (np.array(cers) == np.array([2 / 4])).all()

    cers = compute_norm_cer([['a', 'b', 'c'], ['d', 'e', 'f']], [['c', 'b', 'c'], ['d', 'f', 'e']])
    assert (np.array(cers) == np.array([1/3, 2/3])).all()

    cers = compute_norm_cer([['a', 'b', 'c'], ['c', 'd', 'e', 'f']], [['c', 'b', 'c'], ['d', 'f', 'e']])
    assert (np.array(cers) == np.array([1/3, 3/3])).all()


def test_compute_global_cer():
    def batch_cer(dist, num_ref):
        return np.sum(dist) / np.sum(num_ref)
    dist, num_ref = compute_global_cer([['a', 'b', 'c']], [['c', 'b', 'c']])
    assert batch_cer(dist, num_ref) == 1 / 3

    dist, num_ref = compute_global_cer([['a', 'b', 'c']], [['c', 'b', 'c', 'e']])
    assert batch_cer(dist, num_ref) == 2 / 4

    dist, num_ref = compute_global_cer([['a', 'b', 'c'], ['d', 'e', 'f']], [['c', 'b', 'c'], ['d', 'f', 'e']])
    assert batch_cer(dist, num_ref) == (1 + 2) / 6

    dist, num_ref = compute_global_cer([['a', 'b', 'c'], ['c', 'd', 'e', 'f']], [['c', 'b', 'c'], ['d', 'f', 'e']])
    assert batch_cer(dist, num_ref) == (1 + 3) / 6
