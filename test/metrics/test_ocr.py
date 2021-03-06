import unittest
from ocrstack.metrics import CERMeter


class TestCER(unittest.TestCase):
    def setUp(self):
        self.metric = CERMeter()

    def test_1_sample_same_length(self):
        self.metric.update([['a', 'b', 'c']], [['c', 'b', 'c']])
        assert self.metric.compute() == (1 / 3)

    def test_1_sample_diff_length(self):
        self.metric.update([['a', 'b', 'c']], [['c', 'b', 'c', 'e']])
        assert self.metric.compute() == (2 / 4)

    def test_2_sample_same_length(self):
        self.metric.update([['a', 'b', 'c'], ['d', 'e', 'f']], [['c', 'b', 'c'], ['d', 'f', 'e']])
        assert self.metric.compute() == ((1 + 2) / 6)

    def test_2_sample_diff_length(self):
        self.metric.update([['a', 'b', 'c'], ['c', 'd', 'e', 'f']], [['c', 'b', 'c'], ['d', 'f', 'e']])
        assert self.metric.compute() == ((1 + 3) / 6)

    def test_1_sample_update_2_times(self):
        self.metric.update([['a', 'b', 'c']], [['c', 'b', 'c']])
        self.metric.update([['a', 'b', 'c']], [['c', 'b', 'c']])
        assert self.metric.compute() == ((1 + 1) / 6)
