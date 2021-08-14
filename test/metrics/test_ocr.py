from ocrstack.metrics.ocr import (compute_acc, compute_global_cer,
                                  compute_global_wer, compute_norm_cer,
                                  compute_norm_wer, split_by_token)


def test_split_by_token_normal():
    assert split_by_token(['a', ' ', 'b'], ' ') == [['a'], ['b']]
    assert split_by_token(['a', ' ', 'b', 'c'], ' ') == [['a'], ['b', 'c']]
    assert split_by_token(['a', 'b', 'c'], ' ') == [['a', 'b', 'c']]


def test_split_by_token_spaces():
    # spaces on the right side
    assert split_by_token(['a', ' ', 'b', ' '], ' ') == [['a'], ['b']]
    assert split_by_token(['a', ' ', 'b', ' ', ' '], ' ') == [['a'], ['b']]

    # spaces on the left side
    assert split_by_token([' ', 'a', ' ', 'b'], ' ') == [['a'], ['b']]
    assert split_by_token([' ', ' ', 'a', ' ', 'b'], ' ') == [['a'], ['b']]

    # spaces inside
    assert split_by_token(['a', ' ', ' ', 'b'], ' ') == [['a'], ['b']]

    # spaces both sides
    assert split_by_token([' ', 'a', ' ', 'b', ' '], ' ') == [['a'], ['b']]
    assert split_by_token([' ', ' ', 'a', ' ', ' ', 'b', ' ', ' '], ' ') == [['a'], ['b']]


def test_compute_norm_cer():
    assert compute_norm_cer(['a', 'b', 'c'], ['c', 'b', 'c']) == (1 / 3)
    assert compute_norm_cer(['a', 'b', 'c'], ['c', 'b', 'c', 'e']) == (2 / 4)


def test_compute_global_cer():
    assert compute_global_cer(['a', 'b', 'c'], ['c', 'b', 'c']) == (1, 3)
    assert compute_global_cer(['a', 'b', 'c'], ['c', 'b', 'c', 'e']) == (2, 4)


def test_compute_norm_wer():
    assert compute_norm_wer(['a', ' ', 'b', ' ', 'c'], ['c', ' ', 'b', 'a'], ' ') == 3 / 2
    assert compute_norm_wer(['a', 'b', ' ', 'c'], ['a', 'b', ' ', 'c'], ' ') == 0


def test_compute_global_wer():
    assert compute_global_wer(['a', ' ', 'b', ' ', 'c'], ['c', ' ', 'b', 'a'], ' ') == (3, 2)
    assert compute_global_wer(['a', 'b', ' ', 'c'], ['a', 'b', ' ', 'c'], ' ') == (0, 2)


def test_compute_acc():
    assert compute_acc(['a', ' ', 'b', 'c'], ['a', 'b', 'c']) == 0
    assert compute_acc(['a', ' ', 'b', ' ', 'c'], ['c', 'b', 'a']) == 0
    assert compute_acc(['a', ' ', 'b', ' ', 'c'], ['a', ' ', 'b', ' ', 'c']) == 1
