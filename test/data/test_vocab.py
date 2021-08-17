import pytest
import torch
import torch.nn.functional as F
from ocrstack.data.vocab import CTCVocab, Seq2SeqVocab, Vocab


@pytest.fixture()
def stoi():
    return {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}


def test_default_vocab(stoi):
    vocab = Vocab(stoi, unk=None)
    assert vocab.lookup_index('1') == 1
    assert vocab.lookup_token(1) == '1'

    with pytest.raises(ValueError):
        vocab = Vocab(stoi, unk='0')


def test_out_of_vocab(stoi):
    # out-of-vocab and not passed unk token
    vocab = Vocab(stoi, unk=None)
    with pytest.raises(KeyError):
        vocab.lookup_index('0')

    # out-of-vocab and passed unk token
    vocab = Vocab(stoi, unk='1')
    assert vocab.lookup_index('0') == vocab.lookup_index('1')


def test_s2s_vocab(stoi):

    with pytest.raises(ValueError):
        Seq2SeqVocab(stoi, unk='0')                                     # unk not in stoi

    with pytest.raises(ValueError):
        Seq2SeqVocab(stoi, unk='1', sos='0')                            # unk in stoi, but sos does not

    with pytest.raises(ValueError):
        Seq2SeqVocab(stoi, unk='1', sos='2', eos='0')                   # unk, sos in stoi, but eos does not

    with pytest.raises(ValueError):
        Seq2SeqVocab(stoi, unk='1', sos='2', eos='3', pad='0')          # unk, sos, eos in stoi, but pad does not

    vocab = Seq2SeqVocab(stoi, unk='1', sos='2', eos='3', pad='4')
    assert vocab.SOS == '2'
    assert vocab.EOS == '3'
    assert vocab.PAD == '4'
    assert vocab.SOS_IDX == stoi['2']
    assert vocab.EOS_IDX == stoi['3']
    assert vocab.PAD_IDX == stoi['4']


def test_ctc_vocab(stoi):

    with pytest.raises(ValueError):
        CTCVocab(stoi, unk='0')                     # unk not in stoi

    with pytest.raises(ValueError):
        CTCVocab(stoi, unk='1', blank='0')          # unk in stoi, but blank does not

    vocab = CTCVocab(stoi, unk='1', blank='2')

    assert vocab.BLANK == '2'
    assert vocab.BLANK_IDX == stoi['2']


@pytest.fixture()
def vocab():
    tokens = '1 2 3 4 5 <u> <s> </s> <p>'.split()
    stoi = {s: i for i, s in enumerate(tokens)}
    vocab = Seq2SeqVocab(stoi, '<u>')
    return vocab


@pytest.fixture()
def text():
    return '12321'


@pytest.fixture()
def input_with_sos_eos(vocab: Seq2SeqVocab, text):
    text = [vocab.SOS] + list(text) + [vocab.EOS]
    tensor = torch.tensor(vocab.lookup_indices(text))       # T
    tensor = tensor.unsqueeze_(0)                           # 1, T
    return tensor


def test_translate_seq2seq_sos_eos(vocab: Seq2SeqVocab, input_with_sos_eos, text):
    assert input_with_sos_eos.shape == torch.Size([1, len(text) + 2])
    chars = vocab.translate(input_with_sos_eos, '', keep_sos=True, keep_eos=True, keep_pad=False)
    assert chars[0] == vocab.SOS + text + vocab.EOS


def test_translate_seq2seq_sos_only(vocab: Seq2SeqVocab, input_with_sos_eos, text):
    inputs = input_with_sos_eos[:, :-1]
    assert inputs.shape == torch.Size([1, len(text) + 1])
    chars = vocab.translate(inputs, '', keep_sos=True, keep_eos=True, keep_pad=False)
    assert chars[0] == vocab.SOS + text


def test_translate_seq2seq_eos_only(vocab: Seq2SeqVocab, input_with_sos_eos, text):
    inputs = input_with_sos_eos[:, 1:]
    assert inputs.shape == torch.Size([1, len(text) + 1])
    chars = vocab.translate(inputs, '', keep_sos=True, keep_eos=True, keep_pad=False)
    assert chars[0] == text + vocab.EOS
