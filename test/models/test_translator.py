import pytest
import torch
import torch.nn.functional as F
from ocrstack.data.vocab import Seq2SeqVocab
from ocrstack.models.layers.translator import seq2seq_translate


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
def input_with_sos_eos(vocab, text):
    text = [vocab.SOS] + list(text) + [vocab.EOS]
    tensor = torch.tensor(list(map(vocab.char2int, text)))  # T
    tensor = F.one_hot(tensor, num_classes=len(vocab))      # T, V
    tensor = tensor.unsqueeze_(0)                           # 1, T, V
    return tensor


def test_translate_seq2seq_sos_eos(vocab, input_with_sos_eos, text):
    assert input_with_sos_eos.shape == torch.Size([1, len(text) + 2, len(vocab)])
    chars, probs = seq2seq_translate(input_with_sos_eos, vocab, '', keep_sos=True, keep_eos=True, keep_pad=False)
    assert chars[0] == vocab.SOS + text + vocab.EOS
    assert torch.tensor(probs).eq(1).all()


def test_translate_seq2seq_sos_only(vocab, input_with_sos_eos, text):
    inputs = input_with_sos_eos[:, :-1]
    assert inputs.shape == torch.Size([1, len(text) + 1, len(vocab)])
    chars, probs = seq2seq_translate(inputs, vocab, '', keep_sos=True, keep_eos=True, keep_pad=False)
    assert chars[0] == vocab.SOS + text
    assert torch.tensor(probs).eq(1).all()


def test_translate_seq2seq_eos_only(vocab, input_with_sos_eos, text):
    inputs = input_with_sos_eos[:, 1:]
    assert inputs.shape == torch.Size([1, len(text) + 1, len(vocab)])
    chars, probs = seq2seq_translate(inputs, vocab, '', keep_sos=True, keep_eos=True, keep_pad=False)
    assert chars[0] == text + vocab.EOS
    assert torch.tensor(probs).eq(1).all()
