import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.opts.sequence_decoder import TransformerDecoderAdapter


def test_transformer_decoder_adapter_forward():
    d_model = 10
    nhead = 5
    dim_feedforward = 100
    vocab_size = 5
    batch_size = 2
    src_length = 4
    tgt_length = 3

    tf_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
    tf_decoder = nn.TransformerDecoder(tf_decoder_layer, 1)

    text_embedding = nn.Linear(vocab_size, d_model)
    text_classifier = nn.Linear(d_model, vocab_size)

    sos_token_idx = F.one_hot(torch.tensor([0], dtype=torch.long), vocab_size)
    eos_token_idx = F.one_hot(torch.tensor([1], dtype=torch.long), vocab_size)
    decoder = TransformerDecoderAdapter(text_embedding, text_classifier, sos_token_idx, eos_token_idx, tf_decoder)

    memory = torch.rand(batch_size, src_length, d_model)
    tgt = torch.randint(vocab_size, (batch_size, tgt_length))
    tgt = F.one_hot(tgt, vocab_size).float()

    assert memory.shape == torch.Size([batch_size, src_length, d_model])
    assert tgt.shape == torch.Size([batch_size, tgt_length, vocab_size])

    output = decoder.forward(memory, tgt)
    assert output.shape == torch.Size([batch_size, tgt_length, vocab_size])


def test_transformer_decoder_adapter_decode():
    d_model = 10
    nhead = 5
    dim_feedforward = 100
    batch_size = 2
    src_length = 4
    vocab_size = 5
    tgt_length = 3

    tf_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
    tf_decoder = nn.TransformerDecoder(tf_decoder_layer, 1)
    text_embedding = nn.Linear(vocab_size, d_model)
    text_classifier = nn.Linear(d_model, vocab_size)

    sos_token_idx = F.one_hot(torch.tensor([0], dtype=torch.long), vocab_size)
    eos_token_idx = F.one_hot(torch.tensor([1], dtype=torch.long), vocab_size)
    decoder = TransformerDecoderAdapter(text_embedding, text_classifier, sos_token_idx, eos_token_idx, tf_decoder)

    src = torch.rand(batch_size, src_length, d_model)
    tgt = torch.randint(vocab_size, (batch_size, tgt_length))
    tgt = F.one_hot(tgt, vocab_size).float()

    decoder.eval()
    output = decoder.decode(src, 10)
    assert isinstance(output, tuple)
