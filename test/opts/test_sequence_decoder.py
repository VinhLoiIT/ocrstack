import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.opts.attention import DotProductAttention
from ocrstack.opts.sequence_decoder import (AttentionLSTMDecoder,
                                            TransformerDecoderAdapter)


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

    # sos_token_idx = F.one_hot(torch.tensor([0], dtype=torch.long), vocab_size)
    # eos_token_idx = F.one_hot(torch.tensor([1], dtype=torch.long), vocab_size)
    decoder = TransformerDecoderAdapter(text_embedding, text_classifier, tf_decoder)

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

    sos_onehot = F.one_hot(torch.tensor([0], dtype=torch.long), vocab_size).float()
    eos_onehot = F.one_hot(torch.tensor([1], dtype=torch.long), vocab_size).float()
    decoder = TransformerDecoderAdapter(text_embedding, text_classifier, tf_decoder)

    src = torch.rand(batch_size, src_length, d_model)
    tgt = torch.randint(vocab_size, (batch_size, tgt_length))
    tgt = F.one_hot(tgt, vocab_size).float()

    decoder.eval()
    output = decoder.decode(src, 10, sos_onehot=sos_onehot, eos_onehot=eos_onehot)
    assert isinstance(output, tuple)


def test_attention_lstm_decoder_forward():
    batch_size = 2
    vocab_size = 10
    hidden_size = 20
    tgt_length = 5
    src_length = 4
    context_size = vocab_size + hidden_size
    model = AttentionLSTMDecoder(
        text_embedding=nn.Linear(vocab_size, hidden_size),
        text_classifier=nn.Linear(hidden_size, vocab_size),
        lstm=nn.LSTMCell(context_size, hidden_size),
        attention=DotProductAttention(scaled=True),
    )

    memory = torch.rand(batch_size, src_length, hidden_size)
    tgt = torch.rand(batch_size, tgt_length, vocab_size)
    outputs = model.forward(memory, tgt)
    assert outputs.shape == torch.Size([batch_size, tgt_length, vocab_size])
