import torch.nn as nn

from .conv import resnet_feature
from .conv_attn_rnn import ConvAttnRNN, ConvAttnRNNConfig
from .conv_rnn import ConvRNN
from .conv_seq2seq import ConvSeq2Seq, ConvSeq2SeqConfig
from .layers.attention import DotProductAttention
from .layers.sequence_decoder import (AttentionLSTMDecoder,
                                      TransformerDecoderAdapter)
from .layers.string_decoder import CTCGreedyDecoder, Seq2SeqGreedyDecoder


def resnet18_lstm_ctc(pretrained: bool, vocab, **lstm_kwargs):
    features, input_size = resnet_feature('resnet18', pretrained)
    lstm = nn.LSTM(input_size=input_size, **lstm_kwargs)
    vocab_size = len(vocab)
    return ConvRNN(features, lstm, vocab_size, vocab.BLANK_IDX, CTCGreedyDecoder(vocab))


def resnet18_transformer(pretrained: bool, vocab, d_model, nhead, num_layers, max_length):
    features, feature_size = resnet_feature('resnet18', pretrained)
    vocab_size = len(vocab)
    cfg = ConvSeq2SeqConfig(feature_size, vocab_size, d_model, vocab.SOS_IDX, vocab.EOS_IDX, max_length)
    model = ConvSeq2Seq(
        cfg,
        features,
        TransformerDecoderAdapter(
            text_embedding=nn.Linear(vocab_size, d_model),
            text_classifier=nn.Linear(d_model, vocab_size),
            decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead), num_layers),
        ),
        Seq2SeqGreedyDecoder(vocab, keep_eos=True),
        encoder=None,
    )
    return model


def resnet18_attn_lstm(pretrained: bool, vocab, hidden_size, max_length):
    features, feature_size = resnet_feature('resnet18', pretrained)
    vocab_size = len(vocab)
    cfg = ConvAttnRNNConfig(feature_size, vocab_size, hidden_size, vocab.SOS_IDX, vocab.EOS_IDX, max_length)
    model = ConvAttnRNN(
        cfg,
        features,
        AttentionLSTMDecoder(
            text_embedding=nn.Linear(vocab_size, hidden_size),
            text_classifier=nn.Linear(hidden_size, vocab_size),
            lstm=nn.LSTMCell(vocab_size + hidden_size, hidden_size),
            attention=DotProductAttention(scaled=True),
            teacher_forcing=False,
        ),
        Seq2SeqGreedyDecoder(vocab, keep_eos=True),
        encoder=None,
    )
    return model
