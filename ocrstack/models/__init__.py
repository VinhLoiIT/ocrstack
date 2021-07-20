import torch.nn as nn
from ocrstack.config.config import Config

from .conv import resnet_feature
from .conv_attn_rnn import ConvAttnRNN, ConvAttnRNNConfig
from .conv_rnn import ConvRNN
from .conv_seq2seq import ConvSeq2Seq
from .layers.attention import DotProductAttention
from .layers.sequence_decoder import AttentionLSTMDecoder
from .layers.string_decoder import CTCGreedyDecoder, Seq2SeqGreedyDecoder


def resnet18_lstm_ctc(pretrained: bool, vocab, **lstm_kwargs):
    features, input_size = resnet_feature('resnet18', pretrained)
    lstm = nn.LSTM(input_size=input_size, **lstm_kwargs)
    vocab_size = len(vocab)
    return ConvRNN(features, lstm, vocab_size, vocab.BLANK_IDX, CTCGreedyDecoder(vocab))


def resnet18_transformer(pretrained: bool, vocab, d_model, nhead, num_layers, max_length):
    cfg = Config()
    cfg.MODEL.BACKBONE.TYPE = 'resnet18'
    cfg.MODEL.BACKBONE.FEATURE_SIZE = 512
    cfg.MODEL.BACKBONE.PRETRAINED = pretrained

    cfg.MODEL.ENCODER.TYPE = 'tf_encoder'

    cfg.MODEL.DECODER.TYPE = 'tf_decoder'
    cfg.MODEL.DECODER.D_MODEL = d_model
    cfg.MODEL.DECODER.NUM_HEADS = nhead
    cfg.MODEL.DECODER.NUM_LAYERS = num_layers
    cfg.MODEL.DECODER.MAX_LENGTH = max_length
    cfg.MODEL.DECODER.VOCAB_SIZE = len(vocab)
    cfg.MODEL.DECODER.SOS_IDX = vocab.SOS_IDX
    cfg.MODEL.DECODER.EOS_IDX = vocab.EOS_IDX
    cfg.MODEL.DECODER.PAD_IDX = vocab.PAD_IDX

    model = ConvSeq2Seq(
        cfg,
        Seq2SeqGreedyDecoder(vocab, keep_eos=True),
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
