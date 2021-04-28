from ocrstack.opts.attention import ScaleDotProductAttention
import torch.nn as nn
from ocrstack.models.conv_seq2seq import ConvSeq2Seq
from ocrstack.opts.sequence_decoder import AttentionLSTMDecoder, TransformerDecoderAdapter

from .conv import resnet_feature
from .conv_rnn import ConvRNN

################################################################################
# CRNN models
################################################################################


def resnet18_lstm_ctc(pretrained: bool, vocab_size, **lstm_kwargs):
    features, input_size = resnet_feature('resnet18', pretrained)
    lstm = nn.LSTM(input_size=input_size, **lstm_kwargs)
    return ConvRNN(features, lstm, vocab_size)


def resnet18_gru_ctc(pretrained: bool, vocab_size, **gru_kwargs):
    features, input_size = resnet_feature('resnet18', pretrained)
    lstm = nn.GRU(input_size=input_size, **gru_kwargs)
    return ConvRNN(features, lstm, vocab_size)


def resnet34_lstm_ctc(pretrained: bool, vocab_size, **lstm_kwargs):
    features, input_size = resnet_feature('resnet34', pretrained)
    lstm = nn.LSTM(input_size=input_size, **lstm_kwargs)
    return ConvRNN(features, lstm, vocab_size)


def resnet34_gru_ctc(pretrained: bool, vocab_size, **gru_kwargs):
    features, input_size = resnet_feature('resnet34', pretrained)
    lstm = nn.GRU(input_size=input_size, **gru_kwargs)
    return ConvRNN(features, lstm, vocab_size)


def resnet50_lstm_ctc(pretrained: bool, vocab_size, **lstm_kwargs):
    features, input_size = resnet_feature('resnet50', pretrained)
    lstm = nn.LSTM(input_size=input_size, **lstm_kwargs)
    return ConvRNN(features, lstm, vocab_size)


def resnet50_gru_ctc(pretrained: bool, vocab_size, **gru_kwargs):
    features, input_size = resnet_feature('resnet50', pretrained)
    lstm = nn.GRU(input_size=input_size, **gru_kwargs)
    return ConvRNN(features, lstm, vocab_size)


def resnet18_transformer(pretrained: bool, vocab_size, d_model, nhead, num_layers):
    features, feature_size = resnet_feature('resnet18', pretrained)
    model = ConvSeq2Seq(
        features,
        TransformerDecoderAdapter(
            text_embedding=nn.Linear(vocab_size, d_model),
            text_classifier=nn.Linear(d_model, vocab_size),
            decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead), num_layers),
        ),
        feature_size,
        d_model,
    )
    return model


def resnet18_attn_lstm(pretrained: bool, vocab_size, hidden_size, attn_size):
    features, feature_size = resnet_feature('resnet18', pretrained)
    context_size = vocab_size + attn_size
    model = ConvSeq2Seq(
        features,
        AttentionLSTMDecoder(
            text_embedding=nn.Linear(vocab_size, hidden_size),
            text_classifier=nn.Linear(hidden_size, vocab_size),
            lstm=nn.LSTMCell(context_size, hidden_size),
            attention=ScaleDotProductAttention(attn_size),
        ),
        feature_size,
        hidden_size,
    )
    return model
