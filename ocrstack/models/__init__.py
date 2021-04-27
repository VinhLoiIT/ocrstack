import torch.nn as nn
from ocrstack.models.conv_seq2seq import ConvSeq2Seq
from ocrstack.models.seq2seq import Seq2Seq
from ocrstack.opts.sequence_decoder import TransformerDecoderAdapter

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


def resnet18_transformer(pretrained: bool, vocab_size, d_model, nhead, num_layers, **kwargs):
    features, feature_size = resnet_feature('resnet18', pretrained)
    transformer = Seq2Seq(
        TransformerDecoderAdapter(
            text_embedding=nn.Linear(vocab_size, d_model),
            text_classifier=nn.Linear(d_model, vocab_size),
            sos_onehot=kwargs['sos_onehot'],
            eos_onehot=kwargs['eos_onehot'],
            decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead), num_layers),
        ),
    )
    model = ConvSeq2Seq(features, transformer, feature_size, d_model)
    return model
