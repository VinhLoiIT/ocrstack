from ocrstack.config.config import Config
from ocrstack.data.vocab import CTCVocab, Seq2SeqVocab

from .conv_rnn import GeneralizedCRNN
from .conv_seq2seq import GeneralizedConvSeq2Seq


def resnet18_lstm_ctc(pretrained: bool, vocab: CTCVocab):
    cfg = Config()
    cfg.MODEL.BACKBONE.TYPE = 'resnet18'
    cfg.MODEL.BACKBONE.FEATURE_SIZE = 512
    cfg.MODEL.BACKBONE.PRETRAINED = pretrained

    cfg.MODEL.ENCODER.TYPE = 'avg_pool'
    cfg.MODEL.ENCODER.BATCH_FIRST = True

    cfg.MODEL.DECODER.TYPE = 'lstm'
    cfg.MODEL.DECODER.VOCAB_SIZE = len(vocab)
    cfg.MODEL.DECODER.INPUT_SIZE = cfg.MODEL.BACKBONE.FEATURE_SIZE
    cfg.MODEL.DECODER.HIDDEN_SIZE = 256
    cfg.MODEL.DECODER.NUM_LAYERS = 2
    cfg.MODEL.DECODER.BIAS = True
    cfg.MODEL.DECODER.BATCH_FIRST = True
    cfg.MODEL.DECODER.DROPOUT = 0.1
    cfg.MODEL.DECODER.BIDIRECTIONAL = True

    cfg.MODEL.DECODER.BLANK_IDX = vocab.BLANK_IDX

    model = GeneralizedCRNN(cfg)

    return model


def resnet18_transformer(pretrained: bool, vocab: Seq2SeqVocab):
    cfg = Config()
    cfg.MODEL.BACKBONE.TYPE = 'resnet18'
    cfg.MODEL.BACKBONE.FEATURE_SIZE = 512
    cfg.MODEL.BACKBONE.PRETRAINED = pretrained

    cfg.MODEL.ENCODER.TYPE = 'tf_encoder'

    cfg.MODEL.DECODER.TYPE = 'tf_decoder'
    cfg.MODEL.DECODER.D_MODEL = 512
    cfg.MODEL.DECODER.NUM_HEADS = 8
    cfg.MODEL.DECODER.NUM_LAYERS = 2
    cfg.MODEL.DECODER.MAX_LENGTH = 20

    cfg.MODEL.TEXT_EMBED.EMBED_SIZE = 512
    cfg.MODEL.TEXT_EMBED.VOCAB_SIZE = len(vocab)
    cfg.MODEL.TEXT_EMBED.SOS_IDX = vocab.SOS_IDX
    cfg.MODEL.TEXT_EMBED.EOS_IDX = vocab.EOS_IDX
    cfg.MODEL.TEXT_EMBED.PAD_IDX = vocab.PAD_IDX
    cfg.MODEL.TEXT_EMBED.OUT_BIAS = False
    cfg.MODEL.TEXT_EMBED.SHARE_WEIGHT_IN_OUT = True

    model = GeneralizedConvSeq2Seq(cfg)
    return model


def resnet18_attn_lstm(pretrained: bool, vocab):
    cfg = Config()
    cfg.MODEL.BACKBONE.TYPE = 'resnet18'
    cfg.MODEL.BACKBONE.FEATURE_SIZE = 512
    cfg.MODEL.BACKBONE.PRETRAINED = pretrained
    cfg.MODEL.ENCODER.TYPE = 'tf_encoder'

    cfg.MODEL.DECODER.TYPE = 'attn_lstm'
    cfg.MODEL.DECODER.HIDDEN_SIZE = 512
    cfg.MODEL.DECODER.MAX_LENGTH = 20
    cfg.MODEL.DECODER.TEACHER_FORCING = False

    cfg.MODEL.TEXT_EMBED.EMBED_SIZE = 512
    cfg.MODEL.TEXT_EMBED.VOCAB_SIZE = len(vocab)
    cfg.MODEL.TEXT_EMBED.SOS_IDX = vocab.SOS_IDX
    cfg.MODEL.TEXT_EMBED.EOS_IDX = vocab.EOS_IDX
    cfg.MODEL.TEXT_EMBED.PAD_IDX = vocab.PAD_IDX
    cfg.MODEL.TEXT_EMBED.OUT_BIAS = False
    cfg.MODEL.TEXT_EMBED.SHARE_WEIGHT_IN_OUT = True

    model = GeneralizedConvSeq2Seq(cfg)
    return model
