from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.data.collate import ImageList, TextList
from ocrstack.data.vocab import Seq2SeqVocab
from torch.nn.utils.rnn import pack_padded_sequence

from .component import Classifier, ImageEmbedding, TextEmbedding


class Seq2SeqOutputLayer(nn.Module):
    def __init__(self, mode='greedy'):
        super(Seq2SeqOutputLayer, self).__init__()
        self.mode = mode

    def forward(self, logits: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.training:
            assert targets is not None
            return self.losses(logits, targets)

        if self.mode == 'greedy':
            probs = F.softmax(logits, dim=-1)
            return probs

        raise NotImplementedError()

    def losses(self, logits, targets=None):
        log_probs = F.log_softmax(logits, dim=-1)                      # [B, T, V]

        # prepare loss forward
        lengths = targets.lengths_tensor - 1
        outputs_packed = pack_padded_sequence(log_probs[:, :-1], lengths.cpu(), True)[0]
        targets_packed = pack_padded_sequence(targets.max_probs_idx[:, 1:], lengths.cpu(), True)[0]
        loss = F.cross_entropy(outputs_packed, targets_packed, reduction='mean')

        loss = {
            'cross_entropy_loss': loss
        }
        return loss


class Seq2Seq(nn.Module):
    def __init__(self,
                 vocab: Seq2SeqVocab,
                 img_emb: ImageEmbedding,
                 text_emb: TextEmbedding,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 classifier: Classifier,
                 ):
        super().__init__()
        self.vocab = vocab
        self.img_emb = img_emb
        self.text_emb = text_emb
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.output = Seq2SeqOutputLayer()

        self.register_buffer('sos_token_idx', vocab.onehot(vocab.SOS))
        self.register_buffer('eos_token_idx', vocab.onehot(vocab.EOS))

    def forward(self, inputs: ImageList, targets: Optional[TextList] = None, **kwargs):
        if not self.training:
            assert isinstance(inputs, ImageList), type(inputs)
            return self.inference(inputs, **kwargs)

        assert targets is not None
        images = self.img_emb(inputs.tensor)
        text = self.text_emb(targets.tensor)
        memory = self.encoder(images)
        output = self.decoder(memory, text)
        logits = self.classifier(output)

        loss = {}
        loss.update(self.output(logits, targets))

        return loss

    def inference(self, inputs: ImageList, max_length: int) -> TextList:
        images = self.img_emb(inputs.tensor)     # [B, S, E]
        batch_size = images.size(0)

        assert isinstance(self.sos_token_idx, torch.Tensor)
        assert isinstance(self.eos_token_idx, torch.Tensor)

        memory = self.encoder(images)                                           # [B, S, E]
        predicts = self.sos_token_idx.unsqueeze(0).repeat(batch_size, 1, 1)     # [B, 1, V]
        ends = self.eos_token_idx.argmax(-1).repeat(batch_size).squeeze(-1)     # [B]

        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        lengths = torch.ones(batch_size, dtype=torch.long).fill_(max_length)
        for t in range(max_length):
            text = self.text_emb(predicts)                      # [B, T, E]
            outputs = self.decoder(memory, text)                # [B, T, E]
            logits = self.classifier(outputs[:, [-1]])          # [B, 1, V]
            output = self.output(logits)                        # [B, 1, V]
            predicts = torch.cat([predicts, output], dim=1)     # [B, T + 1, V]

            # set flag for early break
            output = output.squeeze(1).argmax(-1)                           # [B]
            current_end = output == ends                                    # [B]
            current_end = current_end.cpu()
            lengths.masked_fill_(~end_flag & current_end, t + 1)
            end_flag |= current_end
            if end_flag.all():
                break

        return TextList(predicts[:, 1:], lengths.tolist())

    # def to_dict(self, out_dict: Optional[Dict] = None) -> Dict:
    #     out_dict = out_dict or {}
    #     out_dict['vocab'] = {
    #         self.vocab.char2int
    #     }
    #     out_dict['model'] = self.state_dict()
    #     return out_dict
