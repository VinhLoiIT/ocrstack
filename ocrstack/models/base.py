from typing import Optional

from ocrstack.data.collate import Batch
from torch import Tensor
from torch.nn import Module


class ITrainableModel(Module):

    r"""This is a common interface for all trainable modules to be used within `ocrstack`
    """

    def example_inputs(self):
        r"""Example inputs for forward method

        We will use these inputs to generate the TorchScript and Tensorboard graph of your model
        """
        raise NotImplementedError()

    def forward_batch(self, batch: Batch) -> Tensor:
        r"""
        This method will be called during training/validating to receive loss value.

        Args:
            batch: a batch instance from DataLoader. See :class:`Batch`

        Returns:
            loss tensor after forwarding a batch
        """
        raise NotImplementedError()


class IS2SDecode:

    r"""
    This is a common interface for all models based on sequence-to-sequence approach.

    The difference to :class:`ICTCDecode` is the `max_length` parameter in decode functions to
    avoid infinitely decoding.
    """

    def decode_greedy(self, images, max_length, image_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tensor
        r"""Greedy Sequence-To-Sequence decoding

        In fact, it behaves like beamsearch decoding where :code:`beamsize=1` but faster since it does
        not populate a queue to store temporal decoding steps.

        Args:
            images: a tensor of shape :math:`(B, C, H, W)` containing the images
            max_length: a maximum length :math:`L` to decode
            image_mask: a tensor of shape :math:`(B, H, W)` to indicate images content within a batch.

        Return:
            a prediction tensor of shape :math:`(B, L + 2, V)` where :math:`B` is the batch size, :math:`L` is
            the `max_length`, and :math:`V` is the vocabulary size to present the probabilities of each class. It
            should contain both `sos` and `eos` signals.
        """
        raise NotImplementedError()

    def decode_beamsearch(self, images, max_length, beamsize, image_mask=None):
        # type: (Tensor, int, int, Optional[Tensor]) -> Tensor
        r"""Beamsearch Sequence-To-Sequence decoding

        Args:
            images: a tensor of shape :math:`(B, C, H, W)` containing the images
            max_length: a maximum length :math:`L` to decode
            beamsize: the number of beam for beamsearch algorithms
            image_mask: a tensor of shape :math:`(B, H, W)` to indicate images content within a batch.

        Return:
            a prediction tensor of shape :math:`(B, L + 2, V)` where :math:`B` is the batch size, :math:`L` is
            the `max_length`, and :math:`V` is the vocabulary size to present the probabilities of each class. It
            should contain both `sos` and `eos` signals.
        """
        raise NotImplementedError()


class ICTCDecode:

    r"""
    This is a common interface for all models based on CTC approach.

    The difference to :class:`IS2SDecode` is `max_length` parameter in decode functions to
    avoid infinitely decoding.
    """

    def decode_greedy(self, images, image_mask=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        r"""Greedy CTC decoding

        In fact, it behaves like beamsearch decoding where :code:`beamsize=1` but faster since it does
        not populate a queue to store temporal decoding steps.

        Args:
            images: a tensor of shape :math:`(B, C, H, W)` containing the images
            image_mask: a tensor of shape :math:`(B, H, W)` to indicate images content within a batch.

        Return:
            a prediction tensor of shape :math:`(B, L + 2, V)` where :math:`B` is the batch size, :math:`L` is
            the number of steps including the `blank` character, and :math:`V` is the vocabulary size to
            present the probabilities of each class.
        """
        raise NotImplementedError()

    def decode_beamsearch(self, images, beamsize, image_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tensor
        r"""Beamsearch CTC decoding

        Args:
            images: a Tensor of shape :math:`(B, C, H, W)` containing the images
            beamsize: the number of beam for beamsearch algorithms
            image_mask: a Tensor of shape :math:`(B, H, W)` to indicate images content within a batch.

        Return:
            a prediction tensor of shape :math:`(B, L, V)` where :math:`B` is the batch size, :math:`L` is
            the number of steps including the `blank` character, and :math:`V` is the vocabulary size to
            present the probabilities of each class.
        """
        raise NotImplementedError()


class ITrainableS2S(ITrainableModel, IS2SDecode):
    r"""A convenient interface for a trainable sequence-to-sequence model.
    """
    pass


class ITrainableCTC(ITrainableModel, ICTCDecode):
    r"""A convenient interface for a trainable CTC model.
    """
    pass
