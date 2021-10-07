from ocrstack.data.collate import Batch
from ocrstack.ops.sequence_decoder import ICTCDecode, IS2SDecode
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


class ITrainableS2S(ITrainableModel, IS2SDecode):
    r"""A convenient interface for a trainable sequence-to-sequence model.
    """
    pass


class ITrainableCTC(ITrainableModel, ICTCDecode):
    r"""A convenient interface for a trainable CTC model.
    """
    pass
