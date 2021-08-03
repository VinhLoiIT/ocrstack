from typing import List, Tuple

from torch import Tensor
from torch.nn import Module
from ocrstack.data.collate import Batch


class ITrainableModel(Module):

    '''
    This is a common interface for all models based on sequence-to-sequence approach.

    The difference to `ICTCModel` is that `IS2SModel` require `max_length` parameter in decode functions to
    avoid infinitely decoding.

    We will use these methods to interact with your model.
    '''

    def example_inputs(self):
        raise NotImplementedError()

    def forward_batch(self, batch: Batch) -> Tensor:
        '''
        This method will be called during training/validating to receive loss value.

        Returns:
        --------
        - loss: Tensor
        '''
        raise NotImplementedError()

    def predict_batch(self, batch: Batch) -> Tensor:
        raise NotImplementedError()


class IS2SModel(ITrainableModel):

    '''
    This is a common interface for all models based on sequence-to-sequence approach.

    The difference to `ICTCModel` is that `IS2SModel` require `max_length` parameter in decode functions to
    avoid infinitely decoding.

    We will use these methods to interact with your model.
    '''

    def decode_greedy(self, images, image_mask, max_length):
        # type: (Tensor, Tensor, int) -> Tuple[List[str], List[List[float]]]
        '''
        This method is for convenience only. In fact, it is a exactly decoding beamsearch where `beamsize=1`.
        '''
        raise NotImplementedError()

    def decode_beamsearch(self, images, image_mask, max_length, beamsize):
        # type: (Tensor, Tensor, int, int) -> Tuple[List[str], List[List[float]]]
        '''
        '''
        raise NotImplementedError()


class ICTCModel(ITrainableModel):

    '''
    This is a common interface for all models based on CTC approach.

    The difference to `ICTCModel` is that `IS2SModel` require `max_length` parameter in decode functions to
    avoid infinitely decoding.

    We will use these methods to interact with your model.
    '''

    def decode_greedy(self, images, image_mask):
        # type: (Tensor, Tensor) -> Tuple[List[str], List[List[float]]]
        '''
        '''
        raise NotImplementedError()

    def decode_beamsearch(self, images, image_mask, beamsize):
        # type: (Tensor, Tensor, int) -> Tuple[List[str], List[List[float]]]
        '''
        '''
        raise NotImplementedError()
