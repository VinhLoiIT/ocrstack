from typing import Optional

from ocrstack.data.collate import Batch
from torch import Tensor
from torch.nn import Module


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


class IS2SModel(ITrainableModel):

    '''
    This is a common interface for all models based on sequence-to-sequence approach.

    The difference to `ICTCModel` is that `IS2SModel` require `max_length` parameter in decode functions to
    avoid infinitely decoding.

    We will use these methods to interact with your model.
    '''

    def decode_greedy(self, images, max_length, image_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tensor
        '''
        This method is for convenience only. In fact, it is a exactly decoding beamsearch where `beamsize=1`.
        '''
        raise NotImplementedError()

    def decode_beamsearch(self, images, max_length, beamsize, image_mask=None):
        # type: (Tensor, int, int, Optional[Tensor]) -> Tensor
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

    def decode_greedy(self, images, image_mask=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        '''
        '''
        raise NotImplementedError()

    def decode_beamsearch(self, images, beamsize, image_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tensor
        '''
        '''
        raise NotImplementedError()
