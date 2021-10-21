ocrstack.models
================

To make ``ocrstack`` runnable on custom models, we provide some interfaces for you to add to your own pre-defined models in case you already have.


Training Interface
------------------

.. autoclass:: ocrstack.models.base.ITrainableModel
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: training


.. autoclass:: ocrstack.models.base.ITrainableS2S
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: training


.. autoclass:: ocrstack.models.base.ITrainableCTC
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: training


Decode Interface
----------------

.. autoclass:: ocrstack.models.base.IS2SDecode
    :members:
    :undoc-members:

.. autoclass:: ocrstack.models.base.ICTCDecode
    :members:
    :undoc-members:
