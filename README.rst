================================================================
Themeda
================================================================

.. start-badges

|testing badge| |coverage badge| |docs badge| |black badge| |torchapp badge|

.. |testing badge| image:: https://github.com/themeda-devs/themeda/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/themeda-devs/themeda/actions

.. |docs badge| image:: https://github.com/themeda-devs/themeda/actions/workflows/docs.yml/badge.svg
    :target: https://themeda-devs.github.io/themeda
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/296c2f5ddd0a272d5a058401c404489e/raw/coverage-badge.json
    :target: https://themeda-devs.github.io/themeda/coverage/

.. |torchapp badge| image:: https://img.shields.io/badge/MLOpps-torchapp-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/
    
.. end-badges

.. start-quickstart

A model to forecast changes to ecosystems in Australia. 

For more information, see the preprint here: https://ssrn.com/abstract=4681094

Installation
==================================

Install using pip:

.. code-block:: bash

    pip install git+https://github.com/themeda-devs/themeda.git


Usage
==================================

See the options for training a model with the command:

.. code-block:: bash

    themeda train --help

To use the ConvLSTM model, use the `themeda-convlstm` command:

.. code-block:: bash

    themeda-convlstm train --help

See the options for making inferences with the command:

.. code-block:: bash

    themeda infer --help

.. end-quickstart


Credits
==================================

.. start-credits

- Robert Turnbull
- Damien Mannion
- Jessie Wells
- Kabir Manandhar Shrestha
- Attila Balogh
- Rebecca Runting

If you use Themeda, please cite the preprint:

    Turnbull, Robert and Mannion, Damien and Wells, Jessie and Manandhar Shrestha, Kabir and Balogh, Attila and Runting, Rebecca, Themeda: Predicting land cover change using deep learning (January 1, 2024). Available at SSRN: https://ssrn.com/abstract=4681094

.. end-credits


