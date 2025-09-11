================================================================
Themeda
================================================================

.. start-badges

|testing badge| |docs badge| |black badge| |torchapp badge| |doi badge|

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

.. |doi badge| image:: https://img.shields.io/badge/DOI-10.34133%2Fremotesensing.0780-blue
   :target: https://doi.org/10.34133/remotesensing.0780
    
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

If you use Themeda, please cite the paper:

    Robert Turnbull, Damien J. Mannion, Jessie A. Wells, Kabir Manandhar Shrestha, Attila Balogh, Rebecca K. Runting. 'Themeda: Predicting Land Cover Change Using Deep Learning'. *Journal of Remote Sensing* 2025;5:0780. 
    DOI:`10.34133/remotesensing.0780 <https://doi.org/10.34133/remotesensing.0780>`_

.. code-block:: bibtex

    @article {themeda,
        author = {Robert Turnbull  and Damien J. Mannion  and Jessie A. Wells  and Kabir Manandhar Shrestha  and Attila Balogh  and Rebecca K. Runting },
        title = {Themeda: Predicting Land Cover Change Using Deep Learning},
        journal = {Journal of Remote Sensing},
        volume = {5},
        number = {},
        pages = {0780},
        year = {2025},
        doi = {10.34133/remotesensing.0780},
        URL = {https://spj.science.org/doi/abs/10.34133/remotesensing.0780},
        eprint = {https://spj.science.org/doi/pdf/10.34133/remotesensing.0780},
        abstract = {Accurate land cover change prediction is vital for informed land management, and deep learning offers a flexible solution capable of capturing complex ecological dynamics. This paper presents Themeda, a modeling framework to predict land cover one or more years into the future, using artificial neural networks and time series of remotely sensed data from the world’s largest intact savanna, across northern Australia. Themeda incorporates diverse spatiotemporal features, including 33 years of satellite-derived land cover, rainfall, temperature, fire scars, soil properties, and elevation, and generates a probability distribution for the future land cover for each pixel, across possible land cover classes. The model employs a ConvLSTM and a novel Temporal U-Net architecture, extending the U-Net with long short-term memory layers for multi-scale temporal processing. Themeda overcomes limitations of current spatiotemporal models by processing temporal data at multiple spatial scales, capturing local and regional ecological changes effectively. It achieves a 93.4\% pixel-wise validation accuracy for Food and Agriculture Organization Level 3 land cover classes and a Kullback–Leibler divergence of 1.65 × 10−3 for aggregated land cover predictions in 4,000 m × 4,000 m areas, surpassing baseline persistence models. The model performs strongly in predicting unseen test years, demonstrating robust generalizability. These probabilistic outputs and multi-scale temporal processing represent important advances for remote sensing applications, enabling improved ecological forecasting and supporting land use planning across diverse regions.}
    }


.. end-credits


