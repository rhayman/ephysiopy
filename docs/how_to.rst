How-to guide
============

This guide will show you how to use ephysiopy to load and analyse data from both Axona and OpenEphys recordings. Lets say you've recorded data from the entorhinal coretex and want to take a look at the rate maps and so on (Assuming the data has been spike sorted). The first step is to load some data:

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from ephysiopy.tests.conftest import get_path_to_test_axona_data as get_data
    >>> from ephysiopy.io.recording import AxonaTrial
    >>> trial = AxonaTrial(get_data())
    >>> trial.load_pos_data()
    >>> trial.load_neural_data()

The above code will load the position data and the neural data. 
