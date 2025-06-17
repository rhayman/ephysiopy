Load some data
==============

openephys neuropixels / tetrode recordings
------------------------------------------

Import the base class for openephys recordings and tell it where the data is:

.. code-block:: python

    >>> from ephysiopy.io.recording import OpenEphysBase
    >>> trial = OpenEphysBase("/path/to/top_level")

The "/path/to/top_level" bit here means that if your directory hierarchy looks like this:

.. code-block:: none

    ├── 2020-03-20_12-40-15
    ├── Record Node 101
    |    └── settings.xml
             experiment1
    |        └── recording1
    |            ├── structure.oebin
    |            ├── sync_messages.txt
    |            ├── continuous
    |            |   └── Neuropix-PXI-107.0
    |            |       └── continuous.dat
    |            └── events
    ├── Record Node 102

Then the "/path/to/top_level" is the folder "2020-03-20_12-40-15"

The directory structure containing the recording data is traversed and various file locations are noted for later processing of the data in them.

Creating the trial object like this will show the locations of the various files and plugins that have been found in the directory structure:

.. code-block:: python

    >>> trial = OpenEphysBase("/path/to/top_level", verbose=True)
    Loaded settings data

    sync_messages file at: /path/to/2020-03-20_12-40-15/Record Node 101/experiment1/recording1/sync_messages.txt

    Continuous AP data at: /path/to/2020-03-20_12-40-15/Record Node 101/experiment1/recording1/continuous/Neuropix-PXI-107.0/continuous.dat

The pos data is loaded by calling the load_pos_data() method:

.. code-block:: python

    >>> trial.load_pos_data(ppm=300, jumpmax=100, cm=True)

.. note::

    ppm = pixels per metre, used to convert pixel coords to cms.
    jumpmax = maximum "jump" in cms for point to be considered "bad" and smoothed over

Axona data
----------

Where openephys data is organised by folder structure, Axona data is organised by file type. A typical Axona recording session might contain files like this:

.. code-block:: none

    ├── my_trial.set
    ├── my_trial.pos
    ├── my_trial.eeg
    ├── my_trial.1
    ├── my_trial.2
    ├── my_trial.3
    ├── my_trial.4

The AxonaTrial class is used to load the data:

.. code-block:: python

    >>> from ephysiopy.io.recording import AxonaTrial
    >>> trial = AxonaTrial("/path/to/set_file.set")

The same load_pos_data() method is used to load the pos data:

.. code-block:: python

    >>> trial.load_pos_data(ppm=300, jumpmax=100, cm=True)

The tetrode data is lazily loaded in a different way to the openephys data:

.. code-block:: python

    >>> trial.TETRODE[1]
    <ephysiopy.axona.axonaIO.Tetrode at 0x7fcd7f674190>

See the Tetrode class for available methods: :py:class:`ephysiopy.axona.axonaIO.Tetrode`

Plot some data
--------------

The example below uses Axona data, but the same methods are available for openephys data:

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from ephysiopy.tests.conftest import get_path_to_test_axona_data as get_data
    >>> from ephysiopy.io.recording import AxonaTrial
    >>> trial = AxonaTrial(get_data())
    >>> trial.load_pos_data()
    >>> # plot the rate map of cluster 2 on tetrode 3
    >>> trial.plot_rate_map(2,3)
    >>> plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from ephysiopy.tests.conftest import get_path_to_test_axona_data as get_data
    from ephysiopy.io.recording import AxonaTrial
    trial = AxonaTrial(get_data())
    trial.load_pos_data()
    trial.plot_rate_map(2,3)
    plt.show()

