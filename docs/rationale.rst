Rationale
====================

Initially ephysiopy was a re-write of some MATLAB code used for analysing Axona tetrode data. Later I started using openephys for tetrode and neuropixel recordings and so wanted to use tbe same code to analyse both data types/ formats. 

TrialInterface
--------------

There is an abstract base class (:py:class:`ephysiopy.io.recording.TrialInterface`) from which both the OpenEphysBase (:py:class:`ephysiopy.io.recording.OpenEphysBase`) and AxonaTrial (:py:class:`ephysiopy.io.recording.AxonaTrial`) classes inherit and which defines a minimal set of methods for loading the data including:

.. code-block:: none

    load_pos_data()
    load_neural_data()
    load_lfp_data()
    load_ttl()
    load_settings()


The output of calls to most of these methods is an appropriate object that is used for subsequent analysis. For example, load_pos_data() results in the creation of a PosCalcsGeneric object that contains the x-y data and has some methods for smoothing the raw position data and interpolating over missing values for example (:py:class:`ephysiopy.common.ephys_generic.PosCalcsGeneric`). This design pattern is still not fully realised but will be shortly (missing items are the results of load_ttl() and load_neural_data()).

If you have a recording format not covered by the existing classes you can create a new class that inherits from TrialInterface and implement the methods for loading the data from your format.

There are also a collection of get* and plot* functions some of which are covered elsewhere in the documentation. These are used to extract and plot data from the objects created by the load_* methods.
