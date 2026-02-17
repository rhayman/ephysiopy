ephysiopy.io.recording
======================

.. py:module:: ephysiopy.io.recording


Attributes
----------

.. autoapisummary::

   ephysiopy.io.recording.Xml2RecordingKind


Classes
-------

.. autoapisummary::

   ephysiopy.io.recording.AxonaTrial
   ephysiopy.io.recording.OpenEphysBase
   ephysiopy.io.recording.OpenEphysNWB
   ephysiopy.io.recording.RecordingKind
   ephysiopy.io.recording.TrialInterface


Functions
---------

.. autoapisummary::

   ephysiopy.io.recording.find_path_to_ripple_ttl
   ephysiopy.io.recording.make_cluster_ids


Module Contents
---------------

.. py:class:: AxonaTrial(pname, **kwargs)

   Bases: :py:obj:`TrialInterface`


   
   Defines a minimal and required set of methods for loading
   electrophysiology data recorded using Axona or OpenEphys
   (OpenEphysNWB is there but not used)

   :param pname (Path):
   :type pname (Path): The path to the top-level directory containing the recording

   .. attribute:: pname (str)

      :type: the absolute pathname of the top-level data directory

   .. attribute:: settings (dict)

      :type: contains metadata about the trial

   .. attribute:: PosCalcs (PosCalcsGeneric)

      :type: contains the positional data for the trial

   .. attribute:: RateMap

      methods for binning data mostly

      :type: RateMap

   .. attribute:: EEGCalcs

      methods for dealing with LFP data

      :type: EEGCalcs

   .. attribute:: clusterData

      contains results of a spike sorting session (i.e. KiloSort)

      :type: clusterData

   .. attribute:: recording_start_time

      the start time of the recording in seconds

      :type: float

   .. attribute:: sync_message_file

      the location of the sync_message_file (OpenEphys)

      :type: Path

   .. attribute:: ttl_data

      ttl data including timestamps, ids and states

      :type: dict

   .. attribute:: accelerometer_data

      data relating to headstage accelerometers

      :type: np.ndarray

   .. attribute:: path2PosData

      location of the positional data

      :type: Path

   .. attribute:: mask_array

      contains the mask (if applied) for positional data

      :type: np.ma.MaskedArray

   .. attribute:: filter

      contains details of the filter applied to the positional data

      :type: TrialFilter















   ..
       !! processed by numpydoc !!

   .. py:method:: __add__(other)


   .. py:method:: apply_filter(*trial_filter)

      
      Apply a mask to the recorded data. This will mask all the currently
      loaded data (LFP, position etc)

      :param trial_filter: A namedtuple containing the filter
                           name, start and end values
                           name (str): The name of the filter
                           start (float): The start value of the filter
                           end (float): The end value of the filter

                           Valid names are:
                               'dir' - the directional range to filter for
                                   NB Following mathmatical convention, 0/360 degrees is
                                   3 o'clock, 90 degrees is 12 o'clock, 180 degrees is
                                   9 o'clock and 270 degrees
                               'speed' - min and max speed to filter for
                               'xrange' - min and max values to filter x pos values
                               'yrange' - same as xrange but for y pos
                               'time' - the times to keep / remove specified in ms

                           Values are pairs specifying the range of values to filter for
                           from the namedtuple TrialFilter that has fields 'start' and 'end'
                           where 'start' and 'end' are the ranges to filter for
      :type trial_filter: TrialFilter

      :returns: An array of bools that is True where the mask is applied
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_available_clusters_channels(remove0=True)

      
      Slightly laborious and low-level way of getting the cut
      data but it's faster than accessing the TETRODE's as that
      will load the waveforms as well as everything else
















      ..
          !! processed by numpydoc !!


   .. py:method:: get_spike_times(cluster = None, tetrode = None, *args, **kwargs)

      
      Returns the times of an individual cluster

      :param cluster: The cluster(s) to get the spike times for
      :type cluster: int | list
      :param channel: The channel(s) to get the spike times for
      :type channel: int | list

      :returns: the spike times
      :rtype: list | np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_waveforms(cluster, channel, *args, **kwargs)

      
      Returns the waveforms for a given cluster and channel

      :param cluster: The cluster(s) to get the waveforms for
      :type cluster: int | list
      :param channel: The channel(s) to get the waveforms for
      :type channel: int | list

      :returns: the waveforms for the cluster(s) and channel(s)
      :rtype: list | np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: load_cluster_data(*args, **kwargs)

      
      Load the cluster data (Kilosort/ Axona cut/ whatever else
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_lfp(*args, **kwargs)

      
      Load the LFP data
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_neural_data(*args, **kwargs)

      
      Load the neural data
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_pos_data(ppm = 300, jumpmax = 100, *args, **kwargs)

      
      Load the position data

      :param ppm: pixels per metre
      :type ppm: int
      :param jumpmax: max jump in pixels between positions, more
                      than this and the position is interpolated over
      :type jumpmax: int















      ..
          !! processed by numpydoc !!


   .. py:method:: load_settings(*args, **kwargs)

      
      Loads the format specific settings file
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_ttl(*args, **kwargs)


   .. py:attribute:: TETRODE


   .. py:attribute:: _settings
      :value: None



.. py:class:: OpenEphysBase(pname, **kwargs)

   Bases: :py:obj:`TrialInterface`


   
   Defines a minimal and required set of methods for loading
   electrophysiology data recorded using Axona or OpenEphys
   (OpenEphysNWB is there but not used)

   :param pname (Path):
   :type pname (Path): The path to the top-level directory containing the recording

   .. attribute:: pname (str)

      :type: the absolute pathname of the top-level data directory

   .. attribute:: settings (dict)

      :type: contains metadata about the trial

   .. attribute:: PosCalcs (PosCalcsGeneric)

      :type: contains the positional data for the trial

   .. attribute:: RateMap

      methods for binning data mostly

      :type: RateMap

   .. attribute:: EEGCalcs

      methods for dealing with LFP data

      :type: EEGCalcs

   .. attribute:: clusterData

      contains results of a spike sorting session (i.e. KiloSort)

      :type: clusterData

   .. attribute:: recording_start_time

      the start time of the recording in seconds

      :type: float

   .. attribute:: sync_message_file

      the location of the sync_message_file (OpenEphys)

      :type: Path

   .. attribute:: ttl_data

      ttl data including timestamps, ids and states

      :type: dict

   .. attribute:: accelerometer_data

      data relating to headstage accelerometers

      :type: np.ndarray

   .. attribute:: path2PosData

      location of the positional data

      :type: Path

   .. attribute:: mask_array

      contains the mask (if applied) for positional data

      :type: np.ma.MaskedArray

   .. attribute:: filter

      contains details of the filter applied to the positional data

      :type: TrialFilter















   ..
       !! processed by numpydoc !!

   .. py:method:: _get_recording_start_time()


   .. py:method:: apply_filter(*trial_filter)

      
      Apply a mask to the recorded data. This will mask all the currently
      loaded data (LFP, position etc)

      :param trial_filter: A namedtuple containing the filter
                           name, start and end values
                           name (str): The name of the filter
                           start (float): The start value of the filter
                           end (float): The end value of the filter

                           Valid names are:
                               'dir' - the directional range to filter for
                                   NB Following mathmatical convention, 0/360 degrees is
                                   3 o'clock, 90 degrees is 12 o'clock, 180 degrees is
                                   9 o'clock and 270 degrees
                               'speed' - min and max speed to filter for
                               'xrange' - min and max values to filter x pos values
                               'yrange' - same as xrange but for y pos
                               'time' - the times to keep / remove specified in ms

                           Values are pairs specifying the range of values to filter for
                           from the namedtuple TrialFilter that has fields 'start' and 'end'
                           where 'start' and 'end' are the ranges to filter for
      :type trial_filter: TrialFilter

      :returns: An array of bools that is True where the mask is applied
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: find_files(pname_root, experiment_name = 'experiment1', rec_name = 'recording1', **kwargs)


   .. py:method:: get_available_clusters_channels()

      
      Get available clusters and their corresponding channels.

      :returns: A dict where keys are channels and values are lists of clusters
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: get_spike_times(cluster = None, tetrode = None, *args, **kwargs)

      
      Returns the times of an individual cluster

      :param cluster: The cluster(s) to get the spike times for
      :type cluster: int | list
      :param channel: The channel(s) to get the spike times for
      :type channel: int | list

      :returns: the spike times
      :rtype: list | np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_waveforms(cluster, channel, *args, **kwargs)

      
      Gets the waveforms for the specified cluster(s). Ignores the channel input
      and instead returns the waveforms for the four "best" channels for the cluster.
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_accelerometer(target_freq = 50)


   .. py:method:: load_cluster_data(removeNoiseClusters=True, *args, **kwargs)

      
      Load the cluster data (Kilosort/ Axona cut/ whatever else
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_lfp(*args, **kwargs)

      
      Load the LFP data
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_neural_data(*args, **kwargs)

      
      Load the neural data
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_pos_data(ppm = 300, jumpmax = 100, *args, **kwargs)

      
      Load the position data

      :param ppm: pixels per metre
      :type ppm: int
      :param jumpmax: max jump in pixels between positions, more
                      than this and the position is interpolated over
      :type jumpmax: int















      ..
          !! processed by numpydoc !!


   .. py:method:: load_settings(*args, **kwargs)

      
      Loads the format specific settings file
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_ttl(*args, **kwargs)


   .. py:attribute:: channel_count


   .. py:attribute:: kilodata
      :value: None



   .. py:attribute:: rec_kind


   .. py:attribute:: sample_rate
      :value: None



   .. py:attribute:: template_model
      :value: None



.. py:class:: OpenEphysNWB(pname, **kwargs)

   Bases: :py:obj:`OpenEphysBase`


   
   Defines a minimal and required set of methods for loading
   electrophysiology data recorded using Axona or OpenEphys
   (OpenEphysNWB is there but not used)

   :param pname (Path):
   :type pname (Path): The path to the top-level directory containing the recording

   .. attribute:: pname (str)

      :type: the absolute pathname of the top-level data directory

   .. attribute:: settings (dict)

      :type: contains metadata about the trial

   .. attribute:: PosCalcs (PosCalcsGeneric)

      :type: contains the positional data for the trial

   .. attribute:: RateMap

      methods for binning data mostly

      :type: RateMap

   .. attribute:: EEGCalcs

      methods for dealing with LFP data

      :type: EEGCalcs

   .. attribute:: clusterData

      contains results of a spike sorting session (i.e. KiloSort)

      :type: clusterData

   .. attribute:: recording_start_time

      the start time of the recording in seconds

      :type: float

   .. attribute:: sync_message_file

      the location of the sync_message_file (OpenEphys)

      :type: Path

   .. attribute:: ttl_data

      ttl data including timestamps, ids and states

      :type: dict

   .. attribute:: accelerometer_data

      data relating to headstage accelerometers

      :type: np.ndarray

   .. attribute:: path2PosData

      location of the positional data

      :type: Path

   .. attribute:: mask_array

      contains the mask (if applied) for positional data

      :type: np.ma.MaskedArray

   .. attribute:: filter

      contains details of the filter applied to the positional data

      :type: TrialFilter















   ..
       !! processed by numpydoc !!

   .. py:method:: find_files(experiment_name = 'experiment_1', recording_name = 'recording0')


   .. py:method:: load_neural_data(*args, **kwargs)

      
      Load the neural data
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_pos_data(ppm = 300, jumpmax = 100, *args, **kwargs)

      
      Load the position data

      :param ppm: pixels per metre
      :type ppm: int
      :param jumpmax: max jump in pixels between positions, more
                      than this and the position is interpolated over
      :type jumpmax: int















      ..
          !! processed by numpydoc !!


   .. py:method:: load_settings(*args, **kwargs)

      
      Loads the format specific settings file
















      ..
          !! processed by numpydoc !!


.. py:class:: RecordingKind(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   
   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access:

     >>> Color.RED
     <Color.RED: 1>

   - value lookup:

     >>> Color(1)
     <Color.RED: 1>

   - name lookup:

     >>> Color['RED']
     <Color.RED: 1>

   Enumerations can be iterated over, and know how many members they have:

   >>> len(Color)
   3

   >>> list(Color)
   [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

   Methods can be added to enumerations, and members can have their own
   attributes -- see the documentation for details.















   ..
       !! processed by numpydoc !!

   .. py:attribute:: ACQUISITIONBOARD
      :value: 3



   .. py:attribute:: FPGA
      :value: 1



   .. py:attribute:: NEUROPIXELS
      :value: 2



   .. py:attribute:: NWB
      :value: 4



.. py:class:: TrialInterface(pname, **kwargs)

   Bases: :py:obj:`ephysiopy.visualise.plotting.FigureMaker`


   
   Defines a minimal and required set of methods for loading
   electrophysiology data recorded using Axona or OpenEphys
   (OpenEphysNWB is there but not used)

   :param pname (Path):
   :type pname (Path): The path to the top-level directory containing the recording

   .. attribute:: pname (str)

      :type: the absolute pathname of the top-level data directory

   .. attribute:: settings (dict)

      :type: contains metadata about the trial

   .. attribute:: PosCalcs (PosCalcsGeneric)

      :type: contains the positional data for the trial

   .. attribute:: RateMap

      methods for binning data mostly

      :type: RateMap

   .. attribute:: EEGCalcs

      methods for dealing with LFP data

      :type: EEGCalcs

   .. attribute:: clusterData

      contains results of a spike sorting session (i.e. KiloSort)

      :type: clusterData

   .. attribute:: recording_start_time

      the start time of the recording in seconds

      :type: float

   .. attribute:: sync_message_file

      the location of the sync_message_file (OpenEphys)

      :type: Path

   .. attribute:: ttl_data

      ttl data including timestamps, ids and states

      :type: dict

   .. attribute:: accelerometer_data

      data relating to headstage accelerometers

      :type: np.ndarray

   .. attribute:: path2PosData

      location of the positional data

      :type: Path

   .. attribute:: mask_array

      contains the mask (if applied) for positional data

      :type: np.ma.MaskedArray

   .. attribute:: filter

      contains details of the filter applied to the positional data

      :type: TrialFilter















   ..
       !! processed by numpydoc !!

   .. py:method:: __apply_mask_to_subcls__(mask)

      
      Applies a mask to the sub-class specific data
















      ..
          !! processed by numpydoc !!


   .. py:method:: __subclasshook__(subclass)
      :classmethod:



   .. py:method:: _get_map(cluster, channel, var2bin, **kwargs)

      
      This function generates a rate map for a given cluster and channel.

      :param cluster: The cluster(s).
      :type cluster: int or list
      :param channel: The channel(s).
      :type channel: int or list
      :param var2bin: The variable to bin.
      :type var2bin: VariableToBin.XY
      :param \*\*kwargs: Additional keyword arguments for the _get_spike_pos_idx function.
                         - do_shuffle (bool): If True, the rate map will be shuffled by
                         - map_type (MapType): the type of map to generate, default
                                     is MapType.POS but can be any of the options
                                     in MapType
                                              the default number of shuffles (100).
                         - n_shuffles (int): the number of shuffles for the rate map
                                             A list of shuffled rate maps will be returned.
                         - random_seed (int): The random seed to use for the shuffles.
      :type \*\*kwargs: dict, optional

      :returns: The rate map as a numpy array.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: _get_spike_pos_idx(cluster, channel, **kwargs)

      
      Returns the indices into the position data at which some cluster
      on a given channel emitted putative spikes.

      :param cluster: The cluster(s). NB this can be None in which
                      case the "spike times" are equal to the position times, which
                      means data binned using these indices will be equivalent to
                      binning up just the position data alone.
      :type cluster: int | list
      :param channel: The channel identity. Ignored if cluster is None
      :type channel: int | list

      :returns: The indices into the position data at which the spikes
                occurred.
      :rtype: list of np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: _update_filter(val)


   .. py:method:: apply_filter(*trial_filter)

      
      Apply a mask to the recorded data. This will mask all the currently
      loaded data (LFP, position etc)

      :param trial_filter: A namedtuple containing the filter
                           name, start and end values
                           name (str): The name of the filter
                           start (float): The start value of the filter
                           end (float): The end value of the filter

                           Valid names are:
                               'dir' - the directional range to filter for
                                   NB Following mathmatical convention, 0/360 degrees is
                                   3 o'clock, 90 degrees is 12 o'clock, 180 degrees is
                                   9 o'clock and 270 degrees
                               'speed' - min and max speed to filter for
                               'xrange' - min and max values to filter x pos values
                               'yrange' - same as xrange but for y pos
                               'time' - the times to keep / remove specified in ms

                           Values are pairs specifying the range of values to filter for
                           from the namedtuple TrialFilter that has fields 'start' and 'end'
                           where 'start' and 'end' are the ranges to filter for
      :type trial_filter: TrialFilter

      :returns: An array of bools that is True where the mask is applied
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_acorr(cluster, channel, **kwargs)

      
      Computes the cross-correlation for a given cluster and channel.

      :param cluster: The cluster(s).
      :type cluster: int or list
      :param channel: The channel(s).
      :type channel: int or list
      :param \*\*kwargs: Additional keyword arguments passed to the xcorr function.
      :type \*\*kwargs: dict, optional

      :returns: The cross-correlation as a BinnedData object.
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: get_adaptive_map(cluster, channel, **kwargs)

      
      Generates an adaptive map for a given cluster and channel.

      :param cluster: The cluster(s).
      :type cluster: int or list
      :param channel: The channel(s).
      :type channel: int or list
      :param \*\*kwargs: Additional keyword arguments passed to the _get_map function.
      :type \*\*kwargs: dict, optional

      :returns: The adaptive map as a BinnedData object.
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: get_all_maps(channels_clusters, var2bin, maptype, **kwargs)


   .. py:method:: get_available_clusters_channels()
      :abstractmethod:



   .. py:method:: get_binned_spike_times(cluster, channel, bin_into = 'pos')

      
      :param cluster (int | list): The cluster(s).
      :param channel (int | list): The channel(s).
      :param bin_into (str):

      :returns: the spike times binned into the position data
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_eb_map(cluster, channel, **kwargs)

      
      Gets the egocentric boundary map for the cluster(s) and channel.

      :param cluster: The cluster(s) to get the speed vs rate for.
      :type cluster: int, list
      :param channel: The channel(s) number.
      :type channel: int, list
      :param \*\*kwargs:
      :type \*\*kwargs: Additional keyword arguments passed to _get_map

      :returns: the binned data
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: get_field_properties(cluster, channel, **kwargs)

      
      Gets the properties of a given field

      :param cluster: The cluster(s) to get the field properties for
      :type cluster: int | list
      :param channel: The channel(s) to get the field properties for
      :type channel: int | list
      :param \*\*kwargs:
                         partition : str
                             The partition to use for the field properties. This is passed to
                             the fieldproperties function and can be used to specify the partition
                             to use for the field properties.

                             Valid options are 'simple' and 'fancy'

                             Other kwargs get passed to get_rate_map and
                             fieldprops, the most important of which may be
                             how the runs are split in fieldprops (options are
                             'field' and 'clump_runs') which differ depending on
                             if the position data is open-field (field) or linear track
                             in which case you should probably use 'clunmp_runs'

      :returns: A list of FieldProps namedtuples containing the properties of the field
      :rtype: list[FieldProps]















      ..
          !! processed by numpydoc !!


   .. py:method:: get_grid_map(cluster, channel, **kwargs)

      
      Generates a grid map for a given cluster and channel.

      :param cluster: The cluster(s).
      :type cluster: int or list
      :param channel: The channel(s).
      :type channel: int or list
      :param \*\*kwargs: Additional keyword arguments passed to the autoCorr2D function.
      :type \*\*kwargs: dict, optional

      :returns: The grid map as a BinnedData object.
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: get_hd_map(cluster, channel, **kwargs)

      
      Gets the head direction map for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the speed vs rate for.
      :type cluster: int, list
      :param channel: The channel(s) number.
      :type channel: int,  list
      :param \*\*kwargs:
      :type \*\*kwargs: Additional keyword arguments passed to _get_map

      :rtype: BinnedData - the binned data















      ..
          !! processed by numpydoc !!


   .. py:method:: get_linear_rate_map(cluster, channel, **kwargs)

      
      Gets the linear rate map for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the speed vs rate for.
      :type cluster: int, list
      :param channel: The channel(s) number.
      :type channel: int, list
      :param \*\*kwargs: Additional keyword arguments passed to _get_map

      :returns: the binned data
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: get_psth(cluster, channel, **kwargs)

      
      Computes the peri-stimulus time histogram (PSTH) for a given cluster and channel.

      :param cluster: The cluster(s).
      :type cluster: int or list
      :param channel: The channel(s).
      :type channel: int or list
      :param \*\*kwargs: Additional keyword arguments passed to the psth function.
      :type \*\*kwargs: dict, optional

      :returns: The list of time differences between the spikes of the cluster
                and the events (0) and the trials (1)
      :rtype: tuple of lists















      ..
          !! processed by numpydoc !!


   .. py:method:: get_rate_map(cluster, channel, **kwargs)

      
      Gets the rate map for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the speed vs rate for.
      :type cluster: int, list
      :param channel: The channel(s) number.
      :type channel: int, list
      :param \*\*kwargs: Additional keyword arguments passed to _get_map

      :returns: the binned data
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: get_spatial_info_score(cluster, channel, **kwargs)

      
      Computes the spatial information score

      :param cluster: The cluster(s).
      :type cluster: int or list
      :param channel: The channel(s).
      :type channel: int or list
      :param \*\*kwargs: Additional keyword arguments passed to the binning function.
      :type \*\*kwargs: dict, optional

      :returns: The spatial information score
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:method:: get_speed_v_hd_map(cluster, channel, **kwargs)

      
      Gets the speed vs head direction map for the cluster(s) and channel.

      :param cluster: The cluster(s)
      :type cluster: int, list
      :param channel: The channel number.
      :type channel: int, list
      :param \*\*kwargs:
      :type \*\*kwargs: Additional keyword arguments passed to _get_map















      ..
          !! processed by numpydoc !!


   .. py:method:: get_speed_v_rate_map(cluster, channel, **kwargs)

      
      Gets the speed vs rate for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the speed vs rate for.
      :type cluster: int, list
      :param channel: The channel(s) number.
      :type channel: int, list
      :param \*\*kwargs:
      :type \*\*kwargs: Additional keyword arguments passed to _get_map

      :rtype: BinnedData - the binned data















      ..
          !! processed by numpydoc !!


   .. py:method:: get_spike_times(cluster, channel, *args, **kwargs)
      :abstractmethod:


      
      Returns the times of an individual cluster

      :param cluster: The cluster(s) to get the spike times for
      :type cluster: int | list
      :param channel: The channel(s) to get the spike times for
      :type channel: int | list

      :returns: the spike times
      :rtype: list | np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_waveforms(cluster, channel, *args, **kwargs)
      :abstractmethod:


      
      Returns the waveforms for a given cluster and channel

      :param cluster: The cluster(s) to get the waveforms for
      :type cluster: int | list
      :param channel: The channel(s) to get the waveforms for
      :type channel: int | list

      :returns: the waveforms for the cluster(s) and channel(s)
      :rtype: list | np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_xcorr(cluster_a, cluster_b, channel_a, channel_b, **kwargs)

      
      Computes the cross-correlation for a given cluster and channel.

      :param cluster: The cluster(s).
      :type cluster: int or list
      :param channel: The channel(s).
      :type channel: int or list
      :param \*\*kwargs: Additional keyword arguments passed to the xcorr function.
      :type \*\*kwargs: dict, optional

      :returns: The cross-correlation as a BinnedData object.
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: initialise()


   .. py:method:: load_cluster_data(*args, **kwargs)
      :abstractmethod:


      
      Load the cluster data (Kilosort/ Axona cut/ whatever else
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_lfp(*args, **kwargs)
      :abstractmethod:


      
      Load the LFP data
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_neural_data(*args, **kwargs)
      :abstractmethod:


      
      Load the neural data
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_pos_data(ppm = 300, jumpmax = 100, *args, **kws)
      :abstractmethod:


      
      Load the position data

      :param ppm: pixels per metre
      :type ppm: int
      :param jumpmax: max jump in pixels between positions, more
                      than this and the position is interpolated over
      :type jumpmax: int















      ..
          !! processed by numpydoc !!


   .. py:method:: load_settings(*args, **kwargs)
      :abstractmethod:


      
      Loads the format specific settings file
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_ttl(*args, **kwargs)
      :abstractmethod:



   .. py:property:: EEGCalcs


   .. py:property:: PosCalcs

      
      Initializes the FigureMaker object with data from PosCalcs.
















      ..
          !! processed by numpydoc !!


   .. py:property:: RateMap


   .. py:attribute:: _EEGCalcs
      :value: None



   .. py:attribute:: _PosCalcs
      :value: None



   .. py:attribute:: _RateMap
      :value: None



   .. py:attribute:: _accelerometer_data
      :value: None



   .. py:attribute:: _clusterData
      :value: None



   .. py:attribute:: _concatenated
      :value: False



   .. py:attribute:: _filter
      :type:  list
      :value: []



   .. py:attribute:: _mask_array
      :value: None



   .. py:attribute:: _path2PosData
      :value: None



   .. py:attribute:: _pname


   .. py:attribute:: _recording_start_time
      :value: None



   .. py:attribute:: _settings
      :value: None



   .. py:attribute:: _sync_message_file
      :value: None



   .. py:attribute:: _ttl_data
      :value: None



   .. py:property:: accelerometer_data


   .. py:property:: clusterData


   .. py:property:: concatenated


   .. py:property:: filter


   .. py:property:: mask_array


   .. py:property:: path2PosData


   .. py:property:: pname


   .. py:property:: recording_start_time


   .. py:property:: settings


   .. py:property:: sync_message_file


   .. py:property:: ttl_data


.. py:function:: find_path_to_ripple_ttl(trial_root, **kwargs)

   
   Iterates through a directory tree and finds the path to the
   Ripple Detector plugin data and returns its location
















   ..
       !! processed by numpydoc !!

.. py:function:: make_cluster_ids(cluster, channel)

.. py:data:: Xml2RecordingKind

