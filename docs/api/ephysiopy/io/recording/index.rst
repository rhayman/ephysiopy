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

   .. py:method:: apply_filter(*trial_filter)

      
      Apply a mask to the data

      :param trial_filter (TrialFilter):
                                         name, start and end values:
                                             name (str): The name of the filter
                                             start (float): The start value of the filter
                                             end (float): The end value of the filter

                                         Valid names are:
                                             'dir' - the directional range to filter for
                                             'speed' - min and max speed to filter for
                                             'xrange' - min and max values to filter x pos values
                                             'yrange' - same as xrange but for y pos
                                             'time' - the times to keep / remove specified in ms

                                         Values are pairs specifying the range of values to filter for
                                         from the namedtuple TrialFilter that has fields 'start' and 'end'
                                         where 'start' and 'end' are the ranges to filter for

                                         See ephysiopy.common.utils.TrialFilter for more details
      :type trial_filter (TrialFilter): A namedtuple containing the filter

      :returns: **np.ndarray**
      :rtype: An array of bools that is True where the mask is applied















      ..
          !! processed by numpydoc !!


   .. py:method:: get_available_clusters_channels()


   .. py:method:: get_spike_times(cluster = None, tetrode = None, *args, **kwargs)

      
      :param tetrode:
      :type tetrode: int | list
      :param cluster:
      :type cluster: int | list

      :rtype: spike_times (np.ndarray)















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

      :param ppm (int):
      :type ppm (int): pixels per metre
      :param jumpmax (int): than this and the position is interpolated over
      :type jumpmax (int): max jump in pixels between positions, more















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

      
      Get the recording start time from the sync_messages.txt file

      :rtype: start_time (float) - in seconds















      ..
          !! processed by numpydoc !!


   .. py:method:: apply_filter(*trial_filter)

      
      Apply a mask to the data

      :param trial_filter (TrialFilter): name, start and end values
      :type trial_filter (TrialFilter): A namedtuple containing the filter

      :returns: np.array: An array of bools that is True where the mask is applied















      ..
          !! processed by numpydoc !!


   .. py:method:: find_files(pname_root, experiment_name = 'experiment1', rec_name = 'recording1', **kwargs)


   .. py:method:: get_available_clusters_channels()


   .. py:method:: get_spike_times(cluster = None, tetrode = None, *args, **kwargs)

      
      :param cluster (int| list):
      :param tetrode (int | list):

      :returns: **spike_times (list | np.ndarray)**
      :rtype: in seconds















      ..
          !! processed by numpydoc !!


   .. py:method:: load_accelerometer(target_freq = 50)


   .. py:method:: load_cluster_data(removeNoiseClusters=True, *args, **kwargs)

      
      Load the cluster data (Kilosort/ Axona cut/ whatever else
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_lfp(*args, **kwargs)

      
      Valid kwargs are:
      'target_sample_rate' - int
          the sample rate to downsample to from the original
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_neural_data(*args, **kwargs)

      
      Load the neural data
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_pos_data(ppm = 300, jumpmax = 100, *args, **kwargs)

      
      Load the position data

      :param ppm (int):
      :type ppm (int): pixels per metre
      :param jumpmax (int): than this and the position is interpolated over
      :type jumpmax (int): max jump in pixels between positions, more















      ..
          !! processed by numpydoc !!


   .. py:method:: load_settings(*args, **kwargs)

      
      Loads the format specific settings file
















      ..
          !! processed by numpydoc !!


   .. py:method:: load_ttl(*args, **kwargs)

      
      :rtype: loaded (bool) - whether the data was loaded or not

      .. rubric:: Notes

      Valid kwargs:
          StimControl_id (str): This is the string
              "StimControl [0-9][0-9][0-9]" where the numbers
              are the node id in the openephys signal chain
          TTL_channel_number (int): The integer value in the "states.npy"
              file that corresponds to the
              identity of the TTL input on the Digital I/O board on the
              openephys recording system. i.e. if there is input to BNC
              port 3 on the digital I/O board then values of 3 in the
              states.npy file are high TTL values on this input and -3
              are low TTL values. NB This is important as there could well
              be other TTL lines that are active and so the states vector
              will then contain a mix of integer values
          RippleDetector (str): Loads up the TTL data from the Ripple Detector
              plugin

      Sets some keys/values in a dict on 'self'
      called ttl_data, namely:

      ttl_timestamps (list): the times of high ttl pulses in ms
      stim_duration (int): the duration of the ttl pulse in ms















      ..
          !! processed by numpydoc !!


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

      :param ppm (int):
      :type ppm (int): pixels per metre
      :param jumpmax (int): than this and the position is interpolated over
      :type jumpmax (int): max jump in pixels between positions, more















      ..
          !! processed by numpydoc !!


   .. py:method:: load_settings(*args, **kwargs)

      
      Loads the format specific settings file
















      ..
          !! processed by numpydoc !!


.. py:class:: RecordingKind

   Bases: :py:obj:`enum.Enum`


   
   Generic enumeration.

   Derive from this class to define new enumerations.















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

   .. py:method:: __subclasshook__(subclass)
      :classmethod:



   .. py:method:: _get_map(cluster, channel, var2bin, **kwargs)

      
      This function generates a rate map for a given cluster and channel.

      :param cluster (int | list):
      :type cluster (int | list): The cluster(s).
      :param channel (int | list):
      :type channel (int | list): The channel(s).
      :param var2bin (VariableToBin.XY):
      :type var2bin (VariableToBin.XY): The variable to bin. This is an enum that specifies the type of variable to bin.
      :param \*\*kwargs:
                         do_shuffle (bool): If True, the rate map will be shuffled by the default number of shuffles (100).
                                         If the n_shuffles keyword is provided, the rate map will be shuffled by that number of shuffles, and
                                         an array of shuffled rate maps will be returned e.g [100 x nx x ny].
                                         The shuffles themselves are generated by shifting the spike times by a random amount between 30s and the
                                         length of the position data minus 30s. The random amount is drawn from a uniform distribution. In order to preserve
                                         the shifts over multiple calls to this function, the option is provided to set the random seed to a fixed
                                         value using the random_seed keyword.
                                         Default is False
                         n_shuffles (int): The number of shuffles to perform. Default is 100.
                         random_seed (int): The random seed to use for the shuffles. Default is None.

      :returns: **np.ndarray**
      :rtype: The rate map as a numpy array.















      ..
          !! processed by numpydoc !!


   .. py:method:: _get_spike_pos_idx(cluster, channel, **kwargs)

      
      Returns the indices into the position data at which some cluster
      on a given channel emitted putative spikes.

      :param cluster (int | list): case the "spike times" are equal to the position times, which
                                   means data binned using these indices will be equivalent to
                                   binning up just the position data alone.
      :type cluster (int | list): The cluster(s). NB this can be None in which
      :param channel (int | list):
      :type channel (int | list): The channel identity. Ignored if cluster is None

      :returns: **np.ndarray** -- occurred.
      :rtype: The indices into the position data at which the spikes















      ..
          !! processed by numpydoc !!


   .. py:method:: _update_filter(val)


   .. py:method:: apply_filter(*trial_filter)

      
      Apply a mask to the recorded data. This will mask all the currently
      loaded data (LFP, position etc)

      :param trial_filter (TrialFilter): name, start and end values
                                         name (str): The name of the filter
                                         start (float): The start value of the filter
                                         end (float): The end value of the filter

                                         Valid names are:
                                             'dir' - the directional range to filter for
                                             'speed' - min and max speed to filter for
                                             'xrange' - min and max values to filter x pos values
                                             'yrange' - same as xrange but for y pos
                                             'time' - the times to keep / remove specified in ms

                                         Values are pairs specifying the range of values to filter for
                                         from the namedtuple TrialFilter that has fields 'start' and 'end'
                                         where 'start' and 'end' are the ranges to filter for
      :type trial_filter (TrialFilter): A namedtuple containing the filter

      :returns: **np.ndarray**
      :rtype: An array of bools that is True where the mask is applied















      ..
          !! processed by numpydoc !!


   .. py:method:: get_adaptive_map(cluster, channel, **kwargs)


   .. py:method:: get_available_clusters_channels()


   .. py:method:: get_eb_map(cluster, channel, **kwargs)

      
      Gets the egocentric boundary map for the specified cluster(s) and channel.

      :param cluster (int | list):
      :type cluster (int | list): The cluster(s) to get the speed vs rate for.
      :param channel (int | list):
      :type channel (int | list): The channel(s) number.
      :param \*\*kwargs:
      :type \*\*kwargs: Additional keyword arguments passed to _get_map

      :rtype: BinnedData - the binned data















      ..
          !! processed by numpydoc !!


   .. py:method:: get_grid_map(cluster, channel, **kwargs)


   .. py:method:: get_hd_map(cluster, channel, **kwargs)

      
      Gets the head direction map for the specified cluster(s) and channel.

      :param cluster (int | list):
      :type cluster (int | list): The cluster(s) to get the speed vs rate for.
      :param channel (int | list):
      :type channel (int | list): The channel(s) number.
      :param \*\*kwargs:
      :type \*\*kwargs: Additional keyword arguments passed to _get_map

      :rtype: BinnedData - the binned data















      ..
          !! processed by numpydoc !!


   .. py:method:: get_rate_map(cluster, channel, **kwargs)

      
      Gets the rate map for the specified cluster(s) and channel.

      :param cluster (int | list):
      :type cluster (int | list): The cluster(s) to get the speed vs rate for.
      :param channel (int | list):
      :type channel (int | list): The channel(s) number.
      :param \*\*kwargs:
      :type \*\*kwargs: Additional keyword arguments passed to _get_map

      :rtype: BinnedData - the binned data















      ..
          !! processed by numpydoc !!


   .. py:method:: get_speed_v_hd_map(cluster, channel, **kwargs)

      
      Gets the speed vs head direction map for the specified cluster(s) and channel.

      :param cluster (int | list):
      :type cluster (int | list): The cluster(s) to get the speed vs head direction map for.
      :param channel (int | list):
      :type channel (int | list): The channel number.
      :param \*\*kwargs:
      :type \*\*kwargs: Additional keyword arguments passed to _get_map















      ..
          !! processed by numpydoc !!


   .. py:method:: get_speed_v_rate_map(cluster, channel, **kwargs)

      
      Gets the speed vs rate for the specified cluster(s) and channel.

      :param cluster (int | list):
      :type cluster (int | list): The cluster(s) to get the speed vs rate for.
      :param channel (int | list):
      :type channel (int | list): The channel(s) number.
      :param \*\*kwargs:
      :type \*\*kwargs: Additional keyword arguments passed to _get_map

      :rtype: BinnedData - the binned data















      ..
          !! processed by numpydoc !!


   .. py:method:: get_spike_times(cluster, channel, *args, **kwargs)
      :abstractmethod:


      
      Returns the times of an individual cluster
















      ..
          !! processed by numpydoc !!


   .. py:method:: get_spike_times_binned_into_position(cluster, channel)

      
      :param cluster (int | list):
      :type cluster (int | list): The cluster(s).
      :param channel (int | list):
      :type channel (int | list): The channel(s).

      :rtype: np.ndarray - the spike times binned into the position data















      ..
          !! processed by numpydoc !!


   .. py:method:: get_xcorr(cluster, channel, **kwargs)


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


   .. py:method:: load_pos_data(ppm = 300, jumpmax = 100, *args, **kwargs)
      :abstractmethod:


      
      Load the position data

      :param ppm (int):
      :type ppm (int): pixels per metre
      :param jumpmax (int): than this and the position is interpolated over
      :type jumpmax (int): max jump in pixels between positions, more















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

.. py:data:: Xml2RecordingKind

