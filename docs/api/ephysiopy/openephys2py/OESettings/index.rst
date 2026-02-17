ephysiopy.openephys2py.OESettings
=================================

.. py:module:: ephysiopy.openephys2py.OESettings

.. autoapi-nested-parse::

   Classes for parsing information contained in the settings.xml
   file that is saved when recording with the openephys system.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   ephysiopy.openephys2py.OESettings.AbstractProcessorFactory
   ephysiopy.openephys2py.OESettings.AcquisitionBoard
   ephysiopy.openephys2py.OESettings.BandpassFilter
   ephysiopy.openephys2py.OESettings.Channel
   ephysiopy.openephys2py.OESettings.Electrode
   ephysiopy.openephys2py.OESettings.FloatConversion
   ephysiopy.openephys2py.OESettings.IntConversion
   ephysiopy.openephys2py.OESettings.NeuropixPXI
   ephysiopy.openephys2py.OESettings.OEPlugin
   ephysiopy.openephys2py.OESettings.OEStructure
   ephysiopy.openephys2py.OESettings.PosTracker
   ephysiopy.openephys2py.OESettings.ProcessorFactory
   ephysiopy.openephys2py.OESettings.RecordNode
   ephysiopy.openephys2py.OESettings.RhythmFPGA
   ephysiopy.openephys2py.OESettings.RippleDetector
   ephysiopy.openephys2py.OESettings.Settings
   ephysiopy.openephys2py.OESettings.SpikeSorter
   ephysiopy.openephys2py.OESettings.SpikeViewer
   ephysiopy.openephys2py.OESettings.StimControl
   ephysiopy.openephys2py.OESettings.Stream
   ephysiopy.openephys2py.OESettings.TrackMe
   ephysiopy.openephys2py.OESettings.TrackingPort


Functions
---------

.. autoapisummary::

   ephysiopy.openephys2py.OESettings.addValues2Class
   ephysiopy.openephys2py.OESettings.recurseNode


Module Contents
---------------

.. py:class:: AbstractProcessorFactory

   
   Factory class for creating various processor objects.

   .. method:: create_pos_tracker() -> PosTracker

      Create a PosTracker object.

   .. method:: create_rhythm_fpga() -> RhythmFPGA

      Create a RhythmFPGA object.

   .. method:: create_neuropix_pxi() -> NeuropixPXI

      Create a NeuropixPXI object.

   .. method:: create_acquisition_board() -> AcquisitionBoard

      Create an AcquisitionBoard object.

   .. method:: create_spike_sorter() -> SpikeSorter

      Create a SpikeSorter object.

   .. method:: create_track_me() -> TrackMe

      Create a TrackMe object.

   .. method:: create_record_node() -> RecordNode

      Create a RecordNode object.

   .. method:: create_stim_control() -> StimControl

      Create a StimControl object.

   .. method:: create_oe_plugin() -> OEPlugin

      Create an OEPlugin object.

   .. method:: create_ripple_detector() -> RippleDetector

      Create a RippleDetector object.

   .. method:: create_bandpass_filter() -> BandpassFilter

      Create a BandpassFilter object.















   ..
       !! processed by numpydoc !!

   .. py:method:: create_acquisition_board()

      
      Create an AcquisitionBoard object.

      :returns: A new AcquisitionBoard object.
      :rtype: AcquisitionBoard















      ..
          !! processed by numpydoc !!


   .. py:method:: create_bandpass_filter()

      
      Create a BandpassFilter object.

      :returns: A new BandpassFilter object.
      :rtype: BandpassFilter















      ..
          !! processed by numpydoc !!


   .. py:method:: create_neuropix_pxi()

      
      Create a NeuropixPXI object.

      :returns: A new NeuropixPXI object.
      :rtype: NeuropixPXI















      ..
          !! processed by numpydoc !!


   .. py:method:: create_oe_plugin()

      
      Create an OEPlugin object.

      :returns: A new OEPlugin object.
      :rtype: OEPlugin















      ..
          !! processed by numpydoc !!


   .. py:method:: create_pos_tracker()

      
      Create a PosTracker object.

      :returns: A new PosTracker object.
      :rtype: PosTracker















      ..
          !! processed by numpydoc !!


   .. py:method:: create_record_node()

      
      Create a RecordNode object.

      :returns: A new RecordNode object.
      :rtype: RecordNode















      ..
          !! processed by numpydoc !!


   .. py:method:: create_rhythm_fpga()

      
      Create a RhythmFPGA object.

      :returns: A new RhythmFPGA object.
      :rtype: RhythmFPGA















      ..
          !! processed by numpydoc !!


   .. py:method:: create_ripple_detector()

      
      Create a RippleDetector object.

      :returns: A new RippleDetector object.
      :rtype: RippleDetector















      ..
          !! processed by numpydoc !!


   .. py:method:: create_spike_sorter()

      
      Create a SpikeSorter object.

      :returns: A new SpikeSorter object.
      :rtype: SpikeSorter















      ..
          !! processed by numpydoc !!


   .. py:method:: create_stim_control()

      
      Create a StimControl object.

      :returns: A new StimControl object.
      :rtype: StimControl















      ..
          !! processed by numpydoc !!


   .. py:method:: create_track_me()

      
      Create a TrackMe object.

      :returns: A new TrackMe object.
      :rtype: TrackMe















      ..
          !! processed by numpydoc !!


.. py:class:: AcquisitionBoard

   Bases: :py:obj:`OEPlugin`


   
   Documents the Acquisition Board plugin

   .. attribute:: LowCut

      The low cut-off frequency for the acquisition board.

      :type: FloatConversion

   .. attribute:: HighCut

      The high cut-off frequency for the acquisition board.

      :type: FloatConversion















   ..
       !! processed by numpydoc !!

   .. py:attribute:: HighCut
      :type:  FloatConversion


   .. py:attribute:: LowCut
      :type:  FloatConversion


.. py:class:: BandpassFilter

   Bases: :py:obj:`OEPlugin`


   
   Documents the Bandpass Filter plugin

   .. attribute:: name

      The name of the plugin.

      :type: str

   .. attribute:: pluginName

      The display name of the plugin.

      :type: str

   .. attribute:: pluginType

      The type identifier for the plugin.

      :type: int

   .. attribute:: libraryName

      The library name of the plugin.

      :type: str

   .. attribute:: channels

      The list of channels to which the filter is applied.

      :type: list of int

   .. attribute:: low_cut

      The low cut-off frequency for the bandpass filter.

      :type: FloatConversion

   .. attribute:: high_cut

      The high cut-off frequency for the bandpass filter.

      :type: FloatConversion















   ..
       !! processed by numpydoc !!

   .. py:attribute:: channels
      :type:  list[int]
      :value: []



   .. py:attribute:: high_cut
      :type:  FloatConversion


   .. py:attribute:: libraryName
      :value: 'Bandpass Filter'



   .. py:attribute:: low_cut
      :type:  FloatConversion


   .. py:attribute:: name
      :value: 'Bandpass Filter'



   .. py:attribute:: pluginName
      :value: 'Bandpass Filter'



   .. py:attribute:: pluginType
      :value: 1



.. py:class:: Channel

   
   Documents the information attached to each channel.

   .. attribute:: name

      The name of the channel.

      :type: str

   .. attribute:: number

      The channel number, converted from a string.

      :type: int

   .. attribute:: gain

      The gain value, converted from a string.

      :type: float

   .. attribute:: param

      A boolean parameter, converted from a string
      ("1" for True, otherwise False).

      :type: bool

   .. attribute:: record

      A boolean indicating if the channel is recorded,
      converted from a string ("1" for True, otherwise False).

      :type: bool

   .. attribute:: audio

      A boolean indicating if the channel is audio,
      converted from a string ("1" for True, otherwise False).

      :type: bool

   .. attribute:: lowcut

      The low cut frequency, converted from a string.

      :type: float

   .. attribute:: highcut

      The high cut frequency, converted from a string.

      :type: float















   ..
       !! processed by numpydoc !!

   .. py:attribute:: _audio
      :type:  bool
      :value: False



   .. py:attribute:: _gain
      :type:  FloatConversion


   .. py:attribute:: _highcut
      :type:  FloatConversion


   .. py:attribute:: _lowcut
      :type:  FloatConversion


   .. py:attribute:: _number
      :type:  IntConversion


   .. py:attribute:: _param
      :type:  bool
      :value: False



   .. py:attribute:: _record
      :type:  bool
      :value: False



   .. py:property:: audio
      :type: bool


      
      Get the audio status.

      :returns: The audio status.
      :rtype: bool















      ..
          !! processed by numpydoc !!


   .. py:property:: gain
      :type: float


      
      Get the gain value.

      :returns: The gain value.
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:property:: highcut
      :type: float


      
      Get the high cut frequency.

      :returns: The high cut frequency.
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:property:: lowcut
      :type: float


      
      Get the low cut frequency.

      :returns: The low cut frequency.
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:attribute:: name
      :type:  str
      :value: ''



   .. py:property:: number
      :type: int


      
      Get the channel number.

      :returns: The channel number.
      :rtype: int















      ..
          !! processed by numpydoc !!


   .. py:property:: param
      :type: bool


      
      Get the boolean parameter.

      :returns: The boolean parameter.
      :rtype: bool















      ..
          !! processed by numpydoc !!


   .. py:property:: record
      :type: bool


      
      Get the record status.

      :returns: The record status.
      :rtype: bool















      ..
          !! processed by numpydoc !!


.. py:class:: Electrode

   Bases: :py:obj:`object`


   
   Documents the ELECTRODE entries in the settings.xml file.

   .. attribute:: nChannels

      Number of channels for the electrode, default is 0.

      :type: IntConversion

   .. attribute:: id

      ID of the electrode, default is 0.

      :type: IntConversion

   .. attribute:: subChannels

      List of sub-channel indices, default is an empty list.

      :type: list[int]

   .. attribute:: subChannelsThresh

      List of sub-channel thresholds, default is an empty list.

      :type: list[int]

   .. attribute:: subChannelsActive

      List of active sub-channels, default is an empty list.

      :type: list[int]

   .. attribute:: prePeakSamples

      Number of samples before the peak, default is 8.

      :type: IntConversion

   .. attribute:: postPeakSamples

      Number of samples after the peak, default is 32.

      :type: IntConversion















   ..
       !! processed by numpydoc !!

   .. py:attribute:: id
      :type:  IntConversion


   .. py:attribute:: nChannels
      :type:  IntConversion


   .. py:attribute:: postPeakSamples
      :type:  IntConversion


   .. py:attribute:: prePeakSamples
      :type:  IntConversion


   .. py:attribute:: subChannels
      :type:  list[int]
      :value: []



   .. py:attribute:: subChannelsActive
      :type:  list[int]
      :value: []



   .. py:attribute:: subChannelsThresh
      :type:  list[int]
      :value: []



.. py:class:: FloatConversion(*, default)

   
   Descriptor class for converting attribute values to floats.

   :param default: The default value to return if the attribute is not set.
   :type default: float

   .. method:: __set_name__(owner, name)

      Sets the internal name for the attribute.

   .. method:: __get__(obj, type)

      Retrieves the attribute value, returning the default if not set.

   .. method:: __set__(obj, value)

      Sets the attribute value, converting it to a float.















   ..
       !! processed by numpydoc !!

   .. py:method:: __get__(obj, type)

      
      Retrieve the attribute value.

      :param obj: The instance of the owner class.
      :type obj: object
      :param type: The owner class type.
      :type type: type

      :returns: The attribute value or the default value if not set.
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:method:: __set__(obj, value)

      
      Set the attribute value, converting it to a float.

      :param obj: The instance of the owner class.
      :type obj: object
      :param value: The value to set, which will be converted to a float.
      :type value: any















      ..
          !! processed by numpydoc !!


   .. py:method:: __set_name__(owner, name)

      
      Set the internal name for the attribute.

      :param owner: The owner class where the descriptor is defined.
      :type owner: type
      :param name: The name of the attribute.
      :type name: str















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _default


.. py:class:: IntConversion(*, default)

   
   Descriptor class for converting attribute values to integers.

   :param default: The default value to return if the attribute is not set.
   :type default: int

   .. method:: __set_name__(owner, name)

      Sets the internal name for the attribute.

   .. method:: __get__(obj, type)

      Retrieves the attribute value, returning the default if not set.

   .. method:: __set__(obj, value)

      Sets the attribute value, converting it to an integer.















   ..
       !! processed by numpydoc !!

   .. py:method:: __get__(obj, type)

      
      Retrieve the attribute value.

      :param obj: The instance of the owner class.
      :type obj: object
      :param type: The owner class type.
      :type type: type

      :returns: The attribute value or the default value if not set.
      :rtype: int















      ..
          !! processed by numpydoc !!


   .. py:method:: __set__(obj, value)

      
      Set the attribute value, converting it to an integer.

      :param obj: The instance of the owner class.
      :type obj: object
      :param value: The value to set, which will be converted to an integer.
      :type value: any















      ..
          !! processed by numpydoc !!


   .. py:method:: __set_name__(owner, name)

      
      Set the internal name for the attribute.

      :param owner: The owner class where the descriptor is defined.
      :type owner: type
      :param name: The name of the attribute.
      :type name: str















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _default


.. py:class:: NeuropixPXI

   Bases: :py:obj:`OEPlugin`


   
   Documents the Neuropixels-PXI plugin.

   .. attribute:: channel_info

      A list containing information about each channel.

      :type: list of Channel















   ..
       !! processed by numpydoc !!

   .. py:attribute:: channel_info
      :type:  list[Channel]
      :value: []



.. py:class:: OEPlugin

   Bases: :py:obj:`abc.ABC`


   
   Documents an OE plugin.

   .. attribute:: name

      The name of the plugin.

      :type: str

   .. attribute:: insertionPoint

      The insertion point of the plugin, converted from a string.

      :type: IntConversion

   .. attribute:: pluginName

      The name of the plugin.

      :type: str

   .. attribute:: type

      The type of the plugin, converted from a string.

      :type: IntConversion

   .. attribute:: index

      The index of the plugin, converted from a string.

      :type: IntConversion

   .. attribute:: libraryName

      The name of the library.

      :type: str

   .. attribute:: libraryVersion

      The version of the library.

      :type: str

   .. attribute:: processorType

      The type of processor, converted from a string.

      :type: IntConversion

   .. attribute:: nodeId

      The node ID, converted from a string.

      :type: IntConversion

   .. attribute:: channel_count

      The number of channels, converted from a string.

      :type: IntConversion

   .. attribute:: stream

      The data stream associated with the plugin.

      :type: Stream

   .. attribute:: sample_rate

      The sample rate, converted from a string.

      :type: FloatConversion















   ..
       !! processed by numpydoc !!

   .. py:attribute:: channel_count
      :type:  IntConversion


   .. py:attribute:: index
      :type:  IntConversion


   .. py:attribute:: insertionPoint
      :type:  IntConversion


   .. py:attribute:: libraryName
      :type:  str
      :value: ''



   .. py:attribute:: libraryVersion
      :type:  str
      :value: ''



   .. py:attribute:: name
      :type:  str
      :value: ''



   .. py:attribute:: nodeId
      :type:  IntConversion


   .. py:attribute:: pluginName
      :type:  str
      :value: ''



   .. py:attribute:: processorType
      :type:  IntConversion


   .. py:attribute:: sample_rate
      :type:  FloatConversion


   .. py:attribute:: stream
      :type:  Stream


   .. py:attribute:: type
      :type:  IntConversion


.. py:class:: OEStructure(fname)

   Bases: :py:obj:`object`


   
   Loads up the structure.oebin file for Open Ephys flat
   binary format recordings.

   :param fname: The path to the directory containing the structure.oebin file.
   :type fname: Path

   .. attribute:: filename

      List of filenames found.

      :type: list

   .. attribute:: data

      Dictionary containing the data read from the structure.oebin files.

      :type: dict

   .. method:: find_oebin(pname: Path) -> list

      Find all structure.oebin files in the specified path.

   .. method:: read_oebin(fname: Path) -> dict

      Read the structure.oebin file and return its contents as a dictionary.















   ..
       !! processed by numpydoc !!

   .. py:method:: find_oebin(pname)

      
      Find all structure.oebin files in the specified path.

      :param pname: The path to search for structure.oebin files.
      :type pname: Path

      :returns: A list of paths to the found structure.oebin files.
      :rtype: list















      ..
          !! processed by numpydoc !!


   .. py:method:: read_oebin(fname)

      
      Read the structure.oebin file and return its contents as a dictionary.

      :param fname: The path to the structure.oebin file.
      :type fname: Path

      :returns: The contents of the structure.oebin file.
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:attribute:: data


   .. py:attribute:: filename
      :value: []



.. py:class:: PosTracker

   Bases: :py:obj:`OEPlugin`


   
   Documents the PosTracker plugin.

   .. attribute:: Brightness

      Brightness setting for the tracker, default is 20.

      :type: IntConversion

   .. attribute:: Contrast

      Contrast setting for the tracker, default is 20.

      :type: IntConversion

   .. attribute:: Exposure

      Exposure setting for the tracker, default is 20.

      :type: IntConversion

   .. attribute:: LeftBorder

      Left border setting for the tracker, default is 0.

      :type: IntConversion

   .. attribute:: RightBorder

      Right border setting for the tracker, default is 800.

      :type: IntConversion

   .. attribute:: TopBorder

      Top border setting for the tracker, default is 0.

      :type: IntConversion

   .. attribute:: BottomBorder

      Bottom border setting for the tracker, default is 600.

      :type: IntConversion

   .. attribute:: AutoExposure

      Auto exposure setting for the tracker, default is False.

      :type: bool

   .. attribute:: OverlayPath

      Overlay path setting for the tracker, default is False.

      :type: bool

   .. attribute:: sample_rate

      Sample rate setting for the tracker, default is 30.

      :type: IntConversion

   .. method:: load(path2data: Path) -> np.ndarray

      Load Tracking Port data from a specified path.

   .. method:: load_times(path2data: Path) -> np.ndarray

      Load timestamps from a specified path.















   ..
       !! processed by numpydoc !!

   .. py:method:: load(path2data)

      
      Load Tracking Port data from a specified path.

      :param path2data: The path to the directory containing the data file.
      :type path2data: Path

      :returns: A 2D numpy array with the position data.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: load_times(path2data)

      
      Load timestamps from a specified path.

      :param path2data: The path to the directory containing the timestamps file.
      :type path2data: Path

      :returns: A numpy array containing the timestamps.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:attribute:: AutoExposure
      :type:  bool
      :value: False



   .. py:attribute:: BottomBorder
      :type:  IntConversion


   .. py:attribute:: Brightness
      :type:  IntConversion


   .. py:attribute:: Contrast
      :type:  IntConversion


   .. py:attribute:: Exposure
      :type:  IntConversion


   .. py:attribute:: LeftBorder
      :type:  IntConversion


   .. py:attribute:: OverlayPath
      :type:  bool
      :value: False



   .. py:attribute:: RightBorder
      :type:  IntConversion


   .. py:attribute:: TopBorder
      :type:  IntConversion


   .. py:attribute:: sample_rate
      :type:  IntConversion


.. py:class:: ProcessorFactory

   
   Factory class for creating various processor objects
   based on the processor name.

   .. attribute:: factory

      An instance of AbstractProcessorFactory used to create
      processor objects.

      :type: AbstractProcessorFactory

   .. method:: create_processor(proc_name: str)

      Create a processor object based on the processor name.















   ..
       !! processed by numpydoc !!

   .. py:method:: create_processor(proc_name)

      
      Create a processor object based on the processor name.

      :param proc_name: The name of the processor to create.
      :type proc_name: str

      :returns: The created processor object.
      :rtype: object















      ..
          !! processed by numpydoc !!


   .. py:attribute:: factory


.. py:class:: RecordNode

   Bases: :py:obj:`OEPlugin`


   
   Documents the RecordNode plugin.

   .. attribute:: path

      The file path associated with the RecordNode.

      :type: str

   .. attribute:: engine

      The engine used by the RecordNode.

      :type: str

   .. attribute:: recordEvents

      Indicates if events are recorded, converted from a string.

      :type: IntConversion

   .. attribute:: recordSpikes

      Indicates if spikes are recorded, converted from a string.

      :type: IntConversion

   .. attribute:: isMainStream

      Indicates if this is the main stream, converted from a string.

      :type: IntConversion

   .. attribute:: sync_line

      The sync line, converted from a string.

      :type: IntConversion

   .. attribute:: source_node_id

      The source node ID, converted from a string.

      :type: IntConversion

   .. attribute:: recording_state

      The recording state of the RecordNode.

      :type: str















   ..
       !! processed by numpydoc !!

   .. py:attribute:: engine
      :type:  str
      :value: ''



   .. py:attribute:: isMainStream
      :type:  IntConversion


   .. py:attribute:: path
      :type:  str
      :value: ''



   .. py:attribute:: recordEvents
      :type:  IntConversion


   .. py:attribute:: recordSpikes
      :type:  IntConversion


   .. py:attribute:: recording_state
      :type:  str
      :value: ''



   .. py:attribute:: source_node_id
      :type:  IntConversion


   .. py:attribute:: sync_line
      :type:  IntConversion


.. py:class:: RhythmFPGA

   Bases: :py:obj:`OEPlugin`


   
   Documents the Rhythm FPGA plugin.

   .. attribute:: channel_info

      A list containing information about each channel.

      :type: list of Channel















   ..
       !! processed by numpydoc !!

   .. py:attribute:: channel_info
      :type:  list[Channel]
      :value: []



.. py:class:: RippleDetector

   Bases: :py:obj:`OEPlugin`


   
   Documents the Ripple Detector plugin.

   .. attribute:: Ripple_Input

      Input setting for the Ripple Detector, default is -1.

      :type: IntConversion

   .. attribute:: Ripple_Out

      Output setting for the Ripple Detector, default is -1.

      :type: IntConversion

   .. attribute:: Ripple_save

      Save setting for the Ripple Detector, default is -1.

      :type: IntConversion

   .. attribute:: ripple_std

      Standard deviation setting for the Ripple Detector, default is -1.

      :type: FloatConversion

   .. attribute:: time_thresh

      Time threshold setting for the Ripple Detector, default is -1.

      :type: FloatConversion

   .. attribute:: refr_time

      Refractory time setting for the Ripple Detector, default is -1.

      :type: FloatConversion

   .. attribute:: rms_samples

      RMS samples setting for the Ripple Detector, default is -1.

      :type: FloatConversion

   .. attribute:: ttl_duration

      TTL duration setting for the Ripple Detector, default is -1.

      :type: FloatConversion

   .. attribute:: ttl_percent

      TTL percent setting for the Ripple Detector, default is -1.

      :type: FloatConversion

   .. attribute:: mov_detect

      Movement detection setting for the Ripple Detector, default is -1.

      :type: IntConversion

   .. attribute:: mov_input

      Movement input setting for the Ripple Detector, default is -1.

      :type: IntConversion

   .. attribute:: mov_out

      Movement output setting for the Ripple Detector, default is -1.

      :type: IntConversion

   .. attribute:: mov_std

      Movement standard deviation setting for the Ripple Detector,
      default is -1.

      :type: FloatConversion

   .. attribute:: min_time_st

      Minimum time setting for the Ripple Detector, default is -1.

      :type: FloatConversion

   .. attribute:: min_time_mov

      Minimum movement time setting for the Ripple Detector, default is -1.

      :type: FloatConversion

   .. method:: load_ttl(path2TTL: Path, trial_start_time: float) -> dict

      Load TTL data from a specified path and trial start time.















   ..
       !! processed by numpydoc !!

   .. py:method:: load_ttl(path2TTL, trial_start_time)

      
      Load TTL data from a specified path and trial start time.

      :param path2TTL: The path to the directory containing the TTL data files.
      :type path2TTL: Path
      :param trial_start_time: The start time of the trial.
      :type trial_start_time: float

      :returns: A dictionary containing the TTL timestamps and other related data.
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:attribute:: Ripple_Input
      :type:  IntConversion


   .. py:attribute:: Ripple_Out
      :type:  IntConversion


   .. py:attribute:: Ripple_save
      :type:  IntConversion


   .. py:attribute:: min_time_mov
      :type:  FloatConversion


   .. py:attribute:: min_time_st
      :type:  FloatConversion


   .. py:attribute:: mov_detect
      :type:  IntConversion


   .. py:attribute:: mov_input
      :type:  IntConversion


   .. py:attribute:: mov_out
      :type:  IntConversion


   .. py:attribute:: mov_std
      :type:  FloatConversion


   .. py:attribute:: refr_time
      :type:  FloatConversion


   .. py:attribute:: ripple_std
      :type:  FloatConversion


   .. py:attribute:: rms_samples
      :type:  FloatConversion


   .. py:attribute:: time_thresh
      :type:  FloatConversion


   .. py:attribute:: ttl_duration
      :type:  FloatConversion


   .. py:attribute:: ttl_percent
      :type:  FloatConversion


.. py:class:: Settings(pname)

   Bases: :py:obj:`object`


   
   Groups together the other classes in this module and does the actual
   parsing of the settings.xml file.

   :param pname: The pathname to the top-level directory, typically in form
                 YYYY-MM-DD_HH-MM-SS.
   :type pname: str or Path

   .. attribute:: filename

      The path to the settings.xml file.

      :type: str or None

   .. attribute:: tree

      The parsed XML tree of the settings.xml file.

      :type: ElementTree or None

   .. attribute:: processors

      Dictionary of processor objects.

      :type: OrderedDict

   .. attribute:: record_nodes

      Dictionary of record node objects.

      :type: OrderedDict

   .. method:: load()

      Creates a handle to the basic XML document.

   .. method:: parse()

      Parses the basic information about the processors in the
      open-ephys signal chain as described in the settings.xml file(s).

   .. method:: get_processor(key: str)

      Returns the information about the requested processor or an
      empty OEPlugin instance if it's not available.















   ..
       !! processed by numpydoc !!

   .. py:method:: get_processor(key)

      
      Returns the information about the requested processor or an
      empty OEPlugin instance if it's not available.

      :param key: The key of the processor to retrieve.
      :type key: str

      :returns: The requested processor object or an empty OEPlugin instance.
      :rtype: object















      ..
          !! processed by numpydoc !!


   .. py:method:: load()

      
      Creates a handle to the basic XML document.
















      ..
          !! processed by numpydoc !!


   .. py:method:: parse()

      
      Parses the basic information about the processors in the
      open-ephys signal chain as described in the settings.xml file(s).
















      ..
          !! processed by numpydoc !!


   .. py:attribute:: filename
      :value: None



   .. py:attribute:: processors


   .. py:attribute:: record_nodes


   .. py:attribute:: tree
      :value: None



.. py:class:: SpikeSorter

   Bases: :py:obj:`OEPlugin`


   
   Documents an OE plugin.

   .. attribute:: name

      The name of the plugin.

      :type: str

   .. attribute:: insertionPoint

      The insertion point of the plugin, converted from a string.

      :type: IntConversion

   .. attribute:: pluginName

      The name of the plugin.

      :type: str

   .. attribute:: type

      The type of the plugin, converted from a string.

      :type: IntConversion

   .. attribute:: index

      The index of the plugin, converted from a string.

      :type: IntConversion

   .. attribute:: libraryName

      The name of the library.

      :type: str

   .. attribute:: libraryVersion

      The version of the library.

      :type: str

   .. attribute:: processorType

      The type of processor, converted from a string.

      :type: IntConversion

   .. attribute:: nodeId

      The node ID, converted from a string.

      :type: IntConversion

   .. attribute:: channel_count

      The number of channels, converted from a string.

      :type: IntConversion

   .. attribute:: stream

      The data stream associated with the plugin.

      :type: Stream

   .. attribute:: sample_rate

      The sample rate, converted from a string.

      :type: FloatConversion















   ..
       !! processed by numpydoc !!

.. py:class:: SpikeViewer

   Bases: :py:obj:`OEPlugin`


   
   Documents an OE plugin.

   .. attribute:: name

      The name of the plugin.

      :type: str

   .. attribute:: insertionPoint

      The insertion point of the plugin, converted from a string.

      :type: IntConversion

   .. attribute:: pluginName

      The name of the plugin.

      :type: str

   .. attribute:: type

      The type of the plugin, converted from a string.

      :type: IntConversion

   .. attribute:: index

      The index of the plugin, converted from a string.

      :type: IntConversion

   .. attribute:: libraryName

      The name of the library.

      :type: str

   .. attribute:: libraryVersion

      The version of the library.

      :type: str

   .. attribute:: processorType

      The type of processor, converted from a string.

      :type: IntConversion

   .. attribute:: nodeId

      The node ID, converted from a string.

      :type: IntConversion

   .. attribute:: channel_count

      The number of channels, converted from a string.

      :type: IntConversion

   .. attribute:: stream

      The data stream associated with the plugin.

      :type: Stream

   .. attribute:: sample_rate

      The sample rate, converted from a string.

      :type: FloatConversion















   ..
       !! processed by numpydoc !!

.. py:class:: StimControl

   Bases: :py:obj:`OEPlugin`


   
   Documents the StimControl plugin.

   .. attribute:: Device

      Device setting for the StimControl, default is 0.

      :type: IntConversion

   .. attribute:: Duration

      Duration setting for the StimControl, default is 0.

      :type: IntConversion

   .. attribute:: Interval

      Interval setting for the StimControl, default is 0.

      :type: IntConversion

   .. attribute:: Gate

      Gate setting for the StimControl, default is 0.

      :type: IntConversion

   .. attribute:: Output

      Output setting for the StimControl, default is 0.

      :type: IntConversion

   .. attribute:: Start

      Start setting for the StimControl, default is 0.

      :type: IntConversion

   .. attribute:: Stop

      Stop setting for the StimControl, default is 0.

      :type: IntConversion

   .. attribute:: Trigger

      Trigger setting for the StimControl, default is 0.

      :type: IntConversion















   ..
       !! processed by numpydoc !!

   .. py:attribute:: Device
      :type:  IntConversion


   .. py:attribute:: Duration
      :type:  IntConversion


   .. py:attribute:: Gate
      :type:  IntConversion


   .. py:attribute:: Interval
      :type:  IntConversion


   .. py:attribute:: Output
      :type:  IntConversion


   .. py:attribute:: Start
      :type:  IntConversion


   .. py:attribute:: Stop
      :type:  IntConversion


   .. py:attribute:: Trigger
      :type:  IntConversion


.. py:class:: Stream

   
   Documents an OE DataStream.

   .. attribute:: name

      The name of the data stream.

      :type: str

   .. attribute:: description

      A description of the data stream.

      :type: str

   .. attribute:: sample_rate

      The sample rate of the data stream, converted from a string.

      :type: FloatConversion

   .. attribute:: channel_count

      The number of channels in the data stream, converted from a string.

      :type: IntConversion















   ..
       !! processed by numpydoc !!

   .. py:attribute:: channel_count
      :type:  IntConversion


   .. py:attribute:: description
      :type:  str
      :value: ''



   .. py:attribute:: name
      :type:  str
      :value: ''



   .. py:attribute:: sample_rate
      :type:  FloatConversion


.. py:class:: TrackMe

   Bases: :py:obj:`OEPlugin`


   
   Documents the TrackMe plugin.

   .. method:: load(path2data: Path) -> np.ndarray

      Load TrackMe data from a specified path.

   .. method:: load_times(path2data: Path) -> np.ndarray

      Load timestamps from a specified path.

   .. method:: load_frame_count(path2data: Path) -> np.ndarray

      Load frame count data from a specified path.

   .. method:: load_ttl_times(path2data: Path) -> np.ndarray

      Load TTL times from a specified path.















   ..
       !! processed by numpydoc !!

   .. py:method:: load(path2data)

      
      Load TrackMe data from a specified path.

      :param path2data: The path to the directory containing the data file.
      :type path2data: Path

      :returns: A 2D numpy array with the TrackMe data.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: load_frame_count(path2data)

      
      Load frame count data from a specified path.

      :param path2data: The path to the directory containing the data file.
      :type path2data: Path

      :returns: A numpy array containing the frame count data.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: load_times(path2data)

      
      Load timestamps from a specified path.

      :param path2data: The path to the directory containing the timestamps file.
      :type path2data: Path

      :returns: A numpy array containing the timestamps.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: load_ttl_times(path2data)

      
      Load TTL times from a specified path.

      :param path2data: The path to the directory containing the timestamps
                        and states files.
      :type path2data: Path

      :returns: A numpy array containing the TTL times.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


.. py:class:: TrackingPort

   Bases: :py:obj:`OEPlugin`


   
   Documents the Tracking Port plugin which uses Bonsai input
   and Tracking Visual plugin for visualisation within OE
















   ..
       !! processed by numpydoc !!

   .. py:method:: load(path2data)

      
      Load Tracking Port data from a specified path.

      :param path2data: The path to the directory containing the data file.
      :type path2data: Path

      :returns: A 2D numpy array with the position data.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: load_times(path2data)

      
      Load timestamps from a specified path.

      :param path2data: The path to the directory containing the timestamps file.
      :type path2data: Path

      :returns: A numpy array containing the timestamps.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


.. py:function:: addValues2Class(node, cls)

   
   Add values from an XML node to a dataclass instance.

   :param node: The XML element node containing the values.
   :type node: ET.Element
   :param cls: The dataclass instance to which the values will be added.
   :type cls: dataclass















   ..
       !! processed by numpydoc !!

.. py:function:: recurseNode(node, func, cls)

   
   Recursive function that applies a function to each node.

   :param node: The current XML element node.
   :type node: ET.Element
   :param func: The function to apply to each node.
   :type func: Callable
   :param cls: The dataclass instance to pass to the function.
   :type cls: dataclass















   ..
       !! processed by numpydoc !!

