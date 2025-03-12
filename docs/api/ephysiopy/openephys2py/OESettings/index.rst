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

   .. py:method:: create_acquisition_board()


   .. py:method:: create_bandpass_filter()


   .. py:method:: create_neuropix_pxi()


   .. py:method:: create_oe_plugin()


   .. py:method:: create_pos_tracker()


   .. py:method:: create_record_node()


   .. py:method:: create_rhythm_fpga()


   .. py:method:: create_ripple_detector()


   .. py:method:: create_spike_sorter()


   .. py:method:: create_stim_control()


   .. py:method:: create_track_me()


.. py:class:: AcquisitionBoard

   Bases: :py:obj:`OEPlugin`


   
   Documents the Acquisition Board plugin
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: HighCut
      :type:  int
      :value: 0



   .. py:attribute:: LowCut
      :type:  int
      :value: 0



.. py:class:: BandpassFilter

   Bases: :py:obj:`OEPlugin`


   
   Documents the Bandpass Filter plugin
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: channels
      :type:  list[int]
      :value: []



   .. py:attribute:: high_cut
      :type:  float


   .. py:attribute:: libraryName
      :value: 'Bandpass Filter'



   .. py:attribute:: low_cut
      :type:  float


   .. py:attribute:: name
      :value: 'Bandpass Filter'



   .. py:attribute:: pluginName
      :value: 'Bandpass Filter'



   .. py:attribute:: pluginType
      :value: 1



.. py:class:: Channel

   Bases: :py:obj:`object`


   
   Documents the information attached to each channel
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: _audio
      :type:  bool
      :value: False



   .. py:attribute:: _gain
      :type:  float


   .. py:attribute:: _highcut
      :type:  int
      :value: 0



   .. py:attribute:: _lowcut
      :type:  int
      :value: 0



   .. py:attribute:: _number
      :type:  int
      :value: 0



   .. py:attribute:: _param
      :type:  bool
      :value: False



   .. py:attribute:: _record
      :type:  bool
      :value: False



   .. py:property:: audio
      :type: bool



   .. py:property:: gain
      :type: float



   .. py:property:: highcut
      :type: int



   .. py:property:: lowcut
      :type: int



   .. py:attribute:: name
      :type:  str
      :value: ''



   .. py:property:: number
      :type: int



   .. py:property:: param
      :type: bool



   .. py:property:: record
      :type: bool



.. py:class:: Electrode

   Bases: :py:obj:`object`


   
   Documents the ELECTRODE entries in the settings.xml file
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: id
      :type:  int
      :value: 0



   .. py:attribute:: nChannels
      :type:  int
      :value: 0



   .. py:attribute:: postPeakSamples
      :type:  int
      :value: 32



   .. py:attribute:: prePeakSamples
      :type:  int
      :value: 8



   .. py:attribute:: subChannels
      :type:  list[int]
      :value: []



   .. py:attribute:: subChannelsActive
      :type:  list[int]
      :value: []



   .. py:attribute:: subChannelsThresh
      :type:  list[int]
      :value: []



.. py:class:: NeuropixPXI

   Bases: :py:obj:`OEPlugin`


   
   Documents the Neuropixels-PXI plugin
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: channel_info
      :type:  list[Channel]
      :value: []



.. py:class:: OEPlugin

   Bases: :py:obj:`abc.ABC`


   
   Documents an OE plugin
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: channel_count
      :type:  int
      :value: 0



   .. py:attribute:: index
      :type:  int
      :value: 0



   .. py:attribute:: insertionPoint
      :type:  int
      :value: 0



   .. py:attribute:: libraryName
      :type:  str
      :value: ''



   .. py:attribute:: libraryVersion
      :type:  int
      :value: 0



   .. py:attribute:: name
      :type:  str
      :value: ''



   .. py:attribute:: nodeId
      :type:  int
      :value: 0



   .. py:attribute:: pluginName
      :type:  str
      :value: ''



   .. py:attribute:: processorType
      :type:  int
      :value: 0



   .. py:attribute:: sample_rate
      :type:  int
      :value: 0



   .. py:attribute:: stream
      :type:  Stream


   .. py:attribute:: type
      :type:  int
      :value: 0



.. py:class:: OEStructure(fname)

   Bases: :py:obj:`object`


   
   Loads up the structure.oebin file for openephys flat binary
   format recordings
















   ..
       !! processed by numpydoc !!

   .. py:method:: find_oebin(pname)


   .. py:method:: read_oebin(fname)


   .. py:attribute:: data


   .. py:attribute:: filename
      :value: []



.. py:class:: PosTracker

   Bases: :py:obj:`OEPlugin`


   
   Documents the PosTracker plugin
















   ..
       !! processed by numpydoc !!

   .. py:method:: load(path2data)


   .. py:method:: load_times(path2data)


   .. py:attribute:: AutoExposure
      :type:  bool
      :value: False



   .. py:attribute:: BottomBorder
      :type:  int
      :value: 600



   .. py:attribute:: Brightness
      :type:  int
      :value: 20



   .. py:attribute:: Contrast
      :type:  int
      :value: 20



   .. py:attribute:: Exposure
      :type:  int
      :value: 20



   .. py:attribute:: LeftBorder
      :type:  int
      :value: 0



   .. py:attribute:: OverlayPath
      :type:  bool
      :value: False



   .. py:attribute:: RightBorder
      :type:  int
      :value: 800



   .. py:attribute:: TopBorder
      :type:  int
      :value: 0



   .. py:attribute:: sample_rate
      :type:  int
      :value: 30


      
      Custom methods for loading numpy arrays containing position data
      and associated timestamps
















      ..
          !! processed by numpydoc !!


.. py:class:: ProcessorFactory

   .. py:method:: create_processor(proc_name)


   .. py:attribute:: factory


.. py:class:: RecordNode

   Bases: :py:obj:`OEPlugin`


   
   Documents the RecordNode plugin
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: engine
      :type:  str
      :value: ''



   .. py:attribute:: isMainStream
      :type:  int
      :value: 0



   .. py:attribute:: path
      :type:  str
      :value: ''



   .. py:attribute:: recordEvents
      :type:  int
      :value: 0



   .. py:attribute:: recordSpikes
      :type:  int
      :value: 0



   .. py:attribute:: recording_state
      :type:  str
      :value: ''



   .. py:attribute:: source_node_id
      :type:  int
      :value: 0



   .. py:attribute:: sync_line
      :type:  int
      :value: 0



.. py:class:: RhythmFPGA

   Bases: :py:obj:`OEPlugin`


   
   Documents the Rhythm FPGA plugin
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: channel_info
      :type:  list[Channel]
      :value: []



.. py:class:: RippleDetector

   Bases: :py:obj:`OEPlugin`


   
   Documents the Ripple Detector plugin
















   ..
       !! processed by numpydoc !!

   .. py:method:: load_ttl(path2TTL, trial_start_time)


   .. py:attribute:: Ripple_Input
      :type:  int
      :value: 0



   .. py:attribute:: Ripple_Out
      :type:  int
      :value: 0



   .. py:attribute:: Ripple_save
      :type:  int
      :value: 0



   .. py:attribute:: min_time_mov
      :type:  float


   .. py:attribute:: min_time_st
      :type:  float


   .. py:attribute:: mov_detect
      :type:  int
      :value: 0



   .. py:attribute:: mov_input
      :type:  int
      :value: 0



   .. py:attribute:: mov_out
      :type:  int
      :value: 0



   .. py:attribute:: mov_std
      :type:  float


   .. py:attribute:: refr_time
      :type:  float


   .. py:attribute:: ripple_std
      :type:  float


   .. py:attribute:: rms_samples
      :type:  float


   .. py:attribute:: time_thresh
      :type:  float


   .. py:attribute:: ttl_duration
      :type:  int
      :value: 0



   .. py:attribute:: ttl_percent
      :type:  int
      :value: 0



.. py:class:: Settings(pname)

   Bases: :py:obj:`object`


   
   Groups together the other classes in this module and does the actual
   parsing of the settings.xml file

   :param pname: The pathname to the top-level directory, typically in form
                 YYYY-MM-DD_HH-MM-SS
   :type pname: str















   ..
       !! processed by numpydoc !!

   .. py:method:: get_processor(key)

      
      Returns the information about the requested processor or an
      empty OEPlugin instance if it's not available
















      ..
          !! processed by numpydoc !!


   .. py:method:: load()

      
      Creates a handle to the basic xml document
















      ..
          !! processed by numpydoc !!


   .. py:method:: parse()

      
      Parses the basic information about the processors in the
      open-ephys signal chain and as described in the settings.xml
      file(s)
















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


   
   Documents an OE plugin
















   ..
       !! processed by numpydoc !!

.. py:class:: SpikeViewer

   Bases: :py:obj:`OEPlugin`


   
   Documents an OE plugin
















   ..
       !! processed by numpydoc !!

.. py:class:: StimControl

   Bases: :py:obj:`OEPlugin`


   
   Documents the StimControl plugin
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: Device
      :type:  int
      :value: 0



   .. py:attribute:: Duration
      :type:  int
      :value: 0



   .. py:attribute:: Gate
      :type:  int
      :value: 0



   .. py:attribute:: Interval
      :type:  int
      :value: 0



   .. py:attribute:: Output
      :type:  int
      :value: 0



   .. py:attribute:: Start
      :type:  int
      :value: 0



   .. py:attribute:: Stop
      :type:  int
      :value: 0



   .. py:attribute:: Trigger
      :type:  int
      :value: 0



.. py:class:: Stream

   
   Documents an OE DatasSream
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: channel_count
      :type:  int
      :value: 0



   .. py:attribute:: description
      :type:  str
      :value: ''



   .. py:attribute:: name
      :type:  str
      :value: ''



   .. py:attribute:: sample_rate
      :type:  int
      :value: 0



.. py:class:: TrackMe

   Bases: :py:obj:`OEPlugin`


   
   Documents the TrackMe plugin
















   ..
       !! processed by numpydoc !!

   .. py:method:: load(path2data)


   .. py:method:: load_frame_count(path2data)


   .. py:method:: load_times(path2data)


   .. py:method:: load_ttl_times(path2data)


.. py:class:: TrackingPort

   Bases: :py:obj:`OEPlugin`


   
   Documents the Tracking Port plugin which uses Bonsai input
   and Tracking Visual plugin for visualisation within OE
















   ..
       !! processed by numpydoc !!

   .. py:method:: load(path2data)


   .. py:method:: load_times(path2data)


.. py:function:: addValues2Class(node, cls)

.. py:function:: recurseNode(node, func, cls)

   
   Recursive function that applies func to each node
















   ..
       !! processed by numpydoc !!

