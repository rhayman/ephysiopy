ephysiopy.axona.file_headers
============================

.. py:module:: ephysiopy.axona.file_headers


Attributes
----------

.. autoapisummary::

   ephysiopy.axona.file_headers.common_entries
   ephysiopy.axona.file_headers.eeg_entries
   ephysiopy.axona.file_headers.egf_entries
   ephysiopy.axona.file_headers.entries_groups
   ephysiopy.axona.file_headers.entries_to_number
   ephysiopy.axona.file_headers.entries_to_number_one_indexed
   ephysiopy.axona.file_headers.entries_to_number_to_nine
   ephysiopy.axona.file_headers.entries_to_number_to_sixteen
   ephysiopy.axona.file_headers.entries_to_number_to_three
   ephysiopy.axona.file_headers.entries_to_replace_one
   ephysiopy.axona.file_headers.lfp_entries
   ephysiopy.axona.file_headers.pos_entries
   ephysiopy.axona.file_headers.set_meta_info
   ephysiopy.axona.file_headers.singleton_entries
   ephysiopy.axona.file_headers.tetrode_entries


Classes
-------

.. autoapisummary::

   ephysiopy.axona.file_headers.AxonaHeader
   ephysiopy.axona.file_headers.CutHeader
   ephysiopy.axona.file_headers.EEGHeader
   ephysiopy.axona.file_headers.EGFHeader
   ephysiopy.axona.file_headers.LFPHeader
   ephysiopy.axona.file_headers.PosHeader
   ephysiopy.axona.file_headers.SetHeader
   ephysiopy.axona.file_headers.TetrodeHeader


Functions
---------

.. autoapisummary::

   ephysiopy.axona.file_headers.make_cluster_cut_entries
   ephysiopy.axona.file_headers.make_common_entries
   ephysiopy.axona.file_headers.make_cut_header
   ephysiopy.axona.file_headers.make_eeg_entries
   ephysiopy.axona.file_headers.make_egf_entries
   ephysiopy.axona.file_headers.make_pos_entries
   ephysiopy.axona.file_headers.make_set_entries
   ephysiopy.axona.file_headers.make_set_meta
   ephysiopy.axona.file_headers.make_tetrode_entries


Module Contents
---------------

.. py:class:: AxonaHeader

   Bases: :py:obj:`abc.ABC`


   
   Empty Axona header class
















   ..
       !! processed by numpydoc !!

   .. py:method:: __setattr__(name, value)


   .. py:method:: print()


   .. py:attribute:: common
      :type:  dict


.. py:class:: CutHeader

   Bases: :py:obj:`AxonaHeader`


   
   Empty Axona header class
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: common
      :type:  dict


.. py:class:: EEGHeader

   Bases: :py:obj:`LFPHeader`


   
   Empty EEG header class for Axona
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: lfp_entries
      :type:  dict


.. py:class:: EGFHeader

   Bases: :py:obj:`LFPHeader`


   
   Empty EGF header class for Axona
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: lfp_entries
      :type:  dict


.. py:class:: LFPHeader

   Bases: :py:obj:`AxonaHeader`


   
   Empty LFP header class for Axona
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: _n_samples
      :type:  str
      :value: None



   .. py:property:: n_samples


.. py:class:: PosHeader

   Bases: :py:obj:`AxonaHeader`


   
   Empty .pos header class for Axona
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: pos
      :type:  dict


.. py:class:: SetHeader

   Bases: :py:obj:`AxonaHeader`


   
   Empty .set header class for Axona
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: meta_info
      :type:  dict


   .. py:attribute:: set_entries
      :type:  dict


.. py:class:: TetrodeHeader

   Bases: :py:obj:`AxonaHeader`


   
   Empty tetrode header class for Axona
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: tetrode_entries
      :type:  dict


.. py:function:: make_cluster_cut_entries(n_clusters = 31, n_channels = 4, n_params = 2)

   
   Create the cluster entries for the cut file

   :param n_clusters: Number of clusters
   :type n_clusters: int
   :param n_channels: Number of channels
   :type n_channels: int
   :param n_params: Number of parameters
   :type n_params: int

   :returns: String of the cluster entries for the cut file
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: make_common_entries()

.. py:function:: make_cut_header(n_clusters = 31, n_channels = 4, n_params = 2)

   
   Create the header part of the cut file

   :param n_clusters: Number of clusters
   :type n_clusters: int
   :param n_channels: Number of channels
   :type n_channels: int
   :param n_params: Number of parameters
   :type n_params: int

   :returns: Dictionary of the cut file header
   :rtype: dict















   ..
       !! processed by numpydoc !!

.. py:function:: make_eeg_entries()

.. py:function:: make_egf_entries()

.. py:function:: make_pos_entries()

.. py:function:: make_set_entries()

   
   Create the set entries for the .set file
















   ..
       !! processed by numpydoc !!

.. py:function:: make_set_meta()

.. py:function:: make_tetrode_entries()

.. py:data:: common_entries
   :value: [('trial_date', None), ('trial_time', None), ('experimenter', None), ('comments', None),...


.. py:data:: eeg_entries
   :value: [('sample_rate', '250 hz'), ('num_EEG_samples', None), ('EEG_samples_per_position', '5'),...


.. py:data:: egf_entries
   :value: [('sample_rate', '4800 hz'), ('num_EGF_samples', None), ('bytes_per_sample', '2')]


.. py:data:: entries_groups
   :value: [('groups_X_Y', '0')]


.. py:data:: entries_to_number
   :value: [('gain_ch_', '0'), ('filter_ch_', '0'), ('a_in_ch_', '0'), ('b_in_ch_', '0'), ('mode_ch_',...


.. py:data:: entries_to_number_one_indexed
   :value: [('EEG_ch_', '0'), ('saveEEG_ch_', '0'), ('BPFEEG_ch_', '0')]


.. py:data:: entries_to_number_to_nine
   :value: [('slot_chan_', '0')]


.. py:data:: entries_to_number_to_sixteen
   :value: [('collectMask_', '0'), ('stereoMask_', '0'), ('monoMask_', '0'), ('EEGmap_', '0')]


.. py:data:: entries_to_number_to_three
   :value: [('BPFrecord', '0'), ('BPFbit', '0'), ('BPFEEGin', '0')]


.. py:data:: entries_to_replace_one
   :value: [('colmap_1_rmin', '0'), ('colmap_1_rmax', '0'), ('colmap_1_gmin', '0'), ('colmap_1_gmax', '0'),...


.. py:data:: lfp_entries
   :value: [('sw_version', '1.1.0'), ('num_chans', '1'), ('sample_rate', None), ('bytes_per_sample', None)]


.. py:data:: pos_entries
   :value: [('min_x', None), ('max_x', None), ('min_y', None), ('max_y', None), ('window_min_x', None),...


.. py:data:: set_meta_info
   :value: [('sw_version', None), ('ADC_fullscale_mv', None), ('tracker_version', None), ('stim_version',...


.. py:data:: singleton_entries
   :value: [('second_audio', '0'), ('default_filtresp_hp', '0'), ('default_filtkind_hp', '0'),...


.. py:data:: tetrode_entries
   :value: [('num_spikes', None), ('sw_version', '1.1.0'), ('num_chans', '4'), ('timebase', '96000'),...


