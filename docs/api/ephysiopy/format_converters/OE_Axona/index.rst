ephysiopy.format_converters.OE_Axona
====================================

.. py:module:: ephysiopy.format_converters.OE_Axona


Classes
-------

.. autoapisummary::

   ephysiopy.format_converters.OE_Axona.OE2Axona


Module Contents
---------------

.. py:class:: OE2Axona(pname, path2APData = None, pos_sample_rate = 50, channels = 0, **kwargs)

   Bases: :py:obj:`object`


   
   Converts openephys data into Axona files

   Example workflow:

   You have recorded some openephys data using the binary
   format leading to a directory structure something like this:

   M4643_2023-07-21_11-52-02
   ├── Record Node 101
   │ ├── experiment1
   │ │ └── recording1
   │ │     ├── continuous
   │ │     │ └── Acquisition_Board-100.Rhythm Data
   │ │     │     ├── amplitudes.npy
   │ │     │     ├── channel_map.npy
   │ │     │     ├── channel_positions.npy
   │ │     │     ├── cluster_Amplitude.tsv
   │ │     │     ├── cluster_ContamPct.tsv
   │ │     │     ├── cluster_KSLabel.tsv
   │ │     │     ├── continuous.dat
   │ │     │     ├── params.py
   │ │     │     ├── pc_feature_ind.npy
   │ │     │     ├── pc_features.npy
   │ │     │     ├── phy.log
   │ │     │     ├── rez.mat
   │ │     │     ├── similar_templates.npy
   │ │     │     ├── spike_clusters.npy
   │ │     │     ├── spike_templates.npy
   │ │     │     ├── spike_times.npy
   │ │     │     ├── template_feature_ind.npy
   │ │     │     ├── template_features.npy
   │ │     │     ├── templates_ind.npy
   │ │     │     ├── templates.npy
   │ │     │     ├── whitening_mat_inv.npy
   │ │     │     └── whitening_mat.npy
   │ │     ├── events
   │ │     │ ├── Acquisition_Board-100.Rhythm Data
   │ │     │ │ └── TTL
   │ │     │ │     ├── full_words.npy
   │ │     │ │     ├── sample_numbers.npy
   │ │     │ │     ├── states.npy
   │ │     │ │     └── timestamps.npy
   │ │     │ └── MessageCenter
   │ │     │     ├── sample_numbers.npy
   │ │     │     ├── text.npy
   │ │     │     └── timestamps.npy
   │ │     ├── structure.oebin
   │ │     └── sync_messages.txt
   │ └── settings.xml
   └── Record Node 104
       ├── experiment1
       │ └── recording1
       │     ├── continuous
       │     │ └── TrackMe-103.TrackingNode
       │     │     ├── continuous.dat
       │     │     ├── sample_numbers.npy
       │     │     └── timestamps.npy
       │     ├── events
       │     │ ├── MessageCenter
       │     │ │ ├── sample_numbers.npy
       │     │ │ ├── text.npy
       │     │ │ └── timestamps.npy
       │     │ └── TrackMe-103.TrackingNode
       │     │     └── TTL
       │     │         ├── full_words.npy
       │     │         ├── sample_numbers.npy
       │     │         ├── states.npy
       │     │         └── timestamps.npy
       │     ├── structure.oebin
       │     └── sync_messages.txt
       └── settings.xml

   The binary data file is called "continuous.dat" in the
   continuous/Acquisition_Board-100.Rhythm Data folder. There
   is also a collection of files resulting from a KiloSort session
   in that directory.

   Run the conversion code like so:

   >>> from ephysiopy.format_converters.OE_Axona import OE2Axona
   >>> from pathlib import Path
   >>> nChannels = 64
   >>> apData = Path("M4643_2023-07-21_11-52-02/Record Node 101/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data")
   >>> OE = OE2Axona(Path("M4643_2023-07-21_11-52-02"), path2APData=apData, channels=nChannels)
   >>> OE.getOEData()

   The last command will attempt to load position data and also load up
   something called a TemplateModel (from the package phylib) which
   should grab a handle to the neural data. If that doesn't throw
   out errors then try:

   >>> OE.exportPos()

   There are a few arguments you can provide the exportPos() function - see
   the docstring for it below. Basically, it calls a function called
   convertPosData(xy, xyts) where xy is the xy data with shape nsamples x 2
   and xyts is a vector of timestamps. So if the call to exportPos() fails, you
   could try calling convertPosData() directly which returns axona formatted
   position data. If the variable returned from convertPosData() is called axona_pos_data
   then you can call the function:

   writePos2AxonaFormat(pos_header, axona_pos_data)

   Providing the pos_header to it - see the last half of the exportPos function
   for how to create and modify the pos_header as that will need to have
   user-specific information added to it.

   >>> OE.convertTemplateDataToAxonaTetrode()

   This is the main function for creating the tetrode files. It has an optional
   argument called max_n_waves which is used to limit the maximum number of spikes
   that make up a cluster. This defaults to 2000 which means that if a cluster has
   12000 spikes, it will have 2000 spikes randomly drawn from those 12000 (without
   replacement), that will then be saved to a tetrode file. This is mostly a time-saving
   device as if you have 250 clusters and many consist of 10,000's of spikes,
   processing that data will take a long time.

   >>> OE.exportLFP()

   This will save either a .eeg or .egf file depending on the arguments. Check the
   docstring for how to change what channel is chosen for the LFP etc.

   >>> OE.exportSetFile()

   This should save the .set file with all the metadata for the trial.















   ..
       !! processed by numpydoc !!

   .. py:method:: __filterLFP__(data, sample_rate)

      
      Filters the LFP data.

      :param data: The LFP data to be filtered.
      :type data: np.array
      :param sample_rate: The sampling rate of the data.
      :type sample_rate: int

      :returns: The filtered LFP data.
      :rtype: np.array

      .. rubric:: Notes

      Applies a bandpass filter to the LFP data using the specified sample rate.















      ..
          !! processed by numpydoc !!


   .. py:method:: convertPosData(xy, xy_ts)

      
      Performs the conversion of the array parts of the data.

      :param xy: The x and y coordinates.
      :type xy: np.array
      :param xy_ts: The timestamps for the x and y coordinates.
      :type xy_ts: np.array

      :returns: The converted position data.
      :rtype: np.array

      .. rubric:: Notes

      Upsamples the data to the Axona position sampling rate (50Hz) and inserts
      columns into the position array to match the Axona format.















      ..
          !! processed by numpydoc !!


   .. py:method:: convertSpikeData(hdf5_tetrode_data)

      
      Converts spike data from the Open Ephys Spike Sorter format to Axona format tetrode files.

      :param hdf5_tetrode_data: The HDF5 group containing the tetrode data.
      :type hdf5_tetrode_data: h5py._hl.group.Group

      .. rubric:: Notes

      Converts the spike data and timestamps, scales them appropriately, and saves
      them in the Axona tetrode format.















      ..
          !! processed by numpydoc !!


   .. py:method:: convertTemplateDataToAxonaTetrode(max_n_waves=2000, **kwargs)

      
      Converts the data held in a TemplateModel instance into tetrode format Axona data files.

      :param max_n_waves: The maximum number of waveforms to process.
      :type max_n_waves: int, default=2000

      .. rubric:: Notes

      For each cluster, the channel with the peak amplitude is identified, and the
      data is converted to the Axona tetrode format. If a channel from a tetrode is
      missing, the spikes for that channel are zeroed when saved to the Axona format.

      .. rubric:: Examples

      If cluster 3 has a peak channel of 1 then get_cluster_channels() might look like:
      [ 1,  2,  0,  6, 10, 11,  4,  12,  7,  5,  8,  9]
      Here the cluster has the best signal on 1, then 2, 0 etc, but note that channel 3
      isn't in the list. In this case the data for channel 3 will be zeroed
      when saved to Axona format.

      .. rubric:: References

      .. [Rd5d247d6957d-1] https://phy.readthedocs.io/en/latest/api/#phyappstemplatetemplatemodel















      ..
          !! processed by numpydoc !!


   .. py:method:: exportLFP(channel = 0, lfp_type = 'eeg', gain = 5000, **kwargs)

      
      Exports LFP data to file.

      :param channel: The channel number. Default is 0.
      :type channel: int, optional
      :param lfp_type: The type of LFP data. Legal values are 'egf' or 'eeg'. Default is 'eeg'.
      :type lfp_type: str, optional
      :param gain: Multiplier for the LFP data. Default is 5000.
      :type gain: int, optional

      .. rubric:: Notes

      Converts and exports LFP data from the Open Ephys format to the Axona format.
          gain (int): Multiplier for the LFP data.















      ..
          !! processed by numpydoc !!


   .. py:method:: exportPos(ppm=300, jumpmax=100, as_text=False, **kwargs)

      
      Exports position data to either text or Axona format.

      :param ppm: Pixels per meter. Defaults to 300.
      :type ppm: int, optional
      :param jumpmax: Maximum allowed jump in position data. Defaults to 100.
      :type jumpmax: int,def
      :param as_text: If True, exports position data to text format. Defaults to False.
      :type as_text: bool, optional
      :param \*\*kwargs: Additional keyword arguments.















      ..
          !! processed by numpydoc !!


   .. py:method:: exportSetFile(**kwargs)

      
      Wrapper for makeSetData below
















      ..
          !! processed by numpydoc !!


   .. py:method:: exportSpikes()

      
      Exports spiking data.

      .. rubric:: Notes

      Converts spiking data from the Open Ephys format to the Axona format.















      ..
          !! processed by numpydoc !!


   .. py:method:: getOEData()

      
      Loads the nwb file names in filename_root and returns a dict
      containing some of the nwb data relevant for converting to Axona file formats.

      :param filename_root: Fully qualified name of the nwb file.
      :type filename_root: str
      :param recording_name: The name of the recording in the nwb file. Note that
                             the default has changed in different versions of OE from 'recording0'
                             to 'recording1'.
      :type recording_name: str

      :returns: An instance of OpenEphysBase containing the loaded data.
      :rtype: OpenEphysBase















      ..
          !! processed by numpydoc !!


   .. py:method:: makeLFPData(data, eeg_type='eeg', gain=5000)

      
      Downsamples the data and saves the result as either an EGF or EEG file.

      :param data: The data to be downsampled. Must have dtype as np.int16.
      :type data: np.ndarray
      :param eeg_type: The type of LFP data. Legal values are 'egf' or 'eeg'. Default is 'eeg'.
      :type eeg_type: str, optional
      :param gain: The scaling factor. Default is 5000.
      :type gain: int, optional

      .. rubric:: Notes

      Downsamples the data to the specified rate and applies a filter. The data is
      then scaled and saved in the Axona format.















      ..
          !! processed by numpydoc !!


   .. py:method:: makeSetData(lfp_channel=4, **kwargs)

      
      Creates and writes the SET file data.

      :param lfp_channel: The LFP channel number. Default is 4.
      :type lfp_channel: int, optional

      .. rubric:: Notes

      Creates the SET file header and entries based on the provided parameters and
      writes the data to the Axona format.















      ..
          !! processed by numpydoc !!


   .. py:method:: resample(data, src_rate=30, dst_rate=50, axis=0)

      
      Resamples data using FFT.

      :param data: The input data to be resampled.
      :type data: array_like
      :param src_rate: The original sampling rate of the data. Defaults to 30.
      :type src_rate: int, optional
      :param dst_rate: The desired sampling rate of the resampled data. Defaults to 50.
      :type dst_rate: int, optional
      :param axis: The axis along which to resample. Defaults to 0.
      :type axis: int, optional

      :returns: **new_data** -- The resampled data.
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: writeCutData(itet, header, data)

      
      Writes cut data to the Axona format.

      :param itet: The tetrode identifier.
      :type itet: str
      :param header: The header information for the cut file.
      :type header: dataclass
      :param data: The cut data to be written.
      :type data: np.array

      .. rubric:: Notes

      Writes the cut data and header to the Axona format.















      ..
          !! processed by numpydoc !!


   .. py:method:: writeLFP2AxonaFormat(header, data, eeg_type='eeg')

      
      Writes LFP data to the Axona format.

      :param header: The header information for the LFP file.
      :type header: dataclass
      :param data: The LFP data to be written.
      :type data: np.array
      :param eeg_type: The type of LFP data. Legal values are 'egf' or 'eeg'. Default is 'eeg'.
      :type eeg_type: str, optional

      .. rubric:: Notes

      Writes the LFP data and header to the Axona format.















      ..
          !! processed by numpydoc !!


   .. py:method:: writePos2AxonaFormat(header, data)


   .. py:method:: writeSetData(header)

      
      Writes SET data to the Axona format.

      :param header: The header information for the SET file.
      :type header: dataclass

      .. rubric:: Notes

      Writes the SET data and header to the Axona format.















      ..
          !! processed by numpydoc !!


   .. py:method:: writeTetrodeData(itet, header, data)

      
      Writes tetrode data to the Axona format.

      :param itet: The tetrode identifier.
      :type itet: str
      :param header: The header information for the tetrode file.
      :type header: dataclass
      :param data: The tetrode data to be written.
      :type data: np.array

      .. rubric:: Notes

      Writes the tetrode data and header to the Axona format.















      ..
          !! processed by numpydoc !!


   .. py:attribute:: AxonaData


   .. py:attribute:: OE_data
      :value: None



   .. py:attribute:: _settings
      :value: None



   .. py:attribute:: axona_root_name


   .. py:attribute:: bitvolts
      :value: 0.195



   .. py:attribute:: channel_count
      :value: 0



   .. py:attribute:: experiment_name
      :type:  pathlib.Path


   .. py:attribute:: fs
      :value: None



   .. py:attribute:: hp_gain
      :value: 500



   .. py:attribute:: lfp_channel


   .. py:attribute:: lfp_highcut
      :value: None



   .. py:attribute:: lfp_lowcut
      :value: None



   .. py:attribute:: lp_gain
      :value: 15000



   .. py:attribute:: path2APdata
      :type:  pathlib.Path
      :value: None



   .. py:attribute:: pname
      :type:  pathlib.Path


   .. py:attribute:: pos_sample_rate
      :type:  int
      :value: 50



   .. py:attribute:: recording_name
      :value: None



   .. py:property:: settings

      
      Loads the settings data from the settings.xml file
















      ..
          !! processed by numpydoc !!


   .. py:attribute:: tetrodes
      :value: ['1', '2', '3', '4']



