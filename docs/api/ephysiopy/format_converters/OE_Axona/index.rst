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


   .. py:method:: convertPosData(xy, xy_ts)

      
      Performs the conversion of the array parts of the data.

      Note: As well as upsampling the data to the Axona pos sampling rate (50Hz),
      we have to insert some columns into the pos array as Axona format
      expects it like: pos_format: t,x1,y1,x2,y2,numpix1,numpix2
      We can make up some of the info and ignore other bits.















      ..
          !! processed by numpydoc !!


   .. py:method:: convertSpikeData(hdf5_tetrode_data)

      
      Does the spike conversion from OE Spike Sorter format to Axona format tetrode files.

      :param hdf5_tetrode_data: This kind of looks like a dictionary and can,
                                it seems, be treated as one more or less. See http://docs.h5py.org/en/stable/high/group.html
      :type hdf5_tetrode_data: h5py._hl.group.Group















      ..
          !! processed by numpydoc !!


   .. py:method:: convertTemplateDataToAxonaTetrode(max_n_waves=2000, **kwargs)

      
      Converts the data held in a TemplateModel instance into tetrode
      format Axona data files.

      For each cluster, there'll be a channel that has a peak amplitude and this contains that peak channel.
      While the other channels with a large signal in might be on the same tetrode, KiloSort (or whatever) might find
      channels *not* within the same tetrode. For a given cluster, we can extract from the TemplateModel the 12 channels across
      which the signal is strongest using Model.get_cluster_channels(). If a channel from a tetrode is missing from this list then the
      spikes for that channel(s) will be zeroed when saved to Axona format.

      .. rubric:: Example

      If cluster 3 has a peak channel of 1 then get_cluster_channels() might look like:
      [ 1,  2,  0,  6, 10, 11,  4,  12,  7,  5,  8,  9]
      Here the cluster has the best signal on 1, then 2, 0 etc, but note that channel 3 isn't in the list.
      In this case the data for channel 3 will be zeroed when saved to Axona format.

      .. rubric:: References

      1) https://phy.readthedocs.io/en/latest/api/#phyappstemplatetemplatemodel















      ..
          !! processed by numpydoc !!


   .. py:method:: exportLFP(channel = 0, lfp_type = 'eeg', gain = 5000, **kwargs)

      
      Exports LFP data to file.

      :param channel: The channel number.
      :type channel: int
      :param lfp_type: The type of LFP data. Legal values are 'egf' or 'eeg'.
      :type lfp_type: str
      :param gain: Multiplier for the LFP data.
      :type gain: int















      ..
          !! processed by numpydoc !!


   .. py:method:: exportPos(ppm=300, jumpmax=100, as_text=False, **kwargs)


   .. py:method:: exportSetFile(**kwargs)

      
      Wrapper for makeSetData below
















      ..
          !! processed by numpydoc !!


   .. py:method:: exportSpikes()


   .. py:method:: getOEData()

      
      Loads the nwb file names in filename_root and returns a dict
      containing some of the nwb data relevant for converting to Axona file formats.

      :param filename_root: Fully qualified name of the nwb file.
      :type filename_root: str
      :param recording_name: The name of the recording in the nwb file. Note that
                             the default has changed in different versions of OE from 'recording0'
                             to 'recording1'.
      :type recording_name: str















      ..
          !! processed by numpydoc !!


   .. py:method:: makeLFPData(data, eeg_type='eeg', gain=5000)

      
      Downsamples the data in data and saves the result as either an egf or eeg file
      depending on the choice of either eeg_type which can take a value of either 'egf' or 'eeg'.
      Gain is the scaling factor.

      :param data: The data to be downsampled. Must have dtype as np.int16.
      :type data: np.array















      ..
          !! processed by numpydoc !!


   .. py:method:: makeSetData(lfp_channel=4, **kwargs)


   .. py:method:: resample(data, src_rate=30, dst_rate=50, axis=0)

      
      Resamples data using FFT
















      ..
          !! processed by numpydoc !!


   .. py:method:: writeCutData(itet, header, data)


   .. py:method:: writeLFP2AxonaFormat(header, data, eeg_type='eeg')


   .. py:method:: writePos2AxonaFormat(header, data)


   .. py:method:: writeSetData(header)


   .. py:method:: writeTetrodeData(itet, header, data)


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



