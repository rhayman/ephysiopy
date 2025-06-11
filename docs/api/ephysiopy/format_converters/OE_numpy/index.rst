ephysiopy.format_converters.OE_numpy
====================================

.. py:module:: ephysiopy.format_converters.OE_numpy


Classes
-------

.. autoapisummary::

   ephysiopy.format_converters.OE_numpy.OE2Numpy


Module Contents
---------------

.. py:class:: OE2Numpy(filename_root)

   Bases: :py:obj:`object`


   
   Converts openephys data recorded in the nwb format into numpy files.

   .. rubric:: Notes

   Only exports the LFP and TTL files at the moment.















   ..
       !! processed by numpydoc !!

   .. py:method:: exportLFP(channels, output_freq)

      
      Exports LFP data to numpy files.

      :param channels: List of channel indices to export.
      :type channels: list of int
      :param output_freq: The output sampling frequency.
      :type output_freq: int

      .. rubric:: Notes

      The LFP data is resampled to the specified output frequency.















      ..
          !! processed by numpydoc !!


   .. py:method:: exportRaw2Binary(output_fname=None)

      
      Exports raw data to a binary file.

      :param output_fname: The name of the output binary file. If not provided, the output
                           file name is derived from the root file name with a '.bin' extension.
      :type output_fname: str, optional

      .. rubric:: Notes

      The raw data is saved in binary format using numpy's save function.















      ..
          !! processed by numpydoc !!


   .. py:method:: exportTTL()

      
      Exports TTL data to numpy files.

      .. rubric:: Notes

      The TTL state and timestamps are saved as 'ttl_state.npy' and
      'ttl_timestamps.npy' respectively in the specified directory.















      ..
          !! processed by numpydoc !!


   .. py:method:: getOEData(filename_root, recording_name='recording1')

      
      Loads the nwb file names in filename_root and returns a dict
      containing some of the nwb data relevant for converting to Axona
      file formats.

      :param filename_root: Fully qualified name of the nwb file.
      :type filename_root: str
      :param recording_name: The name of the recording in the nwb file. Default is 'recording1'.
                             Note that the default has changed in different versions of OE from
                             'recording0' to 'recording1'.
      :type recording_name: str, optional

      :returns: A dictionary containing the nwb data.Loads the nwb file names in
                filename_root and returns a dict
                containing some of the nwb data relevant for converting to Axona
                file formats.
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: resample(data, src_rate=30, dst_rate=50, axis=0)

      
      Resamples data using FFT.

      :param data: The input data to resample.
      :type data: array_like
      :param src_rate: The source sampling rate. Default is 30.
      :type src_rate: int, optional
      :param dst_rate: The destination sampling rate. Default is 50.
      :type dst_rate: int, optional
      :param axis: The axis along which to resample. Default is 0.
      :type axis: int, optional

      :returns: **new_data** -- The resampled data.
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:attribute:: OE_data
      :value: None



   .. py:attribute:: _settings
      :value: None



   .. py:attribute:: dirname


   .. py:attribute:: experiment_name


   .. py:attribute:: filename_root


   .. py:attribute:: fs
      :value: None



   .. py:attribute:: lfp_highcut
      :value: None



   .. py:attribute:: lfp_lowcut
      :value: None



   .. py:attribute:: recording_name
      :value: None



   .. py:property:: settings

      
      Loads the settings data from the settings.xml file
















      ..
          !! processed by numpydoc !!


