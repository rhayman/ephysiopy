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


   
   Converts openephys data recorded in the nwb format into numpy files

   NB Only exports the LFP and TTL files at the moment















   ..
       !! processed by numpydoc !!

   .. py:method:: exportLFP(channels, output_freq)


   .. py:method:: exportRaw2Binary(output_fname=None)


   .. py:method:: exportTTL()


   .. py:method:: getOEData(filename_root, recording_name='recording1')

      
      Loads the nwb file names in filename_root and returns a dict
      containing some of the nwb data relevant for converting to Axona
      file formats.

      :param filename_root: Fully qualified name of the nwb file.
      :type filename_root: str
      :param recording_name: The name of the recording in the nwb file.
      :type recording_name: str
      :param Note that the default has changed in different versions of OE from:
      :param 'recording0' to 'recording1'.:















      ..
          !! processed by numpydoc !!


   .. py:method:: resample(data, src_rate=30, dst_rate=50, axis=0)

      
      Upsamples data using FFT
















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


