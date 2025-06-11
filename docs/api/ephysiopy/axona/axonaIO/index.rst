ephysiopy.axona.axonaIO
=======================

.. py:module:: ephysiopy.axona.axonaIO


Attributes
----------

.. autoapisummary::

   ephysiopy.axona.axonaIO.BOXCAR
   ephysiopy.axona.axonaIO.MAXSPEED


Classes
-------

.. autoapisummary::

   ephysiopy.axona.axonaIO.ClusterSession
   ephysiopy.axona.axonaIO.EEG
   ephysiopy.axona.axonaIO.IO
   ephysiopy.axona.axonaIO.Pos
   ephysiopy.axona.axonaIO.Stim
   ephysiopy.axona.axonaIO.Tetrode


Module Contents
---------------

.. py:class:: ClusterSession(fname_root)

   Bases: :py:obj:`object`


   
   Loads all the cut file data and timestamps from the data
   associated with the *.set filename given to __init__

   Meant to be a method-replica of the KiloSortSession class
   but really both should inherit from the same meta-class















   ..
       !! processed by numpydoc !!

   .. py:method:: load()


   .. py:attribute:: cluster_id
      :value: None



   .. py:attribute:: fname_root


   .. py:attribute:: good_clusters


   .. py:attribute:: spike_times
      :value: None



   .. py:attribute:: spk_clusters
      :value: None



.. py:class:: EEG(filename_root, eeg_file=1, egf=0)

   Bases: :py:obj:`IO`


   
   Processes eeg data collected with the Axona recording system

   :param filename_root: The fully qualified filename without the suffix
   :type filename_root: str
   :param egf: Whether to read the 'eeg' file or the 'egf' file. 0 is False, 1 is True
   :type egf: int
   :param eeg_file: If more than one eeg channel was recorded from then they are numbered
                    from 1 onwards i.e. trial.eeg, trial.eeg1, trial.eeg2 etc
                    This number specifies that
   :type eeg_file: int















   ..
       !! processed by numpydoc !!

   .. py:attribute:: EEGphase
      :value: None



   .. py:attribute:: eeg


   .. py:attribute:: filename_root


   .. py:attribute:: gain


   .. py:attribute:: header


   .. py:attribute:: polarity
      :value: 1



   .. py:attribute:: sample_rate
      :value: 0



   .. py:attribute:: scaling


   .. py:attribute:: showfigs
      :value: 0



   .. py:attribute:: sig


   .. py:attribute:: x1
      :value: 6



   .. py:attribute:: x2
      :value: 12



.. py:class:: IO(filename_root = '')

   Bases: :py:obj:`object`


   
   Axona data I/O. Also reads .clu files generated from KlustaKwik

   :param filename_root: The fully-qualified filename
   :type filename_root: str















   ..
       !! processed by numpydoc !!

   .. py:method:: getCluCut(tet)

      
      Load a clu file and return as an array of integers

      :param tet: The tetrode the clu file relates to
      :type tet: int

      :returns: **out** -- Data read from the clu file
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: getCut(tet)

      
      Returns the cut file as a list of integers

      :param tet: The tetrode the cut file relates to
      :type tet: int

      :returns: **out** -- The data read from the cut file
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: getData(filename_root)

      
      Returns the data part of an Axona data file i.e. from "data_start" to
      "data_end"

      :param filename_root: Fully qualified path name to the data file
      :type filename_root: str

      :returns: **output** -- The data part of whatever file was fed in
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: getHeader(filename_root)

      
      Reads and returns the header of a specified data file as a dictionary

      :param filename_root: Fully qualified filename of Axona type
      :type filename_root: str

      :returns: **headerDict** -- key - value pairs of the header part of an Axona type file
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: getHeaderVal(header, key)

      
      Get a value from the header as an int

      :param header: The header dictionary to read
      :type header: dict
      :param key: The key to look up
      :type key: str

      :returns: **value** -- The value of `key` as an int
      :rtype: int















      ..
          !! processed by numpydoc !!


   .. py:method:: setCut(filename_root, cut_header, cut_data)


   .. py:method:: setData(filename_root, data)

      
      Writes Axona format data to the given filename

      :param filename_root: The fully qualified filename including the suffix
      :type filename_root: str
      :param data: The data that will be saved
      :type data: ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: setHeader(filename_root, header)

      
      Writes out the header to the specified file

      :param filename_root: A fully qualified path to a file with the relevant suffix at
                            the end (e.g. ".set", ".pos" or whatever)
      :type filename_root: str
      :param header: See ephysiopy.axona.file_headers
      :type header: dataclass















      ..
          !! processed by numpydoc !!


   .. py:attribute:: axona_files


   .. py:attribute:: filename_root
      :value: ''



   .. py:attribute:: other_files


   .. py:attribute:: tetrode_files


.. py:class:: Pos(filename_root, *args, **kwargs)

   Bases: :py:obj:`IO`


   
   Processs position data recorded with the Axona recording system

   :param filename_root: The basename of the file i.e mytrial as opposed to mytrial.pos
   :type filename_root: str

   .. rubric:: Notes

   Currently the only arg that does anything is 'cm' which will convert
   the xy data to cm, assuming that the pixels per metre value has been
   set correctly















   ..
       !! processed by numpydoc !!

   .. py:attribute:: _ppm
      :value: None



   .. py:attribute:: dir


   .. py:attribute:: dir_disp


   .. py:attribute:: filename_root


   .. py:attribute:: header


   .. py:attribute:: led_pix


   .. py:attribute:: led_pos


   .. py:attribute:: nLEDs
      :value: 1



   .. py:attribute:: npos


   .. py:attribute:: posProcessed
      :value: False



   .. py:attribute:: pos_sample_rate


   .. py:property:: ppm


   .. py:attribute:: setheader
      :value: None



   .. py:attribute:: speed


   .. py:attribute:: ts


   .. py:attribute:: xy


.. py:class:: Stim(filename_root, *args, **kwargs)

   Bases: :py:obj:`dict`, :py:obj:`IO`


   
   Processes the stimulation data recorded using Axona

   :param filename_root: The fully qualified filename without the suffix
   :type filename_root: str















   ..
       !! processed by numpydoc !!

   .. py:method:: __getitem__(key)

      
      x.__getitem__(y) <==> x[y]
















      ..
          !! processed by numpydoc !!


   .. py:method:: __setitem__(key, val)

      
      Set self[key] to value.
















      ..
          !! processed by numpydoc !!


   .. py:method:: update(*args, **kwargs)

      
      D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
      If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
      If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
      In either case, this is followed by: for k in F:  D[k] = F[k]
















      ..
          !! processed by numpydoc !!


   .. py:attribute:: filename_root


   .. py:attribute:: timebase


.. py:class:: Tetrode(filename_root, tetrode, volts = True)

   Bases: :py:obj:`IO`


   
   Processes tetrode files recorded with the Axona recording system

   Mostly this class deals with interpolating tetrode and position timestamps
   and getting indices for particular clusters.

   :param filename_root: The fully qualified name of the file without it's suffix
   :type filename_root: str
   :param tetrode: The number of the tetrode
   :type tetrode: int
   :param volts: Whether to convert the data values volts. Default True
   :type volts: bool, optional















   ..
       !! processed by numpydoc !!

   .. py:method:: apply_mask(mask, **kwargs)

      
      Apply a mask to the data

      :param mask:
      :type mask: np.ndarray
      :param The mask to be applied. For use with np.ma.MaskedArray's mask attribute:

      .. rubric:: Notes

      The times inside the bounds are masked ie the mask is set to True
      The mask can be a list of tuples, in which case the mask is applied
      for each tuple in the list.
      mask can be an empty tuple, in which case the mask is removed















      ..
          !! processed by numpydoc !!


   .. py:method:: getClustIdx(cluster)

      
      Get the indices of the position samples corresponding to the cluster

      :param cluster: The cluster whose position indices we want
      :type cluster: int

      :returns: The indices of the position samples, dtype is int
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: getClustSpks(cluster)

      
      Returns the waveforms of `cluster`

      :param cluster: The cluster whose waveforms we want
      :type cluster: int

      :returns: **waveforms** -- The waveforms on all 4 electrodes of the tgtrode so the shape of
                the returned array is [nClusterSpikes, 4, 50]
      :rtype: ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: getClustTS(cluster = None)

      
      Returns the timestamps for a cluster on the tetrode

      :param cluster: The cluster whose timestamps we want
      :type cluster: int

      :returns: The timestamps
      :rtype: np.ndarray

      .. rubric:: Notes

      If None is supplied as input then all timestamps for all clusters
      is returned i.e. getSpkTS() is called















      ..
          !! processed by numpydoc !!


   .. py:method:: getPosSamples()

      
      Returns the pos samples at which the spikes were captured
















      ..
          !! processed by numpydoc !!


   .. py:method:: getSpkTS()

      
      Return all the timestamps for all the spikes on the tetrode
















      ..
          !! processed by numpydoc !!


   .. py:method:: getUniqueClusters()

      
      Returns the unique clusters
















      ..
          !! processed by numpydoc !!


   .. py:attribute:: clusters


   .. py:attribute:: cut


   .. py:attribute:: duration


   .. py:attribute:: filename_root


   .. py:attribute:: header


   .. py:attribute:: nChans


   .. py:attribute:: nSpikes


   .. py:attribute:: posSampleRate


   .. py:attribute:: pos_samples
      :value: None



   .. py:attribute:: samples


   .. py:attribute:: spike_times


   .. py:attribute:: tetrode


   .. py:attribute:: timebase


   .. py:attribute:: volts
      :value: True



   .. py:attribute:: waveforms


.. py:data:: BOXCAR
   :value: 20


.. py:data:: MAXSPEED
   :value: 4.0


