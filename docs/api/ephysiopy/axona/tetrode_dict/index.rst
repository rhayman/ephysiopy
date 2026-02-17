ephysiopy.axona.tetrode_dict
============================

.. py:module:: ephysiopy.axona.tetrode_dict


Classes
-------

.. autoapisummary::

   ephysiopy.axona.tetrode_dict.TetrodeDict


Module Contents
---------------

.. py:class:: TetrodeDict(filename_root, *args, **kwargs)

   Bases: :py:obj:`dict`


   
   A dictionary-like object that returns a Tetrode object when
   a key is requested. The Tetrode object is created on the fly
   if it does not already exist. The Tetrode object is created
   using the axonaIO.Tetrode class. The Tetrode object is stored
   in the dictionary for future use.
















   ..
       !! processed by numpydoc !!

   .. py:method:: __getitem__(key)

      
      Return self[key].
















      ..
          !! processed by numpydoc !!


   .. py:method:: get_all_spike_timestamps(tetrode)

      
      Returns a masked array of spike timestamps for given tetrode

      :returns: A masked array of spike timestamps
      :rtype: np.ma.MaskedArray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_spike_samples(tetrode, cluster)

      
      Returns spike times in seconds for given cluster from given
      tetrode

      :param tetrode: The tetrode number
      :type tetrode: int
      :param cluster: The cluster number
      :type cluster: int

      :returns: The spike times in seconds
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: update(*args, **kwargs)

      
      D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.
      If E is present and has a .keys() method, then does:  for k in E.keys(): D[k] = E[k]
      If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
      In either case, this is followed by: for k in F:  D[k] = F[k]
















      ..
          !! processed by numpydoc !!


   .. py:attribute:: filename_root


   .. py:attribute:: use_volts


   .. py:attribute:: valid_keys


