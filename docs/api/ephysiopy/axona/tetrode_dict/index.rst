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


   
















   ..
       !! processed by numpydoc !!

   .. py:method:: __getitem__(key)

      
      x.__getitem__(y) <==> x[y]
















      ..
          !! processed by numpydoc !!


   .. py:method:: get_spike_samples(tetrode, cluster)

      
      Returns spike times in seconds for given cluster from given
      tetrode
















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


   .. py:attribute:: use_volts


   .. py:attribute:: valid_keys


