ephysiopy.openephys2py.KiloSort
===============================

.. py:module:: ephysiopy.openephys2py.KiloSort


Classes
-------

.. autoapisummary::

   ephysiopy.openephys2py.KiloSort.KiloSortSession


Functions
---------

.. autoapisummary::

   ephysiopy.openephys2py.KiloSort.fileExists


Module Contents
---------------

.. py:class:: KiloSortSession(fname_root)

   Bases: :py:obj:`object`


   
   Loads and processes data from a Kilosort session.

   A kilosort session results in a load of .npy files, a .csv or .tsv file.
   The .npy files contain things like spike times, cluster indices and so on.
   Importantly the .csv (or .tsv) file contains the cluster identities of
   the SAVED part of the phy template-gui (ie when you click "Save" from the
   Clustering menu): this file consists of a header ('cluster_id' and 'group')
   where 'cluster_id' is obvious (relates to identity in spk_clusters.npy),
   the 'group' is a string that contains things like 'noise' or 'unsorted' or
   whatever as the phy user can define their own labels.















   ..
       !! processed by numpydoc !!

   .. py:method:: apply_mask(mask, **kwargs)

      
      Apply a mask to the data.

      :param mask: A tuple (start, end) in seconds specifying the mask range.
      :type mask: tuple

      .. rubric:: Notes

      The times inside the bounds are masked, i.e., the mask is set to True.
      The mask can be a list of tuples, in which case the mask is applied
      for each tuple in the list. The mask can be an empty tuple, in which
      case the mask is removed.















      ..
          !! processed by numpydoc !!


   .. py:method:: get_cluster_spike_times(cluster)

      
      Returns the spike times for a given cluster in samples.

      :param cluster: The cluster ID.
      :type cluster: int

      :returns: The spike times for the specified cluster.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: load()

      
      Load all the relevant files

      There is a distinction between clusters assigned during the automatic
      spike sorting process (here KiloSort2) and the manually curated
      distillation of the automatic process conducted by the user with
      a program such as phy.

      * The file cluster_KSLabel.tsv is output from KiloSort.
          All this information is also contained in the cluster_info.tsv
          file! Not sure about the .csv version (from original KiloSort?)
      * The files cluster_group.tsv or cluster_groups.csv contain
          "group labels" from phy ('good', 'MUA', 'noise' etc).
          One of these (cluster_groups.csv or cluster_group.tsv)
          is from kilosort and the other from kilosort2
          I think these are only amended to once a phy session has been
          run / saved...















      ..
          !! processed by numpydoc !!


   .. py:method:: removeKSNoiseClusters()

      
      Removes "noise" and "mua" clusters from the kilosort labelled stuff
















      ..
          !! processed by numpydoc !!


   .. py:method:: removeNoiseClusters()

      
      Removes clusters with labels 'noise' and 'mua' in self.group
















      ..
          !! processed by numpydoc !!


   .. py:attribute:: amplitudes
      :value: None



   .. py:attribute:: cluster_id
      :value: None



   .. py:attribute:: contamPct
      :value: None



   .. py:attribute:: fname_root


   .. py:attribute:: good_clusters
      :value: []



   .. py:attribute:: mua_clusters
      :value: []



   .. py:attribute:: spike_times
      :value: None



   .. py:attribute:: spk_clusters
      :value: None



.. py:function:: fileExists(pname, fname)

