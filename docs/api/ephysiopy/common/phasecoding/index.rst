ephysiopy.common.phasecoding
============================

.. py:module:: ephysiopy.common.phasecoding


Attributes
----------

.. autoapisummary::

   ephysiopy.common.phasecoding.all_regressors
   ephysiopy.common.phasecoding.cbar_fontsize
   ephysiopy.common.phasecoding.cbar_tick_fontsize
   ephysiopy.common.phasecoding.jet_cmap
   ephysiopy.common.phasecoding.phase_precession_config
   ephysiopy.common.phasecoding.subaxis_title_fontsize


Classes
-------

.. autoapisummary::

   ephysiopy.common.phasecoding.phasePrecession2D


Functions
---------

.. autoapisummary::

   ephysiopy.common.phasecoding._stripAx
   ephysiopy.common.phasecoding.ccc
   ephysiopy.common.phasecoding.ccc_jack
   ephysiopy.common.phasecoding.circCircCorrTLinear
   ephysiopy.common.phasecoding.circRegress
   ephysiopy.common.phasecoding.filter_runs
   ephysiopy.common.phasecoding.getPhaseOfMinSpiking
   ephysiopy.common.phasecoding.plot_field_props
   ephysiopy.common.phasecoding.plot_lfp_segment
   ephysiopy.common.phasecoding.plot_spikes_in_runs_per_field
   ephysiopy.common.phasecoding.shuffledPVal


Module Contents
---------------

.. py:class:: phasePrecession2D(lfp_sig, lfp_fs, xy, spike_ts, pos_ts, pp_config = phase_precession_config, regressors=None)

   Bases: :py:obj:`object`


   
   Performs phase precession analysis for single unit data

   Mostly a total rip-off of code written by Ali Jeewajee for his paper on
   2D phase precession in place and grid cells [R0e7f1ac7e825-1]_

   .. [R0e7f1ac7e825-1] Jeewajee A, Barry C, Douchamps V, Manson D, Lever C, Burgess N.
       Theta phase precession of grid and place cell firing in open
       environments.
       Philos Trans R Soc Lond B Biol Sci. 2013 Dec 23;369(1635):20120532.
       doi: 10.1098/rstb.2012.0532.

   :param lfp_sig: The LFP signal against which cells might precess...
   :type lfp_sig: np.array
   :param lfp_fs: The sampling frequency of the LFP signal
   :type lfp_fs: int
   :param xy: The position data as 2 x num_position_samples
   :type xy: np.array
   :param spike_ts: The times in samples at which the cell fired
   :type spike_ts: np.array
   :param pos_ts: The times in samples at which position was captured
   :type pos_ts: np.array
   :param pp_config: Contains parameters for running the analysis.
                     See phase_precession_config dict in ephysiopy.common.eegcalcs
   :type pp_config: dict















   ..
       !! processed by numpydoc !!

   .. py:method:: _ppRegress(spkDict, whichSpk='first')


   .. py:method:: getLFPPhaseValsForSpikeTS()


   .. py:method:: getPosProps(labels)

      
      Uses the output of partitionFields and returns vectors the same
      length as pos.

      :param tetrode: The tetrode / cluster to examine
      :type tetrode: int
      :param cluster: The tetrode / cluster to examine
      :type cluster: int
      :param peaksXY: The x-y coords of the peaks in the ratemap
      :type peaksXY: array_like
      :param laserEvents: The position indices of on events
      :type laserEvents: array_like
      :param (laser on):

      :returns: Contains a whole bunch of information
                for the whole trial and also on a run-by-run basis (run_dict).
                See the end of this function for all the key / value pairs.
      :rtype: pos_dict, run_dict (dict)















      ..
          !! processed by numpydoc !!


   .. py:method:: getSpikePosIndices(spk_times)


   .. py:method:: getSpikeProps(field_props)


   .. py:method:: getThetaProps(field_props)

      
      Processes the LFP data and inserts into each run within each field
      a segment of LFP data that has had its phase and amplitude extracted
      as well as some other metadata
















      ..
          !! processed by numpydoc !!


   .. py:method:: get_regressor(key)


   .. py:method:: get_regressors()


   .. py:method:: performRegression(**kwargs)

      
      Wrapper function for doing the actual regression which has multiple
      stages.

      Specifically here we partition fields into sub-fields, get a bunch of
      information about the position, spiking and theta data and then
      do the actual regression.

      :param tetrode: The tetrode to examine
      :type tetrode: int
      :param cluster: The cluster to examine
      :type cluster: int
      :param laserEvents: The on times for laser events
      :type laserEvents: array_like, optional
      :param if present. Default is None:

      Valid keyword args:
          plot (bool): whether to plot the results of field partitions, the regression(s)
              etc
      .. seealso::

         ephysiopy.common.eegcalcs.phasePrecession.partitionFields()
         ephysiopy.common.eegcalcs.phasePrecession.getPosProps()
         ephysiopy.common.eegcalcs.phasePrecession.getThetaProps()
         ephysiopy.common.eegcalcs.phasePrecession.getSpikeProps()
         ephysiopy.common.eegcalcs.phasePrecession._ppRegress()















      ..
          !! processed by numpydoc !!


   .. py:method:: plotPPRegression(regressorDict, regressor2plot='pos_d_cum', ax=None)


   .. py:method:: plotRegressor(regressor, ax=None)


   .. py:method:: update_config(pp_config)


   .. py:method:: update_position(ppm, cm)


   .. py:method:: update_rate_map()


   .. py:method:: update_regressor_mask(key, indices)


   .. py:method:: update_regressor_values(key, values)


   .. py:method:: update_regressors(reg_keys)

      
      Create a dict to hold the stats values for
      each regressor
      Default regressors are:
          "spk_numWithinRun",
          "pos_exptdRate_cum",
          "pos_instFR",
          "pos_timeInRun",
          "pos_d_cum",
          "pos_d_meanDir",
          "pos_d_currentdir",
          "spk_thetaBatchLabelInRun"

      NB: The regressors have differing sizes of 'values' depending on the
      type of the regressor:
      spk_* - integer values of the spike number within a run or the theta batch
              in a run, so has a length equal to the number of spikes collected
      pos_* - a bincount of some type so equal to the number of position samples
              collected
      eeg_* - only one at present, the instantaneous firing rate binned into the
              number of eeg samples so equal to that in length















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _pos_ts


   .. py:attribute:: alpha
      :value: 0.05



   .. py:attribute:: conf
      :value: True



   .. py:attribute:: eeg


   .. py:attribute:: filteredEEG


   .. py:attribute:: hyp
      :value: 0



   .. py:attribute:: max_theta
      :value: 12



   .. py:attribute:: min_theta
      :value: 6



   .. py:attribute:: orig_xy


   .. py:attribute:: phase


   .. py:attribute:: phaseAdj


   .. py:property:: pos_ts


   .. py:attribute:: regressor
      :value: 1000



   .. py:property:: spike_eeg_idx


   .. py:property:: spike_pos_idx


   .. py:attribute:: spike_times_in_pos_samples


   .. py:attribute:: spike_ts


   .. py:attribute:: spk_weights


.. py:function:: _stripAx(ax)

.. py:function:: ccc(t, p)

   
   Calculates correlation between two random circular variables
















   ..
       !! processed by numpydoc !!

.. py:function:: ccc_jack(t, p)

   
   Function used to calculate jackknife estimates of correlation
















   ..
       !! processed by numpydoc !!

.. py:function:: circCircCorrTLinear(theta, phi, regressor=1000, alpha=0.05, hyp=0, conf=True)

   
   An almost direct copy from AJs Matlab fcn to perform correlation
   between 2 circular random variables.

   Returns the correlation value (rho), p-value, bootstrapped correlation
   values, shuffled p values and correlation values.

   :param theta: mx1 array containing circular data (radians)
                 whose correlation is to be measured
   :type theta: array_like
   :param phi: mx1 array containing circular data (radians)
               whose correlation is to be measured
   :type phi: array_like
   :param regressor: number of permutations to use to calculate p-value
                     from randomisation and bootstrap estimation of confidence
                     intervals.
                     Leave empty to calculate p-value analytically (NB confidence
                     intervals will not be calculated). Default is 1000.
   :type regressor: int, optional
   :param alpha: hypothesis test level e.g. 0.05, 0.01 etc.
                 Default is 0.05.
   :type alpha: float, optional
   :param hyp: hypothesis to test; -1/ 0 / 1 (-ve correlated /
               correlated in either direction / positively correlated).
               Default is 0.
   :type hyp: int, optional
   :param conf: True or False to calculate confidence intervals
                via jackknife or bootstrap. Default is True.
   :type conf: bool, optional

   .. rubric:: References

   Fisher (1993), Statistical Analysis of Circular Data,
       Cambridge University Press, ISBN: 0 521 56890 0















   ..
       !! processed by numpydoc !!

.. py:function:: circRegress(x, t)

   
   Finds approximation to circular-linear regression for phase precession.

   :param x: n-by-1 list of in-field positions (linear variable)
   :type x: list
   :param t: n-by-1 list of phases, in degrees (converted to radians)
   :type t: list

   .. note:: Neither x nor t can contain NaNs, must be paired (of equal length).















   ..
       !! processed by numpydoc !!

.. py:function:: filter_runs(field_props, min_speed, min_duration, min_spikes = 0)

.. py:function:: getPhaseOfMinSpiking(spkPhase)

.. py:function:: plot_field_props(field_props)

.. py:function:: plot_lfp_segment(field, lfp_sample_rate = 250)

.. py:function:: plot_spikes_in_runs_per_field(field_label, run_starts, run_ends, spikes_in_time, ttls_in_time = None, **kwargs)

   
   Debug plotting to show spikes per run per field found in the ratemap
   as a raster plot

   Args:
   field_label (np.ndarray): The field labels for each position bin
       a vector
   run_start_stop_idx (np.ndarray): The start and stop indices of each run
       has shape (n_runs, 2)
   spikes_in_time (np.ndarray): The number of spikes in each position bin
       a vector

   kwargs:
   separate_plots (bool): If True then each field will be plotted in a
   separate figure

   single_axes (bool): If True will plot all the runs/ spikes in a single
   axis with fields delimited by horizontal lines

   Returns:
   fig, axes (tuple): The figure and axes objects















   ..
       !! processed by numpydoc !!

.. py:function:: shuffledPVal(theta, phi, rho, regressor, hyp)

   
   Calculates shuffled p-values for correlation
















   ..
       !! processed by numpydoc !!

.. py:data:: all_regressors
   :value: ['spk_numWithinRun', 'pos_exptdRate_cum', 'pos_instFR', 'pos_timeInRun', 'pos_d_cum',...


.. py:data:: cbar_fontsize
   :value: 8


.. py:data:: cbar_tick_fontsize
   :value: 6


.. py:data:: jet_cmap

.. py:data:: phase_precession_config

.. py:data:: subaxis_title_fontsize
   :value: 10


