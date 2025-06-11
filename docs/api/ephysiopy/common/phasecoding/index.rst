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

   :param lfp_sig: The LFP signal
   :type lfp_sig: np.ndarray
   :param lfp_fs: The sampling frequency of the LFP signal
   :type lfp_fs: int
   :param xy: The position data as 2 x num_position_samples
   :type xy: np.ndarray
   :param spike_ts: The times in samples at which the cell fired
   :type spike_ts: np.ndarray
   :param pos_ts: The times in samples at which position was captured
   :type pos_ts: np.ndarray
   :param pp_config: Contains parameters for running the analysis.
                     See phase_precession_config dict in ephysiopy.common.eegcalcs
   :type pp_config: dict
   :param regressors: A list of the regressors to use in the analysis
   :type regressors: list

   .. attribute:: orig_xy

      The original position data

      :type: np.ndarray

   .. attribute:: pos_ts

      The position timestamps

      :type: np.ndarray

   .. attribute:: spike_ts

      The spike timestamps

      :type: np.ndarray

   .. attribute:: regressors

      A dictionary containing the regressors and their values

      :type: dict

   .. attribute:: alpha

      The alpha value for hypothesis testing

      :type: float

   .. attribute:: hyp

      The hypothesis to test

      :type: int

   .. attribute:: conf

      Whether to calculate confidence intervals

      :type: bool

   .. attribute:: eeg

      The EEG signal

      :type: np.ndarray

   .. attribute:: min_theta

      The minimum theta frequency

      :type: int

   .. attribute:: max_theta

      The maximum theta frequency

      :type: int

   .. attribute:: filteredEEG

      The filtered EEG signal

      :type: np.ndarray

   .. attribute:: phase

      The phase of the EEG signal

      :type: np.ndarray

   .. attribute:: phaseAdj

      The adjusted phase of the EEG signal as a masked array

      :type: np.ma.MaskedArray

   .. attribute:: spike_times_in_pos_samples

      The spike times in position samples (vector with length = npos)

      :type: np.ndarray

   .. attribute:: spk_weights

      The spike weights (vector with length = npos)

      :type: np.ndarray















   ..
       !! processed by numpydoc !!

   .. py:method:: _ppRegress(spkDict, whichSpk='first')

      
      Perform the regression analysis on the spike data

      :param spkDict: A dictionary containing the spike properties
      :type spkDict: dict
      :param whichSpk: Which spike(s) in a cycle to use in the regression analysis
      :type whichSpk: str

      :returns: A list of the updated regressors
      :rtype: list















      ..
          !! processed by numpydoc !!


   .. py:method:: getPosProps(labels)

      
      Uses the output of partitionFields and returns vectors the same
      length as pos.

      :param labels: The labels of the fields
      :type labels: np.ndarray

      :returns: A list of FieldProps instances (see ephysiopy.common.fieldcalcs.FieldProps)
      :rtype: list of FieldProps















      ..
          !! processed by numpydoc !!


   .. py:method:: getSpikePosIndices(spk_times)

      
      Get the indices of the spikes in the position data
















      ..
          !! processed by numpydoc !!


   .. py:method:: getSpikeProps(field_props)

      
      Extracts the relevant spike properties from the field_props

      :param field_props: A list of FieldProps instances
      :type field_props: list

      :returns: A dictionary containing the spike properties
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: getThetaProps(field_props)

      
      Processes the LFP data and inserts into each run within each field
      a segment of LFP data that has had its phase and amplitude extracted
      as well as some other data

      :param field_props: A list of FieldProps instances
      :type field_props: list[FieldProps]















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

      **kwargs
          do_plot : bool
          whether to plot the results of field partitions, the regression(s)

      .. seealso:: :obj:`ephysiopy.common.eegcalcs.phasePrecession.partitionFields`, :obj:`ephysiopy.common.eegcalcs.phasePrecession.getPosProps`, :obj:`ephysiopy.common.eegcalcs.phasePrecession.getThetaProps`, :obj:`ephysiopy.common.eegcalcs.phasePrecession.getSpikeProps`, :obj:`ephysiopy.common.eegcalcs.phasePrecession._ppRegress`















      ..
          !! processed by numpydoc !!


   .. py:method:: plotRegressor(regressor, ax=None)

      
      Plot the regressor against the phase

      :param regressor: The regressor to plot
      :type regressor: str
      :param ax: The axes to plot on
      :type ax: matplotlib.axes.Axes

      :returns: **ax** -- The axes with the plot
      :rtype: matplotlib.axes.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: update_config(pp_config)

      
      Update the relevant pp_config values
















      ..
          !! processed by numpydoc !!


   .. py:method:: update_position(ppm, cm)

      
      Update the position data based on ppm and cm values
















      ..
          !! processed by numpydoc !!


   .. py:method:: update_rate_map()

      
      Create the ratemap from the position data
















      ..
          !! processed by numpydoc !!


   .. py:method:: update_regressor_mask(key, indices)

      
      Mask entries in the 'values' and 'pha' arrays of the relevant regressor
















      ..
          !! processed by numpydoc !!


   .. py:method:: update_regressor_values(key, values)

      
      Check whether values is a masked array and if not make it one
















      ..
          !! processed by numpydoc !!


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


   .. py:attribute:: regressors
      :value: 1000



   .. py:property:: spike_eeg_idx


   .. py:property:: spike_pos_idx


   .. py:attribute:: spike_times_in_pos_samples


   .. py:attribute:: spike_ts


   .. py:attribute:: spk_weights


.. py:function:: _stripAx(ax)

.. py:function:: ccc(t, p)

   
   Calculates correlation between two random circular variables

   :param t: The first variable
   :type t: np.ndarray
   :param p: The second variable
   :type p: np.ndarray

   :returns: The correlation between the two variables
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: ccc_jack(t, p)

   
   Function used to calculate jackknife estimates of correlation
   between two circular random variables

   :param t: The first variable
   :type t: np.ndarray
   :param p: The second variable
   :type p: np.ndarray

   :returns: The jackknife estimates of the correlation between the two variables
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: circCircCorrTLinear(theta, phi, regressor=1000, alpha=0.05, hyp=0, conf=True)

   
   An almost direct copy from AJs Matlab fcn to perform correlation
   between 2 circular random variables.

   Returns the correlation value (rho), p-value, bootstrapped correlation
   values, shuffled p values and correlation values.

   :param theta: The two circular variables to correlate (in radians)
   :type theta: np.ndarray
   :param phi: The two circular variables to correlate (in radians)
   :type phi: np.ndarray
   :param regressor: number of permutations to use to calculate p-value from randomisation and
                     bootstrap estimation of confidence intervals.
                     Leave empty to calculate p-value analytically (NB confidence
                     intervals will not be calculated).
   :type regressor: int, default=1000
   :param alpha: hypothesis test level e.g. 0.05, 0.01 etc.
   :type alpha: float, default=0.05
   :param hyp: hypothesis to test; -1/ 0 / 1 (-ve correlated / correlated in either direction / positively correlated).
   :type hyp: int, default=0
   :param conf: True or False to calculate confidence intervals via jackknife or bootstrap.
   :type conf: bool, default=True

   .. rubric:: References

   Fisher (1993), Statistical Analysis of Circular Data,
       Cambridge University Press, ISBN: 0 521 56890 0















   ..
       !! processed by numpydoc !!

.. py:function:: circRegress(x, t)

   
   Finds approximation to circular-linear regression for phase precession.

   :param x: The linear variable and the phase variable (in radians)
   :type x: np.ndarray
   :param t: The linear variable and the phase variable (in radians)
   :type t: np.ndarray

   .. rubric:: Notes

   Neither x nor t can contain NaNs, must be paired (of equal length).















   ..
       !! processed by numpydoc !!

.. py:function:: filter_runs(field_props, min_speed, min_duration, min_spikes = 0)

   
   Filter out runs that are too short, too slow or have too few spikes

   :param field_props:
   :type field_props: list of FieldProps
   :param min_speed: the minimum speed for a run
   :type min_speed: float, int
   :param min_duration: the minimum duration for a run
   :type min_duration: int, float
   :param min_spikes: the minimum number of spikes for a run
   :type min_spikes: int, default=0

   :rtype: list of FieldProps

   .. rubric:: Notes

   this modifies the input list















   ..
       !! processed by numpydoc !!

.. py:function:: getPhaseOfMinSpiking(spkPhase)

   
   Returns the phase at which the minimum number of spikes are fired

   :param spkPhase: The phase of the spikes
   :type spkPhase: np.ndarray

   :returns: The phase at which the minimum number of spikes are fired
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: plot_field_props(field_props)

   
   Plots the fields in the list of FieldProps

   :param list of FieldProps:















   ..
       !! processed by numpydoc !!

.. py:function:: plot_lfp_segment(field, lfp_sample_rate = 250)

   
   Plot the lfp segments for a series of runs through a field including
   the spikes emitted by the cell.
















   ..
       !! processed by numpydoc !!

.. py:function:: plot_spikes_in_runs_per_field(field_label, run_starts, run_ends, spikes_in_time, ttls_in_time = None, **kwargs)

   
   Debug plotting to show spikes per run per field found in the ratemap
   as a raster plot

   :param field_label:
   :type field_label: np.ndarray
   :param The field labels for each position bin a vector:
   :param run_starts: The start and stop indices of each run (vectors)
   :type run_starts: np.ndarray
   :param runs_ends: The start and stop indices of each run (vectors)
   :type runs_ends: np.ndarray
   :param spikes_in_time: The number of spikes in each position bin (vector)
   :type spikes_in_time: np.ndarray
   :param ttls_in_time: TTL occurences in time (vector)
   :type ttls_in_time: np.ndarray
   :param \*\*kwargs:
                      separate_plots : bool
                          If True then each field will be plotted in a separate figure
                      single_axes : bool
                          If True will plot all the runs/ spikes in a single axis with fields delimited by horizontal lines

   :returns: **fig, axes** -- The figure and axes objects
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: shuffledPVal(theta, phi, rho, regressor, hyp)

   
   Calculates shuffled p-values for correlation

   :param theta: The two circular variables to correlate (in radians)
   :type theta: np.ndarray
   :param phi: The two circular variables to correlate (in radians)
   :type phi: np.ndarray

   :returns: The shuffled p-value for the correlation between the two variables
   :rtype: float















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

   
   A list of the regressors that can be used in the phase precession analysis
















   ..
       !! processed by numpydoc !!

.. py:data:: subaxis_title_fontsize
   :value: 10


