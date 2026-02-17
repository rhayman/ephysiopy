ephysiopy.phase_precession.phase_precession
===========================================

.. py:module:: ephysiopy.phase_precession.phase_precession


Attributes
----------

.. autoapisummary::

   ephysiopy.phase_precession.phase_precession.subaxis_title_fontsize


Classes
-------

.. autoapisummary::

   ephysiopy.phase_precession.phase_precession.phasePrecessionND


Module Contents
---------------

.. py:class:: phasePrecessionND(T, cluster, channel, pp_config = phase_precession_config, regressors=None, **kwargs)

   Bases: :py:obj:`object`


   
   Performs phase precession analysis for single unit data

   Mostly a rip-off of code written by Ali Jeewajee for his paper on
   2D phase precession in place and grid cells [R1fd823aaaf80-1]_

   .. [R1fd823aaaf80-1] Jeewajee A, Barry C, Douchamps V, Manson D, Lever C, Burgess N.
       Theta phase precession of grid and place cell firing in open
       environments.
       Philos Trans R Soc Lond B Biol Sci. 2013 Dec 23;369(1635):20120532.
       doi: 10.1098/rstb.2012.0532.

   :param T: The trial object holding position, LFP, spiking and ratemap stuff
   :type T: AxonaTrial (or OpenEphysBase eventually)
   :param cluster: the cluster to examine
   :type cluster: int
   :param channel: The channel the cluster was recorded on
   :type channel: int
   :param pp_config: Contains parameters for running the analysis.
                     See phase_precession_config dict in ephysiopy.common.eegcalcs
   :type pp_config: dict
   :param regressors: A list of the regressors to use in the analysis
   :type regressors: list

   .. attribute:: orig_xy

      The original position data

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

   .. py:method:: do_correlation(phase_regressors, **kwargs)

      
      Do the regression(s) for each regressor in the phase_regressors dict,
      optionally plotting the results of the regression

      :param phase_regressors: Dictionary with keys as field label (1,2 etc), each key contains a
                               dictionary with keys 'phase' and optional nummbers of regressors
      :type phase_regressors: dict
      :param plot: Whether to plot the regression results
      :type plot: bool

      .. rubric:: Notes

      This collapses across fields and does the regression for all
      phase and regressor values















      ..
          !! processed by numpydoc !!


   .. py:method:: do_regression(**kwargs)

      
      Wrapper function for doing the actual regression which has multiple
      stages.

      Specifically here we partition fields into sub-fields, get a bunch of
      information about the position, spiking and theta data and then
      do the actual regression.

      **kwargs
          do_plot : bool
              whether to plot the results of the regression(s)
          ax : matplotlib.mat
              The axes to plot into















      ..
          !! processed by numpydoc !!


   .. py:method:: get_phase_reg_per_field(fp, **kwargs)

      
      Extracts the phase and all regressors for all runs through each
      field separately

      :param fp: A list of FieldProps instances
      :type fp: list

      :returns: two-level dictionary holding regression results per field
                first level keys are field number
                second level are the regressors (current_dir etc)
                items in the second dict are the regression results
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: get_pos_props(binned_data = None, var_type = VariableToBin.XY, **kwargs)

      
      Uses the output of fancy_partition and returns vectors the same
      length as pos.

      :param binned_data - BinnedData: optional BinnedData instance. Will be calculated here
                                       if not given
      :param var_type - VariableToBin: defines if we are dealing with 1- or 2D data essentially
      :param \*\*kwargs - keywords:
                                    valid kwargs:
                                        field_threshold - see fancy_partition()
                                        field_threshold_percent - see fancy_partition()
                                        area_threshold - see fancy_partition()

      :returns: A list of FieldProps instances
                (see ephysiopy.common.fieldcalcs.FieldProps)
      :rtype: list of FieldProps















      ..
          !! processed by numpydoc !!


   .. py:method:: get_theta_props(field_props)

      
      Processes the LFP data and inserts into each run within each field
      a segment of LFP data that has had its phase and amplitude extracted
      as well as some other data

      :param field_props: A list of FieldProps instances
      :type field_props: list[FieldProps]

      :returns: The amended list with LFP data added to each run for each field
      :rtype: list of FieldProps















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_regressor(regressor, vals, pha, result, ax=None)

      
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


   .. py:attribute:: _binning_var


   .. py:attribute:: _regressors
      :value: None



   .. py:attribute:: alpha
      :value: 0.05



   .. py:property:: binning_var


   .. py:attribute:: channel


   .. py:attribute:: cluster


   .. py:attribute:: conf
      :value: True



   .. py:attribute:: eeg


   .. py:attribute:: filteredEEG


   .. py:attribute:: hyp
      :value: 0



   .. py:attribute:: lfp_fs


   .. py:attribute:: max_theta
      :value: 12



   .. py:attribute:: min_theta
      :value: 6



   .. py:attribute:: nshuffles
      :value: 1000



   .. py:attribute:: phase


   .. py:attribute:: phaseAdj


   .. py:property:: regressors


   .. py:property:: spike_eeg_idx


   .. py:attribute:: spike_times_in_pos_samples


   .. py:attribute:: spike_ts


   .. py:attribute:: spk_weights


   .. py:attribute:: trial


.. py:data:: subaxis_title_fontsize
   :value: 10


