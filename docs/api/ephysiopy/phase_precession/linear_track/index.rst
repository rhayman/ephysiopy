ephysiopy.phase_precession.linear_track
=======================================

.. py:module:: ephysiopy.phase_precession.linear_track

.. autoapi-nested-parse::

   The main reason for this file is to do some position preprocessing
   for linear track data

   You can either use the x coordinate as the position that gets fed
   into ephysiopy.common.fieldproperties.fieldprops or phi which is
   the euclidean distance along the linear track

   ..
       !! processed by numpydoc !!


Attributes
----------

.. autoapisummary::

   ephysiopy.phase_precession.linear_track.ALPHA
   ephysiopy.phase_precession.linear_track.CONF
   ephysiopy.phase_precession.linear_track.EXCLUDE_SPEEDS
   ephysiopy.phase_precession.linear_track.FIELD_RATE_THRESHOLD
   ephysiopy.phase_precession.linear_track.FIELD_THRESHOLD_PERCENT
   ephysiopy.phase_precession.linear_track.HYPOTHESIS
   ephysiopy.phase_precession.linear_track.MAX_THETA
   ephysiopy.phase_precession.linear_track.MIN_ALLOWED_SPIKE_PHASE
   ephysiopy.phase_precession.linear_track.MIN_FIELD_SIZE_IN_BINS
   ephysiopy.phase_precession.linear_track.MIN_POWER_THRESHOLD
   ephysiopy.phase_precession.linear_track.MIN_RUN_LENGTH
   ephysiopy.phase_precession.linear_track.MIN_SPIKES_PER_RUN
   ephysiopy.phase_precession.linear_track.MIN_THETA
   ephysiopy.phase_precession.linear_track.N_PERMUTATIONS
   ephysiopy.phase_precession.linear_track.POS_SAMPLE_RATE


Functions
---------

.. autoapisummary::

   ephysiopy.phase_precession.linear_track.add_normalised_run_position
   ephysiopy.phase_precession.linear_track.apply_linear_track_filter
   ephysiopy.phase_precession.linear_track.fieldprops_phase_precession
   ephysiopy.phase_precession.linear_track.get_field_props_for_linear_track
   ephysiopy.phase_precession.linear_track.get_run_direction
   ephysiopy.phase_precession.linear_track.plot_linear_runs
   ephysiopy.phase_precession.linear_track.run_phase_analysis


Module Contents
---------------

.. py:function:: add_normalised_run_position(f_props)

   
   Adds the normalised run position to each run through a field in field_props
   where the run x position is normalised with respect to the
   field x position limits and the run direction (east or west)
















   ..
       !! processed by numpydoc !!

.. py:function:: apply_linear_track_filter(T, run_direction=None, var_type=VariableToBin.X, track_end_size=6)

.. py:function:: fieldprops_phase_precession(P, **kwargs)

   
   Run the phase analysis on a linear track trial.

   :param trial (AxonaTrial) - the trial:
   :param cluster (int) - the cluster id:
   :param channel (int) - the channel id:
   :param kwargs (dict) - additional parameters to pass to the:

   :returns: results for that field as the value
   :rtype: dict - a dictionary with the field id as the key and the correlation















   ..
       !! processed by numpydoc !!

.. py:function:: get_field_props_for_linear_track(P, var_type=VariableToBin.X, **kwargs)

   
   Get the field properties for a linear track trial.

   Filters the linear track data based on speed, direction (east
   or west; larger ranges than the usual 90degs are used), and position
   (masks the start and end 12cm of the track)















   ..
       !! processed by numpydoc !!

.. py:function:: get_run_direction(run)

.. py:function:: plot_linear_runs(f_props, var = 'speed', **kwargs)

   
   Plots the runs through the field(s) on a linear track
   as a sort of raster plot with each run as a separate line on
   the y-axis with ticks for each spike occurring on each run.
   For each run the height of
















   ..
       !! processed by numpydoc !!

.. py:function:: run_phase_analysis(trial, cluster, channel, **kwargs)

   
   Run the phase analysis on a linear track trial.

   :param trial (AxonaTrial) - the trial:
   :param cluster (int) - the cluster id:
   :param channel (int) - the channel id:
   :param kwargs (dict) - additional parameters to pass to the:

   :returns: ordered by the field id and the regression
             results for that field
   :rtype: list[RegressionResults] - list of RegressionResults















   ..
       !! processed by numpydoc !!

.. py:data:: ALPHA
   :value: 0.05


.. py:data:: CONF
   :value: True


.. py:data:: EXCLUDE_SPEEDS
   :value: (0, 0.5)


.. py:data:: FIELD_RATE_THRESHOLD
   :value: 0.5


.. py:data:: FIELD_THRESHOLD_PERCENT
   :value: 50


.. py:data:: HYPOTHESIS
   :value: 0


.. py:data:: MAX_THETA
   :value: 10


.. py:data:: MIN_ALLOWED_SPIKE_PHASE

.. py:data:: MIN_FIELD_SIZE_IN_BINS
   :value: 3


.. py:data:: MIN_POWER_THRESHOLD
   :value: 20


.. py:data:: MIN_RUN_LENGTH
   :value: 1


.. py:data:: MIN_SPIKES_PER_RUN
   :value: 3


.. py:data:: MIN_THETA
   :value: 6


.. py:data:: N_PERMUTATIONS
   :value: 1000


.. py:data:: POS_SAMPLE_RATE
   :value: 50


