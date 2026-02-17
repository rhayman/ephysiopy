ephysiopy.phase_precession.plotting
===================================

.. py:module:: ephysiopy.phase_precession.plotting


Functions
---------

.. autoapisummary::

   ephysiopy.phase_precession.plotting._stripAx
   ephysiopy.phase_precession.plotting.add_colorwheel_to_fig
   ephysiopy.phase_precession.plotting.add_fields_to_line_graph
   ephysiopy.phase_precession.plotting.plot_field_and_runs
   ephysiopy.phase_precession.plotting.plot_field_props
   ephysiopy.phase_precession.plotting.plot_lfp_and_spikes_per_run
   ephysiopy.phase_precession.plotting.plot_lfp_run
   ephysiopy.phase_precession.plotting.plot_lfp_segment
   ephysiopy.phase_precession.plotting.plot_phase_precession
   ephysiopy.phase_precession.plotting.plot_phase_v_position
   ephysiopy.phase_precession.plotting.plot_runs_and_precession
   ephysiopy.phase_precession.plotting.plot_spikes_in_runs_per_field
   ephysiopy.phase_precession.plotting.ratemap_line_graph


Module Contents
---------------

.. py:function:: _stripAx(ax)

.. py:function:: add_colorwheel_to_fig(ax)

   
   Add a colorwheel to the given axis.
















   ..
       !! processed by numpydoc !!

.. py:function:: add_fields_to_line_graph(f_props, ax=None)

   
   Add field boundaries to a line graph of the rate map.

   :param f_props: List of FieldProps containing field information.
   :type f_props: list[FieldProps]

   :returns: The axes with the field boundaries added.
   :rtype: matplotlib.axes.Axes















   ..
       !! processed by numpydoc !!

.. py:function:: plot_field_and_runs(trial, field_props)

   
   Plot runs versus time where the colour of the line indicates
   directional heading. The field limits are also plotted and spikes are
   overlaid on the runs. Boxes delineate the runs that have been identified
   in field_props
















   ..
       !! processed by numpydoc !!

.. py:function:: plot_field_props(field_props)

   
   Plots the fields in the list of FieldProps

   :param list of FieldProps:















   ..
       !! processed by numpydoc !!

.. py:function:: plot_lfp_and_spikes_per_run(f_props)

   
   Plot the LFP and spikes per run.

   :param f_props: List of FieldProps containing field information.
   :type f_props: list[FieldProps]

   :returns: The axes with the plot.
   :rtype: matplotlib.axes.Axes















   ..
       !! processed by numpydoc !!

.. py:function:: plot_lfp_run(run, cycle_labels = None, lfp_sample_rate = 250, **kwargs)

   
   Plot the lfp segment for a single run through a field including
   the spikes emitted by the cell.

   .. rubric:: Notes

   There are very small inaccuracies here due to the way the timebase
   is being created from the slice belonging to the run and the way
   the indexing is being done by repeating the indices (repeat_ind)
   of the spike counts binned wrt the LFP sample rate. This shouldn;t
   matter for purposes of plotting - it's only when you zoom in a lot
   that you can see the diffferences between this and the actual spike
   times etc (if you can be arsed to plot them)















   ..
       !! processed by numpydoc !!

.. py:function:: plot_lfp_segment(field, lfp_sample_rate = 250)

   
   Plot the lfp segments for a series of runs through a field including
   the spikes emitted by the cell.
















   ..
       !! processed by numpydoc !!

.. py:function:: plot_phase_precession(phase, normalised_position, slope, intercept, ax=None, **kwargs)

   
   Plot the phase precession of spikes in a field.

   :param field_phase_pos: Dictionary containing the phase and normalised position for each field.
   :type field_phase_pos: dict[str, dict[np.ndarray, np.ndarray]]
   :param ax:
   :type ax: matplotlib.axes.Axes, optional















   ..
       !! processed by numpydoc !!

.. py:function:: plot_phase_v_position(field_props, ax=None, **kwargs)

   
   Plot the phase of the LFP signal at each position in the field.

   :param field_props: List of FieldProps objects containing run and LFP data.
   :type field_props: list[FieldProps]
   :param ax: Axes to plot on. If None, a new figure and axes will be created.
   :type ax: matplotlib.axes.Axes, optional
   :param \*\*kwargs: Additional keyword arguments for plotting.
   :type \*\*kwargs: dict

   :returns: The axes with the plot.
   :rtype: matplotlib.axes.Axes















   ..
       !! processed by numpydoc !!

.. py:function:: plot_runs_and_precession(trial, cluster, channel, field_props)

   
   Plot runs versus time where the colour of the line indicates
   directional heading. The field limits are also plotted and spikes are
   overlaid on the runs. Boxes delineate the runs that have been identified
   in field_props. Also plots phase precession for each field.
















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

.. py:function:: ratemap_line_graph(binned_data, ax=None, **kwargs)

   
   Plot a line graph of the rate map.

   :param binned_data: The binned data containing the rate map.
   :type binned_data: BinnedData
   :param \*\*kwargs: Additional keyword arguments for plotting.
   :type \*\*kwargs: dict

   :returns: The axes with the plot.
   :rtype: matplotlib.axes.Axes















   ..
       !! processed by numpydoc !!

