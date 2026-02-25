# Phase precession

There are basically two different types of phase precession analysis currently
possible but I've attempted to capture both in one class, [phasePrecessionND]().

This design decision is reflected in the way that data is extracted from firing rate maps
and the runs through them in the fieldprops function and the [FieldProps](./field_analysis.md#ephysiopy.common.fieldproperties.FieldProps) and
[RunProps](./field_analysis.md#ephysiopy.common.fieldproperties.fieldprops) classes that function creates

However, it still makes sense to separate out 1D and 2D phase precession analysis to
some degree as they are quite different. That is why there is a separate module for dealing with
linear track data; the functions in the plotting sub-module of the phase_precession module
are concerned with plotting linear track data **not** open-field 2D data.

The 2D phase precession analysis is heavily indebted to the following paper:

  Jeewajee A, Barry C, Douchamps V, Manson D, Lever C, Burgess N.
    Theta phase precession of grid and place cell firing in open
    environments.
    Philos Trans R Soc Lond B Biol Sci. 2013 Dec 23;369(1635):20120532.
    doi: 10.1098/rstb.2012.0532.

As with the analysis of replay data, phase precession analyses are heavily contingent
on how things like firing rate maps are extracted, what the window sizes are for binning up
spike trains, what your inclusion/ exclusion criteria are for things like
good runs/ good firing fields etc. Whilst not infintely flexible, there are many options available
when you run these analyses. Look at the documentation but also the source
code and raise an issue(s) on github if any mistakes are found/ improvements can be made (they
definitely can!)

Many of the settings used in the analysis can be changed by altering the values
in the following dictionaries that are used as input to the phase precession
classes:

::: ephysiopy.phase_precession.config.phase_precession_config

::: ephysiopy.phase_precession.config.all_regressors

::: ephysiopy.phase_precession.phase_precession.phasePrecessionND

## Linear track phase precession

::: ephysiopy.phase_precession.linear_track.run_phase_analysis

::: ephysiopy.phase_precession.linear_track.fieldprops_phase_precession

::: ephysiopy.phase_precession.linear_track.get_field_props_for_linear_track

::: ephysiopy.phase_precession.linear_track.apply_linear_track_filter

::: ephysiopy.phase_precession.linear_track.get_run_direction

::: ephysiopy.phase_precession.linear_track.add_normalised_run_position

::: ephysiopy.phase_precession.linear_track.plot_linear_runs

## Plotting phase precssion results

::: ephysiopy.phase_precession.plotting.plot_phase_precession

::: ephysiopy.phase_precession.plotting.plot_runs_and_precession

::: ephysiopy.phase_precession.plotting.plot_field_and_runs

::: ephysiopy.phase_precession.plotting.plot_phase_v_position

::: ephysiopy.phase_precession.plotting.ratemap_line_graph

::: ephysiopy.phase_precession.plotting.add_fields_to_line_graph

::: ephysiopy.phase_precession.plotting.plot_lfp_and_spikes_per_run

::: ephysiopy.phase_precession.plotting.plot_lfp_segment

::: ephysiopy.phase_precession.plotting.plot_lfp_run

::: ephysiopy.phase_precession.plotting.plot_spikes_in_runs_per_field
