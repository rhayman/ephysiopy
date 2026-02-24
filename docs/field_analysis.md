# Firing field analysis

You can extract various properties of the firing field of a cluster using the 
get_field_properties method the TrialInterface class. This method returns a FieldProps object which contains various properties of the firing field of a cluster.

::: ephysiopy.io.recording.TrialInterface.get_field_properties
    options:
      show_root_heading: true
      show_docstring_examples: true

Plot each FieldProps item to get an idea of how well the segmentation worked etc:

::: ephysiopy.common.fieldcalcs.plot_field_props

The underlying objects are described here:

::: ephysiopy.common.fieldproperties.fieldprops
    options:
      show_root_heading: true
      show_docstring_examples: true

::: ephysiopy.common.fieldproperties.FieldProps
    options:
      show_root_heading: true
      show_docstring_examples: true
      show_docstring_attributes: true
      members:
        - n_spikes
        - normalized_position
        - phase
        - phi
        - pos_phi
        - pos_r
        - projected_direction
        - mean_spiking_var
        - overdispersion
        - runs_expected_spikes
        - smooth_runs
        - spiking_var
        - run_labels
        - cumulative_time
        - cumulative_distance

::: ephysiopy.common.fieldproperties.RunProps
    options:
      show_root_heading: true
      show_docstring_examples: true
      show_docstring_attributes: true
      members:
        - current_direction
        - ndim
        - normed_x
        - phi
        - pos_phi
        - pos_r
        - raw_spike_times
        - rho
        - spike_index
        - xy_dist_to_peak_normed
        - expected_spikes
        - mean_spiking_var
        - overdispersion
        - smooth_xy
        - spiking_var

::: ephysiopy.common.fieldproperties.LFPSegment
    options:
      show_root_heading: true
      show_docstring_examples: true
      show_docstring_attributes: true
      members:
        - raw_spike_times
        - spike_index
        - mean_spiking_var
        - spiking_var
