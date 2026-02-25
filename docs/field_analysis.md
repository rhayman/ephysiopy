# Firing field analysis

You can extract various properties of the firing field of a cluster using the 
get_field_properties method the TrialInterface class. This method returns a FieldProps object which contains various properties of the firing field of a cluster.

::: ephysiopy.io.recording.TrialInterface.get_field_properties

Plot each FieldProps item to get an idea of how well the segmentation worked etc:

::: ephysiopy.common.fieldcalcs.plot_field_props

The underlying objects are described here:

::: ephysiopy.common.fieldproperties.fieldprops

::: ephysiopy.common.fieldproperties.FieldProps

::: ephysiopy.common.fieldproperties.RunProps

::: ephysiopy.common.fieldproperties.LFPSegment

These methods are available to operate on the output of fieldprops (or similarly
the [get_field_properties()](./io.md#ephysiopy.io.recording.TrialInterface.get_field_properties) method of the TrialInterface class).

::: ephysiopy.common.fieldcalcs.sort_fields_by_attr

::: ephysiopy.common.fieldcalcs.get_all_phase

::: ephysiopy.common.fieldcalcs.get_run_times

::: ephysiopy.common.fieldcalcs.get_run

::: ephysiopy.common.fieldcalcs.filter_runs

::: ephysiopy.common.fieldcalcs.filter_for_speed

::: ephysiopy.common.fieldcalcs.infill_ratemap

::: ephysiopy.common.fieldcalcs.get_peak_coords

::: ephysiopy.common.fieldcalcs.simple_partition

::: ephysiopy.common.fieldcalcs.fancy_partition

::: ephysiopy.common.fieldcalcs.get_mean_resultant

::: ephysiopy.common.fieldcalcs.get_mean_resultant_length

::: ephysiopy.common.fieldcalcs.get_mean_resultant_angle

::: ephysiopy.common.fieldcalcs.border_score

::: ephysiopy.common.fieldcalcs.plot_field_props

::: ephysiopy.common.fieldcalcs.kl_spatial_sparsity

::: ephysiopy.common.fieldcalcs.spatial_sparsity

::: ephysiopy.common.fieldcalcs.kldiv_dir

::: ephysiopy.common.fieldcalcs.kldiv

::: ephysiopy.common.fieldcalcs.skaggs_info

::: ephysiopy.common.fieldcalcs.grid_field_props

::: ephysiopy.common.fieldcalcs.gridness

::: ephysiopy.common.fieldcalcs.get_basic_gridscore

::: ephysiopy.common.fieldcalcs.get_expanding_circle_gridscore

::: ephysiopy.common.fieldcalcs.get_deformed_sac_gridscore

::: ephysiopy.common.fieldcalcs.get_thigmotaxis_score






