import numpy as np
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential
import matplotlib.pyplot as plt
from ephysiopy.common.utils import (
    fixAngle,
    labelContigNonZeroRuns,
    getLabelStarts,
    getLabelEnds,
    flatten_list,
    VariableToBin,
    TrialFilter,
)
from ephysiopy.io.recording import AxonaTrial
from ephysiopy.common.fieldcalcs import (
    FieldProps,
    LFPSegment,
    RunProps,
    partitionFields,
    fieldprops,
)
from ephysiopy.common.rhythmicity import LFPOscillations
from ephysiopy.common.phasecoding import (
    get_phase_of_min_spiking,
    circRegress,
    circCircCorrTLinear,
)
from ephysiopy.phase_precession.plotting import (
    add_fields_to_line_graph,
    plot_phase_precession,
)

MIN_THETA = 6
MAX_THETA = 10
# defines start/ end of theta cycle
MIN_ALLOWED_SPIKE_PHASE = np.pi
# percentile power below which theta cycles are rejected
MIN_POWER_THRESHOLD = 20  # percentile - total guess at the value
POS_SAMPLE_RATE = 50
# SOME FIELD THRESHOLDS
FIELD_THRESHOLD_PERCENT = 50
FIELD_RATE_THRESHOLD = 0.5
MIN_FIELD_SIZE_IN_BINS = 3
MIN_SPIKES_PER_RUN = 3
# SOME RUN THRESHOLDS
MIN_RUN_LENGTH = 50
EXCLUDE_SPEEDS = (0, 0.5)  # cm/s
# CONSTANTS FOR THE REGRESSION
N_PERMUTATIONS = 1000
ALPHA = 0.05
# The hypothesis to test:
# -1 = negatively correlated
#  0 = correlated in either direction
# +1 = positively correlated
HYPOTHESIS = 0
# Calculate confidence intervals via jackknife (True) or bootstrap (False)
CONF = True


def run_phase_analysis(
    trial: AxonaTrial, cluster: int, channel: int, **kwargs
) -> list[FieldProps]:
    """
    Run the phase analysis on a linear track trial.
    Returns the field properties with LFP data added.
    """
    run_direction = kwargs.get("run_direction", "e")
    # get the field properties for the linear track
    f_props = get_field_props_for_linear_track(
        trial, cluster, channel, direction=run_direction, **kwargs
    )

    # merge the LFP data with the field properties
    f_props = merge_field_and_lfp(f_props, trial, **kwargs)

    # A given field might be missing all runs now...
    f_props_new = []
    for f in f_props:
        if len(f.runs) > 0:
            f_props_new.append(f)

    # add the normalised run position to each run
    f_props = add_normalised_run_position(f_props_new)

    # create a figure window for the plots now we now how many fields we have
    n_fields = len(f_props)
    if n_fields == 0:
        print("No fields found in this trial.")
        return f_props
    fig, axs = plt.subplots(
        n_fields + 1, 1, figsize=(10, 3 * (n_fields + 1)), layout="constrained"
    )
    axs = flatten_list(axs)
    # plot the rate map
    ax = add_fields_to_line_graph(f_props, ax=axs[0])
    ax.set_xlabel("")

    phase_pos = get_phase_precession_per_field(f_props)

    for i, field in enumerate(phase_pos.keys()):

        fp = f_props[i]
        xmin = fp.binned_data.bin_edges[0][fp.slice[0].start]
        xmax = fp.binned_data.bin_edges[0][fp.slice[0].stop]
        xmid = (xmin + xmax) / 2
        # annotate the ratemap plot with the field id
        axs[0].annotate(
            str(fp.label),
            xy=(xmid, 1.1),
            xycoords=("data", "axes fraction"),
            xytext=(xmid, 1.1),
            color="black",
            fontsize=12,
        )

        regressor = phase_pos[field]["normalised_position"]
        phase = phase_pos[field]["phase"]

        slope, intercept = circRegress(regressor, phase)
        mnx = np.mean(regressor)
        regressor -= mnx
        mxx = np.max(np.abs(regressor)) + np.spacing(1)
        regressor /= mxx
        theta = np.mod(np.abs(slope) * regressor, 2 * np.pi)
        corr_result = circCircCorrTLinear(
            theta, phase, N_PERMUTATIONS, ALPHA, HYPOTHESIS, CONF
        )

        a = plot_phase_precession(
            phase_pos[field]["phase"],
            phase_pos[field]["normalised_position"],
            slope,
            intercept,
            ax=axs[i + 1],
        )
        a.text(1.01, 0.5, corr_result, transform=a.transAxes, fontsize=8)

    if kwargs.get("save_name", None):
        plt.savefig(kwargs.get("save_name"))
        plt.close("all")

    return f_props


def get_run_direction(run: RunProps):
    df = run.xy[0][0] - run.xy[0][-1]
    if df > 0:
        return "w"
    return "e"


def add_normalised_run_position(f_props: list[FieldProps]) -> list[FieldProps]:
    """
    Adds the normalised run position to each run through a field in field_props
    where the run x position is normalised with respect to the
    field x position limits and the run direction (east or west)
    """
    for field in f_props:
        xmin = np.min(field.xy[0])
        xmax = np.max(field.xy[0])
        fp_e = np.linspace(-1, 1, 1000)
        fp_w = np.linspace(1, -1, 1000)
        xp = np.linspace(xmin, xmax, 1000)
        for run in field.runs:
            if get_run_direction(run) == "e":
                x_nrmd = np.interp(run.xy[0], xp, fp_e)
            else:
                x_nrmd = np.interp(run.xy[0], xp, fp_w)
            run.normalised_position = x_nrmd
    return f_props


def mask_short_runs(xy: np.ndarray, min_run_len=50) -> np.ndarray:

    assert isinstance(xy, np.ma.MaskedArray)
    orig_mask = xy[0].mask

    labelled_runs = labelContigNonZeroRuns(np.invert(xy.mask[0]))
    run_starts = getLabelStarts(labelled_runs)
    run_stops = getLabelEnds(labelled_runs)

    # mask runs that are shorter than the minimum run length
    mask = np.zeros_like(xy, dtype=bool)
    for start, stop in zip(run_starts, run_stops):
        if (stop - start) < min_run_len:
            mask[:, start:stop] = True

    # mask the runs in the xy array
    mask = np.logical_or(mask, orig_mask)

    # masked_xy = np.ma.masked_array(xy, mask=mask)

    return mask


def get_field_props_for_linear_track(
    trial: AxonaTrial, cluster: int, channel: int, direction=None, **kwargs
) -> list[FieldProps]:
    """
    Get the field properties for a linear track trial.

    Filters the linear track data based on speed, direction (east
    or west; larger ranges than the usual 90degs are used), and position
    (masks the start and end 12cm of the track)
    """
    # parse kwargs
    field_thresh_prc = kwargs.get("field_threshold_percent", 50)
    field_rate_thresh = kwargs.get("field_rate_threshold", 0.5)
    min_run_len = kwargs.get("min_run_length", 50)
    min_field_sz_bins = kwargs.get("min_field_size_in_bins", 3)
    min_num_spikes = kwargs.get("min_num_spikes", 0)

    # load pos from scratch to make sure we're using the
    # correct ppm
    ppm = int(trial.settings["tracker_pixels_per_metre"])

    trial.PosCalcs.ppm = ppm  # triggers postprocesspos()

    speed_filter = TrialFilter("speed", EXCLUDE_SPEEDS[0], EXCLUDE_SPEEDS[1])
    min_x = np.nanmin(trial.PosCalcs.xy[0].data)
    max_x = np.nanmax(trial.PosCalcs.xy[0].data)
    x_filt0 = TrialFilter("xrange", min_x, min_x + 12)
    x_filt1 = TrialFilter("xrange", max_x - 12, max_x)

    if direction:
        # broaden the directional filters to 180 degrees
        # as runs are getting broken up
        if direction == "e" or direction == "east":
            # filter out east direction - LEAVING onyl west
            # dir_filter0 = TrialFilter("dir", 270, 90)
            dir_filter0 = TrialFilter("dir", "e")
        elif direction == "w" or direction == "west":
            # filter out west direction - LEAVING only east
            # dir_filter0 = TrialFilter("dir", 90, 270)
            dir_filter0 = TrialFilter("dir", "w")

        n = TrialFilter("dir", "n")
        s = TrialFilter("dir", "s")
        trial.apply_filter(x_filt0, x_filt1, speed_filter, dir_filter0, n, s)
    else:
        trial.apply_filter(
            x_filt0,
            x_filt1,
            speed_filter,
        )

    # partition the cells firing into distinct fields
    binned_data = trial.get_rate_map(
        cluster, channel, var_type=VariableToBin.PHI, **kwargs
    )
    # get the field properties
    _, _, label_image, _ = partitionFields(
        binned_data,
        field_threshold_percent=field_thresh_prc,
        field_rate_threshold=field_rate_thresh,
    )

    # Filter out small fields from the image partition here...
    label_image = remove_small_objects(label_image, min_field_sz_bins)
    # and relabel the label image sequentially
    label_image, _, _ = relabel_sequential(label_image)

    pos_data = trial.PosCalcs.xy[0]
    spikes_in_position = trial.get_spike_times_binned_into_position(
        cluster, channel
    ).astype(int)

    field_props = fieldprops(
        label_image,
        binned_data,
        pos_data,
        spikes_in_position,
        min_run_length=min_run_len,
    )

    # remove the runs that have no spikes in them
    for f in field_props:
        runs = []
        for i, run in enumerate(f.runs):
            if run.n_spikes > 0:
                runs.append(run)
        f.runs = runs

    # remove fields that don'thave enough spikes
    field_props = [
        f for f in field_props if f.n_spikes >= min_num_spikes and len(f.runs) > 0
    ]

    return field_props


def merge_field_and_lfp(
    f_props: list[FieldProps],
    trial: AxonaTrial,
    **kwargs,
) -> list[FieldProps]:
    """
    Adds LFP data to the list of field properties.

    Does some processing to remove bad theta cycles based on power,
    length and phase.
    """
    if not trial.EEGCalcs:
        trial.load_lfp()

    lfp_data = trial.EEGCalcs.sig
    # mean normalise the LFP data
    lfp_data = lfp_data - np.mean(lfp_data)

    lfp_fs = trial.EEGCalcs.fs
    L = LFPOscillations(lfp_data, lfp_fs)

    filt_sig, phase, _, _, _ = L.getFreqPhase(lfp_data, [MIN_THETA, MAX_THETA], 2)

    cluster = f_props[0].binned_data.cluster_id[0].Cluster
    channel = f_props[0].binned_data.cluster_id[0].Channel
    spike_times = trial.get_spike_times(cluster, channel)

    spk_phase = phase.copy()

    # unmask masked arrays
    if isinstance(spk_phase, np.ma.MaskedArray):
        spk_phase.mask = False

    min_spiking_phase = get_phase_of_min_spiking(spk_phase)
    phase_adj = fixAngle(
        phase - min_spiking_phase * (np.pi / 180) + MIN_ALLOWED_SPIKE_PHASE
    )
    is_neg_freq = np.diff(np.unwrap(phase_adj)) < 0
    is_neg_freq = np.append(is_neg_freq, is_neg_freq[-1])

    # get start of theta cycles as points where diff > pi
    phase_diff = np.diff(phase_adj)
    cycle_starts = phase_diff[1::] < -np.pi
    cycle_starts = np.append(cycle_starts, True)
    cycle_starts = np.insert(cycle_starts, 0, True)
    cycle_starts[is_neg_freq] = False

    cycle_label = np.cumsum(cycle_starts)

    # calculate power and find low power cycles
    power = np.power(filt_sig, 2)

    cycle_total_valid_power = np.bincount(
        cycle_label[~is_neg_freq], weights=power[~is_neg_freq]
    )
    cycle_valid_bincount = np.bincount(cycle_label[~is_neg_freq])
    cycle_valid_mean_power = cycle_total_valid_power / cycle_valid_bincount
    power_reject_thresh = np.nanpercentile(cycle_valid_mean_power, MIN_POWER_THRESHOLD)

    cycle_has_bad_power = cycle_valid_mean_power < power_reject_thresh

    allowed_theta_len = (
        np.floor((1.0 / MAX_THETA) * lfp_fs).astype(int),
        np.ceil((1.0 / MIN_THETA) * lfp_fs).astype(int),
    )

    cycle_total_bin_count = np.bincount(cycle_label)
    cycle_has_bad_len = np.logical_or(
        cycle_total_bin_count < allowed_theta_len[0],
        cycle_total_bin_count > allowed_theta_len[1],
    )

    # remove data calculated as bad
    is_bad_cycle = np.logical_or(cycle_has_bad_len, cycle_has_bad_power)
    is_in_bad_cycle = is_bad_cycle[cycle_label]
    is_bad = np.logical_or(is_in_bad_cycle, is_neg_freq)

    # apply is_bad to data...
    filt_sig = np.ma.masked_array(filt_sig, mask=is_bad)
    phase_adj = np.ma.MaskedArray(phase_adj, mask=is_bad)
    amp_adj = np.ma.MaskedArray(filt_sig, mask=is_bad)
    cycle_label = np.ma.MaskedArray(cycle_label, mask=is_bad)

    # Now extract the relevant sections of these masked arrays and
    # add them to the relevant run...
    lfp_to_pos_ratio = lfp_fs / POS_SAMPLE_RATE

    for field in f_props:
        for r in field.runs:
            lfp_slice = slice(
                int(r.slice.start * lfp_to_pos_ratio),
                int(r.slice.stop * lfp_to_pos_ratio),
            )
            # add the highest resolution spike timestamps we can
            spk_ts = spike_times[
                np.logical_and(
                    spike_times >= r.slice.start / POS_SAMPLE_RATE,
                    spike_times <= r.slice.stop / POS_SAMPLE_RATE,
                )
            ]
            lfp_segment = LFPSegment(
                field.label,
                r.label,
                lfp_slice,
                spk_ts,
                lfp_data[lfp_slice],
                filt_sig[lfp_slice],
                phase_adj[lfp_slice],
                amp_adj[lfp_slice],
                lfp_fs,
                [MIN_THETA, MAX_THETA],
            )
            r.lfp_segment = lfp_segment

    return f_props


def get_phase_precession_per_field(f_props: list[FieldProps], **kwargs):
    """
    Get the phase and normalized (-1 -> +1) position of spikes in a field.
    """

    phase_pos = {}

    for f in f_props:
        phase = []
        normalised_position = []
        phase_pos[f.label] = {}
        for r in f.runs:
            idx = (r.lfp_segment.spike_times * r.lfp_segment.sample_rate).astype(
                int
            ) - r.lfp_segment.slice.start
            mask = r.lfp_segment.phase[idx].mask
            if np.count_nonzero(np.invert(mask)) > MIN_SPIKES_PER_RUN:
                phase.extend(r.lfp_segment.phase[idx][~mask])
                pos_idx = (r.lfp_segment.spike_times * POS_SAMPLE_RATE).astype(
                    int
                ) - r.slice.start
                normalised_position.extend(r.normalised_position[pos_idx[~mask]])
        phase = np.array(flatten_list(phase))
        normalised_position = np.array(flatten_list(normalised_position))
        phase_pos[f.label]["phase"] = phase
        phase_pos[f.label]["normalised_position"] = normalised_position

    return phase_pos
