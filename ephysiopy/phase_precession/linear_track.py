import numpy as np
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ephysiopy.common.utils import (
    fixAngle,
    labelContigNonZeroRuns,
    getLabelStarts,
    getLabelEnds,
    flatten_list,
    VariableToBin,
    TrialFilter,
)
from ephysiopy.visualise.plotting import colored_line
from ephysiopy.io.recording import AxonaTrial
from ephysiopy.common.fieldcalcs import (
    FieldProps,
    LFPSegment,
    partitionFields,
    fieldprops,
)
from ephysiopy.common.rhythmicity import LFPOscillations
from ephysiopy.common.phasecoding import get_phase_of_min_spiking


MIN_THETA = 6
MAX_THETA = 12
# defines start/ end of theta cycle
MIN_ALLOWED_SPIKE_PHASE = np.pi
# percentile power below which theta cycles are rejected
MIN_POWER_THRESHOLD = 0
POS_SAMPLE_RATE = 50
# SOME FIELD THRESHOLDS
FIELD_THRESHOLD_PERCENT = 50
FIELD_RATE_THRESHOLD = 0.5
MIN_FIELD_SIZE_IN_BINS = 3
# SOME RUN THRESHOLDS
MIN_RUN_LENGTH = 50
EXCLUDE_SPEEDS = (0, 0.01)  # cm/s


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
    trial: AxonaTrial, cluster: int, channel: int, direction="e", **kwargs
) -> list[FieldProps]:
    """
    Get the field properties for a linear track trial.

    Filters the linear track data based on speed, direction, and position
    (masks the start and end 12cm of the track)
    """
    # parse kwargs
    field_thresh_prc = kwargs.get("field_threshold_percent", 50)
    field_rate_thresh = kwargs.get("field_rate_threshold", 0.5)
    min_run_len = kwargs.get("min_run_length", 50)
    min_field_sz_bins = kwargs.get("min_field_size_in_bins", 3)

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
            dir_filter0 = TrialFilter("dir", 270, 90)
        elif direction == "w" or direction == "west":
            # filter out west direction - LEAVING only east
            dir_filter0 = TrialFilter("dir", 90, 270)

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
        cluster, channel, var_type=VariableToBin.X, **kwargs
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
    # mask data in the trial for runs that were discarded forbeing too short

    return field_props


def merge_field_and_lfp(
    field_props: list[FieldProps],
    trial: AxonaTrial,
    **kwargs,
) -> list[FieldProps]:
    """
    Adds LFP data to the field properties.

    Does some processing to remove bad theta cycles based on power,
    length and phase.
    """

    lfp_data = trial.EEGCalcs.sig
    lfp_fs = trial.EEGCalcs.fs
    L = LFPOscillations(lfp_data, lfp_fs)

    filt_sig, phase, _, _, _ = L.getFreqPhase(lfp_data, [MIN_THETA, MAX_THETA], 2)

    cluster = field_props[0].binned_data.cluster_id[0].Cluster
    channel = field_props[0].binned_data.cluster_id[0].Channel
    spike_times = trial.get_spike_times(cluster, channel)

    # spike_lfp_index = np.floor(spike_times * lfp_fs).astype(int)
    # spike_count = np.bincount(spike_lfp_index, minlength=len(lfp_data))
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
    power_reject_thresh = np.percentile(cycle_valid_mean_power, MIN_POWER_THRESHOLD)

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
    phase_adj = np.ma.MaskedArray(phase_adj, mask=is_bad)
    amp_adj = np.ma.MaskedArray(filt_sig, mask=is_bad)
    cycle_label = np.ma.MaskedArray(cycle_label, mask=is_bad)
    # spike_count = np.ma.MaskedArray(spike_count, mask=is_bad)

    # Now extract the relevant sections of these masked arrays and
    # add them to the relevant run...
    lfp_to_pos_ratio = lfp_fs / POS_SAMPLE_RATE

    for field in field_props:
        for run in field.runs:
            lfp_slice = slice(
                int(run._slice.start * lfp_to_pos_ratio),
                int(run._slice.stop * lfp_to_pos_ratio),
            )
            lfp_segment = LFPSegment(
                field.label,
                run.label,
                lfp_slice,
                run.spike_position_index / POS_SAMPLE_RATE,
                lfp_data[lfp_slice],
                filt_sig[lfp_slice],
                phase_adj[lfp_slice],
                amp_adj[lfp_slice],
                lfp_fs,
                [MIN_THETA, MAX_THETA],
            )
            run.lfp_segment = lfp_segment

    return field_props


# TODO: these plotting fncs needed finishing


def plot_field_and_runs(trial: AxonaTrial, field_props: list[FieldProps]):
    """
    Plot runs versus time where the colour of the line indicates
    directional heading. The field limits are also plotted and spikes are
    overlaid on the runs. Boxes delineate the runs that have been identified
    in field_props
    """

    fig, ax = plt.subplots(layout="constrained")
    cluster = field_props[0].binned_data.cluster_id[0].Cluster
    channel = field_props[0].binned_data.cluster_id[0].Channel

    spk_in_pos = trial.get_spike_times_binned_into_position(cluster, channel)
    time = np.arange(0, trial.PosCalcs.duration, 1 / trial.PosCalcs.sample_rate)

    lc = colored_line(
        time,
        trial.PosCalcs.xy[0],
        trial.PosCalcs.dir,
        ax,
        cmap="hsv",
        zorder=1,
    )
    fig.colorbar(lc, orientation="horizontal", label="Direction (degrees)")
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, np.nanmax(trial.PosCalcs.xy[0].data))
    ax.set_yticklabels([])

    for f in field_props:
        # add the field limits as a shaded area
        slice = f.slice[0]
        be = f.binned_data.bin_edges[0]
        ax.axhspan(be[slice.start], be[slice.stop], alpha=0.3)
        for r in f.runs:
            # draw a rectangle that bounds the run in x and time
            ymin = np.nanmin(r.xy[0])
            ymax = np.nanmax(r.xy[0])
            height = ymax - ymin
            xmin = time[r.run_start]
            xmax = time[r.run_stop]
            width = xmax - xmin
            rect = Rectangle(
                (xmin, ymin),
                width,
                height,
                alpha=0.8,
                color="red",
            )
            ax.add_patch(rect)
            # annotate the run with its run number
            ax.annotate(str(r.label), xy=(xmax, ymax))

    idx = np.nonzero(spk_in_pos)[1]
    ax.scatter(time[idx], trial.PosCalcs.xy[0][idx], c="k", s=10, zorder=2)
    ax_histx = ax.inset_axes([1.05, 0, 0.15, 1], sharey=ax)
    add_hist_to_y_axes(
        ax_histx, trial.PosCalcs.xy[0][idx], field_props[0].binned_data.bin_edges[0]
    )
    ax_histx1 = ax.inset_axes([-0.2, 0, 0.15, 1], sharey=ax)
    add_hist_to_y_axes(
        ax_histx1,
        trial.PosCalcs.xy[0][idx],
        field_props[0].binned_data.bin_edges[0],
        flip=True,
    )
    plt.show()


def add_hist_to_y_axes(
    ax_histx,
    data: np.ndarray,
    bin_edges: np.ndarray,
    **kwargs,
):
    """
    Add a histogram to the y axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the histogram to.
    data : np.ndarray
        The data to plot in the histogram.
    bins : int, optional
        Number of bins for the histogram, by default 50.
    **kwargs : dict
        Additional keyword arguments for the histogram.
    """
    # no labels
    ax_histx.tick_params(axis="y", labelleft=False, labelright=False)
    ax_histx.tick_params(axis="x", labeltop=False, labelbottom=False)
    if kwargs.pop("flip", False):
        ax_histx.hist(
            data,
            bins=bin_edges,
            weights=-np.ones_like(data),
            orientation="horizontal",
            **kwargs,
        )
    else:
        ax_histx.hist(data, bins=bin_edges, orientation="horizontal", **kwargs)


def plot_phase_v_position(
    field_props: list[FieldProps],
    ax=None,
    **kwargs,
):
    """
    Plot the phase of the LFP signal at each position in the field.

    Parameters
    ----------
    field_props : list[FieldProps]
        List of FieldProps objects containing run and LFP data.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.
    **kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """

    for field in field_props:
        fig, ax = plt.subplots()
        runs_pos = flatten_list(field.runs_normalized_position)
        runs_phase = flatten_list(field.phase)
        runs_spikes = flatten_list(field.runs_observed_spikes)
        idx = np.nonzero(np.array(runs_spikes))[0]
        ax.scatter(np.array(runs_pos)[idx], np.array(runs_phase)[idx], **kwargs)

        ax.set_xlabel("Position")
        ax.set_ylabel("Phase (radians)")
        plt.show()

    # return ax
