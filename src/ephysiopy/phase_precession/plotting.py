# TODO: these plotting fncs needed finishing
import warnings
import copy
import matplotlib
import numpy as np
from scipy import ndimage
from ephysiopy.common.utils import flatten_list, BinnedData, repeat_ind
from ephysiopy.common.utils import bwperim
from ephysiopy.visualise.plotting import stripAxes, _add_colour_wheel
from ephysiopy.visualise.plotting import colored_line
from ephysiopy.io.recording import AxonaTrial
from ephysiopy.common.fieldcalcs import (
    FieldProps,
    RunProps,
    infill_ratemap,
    filter_runs,
)
from ephysiopy.common.fieldproperties import fieldprops
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import RegularPolyCollection
import matplotlib.colors as colours
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


@stripAxes
def _stripAx(ax):
    return ax


def plot_phase_precession(
    phase, normalised_position, slope, intercept, ax=None, **kwargs
):
    """
    Plot the phase precession of spikes in a field.

    Parameters
    ----------
    field_phase_pos : dict[str, dict[np.ndarray, np.ndarray]]
        Dictionary containing the phase and normalised position for each field.

    ax : matplotlib.axes.Axes, optional

    """

    phase = phase
    normalised_position = normalised_position

    if ax is None:
        fig, ax = plt.subplots()

    # scatter plot of phase vs normalised position
    # repeat the y-axis values for clarity
    mm = (0, -4 * np.pi, -2 * np.pi, 2 * np.pi, 4 * np.pi)
    for m in mm:
        ax.plot((-1, 1), (-slope + intercept + m, slope + intercept + m), "r", lw=3)
        ax.scatter(normalised_position, phase + m, s=6, c="k", **kwargs)

    ax.set_xlim(-1, 1)
    xtick_locs = np.linspace(-1, 1, 3)
    ax.set_xticks(xtick_locs, list(map(str, xtick_locs)))
    ax.set_yticks(sorted(mm), ["-4π", "-2π", "0", "2π", "4π"])
    ax.set_ylim(-2 * np.pi, 4 * np.pi)
    subaxis_title_fontsize = 8
    ax.set_ylabel("Phase", fontsize=subaxis_title_fontsize)
    ax.set_xlabel("Normalised position", fontsize=subaxis_title_fontsize)

    return ax


def plot_runs_and_precession(
    trial: AxonaTrial, cluster: int, channel: int, field_props: list[FieldProps]
):
    """
    Plot runs versus time where the colour of the line indicates
    directional heading. The field limits are also plotted and spikes are
    overlaid on the runs. Boxes delineate the runs that have been identified
    in field_props. Also plots phase precession for each field.
    """
    # warn if the trial has not been masked/ filtered
    if trial.filter is None:
        warnings.warn(
            "Trial has not been filtered. "
            "Consider applying a position filter before plotting runs."
        )

    # need to clearly pull out the runs along the linear track
    # so use fieldprops to do this using a unitary label image
    # as input and some other params that the input
    # field_props would have been generated with
    label_image = np.ones_like(field_props[0]._intensity_image, dtype=int)
    # smooth xy quite a bit as we just want runs going smoothly
    # from one end of the track to the other
    xy = trial.PosCalcs.smoothPos(trial.PosCalcs.xy, window_len=31)

    fp = fieldprops(
        label_image,
        xy=xy[0],
        binned_data=field_props[0].binned_data,
        spike_times=trial.get_spike_times(cluster, channel),
        method="clump_runs",
    )
    # filter out the short duration runs
    fp = filter_runs(fp, ["min_speed", "duration"], [np.greater, np.greater], [0, 0.1])
    # filter for distance traversed

    plt.figure(constrained_layout=True)

    time = np.arange(0, trial.PosCalcs.duration, 1 / trial.PosCalcs.sample_rate)
    run_labels = np.zeros_like(time)
    for i, f in enumerate(fp):
        for r in f.runs:
            run_labels[r.slice] = i

    [plt.plot(r.xy[0], time[r.slice], color="lightgrey", zorder=0) for r in fp[0].runs]

    plt.show()


def plot_field_and_runs(trial: AxonaTrial, field_props: list[FieldProps]):
    """
    Plot runs versus time where the colour of the line indicates
    directional heading. The field limits are also plotted and spikes are
    overlaid on the runs. Boxes delineate the runs that have been identified
    in field_props
    """

    fig, ax = plt.subplots(layout="constrained")
    time = np.arange(0, trial.PosCalcs.duration, 1 / trial.PosCalcs.sample_rate)

    ax.set_ylim(0, time[-1])
    ax.set_xlim(0, np.nanmax(trial.PosCalcs.xy[0].data))
    ax.set_xticklabels([])

    # plot all the position data as a light grey line in the background
    ax.plot(
        trial.PosCalcs.xy[0],
        time,
        color="lightgrey",
        zorder=0,
    )

    xy = field_props[0].runs[0].xy
    tail = (0.55, 0.025) if xy[0, -1] - xy[0, 0] < 0 else (0.45, 0.025)
    head = (0.45, 0.025) if xy[0, -1] - xy[0, 0] < 0 else (0.55, 0.025)

    arrow = mpatches.FancyArrowPatch(tail, head, mutation_scale=100, color="black")
    ax.add_patch(arrow)

    _add_colour_wheel(ax, fig, bbox=(0.05, 0.6, 0.1, 0.1))

    be = field_props[0].binned_data.bin_edges[0]

    for f in field_props:
        # add the field limits as a shaded area
        slice = f.slice[0]
        ax.axvspan(be[slice.start], be[slice.stop], alpha=0.3)
        for irun in f.runs:
            # annotate the run with its run number
            run_time = np.arange(irun.slice.start, irun.slice.stop) / irun.sample_rate
            # draw the run as a black line
            ax.plot(irun.xy[0], run_time, color="black", zorder=1)

            # colour spike positions by phase of firing
            x = irun.spiking_var("xy")
            t = irun.spiking_var("time")
            spike_phase = np.ravel(irun.lfp.spiking_var("phase"))
            norm = colours.Normalize(-np.pi, np.pi, False)
            ax.scatter(
                x,
                t,
                c=spike_phase,
                cmap="hsv",
                norm=norm,
                s=10,
                zorder=2,
            )
            xmax = np.nanmax(irun.xy[0])
            ymax = time[irun.run_stop - 1]
            ax.annotate(str(irun.label), xy=(xmax, ymax))
    # fig.colorbar(lc, orientation="horizontal", label="Direction (degrees)")

    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes(
        "top", 1.2, pad=0.2, sharex=ax, transform=ax.transAxes
    )
    h = field_props[0].binned_data.binned_data[0]
    axHistx.bar(
        be[:-1],
        h,
        width=np.diff(be),
        align="edge",
        color="lightgrey",
        rasterized=True,
    )

    plt.setp(axHistx.get_xticklabels(), visible=False)
    # Label only the min and max of the y-axis
    # max is rounded to the nearest 10
    maxRate = int(np.ceil(np.max(h) / 10.0) * 10)
    axHistx.set_ylim(0, maxRate)
    axHistx.set_yticks((0, maxRate))
    plt.show()


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


def ratemap_line_graph(binned_data: BinnedData, ax=None, **kwargs) -> plt.Axes:
    """
    Plot a line graph of the rate map.

    Parameters
    ----------
    binned_data : BinnedData
        The binned data containing the rate map.
    **kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    x = binned_data.bin_edges[0][:-1]
    y = binned_data.binned_data[0]
    ax.plot(x, y, **kwargs)
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Firing Rate (Hz)")
    return ax


def add_fields_to_line_graph(f_props: list[FieldProps], ax=None) -> plt.Axes:
    """
    Add field boundaries to a line graph of the rate map.

    Parameters
    ----------
    f_props : list[FieldProps]
        List of FieldProps containing field information.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the field boundaries added.
    """
    ax = ratemap_line_graph(f_props[0].binned_data, ax=ax)

    for f in f_props:
        xmin = f.binned_data.bin_edges[0][f.slice[0].start]
        xmax = f.binned_data.bin_edges[0][f.slice[0].stop]
        ax.axvspan(xmin, xmax, alpha=0.3, label=f"{f.label}")

    return ax


def plot_lfp_and_spikes_per_run(f_props: list[FieldProps]) -> plt.Axes:
    """
    Plot the LFP and spikes per run.

    Parameters
    ----------
    f_props : list[FieldProps]
        List of FieldProps containing field information.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    for f in f_props:
        plt.figure()
        nrows = np.ceil(len(f.runs) / 2).astype(int)
        plt.subplot(nrows, 2, 1)
        for i, r in enumerate(f.runs):
            plt.subplot(nrows, 2, i + 1)
            t = np.linspace(
                r.lfp.slice.start / r.lfp.sample_rate,
                r.lfp.slice.stop / r.lfp.sample_rate,
                len(r.lfp.phase),
            )
            plt.plot(t, r.lfp.filtered_signal, label=f"Run {r.label}")
            y = np.interp(r.lfp.spike_times, t, r.lfp.filtered_signal)
            plt.plot(r.lfp.spike_times, y, "ro")
            ax = plt.gca()
            ax.set_xticklabels("")
            ax.set_yticklabels("")
            ax.legend()

    return ax


def plot_field_props(field_props: list[FieldProps]):
    """
    Plots the fields in the list of FieldProps

    Parameters
    ----------
    list of FieldProps
    """
    fig = plt.figure()
    subfigs = fig.subfigures(
        2,
        2,
    )
    ax = subfigs[0, 0].subplots(1, 1)
    # ax = fig.add_subplot(221)
    fig.canvas.manager.set_window_title("Field partitioning and runs")
    outline = np.isfinite(field_props[0]._intensity_image)
    outline = ndimage.binary_fill_holes(outline)
    outline = np.ma.masked_where(np.invert(outline), outline)
    outline_perim = bwperim(outline)
    outline_idx = np.nonzero(outline_perim)
    bin_edges = field_props[0].binned_data.bin_edges
    outline_xy = bin_edges[1][outline_idx[1]], bin_edges[0][outline_idx[0]]
    ax.plot(outline_xy[0], outline_xy[1], "k.", ms=1)
    # PLOT 1
    cmap_arena = matplotlib.colormaps["tab20c_r"].resampled(1)
    ax.pcolormesh(bin_edges[1], bin_edges[0], outline_perim, cmap=cmap_arena)
    # Runs through fields in global x-y coordinates
    max_field_label = np.max([f.label for f in field_props])
    cmap = matplotlib.colormaps["Set1"].resampled(max_field_label)
    [
        [
            ax.plot(r.xy[0], r.xy[1], color=cmap(f.label - 1), label=f.label - 1)
            for r in f.runs
        ]
        for f in field_props
    ]
    # plot the perimeters of the field(s)
    [
        ax.plot(
            f.global_perimeter_coords[0],
            f.global_perimeter_coords[1],
            "k.",
            ms=1,
        )
        for f in field_props
    ]
    [ax.plot(f.xy_at_peak[0], f.xy_at_peak[1], "ko", ms=2) for f in field_props]
    norm = matplotlib.colors.Normalize(1, max_field_label)
    tick_locs = np.linspace(1.5, max_field_label - 0.5, max_field_label)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        ticks=tick_locs,
    )
    cbar.set_ticklabels(list(map(str, [f.label for f in field_props])))
    # ratemaps are plotted with origin in top left so invert y axis
    ax.invert_yaxis()
    ax.set_aspect("equal")
    _stripAx(ax)
    # PLOT 2
    # Runs on the unit circle on a per field basis as it's too confusing to
    # look at all of them on a single unit circle
    n_rows = 2
    n_cols = np.ceil(len(field_props) / n_rows).astype(int)

    ax1 = np.ravel(subfigs[0, 1].subplots(n_rows, n_cols))
    [
        ax1[f.label - 1].plot(
            f.pos_xy[0],
            f.pos_xy[1],
            color=cmap(f.label - 1),
            lw=0.5,
            zorder=1,
        )
        for f in field_props
    ]
    [
        a.add_artist(
            matplotlib.patches.Circle((0, 0), 1, fc="none", ec="lightgrey", zorder=3),
        )
        for a, _ in zip(ax1, field_props)
    ]
    [a.set_xlim(-1, 1) for a in ax1]
    [a.set_ylim(-1, 1) for a in ax1]
    [a.set_title(f.label) for a, f in zip(ax1, field_props)]
    [a.set_aspect("equal") for a in ax1]
    [_stripAx(a) for a in ax1]

    # PLOT 3
    # The runs through the fields coloured by the distance of each xy coord in
    # the field to the peak and angle of each point on the perimeter to
    # the peak
    dist_cmap = matplotlib.colormaps["jet_r"]
    angular_cmap = matplotlib.colormaps["hsv"]
    im = np.zeros_like(field_props[0]._intensity_image).astype(int) * np.nan
    for f in field_props:
        sub_im = f.image * np.nan
        idx = np.nonzero(f.bw_perim)
        # the angles made by the perimeter to the field peak
        sub_im[idx[0], idx[1]] = f.perimeter_angle_from_peak
        im[f.slice] = sub_im
    ax2 = subfigs[1, 0].subplots(1, 1)
    # distances as collections of Rectangles
    distances = np.concatenate(
        [f.xy_dist_to_peak / f.xy_dist_to_peak.max() for f in field_props]
    )
    face_colours = dist_cmap(distances)
    offsets = np.concatenate([f.xy.T for f in field_props])
    rects = RegularPolyCollection(
        numsides=4,
        rotation=0,
        facecolors=face_colours,
        edgecolors=face_colours,
        offsets=offsets,
        offset_transform=ax2.transData,
    )
    ax2.add_collection(rects)
    ax2.pcolormesh(bin_edges[1], bin_edges[0], im, cmap=angular_cmap)
    _stripAx(ax2)

    ax2.invert_yaxis()
    ax2.set_aspect("equal")
    degs_norm = matplotlib.colors.Normalize(0, 360)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=angular_cmap, norm=degs_norm),
        ax=ax2,
    )
    [ax2.plot(f.xy_at_peak[0], f.xy_at_peak[1], "ko", ms=2) for f in field_props]
    # PLOT 4
    # The smoothed ratemap - maybe make this the first sub plot
    ax3 = subfigs[1, 1].subplots(1, 1)
    # smooth the ratemap a bunch
    rmap_to_plot = copy.copy(field_props[0]._intensity_image)
    rmap_to_plot = infill_ratemap(rmap_to_plot)
    ax3.pcolormesh(bin_edges[1], bin_edges[0], rmap_to_plot)
    # add the field labels to the ratemap plot
    [
        ax3.text(f.xy_at_peak[0], f.xy_at_peak[1], str(f.label), ha="left", va="bottom")
        for f in field_props
    ]

    ax3.invert_yaxis()
    ax3.set_aspect("equal")
    _stripAx(ax3)


def plot_lfp_segment(field: FieldProps, lfp_sample_rate: int = 250):
    """
    Plot the lfp segments for a series of runs through a field including
    the spikes emitted by the cell.
    """
    assert hasattr(field.runs[0], "lfp")

    n_rows = 3
    n_cols = np.ceil(len(field.runs) / n_rows).astype(int)
    fig = plt.figure()
    subfigs = fig.subfigures(
        1,
        1,
    )

    ax = np.ravel(subfigs.subplots(n_rows, n_cols))
    for i_run, run in enumerate(field.runs):
        sig = run.lfp.filtered_signal.ravel()
        t = np.linspace(
            run.lfp.slice.start / lfp_sample_rate,
            run.lfp.slice.stop / lfp_sample_rate,
            len(sig),
        )
        ax[i_run].plot(t, sig)
        spike_phase_pos = np.interp(
            run.lfp.spike_times, t, run.lfp.filtered_signal.ravel()
        )
        ax[i_run].plot(run.lfp.spike_times, spike_phase_pos, "ro")
    plt.show()


def plot_lfp_run(
    run: RunProps, cycle_labels: np.ndarray = None, lfp_sample_rate: int = 250, **kwargs
):
    """
    Plot the lfp segment for a single run through a field including
    the spikes emitted by the cell.

    Notes
    -----
    There are very small inaccuracies here due to the way the timebase
    is being created from the slice belonging to the run and the way
    the indexing is being done by repeating the indices (repeat_ind)
    of the spike counts binned wrt the LFP sample rate. This shouldn;t
    matter for purposes of plotting - it's only when you zoom in a lot
    that you can see the diffferences between this and the actual spike
    times etc (if you can be arsed to plot them)
    """

    assert hasattr(run, "lfp")
    cmap = kwargs.get("cmap", "tab10")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # create a colormap for the cycle labels
    colours = run.lfp.cycle_label[0]
    sig = run.lfp.filtered_signal[0]
    t = run.lfp.time
    ax.plot(t, sig.data, color="lightgrey")
    colored_line(t, sig, colours, ax=ax, cmap=cmap)
    inds = repeat_ind(run.lfp.spike_count.ravel().data)
    spike_amp = np.take(sig, inds)
    spike_times = np.take(t, inds)
    ax.plot(spike_times, spike_amp, "ro")
    return ax


def add_colorwheel_to_fig(ax):
    """
    Add a colorwheel to the given axis.

    """
    assert ax.name == "polar"
    ax._direction = 2 * np.pi
    norm = matplotlib.colors.Normalize(0, 2 * np.pi)
    steps = 2056
    cmap = matplotlib.colormaps["hsv"]
    cmap.N = steps
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="horizontal"
    )
    cb.outline.set_visible(False)
    ax.set_axis_off()
    ax.set_rlim([-1, 1])
    return ax


# TODO: Check this works with 1- and 2-D data
def plot_spikes_in_runs_per_field(
    field_label: np.ndarray,
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    spikes_in_time: np.ndarray,
    ttls_in_time: np.ndarray | None = None,
    **kwargs,
):
    """
    Debug plotting to show spikes per run per field found in the ratemap
    as a raster plot

    Parameters
    ----------
    field_label : np.ndarray
    The field labels for each position bin a vector
    run_starts, runs_ends : np.ndarray
        The start and stop indices of each run (vectors)
    spikes_in_time : np.ndarray
        The number of spikes in each position bin (vector)
    ttls_in_time : np.ndarray
        TTL occurences in time (vector)

    **kwargs
        separate_plots : bool
            If True then each field will be plotted in a separate figure
        single_axes : bool
            If True will plot all the runs/ spikes in a single axis with fields delimited by horizontal lines

    Returns
    -------
    fig, axes : tuple
        The figure and axes objects
    """
    spikes_in_time = np.ravel(spikes_in_time)
    if ttls_in_time:
        assert len(spikes_in_time) == len(ttls_in_time)
    run_start_stop_idx = np.array([run_starts, run_ends]).T
    run_field_id = field_label[run_start_stop_idx[:, 0]]
    runs_per_field = np.histogram(run_field_id, bins=range(1, max(run_field_id) + 2))[0]
    max_run_len = np.max(run_start_stop_idx[:, 1] - run_start_stop_idx[:, 0])
    all_slices = np.array([slice(r[0], r[1]) for r in run_start_stop_idx])
    # create the figure window first then do the iteration through fields etc
    master_raster_arr = []
    # a grey colour for the background i.e. how long the run was
    grey = np.array([0.8627, 0.8627, 0.8627, 1])
    # iterate through each field then pull out the
    max_spikes = np.nanmax(spikes_in_time).astype(int) + 1
    orig_cmap = matplotlib.colormaps["spring"].resampled(max_spikes)
    cmap = orig_cmap(np.linspace(0, 1, max_spikes))
    cmap[0, :] = grey
    newcmap = ListedColormap(cmap)
    # some lists to hold the outputs
    # spike count for each run through the field
    master_raster_arr = []
    # list for count of total number of spikes per field
    spikes_per_run = []
    # counts of ttl puleses emitted during each run
    if ttls_in_time is not None:
        ttls_per_field = []
    # collect all the per field spiking, ttls etc first then plot
    # in a separate iteration
    for i, field_id in enumerate(np.unique(run_field_id)):
        # create a temporary array to hold the raster for this fields runs
        raster_arr = np.zeros(shape=(runs_per_field[i], max_run_len)) * np.nan
        ttl_arr = np.zeros(shape=(runs_per_field[i], max_run_len)) * np.nan
        # get the indices into the time binned spikes of the runs
        i_field_slices = all_slices[run_field_id == field_id]
        # breakpoint()
        for j, s in enumerate(i_field_slices):
            i_run_len = s.stop - s.start
            raster_arr[j, 0:i_run_len] = spikes_in_time[s]
            if ttls_in_time is not None:
                ttl_arr[j, 0:i_run_len] = ttls_in_time[s]
        spikes_per_run.append(int(np.nansum(raster_arr)))
        if ttls_in_time:
            ttls_per_field.append(ttl_arr)
        master_raster_arr.append(raster_arr)

    if "separate_plots" in kwargs.keys():
        for i, field_id in enumerate(np.unique(run_field_id)):
            _, ax = plt.subplots(1, 1)
            ax.imshow(master_raster_arr[i], cmap=newcmap, aspect="auto")
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_ylabel(f"Field {field_id}")
    elif "single_axes" in kwargs.keys():
        # deal with master_raster_arr here
        _, ax = plt.subplots(1, 1)
        if ttls_in_time:
            ttls = np.array(flatten_list(ttls_per_field))
            ax.imshow(ttls, cmap=matplotlib.colormaps["bone"])
        spiking_arr = np.array(flatten_list(master_raster_arr))
        ax.imshow(spiking_arr, cmap=newcmap, alpha=0.6)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.hlines(np.cumsum(runs_per_field)[:-1], 0, max_run_len, "regressor")
        ax.set_xlim(0, max_run_len)
        ytick_locs = np.insert(np.cumsum(runs_per_field), 0, 0)
        ytick_locs = np.diff(ytick_locs) // 2 + ytick_locs[:-1]
        ax.set_yticks(ytick_locs, list(map(str, np.unique(run_field_id))))
        ax.set_ylabel("Field ID", rotation=90, labelpad=10)
        ax.set_xlabel("Time (s)")
        ax.set_xticks([0, max_run_len], ["0", f"{(max_run_len) / 50:.2f}"])
        axes2 = ax.twinx()
        axes2.set_yticks(ytick_locs, list(map(str, spikes_per_run)))
        axes2.set_ylim(ax.get_ylim())
        axes2.set_ylabel("Spikes per field", rotation=270, labelpad=10)
