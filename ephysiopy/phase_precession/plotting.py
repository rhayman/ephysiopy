# TODO: these plotting fncs needed finishing
import copy
import matplotlib
import numpy as np
from scipy import ndimage
from ephysiopy.common.utils import flatten_list, BinnedData, repeat_ind
from ephysiopy.common.utils import bwperim
from ephysiopy.visualise.plotting import stripAxes
from ephysiopy.visualise.plotting import colored_line
from ephysiopy.io.recording import AxonaTrial
from ephysiopy.common.fieldcalcs import (
    FieldProps,
    RunProps,
    infill_ratemap,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import RegularPolyCollection


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
        ax.plot((-1, 1), (-slope + intercept + m,
                slope + intercept + m), "r", lw=3)
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

    spk_in_pos = trial.get_binned_spike_times(cluster, channel)
    time = np.arange(0, trial.PosCalcs.duration,
                     1 / trial.PosCalcs.sample_rate)

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
        ax.scatter(np.array(runs_pos)[idx],
                   np.array(runs_phase)[idx], **kwargs)

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
                r.lfp_segment.slice.start / r.lfp_segment.sample_rate,
                r.lfp_segment.slice.stop / r.lfp_segment.sample_rate,
                len(r.lfp_segment.phase),
            )
            plt.plot(t, r.lfp_segment.filtered_signal, label=f"Run {r.label}")
            y = np.interp(r.lfp_segment.spike_times, t,
                          r.lfp_segment.filtered_signal)
            plt.plot(r.lfp_segment.spike_times, y, "ro")
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
            ax.plot(r.xy[0], r.xy[1], color=cmap(
                f.label - 1), label=f.label - 1)
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
    [ax.plot(f.xy_at_peak[0], f.xy_at_peak[1], "ko", ms=2)
     for f in field_props]
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
            matplotlib.patches.Circle(
                (0, 0), 1, fc="none", ec="lightgrey", zorder=3),
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
    [ax2.plot(f.xy_at_peak[0], f.xy_at_peak[1], "ko", ms=2)
     for f in field_props]
    # PLOT 4
    # The smoothed ratemap - maybe make this the first sub plot
    ax3 = subfigs[1, 1].subplots(1, 1)
    # smooth the ratemap a bunch
    rmap_to_plot = copy.copy(field_props[0]._intensity_image)
    rmap_to_plot = infill_ratemap(rmap_to_plot)
    ax3.pcolormesh(bin_edges[1], bin_edges[0], rmap_to_plot)
    # add the field labels to the ratemap plot
    [
        ax3.text(f.xy_at_peak[0], f.xy_at_peak[1],
                 str(f.label), ha="left", va="bottom")
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
    assert hasattr(field.runs[0], "lfp_data")

    n_rows = 3
    n_cols = np.ceil(len(field.runs) / n_rows).astype(int)
    fig = plt.figure()
    subfigs = fig.subfigures(
        1,
        1,
    )

    ax = np.ravel(subfigs.subplots(n_rows, n_cols))
    for i_run, run in enumerate(field.runs):
        sig = run.lfp_data.filtered_signal
        t = np.linspace(
            run.lfp_data.slice.start / lfp_sample_rate,
            run.lfp_data.slice.stop / lfp_sample_rate,
            len(sig),
        )
        ax[i_run].plot(t, sig)
        spike_phase_pos = np.interp(
            run.lfp_data.spike_times, t, run.lfp_data.filtered_signal
        )
        ax[i_run].plot(run.lfp_data.spike_times, spike_phase_pos, "ro")
    plt.show()


def plot_lfp_run(
    run: RunProps, cycle_labels: np.ndarray = None, lfp_sample_rate: int = 250
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
    from ephysiopy.visualise.plotting import colored_line

    assert hasattr(run, "lfp_data")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # create a colormap for the cycle labels
    if cycle_labels is not None:
        colours = cycle_labels[run.lfp_data.slice].data
    sig = run.lfp_data.amplitude
    t = np.linspace(
        run.lfp_data.slice.start / lfp_sample_rate,
        run.lfp_data.slice.stop / lfp_sample_rate,
        len(sig),
    )
    if cycle_labels is None:
        ax.plot(t, sig)
    else:
        colored_line(t, sig, colours, ax=ax, cmap="tab10")
    inds = repeat_ind(run.lfp_data.spike_count)
    spike_amp = np.take(sig, inds)
    spike_times = np.take(t, inds)
    ax.plot(spike_times, spike_amp, "ro")
    plt.show()
