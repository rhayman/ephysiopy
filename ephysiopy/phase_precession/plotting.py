# TODO: these plotting fncs needed finishing

from ephysiopy.common.utils import (
    flatten_list,
    BinnedData,
)

from ephysiopy.visualise.plotting import colored_line
from ephysiopy.io.recording import AxonaTrial
from ephysiopy.common.fieldcalcs import (
    FieldProps,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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

    spk_in_pos = trial.get_spike_times_binned_into_position(cluster, channel)
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
