import functools
import warnings

import matplotlib
import matplotlib.pylab as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.projections import get_projection_class

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

from ephysiopy.axona import tintcolours as tcols
from ephysiopy.common.spikecalcs import xcorr, SpikeCalcsGeneric
from ephysiopy.common.binning import RateMap
from ephysiopy.common.utils import clean_kwargs
from ephysiopy.common import fieldcalcs as fc


def stripAxes(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ax = func(*args, **kwargs)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if "polar" in ax.name:
            ax.set_rticks([])
        else:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
        return ax

    return wrapper


def coloured_line(x, y, c, ax, **lc_kwargs):
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


jet_cmap = matplotlib.colormaps["jet"]
grey_cmap = matplotlib.colormaps["gray_r"]


class FigureMaker(object):
    """
    A mixin class for TrialInterface that deals solely with
    producing graphical output.
    """

    def __init__(self):
        """
        Initializes the FigureMaker object.
        """
        self.PosCalcs = None

        """
        Initializes the FigureMaker object with data from PosCalcs.
        """
        self.RateMap = RateMap(self.PosCalcs)
        self.npos = self.PosCalcs.xy.shape[1]

    def _plot_multiple_clusters(
        self, func, clusters: list, channel: int, **kwargs
    ) -> matplotlib.figure.Figure:
        """
        Plots multiple clusters.

        Args:
            func (function): The function to apply to each cluster.
            clusters (list): The list of clusters to plot.
            channel (int): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """
        fig = plt.figure()
        nrows = int(np.ceil(len(clusters) / 5))
        if "projection" in kwargs.keys():
            proj = kwargs.pop("projection")
        else:
            proj = None
        for i, c in enumerate(clusters):
            ax = fig.add_subplot(nrows, 5, i + 1, projection=proj)
            ts = self.get_spike_times(c, channel)
            func(ts, ax=ax, **kwargs)
        return fig

    @stripAxes
    def plot_rate_map(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Plots the rate map for the specified cluster(s) and channel.

        Args:
            cluster (int): The cluster(s) to get the rate map for.
            channel (int): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """
        rmap = self.get_rate_map(cluster, channel, **kwargs)
        vmax = np.nanmax(np.ravel(rmap.binned_data[0]))
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        kwargs = clean_kwargs(plt.pcolormesh, kwargs)
        ax.pcolormesh(
            rmap.bin_edges[1],
            rmap.bin_edges[0],
            rmap.binned_data[0],
            cmap=jet_cmap,
            edgecolors="face",
            vmax=vmax,
            shading="auto",
            **kwargs
        )
        ax.set_aspect("equal")
        return ax

    @stripAxes
    def plot_hd_map(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Gets the head direction map for the specified cluster(s) and channel.

        Args:
            cluster (int): The cluster(s) to get the head direction map
                for.
            channel (int): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """
        rmap = self.get_hd_map(cluster, channel, **kwargs)
        add_mrv = kwargs.pop("add_mrv", False)
        add_guides = kwargs.pop("add_guides", False)
        strip_axes = kwargs.pop("strip_axes", False)
        fill = kwargs.pop("fill", False)
        ax = kwargs.pop("ax", None)
        kwargs = clean_kwargs(plt.pcolormesh, kwargs)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="polar", **kwargs)
        ax.set_theta_zero_location("N")
        # need to deal with the case where the axis is supplied but
        # is not polar. deal with polar first
        theta = np.deg2rad(rmap.bin_edges[0])
        ax.clear()
        r = (
            rmap.binned_data[0] * self.PosCalcs.sample_rate
        )  # in samples so * pos sample_rate
        r = np.insert(r, -1, r[0])
        hasData = np.any(r > 0)
        if "polar" in ax.name and hasData:
            ax.plot(theta, r)
            if fill:
                ax.fill(theta, r, alpha=0.5)
            ax.set_aspect("equal")

        if add_guides:
            ax.set_rgrids([])

        # See if we should add the mean resultant vector (mrv)
        if add_mrv and hasData:
            from ephysiopy.common.statscalcs import mean_resultant_vector

            veclen = fc.get_mean_resultant_length(rmap.binned_data[0])
            th = fc.get_mean_resultant_angle(rmap.binned_data[0])
            ax.plot(
                [0, th],
                [0, veclen * np.max(rmap.binned_data[0]) * self.PosCalcs.sample_rate],
                "r",
            )
        if "polar" in ax.name:
            ax.set_thetagrids([0, 90, 180, 270])
        if strip_axes:
            return stripAxes(ax)
        return ax

    @stripAxes
    def plot_spike_path(self, cluster=None, channel=None, **kwargs) -> plt.Axes:
        """
        Gets the spike path for the specified cluster(s) and channel.

        Args:
            cluster (int | None): The cluster(s) to get the spike path
                for.
            channel (int | None): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """
        if not self.RateMap:
            self.initialise()
        col = kwargs.pop("c", tcols.colours[1])
        ax = kwargs.pop("ax", None)
        marker = kwargs.pop("marker", "s")
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(
            self.PosCalcs.xy[0, :],
            self.PosCalcs.xy[1, :],
            color=tcols.colours[0],
            zorder=1,
        )
        ax.set_aspect("equal")
        if cluster is not None:
            idx = self._get_spike_pos_idx(cluster, channel)
            ax.scatter(
                self.PosCalcs.xy[0, idx],
                self.PosCalcs.xy[1, idx],
                marker=marker,
                color=col,
                **kwargs
            )
        return ax

    @stripAxes
    def plot_eb_map(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Gets the ego-centric boundary map for the specified cluster(s) and
        channel.

        Args:
            cluster (int): The cluster(s) to get the ego-centric
                boundary map for.
            channel (int): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """
        return_ratemap = kwargs.pop("return_ratemap", False)
        rmap = self.get_eb_map(cluster, channel, range=None, **kwargs)
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="polar")
        # sanitise kwargs before passing on to pcolormesh
        kwargs = clean_kwargs(plt.pcolormesh, kwargs)
        ax.pcolormesh(
            rmap.bin_edges[1],
            rmap.bin_edges[0],
            rmap.binned_data[0],
            edgecolors="face",
            shading="auto",
            **kwargs
        )
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
        # ax.set_xticklabels(np.arange(0, 2*np.pi, np.pi/4))
        ax.set_yticks(np.arange(0, 50, 10))
        ax.set_yticklabels(np.arange(0, 50, 10))
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Distance (cm)")
        if return_ratemap:
            return ax, rmap
        return ax

    @stripAxes
    def plot_eb_spikes(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Gets the ego-centric boundary spikes for the specified cluster(s)
        and channel.

        Args:
            cluster (int): The cluster(s) to get the ego-centric
                boundary spikes for.
            channel (int): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """
        if not self.RateMap:
            self.initialise()
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        # Parse kwargs
        num_dir_bins = kwargs.get("dir_bins", 60)
        rect_size = kwargs.get("ms", 1)
        add_colour_wheel = kwargs.get("add_colour_wheel", False)
        dir_colours = sns.color_palette("hls", num_dir_bins)
        # Process dirrectional data
        idx = self._get_spike_pos_idx(cluster, channel)
        dir_spike_fired_at = self.RateMap.dir[idx]
        idx_of_dir_to_colour = np.floor(
            dir_spike_fired_at / (360 / num_dir_bins)
        ).astype(int)
        rects = [
            Rectangle(
                self.RateMap.xy[:, i],
                width=rect_size,
                height=rect_size,
                clip_box=ax.bbox,
                facecolor=dir_colours[idx_of_dir_to_colour[i]],
                rasterized=True,
            )
            for i in range(len(idx))
        ]
        ax.plot(
            self.PosCalcs.xy[0],
            self.PosCalcs.xy[1],
            c=tcols.colours[0],
            zorder=1,
            alpha=0.3,
        )
        ax.add_collection(PatchCollection(rects, match_original=True))
        if add_colour_wheel:
            ax_col = inset_axes(
                ax,
                width="100%",
                height="100%",
                bbox_to_anchor=(0.75, 0.75, 0.15, 0.15),
                axes_class=get_projection_class("polar"),
                bbox_transform=fig.transFigure,
            )
            ax_col.set_theta_zero_location("N")
            theta = np.linspace(0, 2 * np.pi, 1000)
            phi = np.linspace(0, 1, 2)
            X, Y = np.meshgrid(phi, theta)
            norm = matplotlib.colors.Normalize(0, 2 * np.pi)
            col_map = sns.color_palette("hls", as_cmap=True)
            ax_col.pcolormesh(theta, phi, Y.T, norm=norm, cmap=col_map)
            ax_col.set_yticklabels([])
            ax_col.spines["polar"].set_visible(False)
            ax_col.set_thetagrids([0, 90])
        return ax

    @stripAxes
    def plot_sac(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Gets the spatial autocorrelation for the specified cluster(s) and
        channel.

        Args:
            cluster (int): The cluster(s) to get the spatial
                autocorrelation for.
            channel (int): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """
        sac = self.get_grid_map(cluster, channel)
        measures = fc.grid_field_props(sac)
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        Am = sac.binned_data[0].copy()
        Am[~measures["dist_to_centre"]] = np.nan
        Am = np.ma.masked_invalid(np.atleast_2d(Am))
        x, y = np.meshgrid(sac.bin_edges[1].data, sac.bin_edges[0].data)
        vmax = np.nanmax(np.ravel(sac.binned_data[0]))
        ax.pcolormesh(
            sac.bin_edges[1].data,
            sac.bin_edges[0].data,
            sac.binned_data[0],
            cmap=grey_cmap,
            edgecolors="face",
            vmax=vmax,
            shading="auto",
        )
        import copy

        cmap = copy.copy(jet_cmap)
        cmap.set_bad("w", 0)
        ax.pcolormesh(
            sac.bin_edges[1].data,
            sac.bin_edges[0].data,
            Am,
            cmap=cmap,
            edgecolors="face",
            vmax=vmax,
            shading="auto",
        )
        # horizontal green line at 3 o'clock
        _y = 0, 0
        _x = 0, sac.bin_edges[0][-1]
        ax.plot(_x, _y, c="g")
        mag = measures["scale"] * 0.75
        th = np.linspace(0, measures["orientation"], 50)
        from ephysiopy.common.utils import rect

        [x, y] = rect(mag, th, deg=1)
        # angle subtended by orientation
        ax.plot(x, -y, c="r", **kwargs)
        # plot lines from centre to peaks above middle
        for p in measures["closest_peak_coords"]:
            if p[0] <= measures["dist_to_centre"].shape[0] / 2:
                ax.plot((0, p[1]), (0, p[0]), "k", **kwargs)
        ax.invert_yaxis()
        all_ax = ax.axes
        all_ax.set_aspect("equal")

        return ax

    def plot_speed_v_rate(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Gets the speed versus rate plot for the specified cluster(s) and
        channel.

        By default the distribution of speeds will be plotted as a twin
        axis. To disable set add_speed_hist = False

        Args:
            cluster (int): The cluster(s) to get the speed versus rate
                plot for.
            channel (int): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """
        add_speed_hist = kwargs.pop("add_speed_hist", True)
        rmap = self.get_speed_v_rate_map(cluster, channel, **kwargs)
        # rmap is linear
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        kwargs = clean_kwargs(plt.plot, kwargs)
        ax_colour = "cornflowerblue"
        ax.plot(rmap.bin_edges[0][:-1], rmap.binned_data[0], color=ax_colour, **kwargs)
        ax.set_xlabel("Speed (cm/s)")
        ax.set_ylabel("Rate (Hz)")
        if add_speed_hist:
            ax.spines["left"].set_color(ax_colour)
            ax.tick_params(axis="y", colors=ax_colour)
            ax.yaxis.label.set_color(ax_colour)
            ax2 = ax.twinx()
            ax2_colour = "grey"
            pos_weights = np.ones_like(self.PosCalcs.speed) * (
                1 / self.PosCalcs.sample_rate
            )
            speed_bincounts = np.bincount(
                np.digitize(self.PosCalcs.speed, rmap.bin_edges[0], right=True),
                weights=pos_weights,
            )
            ax2.bar(
                rmap.bin_edges[0],
                speed_bincounts,
                # align="edge",
                alpha=0.5,
                width=np.mean(np.diff(rmap.bin_edges[0])),
                ec="grey",
                fc="grey",
            )
            ax2.set_ylabel("Duration (s)")
            ax2.spines["right"].set_color(ax2_colour)
            ax2.tick_params(axis="y", colors=ax2_colour)
            ax2.yaxis.label.set_color(ax2_colour)

        return ax

    # @stripAxes
    def plot_speed_v_hd(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Gets the speed versus head direction plot for the specified cluster(s)
        and channel.

        Args:
            cluster (int): The cluster(s) to get the speed versus head
                direction plot for.
            channel (int): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """
        rmap = self.get_speed_v_hd_map(cluster, channel, **kwargs)
        im = np.ma.MaskedArray(rmap.binned_data[0], np.isnan(rmap.binned_data[0]))
        # mask low rates...
        # im = np.ma.masked_where(im <= 1, im)
        # ... and where less than 0.5% of data is accounted for
        y, x = np.meshgrid(rmap.bin_edges[0], rmap.bin_edges[1], indexing="ij")
        vmax = np.nanmax(np.ravel(im))
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.pcolormesh(
            x, y, im, cmap=jet_cmap, edgecolors="face", vmax=vmax, shading="auto"
        )
        ax.set_xticks(
            [90, 180, 270], labels=["90", "180", "270"], fontweight="normal", size=6
        )
        ax.set_yticks(
            [10, 20, 30, 40],
            labels=["10", "20", "30", "40"],
            fontweight="normal",
            size=6,
        )
        ax.set_xlabel("Heading", fontweight="normal", size=6)
        return ax

    def plot_xcorr(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Gets the autocorrelogram for the specified cluster(s) and channel.

        Args:
            cluster (int): The cluster(s) to get the autocorrelogram
                for.
            channel (int): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """

        ts = self.get_spike_times(cluster, channel)
        ax = self._getXCorrPlot(ts, **kwargs)
        return ax

    def plot_raster(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Gets the raster plot for the specified cluster(s) and channel.

        Args:
            cluster (int | list): The cluster(s) to get the raster plot for.
            channel (int): The channel number.
            **kwargs: Additional keyword arguments for the function.
        """
        ts = self.get_spike_times(cluster, channel)
        ax = self._getRasterPlot(ts, cluster=cluster, **kwargs)
        return ax

    def plot_power_spectrum(self, **kwargs) -> plt.Axes:
        """
        Gets the power spectrum.

        Args:
            **kwargs: Additional keyword arguments for the function.
        """
        p = self.EEGCalcs.calcEEGPowerSpectrum()
        ax = self._getPowerSpectrumPlot(p[0], p[1], p[2], p[3], p[4], **kwargs)
        return ax

    @stripAxes
    def _plotWaves(self, waves: np.ndarray, ax: matplotlib.axes, **kwargs) -> plt.Axes:
        ax.plot(waves, c="k", **kwargs)
        return ax

    def _getPowerSpectrumPlot(
        self,
        freqs: np.array,
        power: np.array,
        sm_power: np.array,
        band_max_power: float,
        freq_at_band_max_power: float,
        max_freq: int = 50,
        theta_range: tuple = [6, 12],
        ax: matplotlib.axes = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plots the power spectrum. The parameters can be obtained from
        calcEEGPowerSpectrum() in the EEGCalcsGeneric class.

        Args:
            freqs (np.array): The frequencies.
            power (np.array): The power values.
            sm_power (np.array): The smoothed power values.
            band_max_power (float): The maximum power in the band.
            freq_at_band_max_power (float): The frequency at which the maximum
                power in the band occurs.
            max_freq (int, optional): The maximum frequency. Defaults to 50.
            theta_range (tuple, optional): The theta range.
                Defaults to [6, 12].
            ax (matplotlib.axes, optional): The axes to plot on. If None, new
                axes are created.
            **kwargs: Additional keyword arguments for the function.

        Returns:
            matplotlib.axes: The axes with the plot.
        """
        if "strip_axes" in kwargs.keys():
            strip_axes = kwargs.pop("strip_axes")
        else:
            strip_axes = False
        # downsample frequencies and power
        freqs = freqs[0::50]
        power = power[0::50]
        sm_power = sm_power[0::50]
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(freqs, power, alpha=0.5, color=[0.8627, 0.8627, 0.8627])
        ax.plot(freqs, sm_power)
        ax.set_xlim(0, max_freq)
        ylim = [0, np.max(sm_power[freqs < max_freq])]
        if "ylim" in kwargs:
            ylim = kwargs["ylim"]
        ax.set_ylim(ylim)
        ax.set_ylabel("Power")
        ax.set_xlabel("Frequency")
        ax.text(
            x=theta_range[1] / 0.9,
            y=band_max_power,
            s=str(freq_at_band_max_power)[0:4],
            fontsize=20,
        )
        from matplotlib.patches import Rectangle

        r = Rectangle(
            (theta_range[0], 0),
            width=np.diff(theta_range)[0],
            height=np.diff(ax.get_ylim())[0],
            alpha=0.25,
            color="r",
            ec="none",
        )
        ax.add_patch(r)
        if strip_axes:
            return stripAxes(ax)
        return ax

    def _getXCorrPlot(
        self, spk_times: np.array, ax: matplotlib.axes = None, **kwargs
    ) -> plt.Axes:
        """
        Returns an axis containing the autocorrelogram of the spike
        times provided over the range +/-500ms.

        Args:
            spk_times (np.array): Spike times in seconds.
            ax (matplotlib.axes, optional): The axes to plot into. If None,
                new axes are created.
            **kwargs: Additional keyword arguments for the function.
                binsize (int, optional): The size of the bins in ms. Gets
                passed to SpikeCalcsGeneric.xcorr(). Defaults to 1.

        Returns:
            matplotlib.axes: The axes with the plot.
        """
        if "strip_axes" in kwargs.keys():
            strip_axes = kwargs.pop("strip_axes")
        else:
            strip_axes = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if "binsize" in kwargs.keys():
            binsize = kwargs["binsize"]
        else:
            binsize = 0.001
        if "Trange" in kwargs.keys():
            xrange = kwargs.pop("Trange")
        else:
            xrange = [-0.5, 0.5]
        ac = xcorr(spk_times, Trange=xrange, **kwargs)
        c = ac.binned_data[0]
        b = ac.bin_edges[0]
        ax.bar(b[:-1], c, width=binsize, color="k", align="edge")
        ax.set_xlim(xrange)
        ax.set_xticks((xrange[0], 0, xrange[1]))
        ax.set_xticklabels("")
        ax.tick_params(
            axis="both", which="both", left=False, right=False, bottom=False, top=False
        )
        ax.set_yticklabels("")
        ax.xaxis.set_ticks_position("bottom")
        if strip_axes:
            return stripAxes(ax)
        return ax

    def _getRasterPlot(
        self,
        spk_times: np.ndarray,
        dt=(-0.05, 0.1),
        ax: matplotlib.axes = None,
        cluster=0,
        secs_per_bin: int = 0.001,
        **kwargs
    ) -> plt.Axes:
        """
        Plots a raster plot for a specified tetrode/ cluster.

        Args:
            spk_times (np.array): The spike times in samples.
            dt (tuple, optional): The window of time in ms to examine zeroed
                on the event of interest i.e. the first value will probably
                be negative as in the example. Defaults to (-50, 100).
            prc_max (float, optional): The proportion of firing the cell has
                to 'lose' to count as silent; a float between 0 and 1.
                Defaults to 0.5.
            ax (matplotlib.axes, optional): The axes to plot into.
                If not provided a new figure is created. Defaults to None.
            ms_per_bin (int, optional): The number of milliseconds in each bin
                of the raster plot. Defaults to 1.
            sample_rate (float, optional): The sample rate. Defaults to 3e4.
            **kwargs: Additional keyword arguments for the function.

        Returns:
            matplotlib.axes: The axes with the plot.
        """
        assert hasattr(self, "ttl_data")

        if "strip_axes" in kwargs.keys():
            strip_axes = kwargs.pop("strip_axes")
        else:
            strip_axes = False
        S = SpikeCalcsGeneric(spk_times, cluster=cluster)
        S.event_ts = self.ttl_data["ttl_timestamps"]
        x, y = S.psth()
        if ax is None:
            fig = plt.figure(figsize=(4.0, 7.0))
            axScatter = fig.add_subplot(111)
        else:
            axScatter = ax
        histColor = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        axScatter.scatter(x, y, marker=".", s=2, rasterized=False, color=histColor)
        divider = make_axes_locatable(axScatter)
        axScatter.set_xticks((dt[0], 0, dt[1]))
        axScatter.set_xticklabels((str(dt[0]), "0", str(dt[1])))
        axHistx = divider.append_axes(
            "top", 0.95, pad=0.2, sharex=axScatter, transform=axScatter.transAxes
        )
        scattTrans = transforms.blended_transform_factory(
            axScatter.transData, axScatter.transAxes
        )
        stim_pwidth = self.ttl_data["stim_duration"] / 1000
        if stim_pwidth is None:
            raise ValueError("stim duration is None")

        axScatter.add_patch(
            Rectangle(
                (0, 0),
                width=stim_pwidth,
                height=1,
                transform=scattTrans,
                color=[0, 0, 1],
                alpha=0.3,
            )
        )
        histTrans = transforms.blended_transform_factory(
            axHistx.transData, axHistx.transAxes
        )
        axHistx.add_patch(
            Rectangle(
                (0, 0),
                width=stim_pwidth,
                height=1,
                transform=histTrans,
                color=[0, 0, 1],
                alpha=0.3,
            )
        )
        axScatter.set_ylabel("Laser stimulation events", labelpad=-2.5)
        axScatter.set_xlabel("Time to stimulus onset(s)")
        nStms = y[-1]
        axScatter.set_ylim(0, nStms)
        # Label only the min and max of the y-axis
        ylabels = axScatter.get_yticklabels()
        for i in range(1, len(ylabels) - 1):
            ylabels[i].set_visible(False)
        yticks = axScatter.get_yticklines()
        for i in range(1, len(yticks) - 1):
            yticks[i].set_visible(False)

        axHistx.hist(
            x,
            bins=np.arange(dt[0], dt[1] + secs_per_bin, secs_per_bin),
            color=histColor,
            range=dt,
            rasterized=True,
            histtype="stepfilled",
        )
        axHistx.set_ylabel("Spike count", labelpad=-2.5)
        plt.setp(axHistx.get_xticklabels(), visible=False)
        # Label only the min and max of the y-axis
        ylabels = axHistx.get_yticklabels()
        for i in range(1, len(ylabels) - 1):
            ylabels[i].set_visible(False)
        yticks = axHistx.get_yticklines()
        for i in range(1, len(yticks) - 1):
            yticks[i].set_visible(False)
        axHistx.set_xlim(dt)
        axScatter.set_xlim(dt)
        if strip_axes:
            return stripAxes(axScatter)
        return axScatter

    def plotSpectrogramByDepth(
        self,
        nchannels: int = 384,
        nseconds: int = 100,
        maxFreq: int = 125,
        channels: list = [],
        frequencies: list = [],
        frequencyIncrement: int = 1,
        **kwargs
    ):
        """
        Plots a heat map spectrogram of the LFP for each channel.
        Line plots of power per frequency band and power on a subset of
        channels are also displayed to the right and above the main plot.

        Args:
            nchannels (int): The number of channels on the probe.
            nseconds (int, optional): How long in seconds from the start of
                the trial to do the spectrogram for (for speed).
                Default is 100.
            maxFreq (int): The maximum frequency in Hz to plot the spectrogram
                out to. Maximum is 1250. Default is 125.
            channels (list): The channels to plot separately on the top plot.
            frequencies (list): The specific frequencies to examine across
                all channels. The mean from frequency:
                frequency+frequencyIncrement is calculated and plotted on
                the left hand side of the plot.
            frequencyIncrement (int): The amount to add to each value of
                the frequencies list above.
            **kwargs: Additional keyword arguments for the function.
                Valid key value pairs:
                    "saveas" - save the figure to this location, needs absolute
                    path and filename.

        Notes:
            Should also allow kwargs to specify exactly which channels
            and / or frequency bands to do the line plots for.
        """
        if not self.path2LFPdata:
            raise TypeError("Not a probe recording so not plotting")
        import os

        lfp_file = os.path.join(self.path2LFPdata, "continuous.dat")
        status = os.stat(lfp_file)
        nsamples = int(status.st_size / 2 / nchannels)
        mmap = np.memmap(lfp_file, np.int16, "r", 0, (nchannels, nsamples), order="F")
        # Load the channel map NB assumes this is in the AP data
        # location and that kilosort was run there
        channel_map = np.squeeze(
            np.load(os.path.join(self.path2APdata, "channel_map.npy"))
        )
        lfp_sample_rate = 2500
        data = np.array(mmap[channel_map, 0 : nseconds * lfp_sample_rate])
        from ephysiopy.common.ephys_generic import EEGCalcsGeneric

        E = EEGCalcsGeneric(data[0, :], lfp_sample_rate)
        E.calcEEGPowerSpectrum()
        spec_data = np.zeros(shape=(data.shape[0], len(E.sm_power[0::50])))
        for chan in range(data.shape[0]):
            E = EEGCalcsGeneric(data[chan, :], lfp_sample_rate)
            E.calcEEGPowerSpectrum()
            spec_data[chan, :] = E.sm_power[0::50]

        x, y = np.meshgrid(E.freqs[0::50], channel_map)
        import matplotlib.colors as colors
        from matplotlib.pyplot import cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        _, spectoAx = plt.subplots()
        spectoAx.pcolormesh(
            x, y, spec_data, edgecolors="face", cmap="bone", norm=colors.LogNorm()
        )
        spectoAx.set_xlim(0, maxFreq)
        spectoAx.set_ylim(channel_map[0], channel_map[-1])
        spectoAx.set_xlabel("Frequency (Hz)")
        spectoAx.set_ylabel("Channel")
        divider = make_axes_locatable(spectoAx)
        channel_spectoAx = divider.append_axes("top", 1.2, pad=0.1, sharex=spectoAx)
        meanfreq_powerAx = divider.append_axes("right", 1.2, pad=0.1, sharey=spectoAx)
        plt.setp(
            channel_spectoAx.get_xticklabels() + meanfreq_powerAx.get_yticklabels(),
            visible=False,
        )

        # plot mean power across some channels
        mn_power = np.mean(spec_data, 0)
        if not channels:
            channels = range(1, nchannels, 60)
        cols = iter(cm.rainbow(np.linspace(0, 1, len(channels))))
        for chan in channels:
            c = next(cols)
            channel_spectoAx.plot(
                E.freqs[0::50],
                10 * np.log10(spec_data[chan, :] / mn_power),
                c=c,
                label=str(chan),
            )

        channel_spectoAx.set_ylabel("Channel power(dB)")
        channel_spectoAx.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            mode="expand",
            fontsize="x-small",
            ncol=4,
        )

        # plot mean frequencies across all channels
        if not frequencyIncrement:
            freq_inc = 6
        else:
            freq_inc = frequencyIncrement
        if not frequencies:
            lower_freqs = np.arange(1, maxFreq - freq_inc, freq_inc)
        else:
            lower_freqs = frequencies
        upper_freqs = [f + freq_inc for f in lower_freqs]
        cols = iter(cm.nipy_spectral(np.linspace(0, 1, len(upper_freqs))))
        mn_power = np.mean(spec_data, 1)
        for freqs in zip(lower_freqs, upper_freqs):
            freq_mask = np.logical_and(
                E.freqs[0::50] > freqs[0], E.freqs[0::50] < freqs[1]
            )
            mean_power = 10 * np.log10(np.mean(spec_data[:, freq_mask], 1) / mn_power)
            c = next(cols)
            meanfreq_powerAx.plot(
                mean_power,
                channel_map,
                c=c,
                label=str(freqs[0]) + " - " + str(freqs[1]),
            )
        meanfreq_powerAx.set_xlabel("Mean freq. band power(dB)")
        meanfreq_powerAx.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            mode="expand",
            fontsize="x-small",
            ncol=1,
        )
        if "saveas" in kwargs:
            saveas = kwargs["saveas"]
            plt.savefig(saveas)
        plt.show()
