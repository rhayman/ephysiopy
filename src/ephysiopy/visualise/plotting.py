import warnings
import copy
import matplotlib.pylab as plt
import matplotlib.transforms as transforms
from pycircstat2 import Circular
from pycircstat2.utils import rotate_data
from pycircstat2.descriptive import circ_mean_and_r
import numpy as np
from scipy.signal import hilbert
from matplotlib.patches import Rectangle
from matplotlib.collections import QuadMesh
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

from ephysiopy.axona import tintcolours as tcols
from ephysiopy.common.binning import RateMap
from ephysiopy.common.utils import (
    clean_kwargs,
    BinnedData,
    VariableToBin,
    rect,
    flatten_list,
)
from ephysiopy.common import fieldcalcs as fc
from ephysiopy.visualise.utils import (
    saveFigure,
    stripAxes,
    jet_cmap,
    grey_cmap,
    addClusterChannelToAxes,
    _add_colour_wheel,
    _plot_multiple_clusters,
    _plot_patch_collection,
    _plot_pcolormesh,
)


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
        if self.PosCalcs is not None:
            self.RateMap = RateMap(self.PosCalcs)
            self.npos = self.PosCalcs.xy.shape[1]

    @stripAxes
    def _plot_path(self, ax: plt.Axes) -> plt.Axes:
        ax.plot(
            self.PosCalcs.xy[0, :],
            self.PosCalcs.xy[1, :],
            color=tcols.colours[0],
            zorder=1,
        )
        return ax

    @saveFigure
    def plot_rate_map(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> plt.Figure:
        """
        Plots the rate map for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int or list
            The cluster(s) to get the rate map for.
        channel : int or list
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function.
            ax : plt.Axes, optional
                The axes to plot on. If None, new axes are created.
            separate_plots : bool, optional
                If True, each cluster will be plotted on a separate plot.
                Defaults to False.

        Returns
        -------
        plt.Axes
            The axes containing the rate map plot.
        """
        if "var_type" in kwargs.keys():
            if (
                kwargs["var_type"] == VariableToBin.X.value
                or kwargs["var_type"] == VariableToBin.Y.value
                or kwargs["var_type"] == VariableToBin.PHI.value
            ):
                return self.plot_linear_rate_map(cluster, channel, **kwargs)

        rmap = self.get_rate_map(cluster, channel, **kwargs)

        ax = kwargs.pop("ax", None)
        separate_plots = kwargs.pop("separate_plots", False)
        kwargs["cmap"] = kwargs.pop("cmap", jet_cmap)

        kwargs["equal_axes"] = kwargs.pop("equal_axes", True)
        # multiple clusters have been passed in so plot either in
        # one window  or one per cluster
        if len(rmap.binned_data) > 1 and separate_plots:
            for imap in rmap:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax = _plot_pcolormesh(imap, ax, **kwargs)
                ax.set_aspect("equal")
            return fig
        elif len(rmap.binned_data) > 1 and not separate_plots:
            return _plot_multiple_clusters(_plot_pcolormesh, rmap, **kwargs)

        # plot a single cluster in an individual window
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = plt.gcf()

        kwargs = clean_kwargs(plt.pcolormesh, kwargs)
        ax = _plot_pcolormesh(rmap, ax, **kwargs)
        ax.set_aspect("equal")
        return fig

    def plot_linear_rate_map(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> plt.Figure:
        """
        Plots the linear rate map for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int or list
            The cluster(s) to get the linear rate map for.
        channel : int or list
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes containing the linear rate map plot.
        """
        rmap = self.get_linear_rate_map(cluster, channel, **kwargs)

        def _plot_map(rmap: BinnedData, ax: plt.Axes, **kws) -> plt.Axes:
            """
            Plots a single rate map on the given axes.
            """
            label = kws.pop("label", None)
            kws = clean_kwargs(plt.plot, kws)
            # check if the data is masked as the ends of linear tracks
            # are frequently ignored
            mask = np.ma.getmask(rmap.binned_data[0])
            ax.plot(
                rmap.bin_edges[0][:-1][~mask],
                rmap.binned_data[0][~mask],
                label=label,
                **kws,
            )
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Rate (Hz)")
            ax.set_xlim(
                rmap.bin_edges[0][:-1][~mask][0], rmap.bin_edges[0][:-1][~mask][-1]
            )
            return ax

        ax = kwargs.pop("ax", None)
        separate_plots = kwargs.pop("separate_plots", False)

        kwargs["equal_axes"] = kwargs.pop("equal_axes", True)

        if len(rmap.binned_data) > 1 and separate_plots:
            for imap in rmap:
                fig = plt.figure()
                ax = fig.add_subplot(111, **kwargs)
                _plot_map(imap, ax, **kwargs)
            return ax
        elif len(rmap.binned_data) > 1 and not separate_plots:
            return _plot_multiple_clusters(_plot_map, rmap, **kwargs)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        return _plot_map(rmap, ax, **kwargs)

    @saveFigure
    def plot_hd_map(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Gets the head direction map for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int
            The cluster(s) to get the head direction map for.
        channel : int
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes containing the head direction map plot.

        Notes
        -----
        NB Following mathmatical convention, 0/360 degrees is
        3 o'clock, 90 degrees is 12 o'clock, 180 degrees is
        9 o'clock and 270 degrees

        """
        rmap = self.get_hd_map(cluster, channel, **kwargs)

        @stripAxes
        def _plot_single_map(rmap: BinnedData, ax: plt.Axes, **kwargs):
            fill = kwargs.pop("fill", False)
            add_guides = kwargs.pop("add_guides", False)
            add_mrv = kwargs.pop("add_mrv", False)
            col = kwargs.pop("c", None)

            ax.set_theta_zero_location("E")
            theta = np.deg2rad(rmap.bin_edges[0])
            r = rmap.binned_data[0]
            r = np.insert(r, -1, r[0])
            ax.plot(theta, r, color=col)
            if fill:
                ax.fill(theta, r, alpha=0.5)
            ax.set_aspect("equal")

            if add_guides:
                ax.set_rgrids([])

            hasData = np.any(r > 0)

            # See if we should add the mean resultant vector (mrv)
            if add_mrv and hasData:
                th, veclen = circ_mean_and_r(
                    np.deg2rad(rmap.bin_edges[0][:-1]), rmap.binned_data[0]
                )

                ax.plot(
                    [0, th],
                    [
                        0,
                        veclen * np.max(rmap.binned_data[0]),
                        # * self.PosCalcs.sample_rate,
                    ],
                    "r",
                )
            if "polar" in ax.name:
                ax.set_thetagrids([0, 90, 180, 270])

            return ax

        ax = kwargs.pop("ax", None)
        separate_plots = kwargs.pop("separate_plots", False)
        kwargs["projection"] = "polar"
        # multiple clusters have been passed in so plot either in
        # one window  or one per cluster
        if len(rmap.binned_data) > 1 and separate_plots:
            # kwargs = clean_kwargs(plt.pcolormesh, kwargs)
            for imap in rmap:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="polar", **kwargs)
                _plot_single_map(imap, ax, **kwargs)
            return ax
        elif len(rmap.binned_data) > 1 and not separate_plots:
            return _plot_multiple_clusters(_plot_single_map, rmap, **kwargs)

        # plot a single cluster in an individual window
        if ax is None:
            fig = plt.figure()
            kwargs = clean_kwargs(fig.add_subplot, kwargs)
            ax = fig.add_subplot(111, projection="polar", **kwargs)

        return _plot_single_map(rmap, ax, **kwargs)

    # @saveFigure
    def plot_spike_path(self, cluster=None, channel=None, **kws) -> plt.Axes:
        """
        Plots the spikes on the path for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int or None
            The cluster(s) to get the spike path for.
        channel : int or None
            The channel number.
        **kws : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes containing the spike path plot.

        """
        if not self.RateMap:
            self.initialise()
        ax = kws.pop("ax", None)
        separate_plots = kws.pop("separate_plots", False)
        save_as = kws.pop("save_as", None)
        # multiple clusters have been passed in so plot either in
        # one window  or one per cluster

        if cluster is not None or channel is not None:
            pos_idx = self._get_spike_pos_idx(cluster, channel)
            spike_locations = [self.PosCalcs.xy[:, idx] for idx in pos_idx]

            if len(spike_locations) > 1:
                kws["equal_axes"] = kws.pop("equal_axes", True)

                if separate_plots:
                    for idx in spike_locations:
                        if ax is None:
                            fig = plt.figure()
                            ax = fig.add_subplot(111)
                        else:
                            ax = self._plot_path(ax)
                            fig = plt.gcf()
                        _plot_patch_collection(idx, ax, **kws)
                    return ax
                else:
                    kws["func2"] = self._plot_path
                    return _plot_multiple_clusters(
                        _plot_patch_collection, spike_locations, **kws
                    )

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if cluster is None or channel is None:
            ax = self._plot_path(ax)
            return ax

        ax = self._plot_path(ax)
        spike_locations = self.PosCalcs.xy[:, pos_idx[0]]
        ax = _plot_patch_collection(spike_locations, ax, **kws)
        kws["save_as"] = save_as

        return ax

    def plot_eb_map(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Plots the ego-centric boundary map for the specified cluster(s) and
        channel.

        Parameters
        ----------
        cluster : int
            The cluster(s) to get the ego-centric boundary map for.
        channel : int
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes containing the ego-centric boundary map plot.
        """
        rmap = self.get_eb_map(cluster, channel, range=None, **kwargs)

        ax = kwargs.pop("ax", None)
        separate_plots = kwargs.pop("separate_plots", False)

        # multiple clusters have been passed in so plot either in
        # one window  or one per cluster
        if len(rmap.binned_data) > 1 and separate_plots:
            # kwargs = clean_kwargs(plt.pcolormesh, kwargs)
            for imap in rmap:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="polar", **kwargs)
                ax = _plot_pcolormesh(imap, ax, **kwargs)
            return fig
        elif len(rmap.binned_data) > 1 and not separate_plots:
            kwargs["projection"] = "polar"
            return _plot_multiple_clusters(_plot_pcolormesh, rmap, **kwargs)

        # plot a single cluster in an individual window
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="polar", **kwargs)

        kwargs = clean_kwargs(plt.pcolormesh, kwargs)
        ax = _plot_pcolormesh(rmap, ax, **kwargs)
        return fig

    @saveFigure
    def plot_eb_spikes(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Plots the ego-centric boundary spikes for the specified cluster(s)
        and channel.

        Parameters
        ----------
        cluster : int
            The cluster(s) to get the ego-centric boundary spikes for.
        channel : int
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes containing the ego-centric boundary spikes plot.
        """
        if not self.RateMap:
            self.initialise()

        ax = kwargs.pop("ax", None)
        separate_plots = kwargs.get("separate_plots", False)

        pos_idx = self._get_spike_pos_idx(cluster, channel)
        spike_locations = [self.PosCalcs.xy[:, idx] for idx in pos_idx]
        # Parse kwargs
        num_dir_bins = kwargs.get("dir_bins", 60)
        # TODO: add colour wheel?
        add_colour_wheel = kwargs.get("add_colour_wheel", False)
        dir_colours = np.array(sns.color_palette("hls", num_dir_bins))
        # Process dirrectional data into colours for pathces
        indices = [
            np.floor(self.RateMap.dir[idx] / (360 / num_dir_bins)).astype(int)
            for idx in pos_idx
        ]
        idx_of_dir_to_colour = [dir_colours[idx, :] for idx in indices]

        kwargs["zorder"] = 2

        if len(spike_locations):
            kwargs["equal_axes"] = kwargs.pop("equal_axes", True)
            kwargs["c"] = idx_of_dir_to_colour
            if separate_plots:
                for idx in spike_locations:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax = self._plot_path(ax)
                    _plot_patch_collection(idx, ax, **kwargs)
                return fig
            else:
                kwargs["func2"] = self._plot_path
                return _plot_multiple_clusters(
                    _plot_patch_collection, spike_locations, **kwargs
                )

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.set_aspect("equal")

        ax = self._plot_path(ax)
        ax = _plot_patch_collection(spike_locations[0], ax, **kwargs)

        if add_colour_wheel:
            ax = _add_colour_wheel(ax, fig)

        return fig

    @saveFigure
    def plot_sac(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Plots the spatial autocorrelation for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int
            The cluster(s) to get the spatial autocorrelation for.
        channel : int
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes containing the spatial autocorrelation plot.
        """

        @addClusterChannelToAxes
        @stripAxes
        def _plot_single_sac(sac: BinnedData, ax: plt.Axes, **kwargs) -> plt.Axes:
            kwargs["cmap"] = grey_cmap
            ax = _plot_pcolormesh(sac, ax, **kwargs)
            measures = fc.grid_field_props(sac)
            Am = copy.deepcopy(sac)
            Am.binned_data[0][~measures["dist_to_centre"]] = np.nan
            Am.binned_data[0] = np.ma.masked_invalid(np.atleast_2d(Am.binned_data[0]))
            kwargs["cmap"] = jet_cmap

            cmap = copy.copy(jet_cmap)
            cmap.set_bad("w", 0)

            ax = _plot_pcolormesh(Am, ax, **kwargs)
            _y = 0, 0
            _x = 0, sac.bin_edges[0][-1]
            ax.plot(_x, _y, c="g")
            mag = measures["scale"] * 0.75
            th = np.linspace(0, measures["orientation"], 50)

            [x, y] = rect(mag, th, deg=1)
            # angle subtended by orientation
            ax.plot(x, -y, c="r")
            # plot lines from centre to peaks above middle
            for p in measures["closest_peak_coords"]:
                if p[0] <= measures["dist_to_centre"].shape[0] / 2:
                    ax.plot((0, p[1]), (0, p[0]), "k")
            ax.invert_yaxis()
            all_ax = ax.axes
            all_ax.set_aspect("equal")

            return ax

        sac = self.get_grid_map(cluster, channel)
        ax = kwargs.pop("ax", None)
        separate_plots = kwargs.pop("separate_plots", False)

        if len(sac.binned_data) > 1 and separate_plots:
            for imap in sac:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax = _plot_single_sac(imap, ax, **kwargs)
            return fig
        elif len(sac.binned_data) > 1 and not separate_plots:
            return _plot_multiple_clusters(_plot_single_sac, sac, **kwargs)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax = _plot_single_sac(sac, ax, **kwargs)

        return ax
        # return fig

    @saveFigure
    def plot_speed_v_rate(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Plots the speed versus rate plot for the specified cluster(s) and
        channel.

        By default the distribution of speeds will be plotted as a twin
        axis. To disable set add_speed_hist = False

        Parameters
        ----------
        cluster : int
            The cluster(s) to get the speed versus rate plot for.
        channel : int
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes containing the speed versus rate plot.
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

    @saveFigure
    def plot_speed_v_hd(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Plots the speed versus head direction plot for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int
            The cluster(s) to get the speed versus head direction plot for.
        channel : int
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes containing the speed versus head direction plot.
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

    @saveFigure
    @addClusterChannelToAxes
    def plot_acorr(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Plots the autocorrelogram for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int
            The cluster(s) to get the autocorrelogram for.
        channel : int
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function, including:
            binsize : int, optional
                The size of the bins in ms.
                Gets passed to SpikeCalcsGeneric.xcorr().
                Defaults to 1.

        Returns
        -------
        plt.Axes
            The axes containing the autocorrelogram plot.
        """
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        strip_axes = kwargs.get("strip_axes", False)
        binsize = kwargs.get("binsize", 0.001)
        xrange = kwargs.get("Trange", [-0.5, 0.5])
        col = kwargs.get("c", "k")

        binned_data = self.get_acorr(cluster, channel, **kwargs)
        c = binned_data.binned_data[0]
        b = binned_data.bin_edges[0]
        ax.bar(b[:-1], c, width=binsize, color=col, align="edge", zorder=3)
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
        axtrans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.vlines(0, ymin=0, ymax=1, colors="lightgrey", transform=axtrans, zorder=1)

        return ax

    @saveFigure
    def plot_xcorr(
        self, cluster_a: int, channel_a: int, cluster_b: int, channel_b: int, **kwargs
    ) -> plt.Axes:
        """
        Plots the temporal cross-correlogram between cluster_a and cluster_b

        Parameters
        ----------
        cluster_a : int
            first cluster
        channel_a :int
            first channel
        cluster_b : int
            second cluster
        channel_b : int
            second channel

        Returns
        -------
        plt.Axes
            The axes containing the cross-correlogram plot
        """
        if "strip_axes" in kwargs.keys():
            strip_axes = kwargs.pop("strip_axes")
        else:
            strip_axes = False
        ax = kwargs.get("ax", None)
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

        xcorr_binned = self.get_xcorr(
            cluster_a, channel_a, cluster_b, channel_b, Trange=xrange, binsize=binsize
        )
        c = xcorr_binned.binned_data[0]
        b = xcorr_binned.bin_edges[0]
        ax.bar(b[:-1], c, width=binsize, color="k", align="edge", zorder=3)
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
        axtrans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.vlines(0, ymin=0, ymax=1, colors="lightgrey", transform=axtrans, zorder=1)
        return ax

    @saveFigure
    @addClusterChannelToAxes
    def plot_raster(self, cluster: int, channel: int, **kwargs) -> plt.Axes:
        """
        Plots the raster plot for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int
            The cluster(s) to get the raster plot for.
        channel : int
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function, including:
            dt : list
                The range in seconds to plot data over either side of the TTL pulse.
            seconds_per_bin : float
                The number of seconds per bin.

        Returns
        -------
        plt.Axes
            The axes containing the raster plot.
        """
        dt = kwargs.get("dt", [-0.05, 0.1])
        secs_per_bin = kwargs.get("secs_per_bin", 0.001)
        strip_axes = kwargs.pop("strip_axes", False)
        histColor = kwargs.pop("hist_colour", [1 / 255.0, 1 / 255.0, 1 / 255.0])
        x, y = self.get_psth(cluster, channel, **kwargs)
        ax = kwargs.get("ax", None)
        if y:
            if ax is None:
                fig = plt.figure(figsize=(4.0, 7.0))
                axScatter = fig.add_subplot(111)
            else:
                axScatter = ax
            axScatter.scatter(x, y, marker=".", s=2, rasterized=False, color=histColor)
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes(
                "top", 1.2, pad=0.2, sharex=axScatter, transform=axScatter.transAxes
            )
            scattTrans = transforms.blended_transform_factory(
                axScatter.transData, axScatter.transAxes
            )
            stim_pwidth = self.ttl_data["stim_duration"]
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
            ylabel_fs = xlabel_fs = 9
            labelpad = -8
            axScatter.set_ylabel(
                "Laser stimulation events", labelpad=labelpad - 10, fontsize=ylabel_fs
            )
            nStms = y[-1]
            axScatter.set_ylim(0, nStms)
            axScatter.set_yticks((0, nStms))
            axScatter.set_yticklabels(("0", str(nStms + 1)), fontsize=ylabel_fs - 1)
            axScatter.set_xlim(dt)
            axScatter.set_xlabel("Time to laser onset(s)", fontsize=xlabel_fs)
            axScatter.set_xticks((dt[0], 0, dt[1]))
            axScatter.set_xticklabels(
                (str(dt[0]), "0", str(dt[1])), fontsize=xlabel_fs - 1
            )

            h, be = np.histogram(
                x,
                bins=np.arange(dt[0], dt[1] + secs_per_bin, secs_per_bin),
                range=dt,
                density=False,
            )
            axHistx.bar(
                be[:-1],
                h,
                width=secs_per_bin,
                align="edge",
                color=histColor,
                edgecolor="none",
                rasterized=True,
            )
            plt.setp(axHistx.get_xticklabels(), visible=False)
            # Label only the min and max of the y-axis
            # max is rounded to the nearest 10
            maxRate = int(np.ceil(np.max(h) / 10.0) * 10)
            axHistx.set_ylim(0, maxRate)
            axHistx.set_yticks((0, maxRate))
            axHistx.set_yticklabels(("0", str(maxRate)), fontsize=ylabel_fs - 1)
            axHistx.set_xlim(dt)
            axHistx.set_ylabel("Firing rate(Hz)", labelpad=labelpad, fontsize=ylabel_fs)
            fig = plt.gcf()
            fig.canvas.manager.set_window_title(f"Cluster {cluster}")
            if strip_axes:
                return stripAxes(axScatter)
            return axHistx
        else:
            warnings.warn(
                f"PSTH for cluster {
                    cluster
                } is empty. The cell fired no spikes in the period under question"
            )
            return

        return ax

    @saveFigure
    def plot_power_spectrum(self, **kwargs) -> plt.Axes:
        """
        Plots the power spectrum.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to _getPowerSpectrumPlot
        """
        p = self.EEGCalcs.calcEEGPowerSpectrum()
        ax = self._getPowerSpectrumPlot(p[0], p[1], p[2], p[3], p[4], **kwargs)
        return ax

    @saveFigure
    def plot_theta_vs_running_speed(self, **kwargs) -> QuadMesh:
        """
        Plots theta frequency versus running speed.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the function, including:
            low_theta : float
                The lower bound of the theta frequency range (default is 6).
            high_theta : float
                The upper bound of the theta frequency range (default is 12).
            low_speed : float
                The lower bound of the running speed range (default is 2).
            high_speed : float
                The upper bound of the running speed range (default is 50).

        Returns
        -------
        QuadMesh
            The QuadMesh object containing the plot.
        """
        low_theta = kwargs.pop("low_theta", 6)
        high_theta = kwargs.pop("high_theta", 12)
        low_speed = kwargs.pop("low_speed", 2)
        high_speed = kwargs.pop("high_speed", 50)
        theta_filtered_eeg = self.EEGCalcs.butterFilter(low_theta, high_theta)
        hilbert_eeg = hilbert(theta_filtered_eeg)
        inst_freq = (
            self.EEGCalcs.fs / (2 * np.pi) * np.diff(np.unwrap(np.angle(hilbert_eeg)))
        )
        inst_freq = np.insert(inst_freq, -1, inst_freq[-1])
        eeg_times = np.arange(0, len(self.EEGCalcs.sig)) / self.EEGCalcs.fs
        pos_times = self.PosCalcs.xyTS
        idx = np.searchsorted(pos_times, eeg_times)
        idx[idx >= len(pos_times)] = len(pos_times) - 1
        eeg_speed = self.PosCalcs.speed[idx]
        h, edges = np.histogramdd(
            [inst_freq, eeg_speed],
            bins=(
                np.arange(low_theta, high_theta, 0.5),
                np.arange(low_speed, high_speed, 2),
            ),
        )
        hm = np.ma.masked_where(h == 0, h)
        ax = plt.pcolormesh(edges[1], edges[0], hm, cmap=jet_cmap, edgecolors="face")
        return ax

    @addClusterChannelToAxes
    # @saveFigure
    def plot_clusters_theta_phase(
        self, cluster: int, channel: int, **kwargs
    ) -> plt.Axes:
        """
        Plots the theta phase for the specified cluster and channel.

        Parameters
        ----------
        cluster : int
            The cluster to get the theta phase for.
        channel : int
            The channel number.
        **kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes containing the theta phase plot.
        """

        from ephysiopy.common.phasecoding import LFPOscillations

        ax = kwargs.pop("ax", None)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="polar")
        if "polar" not in ax.name:
            raise ValueError("Need a polar axis")
        L = LFPOscillations(self.EEGCalcs.sig, self.EEGCalcs.fs)
        ts = self.get_spike_times(cluster, channel)
        phase, x, y = L.get_theta_phase(ts, **kwargs)  # phase in radians

        phase = rotate_data(phase, np.pi)
        data = Circular(
            phase,
            unit="radian",
            kwargs_median={"method": "deviation", "average_method": "unique"},
        )
        data.plot(
            ax=ax,
            config={
                "axis": False,
                "spine": True,
                "zero_location": "E",
                "clockwise": 1,
                "rose": {"bins": 36},
                "mean": {"color": "r"},  # , "linestyle": "<->"},
                "median": True,
                "density": False,
                # "scatter": {"size": 20},
                # "radius": {"ticks": [0]},
            },
        )
        ax = plt.gca()
        ax.legend().remove()

        return ax

    def plot_phase_precession(
        self,
        cluster: int,
        channel: int,
        run_direction: str = "e",
        field_threshold: float = 1.0,
        field_threshold_percent: float = 150,
        min_run_speed: float = 0.5,
        track_end_size: float = 6.0,
        partition_method: str = "simple",
        return_pp: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """
        Plots the phase precession for the specified cluster and channel.

        Parameters
        ----------
        cluster : int
            The cluster to get the phase precession for.
        channel : int
            The channel number.
        run_direction : str
            Either 'e' or 'w' for east or westbound runs. Defaults to 'e'.
        field_threshold : float
            firing rates below this value in Hz will be considered outside
            the place field. Defaults to 1.0 Hz.
        field_threshold_percent : float
            firing rates below this percentage of the mean firing rate
            will be considered outside the place field.
            Defaults to 150.
        min_run_speed : float
            running speeds below this value in cm/s will be excluded from the
            analysis. Defaults to 0.5 cm/s.
        track_end_size : float
            the size of the track ends in cm. Defaults to 6.0 cm.
        partition_method : str
            the method to use for partitioning the data. Either 'simple' or
            'fancy'. Defaults to 'simple'.
        return_pp: bool
            whether to return the phase precession object. Defaults to False.
        **kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes containing the phase precession plot.

        Notes
        -----
        Assumes linear track data

        See Also
        --------
        ephysiopy.common.fieldcalcs.simple_partition
        ephysiopy.common.fieldcalcs.fancy_partition
        """

        from ephysiopy.phase_precession.linear_track import run_phase_analysis

        run_phase_analysis(
            self,
            cluster,
            channel,
            run_direction=run_direction,
            field_threshold=field_threshold,
            field_threshold_percent=field_threshold_percent,
            min_run_speed=min_run_speed,
            track_end_size=track_end_size,
            partition_method=partition_method,
            return_pp=return_pp,
            plot=True,
            **kwargs,
        )

        ax = plt.gca()
        return ax

    # @saveFigure
    # @stripAxes

    def plot_waveforms(self, cluster: int, channel: int, **kws) -> list[plt.Axes]:
        """
        Plot the waveforms for the selected cluster on the channel (tetrode)

        Parameters
        ----------
        cluster : int
            the cluster
        channel : int
            the channel(s) / tetrode

        Returns
        -------
        plt.Axes
            the axes holding the plot
        """
        # waves should be n_spikes x n_channels x n_samples
        # units are volts so x 1e6 to get microvolts
        waves = self.get_waveforms(cluster, channel) * 1e6
        n_spikes, n_channels, n_samples = waves.shape
        # if n_samples == 82:
        # waves = waves[:, :, 16:66]
        # n_samples = 50
        time = np.linspace(-200, 800, n_samples)

        axs = kws.get("ax", None)
        axes_labels = kws.get("axes_labels", False)

        if axs is None:
            fig, axs = plt.subplots(2, 2, sharey=True)
            axs = flatten_list(axs)
            mn_cols = ["r"] * 4
            alpha = 0.5
        else:
            axs = [axs, axs, axs, axs]
            mn_cols = sns.color_palette("colorblind", 4)
            alpha = 0

        col = [0.8627, 0.8627, 0.8627]

        max_wave = np.max(waves)
        min_wave = np.min(waves)

        mean_waves = np.mean(waves, 0)
        std_waves = np.std(waves, 0) * 3

        for i, ax in enumerate(axs):
            ax.plot(
                time,
                mean_waves[i] + std_waves[i],
                linestyle="-",
                color=col,
                linewidth=1,
                alpha=alpha,
            )
            ax.plot(
                time,
                mean_waves[i] - std_waves[i],
                linestyle="-",
                color=col,
                linewidth=1,
                alpha=alpha,
            )
            ax.plot(time, mean_waves[i], c=mn_cols[i], linewidth=2)
            if axes_labels:
                ax.set_xlim(-250, 850)
                ax.set_xticks([-200, 0, 800])
                ax.set_xticklabels(["-200μs", "0", "800μs"])
                ylim = ax.get_ylim()
                ax.set_yticks([ylim[0], 0, ylim[1]])
                ax.set_yticklabels([f"{int(ylim[0])}μV", "0", f"{int(ylim[1])}μV"])
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
            ax.set_ylim(min_wave, max_wave)

        return axs

    def _getPowerSpectrumPlot(
        self,
        freqs: np.ndarray,
        power: np.ndarray,
        sm_power: np.ndarray,
        band_max_power: float,
        freq_at_band_max_power: float,
        max_freq: int = 50,
        theta_range: tuple = [6, 12],
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Gets the power spectrum. The parameters can be obtained from
        calcEEGPowerSpectrum() in the EEGCalcsGeneric class.

        Parameters
        ----------
        freqs : np.ndarray
            The frequencies.
        power : np.ndarray
            The power values.
        sm_power : np.ndarray
            The smoothed power values.
        band_max_power : float
            The maximum power in the band.
        freq_at_band_max_power : float
            The frequency at which the maximum power in the band occurs.
        max_freq : int, optional
            The maximum frequency. Defaults to 50.
        theta_range : tuple, optional
            The theta range. Defaults to [6, 12].
        ax : plt.Axes, optional
            The axes to plot on. If None, new axes are created.
        **kwargs : dict
            Additional keyword arguments for the function.

        Returns
        -------
        plt.Axes
            The axes with the plot.
        """
        min_freq = kwargs.pop("min_freq", 0)
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
        ax.set_xlim(min_freq, max_freq)
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

    def plotSpectrogramByDepth(
        self,
        nchannels: int = 384,
        nseconds: int = 100,
        maxFreq: int = 125,
        channels: list = [],
        frequencies: list = [],
        frequencyIncrement: int = 1,
        **kwargs,
    ):
        """
        Plots a heat map spectrogram of the LFP for each channel.
        Line plots of power per frequency band and power on a subset of
        channels are also displayed to the right and above the main plot.

        Parameters
        ----------
        nchannels : int
            The number of channels on the probe.
        nseconds : int, optional
            How long in seconds from the start of the trial to do the spectrogram for (for speed).
            Default is 100.
        maxFreq : int
            The maximum frequency in Hz to plot the spectrogram out to. Maximum is 1250. Default is 125.
        channels : list
            The channels to plot separately on the top plot.
        frequencies : list
            The specific frequencies to examine across all channels. The mean from frequency:
            frequency+frequencyIncrement is calculated and plotted on the left hand side of the plot.
        frequencyIncrement : int
            The amount to add to each value of the frequencies list above.
        **kwargs : dict
            Additional keyword arguments for the function. Valid key value pairs:
                "saveas" - save the figure to this location, needs absolute path and filename.

        Notes
        -----
        Should also allow kwargs to specify exactly which channels and / or frequency bands to do the line plots for.
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
