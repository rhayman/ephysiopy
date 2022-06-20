import matplotlib.pylab as plt
import matplotlib
import numpy as np
import functools
from ephysiopy.common.binning import RateMap
from ephysiopy.common.spikecalcs import SpikeCalcsGeneric
from ephysiopy.dacq2py import tintcolours as tcols
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle


# Decorators
def stripAxes(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ax = func(*args, **kwargs)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if 'polar' in ax.name:
            ax.set_rticks([])
        else:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        return ax
    return wrapper


class FigureMaker(object):
    '''
    A mixin class for dacq2py_util and OEKiloPhy that deals solely with
    producing graphical output
    '''
    def __init__(self):
        self.data_loaded = False

    def initialise(self):
        xy = getattr(self, 'xy', None)
        self.npos = xy.shape[1]
        hdir = getattr(self, 'dir', None)
        speed = getattr(self, 'speed', None)
        pos_weights = None
        if hdir is not None:
            if np.ma.is_masked(hdir):
                pos_weights = np.array(~hdir.mask).astype(int)
            else:
                pos_weights = np.ones_like(hdir)
        ppm = getattr(self, 'ppm', 300)
        cmsPerBin = getattr(self, 'cmsPerBin', 3)

        self.RateMapMaker = RateMap(
            xy=xy, hdir=hdir, speed=speed, pos_weights=pos_weights, ppm=ppm,
            xyInCms=True, cmsPerBin=cmsPerBin)
        self.RateMapMaker.x_lims = getattr(self, 'x_lims', None)  # 2-tuple
        self.RateMapMaker.y_lims = getattr(self, 'y_lims', None)
        self.data_loaded = True

    def getSpikePosIndices(self, spk_times: np.array):
        pos_times = getattr(self, 'xyTS')
        idx = np.searchsorted(pos_times, spk_times)
        idx[idx == len(pos_times)] = idx[idx == len(pos_times)] - 1
        return idx

    def makeSummaryPlot(self, spk_times: np.array):
        fig = plt.figure()
        ax = plt.subplot(221)
        self.makeSpikePathPlot(spk_times, ax=ax, markersize=2)
        ax = plt.subplot(222)
        self.makeRateMap(spk_times, ax=ax)
        ax = plt.subplot(223, projection='polar')
        self.makeHDPlot(spk_times, ax=ax)
        ax = plt.subplot(224)
        try:
            self.makeSAC(spk_times, ax=ax)
        except IndexError:
            pass
        return fig

    @stripAxes
    def makeRateMap(self, spk_times: np.array,
                    ax: matplotlib.axes = None) -> matplotlib.axes:
        self.initialise()
        spk_times_in_pos_samples = self.getSpikePosIndices(spk_times)
        spk_weights = np.bincount(
            spk_times_in_pos_samples, minlength=self.npos)
        rmap = self.RateMapMaker.getMap(spk_weights)
        ratemap = np.ma.MaskedArray(rmap[0], np.isnan(rmap[0]), copy=True)
        x, y = np.meshgrid(rmap[1][1][0:-1], rmap[1][0][0:-1])
        vmax = np.nanmax(np.ravel(ratemap))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.pcolormesh(
            x, y, ratemap, cmap=plt.cm.get_cmap("jet"), edgecolors='face',
            vmax=vmax, shading='auto')
        ax.set_aspect('equal')
        return ax

    @stripAxes
    def makeSpikePathPlot(self, spk_times: np.array = None,
                          ax: matplotlib.axes = None,
                          **kwargs) -> matplotlib.axes:
        self.initialise()
        if 'c' in kwargs:
            col = kwargs.pop('c')
        else:
            col = tcols.colours[1]
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.xy[0, :], self.xy[1, :], c=tcols.colours[0], zorder=1)
        ax.set_aspect('equal')
        if spk_times is not None:
            idx = self.getSpikePosIndices(spk_times)
            ax.plot(
                self.xy[0, idx], self.xy[1, idx], 's', c=col, **kwargs)
        return ax

    @stripAxes
    def makeSAC(self, spk_times: np.array = None,
                ax: matplotlib.axes = None, **kwargs) -> matplotlib.axes:
        self.initialise()
        spk_times_in_pos_samples = self.getSpikePosIndices(spk_times)
        spk_weights = np.bincount(
            spk_times_in_pos_samples, minlength=self.npos)
        rmap = self.RateMapMaker.getMap(spk_weights)
        from ephysiopy.common import gridcell
        S = gridcell.SAC()
        nodwell = ~np.isfinite(rmap[0])
        sac = S.autoCorr2D(rmap[0], nodwell)
        measures = S.getMeasures(sac)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax = self.show_SAC(sac, measures, ax)
        return ax

    @stripAxes
    def makeHDPlot(self, spk_times: np.array = None,
                   ax: matplotlib.axes = None, **kwargs) -> matplotlib.axes:
        self.initialise()
        spk_times_in_pos_samples = self.getSpikePosIndices(spk_times)
        spk_weights = np.bincount(
            spk_times_in_pos_samples, minlength=self.npos)
        rmap = self.RateMapMaker.getMap(spk_weights, 'dir')
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')
        # need to deal with the case where the axis is supplied but
        # is not polar. deal with polar first
        theta = np.deg2rad(rmap[1][0])
        ax.clear()
        r = rmap[0]
        r = np.insert(r, -1, r[0])
        if 'polar' in ax.name:
            ax.plot(theta, r)
            if 'fill' in kwargs:
                ax.fill(theta, r, alpha=0.5)
            ax.set_aspect('equal')
        else:
            pass

        # See if we should add the mean resultant vector (mrv)
        if 'add_mrv' in kwargs:
            from ephysiopy.common.statscalcs import mean_resultant_vector
            angles = self.dir[spk_times_in_pos_samples]
            r, th = mean_resultant_vector(np.deg2rad(angles))
            ax.plot([th, th], [0, r*np.max(rmap[0])], 'r')
        if 'polar' in ax.name:
            ax.set_thetagrids([0, 90, 180, 270])
        return ax

    @stripAxes
    def makeSpeedVsRatePlot(
            self, spk_times: np.array, minSpeed: float = 0.0,
            maxSpeed: float = 40.0, sigma: float = 3.0,
            ax: matplotlib.axes = None, **kwargs) -> matplotlib.axes:
        """
        Plots the instantaneous firing rate of a cell against running speed
        Also outputs a couple of measures as with Kropff et al., 2015; the
        Pearsons correlation and the depth of modulation (dom) - see below for
        details
        """
        self.initialise()
        spk_times_in_pos_samples = self.getSpikePosIndices(spk_times)

        speed = np.ravel(self.speed)
        if np.nanmax(speed) < maxSpeed:
            maxSpeed = np.nanmax(speed)
        spd_bins = np.arange(minSpeed, maxSpeed, 1.0)
        # Construct the mask
        speed_filt = np.ma.MaskedArray(speed)
        speed_filt = np.ma.masked_where(speed_filt < minSpeed, speed_filt)
        speed_filt = np.ma.masked_where(speed_filt > maxSpeed, speed_filt)
        from ephysiopy.common.spikecalcs import SpikeCalcsGeneric

        x1 = spk_times_in_pos_samples
        S = SpikeCalcsGeneric(x1)
        spk_sm = S.smoothSpikePosCount(x1, self.xyTS.shape[0], sigma, None)
        spk_sm = np.ma.MaskedArray(spk_sm, mask=np.ma.getmask(speed_filt))
        spd_dig = np.digitize(speed_filt, spd_bins, right=True)
        mn_rate = np.array([np.ma.mean(
            spk_sm[spd_dig == i]) for i in range(0, len(spd_bins))])
        var = np.array([np.ma.std(
            spk_sm[spd_dig == i]) for i in range(0, len(spd_bins))])
        np.array([np.ma.sum(
            spk_sm[spd_dig == i]) for i in range(0, len(spd_bins))])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(
            spd_bins, mn_rate * self.pos_sample_rate,
            yerr=var, color='k')
        ax.set_xlim(spd_bins[0], spd_bins[-1])
        plt.xticks([spd_bins[0], spd_bins[-1]], ['0', '{:.2g}'.format(
            spd_bins[-1])], fontweight='normal', size=6)
        plt.yticks([0, np.nanmax(
            mn_rate)*self.pos_sample_rate], ['0', '{:.2f}'.format(
                np.nanmax(mn_rate))], fontweight='normal', size=6)
        return ax

    @stripAxes
    def makeSpeedVsHeadDirectionPlot(self, spk_times: np.array,
                                     ax: matplotlib.axes = None,
                                     **kwargs) -> matplotlib.axes:
        self.initialise()
        spk_times_in_pos_samples = self.getSpikePosIndices(spk_times)
        idx = np.array(spk_times_in_pos_samples, dtype=int)
        if np.ma.is_masked(self.speed):
            w = self.speed.mask
            w = np.array(~w, dtype=int)
        else:
            w = np.bincount(idx, minlength=self.speed.shape[0])
        dir_bins = np.arange(0, 360, 6)
        spd_bins = np.arange(0, 30, 1)
        h = np.histogram2d(
            self.dir, self.speed, [dir_bins, spd_bins], weights=w)
        from ephysiopy.common.utils import blurImage
        im = blurImage(h[0], 5, ftype='gaussian')
        im = np.ma.MaskedArray(im)
        # mask low rates...
        im = np.ma.masked_where(im <= 1, im)
        # ... and where less than 0.5% of data is accounted for
        x, y = np.meshgrid(dir_bins, spd_bins)
        vmax = np.max(np.ravel(im))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.pcolormesh(x, y, im.T, cmap=plt.cm.get_cmap("jet"),
                      edgecolors='face',
                      vmax=vmax, shading='auto')
        plt.xticks([90, 180, 270], fontweight='normal', size=6)
        plt.yticks([10, 20], fontweight='normal', size=6)
        return ax

    def makePowerSpectrum(
            self, freqs: np.array, power: np.array, sm_power: np.array,
            band_max_power: float, freq_at_band_max_power: float,
            max_freq: int = 50, theta_range: tuple = [6, 12],
            ax: matplotlib.axes = None, **kwargs) -> matplotlib.axes:
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
        ylim = [0, band_max_power / 0.8]
        if 'ylim' in kwargs:
            ylim = kwargs['ylim']
        ax.set_ylim(ylim)
        ax.set_ylabel('Power')
        ax.set_xlabel('Frequency')
        ax.text(
            x=theta_range[1] / 0.9, y=band_max_power,
            s=str(freq_at_band_max_power)[0:4], fontsize=20)
        from matplotlib.patches import Rectangle
        r = Rectangle((
            theta_range[0], 0), width=np.diff(theta_range)[0],
            height=np.diff(ax.get_ylim())[0], alpha=0.25, color='r', ec='none')
        ax.add_patch(r)
        return ax

    def makeXCorr(self, spk_times: np.array, ax: matplotlib.axes = None,
                  **kwargs) -> matplotlib.axes:
        # spk_times in samples provided in seconds but convert to
        # ms for a more display friendly scale
        spk_times = spk_times / 3e4 * 1000.
        S = SpikeCalcsGeneric(spk_times)
        y = S.xcorr(spk_times)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.hist(
                y[y != 0], bins=201, range=[-500, 500],
                color='k', histtype='stepfilled')
        ax.set_xlim(-500, 500)
        ax.set_xticks((-500, 0, 500))
        ax.set_xticklabels('')
        ax.tick_params(
            axis='both', which='both', left=False, right=False,
            bottom=False, top=False)
        ax.set_yticklabels('')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        return ax

    def makeRaster(self, spk_times: np.array, dt=(-50, 100),
                   prc_max: float = 0.5, ax: matplotlib.axes = None,
                   ms_per_bin: int = 1, histtype: str = 'count',
                   **kwargs) -> matplotlib.axes:
        """
        Plots a raster plot for a specified tetrode/ cluster

        Parameters
        ----------
        spk_times: np.array
            The spike times in samples
        dt : 2-tuple
            the window of time in ms to examine zeroed on the event of interest
            i.e. the first value will probably be negative as in the  example
        prc_max : float
            the proportion of firing the cell has to 'lose' to count as
            silent; a float between 0 and 1
        ax - matplotlib.axes
            the axes to plot into. If not provided a new figure is created
        ms_per_bin : int
            The number of milliseconds in each bin of the raster plot
        histtype : str
            either 'count' or 'rate' - the resulting histogram plotted above
            the raster plot will consist of either the counts of spikes in
            ms_per_bin or the mean rate in ms_per_bin
        """
        x1 = spk_times / 3e4 * 1000.  # get into ms
        x1.sort()
        on_good = getattr(self, 'ttl_timestamps')
        dt = np.array(dt)
        irange = on_good[:, np.newaxis] + dt[np.newaxis, :]
        dts = np.searchsorted(x1, irange)
        y = []
        x = []
        for i, t in enumerate(dts):
            tmp = x1[t[0]:t[1]] - on_good[i]
            x.extend(tmp)
            y.extend(np.repeat(i, len(tmp)))
        if ax is None:
            fig = plt.figure(figsize=(4.0, 7.0))
            axScatter = fig.add_subplot(111)
        else:
            axScatter = ax
        axScatter.scatter(x, y, marker='.', s=2, rasterized=False)
        divider = make_axes_locatable(axScatter)
        axScatter.set_xticks((dt[0], 0, dt[1]))
        axScatter.set_xticklabels((str(dt[0]), '0', str(dt[1])))
        axHistx = divider.append_axes("top", 0.95, pad=0.2, sharex=axScatter,
                                      transform=axScatter.transAxes)
        scattTrans = transforms.blended_transform_factory(axScatter.transData,
                                                          axScatter.transAxes)
        stim_pwidth = int(self.settings['stim_pwidth'])
        axScatter.add_patch(
            Rectangle(
                (0, 0), width=stim_pwidth/1000., height=1,
                transform=scattTrans,
                color=[0, 0, 1], alpha=0.5))
        histTrans = transforms.blended_transform_factory(axHistx.transData,
                                                         axHistx.transAxes)
        axHistx.add_patch(Rectangle((0, 0), width=stim_pwidth/1000., height=1,
                          transform=histTrans,
                          color=[0, 0, 1], alpha=0.5))
        axScatter.set_ylabel('Laser stimulation events', labelpad=-18.5)
        axScatter.set_xlabel('Time to stimulus onset(ms)')
        nStms = int(self.STM['num_stm_samples'])
        axScatter.set_ylim(0, nStms)
        # Label only the min and max of the y-axis
        ylabels = axScatter.get_yticklabels()
        for i in range(1, len(ylabels)-1):
            ylabels[i].set_visible(False)
        yticks = axScatter.get_yticklines()
        for i in range(1, len(yticks)-1):
            yticks[i].set_visible(False)

        histColor = [192/255.0, 192/255.0, 192/255.0]
        axHistx.hist(
            x, bins=np.arange(dt[0], dt[1] + ms_per_bin, ms_per_bin),
            color=histColor, alpha=0.6, range=dt, rasterized=True,
            histtype='stepfilled')
        if 'rate' in histtype:
            axHistx.set_ylabel('Rate')
            # mn_rate_pre_stim = np.mean(vals[bins[1:] < 0])
            # idx = np.logical_and(bins[1:] > 0, bins[1:] < 10).nonzero()[0]
            # mn_rate_post_stim = np.mean(vals[idx])
            # above_half_idx = idx[(
            # vals[idx] < mn_rate_pre_stim * prc_max).nonzero()[0]]
            # half_pre_rate_ms = bins[above_half_idx[0]]
            # print('\ntime to {0}% of pre-stimulus rate = {1}ms'.format(*(
            # prc_max * 100, half_pre_rate_ms)))
            # print('mean pre-laser rate = {0}Hz'.format(mn_rate_pre_stim))
            # print('mean 10ms post-laser rate = {0}'.format(
            # mn_rate_post_stim))
        else:
            axHistx.set_ylabel('Spike count', labelpad=-2.5)
        plt.setp(axHistx.get_xticklabels(),
                 visible=False)
        # Label only the min and max of the y-axis
        ylabels = axHistx.get_yticklabels()
        for i in range(1, len(ylabels)-1):
            ylabels[i].set_visible(False)
        yticks = axHistx.get_yticklines()
        for i in range(1, len(yticks)-1):
            yticks[i].set_visible(False)
        axHistx.set_xlim(dt)
        axScatter.set_xlim(dt)
        return axScatter

    '''
    def getRasterHist(
            self, spike_ts: np.array,
            sample_rate: int,
            dt=(-50, 100), hist=True):
        """
        MOVE TO SPIKECALCS

        Calculates the histogram of the raster of spikes during a series of
        events

        Parameters
        ----------
        tetrode : int
        cluster : int
        dt : tuple
            the window of time in ms to examine zeroed on the event of interest
            i.e. the first value will probably be negative as in the example
        hist : bool
            not sure
        """
        spike_ts = spike_ts * float(sample_rate)  # in ms
        spike_ts.sort()
        on_good = getattr(self, 'ttl_timestamps') / sample_rate / float(1000)
        dt = np.array(dt)
        irange = on_good[:, np.newaxis] + dt[np.newaxis, :]
        dts = np.searchsorted(spike_ts, irange)
        y = []
        x = []
        for i, t in enumerate(dts):
            tmp = spike_ts[t[0]:t[1]] - on_good[i]
            x.extend(tmp)
            y.extend(np.repeat(i, len(tmp)))

        if hist:
            nEvents = int(self.STM["num_stm_samples"])
            return np.histogram2d(
                x, y, bins=[np.arange(
                    dt[0], dt[1]+1, 1), np.arange(0, nEvents+1, 1)])[0]
        else:
            return np.histogram(
                x, bins=np.arange(
                    dt[0], dt[1]+1, 1), range=dt)[0]
    '''

    @stripAxes
    def show_SAC(self, A: np.array, inDict: dict,
                 ax: matplotlib.axes = None, **kwargs) -> matplotlib.axes:
        """
        Displays the result of performing a spatial autocorrelation (SAC)
        on a grid cell.

        Uses the dictionary containing measures of the grid cell SAC to
        make a pretty picture

        Parameters
        ----------
        A : array_like
            The spatial autocorrelogram
        inDict : dict
            The dictionary calculated in getmeasures
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            If given the plot will get drawn in these axes. Default None

        Returns
        -------
        fig : matplotlib.Figure instance
            The Figure on which the SAC is shown

        See Also
        --------
        ephysiopy.common.binning.RateMap.autoCorr2D()
        ephysiopy.common.ephys_generic.FieldCalcs.getMeaures()
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        Am = A.copy()
        Am[~inDict['dist_to_centre']] = np.nan
        Am = np.ma.masked_invalid(np.atleast_2d(Am))
        x, y = np.meshgrid(
            np.arange(0, np.shape(A)[1]),
            np.arange(0, np.shape(A)[0]))
        vmax = np.nanmax(np.ravel(A))
        ax.pcolormesh(
            x, y, A, cmap=plt.cm.get_cmap("gray_r"),
            edgecolors='face', vmax=vmax, shading='auto')
        import copy
        cmap = copy.copy(plt.cm.get_cmap("jet"))
        cmap.set_bad('w', 0)
        ax.pcolormesh(
            x, y, Am, cmap=cmap,
            edgecolors='face', vmax=vmax, shading='auto')
        # horizontal green line at 3 o'clock
        _y = (np.shape(A)[0]/2, np.shape(A)[0]/2)
        _x = (np.shape(A)[1]/2, np.shape(A)[0])
        ax.plot(_x, _y, c='g')
        mag = inDict['scale'] * 0.5
        th = np.linspace(0, inDict['orientation'], 50)
        from ephysiopy.common.utils import rect
        [x, y] = rect(mag, th, deg=1)
        # angle subtended by orientation
        ax.plot(
            x + (inDict['dist_to_centre'].shape[1] / 2),
                (inDict['dist_to_centre'].shape[0] / 2) - y, 'r', **kwargs)
        # plot lines from centre to peaks above middle
        for p in inDict['closest_peak_coords']:
            if p[0] <= inDict['dist_to_centre'].shape[0] / 2:
                ax.plot(
                    (inDict['dist_to_centre'].shape[1]/2, p[1]),
                    (inDict['dist_to_centre'].shape[0] / 2, p[0]), 'k', **kwargs)
        ax.invert_yaxis()
        all_ax = ax.axes
        all_ax.set_aspect('equal')
        all_ax.set_xlim((0.5, inDict['dist_to_centre'].shape[1]-1.5))
        all_ax.set_ylim((inDict['dist_to_centre'].shape[0]-.5, -.5))
        return ax
    '''
    def plotDirFilteredRmaps(self, tetrode, cluster, maptype='rmap', **kwargs):
        """
        Plots out directionally filtered ratemaps for the tetrode/ cluster

        Parameters
        ----------
        tetrode : int
        cluster : int
        maptype : str
            Valid values include 'rmap', 'polar', 'xcorr'
        """
        inc = 8.0
        step = 360/inc
        dirs_st = np.arange(-step/2, 360-(step/2), step)
        dirs_en = np.arange(step/2, 360, step)
        dirs_st[0] = dirs_en[-1]

        if 'polar' in maptype:
            _, axes = plt.subplots(
                nrows=3, ncols=3, subplot_kw={'projection': 'polar'})
        else:
            _, axes = plt.subplots(nrows=3, ncols=3)
        ax0 = axes[0][0]  # top-left
        ax1 = axes[0][1]  # top-middle
        ax2 = axes[0][2]  # top-right
        ax3 = axes[1][0]  # middle-left
        ax4 = axes[1][1]  # middle
        ax5 = axes[1][2]  # middle-right
        ax6 = axes[2][0]  # bottom-left
        ax7 = axes[2][1]  # bottom-middle
        ax8 = axes[2][2]  # bottom-right

        max_rate = 0
        for d in zip(dirs_st, dirs_en):
            self.posFilter = {'dir': (d[0], d[1])}
            if 'polar' in maptype:
                rmap = self._getMap(
                    tetrode=tetrode, cluster=cluster, var2bin='dir')[0]
            elif 'xcorr' in maptype:
                x1 = self.TETRODE[tetrode].getClustTS(cluster) / (96000/1000)
                rmap = self.spikecalcs.xcorr(
                    x1, x1, Trange=np.array([-500, 500]))
            else:
                rmap = self._getMap(tetrode=tetrode, cluster=cluster)[0]
            if np.nanmax(rmap) > max_rate:
                max_rate = np.nanmax(rmap)

        from collections import OrderedDict
        dir_rates = OrderedDict.fromkeys(dirs_st, None)

        ax_collection = [ax5, ax2, ax1, ax0, ax3, ax6, ax7, ax8]
        for d in zip(dirs_st, dirs_en, ax_collection):
            self.posFilter = {'dir': (d[0], d[1])}
            npos = np.count_nonzero(np.ma.compressed(~self.POS.dir.mask))
            print("npos = {}".format(npos))
            nspikes = np.count_nonzero(
                np.ma.compressed(
                    ~self.TETRODE[tetrode].getClustSpks(
                        cluster).mask[:, 0, 0]))
            print("nspikes = {}".format(nspikes))
            dir_rates[d[0]] = nspikes  # / (npos/50.0)
            if 'spikes' in maptype:
                self.plotSpikesOnPath(
                    tetrode, cluster, ax=d[2], markersize=4)
            elif 'rmap' in maptype:
                self._plotMap(
                    tetrode, cluster, ax=d[2], vmax=max_rate)
            elif 'polar' in maptype:
                self._plotMap(
                    tetrode, cluster, var2bin='dir', ax=d[2], vmax=max_rate)
            elif 'xcorr' in maptype:
                self.plotXCorr(
                    tetrode, cluster, ax=d[2])
                x1 = self.TETRODE[tetrode].getClustTS(cluster) / (96000/1000)
                print("x1 len = {}".format(len(x1)))
                dir_rates[d[0]] = self.spikecalcs.thetaBandMaxFreq(x1)
                d[2].set_xlabel('')
                d[2].set_title('')
                d[2].set_xticklabels('')
            d[2].set_title("nspikes = {}".format(nspikes))
        self.posFilter = None
        if 'spikes' in maptype:
            self.plotSpikesOnPath(tetrode, cluster, ax=ax4)
        elif 'rmap' in maptype:
            self._plotMap(tetrode, cluster, ax=ax4)
        elif 'polar' in maptype:
            self._plotMap(tetrode, cluster, var2bin='dir', ax=ax4)
        elif 'xcorr' in maptype:
            self.plotXCorr(tetrode, cluster, ax=ax4)
            ax4.set_xlabel('')
            ax4.set_title('')
            ax4.set_xticklabels('')
        return dir_rates
        '''
