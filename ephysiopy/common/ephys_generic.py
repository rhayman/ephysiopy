"""
The classes contained in this module are supposed to be agnostic to recording format
and encapsulate some generic mechanisms for producing things like spike timing
autocorrelograms, power spectrum calculation and so on
"""

import numpy as np
from scipy import signal, spatial, misc, ndimage
import skimage as skimage
import warnings
import matplotlib
import matplotlib.pylab as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle
from ephysiopy.common import binning
from ephysiopy.dacq2py import tintcolours as tcols

class SpikeCalcsGeneric(object):
    """
    Deals with the processing and analysis of spike timing data.

    Parameters
    ----------
    spike_times : array_like
        the times of 'spikes' in the trial
        this should be all spikes as the cluster identity vector _spk_clusters
        is used to pick out the right spikes
    waveforms : np.array, optional
        not sure on shape yet but will be something like a
        a 4 x nSpikes x nSamples (4 for tetrode-based analysis)

    Notes
    -----
    Units for time are provided as per the sample rate but converted internally to milliseconds
    """
    def __init__(self, spike_times, waveforms=None, **kwargs):
        self.spike_times = spike_times
        self.waveforms = waveforms
        self._event_ts = None # the times that events occured i.e. the laser came on
        self._spk_clusters = None # vector of cluster ids, same length as spike_times
        self._event_window = np.array((-0.050, 0.100)) # window, in seconds, either side of the stimulus, to examine
        self._stim_width = None # the width, in ms, of the stimulus
        self._secs_per_bin = 0.001 # used to increase / decrease size of bins in psth
        self._sample_rate = 30000
        self._duration = None

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    def n_spikes(self, cluster=None):
        if cluster is None:
            return len(self.spike_times)
        else:
            if self.spk_clusters is None:
                warnings.warn("No clusters available, please load some into me.")
                return
            else:
                return np.count_nonzero(self._spk_clusters==cluster)

    @property
    def event_ts(self):
        return self._event_ts

    @event_ts.setter
    def event_ts(self, value):
        self._event_ts = value

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self._duration = value

    @property
    def spk_clusters(self):
        return self._spk_clusters

    @spk_clusters.setter
    def spk_clusters(self, value):
        self._spk_clusters = value

    @property
    def event_window(self):
        return self._event_window

    @event_window.setter
    def event_window(self, value):
        self._event_window = value

    @property
    def stim_width(self):
        return self._stim_width

    @stim_width.setter
    def stim_width(self, value):
        self._stim_width = value

    @property
    def _secs_per_bin(self):
        return self.__secs_per_bin

    @_secs_per_bin.setter
    def _secs_per_bin(self, value):
        self.__secs_per_bin = value

    def trial_mean_fr(self, cluster: int)->float:
        # Returns the trial mean firing rate for the cluster
        if self.duration is None:
            warnings.warn("No duration provided, give me one!")
            return
        return self.n_spikes(cluster) / self.duration

    def mean_isi_range(self, cluster: int, n: int)->float:
        """
        Calculates the mean of the autocorrelation from 0 to n milliseconds
        Used to help classify a neruons type (principal, interneuron etc)

        Parameters
        ----------
        cluster : int
            The cluster to analyse
        n : int
            The range in milliseconds to calculate the mean over

        Returns
        -------
        mean_isi_range : float
            The mean of the autocorrelogram between 0 and n milliseconds
        """
        if cluster not in self.spk_clusters:
            warnings.warn("Cluster not available")
            return
        bins = 201
        trange = np.array((-500, 500))
        t = self.spike_times[self.spk_clusters==cluster]
        y = self.xcorr(t, Trange=trange)
        y = y.astype(np.int64) # See xcorr docs
        counts, bins = np.histogram(y[y!=0], bins=bins, range=trange)
        mask = np.logical_and(bins>0, bins<n)
        return np.mean(counts[mask[1:]])

    def xcorr(self, x1: np.ndarray, x2=None, Trange=None, **kwargs)->np.ndarray:
        """
        Calculates the histogram of the ISIs in x1 or x1 vs x2

        Parameters
        ----------
        x1, x2 : array_like
            The times of the spikes emitted by the cluster(s)
            NB must be signed int to accomodate negative times
        Trange : array_like
            Range of times to bin up. Defaults to [-500, +500] in ms

        Returns
        -------
        y : np.ndarray
            The time differences between spike times in x1 over the range
            of times defined Trange
        """
        if x2 is None:
            x2 = x1.copy()
        if Trange is None:
            Trange = np.array([-500, 500])
        if type(Trange) == tuple:
            Trange = np.array(Trange)
        y = []
        irange = x1[:, np.newaxis] + Trange[np.newaxis, :]
        dts = np.searchsorted(x2, irange)
        for i, t in enumerate(dts):
            y.extend(x2[t[0]:t[1]] - x1[i])
        y = np.array(y, dtype=float)
        return y

    def calculatePSTH(self, cluster_id, **kwargs):
        """
        Calculate the PSTH of event_ts against the spiking of a cell

        Parameters
        ----------
        cluster_id : int
            The cluster for which to calculate the psth

        Returns
        -------
        x, y : list
            The list of time differences between the spikes of the cluster
            and the events (x) and the trials (y)
        """
        if self._event_ts is None:
            raise Exception("Need some event timestamps! Aborting")
        if self._spk_clusters is None:
            raise Exception("Need cluster identities! Aborting")
        event_ts = self.event_ts
        event_ts.sort()
        if type(event_ts) == list:
            event_ts = np.array(event_ts)

        spike_times = self.spike_times[self.spk_clusters == cluster_id]
        irange = event_ts[:, np.newaxis] + self.event_window[np.newaxis, :]
        dts = np.searchsorted(spike_times, irange)
        x = []
        y = []
        for i, t in enumerate(dts):
            tmp = spike_times[t[0]:t[1]] - event_ts[i]
            x.extend(tmp)
            y.extend(np.repeat(i,len(tmp)))
        return x, y

    def plotPSTH(self, cluster, fig=None):
        """
        Plots the PSTH for a cluster

        Parameters
        ----------
        cluster : int
            The cluster to examine

        Returns
        -------
        cluster, i : int
            The cluster and a junk variable (not sure why for now)
        """
        x, y = self.calculatePSTH(cluster)
        show = False # used below to show the figure or leave this to the caller
        if fig is None:
            fig = plt.figure(figsize=(4.0,7.0))
            show = True
        scatter_ax = fig.add_subplot(111)
        scatter_ax.scatter(x, y, marker='.', s=2, rasterized=False)
        divider = make_axes_locatable(scatter_ax)
        scatter_ax.set_xticks((self.event_window[0], 0, self.event_window[1]))
        scatter_ax.set_xticklabels((str(self.event_window[0]), '0', str(self.event_window[1])))
        hist_ax = divider.append_axes("top", 0.95, pad=0.2, sharex=scatter_ax,
                                      transform=scatter_ax.transAxes)
        scattTrans = transforms.blended_transform_factory(scatter_ax.transData,
                                                          scatter_ax.transAxes)
        if self.stim_width is not None:
            scatter_ax.add_patch(Rectangle((0, 0), width=self.stim_width, height=1,
                        transform=scattTrans,
                        color=[0, 0, 1], alpha=0.5))
            histTrans = transforms.blended_transform_factory(hist_ax.transData,
                                                             hist_ax.transAxes)
            hist_ax.add_patch(Rectangle((0, 0), width=self.stim_width, height=1,
                              transform=histTrans,
                              color=[0, 0, 1], alpha=0.5))
        scatter_ax.set_ylabel('Laser stimulation events', labelpad=-18.5)
        scatter_ax.set_xlabel('Time to stimulus onset(secs)')
        nStms = int(len(self.event_ts))
        scatter_ax.set_ylim(0, nStms)
        # Label only the min and max of the y-axis
        ylabels = scatter_ax.get_yticklabels()
        for i in range(1, len(ylabels)-1):
            ylabels[i].set_visible(False)
        yticks = scatter_ax.get_yticklines()
        for i in range(1, len(yticks)-1):
            yticks[i].set_visible(False)
        histColor = [192/255.0,192/255.0,192/255.0]
        hist_ax.hist(x, bins=np.arange(self.event_window[0], self.event_window[1] + self._secs_per_bin, self._secs_per_bin),
                             color=histColor, alpha=0.6, range=self.event_window, rasterized=True, histtype='stepfilled')
        hist_ax.set_ylabel("Spike count", labelpad=-2.5)
        plt.setp(hist_ax.get_xticklabels(), visible=False)
        # Label only the min and max of the y-axis
        ylabels = hist_ax.get_yticklabels()
        for i in range(1, len(ylabels)-1):
            ylabels[i].set_visible(False)
        yticks = hist_ax.get_yticklines()
        for i in range(1, len(yticks)-1):
            yticks[i].set_visible(False)
        hist_ax.set_xlim(self.event_window)
        scatter_ax.set_xlim(self.event_window)
        if show:
            plt.show()
        yield cluster, 1

    def plotAllXCorrs(self, clusters, fig=None):
        """
        Plots all xcorrs in a single figure window

        Parameters
        ----------
        clusters : list
            The clusters to plot
        fig : matplotlib.figure instance, optional, default None
            If provided the figure will contain all the axes
        """
        from ephysiopy.dacq2py import spikecalcs
        SpkCalcs = spikecalcs.SpikeCalcs()
        if fig is None:
            fig = plt.figure(figsize=(10,20))

        nrows = np.ceil(np.sqrt(len(clusters))).astype(int)
        fig.subplots_adjust(wspace=0.25,hspace=0.25)
        for i, cluster in enumerate(clusters):
            cluster_idx = np.nonzero(self.spk_clusters == cluster)[0]
            cluster_ts = np.ravel(self.spike_times[cluster_idx])
            ax = fig.add_subplot(nrows,nrows,i+1)
            y = SpkCalcs.xcorr(cluster_ts.T / float(self.sample_rate / 1000)) # ms
            ax.hist(y[y != 0], bins=201, range=[-500, 500], color='k', histtype='stepfilled')
            ax.set_xlim(-500,500)
            ax.set_xticks((-500, 0, 500))
            ax.set_xticklabels((str(-500), '0', str(500)),fontweight='normal', size=8)
            ax.tick_params(axis='both', which='both', left=False, right=False,
                            bottom=False, top=False)
            ax.set_yticklabels('')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title(cluster, fontweight='bold', size=10, pad=1)
        plt.show()

class SpikeCalcsTetrode(SpikeCalcsGeneric):
    """
    Encapsulates methods specific to the geometry inherent in tetrode-based
    recordings
    """
    def __init__(self):
        pass

class SpikeCalcsProbe(SpikeCalcsGeneric):
    """
    Encapsulates methods specific to probe-based recordings
    """
    def __init__(self):
        pass

class EEGCalcsGeneric(object):
    """
    Generic class for processing and analysis of EEG data

    Parameters
    ----------
    sig : array_like
        The signal (of the LFP data)
    fs  : float
        The sample rate
    """
    def __init__(self, sig, fs):
        self.sig = sig
        self.fs = fs
        self.thetaRange = [6,12]
        self.outsideRange = [3,125]
        # for smoothing and plotting of power spectrum
        self.smthKernelWidth = 2
        self.smthKernelSigma = 0.1875
        self.sn2Width = 2
        self.maxFreq = 125
        self.maxPow = None

    def _nextpow2(self, val : int):
        """
        Calculates the next power of 2 that will hold val
        """
        val = val - 1
        val = (val >> 1) | val
        val = (val >> 2) | val
        val = (val >> 4) | val
        val = (val >> 8) | val
        val = (val >> 16) | val
        val = (val >> 32) | val
        return np.log2(val + 1)

    def butterFilter(self, low: float, high: float, order: int=5)->np.ndarray:
        """
        Filters self.sig with a butterworth filter with a bandpass filter
        defined by low and high

        Parameters
        ----------
        low, high : float
            The lower and upper bounds of the bandpass filter
        order : int
            The order of the filter

        Returns
        -------
        filt : np.ndarray
            The filtered signal

        Notes
        -----
        The signal is filtered in both the forward and reverse directions (scipy.signal.filtfilt)
        """
        nyqlim = self.fs / 2
        lowcut = low / nyqlim
        highcut = high / nyqlim
        b, a = signal.butter(order, [lowcut, highcut], btype='band')
        return signal.filtfilt(b, a, self.sig)

    def calcEEGPowerSpectrum(self, **kwargs):
        """
        Calculates the power spectrum of self.sig

        Parameters
        ----------
        None

        Returns
        -------
        Nothing. Sets a bunch of instance variables for the first time including
            freqs : array_like
                The frequencies at which the spectrogram was calculated
            power : array_like
                The power at the frequencies defined above
            sm_power : array_like
                The smoothed power
            bandmaxpower : float
                The maximum power in the theta band
            freqatbandmaxpower : float
                The frequency at which the power is maximum
        """
        nqlim = self.fs / 2
        origlen = len(self.sig)
        fftlen = 2 ** self._nextpow2(origlen).astype(int)
        freqs, power = signal.periodogram(self.sig, self.fs, return_onesided=True, nfft=fftlen)
        ffthalflen = fftlen / 2+1
        binsperhz = (ffthalflen-1) / nqlim
        kernelsigma = self.smthKernelSigma * binsperhz
        smthkernelsigma = 2 * int(4.0 * kernelsigma + 0.5) + 1
        gausswin = signal.gaussian(smthkernelsigma, kernelsigma)
        sm_power = signal.fftconvolve(power, gausswin, 'same')
        sm_power = sm_power / np.sqrt(len(sm_power))
        spectrummaskband = np.logical_and(freqs > self.thetaRange[0], freqs < self.thetaRange[1])
        bandmaxpower = np.max(sm_power[spectrummaskband])
        maxbininband = np.argmax(sm_power[spectrummaskband])
        bandfreqs = freqs[spectrummaskband]
        freqatbandmaxpower = bandfreqs[maxbininband]
        self.freqs = freqs
        self.power = power
        self.sm_power = sm_power
        self.bandmaxpower = bandmaxpower
        self.freqatbandmaxpower = freqatbandmaxpower

    def plotPowerSpectrum(self, **kwargs):
        # calculate
        self.calcEEGPowerSpectrum()
        # plotting
        import matplotlib.pylab as plt
        plt.figure()
        ax = plt.gca()
        freqs = self.freqs[0::50]
        power = self.power[0::50]
        sm_power = self.sm_power[0::50]
        ax.plot(freqs, power, alpha=0.5, color=[0.8627, 0.8627, 0.8627])
        ax.plot(freqs, sm_power)
        ax.set_xlim(0, self.maxFreq)
        if 'ylim' in kwargs.keys():
            ylim = kwargs['ylim']
        else:
            ylim = [0, self.bandmaxpower / 0.8]

        ax.set_ylim(ylim)
        ax.set_ylabel('Power')
        ax.set_xlabel('Frequency')
        ax.text(x=self.thetaRange[1] / 0.9, y=self.bandmaxpower, s=str(self.freqatbandmaxpower)[0:4], fontsize=20)
        from matplotlib.patches import Rectangle
        r = Rectangle((self.thetaRange[0],0), width=np.diff(self.thetaRange)[0], height=np.diff(ax.get_ylim())[0], alpha=0.25, color='r', ec='none')
        ax.add_patch(r)
        plt.show()

    def plotEventEEG(self, event_ts, event_window=(-0.05, 0.1), stim_width=0.01, sample_rate=3e4):
        """
        Plots the mean eeg +- std. dev centred on event timestamps

        Parameters
        ----------
        event_ts : array_like
            The event timestamps in seconds
        event_window : 2-tuple, default = (-0.05, 0.1)
            The pre- and post-stimulus window to examine. In seconds.
            Defaults to the previous 50ms and the subsequent 100ms
        stim_width : float
            The duration of the stimulus. Used for plotting
        sample_rate : float
            The sample rate of the events

        """
        # bandpass filter the raw data first
        from scipy import signal
        nyq = sample_rate / 2
        highlim = 500 / nyq
        b, a = signal.butter(5, highlim, btype='lowpass')
        sig = signal.filtfilt(b, a, self.sig)

        event_idx = np.round(event_ts*sample_rate).astype(int)
        event_window = np.array(event_window)

        max_samples = np.ptp(event_window*sample_rate).astype(int)
        num_events = len(event_ts)
        eeg_array = np.zeros([num_events, max_samples])
        st = int(event_window[0]*sample_rate)
        en = int(event_window[1]*sample_rate)
        for i, eeg_idx in enumerate(event_idx):
            eeg_array[i, :] = sig[eeg_idx+st:eeg_idx+en]
        mn = np.mean(eeg_array, 0)
        se = np.std(eeg_array, 0) / np.sqrt(num_events)
        import matplotlib.pylab as plt
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.transforms as transforms
        from matplotlib.patches import Rectangle
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(np.linspace(event_window[0], event_window[1], len(mn)), mn, yerr=se, rasterized=False)
        ax.set_xlim(event_window)
        axTrans = transforms.blended_transform_factory(ax.transData,
                                                           ax.transAxes)
        ax.add_patch(Rectangle((0, 0), width=stim_width, height=1,
                             transform=axTrans,
                             color=[1, 1, 0], alpha=0.5))
        ax.set_ylabel('LFP ($\mu$V)')
        ax.set_xlabel('Time(s)')
        plt.show()

class PosCalcsGeneric(object):
    """
    Generic class for post-processing of position data
    Uses numpys masked arrays for dealing with bad positions, filtering etc

    Parameters
    ----------
    x, y : array_like
        The x and y positions.
    ppm : int
        Pixels per metre
    cm : boolean
        Whether everything is converted into cms or not
    jumpmax : int
        Jumps in position (pixel coords) greater than this are bad

    Notes
    -----
    The positional data (x,y) is turned into a numpy masked array once this
    class is initialised - that mask is then modified through various
    functions (postprocesspos being the main one).
    """
    def __init__(self, x, y, ppm, cm=True, jumpmax=100):
        assert np.shape(x) == np.shape(y)
        self.xy = np.ma.MaskedArray([x, y])
        self.dir = np.ma.MaskedArray(np.zeros_like(x))
        self.speed = None
        self.ppm = ppm
        self.cm = cm
        self.jumpmax = jumpmax
        self.nleds = np.ndim(x)
        self.npos = len(x)
        self.tracker_params = None
        self.sample_rate = None

    def postprocesspos(self, tracker_params, **kwargs)->tuple:
        """
        Post-process position data

        Parameters
        ----------
        tracker_params : dict
            Same dict as created in OEKiloPhy.Settings.parsePos
            (from module openephys2py)

        Returns
        -------
        xy, hdir : np.ma.MaskedArray
            The post-processed position data

        Notes
        -----
        Several internal functions are called here: speefilter, interpnans, smoothPos
        and calcSpeed. Some internal state/ instance variables are set as well. The
        mask of the positional data (an instance of numpy masked array) is modified
        throughout this method.

        """
        xy = self.xy
        xy = np.ma.MaskedArray(xy, dtype=np.int32)
        x_zero = xy[:, 0] < 0
        y_zero = xy[:, 1] < 0
        xy[np.logical_or(x_zero, y_zero), :] = np.ma.masked

        self.tracker_params = tracker_params
        if 'LeftBorder' in tracker_params:
            min_x = tracker_params['LeftBorder']
            xy[:, xy[0,:] <= min_x] = np.ma.masked
        if 'TopBorder' in tracker_params:
            min_y = tracker_params['TopBorder'] # y origin at top
            xy[:, xy[1,:] <= min_y] = np.ma.masked
        if 'RightBorder' in tracker_params:
            max_x = tracker_params['RightBorder']
            xy[:, xy[0,:] >= max_x] = np.ma.masked
        if 'BottomBorder' in tracker_params:
            max_y = tracker_params['BottomBorder']
            xy[:, xy[1,:] >= max_y] = np.ma.masked
        if 'SampleRate' in tracker_params.keys():
            self.sample_rate = int(tracker_params['SampleRate'])
        else:
            self.sample_rate = 30

        xy = xy.T
        xy = self.speedfilter(xy)
        xy = self.interpnans(xy) # ADJUST THIS SO NP.MASKED ARE INTERPOLATED OVER
        xy = self.smoothPos(xy)
        self.calcSpeed(xy)

        import math
        pos2 = np.arange(0, self.npos-1)
        xy_f = xy.astype(np.float)
        self.dir[pos2] = np.mod(((180/math.pi) * (np.arctan2(-xy_f[1, pos2+1] + xy_f[1,pos2],+xy_f[0,pos2+1]-xy_f[0,pos2]))), 360)
        self.dir[-1] = self.dir[-2]

        hdir = self.dir

        return xy, hdir

    def speedfilter(self, xy):
        """
        Filters speed

        Parameters
        ----------
        xy : np.ma.MaskedArray
            The xy data

        Returns
        -------
        xy : np.ma.MaskedArray
            The xy data with speeds > self.jumpmax masked
        """
        df = np.diff(xy, axis=0)
        disp = np.hypot(df[:,0], df[:,1])
        disp = np.insert(disp, -1, 0)
        xy[disp > self.jumpmax, :] = np.ma.masked
        return xy

    def interpnans(self, xy):
        for i in range(0,np.shape(xy)[-1],2):
            missing = xy.mask.any(axis=-1)
            ok = np.logical_not(missing)
            ok_idx = np.ravel(np.nonzero(np.ravel(ok))[0])#gets the indices of ok poses
            missing_idx = np.ravel(np.nonzero(np.ravel(missing))[0])#get the indices of missing poses
            if len(missing_idx) > 0:
                try:
                    good_data = np.ravel(xy.data[ok_idx,i])
                    good_data1 = np.ravel(xy.data[ok_idx,i+1])
                    xy.data[missing_idx,i] = np.interp(missing_idx,ok_idx,good_data)#,left=np.min(good_data),right=np.max(good_data)
                    xy.data[missing_idx,i+1] = np.interp(missing_idx,ok_idx,good_data1)
                except ValueError:
                    pass
        xy.mask = 0
        print("{} bad/ jumpy positions were interpolated over".format(len(missing_idx)))#this is wrong i think
        return xy

    def smoothPos(self, xy):
        """
        Smooths position data

        Parameters
        ----------
        xy : np.ma.MaskedArray
            The xy data

        Returns
        -------
        xy : array_like
            The smoothed positional data
        """
        # Extract boundaries of window used in recording

        x = xy[:,0].astype(np.float64)
        y = xy[:,1].astype(np.float64)

        from ephysiopy.common.utils import smooth
        # TODO: calculate window_len from pos sampling rate
        # 11 is roughly equal to 400ms at 30Hz (window_len needs to be odd)
        sm_x = smooth(x, window_len=11, window='flat')
        sm_y = smooth(y, window_len=11, window='flat')
        return np.array([sm_x, sm_y])

    def calcSpeed(self, xy):
        """
        Calculates speed

        Parameters
        ---------
        xy : np.ma.MaskedArray
            The xy positional data

        Returns
        -------
        Nothing. Sets self.speed
        """
        speed = np.sqrt(np.sum(np.power(np.diff(xy),2),0))
        speed = np.append(speed, speed[-1])
        if self.cm:
            self.speed = speed * (100 * self.sample_rate / self.ppm) # in cm/s now
        else:
            self.speed = speed

    def upsamplePos(self, xy, upsample_rate=50):
        """
        Upsamples position data from 30 to upsample_rate

        Parameters
        ---------
        xy : np.ma.MaskedArray
            The xy positional data

        upsample_rate : int
            The rate to upsample to

        Returns
        -------
        new_xy : np.ma.MaskedArray
            The upsampled xy positional data

        Notes
        -----
        This is mostly to get pos data recorded using PosTracker at 30Hz
        into Axona format 50Hz data
        """
        from scipy import signal
        denom = np.gcd(upsample_rate, 30)
        new_xy = signal.resample_poly(xy, upsample_rate/denom, 30/denom)
        return new_xy

class MapCalcsGeneric(object):
    """
    Produces graphical output including but not limited to spatial
    analysis of data.
    
    Parameters
    ----------
    xy : array_like
        The positional data usually as a 2D numpy array
    hdir : array_like
        The head direction data usually a 1D numpy array
    pos_ts : array_like
        1D array of timestamps in seconds
    spk_ts : array_like
        1D array of timestamps in seconds
    plot_type : str or list
        Determines the plots produced. Legal values:
        ['map','path','hdir','sac', 'speed']
    
    Notes
    -----
    Output possible: 
    * ratemaps (xy)
    * polar plots (heading direction)
    * grid cell spatial autocorrelograms
    * speed vs rate plots

    It is possible to iterate through instances of this class as it has a yield
    method defined
    """
    def __init__(self, xy, hdir, speed, pos_ts, spk_ts, plot_type='map', **kwargs):
        if (np.argmin(np.shape(xy)) == 1):
            xy = xy.T
        self.xy = xy
        self.hdir = hdir
        self.speed = speed
        self.pos_ts = pos_ts
        if (spk_ts.ndim == 2):
            spk_ts = np.ravel(spk_ts)
        self.spk_ts = spk_ts
        self.plot_type = plot_type
        self.spk_pos_idx = self.__interpSpkPosTimes()
        self.__good_clusters = None
        self.__spk_clusters = None
        self.save_grid_output_location = None
        if ( 'ppm' in kwargs.keys() ):
            self.__ppm = kwargs['ppm']
        else:
            self.__ppm = 400
        if 'pos_sample_rate' in kwargs.keys():
            self.pos_sample_rate = kwargs['pos_sample_rate']
        else:
            self.pos_sample_rate = 30
        if 'save_grid_summary_location' in kwargs.keys():
            self.save_grid_output_location = kwargs['save_grid_summary_location']

    @property
    def good_clusters(self):
        return self.__good_clusters

    @good_clusters.setter
    def good_clusters(self, value):
        self.__good_clusters = value

    @property
    def spk_clusters(self):
        return self.__spk_clusters

    @spk_clusters.setter
    def spk_clusters(self, value):
        self.__spk_clusters = value

    @property
    def ppm(self):
        return self.__ppm

    @ppm.setter
    def ppm(self, value):
        self.__ppm = value

    def plotAll(self):
        """
        Plots rate maps and other graphical output

        Notes
        ----
        This method uses the data provided to the class instance to plot
        various maps into a single figure window for each cluster. The things
        to plot are given in self.plot_type and the list of clusters in self.good_clusters
        """
        if 'all' in self.plot_type:
            what_to_plot = ['map','path','hdir','sac','speed', 'sp_hd']
            fig = plt.figure(figsize=(20,10))
        else:
            what_to_plot = list(self.plot_type)
            if len(what_to_plot) > 1:
                fig = plt.figure(figsize=(20,10))
            else:
                fig = plt.figure(figsize=(20,10))#, constrained_layout=True)
        if 'sac' in what_to_plot:
            from ephysiopy.common import gridcell
            S = gridcell.SAC()
        import matplotlib.gridspec as gridspec
        nrows = np.ceil(np.sqrt(len(self.good_clusters))).astype(int)
        outer = gridspec.GridSpec(nrows, nrows, figure=fig)

        inner_ncols = int(np.ceil(len(what_to_plot) / 2)) # max 2 cols
        if len(what_to_plot) == 1:
            inner_nrows = 1
        else:
            inner_nrows = 2
        for i, cluster in enumerate(self.good_clusters):
            inner = gridspec.GridSpecFromSubplotSpec(inner_nrows,inner_ncols, subplot_spec=outer[i])
            for plot_type_idx, plot_type in enumerate(what_to_plot):
                if 'hdir' in plot_type:
                    ax = fig.add_subplot(inner[plot_type_idx],projection='polar')
                else:
                    ax = fig.add_subplot(inner[plot_type_idx])

                if 'path' in plot_type:
                    self.makeSpikePathPlot(cluster, ax)
                if 'map' in plot_type:
                    rmap = self.makeRateMap(cluster, ax)
                if 'hdir' in plot_type:
                    self.makeHDPlot(cluster, ax, add_mrv=True)
                if 'sac' in plot_type:
                    rmap = self.makeRateMap(cluster)
                    nodwell = ~np.isfinite(rmap[0])
                    sac = S.autoCorr2D(rmap[0], nodwell)
                    d = S.getMeasures(sac)
                    S.show(sac,d,ax)
                if 'speed' in plot_type:
                    self.makeSpeedVsRatePlot(cluster, 0.0, 40.0, 3.0, ax)
                if 'sp_hd' in plot_type:
                    self.makeSpeedVsHeadDirectionPlot(cluster, ax)
                # if first_sub_axis in plot_type: # label the first sub-axis only
                    # ax = fig.add_subplot(inner[plot_type_idx])
                ax.set_title(cluster, fontweight='bold', size=8)
        plt.show()

    def __iter__(self):
        if 'all' in self.plot_type:
            from ephysiopy.common import gridcell
            S = gridcell.SAC()

        for cluster in self.good_clusters:
            print("Cluster {}".format(cluster))
            if 'map' in self.plot_type:
                fig = plt.figure()
                ax = plt.gca()
                self.makeRateMap(cluster, ax)
                fig.show()
            elif 'path' in self.plot_type:
                plt.figure()
                ax = plt.gca()
                self.makeSpikePathPlot(cluster, ax)
                plt.show()
            elif 'hdir' in self.plot_type:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='polar')
                self.makeHDPlot(cluster, ax)
                plt.show()
            elif 'both' in self.plot_type:
                fig, (ax1, ax0) = plt.subplots(1,2)
                # ratemap
                self.makeRateMap(cluster, ax0)
                # path / spikes
                self.makeSpikePathPlot(cluster, ax1)
                plt.show()
            elif 'speed' in self.plot_type:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                self.makeSpeedVsRatePlot(cluster, 0.0, 40.0, 3.0, ax)
            elif 'sp_hd' in self.plot_type:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                self.makeSpeedVsHeadDirectionPlot(cluster, ax)
            elif 'all' in self.plot_type:
                fig = plt.figure(figsize=[9.6, 6.0])
                fig.suptitle("Cluster {}".format(cluster))
                ax1 = fig.add_subplot(2, 3, 1)
                self.makeSpikePathPlot(cluster, ax1)
                ax0 = fig.add_subplot(2, 3, 2)
                rmap = self.makeRateMap(cluster, ax0)
                ax2 = fig.add_subplot(2, 3, 3)
                nodwell = ~np.isfinite(rmap[0])
                sac = S.autoCorr2D(rmap[0], nodwell)
                d = S.getMeasures(sac)
                if self.save_grid_output_location:
                    d['Cluster'] = cluster
                    f = open(self.save_grid_output_location, 'w')
                    f.write(str(d))
                    f.close()
                S.show(sac,d,ax2)
                print("Gridscore: {:.2f}".format(d['gridness']))
                ax3 = fig.add_subplot(2, 3, 4, projection='polar')
                self.makeHDPlot(cluster, ax3, add_mrv=True)
                ax4 = fig.add_subplot(2, 3, 5)
                self.makeSpeedVsRatePlot(cluster, 0.0, 40.0, 3.0, ax4)
                ax5 = fig.add_subplot(2, 3, 6)
                self.makeSpeedVsHeadDirectionPlot(cluster, ax5)
                plt.show()
            yield cluster

    def makeRateMap(self, cluster, ax=None):
        pos_w = np.ones_like(self.pos_ts)
        mapMaker = binning.RateMap(self.xy, None, None, pos_w, ppm=self.ppm)
        spk_w = np.bincount(self.spk_pos_idx, self.spk_clusters==cluster, minlength=self.pos_ts.shape[0])
        # print("nSpikes: {}".format(np.sum(spk_w).astype(int)))
        rmap = mapMaker.getMap(spk_w)
        if ax is None:
            return rmap
        ratemap = np.ma.MaskedArray(rmap[0], np.isnan(rmap[0]), copy=True)
        x, y = np.meshgrid(rmap[1][1][0:-1], rmap[1][0][0:-1][::-1])
        vmax = np.max(np.ravel(ratemap))
        ax.pcolormesh(x, y, ratemap, cmap=cm.jet, edgecolors='face', vmax=vmax)
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        ax.set_aspect('equal')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        return rmap

    def makeSpikePathPlot(self, cluster, ax):
        ax.plot(self.xy[0], self.xy[1], c=tcols.colours[0], zorder=1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        idx = self.spk_pos_idx[self.spk_clusters==cluster]
        spk_colour = tcols.colours[1]
        ax.plot(self.xy[0,idx], self.xy[1,idx],'s',ms=1, c=spk_colour,mec=spk_colour)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def makeHDPlot(self, cluster, ax, **kwargs):
        pos_w = np.ones_like(self.pos_ts)
        mapMaker = binning.RateMap(self.xy, self.hdir, None, pos_w, ppm=self.ppm)
        spk_w = np.bincount(self.spk_pos_idx, self.spk_clusters==cluster, minlength=self.pos_ts.shape[0])
        rmap = mapMaker.getMap(spk_w, 'dir', 'rate')
        if rmap[0].ndim == 1:
            # polar plot
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='polar')
            theta = np.deg2rad(rmap[1][0][1:])
            ax.clear()
            ax.plot(theta, rmap[0])
            ax.set_aspect('equal')
            ax.tick_params(axis='both', which='both', bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False, labeltop=False, labelright=False)
            ax.set_rticks([])

            # See if we should add the mean resultant vector (mrv)
            if 'add_mrv' in kwargs.keys():
                from ephysiopy.dacq2py import statscalcs
                S = statscalcs.StatsCalcs()
                angles = self.hdir[self.spk_pos_idx[self.spk_clusters==cluster]]
                r, th = S.mean_resultant_vector(np.deg2rad(angles))
                # print("Mean resultant vector:")
                # print('\tUnit vector length: {:.3f}\n\tVector angle: {:.2f}'.format(r,np.rad2deg(th)))
                ax.plot([th, th],[0, r*np.max(rmap[0])],'r')
            ax.set_thetagrids([0, 90, 180, 270])

    def makeSpeedVsRatePlot(self, cluster, minSpeed=0.0, maxSpeed=40.0, sigma=3.0, ax=None, **kwargs):
        """
        Plots the instantaneous firing rate of a cell against running speed
        Also outputs a couple of measures as with Kropff et al., 2015; the
        Pearsons correlation and the depth of modulation (dom) - see below for
        details
        """
        speed = np.ravel(self.speed)
        if np.nanmax(speed) < maxSpeed:
            maxSpeed = np.nanmax(speed)
        spd_bins = np.arange(minSpeed, maxSpeed, 1.0)
        # Construct the mask
        speed_filt = np.ma.MaskedArray(speed)
        speed_filt = np.ma.masked_where(speed_filt < minSpeed, speed_filt)
        speed_filt = np.ma.masked_where(speed_filt > maxSpeed, speed_filt)
        from ephysiopy.dacq2py import spikecalcs
        S = spikecalcs.SpikeCalcs()
        x1 = self.spk_pos_idx[self.spk_clusters==cluster]
        spk_sm = S.smoothSpikePosCount(x1, self.pos_ts.shape[0], sigma, None)
        spk_sm = np.ma.MaskedArray(spk_sm, mask=np.ma.getmask(speed_filt))
        from scipy import stats
        stats.mstats.pearsonr(spk_sm, speed_filt)
        spd_dig  = np.digitize(speed_filt, spd_bins, right=True)
        mn_rate = np.array([np.ma.mean(spk_sm[spd_dig==i]) for i in range(0,len(spd_bins))])
        var = np.array([np.ma.std(spk_sm[spd_dig==i]) for i in range(0,len(spd_bins))])
        np.array([np.ma.sum(spk_sm[spd_dig==i]) for i in range(0,len(spd_bins))])
        if ax is not None:
            ax.errorbar(spd_bins, mn_rate * self.pos_sample_rate, yerr=var, color='k')
            ax.set_xlim(spd_bins[0], spd_bins[-1])
            plt.xticks([spd_bins[0], spd_bins[-1]], ['0', '{:.2g}'.format(spd_bins[-1])], fontweight='normal', size=6)
            plt.yticks([0,np.nanmax(mn_rate)*self.pos_sample_rate], ['0', '{:.2f}'.format(np.nanmax(mn_rate))], fontweight='normal', size=6)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

    def makeSpeedVsHeadDirectionPlot(self, cluster, ax):
        idx = self.spk_pos_idx[self.spk_clusters==cluster]
        w = np.bincount(idx, minlength=self.speed.shape[0])
        dir_bins = np.arange(0,360,6)
        spd_bins = np.arange(0,30,1)
        h = np.histogram2d(self.hdir, self.speed, [dir_bins,spd_bins],weights=w)
        b = binning.RateMap()
        im = b.blurImage(h[0],5,ftype='gaussian')
        im = np.ma.MaskedArray(im)
        # mask low rates...
        im = np.ma.masked_where(im<=1, im)
        # ... and where less than 0.5% of data is accounted for
        # all_sp_x_hd_binned = np.histogram2d(self.hdir, self.speed, [dir_bins,spd_bins])[0]
        # im = np.ma.masked_where(all_sp_x_hd_binned < (len(self.speed) * 0.005), im)
        x,y = np.meshgrid(dir_bins, spd_bins)
        ax.pcolormesh(x,y,im.T)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks([90,180,270], fontweight='normal', size=6)
        plt.yticks([10,20], fontweight='normal', size=6)

class FieldCalcs:
	"""
	This class differs from MapCalcsGeneric in that this one is mostly concerned with
	treating rate maps as images as opposed to using the spiking information contained
	within them. It therefore mostly deals with spatial rate maps of place and grid cells.
	"""

	def _blur_image(self, im, n, ny=None, ftype='boxcar'):
		"""
		blurs the image by convolving with a filter ('gaussian' or
		'boxcar') of
		size n. The optional keyword argument ny allows for a different
		size in the y direction.
		"""
		n = int(n)
		if ny is None:
			ny = n
		else:
			ny = int(ny)
		#  keep track of nans
		nan_idx = np.isnan(im)
		im[nan_idx] = 0
		if ftype == 'boxcar':
		    if np.ndim(im) == 1:
			    g = signal.boxcar(n) / float(n)
		    elif np.ndim(im) == 2:
			    g = signal.boxcar([n, ny]) / float(n)
		elif ftype == 'gaussian':
			x, y = np.mgrid[-n:n+1, -ny:ny+1]
			g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		g = g / g.sum()
		if np.ndim(im) == 1:
			g = g[n, :]
		improc = signal.convolve(im, g, mode='same')
		improc[nan_idx] = np.nan
		return improc

	def limit_to_one(self, A, prc=50, min_dist=5):
		"""
		Processes a multi-peaked ratemap (ie grid cell) and returns a matrix
		where the multi-peaked ratemap consist of a single peaked field that is
		a) not connected to the border and b) close to the middle of the ratemap
		"""
		Ac = A.copy()
		Ac[np.isnan(A)] = 0
		# smooth Ac more to remove local irregularities
		n = ny = 5
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		g = g / g.sum()
		Ac = signal.convolve(Ac, g, mode='same')
		peak_mask = skimage.feature.peak_local_max(Ac, min_distance=min_dist,
											 exclude_border=False,
											 indices=False)
		peak_labels = skimage.measure.label(peak_mask, 8)
		field_labels = skimage.morphology.watershed(image=-Ac,
											  markers=peak_labels)
		nFields = np.max(field_labels)
		sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
		labelled_sub_field_mask = np.zeros_like(sub_field_mask)
		sub_field_props = skimage.measure.regionprops(field_labels,
												intensity_image=Ac)
		sub_field_centroids = []
		sub_field_size = []

		for sub_field in sub_field_props:
			tmp = np.zeros(Ac.shape).astype(bool)
			tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
			tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
			sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
			labelled_sub_field_mask[sub_field.label-1,
						   np.logical_and(tmp2, tmp)] = sub_field.label
			sub_field_centroids.append(sub_field.centroid)
			sub_field_size.append(sub_field.area)  # in bins
		sub_field_mask = np.sum(sub_field_mask, 0)
		middle = np.round(np.array(A.shape) / 2)
		normd_dists = sub_field_centroids - middle
		field_dists_from_middle = np.hypot(normd_dists[:, 0], normd_dists[:, 1])
		central_field_idx = np.argmin(field_dists_from_middle)
		central_field = np.squeeze(labelled_sub_field_mask[central_field_idx, :, :])
		# collapse the labelled mask down to an 2d array
		labelled_sub_field_mask = np.sum(labelled_sub_field_mask, 0)
		# clear the border
		cleared_mask = skimage.segmentation.clear_border(central_field)
		# check we've still got stuff in the matrix or fail
		if ~np.any(cleared_mask):
			print('No fields were detected away from edges so nothing returned')
			return None, None, None
		else:
			central_field_props = sub_field_props[central_field_idx]
		return central_field_props, central_field, central_field_idx

	def global_threshold(self, A, prc=50, min_dist=5):
		"""
		Globally thresholds a ratemap and counts number of fields found
		"""
		Ac = A.copy()
		Ac[np.isnan(A)] = 0
		n = ny = 5
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		g = g / g.sum()
		Ac = signal.convolve(Ac, g, mode='same')
		maxRate = np.nanmax(np.ravel(Ac))
		Ac[Ac < maxRate*(prc/float(100))] = 0
		peak_mask = skimage.feature.peak_local_max(Ac, min_distance=min_dist,
											 exclude_border=False,
											 indices=False)
		peak_labels = skimage.measure.label(peak_mask, 8)
		field_labels = skimage.morphology.watershed(image=-Ac,
											  markers=peak_labels)
		nFields = np.max(field_labels)
		return nFields

	def local_threshold(self, A, prc=50, min_dist=5):
		"""
		Locally thresholds a ratemap to take only the surrounding prc amount
		around any local peak
		"""
		Ac = A.copy()
		nanidx = np.isnan(Ac)
		Ac[nanidx] = 0
		# smooth Ac more to remove local irregularities
		n = ny = 5
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		g = g / g.sum()
		Ac = signal.convolve(Ac, g, mode='same')
		peak_mask = skimage.feature.peak_local_max(Ac, min_distance=min_dist,
											 exclude_border=False,
											 indices=False)
		peak_labels = skimage.measure.label(peak_mask, 8)
		field_labels = skimage.morphology.watershed(image=-Ac,
											  markers=peak_labels)
		nFields = np.max(field_labels)
		sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
		sub_field_props = skimage.measure.regionprops(field_labels,
												intensity_image=Ac)
		sub_field_centroids = []
		sub_field_size = []

		for sub_field in sub_field_props:
			tmp = np.zeros(Ac.shape).astype(bool)
			tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
			tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
			sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
			sub_field_centroids.append(sub_field.centroid)
			sub_field_size.append(sub_field.area)  # in bins
		sub_field_mask = np.sum(sub_field_mask, 0)
		A_out = np.zeros_like(A)
		A_out[sub_field_mask.astype(bool)] = A[sub_field_mask.astype(bool)]
		A_out[nanidx] = np.nan
		return A_out

	def getBorderScore(self, A, B=None, shape='square', fieldThresh=0.3, smthKernSig=3,
					circumPrc=0.2, binSize=3.0, minArea=200, debug=False):
		"""
		Calculates a border score totally dis-similar to that calculated in Solstad et al
		(2008)

		Parameters
		----------
		A : array_like
			Should be the ratemap
		B : array_like
			This should be a boolean mask where True (1)
			is equivalent to the presence of a border and False (0)
			is equivalent to 'open space'. Naievely this will be the
			edges of the ratemap but could be used to take account of
			boundary insertions/ creations to check tuning to multiple
			environmental boundaries. Default None: when the mask is
			None then a mask is created that has 1's at the edges of the
			ratemap i.e. it is assumed that occupancy = environmental
			shape
		shape : str
			description of environment shape. Currently
			only 'square' or 'circle' accepted. Used to calculate the
			proportion of the environmental boundaries to examine for
			firing
		fieldThresh : float
			Between 0 and 1 this is the percentage
			amount of the maximum firing rate
			to remove from the ratemap (i.e. to remove noise)
		smthKernSig : float
			the sigma value used in smoothing the ratemap
			(again!) with a gaussian kernel
		circumPrc : float
			The percentage amount of the circumference
			of the environment that the field needs to be to count
			as long enough to make it through
		binSize : float
			bin size in cm
		minArea : float
			min area for a field to be considered
		debug : bool
			If True then some plots and text will be output

		Returns
		-------
		float : the border score

		Notes
		-----
		If the cell is a border cell (BVC) then we know that it should
		fire at a fixed distance from a given boundary (possibly more
		than one). In essence this algorithm estimates the amount of
		variance in this distance i.e. if the cell is a border cell this
		number should be small. This is achieved by first doing a bunch of
		morphological operations to isolate individual fields in the
		ratemap (similar to the code used in phasePrecession.py - see
		the partitionFields method therein). These partitioned fields are then
		thinned out (using skimage's skeletonize) to a single pixel
		wide field which will lie more or less in the middle of the
		(highly smoothed) sub-field. It is the variance in distance from the
		nearest boundary along this pseudo-iso-line that is the boundary measure

		Other things to note are that the pixel-wide field has to have some minimum
		length. In the case of a circular environment this is set to
		20% of the circumference; in the case of a square environment markers
		this is at least half the length of the longest side

		"""
		# need to know borders of the environment so we can see if a field
		# touches the edges, and the perimeter length of the environment
		# deal with square or circles differently
		borderMask = np.zeros_like(A)
		A_rows, A_cols = np.shape(A)
		if 'circle' in shape:
			radius = np.max(np.array(np.shape(A))) / 2.0
			dist_mask = skimage.morphology.disk(radius)
			if np.shape(dist_mask) > np.shape(A):
			    dist_mask = dist_mask[1:A_rows+1, 1:A_cols+1]
			tmp = np.zeros([A_rows + 2, A_cols + 2])
			tmp[1:-1, 1:-1] = dist_mask
			dists = ndimage.morphology.distance_transform_bf(tmp)
			dists = dists[1:-1, 1:-1]
			borderMask = np.logical_xor(dists <= 0, dists < 2)
			# open up the border mask a little
			borderMask = skimage.morphology.binary_dilation(
				borderMask, skimage.morphology.disk(1))
		elif 'square' in shape:
			borderMask[0:3, :] = 1
			borderMask[-3:, :] = 1
			borderMask[:, 0:3] = 1
			borderMask[:, -3:] = 1
			tmp = np.zeros([A_rows + 2, A_cols + 2])
			dist_mask = np.ones_like(A)
			tmp[1:-1, 1:-1] = dist_mask
			dists = ndimage.morphology.distance_transform_bf(tmp)
			# remove edges to make same shape as input ratemap
			dists = dists[1:-1, 1:-1]
		A[np.isnan(A)] = 0
		# get some morphological info about the fields in the ratemap
		# start image processing:
		# get some markers
		# NB I've tried a variety of techniques to optimise this part and the
		# best seems to be the local adaptive thresholding technique which)
		# smooths locally with a gaussian - see the skimage docs for more
		idx = A >= np.nanmax(np.ravel(A)) * fieldThresh
		A_thresh = np.zeros_like(A)
		A_thresh[idx] = A[idx]

		# label these markers so each blob has a unique id
		labels, nFields = ndimage.label(A_thresh)
		# remove small objects
		min_size = int(minArea / binSize) - 1
		if debug:
			plt.figure()
			plt.imshow(A_thresh)
			ax = plt.gca()
			ax.set_title('Before removing small objects')
		skimage.morphology.remove_small_objects(
			labels, min_size=min_size, connectivity=2, in_place=True)
		labels = skimage.segmentation.relabel_sequential(labels)[0]
		nFields = np.max(labels)
		if nFields == 0:
			return np.nan
		# Iterate over the labelled parts of the array labels calculating
		# how much of the total circumference of the environment edge it
		# covers

		fieldAngularCoverage = np.zeros([1, nFields]) * np.nan
		fractionOfPixelsOnBorder = np.zeros([1, nFields]) * np.nan
		fieldsToKeep = np.zeros_like(A)
		for i in range(1, nFields+1):
			fieldMask = np.logical_and(labels == i, borderMask)

			# check the angle subtended by the fieldMask
			if np.sum(fieldMask.astype(int)) > 0:
				s = skimage.measure.regionprops(
				fieldMask.astype(int), intensity_image=A_thresh)[0]
				x = s.coords[:, 0] - (A_cols / 2.0)
				y = s.coords[:, 1] - (A_rows / 2.0)
				subtended_angle = np.rad2deg(np.ptp(np.arctan2(x, y)))
				if subtended_angle > (360 * circumPrc):
					pixelsOnBorder = np.count_nonzero(
					fieldMask) / float(np.count_nonzero(labels == i))
					fractionOfPixelsOnBorder[:, i-1] = pixelsOnBorder
					if pixelsOnBorder > 0.5:
						fieldAngularCoverage[0, i-1] = subtended_angle

				fieldsToKeep = np.logical_or(fieldsToKeep, labels == i)
		if debug:
			_, ax = plt.subplots(4, 1, figsize=(3, 9))
			ax1 = ax[0]
			ax2 = ax[1]
			ax3 = ax[2]
			ax4 = ax[3]
			ax1.imshow(A)
			ax2.imshow(labels)
			ax3.imshow(A_thresh)
			ax4.imshow(fieldsToKeep)
			plt.show()
			for i, f in enumerate(fieldAngularCoverage.ravel()):
				print("angle subtended by field {0} = {1:.2f}".format(i+1, f))
			for i, f in enumerate(fractionOfPixelsOnBorder.ravel()):
				print("% pixels on border for field {0} = {1:.2f}".format(i+1, f))
		fieldAngularCoverage = (fieldAngularCoverage / 360.)
		if np.sum(fieldsToKeep) == 0:
			return np.nan
		rateInField = A[fieldsToKeep]
		# normalize firing rate in the field to sum to 1
		rateInField = rateInField / np.nansum(rateInField)
		dist2WallInField = dists[fieldsToKeep]
		Dm = np.dot(dist2WallInField, rateInField)
		if 'circle' in shape:
			Dm = Dm / radius
		elif 'square' in shape:
			Dm = Dm / (np.max(np.shape(A)) / 2.0)
		borderScore = (fractionOfPixelsOnBorder-Dm) / (fractionOfPixelsOnBorder+Dm)
		return np.max(borderScore)

	def get_field_props(self, A, min_dist=5, neighbours=2, prc=50,
					 plot=False, ax=None, tri=False, verbose=True, **kwargs):
		"""
		Returns a dictionary of properties of the field(s) in a ratemap A

		Parameters
		----------
		A : array_like
			a ratemap (but could be any image)
		min_dist : float
			the separation (in bins) between fields for measures
			such as field distance to make sense. Used to
			partition the image into separate fields in the call to
			skimage.feature.peak_local_max
		neighbours : int
			the number of fields to consider as neighbours to
			any given field. Defaults to 2
		prc : float
			percent of fields to consider
		ax : matplotlib.Axes
			user supplied axis. If None a new figure window is created
		tri : bool
			whether to do Delaunay triangulation between fields
			and add to plot
		verbose : bool
			dumps the properties to the console
		plot : bool
			whether to plot some output - currently consists of the
			ratemap A, the fields of which are outline in a black
			contour. Default False

		Returns
		-------
		result : dict
			The properties of the field(s) in the input ratemap A
		"""

		from scipy.spatial import Delaunay
		from skimage.measure import find_contours
		from sklearn.neighbors import NearestNeighbors
		import gridcell
		import matplotlib.cm as cm

		nan_idx = np.isnan(A)
		Ac = A.copy()
		Ac[np.isnan(A)] = 0
		# smooth Ac more to remove local irregularities
		n = ny = 5
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		g = g / g.sum()
		Ac = signal.convolve(Ac, g, mode='same')
		if 'clear_border' in kwargs.keys():
			clear_border = True
		else:
			clear_border = False
		peak_idx = skimage.feature.peak_local_max(Ac, min_distance=min_dist,
											exclude_border=clear_border,
											indices=True)
		if neighbours > len(peak_idx):
			print('neighbours value of {0} > the {1} peaks found'.format(neighbours, len(peak_idx)))
			print('Reducing neighbours to number of peaks found')
			neighbours = len(peak_idx)
		peak_mask = skimage.feature.peak_local_max(Ac, min_distance=min_dist, exclude_border=clear_border,
											 indices=False)
		peak_labels = skimage.measure.label(peak_mask, 8)
		field_labels = skimage.morphology.watershed(image=-Ac,
											  markers=peak_labels)
		nFields = np.max(field_labels)
		sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
		sub_field_props = skimage.measure.regionprops(field_labels,
												intensity_image=Ac)
		sub_field_centroids = []
		sub_field_size = []

		for sub_field in sub_field_props:
			tmp = np.zeros(Ac.shape).astype(bool)
			tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
			tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
			sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
			sub_field_centroids.append(sub_field.centroid)
			sub_field_size.append(sub_field.area)  # in bins
		sub_field_mask = np.sum(sub_field_mask, 0)
		contours = skimage.measure.find_contours(sub_field_mask, 0.5)
		# find the nearest neighbors to the peaks of each sub-field
		nbrs = NearestNeighbors(n_neighbors=neighbours,
								algorithm='ball_tree').fit(peak_idx)
		distances, _ = nbrs.kneighbors(peak_idx)
		mean_field_distance = np.mean(distances[:, 1:neighbours])

		nValid_bins = np.sum(~nan_idx)
		# calculate the amount of out of field firing
		A_non_field = np.zeros_like(A) * np.nan
		A_non_field[~sub_field_mask.astype(bool)] = A[~sub_field_mask.astype(bool)]
		A_non_field[nan_idx] = np.nan
		out_of_field_firing_prc = (np.count_nonzero(A_non_field > 0) / float(nValid_bins)) * 100
		Ac[np.isnan(A)] = np.nan
		"""
		get some stats about the field ellipticity
		"""
		_, central_field, _ = self.limit_to_one(A, prc=50)
		if central_field is None:
			ellipse_ratio = np.nan
		else:
			contour_coords = find_contours(central_field, 0.5)
			G = gridcell.SAC()
			a = G.__fit_ellipse__(contour_coords[0][:, 0], contour_coords[0][:, 1])
			ellipse_axes = G.__ellipse_axis_length__(a)
			ellipse_ratio = np.min(ellipse_axes) / np.max(ellipse_axes)
		""" using the peak_idx values calculate the angles of the triangles that
		make up a delaunay tesselation of the space if the calc_angles arg is
		in kwargs
		"""
		if 'calc_angs' in kwargs.keys():
			try:
				angs = self.calc_angs(peak_idx)
			except:
				angs = np.nan
		else:
			angs = None

		if plot:
			if ax is None:
				fig = plt.figure()
				ax = fig.add_subplot(111)
			else:
				ax = ax
			Am = np.ma.MaskedArray(Ac, mask=nan_idx, copy=True)
			ax.pcolormesh(Am, cmap=cm.jet, edgecolors='face')
			for c in contours:
				ax.plot(c[:, 1], c[:, 0], 'k')
			# do the delaunay thing
			if tri:
				tri = Delaunay(peak_idx)
				ax.triplot(peak_idx[:, 1], peak_idx[:, 0],
						   tri.simplices.copy(), color='w', marker='o')
			ax.set_xlim(0, Ac.shape[1] - 0.5)
			ax.set_ylim(0, Ac.shape[0] - 0.5)
			ax.set_xticklabels('')
			ax.set_yticklabels('')
			ax.invert_yaxis()
		props = {'Ac': Ac,
					'Peak_rate': np.nanmax(A),
					'Mean_rate': np.nanmean(A),
					'Field_size': np.mean(sub_field_size),
					'Pct_bins_with_firing': (np.sum(sub_field_mask) / nValid_bins) * 100,
					'Out_of_field_firing_prc': out_of_field_firing_prc,
					'Dist_between_fields': mean_field_distance,
					'Num_fields': float(nFields),
					'Sub_field_mask': sub_field_mask,
					'Smoothed_map': Ac,
					'field_labels': field_labels,
					'Peak_idx': peak_idx,
					'angles': angs,
					'contours': contours,
					'ellipse_ratio': ellipse_ratio}

		if verbose:
			print('\nPercentage of bins with firing: {:.2%}'.format(
			np.sum(sub_field_mask) / nValid_bins))
			print('Percentage out of field firing: {:.2%}'.format(
				np.count_nonzero(A_non_field > 0) / float(nValid_bins)))
			print('Peak firing rate: {:.3} Hz'.format(np.nanmax(A)))
			print('Mean firing rate: {:.3} Hz'.format(np.nanmean(A)))
			print('Number of fields: {0}'.format(nFields))
			print('Mean field size: {:.5} cm'.format(np.mean(sub_field_size)))  # 3 is binsize)
			print('Mean inter-peak distance between fields: {:.4} cm'.format(mean_field_distance))
		return props

	def calc_angs(self, points):
		"""
		Calculates the angles for all triangles in a delaunay tesselation of
		the peak points in the ratemap
		"""

		# calculate the lengths of the sides of the triangles
		sideLen = np.hypot(points[:, 0], points[:, 1])
		tri = spatial.Delaunay(points)
		indices, indptr = tri.vertex_neighbor_vertices
		nTris = tri.nsimplex
		outAngs = []
		for k in range(nTris):
			idx = indptr[indices[k]:indices[k+1]]
			a = sideLen[k]
			b = sideLen[idx[0]]
			c = sideLen[idx[1]]
			angA = self._getAng(a, b, c)
			angB = self._getAng(b, c, a)
			angC = self._getAng(c, a, b)
			outAngs.append((angA, angB, angC))
		return np.array(outAngs).T

	def _getAng(self, a, b, c):
		"""
		Given lengths a,b,c of the sides of a triangle this returns the angles
		in degress of all 3 angles
		"""
		return np.degrees(np.arccos((c**2 - b**2 - a**2)/(-2.0 * a * b)))

	def corr_maps(self, map1, map2, maptype='normal'):
		"""
		correlates two ratemaps together ignoring areas that have zero sampling
		"""
		if map1.shape > map2.shape:
			map2 = misc.imresize(map2, map1.shape, interp='nearest', mode='F')
		elif map1.shape < map2.shape:
			map1 = misc.imresize(map1, map2.shape, interp='nearest', mode='F')
		map1 = map1.flatten()
		map2 = map2.flatten()
		if maptype is 'normal':
			valid_map1 = np.logical_or((map1 > 0), ~np.isnan(map1))
			valid_map2 = np.logical_or((map2 > 0), ~np.isnan(map2))
		elif maptype is 'grid':
			valid_map1 = ~np.isnan(map1)
			valid_map2 = ~np.isnan(map2)
		valid = np.logical_and(valid_map1, valid_map2)
		r = np.corrcoef(map1[valid], map2[valid])
		if r.any():
			return r[1][0]
		else:
			return np.nan

	def coherence(self, smthd_rate, unsmthd_rate):
		"""calculates coherence of receptive field via correlation of smoothed
		and unsmoothed ratemaps
		"""
		smthd = smthd_rate.ravel()
		unsmthd = unsmthd_rate.ravel()
		si = ~np.isnan(smthd)
		ui = ~np.isnan(unsmthd)
		idx = ~(~si | ~ui)
		coherence = np.corrcoef(unsmthd[idx], smthd[idx])
		return coherence[1, 0]

	def kldiv_dir(self, polarPlot):
		"""
		Returns a kl divergence for directional firing: measure of directionality.
		Calculates kl diveregence between a smoothed ratemap (probably should be smoothed
		otherwise information theoretic measures don't 'care' about position of bins relative to
		one another) and a pure circular distribution. The larger the divergence the more
		tendancy the cell has to fire when the animal faces a specific direction.

		Parameters
		----------
		polarPlot: 1D-array
			The binned and smoothed directional ratemap

		Returns
		-------
		klDivergence: float
			The divergence from circular of the 1D-array from a uniform circular
			distribution
		"""

		__inc = 0.00001
		polarPlot = np.atleast_2d(polarPlot)
		polarPlot[np.isnan(polarPlot)] = __inc
		polarPlot[polarPlot == 0] = __inc
		normdPolar = polarPlot / float(np.nansum(polarPlot))
		nDirBins = polarPlot.shape[1]
		compCirc = np.ones_like(polarPlot) / float(nDirBins)
		kldivergence = self.kldiv(np.arange(0, nDirBins), normdPolar, compCirc)
		return kldivergence

	def kldiv(self, X, pvect1, pvect2, variant=None):
		"""
		Calculates the Kullback-Leibler or Jensen-Shannon divergence between two distributions.

		kldiv(X,P1,P2) returns the Kullback-Leibler divergence between two
		distributions specified over the M variable values in vector X.  P1 is a
		length-M vector of probabilities representing distribution 1, and P2 is a
		length-M vector of probabilities representing distribution 2.  Thus, the
		probability of value X(i) is P1(i) for distribution 1 and P2(i) for
		distribution 2.  The Kullback-Leibler divergence is given by:

		.. math:: KL(P1(x),P2(x)) = sum_[P1(x).log(P1(x)/P2(x))]

		If X contains duplicate values, there will be an warning message, and these
		values will be treated as distinct values.  (I.e., the actual values do
		not enter into the computation, but the probabilities for the two
		duplicate values will be considered as probabilities corresponding to
		two unique values.)  The elements of probability vectors P1 and P2 must
		each sum to 1 +/- .00001.

		kldiv(X,P1,P2,'sym') returns a symmetric variant of the Kullback-Leibler
		divergence, given by [KL(P1,P2)+KL(P2,P1)]/2

		kldiv(X,P1,P2,'js') returns the Jensen-Shannon divergence, given by
		[KL(P1,Q)+KL(P2,Q)]/2, where Q = (P1+P2)/2.  See the Wikipedia article
		for "Kullback–Leibler divergence".  This is equal to 1/2 the so-called
		"Jeffrey divergence."

		See Also
		--------
		Cover, T.M. and J.A. Thomas. "Elements of Information Theory," Wiley, 1991.

		https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

		Notes
		-----
		This function is taken from one on the Mathworks file exchange
		"""

		if not np.equal(np.unique(X), np.sort(X)).all():
			warnings.warn('X contains duplicate values. Treated as distinct values.',
				UserWarning)
		if not np.equal(np.shape(X), np.shape(pvect1)).all() or not np.equal(np.shape(X), np.shape(pvect2)).all():
			warnings.warn('All inputs must have the same dimension.', UserWarning)
		if (np.abs(np.sum(pvect1) - 1) > 0.00001) or (np.abs(np.sum(pvect2) - 1) > 0.00001):
			warnings.warn('Probabilities don''t sum to 1.', UserWarning)
		if variant:
			if variant == 'js':
				logqvect = np.log2((pvect2 + pvect1) / 2)
				KL = 0.5 * (np.nansum(pvect1 * (np.log2(pvect1) - logqvect)) +
							np.sum(pvect2 * (np.log2(pvect2) - logqvect)))
				return KL
			elif variant == 'sym':
				KL1 = np.nansum(pvect1 * (np.log2(pvect1) - np.log2(pvect2)))
				KL2 = np.nansum(pvect2 * (np.log2(pvect2) - np.log2(pvect1)))
				KL = (KL1 + KL2) / 2
				return KL
			else:
				warnings.warn('Last argument not recognised', UserWarning)
		KL = np.nansum(pvect1 * (np.log2(pvect1) - np.log2(pvect2)))
		return KL

	def skaggsInfo(self, ratemap, dwelltimes):
		"""
		Calculates Skaggs information measure

		Parameters
		----------
		ratemap : array_like
			The binned up ratemap
		dwelltimes: array_like
			Must be same size as ratemap

		Returns
		-------
		bits_per_spike : float
			Skaggs information score

		Notes
		-----
		THIS DATA SHOULD UNDERGO ADAPTIVE BINNING - See adaptiveBin in binning class above
		
		Returns Skaggs et al's estimate of spatial information in bits per spike:

		.. math:: I = sum_{x} p(x).r(x).log(r(x)/r)
		
		"""

		dwelltimes = dwelltimes / 50  # assumed sample rate of 50Hz
		if np.shape(ratemap) > 1:
			ratemap = np.reshape(ratemap, (np.prod(np.shape(ratemap)), 1))
			dwelltimes = np.reshape(dwelltimes, (np.prod(np.shape(dwelltimes)), 1))
		duration = np.nansum(dwelltimes)
		meanrate = np.nansum(ratemap * dwelltimes) / duration
		if meanrate <= 0.0:
			bits_per_spike = np.nan
			return bits_per_spike
		p_x = dwelltimes / duration
		p_r = ratemap / meanrate
		dum = p_x * ratemap
		ind = np.nonzero(dum)[0]
		bits_per_spike = np.nansum(dum[ind] * np.log2(p_r[ind]))
		bits_per_spike = bits_per_spike / meanrate
		return bits_per_spike
