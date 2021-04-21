"""
The classes contained in this module are supposed to be agnostic to recording
format and encapsulate some generic mechanisms for producing
things like spike timing autocorrelograms, power spectrum calculation and so on
"""
import numpy as np
from scipy import signal, spatial, ndimage, stats
from scipy.signal import gaussian
import skimage
from skimage import feature
from skimage.segmentation import watershed
import matplotlib.pylab as plt
from ephysiopy.common import binning
from ephysiopy.common.utils import bwperim
import warnings


class EventsGeneric(object):
    """
    Holds records of events, specifically for now, TTL events produced
    by either the Axona recording system or an Arduino-based plugin I
    (RH) wrote for the open-ephys recording system.

    Idea is to present a generic interface to other classes/ functions
    regardless of how the events were created.

    As a starting point lets base this on the dacq2py STM class which extends
    dict() and dacq2py.axonaIO.IO().

    For a fairly complete description of the nomenclature used for the
    timulation / event parameters see the STM property of the
    axonaIO.Stim() class

    Once a .stm file is loaded the keys for STM are:

    on: np.array
        time in samples of the event
    trial_date: str
    trial_time: str
    experimenter: str
    comments: str
    duration: str
    sw_version: str
    num_chans: str
    timebase: str
    bytes_per_timestamp: str
    data_format: str
    num_stm_samples: str
    posSampRate: int
    eegSampRate: int
    egfSampRate: int
    off: np.array
    stim_params: OrderedDict()
        This has keys:
            Phase_1: str
            Phase_2: str
            Phase_3: str
            etc
                Each of these keys is also a dict with keys:
                    startTime: None
                    duration: int (in seconds)
                    name: str
                    pulseWidth: int (microseconds)
                    pulseRatio: None
                    pulsePause: int (microseconds)

    The most important entries are the on and off numpy arrays and pulseWidth,
    the last mostly for plotting purposes.

    Let's emulate that dict generically so it can be co-opted for use with
    the various types of open-ephys recordings using the Arduino-based plugin
    (called StimControl - see https://github.com/rhayman/StimControl)
    """
    def __init__(self):
        level_one_keys = ['on', 'trial_date', 'trial_time', 'experimenter',
                          'comments', 'duration', 'sw_version', 'num_chans',
                          'timebase', 'bytes_per_timestamp', 'data_format',
                          'num_stm_samples', 'posSampRate', 'eegSampRate',
                          'egfSampRate', 'off', 'stim_params']
        level_two_keys = ['Phase_1', 'Phase_2', 'Phase_3']
        level_three_keys = ['startTime', 'duration', 'name', 'pulseWidth',
                            'pulseRatio', 'pulsePause']

        from collections import OrderedDict
        self.__event_dict = dict.fromkeys(
            level_one_keys)
        self.__event_dict['stim_params'] = OrderedDict.fromkeys(
            level_two_keys)
        for k in self.__event_dict['stim_params'].keys():
            self.__event_dict['stim_params'][k] = dict.fromkeys(
                level_three_keys)


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
        self.thetaRange = [6, 12]
        self.outsideRange = [3, 125]
        # for smoothing and plotting of power spectrum
        self.smthKernelWidth = 2
        self.smthKernelSigma = 0.1875
        self.sn2Width = 2
        self.maxFreq = 125
        self.maxPow = None

    def _nextpow2(self, val: int):
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

    def butterFilter(
            self, low: float, high: float, order: int = 5) -> np.ndarray:
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
        The signal is filtered in both the forward and
        reverse directions (scipy.signal.filtfilt)
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
        A 5-tuple of the following and sets a bunch of member variables:
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

        if 'pad2pow' not in kwargs:
            fftlen = int(np.power(2, self._nextpow2(origlen)))
        else:
            pad2pow = kwargs.pop('pad2pow')
            fftlen = int(np.power(2, pad2pow))

        freqs, power = signal.periodogram(
            self.sig, self.fs, return_onesided=True, nfft=fftlen)
        ffthalflen = fftlen / 2+1
        binsperhz = (ffthalflen-1) / nqlim
        kernelsigma = self.smthKernelSigma * binsperhz
        smthkernelsigma = 2 * int(4.0 * kernelsigma + 0.5) + 1
        gausswin = gaussian(smthkernelsigma, kernelsigma)
        sm_power = signal.fftconvolve(power, gausswin, 'same')
        sm_power = sm_power / np.sqrt(len(sm_power))
        spectrummaskband = np.logical_and(
            freqs > self.thetaRange[0], freqs < self.thetaRange[1])
        bandmaxpower = np.max(sm_power[spectrummaskband])
        maxbininband = np.argmax(sm_power[spectrummaskband])
        bandfreqs = freqs[spectrummaskband]
        freqatbandmaxpower = bandfreqs[maxbininband]
        self.freqs = freqs
        self.power = power
        self.sm_power = sm_power
        self.bandmaxpower = bandmaxpower
        self.freqatbandmaxpower = freqatbandmaxpower
        return freqs, power, sm_power, bandmaxpower, freqatbandmaxpower

    def plotEventEEG(
            self, event_ts, event_window=(-0.05, 0.1), stim_width=0.01,
            sample_rate=3e4):
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
        # nyq = sample_rate / 2
        highlim = 500
        if highlim >= sample_rate/2:
            highlim = (sample_rate-1)/2
        b, a = signal.butter(5, highlim, fs=sample_rate, btype='lowpass')
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
        ax.errorbar(
            np.linspace(
                event_window[0], event_window[1], len(mn)),
            mn,
            yerr=se, rasterized=False)
        ax.set_xlim(event_window)
        axTrans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        ax.add_patch(
            Rectangle(
                (0, 0), width=stim_width, height=1,
                transform=axTrans,
                color=[1, 1, 0], alpha=0.5))
        ax.set_ylabel(r'LFP ($\mu$V)')
        ax.set_xlabel('Time(s)')
        return fig

    def ifftFilter(self, sig, freqs, fs=250):
        """
        Calculates the dft of signal and filters out the frequencies in
        freqs from the result and reconstructs the original signal using
        the inverse fft without those frequencies

        Parameters
        ----------
        sig : np.array
            The LFP signal to be filtered
        freqs: list
            The frequencies to be filtered out
        fs: int
            The sampling frequency of sig

        Returns
        -------
        fftRes: np.array
            The filtered LFP signal
        """
        # from scipy import signal
        nyq = fs / 2.0
        fftRes = np.fft.fft(sig)
        f = nyq * np.linspace(0, 1, len(fftRes)/2)
        f = np.concatenate([f, f - nyq])

        band = 0.0625
        idx = np.zeros([len(freqs), len(f)]).astype(bool)

        for i, freq in enumerate(freqs):
            idx[i, :] = np.logical_and(np.abs(f) < freq+band, np.abs(f) >
                                       freq-band)

        pollutedIdx = np.sum(idx, 0)
        fftRes[pollutedIdx] = np.mean(fftRes)
        return fftRes

    def intrinsic_freq_autoCorr(
            self, spkTimes=None, posMask=None, maxFreq=25,
            acBinSize=0.002, acWindow=0.5, plot=True, posSampleFreq=30,
            spkSampleFreq=30000, **kwargs):
        """
        SEE EPHYSIOPY.COMMON.RHYTHMICITY.COSINEDIRECTIONALTUNING

        Calculates the intrinsic frequency autocorr of a cell

        Parameters
        ----------
        spkTimes: np.array
            times the cell fired in seconds
        posMask: logical mask
            Presumably a mask the same length as the number of pos samples
            where Trues are position samples you want to examine that are
            used to figure out which bit of the spike train to keep
        maxFreq: int
            maximum frequency to consider - used as input to power_spectrum()
        acBinSize: float
            The binsize of the autocorrelogram in seconds
        acWindow: float
            The window to look at in seconds

        Notes
        -----
        Be careful that if you've called dacq2py.Tetrode.getSpkTS()
        that they are divided by
        96000 to get into seconds before using here
        """

        pass


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
        self.sample_rate = 30

    def postprocesspos(self, tracker_params={}, **kwargs) -> tuple:
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
        Several internal functions are called here: speedfilter,
        interpnans, smoothPos and calcSpeed.
        Some internal state/ instance variables are set as well. The
        mask of the positional data (an instance of numpy masked array)
        is modified throughout this method.

        """
        xy = self.xy>
        if 'LeftBorder' in tracker_params:
            min_x = tracker_params['LeftBorder']
            xy[:, xy[0, :] <= min_x] = np.ma.masked
        if 'TopBorder' in tracker_params:
            min_y = tracker_params['TopBorder']  # y origin at top
            xy[:, xy[1, :] <= min_y] = np.ma.masked
        if 'RightBorder' in tracker_params:
            max_x = tracker_params['RightBorder']
            xy[:, xy[0, :] >= max_x] = np.ma.masked
        if 'BottomBorder' in tracker_params:
            max_y = tracker_params['BottomBorder']
            xy[:, xy[1, :] >= max_y] = np.ma.masked
        if 'SampleRate' in tracker_params.keys():
            self.sample_rate = int(tracker_params['SampleRate'])
        else:
            self.sample_rate = 30

        # xy = xy.T
        xy = self.speedfilter(xy)
        xy = self.interpnans(xy)  # ADJUST THIS SO NP.MASKED ARE INTERPOLATED
        xy = self.smoothPos(xy)
        self.calcSpeed(xy)

        import math
        pos2 = np.arange(0, self.npos-1)
        xy_f = xy.astype(float)
        self.dir[pos2] = np.mod(
            ((180/math.pi) * (np.arctan2(
                -xy_f[1, pos2+1] + xy_f[1, pos2], +xy_f[0, pos2+1]-xy_f[
                    0, pos2]))), 360)
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

        disp = np.hypot(xy[0, :], xy[1, :])
        disp = np.diff(disp, axis=0)
        disp = np.insert(disp, -1, 0)
        xy[:, np.abs(disp) > self.jumpmax] = np.ma.masked
        return xy

    def interpnans(self, xy):
        for i in range(0, np.shape(xy)[0], 2):
            missing = xy.mask.any(axis=0)
            ok = np.logical_not(missing)
            ok_idx = np.ravel(np.nonzero(np.ravel(ok))[0])
            missing_idx = np.ravel(np.nonzero(np.ravel(missing))[0])
            if len(missing_idx) > 0:
                try:
                    good_data = np.ravel(xy.data[i, ok_idx])
                    good_data1 = np.ravel(xy.data[i+1, ok_idx])
                    xy.data[i, missing_idx] = np.interp(
                        missing_idx, ok_idx, good_data)
                    xy.data[i+1, missing_idx] = np.interp(
                        missing_idx, ok_idx, good_data1)
                except ValueError:
                    pass
        xy.mask = 0
        print("{} bad positions were interpolated over".format(
            len(missing_idx)))
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
        x = xy[0, :].astype(np.float64)
        y = xy[1, :].astype(np.float64)

        from ephysiopy.common.utils import smooth
        # TODO: calculate window_len from pos sampling rate
        # 11 is roughly equal to 400ms at 30Hz (window_len needs to be odd)
        sm_x = smooth(x, window_len=11, window='flat')
        sm_y = smooth(y, window_len=11, window='flat')
        return np.ma.masked_array([sm_x, sm_y])

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
        speed = np.sqrt(np.sum(np.power(np.diff(xy), 2), 0))
        speed = np.append(speed, speed[-1])
        if self.cm:
            self.speed = speed * (100 * self.sample_rate / self.ppm)
            # in cm/s now
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

        new_x = signal.resample_poly(
            xy[0, :], upsample_rate/denom, 30/denom)
        new_y = signal.resample_poly(
            xy[1, :], upsample_rate/denom, 30/denom)
        return np.array([new_x, new_y])

    def filterPos(self, filter_dict: dict = {}):
        '''
        Filters data based on key/ values in filter_dict
        Meant to replicate a similar function in dacq2py_util.Trial
        called filterPos

        Parameters
        ----------
        filterDict : dict
            Contains the type(s) of filter to be used and the
            range of values to filter for. Values are pairs specifying the
            range of values to filter for NB can take multiple filters and
             iteratively apply them
            legal values are:
            * 'dir' - the directional range to filter for NB this can
                contain 'w','e','s' or 'n'
            * 'speed' - min and max speed to filter for
            * 'xrange' - min and max values to filter x pos values
            * 'yrange' - same as xrange but for y pos
            * 'time' - the times to keep / remove specified in ms

        Returns
        --------
        pos_index_to_keep : ndarray
            The position indices that should be kept
        '''
        if filter_dict is None:
            return
        bool_arr = np.ones(shape=(len(filter_dict), self.npos), dtype=np.bool)
        for idx, key in enumerate(filter_dict):
            if isinstance(filter_dict[key], str):
                if len(filter_dict[key]) == 1 and 'dir' in key:
                    if 'w' in filter_dict[key]:
                        filter_dict[key] = (135, 225)
                    elif 'e' in filter_dict[key]:
                        filter_dict[key] = (315, 45)
                    elif 's' in filter_dict[key]:
                        filter_dict[key] = (225, 315)
                    elif 'n' in filter_dict[key]:
                        filter_dict[key] = (45, 135)
                else:
                    raise ValueError("filter must contain a key / value pair")
            if 'speed' in key:
                if filter_dict[key][0] > filter_dict[key][1]:
                    raise ValueError("First value must be less \
                        than the second one")
                else:
                    bool_arr[idx, :] = np.logical_and(
                        self.speed > filter_dict[key][0],
                        self.speed < filter_dict[key][1])
            elif 'dir' in key:
                if filter_dict[key][0] < filter_dict[key][1]:
                    bool_arr[idx, :] = np.logical_and(
                        self.dir > filter_dict[key][0],
                        self.dir < filter_dict[key][1])
                else:
                    bool_arr[idx, :] = np.logical_or(
                        self.dir > filter_dict[key][0],
                        self.dir < filter_dict[key][1])
            elif 'xrange' in key:
                bool_arr[idx, :] = np.logical_and(
                    self.xy[0, :] > filter_dict[key][0],
                    self.xy[0, :] < filter_dict[key][1])
            elif 'yrange' in key:
                bool_arr[idx, :] = np.logical_and(
                    self.xy[1, :] > filter_dict[key][0],
                    self.xy[1, :] < filter_dict[key][1])
            elif 'time' in key:
                # takes the form of 'from' - 'to' times in SECONDS
                # such that only pos's between these ranges are KEPT
                filter_dict[key] = filter_dict[key] * self.sample_rate
                if filter_dict[key].ndim == 1:
                    bool_arr[idx, filter_dict[
                        key][0]:filter_dict[key][1]] = False
                else:
                    for i in filter_dict[key]:
                        bool_arr[idx, i[0]:i[1]] = False
                bool_arr = ~bool_arr
            else:
                print("Unrecognised key in dict")
                pass
        return np.expand_dims(np.any(~bool_arr, axis=0), 0)


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
    speed : array like
        The speed data, usually a 1D numpy array
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
    def __init__(
            self, xy, hdir, speed, pos_ts, spk_ts, plot_type='map', **kwargs):
        if (np.argmin(np.shape(xy)) == 1):
            xy = xy.T
        assert(xy.ndim == 2)
        assert(xy.shape[1] == hdir.shape[0] == speed.shape[0])
        self.xy = xy
        self.hdir = hdir
        self.speed = speed
        self.pos_ts = pos_ts
        if (spk_ts.ndim == 2):
            spk_ts = np.ravel(spk_ts)
        self.spk_ts = spk_ts  # All spike times regardless of cluster id
        if type(plot_type) is str:
            self.plot_type = [plot_type]
        else:
            self.plot_type = list(plot_type)
        self.spk_pos_idx = self.__interpSpkPosTimes__()
        self.__good_clusters = None
        self.__spk_clusters = None
        self.save_grid_output_location = None
        if ('ppm' in kwargs):
            self.__ppm = kwargs['ppm']
        else:
            self.__ppm = 400
        if 'pos_sample_rate' in kwargs:
            self.pos_sample_rate = kwargs['pos_sample_rate']
        else:
            self.pos_sample_rate = 30
        if 'save_grid_summary_location' in kwargs:
            self.save_grid_output_location = kwargs[
                'save_grid_summary_location']

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

    def __interpSpkPosTimes__(self):
        """
        Interpolates spike times into indices of position data
        NB Assumes pos times have been zeroed correctly - see comments in
        OEKiloPhy.OpenEphysNWB function __alignTimeStamps__()
        """
        idx = np.searchsorted(self.pos_ts, self.spk_ts)
        idx[idx == len(self.pos_ts)] = len(self.pos_ts) - 1
        return idx
    '''
    def plotAll(self, **kwargs):
        """
        Plots rate maps and other graphical output

        Notes
        ----
        This method uses the data provided to the class instance to plot
        various maps into a single figure window for each cluster. The things
        to plot are given in self.plot_type and the list of clusters in
        self.good_clusters
        """
        if 'all' in self.plot_type:
            what_to_plot = ['map', 'path', 'hdir', 'sac', 'speed', 'sp_hd']
            fig = plt.figure(figsize=(20, 10))
        else:
            fig = plt.gcf()
            if type(self.plot_type) is str:
                what_to_plot = [self.plot_type]  # turn into list
            else:
                what_to_plot = self.plot_type

        import matplotlib.gridspec as gridspec
        nrows = np.ceil(np.sqrt(len(self.good_clusters))).astype(int)
        outer = gridspec.GridSpec(nrows, nrows, figure=fig)

        inner_ncols = int(np.ceil(len(what_to_plot) / 2))  # max 2 cols
        if len(what_to_plot) == 1:
            inner_nrows = 1
        else:
            inner_nrows = 2

        try:
            iter(self.good_clusters)
        except Exception:
            self.good_clusters = [self.good_clusters]

        for i, cluster in enumerate(self.good_clusters):
            inner = gridspec.GridSpecFromSubplotSpec(
                inner_nrows, inner_ncols, subplot_spec=outer[i])
            for plot_type_idx, plot_type in enumerate(what_to_plot):
                if 'hdir' in plot_type:
                    ax = fig.add_subplot(
                        inner[plot_type_idx], projection='polar')
                else:
                    ax = fig.add_subplot(inner[plot_type_idx])
                if 'path' in plot_type:
                    self.makeSpikePathPlot(cluster, ax, **kwargs)
                if 'map' in plot_type:
                    self.makeRateMap(cluster, ax)
                if 'hdir' in plot_type:
                    self.makeHDPlot(cluster, ax, add_mrv=True, **kwargs)
                if 'sac' in plot_type:
                    self.makeSAC(cluster, ax)
                if 'speed' in plot_type:
                    self.makeSpeedVsRatePlot(cluster, ax, 0.0, 40.0, 3.0)
                if 'sp_hd' in plot_type:
                    self.makeSpeedVsHeadDirectionPlot(cluster, ax)
                ax.set_title(cluster, fontweight='bold', size=8)
        return fig
    '''
    def getSpatialStats(self, cluster):
        # HWPD 20200527
        """
        Adds summary of various spatial metrics in a dataframe

        Parameters
        ----------
        cluster : list, numpy array
            cell IDs to summarise (these will be recorded in the dataframe)
        """

        import pandas as pd
        try:
            iter(cluster)
        except Exception:
            cluster = [cluster]

        from ephysiopy.common import gridcell
        gridness, scale, orientation, HDtuning, HDangle, \
            speedCorr, speedMod = [], [], [], [], [], [], []
        for _, cl in enumerate(cluster):
            self.makeRateMap(cl, None)
            try:
                pos_w = np.ones_like(self.pos_ts)
                mapMaker = binning.RateMap(
                    self.xy, None, None, pos_w, ppm=self.ppm)
                spk_w = np.bincount(
                    self.spk_pos_idx, self.spk_clusters == cluster,
                    minlength=self.pos_ts.shape[0])
                rmap = mapMaker.getMap(spk_w)
                S = gridcell.SAC()
                nodwell = ~np.isfinite(rmap[0])
                sac = S.autoCorr2D(rmap[0], nodwell)
                m = S.getMeasures(sac)
            except Exception:
                m['gridness'] = np.nan
            try:
                r, th = self.getHDtuning(cl)
            except Exception:
                r = th = np.nan
            try:
                spC, spM = self.getSpeedTuning(cl)
            except Exception:
                spC = spM = np.nan

            gridness.append(m['gridness'])
            scale.append(m['scale'])
            orientation.append(m['orientation'])
            HDtuning.append(r)
            HDangle.append(th)
            speedCorr.append(spC)
            speedMod.append(spM)

        d = {
            'id': cluster, 'gridness': gridness, 'scale': scale,
            'orientation': orientation, 'HDtuning': HDtuning,
            'HDangle': HDangle, 'speedCorr': speedCorr, 'speedMod': speedMod}
        self.spatialStats = pd.DataFrame(d)
        return self.spatialStats

    def getHDtuning(self, cluster):
        # HWPD 20200527
        """
        Uses RH's head direction tuning function, just returns metric
        """
        from ephysiopy.common import statscalcs
        S = statscalcs.StatsCalcs()
        angles = self.hdir[self.spk_pos_idx[self.spk_clusters == cluster]]
        r, th = S.mean_resultant_vector(np.deg2rad(angles))
        return r, th

    def getSpeedTuning(self, cluster, minSpeed=0.0, maxSpeed=40.0, sigma=3.0):
        """
        Uses RH's speed tuning function, just returns metric and doesn't plot
        """
        # HWPD 20200527
        speed = np.ravel(self.speed)
        if np.nanmax(speed) < maxSpeed:
            maxSpeed = np.nanmax(speed)
        spd_bins = np.arange(minSpeed, maxSpeed, 1.0)
        # Construct the mask
        speed_filt = np.ma.MaskedArray(speed)
        speed_filt = np.ma.masked_where(speed_filt < minSpeed, speed_filt)
        speed_filt = np.ma.masked_where(speed_filt > maxSpeed, speed_filt)
        x1 = self.spk_pos_idx[self.spk_clusters == cluster]
        from ephysiopy.common.ephys_generic import SpikeCalcsGeneric
        S = SpikeCalcsGeneric(x1)
        spk_sm = S.smoothSpikePosCount(x1, self.pos_ts.shape[0], sigma, None)
        spk_sm = np.ma.MaskedArray(spk_sm, mask=np.ma.getmask(speed_filt))
        from scipy import stats
        speedCorr = stats.mstats.pearsonr(spk_sm, speed_filt)
        spd_dig = np.digitize(speed_filt, spd_bins, right=True)
        mn_rate = np.array([np.ma.mean(
            spk_sm[spd_dig == i]) for i in range(0, len(spd_bins))])
        speedCorr = speedCorr[0]  # HWPD 20200527
        speedCurve = mn_rate * self.pos_sample_rate
        speedMod = np.max(speedCurve) - np.min(speedCurve)  # HWPD 20200527
        return speedCorr, speedMod


class FieldCalcs:
    """
    This class differs from MapCalcsGeneric in that this one is mostly
    concerned with treating rate maps as images as opposed to using
    the spiking information contained within them. It therefore mostly
    deals with spatial rate maps of place and grid cells.
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
            ny = int(n)
        else:
            ny = int(ny)
        #  keep track of nans
        nan_idx = np.isnan(im)
        im[nan_idx] = 0
        if ftype == 'boxcar':
            if np.ndim(im) == 1:
                g = signal.boxcar(n) / float(n)
            elif np.ndim(im) == 2:
                g = signal.boxcar(n) / float(n)
                g = np.tile(g, (1, ny, 1))
                g = g / g.sum()
                g = np.squeeze(g)  # extra dim introduced in np.tile above
        elif ftype == 'gaussian':
            x, y = np.mgrid[-n:n+1, int(0-ny):ny+1]
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
        a) not connected to the border and b) close to the middle of the
        ratemap
        """
        Ac = A.copy()
        Ac[np.isnan(A)] = 0
        # smooth Ac more to remove local irregularities
        n = ny = 5
        x, y = np.mgrid[-n:n+1, -ny:ny+1]
        g = np.exp(-(x**2/float(n) + y**2/float(ny)))
        g = g / g.sum()
        Ac = signal.convolve(Ac, g, mode='same')
        peak_mask = feature.peak_local_max(
            Ac, min_distance=min_dist,
            exclude_border=False,
            indices=False)
        peak_labels = skimage.measure.label(peak_mask, 8)
        field_labels = watershed(
            image=-Ac, markers=peak_labels)
        nFields = np.max(field_labels)
        sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
        labelled_sub_field_mask = np.zeros_like(sub_field_mask)
        sub_field_props = skimage.measure.regionprops(
            field_labels, intensity_image=Ac)
        sub_field_centroids = []
        sub_field_size = []

        for sub_field in sub_field_props:
            tmp = np.zeros(Ac.shape).astype(bool)
            tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
            tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
            sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(
                tmp2, tmp)
            labelled_sub_field_mask[
                sub_field.label-1, np.logical_and(tmp2, tmp)] = sub_field.label
            sub_field_centroids.append(sub_field.centroid)
            sub_field_size.append(sub_field.area)  # in bins
        sub_field_mask = np.sum(sub_field_mask, 0)
        middle = np.round(np.array(A.shape) / 2)
        normd_dists = sub_field_centroids - middle
        field_dists_from_middle = np.hypot(
            normd_dists[:, 0], normd_dists[:, 1])
        central_field_idx = np.argmin(field_dists_from_middle)
        central_field = np.squeeze(
            labelled_sub_field_mask[central_field_idx, :, :])
        # collapse the labelled mask down to an 2d array
        labelled_sub_field_mask = np.sum(labelled_sub_field_mask, 0)
        # clear the border
        cleared_mask = skimage.segmentation.clear_border(central_field)
        # check we've still got stuff in the matrix or fail
        if ~np.any(cleared_mask):
            print(
                'No fields were detected away from edges so nothing returned')
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
        peak_mask = feature.peak_local_max(
            Ac, min_distance=min_dist,
            exclude_border=False,
            indices=False)
        peak_labels = skimage.measure.label(peak_mask, 8)
        field_labels = watershed(
            image=-Ac, markers=peak_labels)
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
        peak_mask = feature.peak_local_max(
            Ac, min_distance=min_dist, exclude_border=False,
            indices=False)
        peak_labels = skimage.measure.label(peak_mask, 8)
        field_labels = watershed(
            image=-Ac, markers=peak_labels)
        nFields = np.max(field_labels)
        sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
        sub_field_props = skimage.measure.regionprops(
            field_labels, intensity_image=Ac)
        sub_field_centroids = []
        sub_field_size = []

        for sub_field in sub_field_props:
            tmp = np.zeros(Ac.shape).astype(bool)
            tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
            tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
            sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(
                tmp2, tmp)
            sub_field_centroids.append(sub_field.centroid)
            sub_field_size.append(sub_field.area)  # in bins
        sub_field_mask = np.sum(sub_field_mask, 0)
        A_out = np.zeros_like(A)
        A_out[sub_field_mask.astype(bool)] = A[sub_field_mask.astype(bool)]
        A_out[nanidx] = np.nan
        return A_out

    def getBorderScore(
            self, A, B=None, shape='square', fieldThresh=0.3, smthKernSig=3,
            circumPrc=0.2, binSize=3.0, minArea=200, debug=False):
        """
        Calculates a border score totally dis-similar to that calculated in
        Solstad et al (2008)

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
        nearest boundary along this pseudo-iso-line that is the boundary
        measure

        Other things to note are that the pixel-wide field has to have some
        minimum length. In the case of a circular environment this is set to
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
                print("% pixels on border for \
                    field {0} = {1:.2f}".format(i+1, f))
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
        borderScore = (fractionOfPixelsOnBorder-Dm) / (
            fractionOfPixelsOnBorder+Dm)
        return np.max(borderScore)

    def get_field_props(
            self, A, min_dist=5, neighbours=2, prc=50,
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
            feature.peak_local_max
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

        from skimage.measure import find_contours
        from sklearn.neighbors import NearestNeighbors

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
        peak_idx = feature.peak_local_max(
            Ac, min_distance=min_dist,
            exclude_border=clear_border, indices=True)
        if neighbours > len(peak_idx):
            print('neighbours value of {0} > the {1} peaks found'.format(
                neighbours, len(peak_idx)))
            print('Reducing neighbours to number of peaks found')
            neighbours = len(peak_idx)
        peak_mask = feature.peak_local_max(
            Ac, min_distance=min_dist, exclude_border=clear_border,
            indices=False)
        peak_labels = skimage.measure.label(peak_mask, 8)
        field_labels = watershed(
            image=-Ac, markers=peak_labels)
        nFields = np.max(field_labels)
        sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
        sub_field_props = skimage.measure.regionprops(
            field_labels, intensity_image=Ac)
        sub_field_centroids = []
        sub_field_size = []

        for sub_field in sub_field_props:
            tmp = np.zeros(Ac.shape).astype(bool)
            tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
            tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
            sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(
                tmp2, tmp)
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
        A_non_field[~sub_field_mask.astype(bool)] = A[
            ~sub_field_mask.astype(bool)]
        A_non_field[nan_idx] = np.nan
        out_of_field_firing_prc = (np.count_nonzero(
            A_non_field > 0) / float(nValid_bins)) * 100
        Ac[np.isnan(A)] = np.nan
        """
        get some stats about the field ellipticity
        """
        _, central_field, _ = self.limit_to_one(A, prc=50)
        if central_field is None:
            ellipse_ratio = np.nan
        else:
            contour_coords = find_contours(central_field, 0.5)
            a = self.__fit_ellipse__(
                contour_coords[0][:, 0], contour_coords[0][:, 1])
            ellipse_axes = self.__ellipse_axis_length__(a)
            ellipse_ratio = np.min(ellipse_axes) / np.max(ellipse_axes)
        """ using the peak_idx values calculate the angles of the triangles that
        make up a delaunay tesselation of the space if the calc_angles arg is
        in kwargs
        """
        if 'calc_angs' in kwargs.keys():
            try:
                angs = self.calc_angs(peak_idx)
            except Exception:
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
            ax.pcolormesh(Am, cmap=plt.cm.get_cmap("jet"), edgecolors='face')
            for c in contours:
                ax.plot(c[:, 1], c[:, 0], 'k')
            # do the delaunay thing
            if tri:
                tri = spatial.Delaunay(peak_idx)
                ax.triplot(peak_idx[:, 1], peak_idx[:, 0],
                           tri.simplices.copy(), color='w', marker='o')
            ax.set_xlim(0, Ac.shape[1] - 0.5)
            ax.set_ylim(0, Ac.shape[0] - 0.5)
            ax.set_xticklabels('')
            ax.set_yticklabels('')
            ax.invert_yaxis()
        props = {
            'Ac': Ac, 'Peak_rate': np.nanmax(A), 'Mean_rate': np.nanmean(A),
            'Field_size': np.mean(sub_field_size),
            'Pct_bins_with_firing': (np.sum(
                sub_field_mask) / nValid_bins) * 100,
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
            print('Mean field size: {:.5} cm'.format(np.mean(sub_field_size)))
            print('Mean inter-peak distance between \
                fields: {:.4} cm'.format(mean_field_distance))
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
        in degress of the angle between the first 2 args
        """
        return np.degrees(np.arccos((c**2 - b**2 - a**2)/(-2.0 * a * b)))

    def corr_maps(self, map1, map2, maptype='normal'):
        """
        correlates two ratemaps together ignoring areas that have zero sampling
        """
        if map1.shape > map2.shape:
            map2 = skimage.transform.resize(map2, map1.shape, mode='reflect')
        elif map1.shape < map2.shape:
            map1 = skimage.transform.resize(map1, map2.shape, mode='reflect')
        map1 = map1.flatten()
        map2 = map2.flatten()
        if 'normal' in maptype:
            valid_map1 = np.logical_or((map1 > 0), ~np.isnan(map1))
            valid_map2 = np.logical_or((map2 > 0), ~np.isnan(map2))
        elif 'grid' in maptype:
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
        Returns a kl divergence for directional firing: measure of
        directionality.
        Calculates kl diveregence between a smoothed ratemap (probably
        should be smoothed otherwise information theoretic measures
        don't 'care' about position of bins relative to
        one another) and a pure circular distribution.
        The larger the divergence the more tendancy the cell has to fire
        when the animal faces a specific direction.

        Parameters
        ----------
        polarPlot: 1D-array
            The binned and smoothed directional ratemap

        Returns
        -------
        klDivergence: float
            The divergence from circular of the 1D-array from a
            uniform circular distribution
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
        Calculates the Kullback-Leibler or Jensen-Shannon divergence between
        two distributions.

        kldiv(X,P1,P2) returns the Kullback-Leibler divergence between two
        distributions specified over the M variable values in vector X.
        P1 is a length-M vector of probabilities representing distribution 1;
        P2 is a length-M vector of probabilities representing distribution 2.
         Thus, the probability of value X(i) is P1(i) for distribution 1 and
        P2(i) for distribution 2.

        The Kullback-Leibler divergence is given by:

        .. math:: KL(P1(x),P2(x)) = sum_[P1(x).log(P1(x)/P2(x))]

        If X contains duplicate values, there will be an warning message,
        and these values will be treated as distinct values.  (I.e., the
        actual values do not enter into the computation, but the probabilities
        for the two duplicate values will be considered as probabilities
        corresponding to two unique values.).
        The elements of probability vectors P1 and P2 must
        each sum to 1 +/- .00001.

        kldiv(X,P1,P2,'sym') returns a symmetric variant of the
        Kullback-Leibler divergence, given by [KL(P1,P2)+KL(P2,P1)]/2

        kldiv(X,P1,P2,'js') returns the Jensen-Shannon divergence, given by
        [KL(P1,Q)+KL(P2,Q)]/2, where Q = (P1+P2)/2.  See the Wikipedia article
        for "Kullback–Leibler divergence".  This is equal to 1/2 the so-called
        "Jeffrey divergence."

        See Also
        --------
        Cover, T.M. and J.A. Thomas. "Elements of Information Theory," Wiley,
        1991.

        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

        Notes
        -----
        This function is taken from one on the Mathworks file exchange
        """

        if not np.equal(np.unique(X), np.sort(X)).all():
            warnings.warn(
                'X contains duplicate values. Treated as distinct values.',
                UserWarning)
        if not np.equal(
            np.shape(X), np.shape(pvect1)).all() or not np.equal(
                np.shape(X), np.shape(pvect2)).all():
            warnings.warn(
                'All inputs must have the same dimension.', UserWarning)
        if (np.abs(
            np.sum(pvect1) - 1) > 0.00001) or (np.abs(
                np.sum(pvect2) - 1) > 0.00001):
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

    def skaggsInfo(self, ratemap, dwelltimes, **kwargs):
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
        THIS DATA SHOULD UNDERGO ADAPTIVE BINNING
        See adaptiveBin in binning class above

        Returns Skaggs et al's estimate of spatial information
        in bits per spike:

        .. math:: I = sum_{x} p(x).r(x).log(r(x)/r)

        """
        if 'sample_rate' in kwargs:
            sample_rate = kwargs['sample_rate']
        else:
            sample_rate = 50

        dwelltimes = dwelltimes / sample_rate  # assumed sample rate of 50Hz
        if ratemap.ndim > 1:
            ratemap = np.reshape(
                ratemap, (np.prod(np.shape(ratemap)), 1))
            dwelltimes = np.reshape(
                dwelltimes, (np.prod(np.shape(dwelltimes)), 1))
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

    def getGridFieldMeasures(
            self, A, maxima='centroid', field_extent_method=2, allProps=True,
            **kwargs):
        """
        Extracts various measures from a spatial autocorrelogram

        Parameters
        ----------
        A : array_like
            The spatial autocorrelogram (SAC)
        maxima : str, optional
            The method used to detect the peaks in the SAC.
            Legal values are 'single' and 'centroid'. Default 'centroid'
        field_extent_method : int, optional
            The method used to delimit the regions of interest in the SAC
            Legal values:
            * 1 - uses the half height of the ROI peak to limit field extent
            * 2 - uses a watershed method to limit field extent
            Default 2
        allProps : bool, optional
            Whether to return a dictionary that contains the attempt to fit an
            ellipse around the edges of the central size peaks. See below
            Default True

        Returns
        -------
        props : dict
            A dictionary containing measures of the SAC. Keys include:
            * gridness score
            * scale
            * orientation
            * coordinates of the peaks (nominally 6) closest to SAC centre
            * a binary mask around the extent of the 6 central fields
            * values of the rotation procedure used to calculate gridness
            * ellipse axes and angle (if allProps is True and the it worked)

        Notes
        -----
        The output from this method can be used as input to the show() method
        of this class.
        When it is the plot produced will display a lot more informative.

        See Also
        --------
        ephysiopy.common.binning.autoCorr2D()

        """
        A_tmp = A.copy()
        A_tmp[~np.isfinite(A)] = -1
        A_tmp[A_tmp <= 0] = -1
        A_sz = np.array(np.shape(A))
        # [STAGE 1] find peaks & identify 7 closest to centre
        if 'min_distance' in kwargs.keys():
            min_distance = kwargs.pop('min_distance')
        else:
            min_distance = np.ceil(np.min(A_sz / 2) / 8.).astype(int)
        import skimage.feature
        peaksMask = feature.peak_local_max(
            A_tmp, indices=False, min_distance=min_distance,
            exclude_border=False)
        import skimage
        peaksLabel = skimage.measure.label(peaksMask, connectivity=2)
        if maxima == 'centroid':
            S = skimage.measure.regionprops(peaksLabel)
            xyCoordPeaks = np.fliplr(
                np.array([(x['Centroid'][1], x['Centroid'][0]) for x in S]))
        elif maxima == 'single':
            xyCoordPeaks = np.fliplr(np.rot90(
                np.array(np.nonzero(
                    peaksLabel))))  # flipped so xy instead of yx
        # Convert so the origin is at the centre of the SAC
        centralPoint = np.ceil(A_sz/2).astype(int)
        xyCoordPeaksCentral = xyCoordPeaks - centralPoint
        # calculate distance of peaks from centre and find 7 closest
        # NB one is central peak - dealt with later
        peaksDistToCentre = np.hypot(
            xyCoordPeaksCentral[:, 1], xyCoordPeaksCentral[:, 0])
        orderOfClose = np.argsort(peaksDistToCentre)
        # Get id and coordinates of closest peaks1
        # NB closest peak at index 0 will be centre
        closestPeaks = orderOfClose[0:np.min((7, len(orderOfClose)))]
        closestPeaksCoord = xyCoordPeaks[closestPeaks, :]
        closestPeaksCoord = np.floor(closestPeaksCoord).astype(int)
        # [Stage 2] Expand peak pixels into the surrounding half-height region
        if field_extent_method == 1:
            peakLabel = np.zeros((A.shape[0], A.shape[1], len(closestPeaks)))
            perimeterLabel = np.zeros_like(peakLabel)
            for i in range(len(closestPeaks)):
                peakLabel[:, :, i], perimeterLabel[:, :, i] = \
                    self.__findPeakExtent__(
                    A, closestPeaks[i], closestPeaksCoord[i])
            fieldsLabel = np.max(peakLabel, 2)
            fieldsMask = fieldsLabel > 0
        elif field_extent_method == 2:
            # 2a find the inverse drainage bin for each peak
            fieldsLabel = watershed(image=-A_tmp, markers=peaksLabel)
            # 2b. Work out what threshold to use in each drainage-basin
            nZones = np.max(fieldsLabel.ravel())
            fieldIDs = fieldsLabel[
                closestPeaksCoord[:, 0], closestPeaksCoord[:, 1]]
            thresholds = np.ones((nZones, 1)) * np.inf
            # set thresholds for each sub-field at half-maximum
            thresholds[fieldIDs - 1, 0] = A[
                closestPeaksCoord[:, 0], closestPeaksCoord[:, 1]] / 2
            fieldsMask = np.zeros((A.shape[0], A.shape[1], nZones))
            for field in fieldIDs:
                sub = fieldsLabel == field
                fieldsMask[:, :, field-1] = np.logical_and(
                    sub, A > thresholds[field-1])
                # TODO: the above step can fragment a sub-field in
                # poorly formed SACs
                # need to deal with this...perhaps by only retaining
                # the largest  sub-sub-field
                labelled_sub_field = skimage.measure.label(
                    fieldsMask[:, :, field-1], connectivity=2)
                sub_props = skimage.measure.regionprops(labelled_sub_field)
                if len(sub_props) > 1:
                    distFromCentre = []
                    for s in range(len(sub_props)):
                        centroid = sub_props[s]['Centroid']
                        distFromCentre.append(
                            np.hypot(centroid[0]-A_sz[1], centroid[1]-A_sz[0]))
                    idx = np.argmin(distFromCentre)
                    tmp = np.zeros_like(A)
                    tmp[
                        sub_props[idx]['Coordinates'][:, 0],
                        sub_props[idx]['Coordinates'][:, 1]] = 1
                    fieldsMask[:, :, field-1] = tmp.astype(bool)
            fieldsMask = np.max(fieldsMask, 2).astype(bool)
            fieldsLabel[~fieldsMask] = 0
        fieldPerim = bwperim(fieldsMask)
        fieldsLabel = fieldsLabel.astype(int)
        # [Stage 3] Calculate a couple of metrics based on the closest peaks
        # Find the (mean) autoCorr value at the closest peak pixels
        nPixelsInLabel = np.bincount(fieldsLabel.ravel())
        sumRInLabel = np.bincount(fieldsLabel.ravel(), weights=A.ravel())
        meanRInLabel = sumRInLabel[closestPeaks+1] / nPixelsInLabel[
            closestPeaks+1]
        # get scale of grid
        closestPeakDistFromCentre = peaksDistToCentre[closestPeaks[1:]]
        scale = np.median(closestPeakDistFromCentre.ravel())
        # get orientation
        try:
            orientation = self.getGridOrientation(
                xyCoordPeaksCentral, closestPeaks)
        except Exception:
            orientation = np.nan
        # calculate gridness
        # THIS STEP MASKS THE MIDDLE AND OUTER PARTS OF THE SAC
        #
        # crop to the central region of the image and remove central peak
        x = np.linspace(-centralPoint[0], centralPoint[0], A_sz[0])
        y = np.linspace(-centralPoint[1], centralPoint[1], A_sz[1])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        dist2Centre = np.hypot(xx, yy)
        maxDistFromCentre = np.nan
        if len(closestPeaks) >= 7:
            maxDistFromCentre = np.max(dist2Centre[fieldsMask])
        if np.logical_or(
            np.isnan(
                maxDistFromCentre), maxDistFromCentre >
                np.min(np.floor(A_sz/2))):
            maxDistFromCentre = np.min(np.floor(A_sz/2))
        gridnessMaskAll = dist2Centre <= maxDistFromCentre
        centreMask = fieldsLabel == fieldsLabel[
            centralPoint[0], centralPoint[1]]
        gridnessMask = np.logical_and(gridnessMaskAll, ~centreMask)
        W = np.ceil(maxDistFromCentre).astype(int)
        autoCorrMiddle = A.copy()
        autoCorrMiddle[~gridnessMask] = np.nan
        autoCorrMiddle = autoCorrMiddle[
            -W + centralPoint[0]:W + centralPoint[0],
            -W+centralPoint[1]:W+centralPoint[1]]
        # crop the edges of the middle if there are rows/ columns of nans
        if np.any(np.all(np.isnan(autoCorrMiddle), 1)):
            autoCorrMiddle = np.delete(
                autoCorrMiddle, np.nonzero((np.all(
                    np.isnan(autoCorrMiddle), 1)))[0][0], 0)
        if np.any(np.all(np.isnan(autoCorrMiddle), 0)):
            autoCorrMiddle = np.delete(
                autoCorrMiddle, np.nonzero((np.all(
                    np.isnan(autoCorrMiddle), 0)))[0][0], 1)
        if 'step' in kwargs.keys():
            step = kwargs.pop('step')
        else:
            step = 30
        try:  # HWPD
            gridness, rotationCorrVals, rotationArr = self.getGridness(
                autoCorrMiddle, step=step)
        except Exception:  # HWPD
            gridness, rotationCorrVals, rotationArr = np.nan, np.nan, np.nan
        # attempt to fit an ellipse to the closest peaks
        if allProps:
            try:
                a = self.__fit_ellipse__(
                    closestPeaksCoord[1:, 0], closestPeaksCoord[1:, 1])
                im_centre = self.__ellipse_center__(a)
                ellipse_axes = self.__ellipse_axis_length__(a)
                ellipse_angle = self.__ellipse_angle_of_rotation__(a)
    #            ang =  ang + np.pi
                ellipseXY = self.__getellipseXY__(
                    ellipse_axes[0], ellipse_axes[1], ellipse_angle, im_centre)
                # get the min containing circle given the eliipse minor axis
                circleXY = self.__getcircleXY__(
                    im_centre, np.min(ellipse_axes))
            except Exception:
                im_centre = centralPoint
                ellipse_angle = None
                ellipse_axes = (None, None)
                ellipseXY = None
                circleXY = None
        else:
            ellipseXY = None
            circleXY = None
            ellipse_axes = None
            ellipse_angle = None
            im_centre = centralPoint
        # collect all the following keywords into a dict for output
        dictKeys = (
            'gridness', 'scale', 'orientation', 'closestPeaksCoord',
            'gridnessMaskAll', 'gridnessMask', 'ellipse_axes',
            'ellipse_angle', 'ellipseXY', 'circleXY', 'im_centre',
            'rotationArr', 'rotationCorrVals')
        outDict = dict.fromkeys(dictKeys, np.nan)
        for thiskey in outDict.keys():
            outDict[thiskey] = locals()[thiskey]
            # neat trick: locals is a dict holding all locally scoped variables
        return outDict

    def getGridOrientation(self, peakCoords, closestPeakIdx):
        """
        Calculates the orientation angle of a grid field.

        The orientation angle is the angle of the first peak working
        counter-clockwise from 3 o'clock

        Parameters
        ----------
        peakCoords : array_like
            The peak coordinates as pairs of xy
        closestPeakIdx : array_like
            A 1D array of the indices in peakCoords of the peaks closest
            to the centre of the SAC

        Returns
        -------
        peak_orientation : float
            The first value in an array of the angles of the peaks in the SAC
            working counter-clockwise from a line extending from the
            middle of the SAC to 3 o'clock.
        """
        if len(closestPeakIdx) == 1:
            return np.nan
        else:
            from .utils import polar
            closestPeaksCoordCentral = peakCoords[closestPeakIdx[1::]]
            theta = polar(
                closestPeaksCoordCentral[:, 1],
                -closestPeaksCoordCentral[:, 0], deg=1)[1]
            return np.sort(theta.compress(theta > 0))[0]

    def getGridness(self, image, step=30):
        """
        Calculates the gridness score in a grid cell SAC.

        Briefly, the data in `image` is rotated in `step` amounts and
        each rotated array is correlated with the original.
        The maximum of the values at 30, 90 and 150 degrees
        is the subtracted from the minimum of the values at 60, 120
        and 180 degrees to give the grid score.

        Parameters
        ----------
        image : array_like
            The spatial autocorrelogram
        step : int, optional
            The amount to rotate the SAC in each step of the rotational
            correlation procedure

        Returns
        -------
        gridmeasures : 3-tuple
            The gridscore, the correlation values at each `step` and
            the rotational array

        Notes
        -----
        The correlation performed is a Pearsons R. Some rescaling of the
        values in `image` is performed following rotation.

        See Also
        --------
        skimage.transform.rotate : for how the rotation of `image` is done
        skimage.exposure.rescale_intensity : for the resscaling following
        rotation

        """
        # TODO: add options in here for whether the full range of correlations
        # are wanted or whether a reduced set is wanted (i.e. at the 30-tuples)
        from collections import OrderedDict
        rotationalCorrVals = OrderedDict.fromkeys(
            np.arange(0, 181, step), np.nan)
        rotationArr = np.zeros(len(rotationalCorrVals)) * np.nan
        # autoCorrMiddle needs to be rescaled or the image rotation falls down
        # as values are cropped to lie between 0 and 1.0
        in_range = (np.nanmin(image), np.nanmax(image))
        out_range = (0, 1)
        import skimage
        autoCorrMiddleRescaled = skimage.exposure.rescale_intensity(
            image, in_range, out_range)
        origNanIdx = np.isnan(autoCorrMiddleRescaled.ravel())
        for idx, angle in enumerate(rotationalCorrVals.keys()):
            rotatedA = skimage.transform.rotate(
                autoCorrMiddleRescaled, angle=angle, cval=np.nan, order=3)
            # ignore nans
            rotatedNanIdx = np.isnan(rotatedA.ravel())
            allNans = np.logical_or(origNanIdx, rotatedNanIdx)
            # get the correlation between the original and rotated images
            rotationalCorrVals[angle] = stats.pearsonr(
                autoCorrMiddleRescaled.ravel()[~allNans],
                rotatedA.ravel()[~allNans])[0]
            rotationArr[idx] = rotationalCorrVals[angle]
        gridscore = np.min(
            (
                rotationalCorrVals[60],
                rotationalCorrVals[120])) - np.max(
                (
                    rotationalCorrVals[150],
                    rotationalCorrVals[30],
                    rotationalCorrVals[90]))
        return gridscore, rotationalCorrVals, rotationArr

    def deformSAC(self, A, circleXY=None, ellipseXY=None):
        """
        Deforms a SAC that is non-circular to be more circular

        Basically a blatant attempt to improve grid scores, possibly
        introduced in a paper by Matt Nolan...

        Parameters
        ----------
        A : array_like
            The SAC
        circleXY : array_like
            The xy coordinates defining a circle. Default None.
        ellipseXY : array_like
            The xy coordinates defining an ellipse. Default None.

        Returns
        -------
        deformed_sac : array_like
            The SAC deformed to be more circular

        See Also
        --------
        ephysiopy.common.ephys_generic.FieldCalcs.getGridFieldMeasures
        skimage.transform.AffineTransform
        skimage.transform.warp
        skimage.exposure.rescale_intensity
        """
        if circleXY is None or ellipseXY is None:
            SAC_stats = self.getGridFieldMeasures(A)
            circleXY = SAC_stats['circleXY']
            ellipseXY = SAC_stats['ellipseXY']
            # The ellipse detection stuff might have failed, if so
            # return the original SAC
            if circleXY is None:
                return A

        if circleXY.shape[0] == 2:
            circleXY = circleXY.T
        if ellipseXY.shape[0] == 2:
            ellipseXY = ellipseXY.T

        tform = skimage.transform.AffineTransform()
        try:
            tform.estimate(ellipseXY, circleXY)
        except np.linalg.LinAlgError:  # failed to converge
            print("Failed to estimate ellipse. Returning original SAC")
            return A

        """
        the transformation algorithms used here crop values < 0 to 0. Need to
        rescale the SAC values before doing the deformation and then rescale
        again so the values assume the same range as in the unadulterated SAC
        """
        A[np.isnan(A)] = 0
        SACmin = np.nanmin(A.flatten())
        SACmax = np.nanmax(A.flatten())  # should be 1 if autocorr
        AA = A + 1
        deformedSAC = skimage.transform.warp(
            AA / np.nanmax(AA.flatten()), inverse_map=tform.inverse, cval=0)
        return skimage.exposure.rescale_intensity(
            deformedSAC, out_range=(SACmin, SACmax))

    def __findPeakExtent__(self, A, peakID, peakCoord):
        """
        Finds extent of field that belongs to each peak.

        The extent is defined as the area that falls under the half-height.

        Parameters
        ----------
        A : array_like
            The SAC
        peakID : array_like
            I think this is a list of the peak identities i.e. [1, 2, 3 etc]
        peakCoord : array_like
            xy coordinates into A that contain the full peaks

        Returns
        -------
        out : 2-tuple
            Consisting of the labelled peaks and their labelled perimeters
        """
        peakLabel = np.zeros((A.shape[0], A.shape[1]))
        perimeterLabel = np.zeros_like(peakLabel)

        # define threshold to use - currently this is half-height
        halfHeight = A[peakCoord[1], peakCoord[0]] * .5
        aboveHalfHeightLabel = ndimage.label(
            A > halfHeight, structure=np.ones((3, 3)))[0]
        peakIDTmp = aboveHalfHeightLabel[peakCoord[1], peakCoord[0]]
        peakLabel[aboveHalfHeightLabel == peakIDTmp] = peakID
        perimeterLabel[bwperim(aboveHalfHeightLabel == peakIDTmp)] = peakID
        return peakLabel, perimeterLabel

    def __getcircleXY__(self, centre, radius):
        """
        Calculates xy coordinate pairs that define a circle

        Parameters
        ----------
        centre : array_like
            The xy coordinate of the centre of the circle
        radius : int
            The radius of the circle

        Returns
        -------
        circ : array_like
            100 xy coordinate pairs that describe the circle
        """
        npts = 100
        t = np.linspace(0+(np.pi/4), (2*np.pi)+(np.pi/4), npts)
        r = np.repeat(radius, npts)
        x = r * np.cos(t) + centre[1]
        y = r * np.sin(t) + centre[0]
        return np.array((x, y))

    def __getellipseXY__(self, a, b, ang, im_centre):
        """
        Calculates xy coordinate pairs that define an ellipse

        Parameters
        ----------
        a, b : float
            The major and minor axes of the ellipse respectively
        ang : float
            The angle of orientation of the ellipse
        im_centre : array_like
            The xy coordinate of the centre of the ellipse

        Returns
        -------
        ellipse : array_like
            100 xy coordinate pairs that describe the ellipse
        """
        pts = 100
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        theta = np.linspace(0, 2*np.pi, pts)
        X = a*np.cos(theta)*cos_a - sin_a*b*np.sin(theta) + im_centre[1]
        Y = a*np.cos(theta)*sin_a + cos_a*b*np.sin(theta) + im_centre[0]
        return np.array((X, Y))

    def __fit_ellipse__(self, x, y):
        """
        Does a best fits of an ellipse to the x/y coordinates provided

        Parameters
        ----------
        x, y : array_like
            The x and y coordinates

        Returns
        -------
        a : array_like
            The xy coordinate pairs that fit
        """
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:, n]
        return a

    def __ellipse_center__(self, a):
        """
        Finds the centre of an ellipse

        Parameters
        ----------
        a : array_like
            The values that describe the ellipse; major, minor axes etc

        Returns
        -------
        xy_centre : array_like
            The xy coordinates of the centre of the ellipse
        """
        b, c, d, f, _, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        num = b*b-a*c
        x0 = (c*d-b*f)/num
        y0 = (a*f-b*d)/num
        return np.array([x0, y0])

    def __ellipse_angle_of_rotation__(self, a):
        """
        Finds the angle of rotation of an ellipse

        Parameters
        ----------
        a : array_like
            The values that describe the ellipse; major, minor axes etc

        Returns
        -------
        angle : array_like
            The angle of rotation of the ellipse
        """
        b, c, _, _, _, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        return 0.5*np.arctan(2*b/(a-c))

    def __ellipse_axis_length__(self, a):
        """
        Finds the axis length of an ellipse

        Parameters
        ----------
        a : array_like
            The values that describe the ellipse; major, minor axes etc

        Returns
        -------
        axes_length : array_like
            The length of the major and minor axes (I think)
        """
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        _up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1 = (b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2 = (b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1 = np.sqrt(_up/np.abs(down1))
        res2 = np.sqrt(_up/np.abs(down2))
        return np.array([res1, res2])
