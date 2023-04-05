"""
The classes contained in this module are supposed to be agnostic to recording
format and encapsulate some generic mechanisms for producing
things like spike timing autocorrelograms, power spectrum calculation and so on
"""
import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.interpolate import griddata


class EventsGeneric(object):
    """
    Holds records of events, specifically for now, TTL events produced
    by either the Axona recording system or an Arduino-based plugin I
    (RH) wrote for the open-ephys recording system.

    Idea is to present a generic interface to other classes/ functions
    regardless of how the events were created.

    As a starting point lets base this on the axona STM class which extends
    dict() and axona.axonaIO.IO().

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
    timebase: strxy_ts[3] = juce::uint32(frameTime * 1e6);
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
        level_one_keys = [
            "on",
            "trial_date",
            "trial_time",
            "experimenter",
            "comments",
            "duration",
            "sw_version",
            "num_chans",
            "timebase",
            "bytes_per_timestamp",
            "data_format",
            "num_stm_samples",
            "posSampRate",
            "eegSampRate",
            "egfSampRate",
            "off",
            "stim_params",
        ]
        level_two_keys = ["Phase_1", "Phase_2", "Phase_3"]
        level_three_keys = [
            "startTime",
            "duration",
            "name",
            "pulseWidth",
            "pulseRatio",
            "pulsePause",
        ]

        from collections import OrderedDict

        self._event_dict = dict.fromkeys(level_one_keys)
        self._event_dict["stim_params"] = OrderedDict.fromkeys(level_two_keys)
        for k in self._event_dict["stim_params"].keys():
            self._event_dict["stim_params"][k] = dict.fromkeys(
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
                     self,
                     low: float,
                     high: float,
                     order: int = 5) -> np.ndarray:
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
        b, a = signal.butter(order, [lowcut, highcut], btype="band")
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

        if "pad2pow" not in kwargs:
            fftlen = int(np.power(2, self._nextpow2(origlen)))
        else:
            pad2pow = kwargs.pop("pad2pow")
            fftlen = int(np.power(2, pad2pow))

        freqs, power = signal.periodogram(
            self.sig, self.fs, return_onesided=True, nfft=fftlen
        )
        ffthalflen = fftlen / 2 + 1
        binsperhz = (ffthalflen - 1) / nqlim
        kernelsigma = self.smthKernelSigma * binsperhz
        smthkernelsigma = 2 * int(4.0 * kernelsigma + 0.5) + 1
        gausswin = signal.gaussian(smthkernelsigma, kernelsigma)
        sm_power = signal.fftconvolve(power, gausswin, "same")
        sm_power = sm_power / np.sqrt(len(sm_power))
        spectrummaskband = np.logical_and(
            freqs > self.thetaRange[0], freqs < self.thetaRange[1]
        )
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
        f = nyq * np.linspace(0, 1, int(len(fftRes) / 2))
        f = np.concatenate([f, f - nyq])

        band = 0.0625
        idx = np.zeros([len(freqs), len(f)]).astype(bool)

        for i, freq in enumerate(freqs):
            idx[i, :] = np.logical_and(
                np.abs(f) < freq + band, np.abs(f) > freq - band)

        pollutedIdx = np.sum(idx, 0)
        fftRes[pollutedIdx] = np.mean(fftRes)
        return fftRes


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
    **kwargs:
        a dict[str, float] called 'tracker_params' is used to limit the
        range of valid xy positions - 'bad' positions are masked out
        and interpolated over

    Notes
    -----
    The positional data (x,y) is turned into a numpy masked array once this
    class is initialised - that mask is then modified through various
    functions (postprocesspos being the main one).
    """

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        ppm: float,
        cm: bool = True,
        jumpmax: float = 100,
        **kwargs,
    ):
        assert np.shape(x) == np.shape(y)
        self.orig_xy: np.ma.MaskedArray = np.ma.MaskedArray([x, y])
        self._xy = None
        self._xyTS = None
        self._dir = np.ma.MaskedArray(np.zeros_like(x))
        self._speed = None
        self._ppm = ppm
        self.cm = cm
        self._jumpmax = jumpmax
        self.nleds = np.ndim(x)
        self.npos = len(x)
        if "tracker_params" in kwargs:
            self.tracker_params = kwargs["tracker_params"]
        else:
            self.tracker_params = {}
        self._sample_rate = 30

    @property
    def xy(self) -> np.ma.MaskedArray:
        return self._xy

    @xy.setter
    def xy(self, value) -> None:
        self._xy: np.ma.MaskedArray = value

    @property
    def xyTS(self):
        return self._xyTS

    @xyTS.setter
    def xyTS(self, val):
        self._xyTS = val

    @property
    def dir(self) -> np.ma.MaskedArray:
        return self._dir

    @dir.setter
    def dir(self, value) -> None:
        self._dir: np.ma.MaskedArray = value

    @property
    def ppm(self) -> float:
        return self._ppm

    @ppm.setter
    def ppm(self, value) -> None:
        self._ppm: float = value
        self.postprocesspos(self.tracker_params)

    @property
    def jumpmax(self):
        return self._jumpmax

    @jumpmax.setter
    def jumpmax(self, val):
        self._jumpmax = val
        self.postprocesspos(self.tracker_params)

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = value

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, val):
        self._sample_rate = val

    def postprocesspos(
                       self,
                       tracker_params: "dict[str, float]" = {},
                       **kwargs) -> None:
        """
        Post-process position data

        Parameters
        ----------
        tracker_params : dict
            Same dict as created in OESettings.Settings.parse
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
        xy = self.orig_xy
        x_zero = xy[0, :] < 0
        y_zero = xy[1, :] < 0
        xy[:, np.logical_or(x_zero, y_zero)] = np.ma.masked

        self.tracker_params: dict[str, float] = tracker_params
        if "AxonaBadValue" in tracker_params:
            bad_val = tracker_params["AxonaBadValue"]
            xy = np.ma.masked_equal(xy, bad_val)
        if "LeftBorder" in tracker_params:
            min_x = tracker_params["LeftBorder"]
            xy[:, xy[0, :] <= min_x] = np.ma.masked
        if "TopBorder" in tracker_params:
            min_y = tracker_params["TopBorder"]  # y origin at top
            xy[:, xy[1, :] <= min_y] = np.ma.masked
        if "RightBorder" in tracker_params:
            max_x = tracker_params["RightBorder"]
            xy[:, xy[0, :] >= max_x] = np.ma.masked
        if "BottomBorder" in tracker_params:
            max_y = tracker_params["BottomBorder"]
            xy[:, xy[1, :] >= max_y] = np.ma.masked
        if "SampleRate" in tracker_params:
            self.sample_rate = int(tracker_params["SampleRate"])
        else:
            self.sample_rate = 30

        if self.cm:
            xy = xy / (self._ppm / 100.0)

        xy = self.speedfilter(xy)
        xy = self.interpnans(xy)
        xy = self.smoothPos(xy)
        self.calcSpeed(xy)
        self._xy = xy
        self._dir = self.calcHeadDirection(xy)

    def calcHeadDirection(self, xy: np.ma.MaskedArray) -> np.ma.MaskedArray:
        import math

        pos2 = np.arange(0, self.npos - 1)
        xy_f = xy.astype(float)
        self.dir[pos2] = np.mod(
            (
                (180 / math.pi)
                * (
                    np.arctan2(
                        -xy_f[1, pos2 + 1] + xy_f[1, pos2],
                        +xy_f[0, pos2 + 1] - xy_f[0, pos2],
                    )
                )
            ),
            360,
        )
        self.dir[-1] = self.dir[-2]
        return self.dir

    def speedfilter(self, xy: np.ma.MaskedArray):
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

        disp: np.ma.MaskedArray = np.hypot(xy[0], xy[1])
        disp: np.ma.MaskedArray = np.diff(disp, axis=0)
        disp: np.ma.MaskedArray = np.insert(disp, -1, 0)
        if self.cm:
            jumpmax: float = self.ppm / self.jumpmax
        else:
            jumpmax: float = self.jumpmax
        jumps: np.ma.MaskedArray = np.abs(disp) > jumpmax
        x: np.ma.MaskedArray = xy[0]
        y: np.ma.MaskedArray = xy[1]
        x: np.ma.MaskedArray = np.ma.masked_where(jumps, x)
        y: np.ma.MaskedArray = np.ma.masked_where(jumps, y)
        if getattr(self, "mask_min_values", True):
            x: np.ma.MaskedArray = np.ma.masked_equal(x, np.min(x))
            y: np.ma.MaskedArray = np.ma.masked_equal(y, np.min(y))
        xy = np.ma.array([x, y])
        return xy

    def interpnans(self, xy: np.ma.MaskedArray) -> np.ma.MaskedArray:
        n_masked: int = np.count_nonzero(xy.mask)
        if n_masked > 2:
            xm: np.ma.MaskedArray = xy[0]
            ym: np.ma.MaskedArray = xy[1]
            idx: np.ndarray = np.arange(0, len(xm))
            xi = griddata(idx[~xm.mask], xm[~xm.mask], idx, method="linear")
            yi = griddata(idx[~ym.mask], ym[~ym.mask], idx, method="linear")
            print(f"Interpolated over {n_masked} bad values")
            return np.ma.MaskedArray([xi, yi])
        else:
            return xy

    def smoothPos(self, xy: np.ma.MaskedArray):
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
        sm_x = smooth(x, window_len=11, window="flat")
        sm_y = smooth(y, window_len=11, window="flat")
        return np.ma.masked_array([sm_x, sm_y])

    def calcSpeed(self, xy: np.ma.MaskedArray):
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
        speed = np.abs(np.ma.ediff1d(np.hypot(xy[0], xy[1])))
        self.speed = np.append(speed, speed[-1])
        if self.cm:
            self.speed = self.speed * self.sample_rate

    def upsamplePos(self, xy: np.ma.MaskedArray, upsample_rate: int = 50):
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
        -----E =
        This is mostly to get pos data recorded using PosTracker at 30Hz
        into Axona format 50Hz data
        """
        from scipy import signal

        denom = np.gcd(upsample_rate, 30)

        new_x = signal.resample_poly(
            xy[0, :], upsample_rate / denom, 30 / denom)
        new_y = signal.resample_poly(
            xy[1, :], upsample_rate / denom, 30 / denom)
        return np.array([new_x, new_y])

    def filterPos(self, filt: dict = {}):
        """
        Filters data based on key/ values in filt
        Meant to replicate a similar function in axona_util.Trial
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
        """
        if filt is None:
            self.xy.mask = False
            self.dir.mask = False
            self.speed.mask = False
            return False
        bool_arr = np.ones(shape=(len(filt), self.npos), dtype=bool)
        for idx, key in enumerate(filt):
            if isinstance(filt[key], str):
                if len(filt[key]) == 1 and "dir" in key:
                    if "w" in filt[key]:
                        filt[key] = (135, 225)
                    elif "e" in filt[key]:
                        filt[key] = (315, 45)
                    elif "s" in filt[key]:
                        filt[key] = (225, 315)
                    elif "n" in filt[key]:
                        filt[key] = (45, 135)
                else:
                    raise ValueError("filter must contain a key / value pair")
            if "speed" in key:
                if filt[key][0] > filt[key][1]:
                    raise ValueError(
                        "First value must be less \
                        than the second one"
                    )
                else:
                    bool_arr[idx, :] = np.logical_and(
                        self.speed > filt[key][0],
                        self.speed < filt[key][1],
                    )
            elif "dir" in key:
                if filt[key][0] < filt[key][1]:
                    bool_arr[idx, :] = np.logical_and(
                        self.dir > filt[key][0], self.dir < filt[key][1]
                    )
                else:
                    bool_arr[idx, :] = np.logical_or(
                        self.dir > filt[key][0], self.dir < filt[key][1]
                    )
            elif "xrange" in key:
                bool_arr[idx, :] = np.logical_and(
                    self.xy[0, :] > filt[key][0],
                    self.xy[0, :] < filt[key][1],
                )
            elif "yrange" in key:
                bool_arr[idx, :] = np.logical_and(
                    self.xy[1, :] > filt[key][0],
                    self.xy[1, :] < filt[key][1],
                )
            elif "time" in key:
                # takes the form of 'from' - 'to' times in SECONDS
                # such that only pos's between these ranges are KEPT
                for i in filt[key]:
                    bool_arr[idx, i * self.sample_rate: i *
                             self.sample_rate] = False
                bool_arr = ~bool_arr
            else:
                raise KeyError("Unrecognised key")
        mask = np.expand_dims(np.any(~bool_arr, axis=0), 0)
        self.xy.mask = mask
        self.dir.mask = mask
        self.speed.mask = mask
        return mask
