"""
The classes contained in this module are supposed to be agnostic to recording
format and encapsulate some generic mechanisms for producing
things like spike timing autocorrelograms, power spectrum calculation and so on
"""

import numpy as np
from scipy import signal
from scipy.interpolate import griddata
import astropy.convolution as cnv


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

    Attributes
    ----------
    on : np.array
        time in samples of the event
    trial_date : str
    trial_time : str
    experimenter : str
    comments : str
    duration : str
    sw_version : str
    num_chans : str
    timebase : str
    bytes_per_timestamp : str
    data_format : str
    num_stm_samples : str
    posSampRate : int
    eegSampRate : int
    egfSampRate : int
    off : np.ndarray
    stim_params : OrderedDict
        This has keys:
            Phase_1 : str
            Phase_2 : str
            Phase_3 : str
            etc
            Each of these keys is also a dict with keys:
                startTime: None
                duration: int
                    in seconds
                name: str
                pulseWidth: int
                    microseconds
                pulseRatio: None
                pulsePause: int
                    microseconds

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
    sig : np.ndarray
        The signal (of the LFP data)
    fs : float
        The sample rate
    """

    def __init__(self, sig, fs):
        self.sig = np.ma.MaskedArray(sig)
        self.fs = fs
        self.thetaRange = [6, 12]
        self.outsideRange = [3, 125]
        # for smoothing and plotting of power spectrum
        self.smthKernelWidth = 2
        self.smthKernelSigma = 0.1875
        self.sn2Width = 2
        self.maxFreq = 125
        self.maxPow = None

    def __add__(self, other):
        """
        Adds two EEGCalcsGeneric objects together

        Parameters
        ----------
        other : EEGCalcsGeneric
            The other EEGCalcsGeneric object to add

        Returns
        -------
        EEGCalcsGeneric
            A new EEGCalcsGeneric object with the combined sig and fs
        """
        if not isinstance(other, EEGCalcsGeneric):
            raise TypeError("Can only add another EEGCalcsGeneric object")

        new_sig = np.ma.concatenate((self.sig, other.sig))
        return EEGCalcsGeneric(new_sig, self.fs)

    def apply_mask(self, mask) -> None:
        """
        Applies a mask to the signal

        Parameters
        ----------
        mask : np.ndarray
            The mask to be applied. For use with np.ma.MaskedArray's mask attribute

        Notes
        -----
        If mask is empty, the mask is removed
        The mask should be a list of tuples, each tuple containing
        the start and end times of the mask i.e. [(start1, end1), (start2, end2)]
        everything inside of these times is masked
        """
        if np.any(mask):
            ratio = int(len(self.sig) / np.shape(mask)[1])
            mask = np.repeat(mask, ratio)
            # mask now might be shorter than self.sig so we need to pad it
            if len(mask) < len(self.sig):
                mask = np.pad(mask, (0, len(self.sig) - len(mask)))
            self.sig.mask = mask
        else:
            self.sig.mask = False

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

    def butterFilter(self, low: float, high: float, order: int = 5) -> np.ndarray:
        """
         Filters self.sig with a butterworth filter with a bandpass filter
         defined by low and high

        Parameters
        ----------
         low, high : float
             the lower and upper bounds of the bandpass filter
         order : int
             the order of the filter

         Returns
         -------
         filt : np.ndarray
             the filtered signal

         Notes
         -----
         the signal is filtered in both the forward and
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

        Returns
        -------
        psd : tuple[np.ndarray, float,...]
        A 5-tuple of the following and sets a bunch of member variables:
        freqs (array_like): The frequencies at which the spectrogram
        was calculated
        power (array_like): The power at the frequencies defined above
        sm_power (array_like): The smoothed power
        bandmaxpower (float): The maximum power in the theta band
        freqatbandmaxpower (float): The frequency at which the power
        is maximum
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
        gausswin = signal.windows.gaussian(smthkernelsigma, kernelsigma)
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
        sig : np.ndarray
            the LFP signal to be filtered
        freqs : list
            the frequencies to be filtered out
        fs : int
            the sampling frequency of sig

        Returns
        -------
        fftRes : np.ndarray
            the filtered LFP signal
        """
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
    x, y : np.ndarray
        the x and y positions
    ppm : int
        Pixels per metre
    convert2cm : bool
        Whether everything is converted into cms or not
    jumpmax : int
        Jumps in position (pixel coords) > than this are bad
    **kwargs:
        a dict[str, float] called 'tracker_params' is used to limit
        the range of valid xy positions - 'bad' positions are masked out
        and interpolated over

    Attributes
    ----------
    orig_xy : np.ndarray
        the original xy coordinates, never modified directly
    npos : int
        the number of position samples
    xy : np.ndarray
        2 x npos array
    convert2cm : bool
        whether to convert the xy position data to cms or not
    duration : float
        the trial duration in seconds
    xyTS : np.ndarray
        the timestamps the position data was recorded at. npos long vector
    dir : np.ndarray
        the directional data. In degrees
    ppm : float
        the number of pixels per metre
    jumpmax : float
        the minimum jump between consecutive positions before a jump is considered 'bad'
        and smoothed over
    speed : np.ndarray
        the speed data, extracted from a difference of xy positions. npos long vector
    sample_rate : int
        the sample rate of the position data

    Notes
    -----
    The positional data (x,y) is turned into a numpy masked array once this
    class is initialised - that mask is then modified through various
    functions (postprocesspos being the main one).
    """

    def __init__(
        self,
        x: np.ndarray | list,
        y: np.ndarray | list,
        ppm: float,
        convert2cm: bool = True,
        jumpmax: float = 100,
        **kwargs,
    ):
        assert np.shape(x) == np.shape(y)
        self.orig_xy: np.ma.MaskedArray = np.ma.MaskedArray([x, y])
        self._xy = np.ma.MaskedArray(np.zeros(shape=[2, len(x)]))
        self._xyTS = None
        self._dir = np.ma.MaskedArray(np.zeros_like(x))
        self._speed = np.ma.MaskedArray(np.zeros_like(x))
        self._ppm = ppm
        self._convert2cm = convert2cm
        self._jumpmax = jumpmax
        self.nleds = np.ndim(x)
        if "tracker_params" in kwargs:
            self.tracker_params = kwargs["tracker_params"]
        else:
            self.tracker_params = {}
        self._sample_rate = 30

    def __add__(self, other):
        """
        Adds two PosCalcsGeneric objects together

        Parameters
        ----------
        other : PosCalcsGeneric
            The other PosCalcsGeneric object to add

        Returns
        -------
        PosCalcsGeneric
            A new PosCalcsGeneric object with the combined xy data
        """
        if not isinstance(other, PosCalcsGeneric):
            raise TypeError("Can only add another PosCalcsGeneric object")
        new_xy = np.ma.concatenate((self.orig_xy, other.orig_xy), axis=1)
        P = PosCalcsGeneric(new_xy[0], new_xy[1], self.ppm, self.convert2cm)
        P._xyTS = (
            np.ma.concatenate((self.xyTS, other.xyTS + self.duration), axis=0)
            if self.xyTS is not None
            else None
        )
        return P

    @property
    def npos(self):
        return len(self.orig_xy.T)

    @property
    def xy(self) -> np.ma.MaskedArray:
        return self._xy

    @xy.setter
    def xy(self, value) -> None:
        self._xy = np.ma.MaskedArray(value)

    @property
    def convert2cm(self) -> bool:
        return self._convert2cm

    @convert2cm.setter
    def convert2cm(self, val) -> None:
        self._convert2cm = val
        self.postprocesspos(self.tracker_params)

    @property
    def duration(self) -> float:
        return self.npos / self.sample_rate

    @property
    def xyTS(self) -> np.ma.MaskedArray | None:
        return self._xyTS

    @xyTS.setter
    def xyTS(self, val) -> None:
        self._xyTS = np.ma.MaskedArray(val)

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
    def jumpmax(self) -> float:
        return self._jumpmax

    @jumpmax.setter
    def jumpmax(self, val) -> None:
        self._jumpmax = val
        self.postprocesspos(self.tracker_params)

    @property
    def speed(self) -> np.ma.MaskedArray:
        return self._speed

    @speed.setter
    def speed(self, value) -> None:
        self._speed = value

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, val) -> None:
        self._sample_rate = val

    def postprocesspos(self, tracker_params: "dict[str, float]" = {}):
        """
        Post-process position data

        Parameters
        ----------
        tracker_params : dict
            Same dict as created in OESettings.Settings.parse
            (from module openephys2py)

        Notes
        -----
        Several internal functions are called here: speedfilter,
        interpnans, smoothPos and calcSpeed.
        Some internal state/ instance variables are set as well. The
        mask of the positional data (an instance of numpy masked array)
        is modified throughout this method.
        """
        xy = self.orig_xy.copy()
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

        if self.convert2cm:
            xy = xy / (self._ppm / 100.0)
        xy = self.speedfilter(xy)
        xy = self.interpnans(xy)
        xy = self.smoothPos(xy)
        self.calcSpeed(xy)
        self.smooth_speed(self.speed)
        self._xy = xy
        self._dir = self.calcHeadDirection(xy)

    def calcHeadDirection(self, xy: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """
        Calculates the head direction from the xy data

        Parameters
        ----------
        xy : np.ma.MaskedArray
            The xy data

        Returns
        -------
        np.ma.MaskedArray
            The head direction data

        """
        # keep track of valid/ non-valid indices
        good = np.isfinite(xy[0].data)
        xy_f = xy.astype(float)
        self.dir = np.mod(
            np.arctan2(np.diff(xy_f[1]), np.diff(xy_f[0])) * (180 / np.pi), 360
        )
        self.dir = np.append(self.dir, self.dir[-1])
        self.dir[~good] = np.ma.masked
        return self.dir

    def speedfilter(self, xy: np.ma.MaskedArray):
        """
        Filters speed

        Args:
            xy (np.ma.MaskedArray): The xy data

        Returns:
            xy (np.ma.MaskedArray): The xy data with speeds >
            self.jumpmax masked
        """

        disp = np.hypot(xy[0], xy[1])
        disp = np.diff(disp, axis=0)
        disp = np.insert(disp, -1, 0)
        if self.convert2cm:
            jumpmax: float = self.ppm / self.jumpmax
        else:
            jumpmax: float = self.jumpmax
        jumps = np.abs(disp) > jumpmax
        x = xy[0]
        y = xy[1]
        x = np.ma.masked_where(jumps, x)
        y = np.ma.masked_where(jumps, y)
        if getattr(self, "mask_min_values", True):
            x = np.ma.masked_equal(x, np.min(x))
            y = np.ma.masked_equal(y, np.min(y))
        xy = np.ma.array([x, y])
        return xy

    def smooth_speed(self, speed: np.ma.MaskedArray, window_len: int = 21):
        """
        Smooth speed data with a window a little bit bigger than the usual
        400ms window used for smoothing position data

        NB Uses a box car filter as with Axona
        """
        g = cnv.Box1DKernel(window_len)
        speed = cnv.convolve(speed, g, boundary="extend")
        self.speed = np.ma.MaskedArray(speed)

    def interpnans(self, xy: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """
        Interpolates over bad values in the xy data

        Parameters
        ----------
        xy : np.ma.MaskedArray

        Returns
        -------
        np.ma.MaskedArray
            The interpolated xy data
        """
        n_masked = np.count_nonzero(xy.mask)
        if n_masked > 2:
            xm = xy[0]
            ym = xy[1]
            idx = np.arange(0, len(xm))
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
        ----------
        xy : np.ma.MaskedArray
            The xy positional data

        """
        speed = np.ma.MaskedArray(
            np.abs(np.ma.ediff1d(np.hypot(xy[0], xy[1]))))
        self.speed = np.append(speed, speed[-1])
        self.speed = self.speed * self.sample_rate

    def upsamplePos(self, xy: np.ma.MaskedArray, upsample_rate: int = 50):
        """
        Upsamples position data from 30 to upsample_rate

        Parameters
        ----------
        xy : np.ma.MaskedArray
            The xy positional data
        upsample_rate : int
            The rate to upsample to

        Returns
        -------
        np.ma.MaskedArray
            The upsampled xy positional data

        Notes
        -----
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

    def apply_mask(self, mask: np.ndarray):
        """
        Applies a mask to the position data

        Parameters
        ----------
        mask : np.ndarray
            The mask to be applied.

        Notes
        -----
        If mask is empty, the mask is removed
        The mask should be a list of tuples, each tuple containing
        the start and end times of the mask i.e. [(start1, end1), (start2, end2)]
        everything inside of these times is masked
        """
        self.xy.mask = mask
        self.xyTS.mask = mask
        self.dir.mask = mask
        self.speed.mask = mask


"""
Methods for quantifying data from AUX channels from the openephys
headstages.

Idea is to find periods of quiescence for ripple/ MUA/ replay analysis
"""


def downsample_aux(
    data: np.ndarray, source_freq: int = 30000, target_freq: int = 50, axis=-1
) -> np.ndarray:
    """
    Downsamples the default 30000Hz AUX signal to a default of 500Hz

    Parameters
    ----------
    data : np.ndarray
        the source data
    source_freq : int
        the sampling frequency of data
    target_freq : int
        the desired output frequency of the data
    axis : int
        the axis along which to apply the resampling

    Returns
    -------
    np.ndarray
        the downsampled data
    """
    denom = np.gcd(int(source_freq), int(target_freq))
    sig = signal.resample_poly(
        data.astype(float),
        target_freq / denom,
        source_freq / denom,
        axis,
        padtype="line",
    )
    return sig


def calculate_rms_and_std(
    sig: np.ndarray, time_window: list = [0, 10], fs: int = 50
) -> tuple:
    """
    Calculate the root mean square value for time_window (in seconds)

    Parameters
    ----------
    sig : np.ndarray
        the downsampled AUX data (single channel)
    time_window : list
        the range of times in seconds to calculate the RMS for
    fs: int
        the sampling frequency of sig

    Returns
    -------
    tuple of np.ndarray
        the RMS and standard deviation of the signal
    """
    rms = np.nanmean(
        np.sqrt(
            np.power(sig[int(time_window[0] * fs): int(time_window[1] * fs)], 2))
    )
    std = np.nanstd(
        np.sqrt(
            np.power(sig[int(time_window[0] * fs): int(time_window[1] * fs)], 2))
    )
    return rms, std


def find_high_amp_long_duration(
    raw_signal: np.ndarray,
    fs: int,
    amp_std: int = 3,
    duration_range: list = [0.03, 0.11],
    duration_std: int = 1,
) -> np.ma.MaskedArray:
    """
    Find periods of high amplitude and long duration in the ripple bandpass
    filtered signal.

    Parameters
    ----------
    raw_signal : np.ndarray
        the raw LFP signal which will be filtered here
    fs : int
        the sampliing frequency of the raw signal
    amp_std : int
        the signal needs to be this many standard deviations above the mean
    duration :list of int
        the minimum and maximum durations in seconds for the ripples
    duration_std :int
        how many standard deviations above the mean the ripples should
        be for 'duration' ms

    Returns
    -------
    np.ma.MaskedArray
        the bandpass filtered LFP that has been masked outside of epochs that don't meet the above thresholds

    Notes
    -----
    From Todorova & Zugaro (supp info):

    "To detect ripple events, we first detrended the LFP signals and used the Hilbert transform
    to compute the ripple band (100–250 Hz) amplitude for each channel recorded from the
    CA1 pyramidal layer. We then averaged these amplitudes, yielding the mean instanta-
    neous ripple amplitude. To exclude events of high spectral power not specific to the ripple
    band, we then subtracted the mean high-frequency (300–500 Hz) amplitude (if the differ-
    ence was negative, we set it to 0). Finally, we z-scored this signal, yielding a corrected
    and normalized ripple amplitude R(t). Ripples were defined as events where R(t) crossed
    a threshold of 3 s.d. and remained above 1 s.d. for 30 to 110 ms."

    References
    ----------
    Todorova & Zugaro, 2019. Isolated cortical computations during delta waves support memory consolidation. 366: 6463
    doi: 10.1126/science.aay0616
    """
    from scipy.signal import detrend, hilbert
    from ephysiopy.common.utils import get_z_score

    E = EEGCalcsGeneric(raw_signal, fs)
    detrended_lfp = detrend(raw_signal)
    E.sig = detrended_lfp

    def get_filtered_amplitude(E: EEGCalcsGeneric, filt: list) -> np.ndarray:
        f_lfp = E.butterFilter(filt[0], filt[1])
        analytic_lfp = hilbert(f_lfp)
        return np.abs(analytic_lfp)

    ripple_amplitude = get_filtered_amplitude(E, [100, 250])
    high_freq_amplitude = get_filtered_amplitude(E, [300, 499])

    amplitude_df = ripple_amplitude - high_freq_amplitude
    amplitude_df[amplitude_df < 0] = 0

    Rt = get_z_score(amplitude_df)
    Rt_mean = np.nanmean(Rt)
    Rt_std = np.nanstd(Rt)
    print(f"Rt mean: {Rt_mean}\nRt std: {Rt_std}")

    correct_duration_events = np.ma.masked_where(
        Rt < (Rt_mean + Rt_std * duration_std), Rt
    )
    candidate_slices = np.ma.clump_unmasked(correct_duration_events)
    mask = np.ones_like(Rt, dtype=bool)
    for i_slice in candidate_slices:
        dt = (i_slice.stop - i_slice.start) / fs
        if np.logical_and(dt > duration_range[0], dt < duration_range[1]):
            if np.any(Rt[i_slice] > (Rt_mean + Rt_std * amp_std)):
                mask[i_slice] = False

    return np.ma.MaskedArray(Rt, mask=mask)
