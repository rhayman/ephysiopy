from collections import namedtuple
from collections.abc import Sequence
from scipy import interpolate
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from ephysiopy.common.utils import TrialFilter


# a namedtuple to hold some metrics from the KiloSort run
KSMetaTuple = namedtuple("KSMeta", "Amplitude group KSLabel ContamPct ")

# ── Sample rates ────────────────────────────────────────────────────────────
FS_50K = 50_000  # Hz
FS_30K = 30_000  # Hz


def peak_to_trough_time(
    ap: np.ndarray,
    fs: int,
    search_window_ms: float = 2.0,
) -> dict:
    """
    Calculate the peak-to-trough time of a single action potential waveform.

    Parameters
    ----------
    ap : np.ndarray
        1-D array containing one action potential waveform (voltage, µV or mV).
    fs : int
        Sampling frequency in Hz (e.g. 50_000 or 30_000).
    search_window_ms : float
        How many milliseconds after the peak to search for the trough.
        Default is 2.0 ms (covers typical neuronal APs).

    Returns
    -------
    dict with keys:
        peak_idx        – sample index of the peak
        trough_idx      – sample index of the trough
        peak_to_trough_samples  – difference in samples
        peak_to_trough_ms       – difference in milliseconds
        peak_value      – voltage at the peak
        trough_value    – voltage at the trough
    """
    samples_per_ms = fs / 1_000.0
    search_window_samples = int(np.round(search_window_ms * samples_per_ms))

    # ── Peak: global maximum within the waveform ────────────────────────────
    peak_idx = int(np.argmax(ap))

    # ── Trough: minimum AFTER the peak, within the search window ────────────
    search_end = min(peak_idx + search_window_samples, len(ap))
    if peak_idx >= search_end:
        raise ValueError(
            f"Peak is at the very end of the snippet (idx={peak_idx}); "
            "extend the waveform or reduce search_window_ms."
        )

    post_peak = ap[peak_idx:search_end]
    trough_idx = peak_idx + int(np.argmin(post_peak))

    # ── Timing ───────────────────────────────────────────────────────────────
    delta_samples = trough_idx - peak_idx
    delta_ms = delta_samples / samples_per_ms

    return {
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "peak_to_trough_samples": delta_samples,
        "peak_to_trough_ms": delta_ms,
        "peak_value": ap[peak_idx],
        "trough_value": ap[trough_idx],
    }


def get_param(waveforms, param="Amp", t=200, fet=1, **kws) -> np.ndarray:
    """
    Returns the requested parameter from a spike train as a numpy array.

    Parameters
    ----------
    waveforms : np.ndarray
        Array shape can be nSpikes x nSamples OR
        nSpikes x nElectrodes x nSamples.
    param : str, default='Amp'
        Valid values are:
        - 'Amp': peak-to-trough amplitude
        - 'P': height of peak
        - 'T': depth of trough
        - 'Vt': height at time t
        - 'tP': time of peak (in seconds)
        - 'tT': time of trough (in seconds)
        - 'PCA': first n fet principal components (defaults to 1)
    t : int, default=200
        The time used for Vt
    fet : int, default=1
        The number of principal components (use with param 'PCA').

    **kws : dict
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        The requested parameter as a numpy array.

    """
    from sklearn.decomposition import PCA

    spike_window = kws.get("spike_window", 1000)

    if param == "Amp":
        return np.ptp(waveforms, axis=-1)
    elif param == "P":
        return np.max(waveforms, axis=-1)
    elif param == "T":
        return np.min(waveforms, axis=-1)
    elif param == "Vt":
        idx = t if t < waveforms.shape[-1] else waveforms.shape[-1] - 1
        return waveforms[..., idx]
    elif param == "tP":
        idx = np.argmax(waveforms, axis=-1)
        return idx / waveforms.shape[-1] / spike_window
    elif param == "tT":
        idx = np.argmin(waveforms, axis=-1)
        return idx / waveforms.shape[-1] / spike_window
    elif param == "PCA":
        pca = PCA(n_components=fet)
        if waveforms.ndim == 2:
            return pca.fit_transform(waveforms).squeeze()
        elif waveforms.ndim == 3:
            out = []
            for i in range(waveforms.shape[1]):
                wf = waveforms[:, i, :]
                if not np.any(np.isnan(wf)):
                    pc = pca.fit_transform(wf)
                    out.append(pc)
            return np.concatenate(out, axis=1)


def get_peak_to_trough_time(waveforms: np.ndarray, spike_window=1000) -> np.ndarray:
    """
    Returns the time in seconds of the peak to trough in a waveform.

    Parameters
    ----------
    waveforms : np.ndarray
        The waveforms to calculate the peak to trough time for.

    Returns
    -------
    np.ndarray
        The time of the peak to trough in seconds.
    """
    peak_times = get_param(waveforms, "tP", spike_window=spike_window)
    trough_times = get_param(waveforms, "tT", spike_window=spike_window)
    return np.mean(trough_times - peak_times)


class WaveformCalcsGeneric(object):
    """
    Deals with the processing and analysis of spike data.
    There should be one instance of this class per cluster in the
    recording session. NB this differs from previous versions of this
    class where there was one instance per recording session and clusters
    were selected by passing in the cluster id to the methods.

    NB Axona waveforms are nSpikes x nChannels x nSamples - this boils
    down to nSpikes x 4 x 50
    NB KiloSort waveforms are nSpikes x nSamples x nChannels - these are ordered
    by 'best' channel first and then the rest of the channels. This boils
    down to nSpikes x 61 x 12 SO THIS NEEDS TO BE CHANGED to
    nSpikes x nChannels x nSamples

    Parameters
    ----------
    spike_times : np.ndarray
        The times of spikes in the trial in seconds.
    cluster : int
        The cluster ID.
    waveforms : np.ndarray, optional
        An nSpikes x nChannels x nSamples array.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    spike_times : np.ma.MaskedArray
        The times of spikes in the trial in seconds.
    _waves : np.ma.MaskedArray or None
        The waveforms of the spikes.
    cluster : int
        The cluster ID.
    n_spikes : int
        the total number of spikes for the current cluster
    duration : float, int
        total duration of the trial in seconds
    event_ts : np.ndarray or None
        The times that events occurred in seconds.
    event_window : np.ndarray
        The window, in seconds, either side of the stimulus, to examine.
    stim_width : float or None
        The width, in ms, of the stimulus.
    secs_per_bin : float
        The size of bins in PSTH.
    sample_rate : int
        The sample rate of the recording.
    pos_sample_rate : int
        The sample rate of the position data.
    pre_spike_samples : int
        The number of samples before the spike.
    post_spike_samples : int
        The number of samples after the spike.
    KSMeta : KSMetaTuple
        The metadata from KiloSort.
    """

    def __init__(
        self,
        waveforms: np.ndarray,
        spike_times: np.ndarray,
        cluster: int,
        **kwargs,
    ):
        self.spike_times = np.ma.MaskedArray(spike_times)  # IN SECONDS
        # if waveforms.shape[-1] > 50:
        # this is a hack to deal with the fact that KiloSort waveforms are 82 samples long
        # and I want them comparable with Axona which is 50
        # waveforms = waveforms[:, :, 16:66]
        n_spikes, n_channels, n_samples = waveforms.shape
        assert self.n_spikes == n_spikes, (
            "Number of spike times does not match number of waveforms"
        )
        self._waves = np.ma.MaskedArray(waveforms)
        self.cluster = cluster
        self._event_ts = None  # the times that events occured IN SECONDS
        # window, in seconds, either side of the stimulus, to examine
        self._event_window = np.array((-0.050, 0.100))
        self._stim_width = None  # the width, in ms, of the stimulus
        # used to increase / decrease size of bins in psth
        self._secs_per_bin = 0.001
        self._sample_rate = 50000
        self._pos_sample_rate = 50
        self._duration = None
        self._invert_waveforms = False
        # these values should be specific to Axona data
        # this I think is wrong as the pre capture buffer
        # should be 200ms and 800ms post trigger
        # whereas these values are equal to 330 microseconds
        # and 708 microseconds if sampling is at 48kHz
        self._pre_spike_samples = 10
        self._post_spike_samples = 40
        # if waveforms is not None:
        #     if self.n_samples > 50:
        # self.pre_spike_samples = 41
        # self.post_spike_samples = 41
        # self.sample_rate = 30000
        # these values are specific to OE data
        # if sample rate is 3kHz and the number of samples
        # captured per waveform is 82 then it looks from
        # plotting the waveforms that the pre spike samples is
        # 40 so 1.33 milliseconds and the pos spike samples is
        # 1.40 milliseconds
        # values from running KS
        self._ksmeta = KSMetaTuple(None, None, None, None)
        # update the __dict__ attribute with the kwargs
        self.__dict__.update(kwargs)

    @property
    def sample_rate(self) -> int | float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int | float) -> None:
        self._sample_rate = value

    @property
    def pos_sample_rate(self) -> int | float:
        return self._pos_sample_rate

    @pos_sample_rate.setter
    def pos_sample_rate(self, value: int | float) -> None:
        self._pos_sample_rate = value

    @property
    def pre_spike_samples(self) -> int:
        return self._pre_spike_samples

    @pre_spike_samples.setter
    def pre_spike_samples(self, value: int) -> None:
        self._pre_spike_samples = int(value)

    @property
    def post_spike_samples(self) -> int:
        return self._post_spike_samples

    @post_spike_samples.setter
    def post_spike_samples(self, value: int) -> None:
        self._post_spike_samples = int(value)

    # its possible due to referencing that waveforms are inverted
    # so add the option to correct that here
    @property
    def invert_waveforms(self) -> bool:
        return self._invert_waveforms

    @invert_waveforms.setter
    def invert_waveforms(self, val: bool) -> None:
        self._invert_waveforms = val

    def waveforms(self, channel_id: Sequence = None) -> np.ndarray | None:
        """
        Returns the waveforms of the cluster.

        Parameters
        ----------
        channel_id : Sequence, optional
            The channel IDs to return the waveforms for.
            If None, returns waveforms for all channels.

        Returns
        -------
        np.ndarray | None
            The waveforms of the cluster,
            or None if no waveforms are available.


        """
        if self._waves is not None:
            scaling = 1
            if self.invert_waveforms:
                scaling = -1
            if channel_id is None:
                return self._waves[:, :, :] * scaling
            else:
                if isinstance(channel_id, int):
                    channel_id = [channel_id]
                return self._waves[:, channel_id, :] * scaling
        else:
            return None

    @property
    def n_spikes(self):
        """
        Returns the number of spikes in the cluster

        Returns
        -------
        int
            The number of spikes in the cluster
        """
        return np.ma.count(self.spike_times)

    @property
    def n_channels(self) -> int | None:
        """
        Returns the number of channels in the waveforms.

        Returns
        -------
        int | None
            The number of channels in the waveforms,
            or None if no waveforms are available.
        """
        if self._waves is not None:
            return self._waves.shape[1]
        else:
            return None

    @property
    def n_samples(self) -> int | None:
        """
        Returns the number of samples in the waveforms.

        Returns
        -------
        int | None
            The number of samples in the waveforms,
            or None if no waveforms are available.
        """
        if self._waves is not None:
            return self._waves.shape[2]
        else:
            return None

    @property
    def event_ts(self) -> np.ndarray:
        return self._event_ts

    @event_ts.setter
    def event_ts(self, value: np.ndarray) -> None:
        self._event_ts = value

    @property
    def duration(self) -> float | int | None:
        return self._duration

    @duration.setter
    def duration(self, value: float | int | None):
        self._duration = value

    @property
    def KSMeta(self) -> KSMetaTuple:
        return self._ksmeta

    def update_KSMeta(self, value: dict) -> None:
        """
        Takes in a TemplateModel instance from a phy session and
        parses out the relevant metrics for the cluster and places
        into the namedtuple KSMeta.

        Parameters
        ----------
        value : dict
            A dictionary containing the relevant metrics for the cluster.

        """
        metavals = []
        for f in KSMetaTuple._fields:
            if f in value.keys():
                if self.cluster in value[f].keys():
                    metavals.append(value[f][self.cluster])
                else:
                    metavals.append(None)
            else:
                metavals.append(None)
        self._ksmeta = KSMetaTuple(*metavals)

    @property
    def event_window(self) -> np.ndarray:
        return self._event_window

    @event_window.setter
    def event_window(self, value: np.ndarray):
        self._event_window = value

    @property
    def stim_width(self) -> int | float | None:
        return self._stim_width

    @stim_width.setter
    def stim_width(self, value: int | float | None):
        self._stim_width = value

    @property
    def secs_per_bin(self) -> float | int:
        return self._secs_per_bin

    @secs_per_bin.setter
    def secs_per_bin(self, value: float | int):
        self._secs_per_bin = value

    def apply_filter(self, *trial_filter: TrialFilter) -> None:
        """
        Applies a mask to the spike times.

        Parameters
        ----------
        trial_filter : TrialFilter
            The filter
        """
        if trial_filter:
            for i_filter in trial_filter:
                assert isinstance(i_filter, TrialFilter), "Filter must be a TrialFilter"
                assert i_filter.name == "time", "Only time filters are supported"
        self.spike_times.mask = False
        if np.any(self._waves) and self._waves is not None:
            self._waves.mask = False
        if not trial_filter or len(trial_filter) == 0:
            if np.any(self._waves) and self._waves is not None:
                self._waves.mask = False
            self.spike_times.mask = False
        else:
            mask = np.zeros_like(self.spike_times, dtype=bool)
            for i_filter in trial_filter:
                i_mask = np.logical_and(
                    self.spike_times > i_filter.start, self.spike_times < i_filter.end
                )
                mask = np.logical_or(mask, i_mask)
            self.spike_times.mask = mask
            if self._waves is not None:
                self._waves.mask = mask

    def trial_mean_fr(self) -> float:
        # Returns the trial mean firing rate for the cluster
        if self.duration is None:
            raise IndexError("No duration provided, give me one!")
        return self.n_spikes / self.duration

    def mean_waveform(self, channel_id: Sequence = None):
        """
        Returns the mean waveform and standard error of the mean (SEM) for a
        given spike train on a particular channel.

        Parameters
        ----------
        channel_id : Sequence, optional
            The channel IDs to return the mean waveform for. If None, returns
            mean waveforms for all channels.

        Returns
        -------
        tuple
            A tuple containing:
            - mn_wvs (np.ndarray): The mean waveforms, usually 4x50 for tetrode recordings.
            - std_wvs (np.ndarray): The standard deviations of the waveforms, usually 4x50 for tetrode recordings.
        """
        x = self.waveforms(channel_id)
        if x is not None:
            return np.mean(x, axis=0), np.std(x, axis=0)
        else:
            return None

    def get_best_channel(self) -> int | None:
        """
        Returns the channel with the highest mean amplitude of the waveforms.

        Returns
        -------
        int | None
            The index of the channel with the highest mean amplitude,
            or None if no waveforms are available.
        """
        wvs = self.waveforms()
        if wvs is not None:
            amps = np.mean(np.ptp(wvs, axis=-1), axis=0)
            return np.argmax(amps)
        else:
            return None

    def estimate_AHP(self) -> float | None:
        """
        Estimate the decay time for the AHP of the waveform of the
        best channel for the current cluster.

        Returns
        -------
        float | None
            The estimated AHP decay time in microseconds,
            or None if no waveforms are available.
        """
        best_chan = self.get_best_channel()

        if not best_chan:
            return None

        waveform, _ = self.mean_waveform(best_chan)

        if waveform is None:
            return None

        # get the times
        times = np.linspace(0, 1000, self.n_samples)  # in microseconds
        pre_spike = int(self.pre_spike_samples * (1000000 / self.sample_rate))
        post_spike = int(self.post_spike_samples * (1000000 / self.sample_rate))

        f = interpolate.interp1d(
            times, np.linspace(-(pre_spike), post_spike, self.n_samples), "nearest"
        )
        f_t = f(times)

        # get the baseline voltage (for Axona this should be the
        # 200 microseconds before the spike)
        baseline_voltage = np.mean(waveform[f_t < 0])
        # make sure this min measure isafter the spike would
        # have triggered the capture of the buffer i.e. after t=0
        min_voltage = np.min(waveform[f_t >= 0])
        half_voltage = (baseline_voltage + min_voltage) / 2

        # get the index of the minima of the waveform
        t_thresh = np.nonzero(f_t > 0)[0][0]
        idx = np.argmin(waveform[f_t > 0]) + t_thresh

        # find the time it takes to return to baseline voltage
        X = np.atleast_2d(f_t[idx:]).T
        y = np.atleast_2d(waveform[idx:]).T

        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        lin2 = LinearRegression()
        lin2.fit(X_poly, y)

        tn = np.linspace(f_t[idx], 1500, 1000).reshape(-1, 1)
        yn = lin2.predict(poly.fit_transform(tn))

        # predict the time it takes to return to half voltage
        ahp_time = None
        for i, v in enumerate(yn):
            if v >= half_voltage:
                ahp_time = tn[i][0]
                break

        try:
            return ahp_time - f_t[idx]
        except Exception:
            return None

    def plot_waveforms(self, n_waveforms: int = 2000, n_channels: int = 4):
        """
        Plots the waveforms of the cluster.

        Parameters
        ----------
        n_waveforms : int, optional
            The number of waveforms to plot.
        n_channels : int, optional
            The number of channels to plot.

        Returns
        -------
        None
        """
        if self._waves is None:
            raise ValueError("No waveforms available for this cluster.")

        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(n_channels, 1, figsize=(5, 10))
        for i in range(n_channels):
            axes[i].plot(self.waveforms()[:n_waveforms, i, :].T, c="gray")
            # plot mean waveform on top
            axes[i].plot(np.mean(self.waveforms()[:n_waveforms, i, :], axis=0), c="red")
            axes[i].set_title(f"Channel {i}")
            axes[i].set_xlabel("Time (ms)")
            axes[i].set_ylabel("Amplitude (uV)")
        plt.tight_layout()
        plt.show()
