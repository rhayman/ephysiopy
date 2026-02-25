import os
import warnings
import matplotlib.pylab as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path, PurePath
from ephysiopy.common.ephys_generic import EEGCalcsGeneric

from ephysiopy.common.utils import PowerSpectrumParams, window_rms, find_runs, nextpow2
from ephysiopy.openephys2py.KiloSort import KiloSortSession
from ephysiopy.visualise.plotting import FigureMaker, saveFigure

from scipy import signal

"""
Dataclass for collecting the results of frequency/ phase
analysis of LFP

See LFPOscillations.getFreqPhase()
"""


def power_spectrum(
    params: PowerSpectrumParams,
    plot=True,
    pad2pow=None,
    ymax=None,
    **kwargs,
) -> dict:
    """
    Method used by eeg_power_spectra and intrinsic_freq_autoCorr.
    Signal in must be mean normalized already.

    Parameters
    ----------
    eeg : np.ndarray
        The EEG signal to analyze.
    plot : bool, optional
        Whether to plot the resulting power spectrum (default is True).
    binWidthSecs : float, optional
        The bin width in seconds for the power spectrum.
    maxFreq : float, optional
        The upper limit of the power spectrum frequency range
        (default is 25).
    pad2pow : int, optional
        The power of 2 to pad the signal to (default is None).
    ymax : float, optional
        The maximum y-axis value for the plot (default is None).
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    dict
        A dictionary containing the power spectrum and other
        related metrics.
            "maxFreq", (float) - frequency at which max power in theta band
                                    occurs
            "Power", (np.ndarray) - smoothed power values
            "Freqs", (np.ndarray) - frequencies corresponding to power
                                    values
            "s2n", - signal to noise ratio
            "Power_raw", (np.ndarray) - raw power values
            "k", (np.ndarray) - smoothing kernel
            "kernelLen", (float) - length of smoothing kernel
            "kernelSig", (float) - sigma of smoothing kernel
            "binsPerHz", (float) - bins per Hz in the power spectrum
            "kernelLen", (float) - length of the smoothing kernel


    """
    eeg = params.signal
    # Get raw power spectrum
    nqLim = 1 / params.bin_width_in_secs / 2.0
    origLen = len(eeg)
    if pad2pow is None:
        fftLen = int(np.power(2, nextpow2(origLen)))
    else:
        fftLen = int(np.power(2, pad2pow))
    fftHalfLen = int(fftLen / float(2) + 1)

    fftRes = np.fft.fft(eeg, fftLen)
    # get power density from fft and discard second half of spectrum
    _power = np.power(np.abs(fftRes), 2) / origLen
    power = np.delete(_power, np.s_[fftHalfLen::])
    power[1:-2] = power[1:-2] * 2

    # calculate freqs and crop spectrum to requested range
    freqs = nqLim * np.linspace(0, 1, fftHalfLen)
    freqs = freqs[freqs <= params.max_frequency].T
    power = power[0 : len(freqs)]

    # smooth spectrum using gaussian kernel
    binsPerHz = (fftHalfLen - 1) / nqLim
    kernelLen = int(np.round(params.smoothing_kernel_width * binsPerHz))
    kernelSig = params.smoothing_kernel_sigma * binsPerHz

    k = signal.windows.gaussian(kernelLen, kernelSig) / (kernelLen / 2 / 2)
    power_sm = signal.fftconvolve(power, k[::-1], mode="same")

    # calculate some metrics
    # find max in theta band
    spectrumMaskBand = np.logical_and(
        freqs > params.theta_range[0], freqs < params.theta_range[1]
    )
    bandMaxPower = np.max(power_sm[spectrumMaskBand])
    maxBinInBand = np.argmax(power_sm[spectrumMaskBand])
    bandFreqs = freqs[spectrumMaskBand]
    freqAtBandMaxPower = bandFreqs[maxBinInBand]
    # self.maxBinInBand = maxBinInBand
    # self.freqAtBandMaxPower = freqAtBandMaxPower
    # self.bandMaxPower = bandMaxPower

    # find power in small window around peak and divide by power in rest
    # of spectrum to get snr
    spectrumMaskPeak = np.logical_and(
        freqs > freqAtBandMaxPower - params.signal_to_noise_width / 2,
        freqs < freqAtBandMaxPower + params.signal_to_noise_width / 2,
    )
    s2n = np.nanmean(power_sm[spectrumMaskPeak]) / np.nanmean(
        power_sm[~spectrumMaskPeak]
    )
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if ymax is None:
            ymax = np.min([2 * np.max(power), np.max(power_sm)])
            if ymax == 0:
                ymax = 1
        ax.plot(freqs, power, c=[0.9, 0.9, 0.9])
        # ax.hold(True)
        ax.plot(freqs, power_sm, "k", lw=2)
        ax.axvline(params.theta_range[0], c="b", ls="--")
        ax.axvline(params.theta_range[1], c="b", ls="--")
        _, stemlines, _ = ax.stem([freqAtBandMaxPower], [bandMaxPower], linefmt="r")
        # plt.setp(stemlines, 'linewidth', 2)
        ax.fill_between(
            freqs,
            0,
            power_sm,
            where=spectrumMaskPeak,
            color="r",
            alpha=0.25,
            zorder=25,
        )
        # ax.set_ylim(0, ymax)
        # ax.set_xlim(0, self.xmax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power density (W/Hz)")
    out_dict = {
        "maxFreq": freqAtBandMaxPower,
        "Power": power_sm,
        "Freqs": freqs,
        "s2n": s2n,
        "Power_raw": power,
        "k": k,
        "kernelLen": kernelLen,
        "kernelSig": kernelSig,
        "binsPerHz": binsPerHz,
    }
    return out_dict


def intrinsic_freq_autoCorr(
    spkTimes=None,
    posMask=None,
    pos_sample_rate=50,
    maxFreq=25,
    acBinSize=0.002,
    acWindow=0.5,
    plot=True,
    **kwargs,
):
    """
    Taken and adapted from ephysiopy.common.eegcalcs.EEGCalcs

    Parameters
    ----------
    spkTimes : np.array
        Times in seconds of the cells firing
    posMask : np.array
        Boolean array corresponding to the length of spkTimes where True is stuff to keep
    maxFreq : float
        The maximum frequency to do the power spectrum out to
    acBinSize : float
        The bin size of the autocorrelogram in seconds
    acWindow : float
        The range of the autocorr in seconds
    plot : bool
        Whether to plot the resulting autocorrelogram and power spectrum


    Returns
    -------
    dict
        A dictionary containing the power spectrum and other related metrics

    Notes
    -----
    Make sure all times are in seconds
    """
    acBinsPerPos = 1.0 / pos_sample_rate / acBinSize
    acWindowSizeBins = np.round(acWindow / acBinSize)
    binCentres = np.arange(0.5, len(posMask) * acBinsPerPos) * acBinSize
    spkTrHist, _ = np.histogram(spkTimes, bins=binCentres)

    # split the single histogram into individual chunks
    splitIdx = np.nonzero(np.diff(posMask.astype(int)))[0] + 1
    splitMask = np.split(posMask, splitIdx)
    splitSpkHist = np.split(spkTrHist, (splitIdx * acBinsPerPos).astype(int))
    histChunks = []
    for i in range(len(splitSpkHist)):
        if np.all(splitMask[i]):
            if np.sum(splitSpkHist[i]) > 2:
                if len(splitSpkHist[i]) > int(acWindowSizeBins) * 2:
                    histChunks.append(splitSpkHist[i])
    autoCorrGrid = np.zeros((int(acWindowSizeBins) + 1, len(histChunks)))
    chunkLens = []

    print(f"num chunks = {len(histChunks)}")
    for i in range(len(histChunks)):
        lenThisChunk = len(histChunks[i])
        chunkLens.append(lenThisChunk)
        tmp = np.zeros(lenThisChunk * 2)
        tmp[lenThisChunk // 2 : lenThisChunk // 2 + lenThisChunk] = histChunks[i]
        tmp2 = signal.fftconvolve(
            tmp, histChunks[i][::-1], mode="valid"
        )  # the autocorrelation
        autoCorrGrid[:, i] = (
            tmp2[lenThisChunk // 2 : lenThisChunk // 2 + int(acWindowSizeBins) + 1]
            / acBinsPerPos
        )

    totalLen = np.sum(chunkLens)
    autoCorrSum = np.nansum(autoCorrGrid, 1) / totalLen
    meanNormdAc = autoCorrSum[1::] - np.nanmean(autoCorrSum[1::])
    # return meanNormdAc
    P = PowerSpectrumParams(
        meanNormdAc,
        bin_width_in_secs=acBinSize,
        max_frequency=maxFreq,
    )
    out = power_spectrum(
        P,
        plot=False,
        **kwargs,
    )
    out.update({"meanNormdAc": meanNormdAc})
    if plot:
        fig = plt.gcf()
        ax = fig.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.imshow(
            autoCorrGrid,
            extent=[
                maxFreq * 0.6,
                maxFreq,
                np.max(out["Power"]) * 0.6,
                ax.get_ylim()[1],
            ],
        )
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
    return out


class Rippler(object):
    """
    Does some spectrographic analysis and plots of LFP data
    looking specifically at the ripple band

    NB This is tied pretty specifically to an experiment that
    uses TTL pulses to trigger some 'event' / 'events'...

    Until I modified the Ripple Detector plugin the duration of the TTL
    pulses was variable with a more or less bimodal distribution which
    is why there is a separate treatment of short and long duration TTL pulses below

    """

    n_channels = 64
    # time units are seconds, frequencies and sample rates in Hz
    pre_ttl = 0.05
    post_ttl = 0.2
    min_ttl_duration = 0.01
    # Not all TTL "events" in OE parlance result in a laser pulse as I modified
    # the plugin so that only x percent are sent to the ttl "out" line that goes
    # to the laser. All TTL events *are* recorded however on a separate TTL line
    # that here I am calling ttl_all_line as opposed to ttl_out_line which is the
    # line that goes to the laser - these values are overwritten when the Ripple
    # Detector plugin settings are loaded in __init__
    ttl_all_line = 4
    ttl_out_line = 1
    ttl_percent = (
        100  # percentage of the ripple detections that get propagated to laser
    )
    ttl_duration = 0.05  # minimum duration of TTL pulse in seconds
    low_band = 120  # Hz
    high_band = 250  # Hz
    bit_volts = 0.1949999928474426  # available in the structure.oebin file
    # some parameters for the FFT stuff
    gaussian_window = 12  # in samples
    gaussian_std = 5
    lfp_plotting_scale = (
        500  # this is the scale/range I was looking at the ripple filtered lfp signal
    )
    ripple_std_dev = 2
    ripple_min_duration_ms = 20

    def __init__(self, trial_root: Path, signal: np.ndarray, fs: int):
        """
        Initializes the Rippler class.

        Parameters
        ----------
        trial_root : Path
            Location of the root recording directory, used to load ttls etc.
        signal : np.ndarray
            The LFP signal (usually downsampled to about 500-1000Hz).
        fs : int
            The sampling rate of the signal.

        """

        from ephysiopy.io.recording import OpenEphysBase

        self.pname_for_trial = trial_root
        self.orig_sig = signal
        self.fs = fs
        trial = OpenEphysBase(trial_root)
        trial._get_recording_start_time()
        self.settings = trial.settings
        LFP = EEGCalcsGeneric(signal, fs)
        self.LFP = LFP
        detector_settings = self.settings.get_processor("Ripple")
        ttl_data = detector_settings.load_ttl(
            trial.path2RippleDetector, trial.recording_start_time
        )
        # breakpoint()

        # pname_for_ttl_data = self._find_path_to_ripple_ttl(self.pname_for_trial)
        # sync_file = pname_for_ttl_data.parents[2] / Path("sync_messages.txt")
        # recording_start_time = self._load_start_time(sync_file)
        # ttl_ts = np.load(pname_for_ttl_data / "timestamps.npy") - trial.recording_start_time
        # ttl_states = np.load(pname_for_ttl_data / "states.npy")
        # all_ons = ttl_ts[ttl_states == detector_settings.Ripple_save]
        # laser_ons = ttl_ts[ttl_states == detector_settings.Ripple_Out]
        # laser_offs = ttl_ts[ttl_states == detector_settings.Ripple_Out * -1]
        # no_laser_ons = np.lib.setdiff1d(all_ons, laser_ons)

        # self.all_on_ts = all_ons
        # self.ttl_states = ttl_states
        # self.all_ts = ttl_ts
        self.laser_on_ts = ttl_data["ttl_timestamps"]
        self.laser_off_ts = ttl_data["ttl_timestamps_off"]
        self.no_laser_on_ts = ttl_data["no_laser_ttls"]

        filtered_eeg = LFP.butterFilter(self.low_band, self.high_band)
        filtered_eeg *= self.bit_volts
        self.filtered_eeg = filtered_eeg
        self.eeg_time = np.linspace(
            0,
            LFP.sig.shape[0] / self.fs,
            LFP.sig.shape[0],
        )  # in seconds

    def update_bandpass(self, low=None, high=None):
        """
        Updates the bandpass filter settings.

        Parameters
        ----------
        low : int, optional
            The low frequency for the bandpass filter.
        high : int, optional
            The high frequency for the bandpass filter.
        """
        if low is None:
            low = self.low_band
        self.low_band = low
        if high is None:
            high = self.high_band
        self.high_band = high
        filtered_eeg = self.LFP.butterFilter(low, high)
        filtered_eeg *= self.bit_volts
        self.filtered_eeg = filtered_eeg

    def _load_start_time(self, path_to_sync_message_file: Path):
        """
        Returns the start time contained in a sync file from OE.

        Parameters
        ----------
        path_to_sync_message_file : Path
            Path to the sync message file.

        Returns
        -------
        float
            The start time in seconds.
        """
        recording_start_time = 0
        with open(path_to_sync_message_file, "r") as f:
            sync_strs = f.read()
            sync_lines = sync_strs.split("\n")
            for line in sync_lines:
                if "Start Time" in line:
                    tokens = line.split(":")
                    start_time = int(tokens[-1])
                    sample_rate = int(tokens[0].split("@")[-1].strip().split()[0])
                    recording_start_time = start_time / float(sample_rate)
        return recording_start_time

    def _find_path_to_continuous(self, trial_root: Path, **kwargs) -> Path:
        """
        Iterates through a directory tree and finds the path to the
        Ripple Detector plugin data and returns its location.

        Parameters
        ----------
        trial_root : Path
            The root directory of the trial.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Path
            The path to the continuous data.
        """
        exp_name = kwargs.pop("experiment", "experiment1")
        rec_name = kwargs.pop("recording", "recording1")
        folder_match = (
            trial_root
            / Path("Record Node [0-9][0-9][0-9]")
            / Path(exp_name)
            / Path(rec_name)
            / Path("events")
            / Path("Acquisition_Board-[0-9][0-9][0-9].*")
        )
        for d, c, f in os.walk(trial_root):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if "continuous.dat" in ff:
                        if PurePath(d).match(str(folder_match)):
                            return Path(d)
        return Path()

    def _find_path_to_ripple_ttl(self, trial_root: Path, **kwargs) -> Path:
        """
        Iterates through a directory tree and finds the path to the
        Ripple Detector plugin data and returns its location.

        Parameters
        ----------
        trial_root : Path
            The root directory of the trial.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Path
            The path to the ripple TTL data.

        """
        exp_name = kwargs.pop("experiment", "experiment1")
        rec_name = kwargs.pop("recording", "recording1")
        ripple_match = (
            trial_root
            / Path("Record Node [0-9][0-9][0-9]")
            / Path(exp_name)
            / Path(rec_name)
            / Path("events")
            / Path("Ripple_Detector-[0-9][0-9][0-9].*")
            / Path("TTL")
        )
        for d, c, f in os.walk(trial_root):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if "timestamps.npy" in ff:
                        if PurePath(d).match(str(ripple_match)):
                            return Path(d)
        return Path()

    @saveFigure
    def plot_filtered_lfp_chunk(
        self, start_time: float, end_time: float, **kwargs
    ) -> plt.Axes:
        """
        Plots a chunk of the filtered LFP signal between the specified start and end times.

        Parameters
        ----------
        start_time : float
            The start time of the chunk to plot, in seconds.
        end_time : float
            The end time of the chunk to plot, in seconds.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        plt.Axes
            The matplotlib axes object with the plot.
        """
        idx = np.logical_and(
            self.eeg_time > start_time - self.pre_ttl,
            self.eeg_time < end_time + self.post_ttl,
        )

        eeg_chunk = self.filtered_eeg[idx]

        normed_time = np.linspace(
            -int(self.pre_ttl * 1000), int(self.post_ttl * 1000), len(eeg_chunk)
        )  # in ms
        _, ax1 = plt.subplots(figsize=(6.0, 4.0))  # enlarge plot a bit
        ax1.plot(normed_time, eeg_chunk)

        trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.vlines(
            [0, int(self.post_ttl * 1000)],
            ymin=0,
            ymax=1,
            colors="r",
            linestyles="--",
            transform=trans,
        )
        ax1.set_xlabel("Time to TTL(ms)")
        return ax1

    def plot_rasters(self, laser_on: bool):
        """
        Plots raster plots for the given laser condition.

        Parameters
        ----------
        laser_on : bool
            If True, plots rasters for laser on condition. If False, plots rasters for no laser condition.
        """
        F = FigureMaker()
        self.path2APdata = self._find_path_to_continuous(self.pname_for_trial)
        K = KiloSortSession(self.path2APdata)
        F.ttl_data = {}
        if laser_on:
            F.ttl_data["ttl_timestamps"] = self.laser_on_ts
            ttls = np.array([self.laser_on_ts, self.laser_off_ts]).T
            F.ttl_data["stim_duration"] = (
                np.max(np.diff(ttls)) * 1000
            )  # needs to be in ms
        else:
            F.ttl_data["ttl_timestamps"] = self.no_laser_on_ts
            F.ttl_data["stim_duration"] = self.ttl_duration
        K.load()
        K.removeNoiseClusters()
        K.removeKSNoiseClusters()
        for c in K.good_clusters:
            ts = K.get_cluster_spike_times(c) / 3e4
            F._getRasterPlot(spk_times=ts, cluster=c)
            plt.show()

    @saveFigure
    def _plot_ripple_lfp_with_ttl(self, i_time: float, **kwargs):
        eeg_chunk = self.filtered_eeg[
            np.logical_and(
                self.eeg_time > i_time - self.pre_ttl,
                self.eeg_time < i_time + self.post_ttl,
            )
        ]
        eeg_chunk_time = self.eeg_time[
            np.logical_and(
                self.eeg_time > i_time - self.pre_ttl,
                self.eeg_time < i_time + self.post_ttl,
            )
        ]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        axTrans = transforms.blended_transform_factory(ax.transData, ax.transData)
        ax.plot(eeg_chunk_time, eeg_chunk)
        ax.add_patch(
            Rectangle(
                (i_time, -self.lfp_plotting_scale),
                width=0.1,
                height=1000,
                transform=axTrans,
                color=[0, 0, 1],
                alpha=0.3,
            )
        )

        ax.set_ylim(-self.lfp_plotting_scale, self.lfp_plotting_scale)
        return ax

    def plot_and_save_ripple_band_lfp_with_ttl(self, **kwargs):
        """
        Plots and saves the ripple band LFP signal with TTL events.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.
        """
        for i_time in self.laser_on_ts:
            self._plot_ripple_lfp_with_ttl(i_time, **kwargs)

    @saveFigure
    def plot_mean_spectrograms(self, **kwargs) -> plt.Figure:
        """
        Plots the spectrograms of the LFP signal for both laser on
        and laser off conditions.
        """
        figsize = kwargs.pop("figsize", (12.0, 4.0))
        fig = plt.figure(figsize=figsize)
        ax, ax1 = fig.subplots(1, 2)
        fig, im, spec = self.plot_mean_spectrogram(laser_on=False, ax=ax, **kwargs)
        fig, im1, spec1 = self.plot_mean_spectrogram(laser_on=True, ax=ax1, **kwargs)
        self.laser_off_spectrogram = spec
        self.laser_on_spectrogram = spec1
        spec = np.mean(spec, 0)
        spec1 = np.mean(spec1, 0)
        min_im = np.min([np.min(spec), np.min(spec1)])
        max_im = np.max([np.max(spec), np.max(spec1)])
        im.set_clim((min_im, max_im))
        im1.set_clim((min_im, max_im))
        ax1.set_ylabel("")
        n_no_laser_ttls = len(self.no_laser_on_ts)
        n_laser_ttls = len(self.laser_on_ts)
        ax.set_title(f"Laser off ({n_no_laser_ttls} events)")
        ax1.set_title(f"Laser on ({n_laser_ttls} events)")
        cb_ax = fig.add_axes([0.91, 0.124, 0.01, 0.754])
        fig.colorbar(
            im1,
            label="Power Spectral Density " + r"$20\,\log_{10}|S_x(t, f)|$ in dB",
            cax=cb_ax,
        )
        return fig

    def plot_mean_spectrogram(self, laser_on: bool = False, ax=None, **kwargs):
        """
        Plots the mean spectrograms for both laser on and laser off conditions.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        plt.Figure
            The matplotlib figure object with the plots.

        """
        norm = kwargs.pop("norm", None)
        ttls = np.array([self.laser_on_ts, self.laser_off_ts]).T
        # max_duration used in plotting output below
        ttl_duration = np.mean(np.diff(ttls))

        if not laser_on:
            ttls = np.array(
                [self.no_laser_on_ts, self.no_laser_on_ts + (self.ttl_duration)]
            ).T
        # breakpoint()
        spectrograms = []
        rows = []
        cols = []
        for ttl in ttls:
            (
                SFT,
                N,
                spec,
            ) = self.get_spectrogram(ttl[0], ttl[1])
            r, c = np.shape(spec)
            rows.append(r)
            cols.append(c)
            spectrograms.append(spec)

        # some spectrograms might be slightly different shapes so
        # truncate to the shortest length in each dimension
        min_rows = np.min(rows)
        min_cols = np.min(cols)
        spec_array = np.empty(shape=[len(ttls), min_rows, min_cols])
        for i, s in enumerate(spectrograms):
            spec_array[i, :, :] = s[0:min_rows, 0:min_cols]

        if ax is None:
            fig1, ax1 = plt.subplots(figsize=(6.0, 4.0))  # enlarge plot a bit
        else:
            ax1 = ax
            fig1 = plt.gcf()
        t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
        ax1.set(
            xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, "
            + rf"$\Delta t = {SFT.delta_t:g}\,$s)",
            ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, "
            + rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
            xlim=(t_lo, t_hi),
        )
        trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.vlines(
            [
                self.pre_ttl,
                self.pre_ttl + ttl_duration,
            ],
            ymin=0,
            ymax=1,
            colors="r",
            linestyles="--",
            transform=trans,
        )
        # add an annotation for the ttl duration next in between
        # the vertical red dashed lines
        ttl_duration_ms = ttl_duration  # * 1000
        ax1.annotate(
            f"{ttl_duration_ms:.2f}\n ms",
            xy=(self.pre_ttl + ttl_duration / 2, 0.8),
            xytext=(self.pre_ttl + ttl_duration / 2, 0.8),
            xycoords=trans,
            textcoords=trans,
            ha="center",
            va="bottom",
            color="r",
            fontsize="small",
        )
        # imshow not respecting the image extents so use pcolormesh
        # mean_spec_array = np.mean(spec_array, 0)
        # X, Y = np.meshgrid(
        #     np.linspace(SFT.extent(N)[0], SFT.extent(N)[1], mean_spec_array.shape[1]),
        #     np.linspace(SFT.extent(N)[2], SFT.extent(N)[3], mean_spec_array.shape[0]),
        # )
        # breakpoint()
        # im1 = ax1.pcolormesh(
        #     X, Y, np.mean(spec_array, 0), cmap="magma", norm=norm, edgecolors="face"
        # )
        im1 = ax1.imshow(
            np.mean(spec_array, 0),
            origin="lower",
            aspect="auto",
            extent=SFT.extent(N),
            cmap="magma",
            norm=norm,
        )
        return fig1, im1, spec_array

    def get_spectrogram(self, start_time: float, end_time: float, plot=False) -> tuple:
        """
        Computes the spectrogram of the filtered LFP signal between the specified start and end times.

        Parameters
        ----------
        start_time : float
            The start time of the chunk to analyze, in seconds.
        end_time : float
            The end time of the chunk to analyze, in seconds.
        plot : bool, optional
            Whether to plot the resulting spectrogram (default is False).

        Returns
        -------
        tuple
            A tuple containing the ShortTimeFFT object, the number of samples, and the spectrogram array.
        """
        eeg_chunk = self.filtered_eeg[
            np.logical_and(
                self.eeg_time > start_time - self.pre_ttl,
                self.eeg_time < start_time + self.post_ttl,
            )
        ]
        # breakpoint()

        win = signal.windows.gaussian(
            self.gaussian_window, std=self.gaussian_std, sym=True
        )
        SFT = signal.ShortTimeFFT(win, hop=1, fs=self.fs, mfft=256, scale_to="psd")
        Sx2 = SFT.spectrogram(eeg_chunk)
        N = len(eeg_chunk)

        if plot:
            fig1, ax1 = plt.subplots(figsize=(6.0, 4.0))  # enlarge plot a bit
            t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
            ax1.set_title(
                rf"Spectrogram ({SFT.m_num * SFT.T:g}$\,s$ Gaussian "
                + rf"window, $\sigma_t={self.gaussian_std * SFT.T:g}\,$s)"
            )
            ax1.set(
                xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, "
                + rf"$\Delta t = {SFT.delta_t:g}\,$s)",
                ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, "
                + rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
                xlim=(t_lo, t_hi),
            )
            trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
            ax1.vlines(
                [self.pre_ttl, self.pre_ttl + (start_time)],
                ymin=0,
                ymax=1,
                colors="r",
                linestyles="--",
                transform=trans,
            )

            im1 = ax1.imshow(
                np.abs(Sx2),
                origin="lower",
                aspect="auto",
                extent=SFT.extent(N),
                cmap="magma",
            )
            fig1.colorbar(
                im1,
                label="Power Spectral Density " + r"$20\,\log_{10}|S_x(t, f)|$ in dB",
            )
            plt.show()
        return SFT, N, np.abs(Sx2)

    @saveFigure
    def plot_mean_rippleband_power(self, **kwargs) -> plt.Axes | None:
        """
        Plots the mean power in the ripple band for the laser on and no laser conditions.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        plt.Axes | None
            The matplotlib axes object with the plot, or None if no data is available.

        """
        if np.any(self.laser_on_spectrogram) and np.any(self.laser_off_spectrogram):
            ax = kwargs.pop("ax", None)
            freqs = np.linspace(
                0, int(self.fs / 2), int(self.laser_off_spectrogram.shape[1])
            )
            idx = np.logical_and(freqs >= self.low_band, freqs <= self.high_band)
            mean_power_on = np.mean(self.laser_on_spectrogram[:, idx, :], axis=(0, 1))
            mean_power_no = np.mean(self.laser_off_spectrogram[:, idx, :], axis=(0, 1))
            mean_power_on_time = np.linspace(
                0 - self.pre_ttl, self.post_ttl, len(mean_power_on)
            )
            mean_power_off_time = np.linspace(
                0 - self.pre_ttl, self.post_ttl, len(mean_power_no)
            )
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)

            plt.plot(
                mean_power_on_time,
                mean_power_on,
                "blue",
                label="on",
            )
            plt.plot(
                mean_power_off_time,
                mean_power_no,
                "k",
                label="off",
            )
            ax = plt.gca()
            ax.set_xlabel("Time(s)")
            ax.set_ylabel("Power")
            ax.set_title(f"Mean power between {self.low_band} - {self.high_band}Hz")
            axTrans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.add_patch(
                Rectangle(
                    (0, 0),
                    width=0.1,
                    height=1,
                    transform=axTrans,
                    color=[0, 0, 1],
                    alpha=0.3,
                    label="Laser on",
                )
            )
            plt.legend()
            plt.show()
            return ax

    def _find_high_power_periods(self, n: int = 3, t: int = 10) -> np.ndarray:
        """
        Find periods where the power in the ripple band is above n standard deviations
        for t samples. Meant to recapitulate the algorithm from the Ripple Detector
        plugin.

        Parameters
        ----------
        n : int, optional
            The number of standard deviations above the mean power to consider as high power (default is 3).
        t : int, optional
            The number of samples for which the power must be above the threshold (default is 10).

        Returns
        -------
        np.ndarray
            An array of indices where the power is above the threshold for the specified duration.


        """
        pass
        # get some detection parameters from the Ripple Detector plugin
        # settings = Settings(self.pname_for_trial)
        # proc = settings.get_processor("Ripple")
        # rms_window = float(getattr(proc, "rms_samples"))
        # ripple_detect_channel = int(getattr(proc, "Ripple_Input"))
        # ripple_std = int(getattr(proc, "ripple_std"))
        # time_thresh = int(getattr(proc, "time_thresh"))
        # rms_sig = window_rms(self.filtered_eeg, rms_window)

    def filter_timestamps_for_real_ripples(self):
        """
        Filter out low power and short duration events from the list of timestamps
        """
        laser_on_keep_indices, laser_on_run_lens = (
            self._calc_ripple_chunks_duration_power("laser")
        )
        no_laser_keep_indices, no_laser_run_lens = (
            self._calc_ripple_chunks_duration_power("no_laser")
        )
        self.laser_on_run_lens = laser_on_run_lens
        self.no_laser_run_lens = no_laser_run_lens
        self.laser_on_ts = self.laser_on_ts[laser_on_keep_indices]
        self.laser_off_ts = self.laser_off_ts[laser_on_keep_indices]
        self.no_laser_on_ts = self.no_laser_on_ts[no_laser_keep_indices]

    def _calc_ripple_chunks_duration_power(self, ttl_type="no_laser") -> tuple:
        """
        Find the indices and durations of the events that have sufficient
        duration and power to be considered ripples.

        Parameters
        ----------
        ttl_type : str, default='no_laser'
            which bit of the trial to do the calculation for
            Either 'no_laser' or 'laser'

        Returns
        -------
        tuple
            the run indices to keep and the run durations in ms

        """
        n_samples = int((self.post_ttl + self.pre_ttl) * 1000)
        times = [0]
        if ttl_type == "no_laser":
            times = self.no_laser_on_ts
        elif ttl_type == "laser":
            times = self.laser_on_ts
        else:
            warnings.warn(
                "ttl_type not recognised. Must be one of 'laser' or 'no_laser'"
            )
            return ([],)
        eeg_chunks = np.zeros(shape=[len(times), n_samples])
        rms_signal = window_rms(self.filtered_eeg, 12)

        # Get segments of the root mean squared and smoothed LFP signal
        for i, t in enumerate(times):
            idx = np.logical_and(
                self.eeg_time > t - self.pre_ttl, self.eeg_time < t + self.post_ttl
            )
            eeg_chunks[i, :] = rms_signal[idx]

        # Square the whole filtered LFP signal and calculate the mean power
        mean_power = np.mean(rms_signal)
        std_dev_power = np.std(rms_signal)

        # Find ripples that are ripple_std_dev standard deviations over the
        # mean power to demarcate the start and end of the ripples and longer
        # than ripple_min_duration_ms
        indices_to_keep = []
        run_lens = []
        for idx, chunk in enumerate(eeg_chunks):
            high_power = chunk > mean_power + std_dev_power * self.ripple_std_dev
            run_vals, _, run_lengths = find_runs(high_power)
            if len(run_vals > 1):
                try:
                    if run_vals[0] is True:
                        run_length = run_lengths[0]
                    else:  # second run_val must be True
                        run_length = run_lengths[1]
                    if run_length > self.ripple_min_duration_ms:
                        indices_to_keep.append(idx)
                        run_lens.append(run_length)
                except IndexError:
                    pass
        return indices_to_keep, run_lens
