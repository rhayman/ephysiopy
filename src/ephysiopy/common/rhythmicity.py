import os
import warnings
import matplotlib
import matplotlib.pylab as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path, PurePath
from typing import Callable
from ephysiopy.common.ephys_generic import PosCalcsGeneric, EEGCalcsGeneric, nextpow2

from ephysiopy.common.statscalcs import mean_resultant_vector
from ephysiopy.common.utils import window_rms, find_runs
from ephysiopy.openephys2py.KiloSort import KiloSortSession
from ephysiopy.visualise.plotting import FigureMaker, saveFigure
from ephysiopy.io.recording import OpenEphysBase, AxonaTrial

from scipy import signal
from scipy import stats
from scipy.special import i0

from pactools import Comodulogram, REFERENCES
import pywt
from dataclasses import dataclass, field
from typing import List

"""
Dataclass for collecting the results of frequency/ phase
analysis of LFP

See LFPOscillations.getFreqPhase()
"""


@dataclass(frozen=True)
class FreqPhase:
    filt_sig: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray
    amplitude_filtered: np.ndarray
    inst_freq: np.ndarray


@dataclass
class PowerSpectrumParams:
    """
    Dataclass for holding the parameters for calculating a power
    spectrum as this was being used in several classes and needed
    refactoring out into a standalone function
    """

    signal: np.ndarray
    smoothing_kernel_width: float = 2
    smoothing_kernel_sigma: float = 0.1875
    signal_to_noise_width: float = 2
    theta_range: List = field(default_factory=lambda: [6, 12])
    max_frequency: float = 25
    bin_width_in_secs: float = 1 / 250
    pad_to_power: int = lambda: int(nextpow2(len(signal)))


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


# this class needs refactoring to make use of fieldprops for the
# fitering operations like getRunsOfMinLength etc.
# it's also just generally a mess, using old functions and conventions


class CosineDirectionalTuning(object):
    """
    Produces output to do with Welday et al (2011) like analysis
    of rhythmic firing a la oscialltory interference model
    """

    def __init__(
        self,
        trial: AxonaTrial | OpenEphysBase,
        channel: int,
    ):
        """
        Parameters
        ----------
        spike_times : np.ndarray
            Spike times (in seconds)
        pos_times : np.ndarray
            Position times (in seconds)
        spk_clusters : np.ndarray
            Spike clusters
        x, y : np.ndarray
            Position coordinates
        tracker_params : dict
            From the PosTracker as created in OESettings.Settings.parse

        Attributes
        ----------
        spike_times : np.ndarray
            Spike times
        pos_times : np.ndarray
            Position times
        spk_clusters : np.ndarray
            Spike clusters
        pos_sample_rate : int
            Position sample rate
        spk_sample_rate : float
            Spike sample rate
        min_runlength : float
            Minimum run length
        xy : np.ndarray
            Position coordinates
        hdir : np.ndarray
            Head direction
        speed : np.ndarray
            Speed
        pos_samples_for_spike : np.ndarray
            Position samples for spike
        posCalcs : PosCalcsGeneric
            Position calculations (see ephysiopy.common.ephys_generic.PosCalcsGeneric)
        spikeCalcs : SpikeCalcsGeneric
            Spike calculations (see ephysiopy.common.spikecalcs.SpikeCalcsGeneric)
        smthKernelWidth : int
            Smoothing kernel width (for LFP data)
        smthKernelSigma : float
            Smoothing kernel sigma (for LFP data)
        sn2Width : int
            SN2 width (for LFP data)
        thetaRange : list
            Minimum to maximum theta range
        xmax : int
            Maximum x value


        Notes
        -----
        All timestamps should be given in sub-millisecond accurate seconds
        and pos_xy in cms
        """
        if trial.PosCalcs is None:
            trial.load_pos_data()

        self.trial = trial

        self.spike_times = trial.get_spike_times(tetrode=channel)

        self.pos_times = trial.PosCalcs.xyTS
        # self.spk_clusters = spk_clusters
        # Make sure spike times are within the range of the position times
        idx_to_keep = self.spike_times < self.pos_times[-1]
        self.spike_times = self.spike_times[idx_to_keep]
        # self.spk_clusters = self.spk_clusters[idx_to_keep]
        self._pos_sample_rate = self.trial.PosCalcs.sample_rate
        self._spk_sample_rate = 3e4  # ? for OpenEphysBase
        self._pos_samples_for_spike = None
        self._min_runlength = 0.4  # in seconds
        self.posCalcs = trial.PosCalcs
        # self.spikeCalcs.spk_clusters = spk_clusters
        xy = trial.PosCalcs.xy
        hdir = trial.PosCalcs.dir
        self._xy = xy
        self._hdir = hdir
        self._speed = self.posCalcs.speed
        # TEMPORARY FOR POWER SPECTRUM STUFF
        # self.smthKernelWidth = 2
        # self.smthKernelSigma = 0.1875
        # self.sn2Width = 2
        # self.thetaRange = [7, 11]
        # self.xmax = 11

    @property
    def spk_sample_rate(self):
        return self._spk_sample_rate

    @spk_sample_rate.setter
    def spk_sample_rate(self, value):
        self._spk_sample_rate = value

    @property
    def pos_sample_rate(self):
        return self._pos_sample_rate

    @pos_sample_rate.setter
    def pos_sample_rate(self, value):
        self._pos_sample_rate = value

    @property
    def min_runlength(self):
        return self._min_runlength

    @min_runlength.setter
    def min_runlength(self, value):
        self._min_runlength = value

    @property
    def xy(self):
        return self._xy

    @xy.setter
    def xy(self, value):
        self._xy = value

    @property
    def hdir(self):
        return self._hdir

    @hdir.setter
    def hdir(self, value):
        self._hdir = value

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = value

    @property
    def pos_samples_for_spike(self):
        return self._pos_samples_for_spike

    @pos_samples_for_spike.setter
    def pos_samples_for_spike(self, value):
        self._pos_samples_for_spike = value

    def _rolling_window(self, a: np.array, window: int) -> np.ndarray:
        """
        Returns a view of the array a using a window length of window

        Parameters
        ----------
        a : np.array
            The array to be windowed
        window : int
            The window length

        Returns
        -------
        np.array
            The windowed array

        Notes
        -----
        Taken from:
        https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
        """
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def getPosIndices(self):
        self.pos_samples_for_spike = np.floor(
            self.spike_times * self.pos_sample_rate
        ).astype(int)

    def getClusterPosIndices(self, clust: int) -> np.array:
        if self.pos_samples_for_spike is None:
            self.getPosIndices()
        clust_pos_idx = self.pos_samples_for_spike[self.spk_clusters == clust]
        clust_pos_idx[clust_pos_idx >= len(self.pos_times)] = len(self.pos_times) - 1
        return clust_pos_idx

    def getClusterSpikeTimes(self, cluster: int):
        ts = self.spike_times[self.spk_clusters == cluster]
        if self.pos_samples_for_spike is None:
            self.getPosIndices()
        return ts

    def getDirectionalBinPerPosition(self, binwidth: int):
        """
        Digitizes the directional bin each position sample belongs to.

        Direction is in degrees as that what is created by me in some of the
        other bits of this package.

        Parameters
        ----------
        binwidth : int
            The bin width in degrees.

        Returns
        -------
        np.ndarray
            A digitization of which directional bin each position sample belongs to.
        """

        bins = np.arange(0, 360, binwidth)
        return np.digitize(self.hdir, bins)

    def getDirectionalBinForCluster(self, cluster: int):
        b = self.getDirectionalBinPerPosition(45)
        cluster_pos = self.getClusterPosIndices(cluster)
        # idx_to_keep = cluster_pos < len(self.pos_times)
        # cluster_pos = cluster_pos[idx_to_keep]
        return b[cluster_pos]

    def getRunsOfMinLength(self):
        """
        Identifies runs of at least self.min_runlength seconds long,
        which at 30Hz pos sampling rate equals 12 samples, and
        returns the start and end indices at which
        the run was occurred and the directional bin that run belongs to.

        Returns
        -------
        np.array
            The start and end indices into pos samples of the run
            and the directional bin to which it belongs.
        """

        b = self.getDirectionalBinPerPosition(45)
        # nabbed from SO
        from itertools import groupby

        grouped_runs = [(k, sum(1 for i in g)) for k, g in groupby(b)]
        grouped_runs = np.array(grouped_runs)
        run_start_indices = np.cumsum(grouped_runs[:, 1]) - grouped_runs[:, 1]
        min_len_in_samples = int(self.pos_sample_rate * self.min_runlength)
        min_len_runs_mask = grouped_runs[:, 1] >= min_len_in_samples
        ret = np.array(
            [run_start_indices[min_len_runs_mask], grouped_runs[min_len_runs_mask, 1]]
        ).T
        # ret contains run length as last column
        ret = np.insert(ret, 1, np.sum(ret, 1), 1)
        ret = np.insert(ret, 2, grouped_runs[min_len_runs_mask, 0], 1)
        return ret[:, 0:3]

    def speedFilterRuns(self, runs: np.array, minspeed=5.0):
        """
        Given the runs identified in getRunsOfMinLength, filter for speed
        and return runs that meet the min speed criteria.

        The function goes over the runs with a moving window of length equal
        to self.min_runlength in samples and sees if any of those segments
        meet the speed criteria and splits them out into separate runs if true.

        Notes
        -----
        For now this means the same spikes might get included in the
        autocorrelation procedure later as the moving window will use
        overlapping periods - can be modified later.

        Parameters
        ----------
        runs : np.array
            Generated from getRunsOfMinLength, shape (3, nRuns)
        minspeed : float
            Min running speed in cm/s for an epoch (minimum epoch length
            defined previously in getRunsOfMinLength as minlength, usually 0.4s)

        Returns
        -------
        np.array
            A modified version of the "runs" input variable, shape (3, nRuns)
        """
        pass
        # minlength_in_samples = int(self.pos_sample_rate * self.min_runlength)
        # run_list = runs.tolist()
        # all_speed = np.array(self.speed)
        # for start_idx, end_idx, dir_bin in run_list:
        #     this_runs_speed = all_speed[start_idx:end_idx]
        # this_runs_runs = self._rolling_window(this_runs_speed, minlength_in_samples)
        # run_mask = np.all(this_runs_runs > minspeed, 1)
        # if np.any(run_mask):
        #     print("got one")

    """
    def testing(self, cluster: int):
        ts = self.getClusterSpikeTimes(cluster)
        pos_idx = self.getClusterPosIndices(cluster)

        dir_bins = self.getDirectionalBinPerPosition(45)
        cluster_dir_bins = dir_bins[pos_idx.astype(int)]

        from scipy.signal import periodogram, boxcar, filtfilt

        acorrs = []
        max_freqs = []
        max_idx = []
        isis = []

        acorr_range = np.array([-500, 500])
        for i in range(1, 9):
            this_bin_indices = cluster_dir_bins == i
            this_ts = ts[this_bin_indices]  # in seconds still so * 1000 for ms
            y = self.spikeCalcs.xcorr(this_ts*1000, Trange=acorr_range)
            isis.append(y)
            corr, acorr_bins = np.histogram(
                y[y != 0], bins=501, range=acorr_range)
            freqs, power = periodogram(corr, fs=200, return_onesided=True)
            # Smooth the power over +/- 1Hz
            b = boxcar(3)
            h = filtfilt(b, 3, power)
            # Square the amplitude first
            sqd_amp = h ** 2
            # Then find the mean power in the +/-1Hz band either side of that
            theta_band_max_idx = np.nonzero(
                sqd_amp == np.max(
                    sqd_amp[np.logical_and(freqs > 6, freqs < 11)]))[0][0]
            max_freq = freqs[theta_band_max_idx]
            acorrs.append(corr)
            max_freqs.append(max_freq)
            max_idx.append(theta_band_max_idx)
        return isis, acorrs, max_freqs, max_idx, acorr_bins

    def plotXCorrsByDirection(self, cluster: int):
        acorr_range = np.array([-500, 500])
        # plot_range = np.array([-400,400])
        nbins = 501
        isis, acorrs, max_freqs, max_idx, acorr_bins = self.testing(cluster)
        bin_labels = np.arange(0, 360, 45)
        fig, axs = plt.subplots(8)
        pts = []
        for i, a in enumerate(isis):
            axs[i].hist(
                a[a != 0], bins=nbins, range=acorr_range,
                color='k', histtype='stepfilled')
            # find the max of the first positive peak
            corr, _ = np.histogram(a[a != 0], bins=nbins, range=acorr_range)
            axs[i].set_xlim(acorr_range)
            axs[i].set_ylabel(str(bin_labels[i]))
            axs[i].set_yticklabels('')
            if i < 7:
                axs[i].set_xticklabels('')
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['left'].set_visible(False)
        plt.show()
        return pts
    """

    def intrinsic_freq_autoCorr(
        self,
        spkTimes=None,
        posMask=None,
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
        acBinsPerPos = 1.0 / self.pos_sample_rate / acBinSize
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


class LFPOscillations(object):
    """
    Does stuff with the LFP such as looking at nested oscillations
    (theta/ gamma coupling), the modulation index of such phenomena,
    filtering out certain frequencies in the LFP, getting the instantaneous
    phase and amplitude and so on

    """

    def __init__(self, sig, fs, **kwargs):
        self.sig = sig
        self.fs = fs
        # these member variables are used in power_spectrum()
        self.smthKernelWidth = 2
        self.smthKernelSigma = 0.1875
        self.sn2Width = 2
        self.thetaRange = [6, 12]
        self.xmax = 11

    def getFreqPhase(self, sig, band2filter: list, ford=3) -> FreqPhase:
        """
        Uses the Hilbert transform to calculate the instantaneous phase and
        amplitude of the time series in sig.

        Parameters
        ----------
        sig : np.array
            The signal to be analysed.
        band2filter : list
            The two frequencies to be filtered for.
        ford : int, optional
            The order for the Butterworth filter (default is 3).

        Returns
        -------
        tuple
            A tuple containing the filtered signal, phase, amplitude,
            amplitude filtered, and instantaneous frequency.

        """
        if sig is None:
            sig = self.sig
        band2filter = np.array(band2filter, dtype=float)

        b, a = signal.butter(ford, band2filter / (self.fs / 2), btype="bandpass")

        filt_sig = signal.filtfilt(b, a, sig, padtype="odd")
        hilbert_sig = signal.hilbert(filt_sig)
        phase = np.angle(hilbert_sig)
        amplitude = np.abs(hilbert_sig)
        inst_freq = self.fs / (2 * np.pi) * np.diff(np.unwrap(phase))
        inst_freq = np.insert(inst_freq, -1, inst_freq[-1])
        amplitude_filtered = signal.filtfilt(b, a, amplitude, padtype="odd")
        F = FreqPhase(filt_sig, phase, amplitude, amplitude_filtered, inst_freq)
        return F

    def plot_cwt(
        self,
        sig: np.ndarray,
        Pos: PosCalcsGeneric,
        start: float,
        stop: float,
        FREQ_BAND=(20, 90),
        **kwargs,
    ):
        """
        Plots the continuous wavelet transform of the signal

        Parameters
        ----------
        sig : np.ndarray
            The signal to be analysed.
        Pos : PosCalcsGeneric
            The position object containing speed and time information.
        start : float
            The start time for the plot (in seconds).
        stop : float
            The stop time for the plot (in seconds).
        FREQ_BAND : tuple, optional
            The frequency band to be highlighted (default is (20, 90)).
        """
        wavelet = "cmor1.0-1.0"
        scales = np.geomspace(2, 140, num=100)
        _sig = sig[int(start * self.fs) : int(stop * self.fs)]
        cwtmatr, freqs = pywt.cwt(
            _sig, scales, wavelet, sampling_period=1 / self.fs, method="fft"
        )
        power = np.abs(cwtmatr[:-1, :-1]) ** 2
        t = np.linspace(start, stop, len(_sig))
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].pcolormesh(
            t,
            freqs,
            power,
            # norm="log",
            vmax=np.percentile(power, 99),
            vmin=0,
            cmap="jet",
        )
        ax[0].set_yscale("log")
        # fig.colorbar(im, ax=ax[0], label="Power")
        ax[1].plot(t, _sig, "k")
        ax[1].set_ylabel("LFP (a.u.)")
        # plot speed
        s = slice(int(start * Pos.sample_rate), int(stop * Pos.sample_rate))
        speed = Pos.speed[s]
        t = np.linspace(start, stop, len(speed))
        ax[2].plot(t, speed, "k")
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("Speed (cm/s)")
        plt.tight_layout()
        return fig, ax

    def get_comodulogram(self, low_freq_band=[1, 12], **kwargs):
        """
        Computes the comodulogram of phase-amplitude coupling
        between different frequency bands.

        Parameters
        ----------
        low_freq_band : list
            The low frequency band - what the pactools module calls
            the "driver" frequency
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The computed comodulogram.

        Notes
        -----
        This method is a placeholder and needs to be implemented.
        """
        fs = self.fs
        signal = self.sig

        method = kwargs.get("method", "duprelatour")

        low_fq_width = 1.0  # Hz

        low_fq_range = np.linspace(low_freq_band[0], low_freq_band[1], 50)

        estimator = Comodulogram(
            fs=fs,
            low_fq_range=low_fq_range,
            low_fq_width=low_fq_width,
            method=method,
            progress_bar=False,
        )

        estimator.fit(signal)

        if kwargs.get("plot", False):
            if kwargs.get("ax", None):
                ax = kwargs.get("ax")
            else:
                fig, ax = plt.subplots(figsize=(4, 3))

            vmin = kwargs.get("vmin", None)
            vmax = kwargs.get("vmax", None)

            estimator.plot(titles=[REFERENCES[method]], axs=[ax], vmin=vmin, vmax=vmax)
            ax.set_title("Comodulogram")
            return estimator, ax

        return estimator

    def get_oscillatory_epochs(
        self,
        out_window_size: float = 0.4,
        FREQ_BAND=(
            20,
            90,
        ),
        **kwargs,
    ) -> np.ndarray:
        """
        Uses the continuous wavelet transform to find epochs
        of high oscillatory power in the LFP

        Parameters
        ----------
        out_window_size : float, optional
            The size of the output window in seconds (default is 0.4).
        Returns
        -------
        dict
            A dictionary where keys are the center time of the oscillatory
            window and values are the LFP signal in that window.
        Notes
        -----
        Uses a similar method to jun et al., but expands the window
        for candidate oscillatory windows in a better way

        References
        ----------
        Jun et al., 2020, Neuron 107, 1095–1112
        https://doi.org/10.1016/j.neuron.2020.06.023
        """
        wavelet = kwargs.get("wavelet", "cmor1.0-1.0")
        sd_threshold = kwargs.get("sd_threshold", 2)

        scales = np.geomspace(2, 1024, num=100)
        cwtmatr, freqs = pywt.cwt(
            self.sig, scales, wavelet, sampling_period=1 / self.fs, method="fft"
        )
        # cwtmatr is complex - amplitude is the real part,
        # phase the imaginary part
        # so power is the square of the abs
        power = np.abs(cwtmatr[:-1, :-1]) ** 2
        # get the mean power in the frequency band
        mean_band_power = np.mean(
            power[(freqs >= FREQ_BAND[0]) & (freqs <= FREQ_BAND[1]), :], axis=0
        )
        # Jun et al define periods of high oscillatory power as those
        # over 2 SDs of the mean power
        threshold = np.std(mean_band_power) * sd_threshold
        high_power_mask = mean_band_power > np.mean(mean_band_power) + threshold
        # find the runs of True's in the high_power_mask
        # these are epochs with high gamma power
        vals, run_starts, run_lens = find_runs(high_power_mask.astype(int))
        # calculate the maxima of the amplitude for each segment
        # of the frequency band pass version of the LFP
        F = self.getFreqPhase(self.sig, band2filter=list(FREQ_BAND))
        amplitude_filtered = F.amplitude_filtered
        good_runs = np.nonzero(vals == 1)[0]

        # for each of these epochs get a window of the raw LFP signal
        # centred on the maximum band power
        t = int(out_window_size / (1 / self.fs))
        oscillatory_windows = {}

        for i, run_idx in enumerate(good_runs):
            run_start = run_starts[run_idx]
            s = slice(run_start, run_start + run_lens[run_idx])
            run_max_idx = np.argmax(amplitude_filtered[s], 0) + run_start
            sig_slice = slice(run_max_idx - int(t / 2), run_max_idx + int(t / 2))
            slice_in_seconds = sig_slice.start / self.fs, sig_slice.stop / self.fs
            oscillatory_windows[slice_in_seconds] = self.sig[sig_slice]

        return oscillatory_windows

    def modulationindex(
        self,
        sig=None,
        nbins=20,
        forder=2,
        thetaband=[6, 12],
        gammaband=[20, 90],
        plot=False,
    ) -> float:
        """
        Calculates the modulation index of theta and gamma oscillations.
        Specifically, this is the circular correlation between the phase of
        theta and the power of gamma.

        Parameters
        ----------
        sig : np.array, optional
            The LFP signal. If None, uses the signal provided during
            initialization.
        nbins : int, optional
            The number of bins in the circular range 0 to 2*pi (default is 20).
        forder : int, optional
            The order of the Butterworth filter (default is 2).
        thetaband : list, optional
            The lower and upper bands of the theta frequency range
            (default is [6, 12]).
        gammaband : list, optional
            The lower and upper bands of the gamma frequency range
            (default is [20, 90]).
        plot : bool, optional
            Whether to plot the results (default is True).

        Returns
        -------
        float
            The modulation index.

        Notes
        -----
        The modulation index is a measure of the strength of phase-amplitude
        coupling between theta and gamma oscillations.

        """
        if sig is None:
            sig = self.sig
        sig = sig - np.ma.mean(sig)
        if np.ma.is_masked(sig):
            sig = np.ma.compressed(sig)
        F = self.getFreqPhase(sig, thetaband, forder)
        lowphase = F.phase
        F = self.getFreqPhase(sig, gammaband, forder)
        highamp = F.amplitude
        inc = 2 * np.pi / nbins
        a = np.arange(-np.pi + inc / 2, np.pi, inc)
        dt = np.array([-inc / 2, inc / 2])
        pbins = a[:, np.newaxis] + dt[np.newaxis, :]
        amp = np.zeros((nbins))
        phaselen = np.arange(len(lowphase))
        for i in range(nbins):
            pts = np.nonzero(
                (lowphase >= pbins[i, 0]) * (lowphase < pbins[i, 1]) * phaselen
            )
            amp[i] = np.mean(highamp[pts])
        amp = amp / np.sum(amp)
        from ephysiopy.common.statscalcs import circ_r

        mi = circ_r(pbins[:, 1], amp)
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)
            w = np.pi / (nbins / 2)
            ax.bar(pbins[:, 1], amp, width=w)
            ax.set_title("Modulation index={0:.5f}".format(mi))
        return mi

    def power_spectrum(
        self,
        eeg=None,
        plot=True,
        binWidthSecs=(1 / 250),
        maxFreq=25,
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
        if eeg is None:
            eeg = self.sig

        P = PowerSpectrumParams(eeg)
        return power_spectrum(P, plot, pad2pow, ymax, **kwargs)

    def plv(
        self,
        sig=None,
        forder=2,
        thetaband=[4, 8],
        gammaband=[30, 80],
        plot=True,
        **kwargs,
    ):
        """
        Computes the phase-amplitude coupling (PAC) of nested oscillations.
        More specifically this is the phase-locking value (PLV) between two
        nested oscillations in EEG data, in this case theta (default 4-8Hz)
        and gamma (defaults to 30-80Hz). A PLV of unity indicates perfect phase
        locking (here PAC) and a value of zero indicates no locking (no PAC).

        Parameters
        ----------
        sig : np.array, optional
            The LFP signal. If None, uses the signal provided during initialization.
        forder : int, optional
            The order of the Butterworth filter (default is 2).
        thetaband : list, optional
            The lower and upper bands of the theta frequency range (default is [4, 8]).
        gammaband : list, optional
            The lower and upper bands of the gamma frequency range (default is [30, 80]).
        plot : bool, optional
            Whether to plot the resulting binned up polar plot which shows the amplitude
            of the gamma oscillation found at different phases of the theta oscillation
            (default is True).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        float
            The value of the phase-amplitude coupling (PLV).

        """

        if sig is None:
            sig = self.sig
        sig = sig - np.ma.mean(sig)
        if np.ma.is_masked(sig):
            sig = np.ma.compressed(sig)

        F = self.getFreqPhase(sig, thetaband, forder)
        lowphase = F.phase
        F = self.getFreqPhase(sig, gammaband, forder)
        highamp_f = F.amplitude_filtered

        highampphase = np.angle(signal.hilbert(highamp_f))
        phasedf = highampphase - lowphase
        phasedf = np.exp(1j * phasedf)
        phasedf = np.angle(phasedf)
        from ephysiopy.common.statscalcs import circ_r

        plv = circ_r(phasedf)
        th = np.linspace(0.0, 2 * np.pi, 20, endpoint=False)
        h, _ = np.histogram(phasedf, bins=20)
        h = h / float(len(phasedf))

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)
            w = np.pi / 10
            ax.bar(th, h, width=w, bottom=0.0)
        return plv, th, h

    def filterForLaser(self, sig=None, width=0.125, dip=15.0, stimFreq=6.66):
        """
        Attempts to filter out frequencies from optogenetic experiments where
        the frequency of laser stimulation was at 6.66Hz.

        Parameters
        ----------
        sig : np.array, optional
            The signal to be filtered. If None, uses the signal provided during initialization.
        width : float, optional
            The width of the filter (default is 0.125).
        dip : float, optional
            The dip of the filter (default is 15.0).
        stimFreq : float, optional
            The frequency of the laser stimulation (default is 6.66Hz).

        Returns
        -------
        np.array
            The filtered signal.
        """
        from scipy.signal import filtfilt, firwin, kaiserord

        nyq = self.fs / 2.0
        width = width / nyq
        dip = dip
        N, beta = kaiserord(dip, width)
        print("N: {0}\nbeta: {1}".format(N, beta))
        upper = np.ceil(nyq / stimFreq)
        c = np.arange(stimFreq, upper * stimFreq, stimFreq)
        dt = np.array([-0.125, 0.125])
        cutoff_hz = dt[:, np.newaxis] + c[np.newaxis, :]
        cutoff_hz = cutoff_hz.ravel()
        cutoff_hz = np.append(cutoff_hz, nyq - 1)
        cutoff_hz.sort()
        cutoff_hz_nyq = cutoff_hz / nyq
        taps = firwin(N, cutoff_hz_nyq, window=("kaiser", beta))
        if sig is None:
            sig = self.sig
        fx = filtfilt(taps, [1.0], sig)
        return fx

    def theta_running(
        self,
        pos_data: PosCalcsGeneric,
        lfp_data: EEGCalcsGeneric,
        plot: bool = True,
        **kwargs,
    ) -> tuple[np.ma.MaskedArray, ...]:
        """
        Returns metrics to do with the theta frequency/power and
        running speed/acceleration.

        Parameters
        ----------
        pos_data : PosCalcsGeneric
            Position data object containing position and speed information.
        lfp_data : EEGCalcsGeneric
            LFP data object containing the LFP signal and sampling rate.
        plot : bool
            Whether to plot the results (default is True).
        **kwargs : dict
            Additional keyword arguments:
                low_theta : float
                    Lower bound of theta frequency (default is 6).
                high_theta : float
                    Upper bound of theta frequency (defaultt is 12).
                low_speed : float
                    Lower bound of running speed (data is masked
                    below this value)
                high_speed : float
                    Upper bound of running speed (data is masked
                    above this value)
                nbins : int
                    Number of bins into which to bin data (Same
                    number for both speed and theta)

        Returns
        -------
        tuple[np.ma.MaskedArray, ...]
            A tuple containing masked arrays for speed and theta frequency.

        Notes
        -----
        The function calculates the instantaneous frequency of the theta band
        and interpolates the running speed to match the LFP data. It then
        creates a 2D histogram of theta frequency vs. running speed and
        overlays the mean points for each speed bin. The function also
        performs a linear regression to find the correlation between
        speed and theta frequency.

        """
        low_theta = kwargs.pop("low_theta", 6)
        high_theta = kwargs.pop("high_theta", 12)
        low_speed = kwargs.pop("low_speed", 2)
        high_speed = kwargs.pop("high_speed", 35)
        nbins = kwargs.pop("nbins", 13)
        F = self.getFreqPhase(lfp_data.sig, band2filter=[low_theta, high_theta])
        inst_freq = F.inst_freq
        # interpolate speed to match the frequency of the LFP data
        eeg_time = np.linspace(
            0, lfp_data.sig.shape[0] / lfp_data.fs, len(lfp_data.sig)
        )
        pos_time = np.linspace(0, pos_data.duration, pos_data.npos)
        interpolated_speed = np.interp(eeg_time, pos_time, pos_data.speed)
        h, e = np.histogramdd(
            [inst_freq, interpolated_speed],
            bins=(
                np.linspace(low_theta, high_theta, nbins),
                np.linspace(low_speed, high_speed, nbins),
            ),
        )
        # overlay the mean points for each speed bin
        spd_bins = np.linspace(low_speed, high_speed, nbins)

        def __freq_calc__(fn: Callable) -> list:
            return [
                fn(
                    inst_freq[
                        np.logical_and(interpolated_speed > s1, interpolated_speed < s2)
                    ]
                )
                for s1, s2 in zip(spd_bins[:-1], spd_bins[1:])
            ]

        mean_freqs = __freq_calc__(np.mean)
        counts = [
            np.count_nonzero(np.logical_and(pos_data.speed >= s1, pos_data.speed < s2))
            for s1, s2 in zip(spd_bins[:-1], spd_bins[1:])
        ]
        std_freqs = __freq_calc__(np.std) / np.sqrt(counts)

        # mask the speed and lfp vectors so we can return these based
        # on the low/high bounds of speed & theta for doing correlations/
        # stats later
        speed_masked = np.ma.masked_outside(interpolated_speed, low_speed, high_speed)
        theta_masked = np.ma.masked_outside(inst_freq, low_theta, high_theta)
        # extract both masks, combine and re-apply
        mask = np.logical_or(speed_masked.mask, theta_masked.mask)
        speed_masked.mask = mask
        theta_masked.mask = mask
        # do the linear regression
        # alternative argument here says we expect the correlation
        # to be positive
        res = stats.linregress(
            speed_masked.compressed(), theta_masked.compressed(), alternative="greater"
        )

        if plot:
            plt.pcolormesh(
                e[1],
                e[0],
                h,
                cmap=matplotlib.colormaps["bone_r"],
                norm=matplotlib.colors.LogNorm(),
            )
            plt.colorbar()
            plt.errorbar(
                x=spd_bins[1:] - low_speed, y=mean_freqs, yerr=std_freqs, fmt="r."
            )
            ax = plt.gca()
            ax.set_ylim((low_theta, high_theta))
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Running speed (cm/s)")
            ax.plot(
                spd_bins[1:] - low_speed,
                res.intercept + res.slope * (spd_bins[1:] - low_speed),
                "r--",
            )
            ax.set_title(
                f"""r = {res.rvalue:.2f}, p = {res.pvalue:.3f}, intercept = {
                    res.intercept:.2f}"""
            )

        return res, speed_masked, theta_masked

    def get_mean_resultant_vector(self, spike_times: np.ndarray, **kws) -> np.ndarray:
        """
        Calculates the mean phase at which the cluster emitted spikes
        and the length of the mean resultant vector.

        Parameters
        ----------
        lfp_data (np.ndarray) - the LFP signal

        fs (float) - the sample rate of the LFP signal

        Returns
        -------
        tuple (float, float) - the mean resultant vector length and mean
                               mean resultant direction
        Notes
        -----
        For similar approach see Boccara et al., 2010.
        doi: 10.1038/nn.2602

        """
        MIN_THETA = kws.get("min_theta", 6)
        MAX_THETA = kws.get("max_theta", 12)

        F = self.getFreqPhase(self.sig, [MIN_THETA, MAX_THETA], 2)
        phase = F.phase
        idx = (spike_times * self.fs).astype(int)

        return mean_resultant_vector(phase[idx])

    def get_theta_phase(self, cluster_times: np.ndarray, **kwargs):
        """
        Calculates the phase of theta at which a cluster emitted spikes
        and returns a fit to a vonmises distribution.

        Parameters
        ----------
        cluster_times : np.ndarray
            The times the cluster emitted spikes in seconds.

        Notes
        -----
        kwargs can include:
            low_theta : int
                Low end for bandpass filter.
            high_theta : int
                High end for bandpass filter.

        Returns
        -------
        tuple
            A tuple containing the phase of theta at which the cluster
            emitted spikes, the x values for the vonmises distribution,
            and the y values for the vonmises distribution.

        """
        low_theta = kwargs.pop("low_theta", 6)
        high_theta = kwargs.pop("high_theta", 12)
        F = self.getFreqPhase(self.sig, [low_theta, high_theta])
        phase = F.phase
        # get indices into the phase vector
        phase_idx = np.array(cluster_times * self.fs, dtype=int)
        # It's possible that there are indices higher than the length of
        # the phase vector so lets set them to the last index
        bad_idx = np.nonzero(phase_idx > len(phase))[0]
        phase_idx[bad_idx] = len(phase) - 1
        # get some stats for fitting to a vonmises
        kappa, loc, _ = stats.vonmises.fit(phase[phase_idx])
        x = np.linspace(-np.pi, np.pi, num=501)
        y = np.exp(kappa * np.cos(x - loc)) / (2 * np.pi * i0(kappa))
        return phase[phase_idx], x, y

    def spike_xy_phase_plot(
        self,
        cluster: int,
        pos_data: PosCalcsGeneric,
        lfp_data: EEGCalcsGeneric,
        cluster_times: np.ndarray,
    ) -> plt.Axes:
        """
        Produces a plot of the phase of theta at which each spike was
        emitted. Each spike is plotted according to the x-y location the
        animal was in when it was fired and the colour of the marker
        corresponds to the phase of theta at which it fired.

        Parameters
        ----------
        cluster : int
            The cluster number.
        pos_data : PosCalcsGeneric
            Position data object containing position and speed information.
        phy_data : TemplateModel
            Phy data object containing spike times and clusters.
        lfp_data : EEGCalcsGeneric
            LFP data object containing the LFP signal and sampling rate.

        Returns
        -------
        plt.Axes
            The matplotlib axes object with the plot.
        """
        F = self.getFreqPhase(lfp_data.sig, [6, 12])
        phase = F.phase
        # get indices into the phase vector
        phase_idx = np.array(cluster_times * self.fs, dtype=int)
        # It's possible that there are indices higher than the length of
        # the phase vector so lets set them to the last index
        bad_idx = np.nonzero(phase_idx > len(phase))[0]
        phase_idx[bad_idx] = len(phase) - 1
        # get indices into the position data
        pos_idx = np.array(cluster_times * pos_data.sample_rate, dtype=int)
        bad_idx = np.nonzero(pos_idx >= len(pos_data.xyTS))[0]
        pos_idx[bad_idx] = len(pos_data.xyTS) - 1
        # add PI to phases to remove negativity
        # cluster_phases = phase[phase_idx]
        # TODO: create the colour map for phase and plot
        spike_xy = pos_data.xy[:, pos_idx]
        spike_phase = phase[phase_idx]
        cmap = matplotlib.colormaps["hsv"]
        fig, ax = plt.subplots()
        ax.plot(pos_data.xy[0], pos_data.xy[1], color="lightgrey", zorder=0)
        ax.scatter(spike_xy[0], spike_xy[1], c=spike_phase, cmap=cmap, zorder=1)
        return ax


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
