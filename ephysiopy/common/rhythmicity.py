import matplotlib.pylab as plt
import numpy as np
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.common.ephys_generic import EEGCalcsGeneric
from ephysiopy.common.spikecalcs import SpikeCalcsGeneric
from ephysiopy.openephys2py.KiloSort import KiloSortSession
from scipy import signal


class CosineDirectionalTuning(object):
    """
    Produces output to do with Welday et al (2011) like analysis
    of rhythmic firing a la oscialltory interference model
    """

    def __init__(
        self,
        spike_times: np.array,
        pos_times: np.array,
        spk_clusters: np.array,
        x: np.array,
        y: np.array,
        tracker_params={},
    ):
        """
        Parameters
        ----------
        spike_times - 1d np.array
        pos_times - 1d np.array
        spk_clusters - 1d np.array
        x and y - 1d np.array
        tracker_params - dict - from the PosTracker as created in
            OESettings.Settings.parse

        NB All timestamps should be given in sub-millisecond accurate
             seconds and pos_xy in cms
        """
        self.spike_times = spike_times
        self.pos_times = pos_times
        self.spk_clusters = spk_clusters
        """
        There can be more spikes than pos samples in terms of sampling as the
        open-ephys buffer probably needs to finish writing and the camera has
        already stopped, so cut of any cluster indices and spike times
        that exceed the length of the pos indices
        """
        idx_to_keep = self.spike_times < self.pos_times[-1]
        self.spike_times = self.spike_times[idx_to_keep]
        self.spk_clusters = self.spk_clusters[idx_to_keep]
        self._pos_sample_rate = 30
        self._spk_sample_rate = 3e4
        self._pos_samples_for_spike = None
        self._min_runlength = 0.4  # in seconds
        self.posCalcs = PosCalcsGeneric(
            x, y, 230, cm=True, jumpmax=100, tracker_params=tracker_params
        )
        self.spikeCalcs = SpikeCalcsGeneric(spike_times)
        self.spikeCalcs.spk_clusters = spk_clusters
        self.posCalcs.postprocesspos(tracker_params)
        xy = self.posCalcs.xy
        hdir = self.posCalcs.dir
        self.posCalcs.calcSpeed(xy)
        self._xy = xy
        self._hdir = hdir
        self._speed = self.posCalcs.speed
        # TEMPORARY FOR POWER SPECTRUM STUFF
        self.smthKernelWidth = 2
        self.smthKernelSigma = 0.1875
        self.sn2Width = 2
        self.thetaRange = [7, 11]
        self.xmax = 11

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

    def _rolling_window(self, a: np.array, window: int):
        """
        Totally nabbed from SO:
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
        clust_pos_idx[clust_pos_idx >= len(self.pos_times)] = (
            len(self.pos_times) - 1
        )
        return clust_pos_idx

    def getClusterSpikeTimes(self, cluster: int):
        ts = self.spike_times[self.spk_clusters == cluster]
        if self.pos_samples_for_spike is None:
            self.getPosIndices()
        return ts

    def getDirectionalBinPerPosition(self, binwidth: int):
        """
        Direction is in degrees as that what is created by me in some of the
        other bits of this package.

        Parameters
        ----------
        binwidth : int - binsizethe bin width in degrees

        Outputs
        -------
        A digitization of which directional bin each position sample belongs to
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
        the run was occurred and the directional bin that run belongs to

        Returns
        -------
        np.array - the start and end indices into position samples of the run
                          and the directional bin to which it belongs
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
            [run_start_indices[min_len_runs_mask],
                grouped_runs[min_len_runs_mask, 1]]
        ).T
        # ret contains run length as last column
        ret = np.insert(ret, 1, np.sum(ret, 1), 1)
        ret = np.insert(ret, 2, grouped_runs[min_len_runs_mask, 0], 1)
        return ret[:, 0:3]

    def speedFilterRuns(self, runs: np.array, minspeed=5.0):
        """
        Given the runs identified in getRunsOfMinLength, filter for speed
        and return runs that meet the min speed criteria

        The function goes over the runs with a moving window of length equal
        to self.min_runlength in samples and sees if any of those segments
        meets the speed criteria and splits them out into separate runs if true

        NB For now this means the same spikes might get included in the
        autocorrelation procedure later as the
        moving window will use overlapping periods - can be modified later

        Parameters
        ----------
        runs - 3 x nRuns np.array generated from getRunsOfMinLength
        minspeed - float - min running speed in cm/s for an epoch (minimum
                                        epoch length defined previously
                            in getRunsOfMinLength as minlength, usually 0.4s)

        Returns
        -------
        3 x nRuns np.array - A modified version of the "runs" input variable
        """
        minlength_in_samples = int(self.pos_sample_rate * self.min_runlength)
        run_list = runs.tolist()
        all_speed = np.array(self.speed)
        for start_idx, end_idx, dir_bin in run_list:
            this_runs_speed = all_speed[start_idx:end_idx]
            this_runs_runs = self._rolling_window(
                this_runs_speed, minlength_in_samples)
            run_mask = np.all(this_runs_runs > minspeed, 1)
            if np.any(run_mask):
                print("got one")

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
        This is taken and adapted from ephysiopy.common.eegcalcs.EEGCalcs

        Parameters
        ----------
        spkTimes - np.array of times in seconds of the cells firing
        posMask - boolean array corresponding to the length of spkTimes
                            where True is stuff to keep
        maxFreq - the maximum frequency to do the power spectrum out to
        acBinSize - the bin size of the autocorrelogram in seconds
        acWindow - the range of the autocorr in seconds

        NB Make sure all times are in seconds
        """
        acBinsPerPos = 1.0 / self.pos_sample_rate / acBinSize
        acWindowSizeBins = np.round(acWindow / acBinSize)
        binCentres = np.arange(0.5, len(posMask) * acBinsPerPos) * acBinSize
        spkTrHist, _ = np.histogram(spkTimes, bins=binCentres)

        # split the single histogram into individual chunks
        splitIdx = np.nonzero(np.diff(posMask.astype(int)))[0] + 1
        splitMask = np.split(posMask, splitIdx)
        splitSpkHist = np.split(
            spkTrHist, (splitIdx * acBinsPerPos).astype(int))
        histChunks = []
        for i in range(len(splitSpkHist)):
            if np.all(splitMask[i]):
                if np.sum(splitSpkHist[i]) > 2:
                    if len(splitSpkHist[i]) > int(acWindowSizeBins) * 2:
                        histChunks.append(splitSpkHist[i])
        autoCorrGrid = np.zeros((int(acWindowSizeBins) + 1, len(histChunks)))
        chunkLens = []
        from scipy import signal

        print(f"num chunks = {len(histChunks)}")
        for i in range(len(histChunks)):
            lenThisChunk = len(histChunks[i])
            chunkLens.append(lenThisChunk)
            tmp = np.zeros(lenThisChunk * 2)
            tmp[lenThisChunk // 2: lenThisChunk //
                2 + lenThisChunk] = histChunks[i]
            tmp2 = signal.fftconvolve(
                tmp, histChunks[i][::-1], mode="valid"
            )  # the autocorrelation
            autoCorrGrid[:, i] = (
                tmp2[lenThisChunk // 2: lenThisChunk //
                     2 + int(acWindowSizeBins) + 1]
                / acBinsPerPos
            )

        totalLen = np.sum(chunkLens)
        autoCorrSum = np.nansum(autoCorrGrid, 1) / totalLen
        meanNormdAc = autoCorrSum[1::] - np.nanmean(autoCorrSum[1::])
        # return meanNormdAc
        out = self.power_spectrum(
            eeg=meanNormdAc,
            binWidthSecs=acBinSize,
            maxFreq=maxFreq,
            pad2pow=16,
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

    def power_spectrum(
        self,
        eeg,
        plot=True,
        binWidthSecs=None,
        maxFreq=25,
        pad2pow=None,
        ymax=None,
        **kwargs,
    ):
        """
        Method used by eeg_power_spectra and intrinsic_freq_autoCorr
        Signal in must be mean normalised already
        """

        # Get raw power spectrum
        nqLim = 1 / binWidthSecs / 2.0
        origLen = len(eeg)
        # if pad2pow is None:
        # 	fftLen = int(np.power(2, self._nextpow2(origLen)))
        # else:
        fftLen = int(np.power(2, pad2pow))
        fftHalfLen = int(fftLen / float(2) + 1)

        fftRes = np.fft.fft(eeg, fftLen)
        # get power density from fft and discard second half of spectrum
        _power = np.power(np.abs(fftRes), 2) / origLen
        power = np.delete(_power, np.s_[fftHalfLen::])
        power[1:-2] = power[1:-2] * 2

        # calculate freqs and crop spectrum to requested range
        freqs = nqLim * np.linspace(0, 1, fftHalfLen)
        freqs = freqs[freqs <= maxFreq].T
        power = power[0: len(freqs)]

        # smooth spectrum using gaussian kernel
        binsPerHz = (fftHalfLen - 1) / nqLim
        kernelLen = np.round(self.smthKernelWidth * binsPerHz)
        kernelSig = self.smthKernelSigma * binsPerHz
        from scipy import signal

        k = signal.gaussian(kernelLen, kernelSig) / (kernelLen / 2 / 2)
        power_sm = signal.fftconvolve(power, k[::-1], mode="same")

        # calculate some metrics
        # find max in theta band
        spectrumMaskBand = np.logical_and(
            freqs > self.thetaRange[0], freqs < self.thetaRange[1]
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
            freqs > freqAtBandMaxPower - self.sn2Width / 2,
            freqs < freqAtBandMaxPower + self.sn2Width / 2,
        )
        s2n = np.nanmean(power_sm[spectrumMaskPeak]) / np.nanmean(
            power_sm[~spectrumMaskPeak]
        )
        self.freqs = freqs
        self.power_sm = power_sm
        self.spectrumMaskPeak = spectrumMaskPeak
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
            ax.axvline(self.thetaRange[0], c="b", ls="--")
            ax.axvline(self.thetaRange[1], c="b", ls="--")
            _, stemlines, _ = ax.stem([freqAtBandMaxPower], [
                                      bandMaxPower], linefmt="r")
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
            "kernelLen": kernelLen,
        }
        return out_dict


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

    def getFreqPhase(self, sig, band2filter: list, ford=3):
        """
        Uses the Hilbert transform to calculate the instantaneous phase and
        amplitude of the time series in sig

        Parameters
        ----------
        sig: np.array
            The signal to be analysed
        ford: int
            The order for the Butterworth filter
        band2filter: list
            The two frequencies to be filtered for e.g. [6, 12]
        """
        if sig is None:
            sig = self.sig
        band2filter = np.array(band2filter, dtype=float)

        b, a = signal.butter(ford, band2filter /
                             (self.fs / 2), btype="bandpass")

        filt_sig = signal.filtfilt(b, a, sig, padtype="odd")
        phase = np.angle(signal.hilbert(filt_sig))
        amplitude = np.abs(signal.hilbert(filt_sig))
        amplitude_filtered = signal.filtfilt(b, a, amplitude, padtype="odd")
        return filt_sig, phase, amplitude, amplitude_filtered

    def modulationindex(
        self,
        sig=None,
        nbins=20,
        forder=2,
        thetaband=[4, 8],
        gammaband=[30, 80],
        plot=True,
    ):
        """
        Calculates the modulation index of theta and gamma oscillations
        Specifically this is the circular correlation between the phase of
        theta and the power of theta

        Parameters
        ----------
        sig; np.array
            The LFP signal
        nbins: int
            The number of bins in the circular range 0 to 2*pi
        forder: int
            The order of the butterworth filter
        thetaband: list
            The lower and upper bands of the theta frequency range
        gammaband: list
            The lower and upper bands of the gamma frequency range
        plot: bool
            Show some pics or not

        """
        if sig is None:
            sig = self.sig
        sig = sig - np.ma.mean(sig)
        if np.ma.is_masked(sig):
            sig = np.ma.compressed(sig)
        _, lowphase, _, _ = self.getFreqPhase(sig, thetaband, forder)
        _, _, highamp, _ = self.getFreqPhase(sig, gammaband, forder)
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
        locking (here PAC) and a value of zero indicates no locking (no PAC)

        Parameters
        ----------
        eeg: numpy array
            the eeg data itself. This is a 1-d array which can be masked or not
        forder: int
            the order of the filter(s) applied to the eeg data
        thetaband/ gammaband: list/ array
            the range of values to bandpass filter for for the theta and gamma
            ranges
        plot: bool (default True)
            whether to plot the resulting binned up polar plot which shows the
            amplitude of the gamma oscillation found at different phases of the
            theta oscillation
        Returns
        -------
        plv: float
            the value of the phase-amplitude coupling
        """

        if sig is None:
            sig = self.sig
        sig = sig - np.ma.mean(sig)
        if np.ma.is_masked(sig):
            sig = np.ma.compressed(sig)

        _, lowphase, _, _ = self.getFreqPhase(sig, thetaband, forder)
        _, _, _, highamp_f = self.getFreqPhase(sig, gammaband, forder)

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
        In some of the optogenetic experiments I ran the frequency of laser
        stimulation was at 6.66Hz - this method attempts to filter those
        frequencies out

        NB: This never worked as well as I would have liked as it required
        tailoring for each trial almost. Maybe a better way to do this using
        mean power or something...
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
    

    def spike_phase_plot(self, cluster: int,
                         pos_data: PosCalcsGeneric,
                         cluster_data: KiloSortSession,
                         lfp_data: EEGCalcsGeneric) -> None:
        '''
        Produces a plot of the phase of theta at which each spike was
        emitted. Each spike is plotted according to the x-y location the
        animal was in when it was fired and the colour of the marker 
        corresponds to the phase of theta at which it fired.
        '''
        _, phase, _, _ = self.getFreqPhase(
            lfp_data.sig, [6, 12])
        cluster_times = cluster_data.spk_times[cluster_data.spk_clusters==cluster]
        # cluster_times in samples (@30000Hz)
        # get indices into the phase vector
        phase_idx = np.array(cluster_times/(3e4/self.fs), dtype=int)
        # It's possible that there are indices higher than the length of 
        # the phase vector so lets set them to the last index
        bad_idx = np.nonzero(phase_idx > len(phase))[0]
        phase_idx[bad_idx] = len(phase) - 1
        # get indices into the position data
        pos_idx = np.array(cluster_times/(3e4/pos_data.sample_rate), dtype=int)
        bad_idx = np.nonzero(pos_idx >= len(pos_data.xyTS))[0]
        pos_idx[bad_idx] = len(pos_data.xyTS) - 1
        # add PI to phases to remove negativity
        cluster_phases = phase[phase_idx]