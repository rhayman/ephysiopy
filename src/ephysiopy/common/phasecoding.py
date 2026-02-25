import matplotlib
import matplotlib.cm
import matplotlib.pylab as plt
import numpy as np
from typing import Callable
from ephysiopy.common.utils import (
    fixAngle,
    find_runs,
    FreqPhase,
    PowerSpectrumParams,
)
from ephysiopy.common.ephys_generic import PosCalcsGeneric, EEGCalcsGeneric
from ephysiopy.common.rhythmicity import power_spectrum
from scipy import ndimage, signal
import pywt
import pycwt
from pactools import Comodulogram, REFERENCES
from scipy import stats
from scipy.special import i0
from ephysiopy.common.statscalcs import mean_resultant_vector

jet_cmap = matplotlib.colormaps["jet"]
cbar_fontsize = 8
cbar_tick_fontsize = 6


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
        speed: np.ndarray,
        pos_sample_rate: float,
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
        s = slice(int(start * pos_sample_rate), int(stop * pos_sample_rate))
        speed = speed[s]
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


def get_cycle_labels(
    spike_phase: np.ndarray, min_allowed_min_spike_phase: float
) -> tuple[np.ndarray, ...]:
    """
    Get the cycle labels for a given phase array

    Parameters
    ----------
    phase : np.ndarray
        The phases at which the spikes were fired.
    min_allowed_min_spike_phase : float
        The minimum allowed phase for cycles to start.

    Returns
    -------
    np.ndarray
        The cycle labels for the phase array
    """
    # force phase to lie between 0 and 2PI
    minSpikingPhase = get_phase_of_min_spiking(spike_phase)
    phaseAdj = fixAngle(
        spike_phase - minSpikingPhase * (np.pi / 180) + min_allowed_min_spike_phase
    )
    isNegFreq = np.diff(np.unwrap(phaseAdj)) < 0
    isNegFreq = np.append(isNegFreq, isNegFreq[-1])
    # get start of theta cycles as points where diff > pi
    phaseDf = np.diff(phaseAdj)
    cycleStarts = phaseDf[1::] < -np.pi
    cycleStarts = np.append(cycleStarts, True)
    cycleStarts = np.insert(cycleStarts, 0, True)
    cycleStarts[isNegFreq] = False
    cycleLabel = np.cumsum(cycleStarts)

    return cycleLabel, phaseAdj


def get_phase_of_min_spiking(spkPhase: np.ndarray) -> float:
    """
    Returns the phase at which the minimum number of spikes are fired

    Parameters
    ----------
    spkPhase : np.ndarray
        The phase of the spikes

    Returns
    -------
    float
        The phase in degrees at which the minimum number of spikes are fired


    """
    kernelLen = 180
    kernelSig = kernelLen / 4

    regressor = signal.windows.gaussian(kernelLen, kernelSig)
    bins = np.arange(-179.5, 180, 1)
    phaseDist, _ = np.histogram(spkPhase / np.pi * 180, bins=bins)
    phaseDist = ndimage.convolve(phaseDist, regressor)
    phaseMin = bins[
        int(np.ceil(np.nanmean(np.nonzero(phaseDist == np.min(phaseDist))[0])))
    ]
    return phaseMin


def get_bad_cycles(
    filtered_eeg: np.ndarray,
    negative_freqs: np.ndarray,
    cycle_labels: np.array,
    min_power_percent_threshold: float,
    min_theta: float,
    max_theta: float,
    lfp_fs: float,
) -> np.ndarray:
    """
    Get the cycles that are bad based on their length and power

    Parameters
    ----------
    filtered_eeg : np.ndarray
        The filtered EEG signal
    negative_freqs : np.ndarray
        A boolean array indicating negative frequencies
    cycle_labels : np.ndarray
        The cycle labels for the phase array
    min_power_percent_threshold : float
        The minimum power percent threshold for rejecting cycles
    min_theta : float
        The minimum theta frequency
    max_theta : float
        The maximum theta frequency
    lfp_fs : float
        The sampling frequency of the LFP signal

    Returns
    -------
    np.ndarray
        A boolean array indicating bad cycles
    """
    power = np.power(filtered_eeg, 2)
    min_bins = np.max(cycle_labels) + 1
    cycle_valid_power = np.bincount(
        cycle_labels[~negative_freqs],
        weights=power[~negative_freqs],
        minlength=min_bins,
    )
    cycle_valid_bincount = np.bincount(
        cycle_labels[~negative_freqs], minlength=min_bins
    )
    cycle_valid_mn_power = cycle_valid_power / cycle_valid_bincount
    power_rejection_thresh = np.nanpercentile(
        cycle_valid_mn_power, min_power_percent_threshold
    )
    # get the cycles that are below the rejection threshold
    bad_power_cycle = cycle_valid_mn_power < power_rejection_thresh

    # find cycle too long or too short
    allowed_cycle_len = (
        np.floor((1.0 / max_theta) * lfp_fs).astype(int),
        np.ceil((1.0 / min_theta) * lfp_fs).astype(int),
    )
    cycle_bincount_total = np.bincount(cycle_labels, minlength=min_bins)
    bad_len_cycles = np.logical_or(
        cycle_bincount_total < allowed_cycle_len[0],
        cycle_bincount_total > allowed_cycle_len[1],
    )
    bad_cycle = np.logical_or(bad_len_cycles, bad_power_cycle)
    in_bad_cycle = bad_cycle[cycle_labels]
    is_bad = np.logical_or(in_bad_cycle, negative_freqs)

    return is_bad


# ----------- WHOLE LFP ANALYSIS -----------
# these methods look at the whole LFP i.e are
# not limited to looking at spiking and its
# relation to the LFP


def theta_filter_lfp(lfp: np.ndarray, fs: float, **kwargs):
    """
    Processes an LFP signal for theta cycles, filtering
    out bad cycles (low power, too long/ short etc) and
    applying labels to each cycle etc
    """
    # get/ set some args
    min_power_percent_threshold = kwargs.pop("min_power_percent_threshold", 0)
    min_theta = kwargs.pop("min_theta", 6)
    max_theta = kwargs.pop("max_theta", 12)

    L = LFPOscillations(lfp, fs)
    freq_phase = L.getFreqPhase(lfp, (6, 12))
    phase = freq_phase.phase

    is_neg_freq = np.diff(np.unwrap(phase)) < 0
    is_neg_freq = np.append(is_neg_freq, is_neg_freq[-1])
    # get start of theta cycles as points where diff > pi
    phase_df = np.diff(phase)
    cycle_starts = phase_df[1::] < -np.pi
    cycle_starts = np.append(cycle_starts, True)
    cycle_starts = np.insert(cycle_starts, 0, True)
    cycle_starts[is_neg_freq] = False
    cycle_label = np.cumsum(cycle_starts)

    # get the "bad" theta cycles and use this to mask
    # the lfp data
    is_bad = get_bad_cycles(
        freq_phase.filt_sig,
        is_neg_freq,
        cycle_label,
        min_power_percent_threshold,
        min_theta,
        max_theta,
        fs,
    )
    phase = np.ma.MaskedArray(phase, mask=is_bad)
    cycle_label = np.ma.MaskedArray(cycle_label, mask=is_bad)
    lfp = np.ma.MaskedArray(freq_phase.filt_sig, mask=is_bad)
    return phase, cycle_label, lfp


def get_cross_wavelet(
    theta_phase: np.ndarray,
    theta_lfp: np.ndarray,
    gamma_lfp: np.ndarray,
    fs: float,
    **kwargs,
):
    """
    Get the cross wavelet transform between the theta and gamma LFP signals
    """
    # get some args
    min_lfp_chunk_secs = kwargs.pop("min_lfp_chunk_secs", 0.1)
    min_lfp_len = int(min_lfp_chunk_secs / (1 / fs))

    # get the indices of the minimum phases
    min_phase_idx = signal.argrelmin(theta_phase)[0]

    _s = [
        slice(min_phase_idx[i - 1], min_phase_idx[i])
        for i in range(1, len(min_phase_idx))
    ]
    slices = [ss for ss in _s if (ss.stop - ss.start > min_lfp_len)]

    all_spectrograms = np.zeros(shape=(len(slices), 100, 360)) * np.nan

    for i, i_slice in enumerate(slices):
        i_phase = theta_phase[i_slice]
        xwt = pycwt.xwt(
            theta_lfp[i_slice],
            gamma_lfp[i_slice],
            1 / fs,
            dj=1 / 6,
            s0=20 * (1 / fs),
            # J=-1,
        )
        power = np.abs(xwt[0]) ** 2

        # i_phase ranges from -pi to +pi
        # change to degrees and 0 to 360
        i_phase = np.degrees(i_phase)
        i_phase = np.remainder(i_phase + 180, 360).astype(int)
        i_spectrogram = np.zeros(shape=(np.shape, 360)) * np.nan
        i_spectrogram[:, i_phase] = power
        all_spectrograms[i, :, :] = i_spectrogram

    return all_spectrograms, xwt[-2]


def get_theta_cycle_spectogram(
    phase: np.ndarray,
    cycle_label: np.ndarray,
    filt_lfp: np.ndarray,
    lfp: np.ndarray,
    fs: float,
    **kwargs,
):
    """
    Get a spectrogram of the theta cycles in the LFP
    """
    # get some args
    min_lfp_chunk_secs = kwargs.pop("min_lfp_chunk_secs", 0.1)
    min_lfp_len = int(min_lfp_chunk_secs / (1 / fs))

    # get the indices of the minimum phases
    min_phase_idx = signal.argrelmin(phase)[0]

    _s = [
        slice(min_phase_idx[i - 1], min_phase_idx[i])
        for i in range(1, len(min_phase_idx))
    ]
    slices = [ss for ss in _s if (ss.stop - ss.start > min_lfp_len)]

    all_spectrograms = np.zeros(shape=(len(slices), 100, 360)) * np.nan

    wavelet = "cmor1.0-1.0"
    scales = np.geomspace(2, 140, num=100)

    for i, i_slice in enumerate(slices):
        i_phase = phase[i_slice]
        cwtmatr, freqs = pywt.cwt(
            lfp[i_slice], scales, wavelet, sampling_period=1 / fs, method="fft"
        )
        power = np.abs(cwtmatr) ** 2

        # i_phase ranges from -pi to +pi
        # change to degrees and 0 to 360
        i_phase = np.degrees(i_phase)
        i_phase = np.remainder(i_phase + 180, 360).astype(int)
        i_spectrogram = np.zeros(shape=(len(scales), 360)) * np.nan
        i_spectrogram[:, i_phase] = power
        all_spectrograms[i, :, :] = i_spectrogram

    return all_spectrograms, freqs


def detect_oscillation_episodes(lfp: np.ndarray, fs: float):
    scales = np.geomspace(2, 140, num=100)
    cwtmatr, freqs = pywt.cwt(
        lfp, scales, wavelet="cmor1.0-1.0", sampling_period=1 / fs
    )
    power = np.abs(cwtmatr) ** 2

    freq_band = (20, 40)
    freq_idx = np.where(np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1]))[0]
    mean_power = np.mean(power[freq_idx, :], axis=0)
    power_threshold = np.percentile(mean_power, 97.72)

    is_high_power = mean_power >= power_threshold

    values, starts, lengths = find_runs(is_high_power)

    # cut 160ms windows around the centres of the high power episodes
    half_win = int(0.08 * fs)
    episode_slices = []
    for v, s, lens in zip(values, starts, lengths):
        if v:
            c = s + lens // 2
            episode_slices.append(
                slice(max(0, c - half_win), min(len(lfp), c + half_win))
            )

    # make sure each episode is separated by at least 100ms
    min_separation = int(0.1 * fs)
    filtered_slices = []
    if len(episode_slices) > 0:
        filtered_slices.append(episode_slices[0])
        for es in episode_slices[1:]:
            if es.start - filtered_slices[-1].stop >= min_separation:
                filtered_slices.append(es)

    # get the indices of the max amplitude within each episode
    peak_indices = []
    for es in filtered_slices:
        ep = lfp[es]
        peak_idx = np.argmax(np.abs(ep))
        peak_indices.append(es.start + peak_idx)

    # extract 400ms windows around each peak
    half_win = int(0.2 * fs)
    final_slices = []
    for pi in peak_indices:
        final_slices.append(slice(max(0, pi - half_win), min(len(lfp), pi + half_win)))

    return final_slices, freqs
