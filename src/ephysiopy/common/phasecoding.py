import matplotlib
import matplotlib.cm
import numpy as np
from ephysiopy.common.rhythmicity import LFPOscillations
from ephysiopy.common.utils import (
    fixAngle,
    find_runs,
)
from scipy import ndimage, signal
import pywt
import pycwt

jet_cmap = matplotlib.colormaps["jet"]
cbar_fontsize = 8
cbar_tick_fontsize = 6


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
