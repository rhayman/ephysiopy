import numpy as np
from scipy.signal import butter, filtfilt
from ephysiopy.common.utils import memmapBinaryFile
from ephysiopy.io.recording import TrialInterface as Trial

bit_volts = 0.1949999928474426  # available in the structure.oebin file


def get_raw_cluster_spikes(trial: Trial, cluster: int, **kws) -> np.ndarray:
    """
    Get the raw cluster spikes for a given trial.

    Parameters
    ----------
    trial : Trial
        The trial to get the raw cluster spikes for.
    cluster : int
        The cluster to get the raw cluster spikes for.

    Returns
    -------
    np.ndarray
        The raw cluster spikes for the given trial.
    """
    dt = kws.get("dt", 0.002)  # default to 2 ms window around the spike time
    sample_rate = trial.template_model.sample_rate
    # Get the channels the cluster was active on
    channels = trial.template_model.get_cluster_channels(int(cluster))
    channels = channels[1:5]  # get the top 4 channels

    # get the timestamps of the spikes for the given cluster
    spike_times_secs = trial.get_spike_times(cluster, 0)  # in seconds
    spike_times_samples = (spike_times_secs * sample_rate).astype(int)

    dat_file_path = trial.template_model.dat_path[0]

    # to map correctly we need to know the number of channels in the dat file
    num_channels = trial.template_model.n_channels_dat

    mapped_data = memmapBinaryFile(dat_file_path, n_channels=num_channels)

    # Extract centered slices around each spike time for the specified channels
    raw_cluster_spikes = _extract_centered_slices(
        mapped_data, channels, spike_times_samples, width=int(dt * sample_rate)
    )
    raw_cluster_spikes = (
        raw_cluster_spikes.astype(float) * bit_volts
    )  # convert to microvolts
    filtered_spikes = np.array(
        [
            bandpass_filter(raw_cluster_spikes[i], sample_rate)
            for i in range(raw_cluster_spikes.shape[0])
        ]
    )
    return filtered_spikes


def _extract_centered_slices(memmap_arr, row_indices, col_indices, width):
    """
    Extracts slices of shape (len(row_indices), width, width) from a 2D memmap array,
    centered at each (r, c) pair from row_indices and col_indices.

    Parameters:
        memmap_arr (np.memmap): 2D memory-mapped array of int16.
        row_indices (array-like): Indices into rows (length N).
        col_indices (array-like): Indices into columns (length N).
        width (int): Width of the square slice (must be odd).

    Returns:
        np.ndarray: Array of shape (N, width, width) with extracted slices.
    """
    if width % 2 == 1:
        width += 1  # Ensure width is even for symmetric padding
    half = width // 2
    slices = []
    for r in row_indices:
        row_slices = []
        for c in col_indices:
            c_start = max(c - half, 0)
            c_end = min(c + half + 1, memmap_arr.shape[1])
            slice_ = memmap_arr[r, c_start:c_end]
            pad_left = max(0, half - c)
            pad_right = max(0, (c + half + 1) - memmap_arr.shape[1])
            slice_ = np.pad(slice_, (pad_left, pad_right), mode="constant")
            row_slices.append(slice_)
        slices.append(np.stack(row_slices))
    return np.stack(slices)


def bandpass_filter(data, fs, lowcut=360, highcut=7000, order=3):
    """
    Bandpass filter for 2D array data (N, width) between lowcut and highcut Hz.

    Parameters:
        data (np.ndarray): Input array of shape (N, width).
        fs (float): Sampling frequency in Hz.
        lowcut (float): Low cutoff frequency in Hz.
        highcut (float): High cutoff frequency in Hz.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered data of same shape.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    # Apply along the last axis (width)
    return filtfilt(b, a, data.astype(float), axis=1)
