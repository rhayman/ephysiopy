"""
A comparison of the spiking output from KiloSort and phy and the raw data as recorded
from openephys and the same data high-pass filtered.
"""

from ephysiopy.io.recording import *
from ephysiopy.common.utils import find_runs
from scipy import signal
import matplotlib.pylab as plt

BIT_VOLTS = 0.1949999928474426

data_root = Path("/home/robin/Documents/Science/SST_data_and_paper/SST_data/raw/")
pname = Path(data_root / Path("RHA1-00062_2023-07-19_10-26-59"))

trial = OpenEphysBase(pname, verbose=True)
trial.load_pos_data()
trial.load_neural_data()

raw_data = memmapBinaryFile(trial.path2APdata / Path("continuous.dat"), n_channels=64)
sample_rate = int(3e4)
cluster = 36

# get the KS/ phy waveforms
phy_waveforms = trial.template_model.get_cluster_spike_waveforms(cluster)

# for now only look at really high amplitude events...
min_amplitude = 50  # not sure if this is in sensible units or arbitrary
# ... on the channel with the best signal
best_channel = trial.template_model.get_cluster_channels(cluster)[0]
# get the relevant indices into the various vectors
cluster_idx = trial.template_model.spike_clusters == cluster
spike_times = trial.template_model.spike_times[cluster_idx]
spike_amplitudes = trial.template_model.amplitudes[cluster_idx]

# find runs where there are a bunch of spikes separated by no less than time_df seconds
time_df = 2

high_amplitude_times = np.ma.masked_where(spike_amplitudes < min_amplitude, spike_times)

# get the runs of spikes...
run_values, run_starts, run_lengths = find_runs(high_amplitude_times.mask)

min_run_length = 5
time_at_edges = 0.5  # seconds to plot either side of the current chunk

# run_values inverted here as high_amplitude_times is
# masked where low amplitude values occur
candidate_runs_indices = np.where(
    np.logical_and(run_lengths >= min_run_length, ~run_values)
)[0]

if candidate_runs_indices:
    idx = candidate_runs_indices[0]
    i_slice = slice(run_starts[idx], run_starts[idx] + run_lengths[idx])
    candidate_spike_times = spike_times[i_slice]
    i_spike_amplitudes = spike_amplitudes[i_slice]

    start_in_seconds = candidate_spike_times[0] - time_at_edges
    end_in_seconds = candidate_spike_times[-1] + time_at_edges
    start_time_samples = int(start_in_seconds * sample_rate)
    end_time_samples = int(end_in_seconds * sample_rate)

    raw_chunk = raw_data[best_channel, start_time_samples:end_time_samples]
    time = np.linspace(
        start_in_seconds, end_in_seconds, end_time_samples - start_time_samples
    )
    # plot the raw chunk
    # plt.plot(time, raw_chunk, "black")
    # high pass filter the raw signal and plot
    sos = signal.butter(
        2, np.array([300, 6000]), fs=sample_rate, btype="bandpass", output="sos"
    )
    f_raw = signal.sosfiltfilt(sos, raw_chunk, axis=0)
    plt.plot(time, f_raw, "green")

    # plot the waveforms as extracted from KS/ used in phy
    t_inc = 41 * (1 / sample_rate)
    w = phy_waveforms[i_slice, :, 0]
    i = 0
    for iwave, itime in zip(w, candidate_spike_times):
        tt = np.linspace(itime - t_inc, itime + t_inc, iwave.shape[0])
        plt.plot(tt, iwave * BIT_VOLTS, "orange")
        i += 1

    # plot vertical lines for the spike times
    ax = plt.gca()
    ylim = ax.get_ylim()

    ax.vlines(candidate_spike_times, ylim[0], ylim[1])

else:
    print(
        f"No runs of spikes long enough for cluster {cluster} and a min \
            time difference of {time_df}. Try another time difference and// or cluster"
    )
