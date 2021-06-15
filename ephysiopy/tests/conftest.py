import pytest
import numpy as np
from pathlib import Path
import os
from ephysiopy.common.ephys_generic import PosCalcsGeneric


@pytest.fixture
def basic_xy():
    '''
    Returns a random 2D walk as x, y tuple
    '''
    path = Path(__file__).parents[0] / 'data'
    xy_test_data_path = Path(path, "random_walk_xy.npy")
    xy = np.load(xy_test_data_path)
    x = xy[0]
    y = xy[1]
    return x, y


@pytest.fixture
def path_to_axona_data():
    path = Path(__file__).parents[0] / 'data'
    axona_data_path = Path(path, "M845_140919t1rh.set")
    return os.path.join(axona_data_path)


@pytest.fixture
def path_to_OE_settings():
    path = Path(__file__).parents[0] / 'data'
    settings_path = Path(path)
    return os.path.join(settings_path)


@pytest.fixture
def path_to_OE_spikeSorter_settings():
    path = Path(__file__).parents[0] / 'data/spike_sorter_settings_file'
    settings_path = Path(path)
    return os.path.join(settings_path)


@pytest.fixture
def basic_eeg():
    # Lifted this from the scipy.signal.periodogram page
    # Generate a test signal, a 2 Vrms sine wave at 8.5 Hz
    # corrupted by 1.1 V**2/Hz of white noise sampled
    # at 250Hz.
    fs = 250
    N = 1e5
    amp = 2*np.sqrt(2)
    freq = 8.5
    noise_power = 1.1 * fs / 2
    time = np.arange(N) / fs
    x = amp*np.sin(2*np.pi*freq*time)
    x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    return x, fs


@pytest.fixture
def basic_spike_times_and_cluster_ids():
    # Sample rate for open-ephys stuff is 3e4
    # Lets make up 10 seconds of spiking data
    # Make sure we start from the same state
    np.random.seed(21)
    t = np.random.rand(30000*10, 1)
    spike_times = np.nonzero(t > 0.95)[0]  # about 15000 spike times
    # Now we need the cluster ids, same len as spike_times
    cluster_ids = np.zeros_like(spike_times)
    idx = np.nonzero(np.mod(spike_times, 49))[0]
    cluster_ids[idx] = 1
    return spike_times, cluster_ids


@pytest.fixture
def basic_ratemap():
    x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
    r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
    return r


@pytest.fixture
def basic_PosCalcs(basic_xy):
    '''
    Returns a PosCalcsGeneric instance initialised with some random
    walk xy data
    '''
    x = basic_xy[0]
    y = basic_xy[1]
    ppm = 300  # pixels per metre value
    return PosCalcsGeneric(x, y, ppm)
