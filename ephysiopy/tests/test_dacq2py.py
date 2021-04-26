import pytest
import numpy as np
from ephysiopy.dacq2py.dacq2py_util import AxonaTrial


def test_load_axona_trial(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()


@pytest.mark.mpl_image_compare
def test_plot_axona_path(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotSpikesOnPath(plot=False)
    return ax


@pytest.mark.mpl_image_compare
def test_plot_axona_spikes_on_path(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotSpikesOnPath(1, 1, plot=False)
    return ax


@pytest.mark.mpl_image_compare
def test_plot_axona_ratemap(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotRateMap(1, 1, plot=False)
    return ax


@pytest.mark.mpl_image_compare
def test_plot_axona_HD_map(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotHDMap(1, 1, plot=False)
    return ax


@pytest.mark.mpl_image_compare
def test_plot_axona_HD_map_with_mrv(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotHDMap(1, 1, plot=False, add_mrv=True)
    return ax


@pytest.mark.mpl_image_compare
def test_plot_axona_SAC(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotSAC(1, 1, plot=False)
    return ax


@pytest.mark.mpl_image_compare
def test_plot_axona_speed_vs_rate(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotSpeedVsRate(1, 1, plot=False)
    return ax


@pytest.mark.mpl_image_compare
def test_plot_axona_speed_vs_HD(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotSpeedVsHeadDirection(1, 1, plot=False)
    return ax


@pytest.mark.mpl_image_compare
def test_plot_axona_EEG_power(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotEEGPower('eeg', plot=False)
    return ax


@pytest.mark.mpl_image_compare
def test_plot_axona_EGF_power(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotEEGPower('egf', plot=False)
    return ax


@pytest.mark.mpl_image_compare
def test_plot_axona_xcorr(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    ax = T.plotXCorr(1, 1, plot=False)
    return ax


def test_axona_properties(path_to_axona_data):
    from ephysiopy.dacq2py.axonaIO import EEG
    T = AxonaTrial(path_to_axona_data)
    T.load()
    assert(isinstance(T.STM, dict))
    assert(isinstance(T.settings, dict))
    assert(isinstance(T.EEG, EEG))
    assert(isinstance(T.EGF, EEG))
    T.TETRODE
    spike_times = T.TETRODE.get_spike_ts(1, 1)
    assert(isinstance(spike_times, np.ndarray))
    with pytest.raises(KeyError):
        T.TETRODE[99]
    with pytest.raises(Exception):
        T.TETRODE.get_spike_ts(99, 99)
