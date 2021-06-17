import pytest
import numpy as np
import matplotlib.pylab as plt
from ephysiopy.dacq2py.dacq2py_util import AxonaTrial


def test_load_axona_trial(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()


@pytest.mark.mpl_image_compare
def test_plot_axona_path(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotSpikesOnPath(plot=False)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_axona_spikes_on_path(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotSpikesOnPath(1, 1, plot=False)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_axona_ratemap(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotRateMap(1, 1, plot=False)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_axona_HD_map(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotHDMap(1, 1, plot=False)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_axona_HD_map_with_mrv(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotHDMap(1, 1, plot=False, add_mrv=True, fill=True)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_axona_SAC(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotSAC(1, 1, plot=False)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_axona_speed_vs_rate(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotSpeedVsRate(1, 1, plot=False, maxSpeed=1e6)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_axona_speed_vs_HD(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotSpeedVsHeadDirection(1, 1, plot=False)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_axona_EEG_power(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotEEGPower('eeg', ylim=25, plot=False)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_axona_EGF_power(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotEEGPower('egf', plot=False)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_axona_xcorr(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    T.plotXCorr(1, 1, plot=False)
    fig = plt.gcf()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_raster(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    old_events = T.ttl_timestamps.copy()
    T.ttl_timestamps = None
    T.plotRaster(1, 1, plot=False)
    T.ttl_timestamps = old_events
    T.plotRaster(1, 1, plot=False)
    fig = plt.gcf()
    ax = plt.gca()
    T.plotRaster(1, 1, ax=ax, histtype='rate')
    return fig


def test_axona_properties(path_to_axona_data):
    from ephysiopy.dacq2py.axonaIO import EEG
    T = AxonaTrial(path_to_axona_data)
    T.load()
    assert(isinstance(T.STM, dict))
    assert(isinstance(T.settings, dict))
    assert(isinstance(T.EEG, EEG))
    assert(isinstance(T.EGF, EEG))
    T.TETRODE
    spike_times = T.TETRODE.get_spike_samples(1, 1)
    assert(isinstance(spike_times, np.ndarray))
    with pytest.raises(KeyError):
        T.TETRODE[16]
    with pytest.raises(Exception):
        T.TETRODE.get_spike_samples(99, 99)


def test_plot_summary(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spike_times = T.TETRODE.get_spike_samples(1, 1)
    T.makeSummaryPlot(spike_times)
