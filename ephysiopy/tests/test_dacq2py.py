# import pytest
# import numpy as np
# import matplotlib.pylab as plt
# from ephysiopy.io.recording import AxonaTrial
# from pathlib import Path


# def get_spike_times(T: AxonaTrial):
#     T.load_pos_data(Path(T.pname))
#     return T.get_spike_times(1, 1)


# def test_load_axona_trial(path_to_axona_data):
#     AxonaTrial(path_to_axona_data)


# @pytest.mark.mpl_image_compare
# def test_plot_axona_path(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     get_spike_times(T)
#     T.makeSpikePathPlot()
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_axona_spikes_on_path(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T.makeSpikePathPlot(ts)
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_axona_ratemap(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T.makeRateMap(ts)
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_axona_HD_map(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T.makeHDPlot(ts)
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_axona_HD_map_with_mrv(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T.makeHDPlot(ts, add_mrv=True, fill=True)
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_axona_SAC(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T.makeSAC(ts)
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_axona_speed_vs_rate(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T.makeSpeedVsRatePlot(ts, maxSpeed=1e6)
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_axona_speed_vs_HD(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T.makeSpeedVsHeadDirectionPlot(ts)
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_axona_EEG_power(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load()
#     T.plotEEGPower('eeg', ylim=25)
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_axona_EGF_power(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load()
#     T.plotEEGPower('egf')
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_axona_xcorr(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load()
#     T.plotXCorr(1, 1)
#     fig = plt.gcf()
#     return fig


# @pytest.mark.mpl_image_compare
# def test_plot_raster(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load()
#     old_events = T.ttl_timestamps.copy()
#     T.ttl_timestamps = None
#     T.plotRaster(1, 1)
#     T.ttl_timestamps = old_events
#     T.plotRaster(1, 1)
#     fig = plt.gcf()
#     ax = plt.gca()
#     T.plotRaster(1, 1, ax=ax, histtype='rate')
#     return fig


# def test_axona_properties(path_to_axona_data):
#     from ephysiopy.dacq2py.axonaIO import EEG
#     T = AxonaTrial(path_to_axona_data)
#     T.load()
#     assert isinstance(T.STM, dict)
#     assert isinstance(T.settings, dict)
#     assert isinstance(T.EEG, EEG)
#     assert isinstance(T.EGF, EEG)
#     T.TETRODE
#     spike_times = T.get_spike_times(1, 1)
#     assert isinstance(spike_times, np.ndarray)
#     with pytest.raises(KeyError):
#         T.TETRODE[16]
#     with pytest.raises(Exception):
#         T.get_spike_times(99, 99)


# def test_plot_summary(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load_pos_data()
#     spike_times = T.get_spike_times(1, 1)
#     T.makeSummaryPlot(spike_times)
