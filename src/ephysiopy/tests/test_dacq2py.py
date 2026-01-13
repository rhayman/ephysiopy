# import pytest
# import numpy as np
# from ephysiopy.io.recording import AxonaTrial
# from pathlib import Path


# def get_spike_times(T: AxonaTrial):
#     T.load_pos_data()
#     return T.get_spike_times(1, 1)


# def test_load_axona_trial(path_to_axona_data):
#     AxonaTrial(path_to_axona_data)


# def test_plot_axona_path(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     get_spike_times(T)
#     T._getSpikePathPlot()


# def test_plot_axona_spikes_on_path(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T._getSpikePathPlot(ts)


# def test_plot_axona_ratemap(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T._getRateMapPlot(ts)


# def test_plot_axona_HD_map(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T._getHDPlot(ts)


# def test_plot_axona_HD_map_with_mrv(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T._getHDPlot(ts, add_mrv=True, fill=True)


# # def test_plot_axona_SAC(path_to_axona_data):
# #     T = AxonaTrial(path_to_axona_data)
# #     ts = get_spike_times(T)
# #     T._getSACPlot(ts)


# def test_plot_axona_speed_vs_rate(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T._getSpeedVsRatePlot(ts, maxSpeed=1e6)


# def test_plot_axona_speed_vs_HD(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T._getSpeedVsHeadDirectionPlot(ts)


# def test_plot_axona_EEG_power(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load_lfp(Path(path_to_axona_data))
#     T.EEGCalcs.calcEEGPowerSpectrum()
#     T._getPowerSpectrumPlot(T.EEGCalcs.freqs,
#                         T.EEGCalcs.power,
#                         T.EEGCalcs.sm_power,
#                         T.EEGCalcs.bandmaxpower,
#                         T.EEGCalcs.freqatbandmaxpower)


# def test_plot_axona_EGF_power(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load_lfp(Path(path_to_axona_data), "egf")
#     T.EEGCalcs.calcEEGPowerSpectrum()
#     T._getPowerSpectrumPlot(T.EEGCalcs.freqs,
#                         T.EEGCalcs.power,
#                         T.EEGCalcs.sm_power,
#                         T.EEGCalcs.bandmaxpower,
#                         T.EEGCalcs.freqatbandmaxpower)


# def test_plot_axona_xcorr(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     ts = get_spike_times(T)
#     T._getXCorrPlot(ts)


# # # def test_plot_raster(path_to_axona_data):
# #     T = AxonaTrial(path_to_axona_data)
# #     ts = get_spike_times(T)
# #     ax = plt.gca()
# #     T._getRasterPlot(ts, ax=ax, histtype='rate')
# #     plt.close('all')
# #     T._getRasterPlot(ts, histtype='rate')
# #     fig = plt.gcf()
    
# #     return fig


# def test_axona_properties(path_to_axona_data):
#     from ephysiopy.common.ephys_generic import EEGCalcsGeneric
#     T = AxonaTrial(path_to_axona_data)
#     # assert isinstance(T.STM, dict)
#     # assert isinstance(T.settings, dict)
#     T.load_lfp(path_to_axona_data)
#     assert isinstance(T.EEGCalcs, EEGCalcsGeneric)
#     T.TETRODE
#     spike_times = T.get_spike_times(1, 1)
#     assert isinstance(spike_times, np.ndarray)
#     with pytest.raises(KeyError):
#         T.TETRODE[16]
#     with pytest.raises(Exception):
#         T.get_spike_times(99, 99)
