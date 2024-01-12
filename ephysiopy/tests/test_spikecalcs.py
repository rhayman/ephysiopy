import numpy as np
import pytest
from ephysiopy.common.spikecalcs import SpikeCalcsGeneric, get_param
from ephysiopy.common.spikecalcs import SpikeCalcsAxona, cluster_quality
from ephysiopy.io.recording import AxonaTrial


def get_spikecalcs_instance(path_to_axona_data) -> SpikeCalcsGeneric:
    T = AxonaTrial(path_to_axona_data)
    T.load_pos_data()
    cut = T.TETRODE[1].cut
    spk_ts = T.TETRODE[1].spk_ts[cut == 1]
    waves = T.TETRODE[1].waveforms[cut == 1]
    S = SpikeCalcsGeneric(spk_ts, 1, waveforms=waves)
    return S


def test_spikecalcs_init(path_to_axona_data):
    S = get_spikecalcs_instance(path_to_axona_data)
    S.n_spikes
    S.event_window = [-50, 100]
    S.stim_width
    S.stim_width = 10
    S._secs_per_bin
    S._secs_per_bin = 0.5
    S.sample_rate
    S.sample_rate = 30000
    S.pre_spike_samples
    S.pre_spike_samples = 18
    S.post_spike_samples
    S.post_spike_samples = 30
    S.n_spikes
    S.duration
    with pytest.raises(IndexError):
        S.trial_mean_fr()
    S.duration = 50.
    fr = S.trial_mean_fr()
    assert (isinstance(fr, float))
    S.spk_clusters = None


def test_mean_isi_range(path_to_axona_data):
    S = get_spikecalcs_instance(path_to_axona_data)
    r = S.mean_isi_range(50)
    assert (isinstance(r, float))


def test_xcorr(path_to_axona_data):
    S = get_spikecalcs_instance(path_to_axona_data)
    S.acorr()
    y, bins = S.acorr(Trange=[-100, 100])
    assert (isinstance(y, np.ndarray))


def test_mean_waveforms(path_to_axona_data):
    S = get_spikecalcs_instance(path_to_axona_data)
    S.mean_waveform()
    with pytest.raises(IndexError):
        S.mean_waveform(9999)
    S.waveforms(1)
    S._waves = None
    S.mean_waveform(1)


def test_cluster_quality(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    cut = T.TETRODE[1].cut
    waves = T.TETRODE[1].waveforms
    L_ratio, isolation_dist = cluster_quality(waves, cut, 1)
    assert (isinstance(L_ratio, float))
    assert (isinstance(isolation_dist, float))


'''
                        SPIKECALCSAXONA
'''


def test_get_param(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    waveforms = T.TETRODE[1].waveforms
    waveforms = waveforms[T.TETRODE[1].cut == 1, :, :]
    params = ['Amp', 'P', 'T', 'Vt', 'tP', 'tT', 'PCA']
    for param in params:
        get_param(waveforms, param=param)


def test_half_amp_duration(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    waveforms = T.TETRODE[1].waveforms
    waveforms = waveforms[T.TETRODE[1].cut == 1, :, :]
    spk_ts = T.TETRODE.get_spike_samples(1, 1)
    S = SpikeCalcsAxona(spk_ts, 1)
    S.half_amp_dur(waveforms)


def test_p2t_time(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load_pos_data()
    waveforms = T.TETRODE[1].waveforms
    waveforms = waveforms[T.TETRODE[1].cut == 1, :, :]
    spk_ts = T.TETRODE.get_spike_samples(1, 1)
    S = SpikeCalcsAxona(spk_ts, 1)
    S.p2t_time(waveforms)


def test_plot_cluster_space(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load_pos_data()
    waveforms = T.TETRODE[1].waveforms
    waveforms = waveforms[T.TETRODE[1].cut == 1, :, :]
    spk_ts = T.TETRODE.get_spike_samples(1, 1)
    S = SpikeCalcsAxona(spk_ts, 1)
    S.plotClusterSpace(waveforms)
