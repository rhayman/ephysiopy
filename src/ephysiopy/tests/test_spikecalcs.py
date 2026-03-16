import pytest
from ephysiopy.common.spikingcalcs import cluster_quality, SpikeCalcsGeneric
from ephysiopy.common.waveformcalcs import get_param
from ephysiopy.io.recording import AxonaTrial
from ephysiopy.common.utils import BinnedData


def get_spikecalcs_instance(path_to_axona_data) -> SpikeCalcsGeneric:
    T = AxonaTrial(path_to_axona_data)
    T.load_pos_data()
    cut = T.TETRODE[3].cut
    spk_ts = T.TETRODE[3].spike_times[cut == 1]
    S = SpikeCalcsGeneric(spk_ts, 1)
    return S


def test_spikecalcs_init(path_to_axona_data):
    S = get_spikecalcs_instance(path_to_axona_data)
    S.n_spikes
    S.event_window = [-50, 100]
    S._secs_per_bin
    S._secs_per_bin = 0.5
    S.sample_rate
    S.sample_rate = 30000
    S.duration
    with pytest.raises(IndexError):
        S.trial_mean_fr()
    S.duration = 50.0
    fr = S.trial_mean_fr()
    assert isinstance(fr, float)


def test_mean_isi_range(path_to_axona_data):
    S = get_spikecalcs_instance(path_to_axona_data)
    r = S.mean_isi_range(50)
    assert isinstance(r, float)


def test_xcorr(path_to_axona_data):
    S = get_spikecalcs_instance(path_to_axona_data)
    S.acorr()
    ac = S.acorr(Trange=[-100, 100])
    assert isinstance(ac, BinnedData)


def test_cluster_quality(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    cut = T.TETRODE[3].cut
    waves = T.TETRODE[3].waveforms
    L_ratio, isolation_dist = cluster_quality(waves, cut, 1)
    assert isinstance(L_ratio, float)
    assert isinstance(isolation_dist, float)


"""
                        SPIKECALCSAXONA
"""


def test_get_param(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    waveforms = T.TETRODE[3].waveforms
    waveforms = waveforms[T.TETRODE[3].cut == 1, :, :]
    params = ["Amp", "P", "T", "Vt", "tP", "tT", "PCA"]
    for param in params:
        get_param(waveforms, param=param)
