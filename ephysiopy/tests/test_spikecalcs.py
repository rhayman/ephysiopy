import numpy as np
import pytest
from ephysiopy.common.spikecalcs import SpikeCalcsGeneric
from ephysiopy.common.spikecalcs import SpikeCalcsAxona
from ephysiopy.dacq2py.dacq2py_util import AxonaTrial


def test_spikecalcs_init(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spk_ts = T.TETRODE[1].spk_ts
    spk_clusters = T.TETRODE[1].cut
    waveforms = T.TETRODE[1].waveforms
    S = SpikeCalcsGeneric(spk_ts, waveforms=waveforms)
    S.n_spikes()
    S.spk_clusters = None
    S.n_spikes(1)
    S.spk_clusters = spk_clusters
    S.n_spikes(1)
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
    S.trial_mean_fr(1)
    S.duration = len(T.dir) / 50.
    fr = S.trial_mean_fr(1)
    assert(isinstance(fr, float))


def test_mean_isi_range(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spk_ts = T.TETRODE[1].spk_ts
    spk_clusters = T.TETRODE[1].cut
    waveforms = T.TETRODE[1].waveforms
    S = SpikeCalcsGeneric(spk_ts, waveforms=waveforms)
    S.spk_clusters = spk_clusters
    r = S.mean_isi_range(1, 50)
    assert(isinstance(r, float))
    S.mean_isi_range(999, 50)


def test_xcorr(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spk_ts = T.TETRODE.get_spike_ts(3, 3)
    S = SpikeCalcsGeneric(T.TETRODE[1].spk_ts)
    S.xcorr(spk_ts)
    y = S.xcorr(spk_ts, Trange=[-100, 100])
    assert(isinstance(y, np.ndarray))


def test_mean_waveforms(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spk_ts = T.TETRODE[1].spk_ts
    # spk_clusters = T.TETRODE[1].cut
    # waveforms = T.TETRODE[1].waveforms
    S = SpikeCalcsGeneric(spk_ts)
    S.getClusterWaveforms(1, 1)
    S.waveforms = T.TETRODE[1].waveforms
    S.spk_clusters = T.TETRODE[1].cut
    S.getMeanWaveform(1, 1)
    with pytest.raises(IndexError):
        S.getMeanWaveform(9999, 1)
    S.getClusterWaveforms(1, 1)
    S.waveforms = 1
    with pytest.raises(IndexError):
        S.getMeanWaveform(9999, 1)
    S.waveforms = None
    S.spk_clusters = None
    S.getMeanWaveform(1, 1)


'''
                        SPIKECALCSAXONA
'''


def test_get_param(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    waveforms = T.TETRODE[1].waveforms
    waveforms = waveforms[T.TETRODE[1].cut == 1, :, :]
    spk_ts = T.TETRODE.get_spike_ts(1, 1)
    S = SpikeCalcsAxona(spk_ts)
    params = ['Amp', 'P', 'T', 'Vt', 'tP', 'tT', 'PCA']
    for param in params:
        S.getParam(waveforms, param=param)


def test_half_amp_duration(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    waveforms = T.TETRODE[1].waveforms
    waveforms = waveforms[T.TETRODE[1].cut == 1, :, :]
    spk_ts = T.TETRODE.get_spike_ts(1, 1)
    S = SpikeCalcsAxona(spk_ts)
    S.half_amp_dur(waveforms)


def test_p2t_time(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    waveforms = T.TETRODE[1].waveforms
    waveforms = waveforms[T.TETRODE[1].cut == 1, :, :]
    spk_ts = T.TETRODE.get_spike_ts(1, 1)
    S = SpikeCalcsAxona(spk_ts)
    S.p2t_time(waveforms)


@pytest.mark.mpl_image_compare
def test_plot_cluster_space(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    waveforms = T.TETRODE[1].waveforms
    waveforms = waveforms[T.TETRODE[1].cut == 1, :, :]
    spk_ts = T.TETRODE.get_spike_ts(1, 1)
    S = SpikeCalcsAxona(spk_ts)
    fig = S.plotClusterSpace(waveforms)
    return fig
