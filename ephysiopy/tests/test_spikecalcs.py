from ephysiopy.common.spikecalcs import SpikeCalcsGeneric
from ephysiopy.dacq2py.dacq2py_util import AxonaTrial


def test_spikecalcs_init(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spk_ts = T.TETRODE[1].spk_ts
    spk_clusters = T.TETRODE[1].cut
    waveforms = T.TETRODE[1].waveforms
    S = SpikeCalcsGeneric(spk_ts, waveforms=waveforms)
    S.spk_clusters = spk_clusters
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
