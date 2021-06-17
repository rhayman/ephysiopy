import pytest
import numpy as np
import matplotlib.pylab as plt
from ephysiopy.dacq2py.dacq2py_util import AxonaTrial
from ephysiopy.common.rhythmicity import CosineDirectionalTuning
from ephysiopy.common.rhythmicity import LFPOscillations


def test_cosine_init(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    C = CosineDirectionalTuning(
        T.TETRODE[1].spk_ts,
        T.xyTS,
        T.TETRODE[1].cut,
        T.xy[0, :],
        T.xy[1, :]
    )
    C.getPosIndices()
    clust_pos_indices = C.getClusterPosIndices(1)
    assert(isinstance(clust_pos_indices, np.ndarray))
    ts = C.getClusterSpikeTimes(1)
    assert(isinstance(ts, np.ndarray))
    dbs = C.getDirectionalBinPerPosition(6)
    assert(isinstance(dbs, np.ndarray))
    dbs = C.getDirectionalBinForCluster(1)
    assert(isinstance(dbs, np.ndarray))
    C.spk_sample_rate
    C.spk_sample_rate = 48000
    C.pos_sample_rate
    C.pos_sample_rate = 50
    C.min_runlength
    C.min_runlength = 5
    C.xy
    C.xy = 1
    C.hdir
    C.hdir = 1
    C.speed
    C.speed = 1
    C.pos_samples_for_spike
    C.pos_samples_for_spike = 1


def test_the_rest_of_CDT(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    C = CosineDirectionalTuning(
        T.TETRODE[1].spk_ts,
        T.xyTS,
        T.TETRODE[1].cut,
        T.xy[0, :],
        T.xy[1, :]
    )
    runs = C.getRunsOfMinLength()
    C.speedFilterRuns(runs)
    spk_ts = T.TETRODE.get_spike_samples(1, 1)
    pos_mask = np.ones_like(spk_ts).astype(bool)
    C.intrinsic_freq_autoCorr(spk_ts, pos_mask)


def test_LFP_oscillations(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    sig = T.EEG.sig
    fs = T.EEG.sample_rate
    LFP_Osc = LFPOscillations(sig, fs)
    LFP_Osc.getFreqPhase(sig, [6, 12])
    LFP_Osc.modulationindex(sig)
    LFP_Osc.plv(sig)
    LFP_Osc.filterForLaser(sig)
