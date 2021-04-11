import pytest
import numpy as np
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.common.ephys_generic import EEGCalcsGeneric
from ephysiopy.common.ephys_generic import SpikeCalcsGeneric
from ephysiopy.common.ephys_generic import SpikeCalcsTetrode
from ephysiopy.common.ephys_generic import MapCalcsGeneric
from ephysiopy.common.ephys_generic import FieldCalcs


# Fixtures setting up/ returning instances of the various classes to be tested
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


@pytest.fixture
def basic_MapCalcs(basic_PosCalcs, basic_SpikeCalcs):
    # MapCalcs needs xy, hdir, speed and pos and spike timestamps (in secs)
    P = basic_PosCalcs
    xy, hdir = P.postprocesspos()
    # only have 10 seconds of spiking data so limit the pos stuff to that too
    P.xy = P.xy[:, 0:10*P.sample_rate]
    P.dir = P.dir[0:10*P.sample_rate]
    P.speed = P.speed[0:10*P.sample_rate]
    P.npos = 10*P.sample_rate
    spk_ts = basic_SpikeCalcs.spike_times
    spk_ts = spk_ts / 3e4  # only have 10 secs worth
    pos_ts = np.arange(0, P.npos) / 30.  # have about 167 seconds worth
    M = MapCalcsGeneric(P.xy, P.dir, P.speed, pos_ts, spk_ts, ppm=300)
    M.spk_clusters = basic_SpikeCalcs.spk_clusters
    return M


@pytest.fixture
def basic_EEGCalcs(basic_eeg):
    sig = basic_eeg[0]
    fs = basic_eeg[1]
    return EEGCalcsGeneric(sig, fs)


@pytest.fixture
def basic_SpikeCalcs(basic_spike_times_and_cluster_ids):
    spike_times, cluster_ids = basic_spike_times_and_cluster_ids
    S = SpikeCalcsGeneric(spike_times)
    S._spk_clusters = cluster_ids.astype(int)
    S.sample_rate = 30000
    event_ts = np.arange(20, 380, 10)
    S.event_ts = event_ts / S.sample_rate
    S.duration = 10
    return S


@pytest.fixture
def basic_SpikeCalcsTetrode(basic_spike_times_and_cluster_ids):
    spike_times, cluster_ids = basic_spike_times_and_cluster_ids
    S = SpikeCalcsTetrode(spike_times)
    S._spk_clusters = cluster_ids.astype(int)
    S.sample_rate = 30000
    event_ts = np.arange(20, 380, 10)
    S.event_ts = event_ts / S.sample_rate
    S.duration = 10
    return S


# -----------------------------------------------------------------------
# ------------ PosCalcsGeneric testing ----------------------
# -----------------------------------------------------------------------
def test_speedfilter(basic_PosCalcs, basic_xy):
    xy = np.ma.masked_array([basic_xy[0], basic_xy[1]])
    new_xy = basic_PosCalcs.speedfilter(xy)
    assert(new_xy.ndim == 2)
    assert(xy.shape == new_xy.shape)


def test_interpnans(basic_PosCalcs, basic_xy):
    xy = np.ma.masked_array([basic_xy[0], basic_xy[1]])
    new_xy = basic_PosCalcs.interpnans(xy)
    assert(new_xy.ndim == 2)
    assert(xy.shape == new_xy.shape)


def test_smoothPos(basic_PosCalcs, basic_xy):
    xy = np.ma.masked_array([basic_xy[0], basic_xy[1]])
    new_xy = basic_PosCalcs.smoothPos(xy)
    assert(new_xy.ndim == 2)
    assert(xy.shape == new_xy.shape)


def test_calcSpeed(basic_PosCalcs, basic_xy):
    xy = np.ma.masked_array([basic_xy[0], basic_xy[1]])
    basic_PosCalcs.calcSpeed(xy)
    speed = basic_PosCalcs.speed
    assert(speed.ndim == 1)
    assert(xy.shape[1] == speed.shape[0])


# postprocesspos calls the functions in the above 4 tests
def test_postprocesspos(basic_PosCalcs):
    xy, hdir = basic_PosCalcs.postprocesspos({})
    assert(xy.ndim == 2)
    assert(hdir.ndim == 1)
    assert(xy.shape[1] == hdir.shape[0])


def test_upsamplePos(basic_PosCalcs, basic_xy):
    xy = np.ma.masked_array([basic_xy[0], basic_xy[1]])
    # default upsample rate is 50Hz (for Axona)
    # default sample rate for me is 30Hz
    new_xy = basic_PosCalcs.upsamplePos(xy)
    new_len = np.ceil(xy.shape[1] * (50 / 30.)).astype(int)
    assert(new_xy.ndim == 2)
    assert(new_xy.shape[1] == new_len)


# -----------------------------------------------------------------------
# ------------ EEGCalcsGeneric testing ----------------------
# -----------------------------------------------------------------------
def test_butter_filter(basic_EEGCalcs):
    filt = basic_EEGCalcs.butterFilter(10, 50)
    assert(isinstance(filt, np.ndarray))


def test_calc_power_spectrum(basic_EEGCalcs):
    basic_EEGCalcs.calcEEGPowerSpectrum()


def test_nextpow2(basic_EEGCalcs):
    val = basic_EEGCalcs._nextpow2(10001)
    assert(isinstance(val, float))


@pytest.mark.mpl_image_compare
def test_plot_power_spectrum(basic_EEGCalcs):
    fig = basic_EEGCalcs.plotPowerSpectrum()
    return fig


@pytest.mark.mpl_image_compare
def test_plot_event_eef(basic_EEGCalcs):
    # Need to supply the fcn to be tested with some
    # timestamps in seconds for when 'events' occurred
    # The artificial eeg created and a member of basic_EEGCalcs
    # is sampled at 250Hz and 1e5 samples long (400 seconds)
    event_ts = np.arange(20, 380, 10)
    fig = basic_EEGCalcs.plotEventEEG(event_ts, sample_rate=250)
    return fig


# -----------------------------------------------------------------------
# ------------ SpikeCalcsGeneric testing ----------------------
# -----------------------------------------------------------------------
def test_trial_mean_firing_rate(basic_SpikeCalcs):
    fr = basic_SpikeCalcs.trial_mean_fr(1)
    assert(isinstance(fr, float))


def test_count_spikes(basic_SpikeCalcs):
    n = basic_SpikeCalcs.n_spikes(1)
    assert(isinstance(n, int))


def test_mean_isi_range(basic_SpikeCalcs):
    mn_count = basic_SpikeCalcs.mean_isi_range(1, 500)
    assert(isinstance(mn_count, float))


def test_xcorr(basic_SpikeCalcs):
    c1_times = basic_SpikeCalcs.spike_times[basic_SpikeCalcs.spk_clusters == 1]
    xc = basic_SpikeCalcs.xcorr(c1_times)
    assert(isinstance(xc, np.ndarray))


def test_calculate_psth(basic_SpikeCalcs):
    x, y = basic_SpikeCalcs.calculatePSTH(1)
    assert(isinstance(x, list))
    assert(isinstance(y, list))


@pytest.mark.mpl_image_compare
def test_plot_psth(basic_SpikeCalcs):
    basic_SpikeCalcs.stim_width = 0.01  # * \
    basic_SpikeCalcs.spike_times = basic_SpikeCalcs.spike_times / \
        basic_SpikeCalcs.sample_rate
    fig = basic_SpikeCalcs.plotPSTH(1)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_all_xcorrs(basic_SpikeCalcs):
    fig = basic_SpikeCalcs.plotAllXCorrs(1)
    return fig


def test_theta_mod_idx(basic_SpikeCalcs):
    c1_times = basic_SpikeCalcs.spike_times[basic_SpikeCalcs.spk_clusters == 1]
    tm = basic_SpikeCalcs.thetaModIdx(c1_times)
    assert(isinstance(tm, float))


def test_theta_mod_idx2(basic_SpikeCalcs):
    c1_times = basic_SpikeCalcs.spike_times[basic_SpikeCalcs.spk_clusters == 1]
    tm = basic_SpikeCalcs.thetaModIdxV2(c1_times)
    assert(isinstance(tm, float))


def test_theta_band_max_freq(basic_SpikeCalcs):
    c1_times = basic_SpikeCalcs.spike_times[basic_SpikeCalcs.spk_clusters == 1]
    tm = basic_SpikeCalcs.thetaBandMaxFreq(c1_times)
    assert(isinstance(tm, float))


def test_smooth_spike_pos_count(basic_SpikeCalcs):
    c1_times = basic_SpikeCalcs.spike_times[
        basic_SpikeCalcs.spk_clusters == 1]
    # Assume a 10 second trial sampled at 30Hz so times are congruous
    # with the spiking data
    sm_spks = basic_SpikeCalcs.smoothSpikePosCount(c1_times, npos=3000)
    assert(isinstance(sm_spks, np.ndarray))


# -----------------------------------------------------------------------
# ------------ SpikeCalcsTetrode testing ----------------------
# -----------------------------------------------------------------------
@pytest.mark.mpl_image_compare
def test_plot_ifr_sp_corr(basic_SpikeCalcsTetrode, basic_xy):
    # Assume a 10 second trial sampled at 30Hz so times are congruous
    # with the spiking data
    xy = np.array(basic_xy)
    xy = xy[:, 0:300]  # 300 samples (10 secs)
    speed = np.hypot(xy[0, :], xy[1, :]) / 300  # ppm=300
    speed = speed / (1/30.)  # pos sample rate=30
    # Need to provide the indices of position at which the cell fired
    # so need to do some timebase conversion stuff
    c1_times = basic_SpikeCalcsTetrode.spike_times[
        basic_SpikeCalcsTetrode.spk_clusters == 1]
    c1_pos_idx = np.floor(c1_times / 3e4 * 30).astype(int)
    fig = basic_SpikeCalcsTetrode.ifr_sp_corr(c1_pos_idx, speed, plot=True)
    return fig


# -----------------------------------------------------------------------
# ------------ MapCalcsGeneric testing ----------------------
# -----------------------------------------------------------------------
def test_interp_spike_pos_times(basic_MapCalcs):
    idx = basic_MapCalcs.__interpSpkPosTimes__()
    assert(isinstance(idx, np.ndarray))


@pytest.mark.mpl_image_compare
def test_plot_all(basic_MapCalcs):
    M = basic_MapCalcs
    # this should call all member plotting fcns of MapCalcsGeneric
    M.plot_type = 'all'
    M.good_clusters = [1]
    fig = M.plotAll()
    return fig


def test_get_spatial_stats(basic_MapCalcs):
    val = basic_MapCalcs.getSpatialStats(1)
    import pandas as pd
    assert(isinstance(val, pd.DataFrame))


def test_get_hd_tuning(basic_MapCalcs):
    r, th = basic_MapCalcs.getHDtuning(1)
    assert(isinstance(r, float))
    assert(isinstance(th, float))


def test_get_speed_tuning(basic_MapCalcs):
    speed_corr, speed_mod = basic_MapCalcs.getSpeedTuning(1)
    assert(isinstance(speed_corr, float))
    assert(isinstance(speed_mod, float))


# -----------------------------------------------------------------------
# ------------ FieldCalcs testing --------------------------------
# -----------------------------------------------------------------------
def test_blur_image(basic_ratemap):
    F = FieldCalcs()
    blurred = F._blur_image(basic_ratemap, n=5)
    assert(isinstance(blurred, np.ndarray))


def test_limit_to_one(basic_ratemap):
    F = FieldCalcs()
    rprops, middle_field, middle_field_id = F.limit_to_one(basic_ratemap)
    assert(isinstance(middle_field, np.ndarray))


def test_global_threshold(basic_ratemap):
    F = FieldCalcs()
    F.global_threshold(basic_ratemap)


def test_local_threshold(basic_ratemap):
    F = FieldCalcs()
    A = F.local_threshold(basic_ratemap)
    assert(isinstance(A, np.ndarray))


def test_get_border_score(basic_ratemap):
    F = FieldCalcs()
    bs = F.getBorderScore(basic_ratemap)


def test_get_field_props(basic_ratemap):
    F = FieldCalcs()
    fp = F.get_field_props(basic_ratemap)
    assert(isinstance(fp, dict))


# def test_calc_angs(basic_ratemap):
#     F = FieldCalcs()
#     fp = F.get_field_props(basic_ratemap)
#     peak_idx = fp['Peak_idx']
#     angs = F.calc_angs(peak_idx)
#     assert(isinstance(angs, np.ndarray))
