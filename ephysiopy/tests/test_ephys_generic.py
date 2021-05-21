import pytest
import numpy as np
from ephysiopy.common.ephys_generic import EEGCalcsGeneric
from ephysiopy.common.spikecalcs import SpikeCalcsGeneric
from ephysiopy.common.spikecalcs import SpikeCalcsTetrode
from ephysiopy.common.ephys_generic import MapCalcsGeneric
from ephysiopy.common.ephys_generic import EventsGeneric


# Fixtures setting up/ returning instances of the various classes to be tested
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
    sig, fs = basic_eeg
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
# ------------ EventsGeneric testing ----------------------
# -----------------------------------------------------------------------
def test_events_class():
    EventsGeneric()


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
    all_masked = np.ma.masked_all(np.shape(xy))
    with pytest.raises(ValueError):
        basic_PosCalcs.interpnans(all_masked)


def test_smoothPos(basic_PosCalcs, basic_xy):
    xy = np.ma.masked_array([basic_xy[0], basic_xy[1]])
    new_xy = basic_PosCalcs.smoothPos(xy)
    assert(new_xy.ndim == 2)
    assert(xy.shape == new_xy.shape)


def test_calcSpeed(basic_PosCalcs, basic_xy):
    xy = np.ma.masked_array([basic_xy[0], basic_xy[1]])
    basic_PosCalcs.cm = False
    basic_PosCalcs.calcSpeed(xy)
    basic_PosCalcs.cm = True
    basic_PosCalcs.calcSpeed(xy)
    speed = basic_PosCalcs.speed
    assert(speed.ndim == 1)
    assert(xy.shape[1] == speed.shape[0])


def test_filter_pos(basic_PosCalcs):
    basic_PosCalcs.postprocesspos({})
    x_min = np.min(basic_PosCalcs.xy[0, :])
    y_min = np.min(basic_PosCalcs.xy[1, :])
    x_max = np.max(basic_PosCalcs.xy[0, :])
    y_max = np.max(basic_PosCalcs.xy[1, :])
    pos_filter = {'dir': 'w',
                  'dir1': 'e',
                  'dir2': 'n',
                  'dir3': 's',
                  'speed': [1, 10],
                  'time': [0, 3],
                  'xrange': [x_min+10, x_max-10],
                  'yrange': [y_min+10, y_max-10]}
    this_filt = {}
    for k in pos_filter.keys():
        this_filt[k] = pos_filter[k]
        val = basic_PosCalcs.filterPos(this_filt)
        assert(isinstance(val, np.ndarray))
    basic_PosCalcs.filterPos(None)
    with pytest.raises(ValueError):
        basic_PosCalcs.filterPos({'dir': 'gxg'})
    with pytest.raises(ValueError):
        basic_PosCalcs.filterPos({'speed': [20, 10]})
    with pytest.raises(KeyError):
        basic_PosCalcs.filterPos({'blert': [20, 10]})


# postprocesspos calls the functions in the above 4 tests
def test_postprocesspos(basic_PosCalcs):
    xy, hdir = basic_PosCalcs.postprocesspos({})
    assert(xy.ndim == 2)
    assert(hdir.ndim == 1)
    assert(xy.shape[1] == hdir.shape[0])
    tracker_dict = {
        'LeftBorder': np.min(xy[0, :]),
        'RightBorder': np.max(xy[0, :]),
        'TopBorder': np.min(xy[1, :]),
        'BottomBorder': np.max(xy[1, :]),
        'SampleRate': 30,
    }
    xy, hdir = basic_PosCalcs.postprocesspos(tracker_dict)
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
    basic_EEGCalcs.calcEEGPowerSpectrum(pad2pow=20)


def test_nextpow2(basic_EEGCalcs):
    val = basic_EEGCalcs._nextpow2(10001)
    assert(isinstance(val, float))


def test_ifft_filter(basic_EEGCalcs):
    val = basic_EEGCalcs.ifftFilter(
        basic_EEGCalcs.sig, [50, 60], basic_EEGCalcs.fs)
    assert(isinstance(val, np.ndarray))


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
def test_map_calcs_init(basic_MapCalcs):
    xy = basic_MapCalcs.xy
    hdir = basic_MapCalcs.hdir
    speed = basic_MapCalcs.speed
    pos_ts = basic_MapCalcs.pos_ts
    spk_ts = basic_MapCalcs.spk_ts
    MapCalcsGeneric(
        xy, hdir, speed, pos_ts, spk_ts,
        plot_type="map",
        ppm=300,
        pos_sample_rate=30
        )
    M = MapCalcsGeneric(
        xy, hdir, speed, pos_ts, spk_ts,
        plot_type=["map"],
        pos_sample_rate=30
        )
    M.good_clusters
    M.good_clusters = [1, 2]
    M.spk_clusters
    M.spk_clusters = [1, 2]
    M.ppm
    M.ppm = 300


def test_interp_spike_pos_times(basic_MapCalcs):
    idx = basic_MapCalcs.__interpSpkPosTimes__()
    assert(isinstance(idx, np.ndarray))


def test_get_spatial_stats(basic_MapCalcs):
    val = basic_MapCalcs.getSpatialStats(1)
    assert(isinstance(val, dict))
    val = basic_MapCalcs.getSpatialStats([1])


def test_get_hd_tuning(basic_MapCalcs):
    r, th = basic_MapCalcs.getHDtuning(1)
    assert(isinstance(r, float))
    assert(isinstance(th, float))


def test_get_speed_tuning(basic_MapCalcs):
    speed_corr, speed_mod = basic_MapCalcs.getSpeedTuning(1)
    assert(isinstance(speed_corr, float))
    assert(isinstance(speed_mod, float))
    basic_MapCalcs.getSpeedTuning(1, maxSpeed=2000)


# -----------------------------------------------------------------------
# ------------------------- Misc testing --------------------------------
# -----------------------------------------------------------------------


def test_tint_colours():
    from ephysiopy.dacq2py import tintcolours
    assert(isinstance(tintcolours.colours, list))


def test_about():
    from ephysiopy import __about__
    assert(isinstance(__about__.__author__, str))


def test_init():
    import ephysiopy
    assert(isinstance(ephysiopy.lfp_highcut, int))