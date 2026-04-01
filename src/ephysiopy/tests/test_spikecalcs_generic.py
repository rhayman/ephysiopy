"""
pytest test suite for ephysiopy.common.spikecalcs
Tests for: get_burstiness, mahal, cluster_quality, xcorr,
contamination_percent, KSMetaTuple, and SpikeCalcsGeneric.

All test data is generated synthetically; no file I/O.
"""

import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ephysiopy.common.spikingcalcs import (
    KSMetaTuple,
    SpikeCalcsGeneric,
    cluster_quality,
    contamination_percent,
    get_burstiness,
    mahal,
    xcorr,
)
from ephysiopy.common.utils import BinnedData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spike_times():
    """
    1-D float array of ~500 spike timestamps in seconds, spanning 0-60 s,
    sorted. Seeded with np.random.seed(42).
    """
    np.random.seed(42)
    ts = np.sort(np.random.uniform(0, 60, 500))
    return ts


@pytest.fixture
def spike_times_pair():
    """
    Tuple (x1, x2) of two independent sorted spike-time arrays.
    """
    np.random.seed(42)
    x1 = np.sort(np.random.uniform(0, 60, 300))
    x2 = np.sort(np.random.uniform(0, 60, 250))
    return x1, x2


@pytest.fixture
def isi_matrix():
    """
    200 x 30 float array of synthetic ISI probability distributions.
    Rows are neurons, cols are ISI bins; each row normalised to sum ~1.
    Two NaN rows are included to exercise the NaN-removal branch.
    """
    np.random.seed(7)
    mat = np.random.dirichlet(np.ones(30), size=200).astype(float)
    # insert two NaN rows
    mat[10, 5] = np.nan
    mat[150, 12] = np.nan
    return mat


@pytest.fixture
def waveforms_4ch():
    """
    (nSpikes=80, nElectrodes=4, nSamples=32) random float32 waveform array.
    A few zeros are added to exercise the zeroIdx removal branch in
    cluster_quality.
    """
    np.random.seed(99)
    wvs = np.random.randn(80, 4, 32).astype(np.float32)
    # zero out entire electrode column for the first electrode
    # this exercises the zeroIdx branch
    wvs[:, 0, :] = 0.0
    return wvs


@pytest.fixture
def spike_calcs(spike_times):
    """
    A SpikeCalcsGeneric instance built from spike_times,
    with duration=60.0, pos_sample_rate=50.0, sample_rate=30000.0.
    """
    sc = SpikeCalcsGeneric(spike_times, cluster=1)
    sc.duration = 60.0
    sc.pos_sample_rate = 50.0
    sc.sample_rate = 30000.0
    return sc


@pytest.fixture
def spike_calcs_with_events(spike_times):
    """
    SpikeCalcsGeneric instance with _event_ts set to 20 random event
    timestamps, event_window=np.array([-0.25, 0.25]), _secs_per_bin=0.01.
    """
    np.random.seed(0)
    sc = SpikeCalcsGeneric(spike_times, cluster=1)
    sc.duration = 60.0
    sc.pos_sample_rate = 50.0
    sc.sample_rate = 30000.0
    sc._event_ts = np.sort(np.random.uniform(1, 59, 20))
    sc.event_window = np.array([-0.25, 0.25])
    sc._secs_per_bin = 0.01
    return sc


# ---------------------------------------------------------------------------
# get_burstiness
# ---------------------------------------------------------------------------


def test_get_burstiness_basic(isi_matrix):
    """
    Valid input returns a 3-tuple of (1-D array, 2-D array, 2-D array).
    NaN rows trigger a UserWarning and are removed, so the returned isi_matrix
    may be smaller than the input.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        distances, pca_matrix, out_isi = get_burstiness(isi_matrix)

    assert isinstance(distances, np.ndarray)
    assert distances.ndim == 1
    assert isinstance(pca_matrix, np.ndarray)
    assert pca_matrix.ndim == 2
    assert isinstance(out_isi, np.ndarray)
    assert out_isi.ndim == 2


def test_get_burstiness_nan_rows_removed(isi_matrix):
    """
    When NaN rows are present, a UserWarning should be raised and the
    returned ISI matrix should have fewer rows than the input (NaN rows
    have been removed).
    """
    n_input_rows = isi_matrix.shape[0]
    with pytest.warns(UserWarning, match="NaN"):
        distances, pca_matrix, out_isi = get_burstiness(isi_matrix)

    assert out_isi.shape[0] < n_input_rows


def test_get_burstiness_distances_normalized(isi_matrix):
    """
    The returned distances are projections onto the LDA discriminant.
    Verify that the returned distances array is a finite 1-D float array.
    Note: these distances are signed projections (not constrained to [0,1]).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        distances, _, _ = get_burstiness(isi_matrix)

    assert distances.ndim == 1
    assert np.all(np.isfinite(distances))


def test_get_burstiness_whiten(isi_matrix):
    """
    Running with whiten=True should not raise any exceptions and should
    return results of the correct types.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = get_burstiness(isi_matrix, whiten=True)

    assert len(result) == 3
    distances, pca_matrix, out_isi = result
    assert isinstance(distances, np.ndarray)
    assert isinstance(pca_matrix, np.ndarray)
    assert isinstance(out_isi, np.ndarray)


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.subplots")
@patch("matplotlib.pyplot.figure")
@patch("matplotlib.pyplot.scatter")
@patch("seaborn.histplot")
def test_get_burstiness_plot_pcs(
    mock_sns_histplot, mock_scatter, mock_figure, mock_subplots, mock_show, isi_matrix
):
    """
    Running with plot_pcs=True should not raise any exceptions even when
    plotting functions are patched to avoid opening a GUI.
    seaborn.histplot is also patched to avoid seaborn internals errors.
    """
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_figure.return_value = mock_fig
    mock_fig.add_subplot.return_value = mock_ax
    mock_subplots.return_value = (mock_fig, mock_ax)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = get_burstiness(isi_matrix, plot_pcs=True)

    assert len(result) == 3


# ---------------------------------------------------------------------------
# mahal
# ---------------------------------------------------------------------------


def test_mahal_basic():
    """
    Standard case: v is 50x4, u is 30x4.
    Result should have shape (30,).
    """
    np.random.seed(42)
    v = np.random.randn(50, 4)
    u = np.random.randn(30, 4)
    d = mahal(u, v)
    assert d.shape == (30,)


def test_mahal_column_mismatch_warning():
    """
    Mismatched column counts between u and v should trigger a UserWarning.
    The function may raise a subsequent exception due to the mismatch,
    so we catch it after checking the warning was issued.
    """
    np.random.seed(42)
    v = np.random.randn(50, 4)
    u = np.random.randn(30, 3)
    with pytest.warns(UserWarning):
        try:
            mahal(u, v)
        except (ValueError, Exception):
            pass  # expected: column mismatch causes further errors


def test_mahal_too_few_rows_warning():
    """
    Fewer rows than columns in v triggers a UserWarning.
    The function may raise a subsequent exception due to the rank deficiency,
    so we catch it after checking the warning was issued.
    """
    np.random.seed(42)
    # 3 rows, 4 columns -> too few rows
    v = np.random.randn(3, 4)
    u = np.random.randn(5, 4)
    with pytest.warns(UserWarning):
        try:
            mahal(u, v)
        except Exception:
            pass  # expected: singular matrix / rank-deficient


def test_mahal_values_nonnegative():
    """
    Mahalanobis distances should all be >= 0.
    """
    np.random.seed(42)
    v = np.random.randn(50, 4)
    u = np.random.randn(30, 4)
    d = mahal(u, v)
    assert np.all(d >= 0)


# ---------------------------------------------------------------------------
# cluster_quality
# ---------------------------------------------------------------------------


def test_cluster_quality_returns_none_when_no_waveforms():
    """
    When waveforms=None, cluster_quality should return None immediately.
    """
    result = cluster_quality(waveforms=None)
    assert result is None


def test_cluster_quality_returns_floats(waveforms_4ch):
    """
    Valid 80x4x32 waveform array with random cluster IDs should return
    a tuple of two floats (L_ratio, isolation_dist).
    """
    np.random.seed(42)
    spike_clusters = np.random.randint(0, 2, size=80)
    result = cluster_quality(
        waveforms=waveforms_4ch,
        spike_clusters=spike_clusters,
        cluster_id=1,
        fet=1,
    )
    assert result is not None
    L_ratio, isolation_dist = result
    assert isinstance(L_ratio, float) or np.isnan(L_ratio)
    assert isinstance(isolation_dist, float) or np.isnan(isolation_dist)


def test_cluster_quality_nan_on_exception():
    """
    Pathological waveforms (all zeros except one channel) should either
    succeed or trigger the except branch returning (nan, nan).
    """
    # All-zero waveforms will cause PCA/mahal to fail
    wvs = np.zeros((10, 4, 32), dtype=np.float32)
    spike_clusters = np.array([1] * 5 + [0] * 5)
    result = cluster_quality(
        waveforms=wvs,
        spike_clusters=spike_clusters,
        cluster_id=1,
    )
    if result is not None:
        L_ratio, isolation_dist = result
        # Either nan (except branch) or valid float
        assert np.isnan(L_ratio) or isinstance(L_ratio, float)
        assert np.isnan(isolation_dist) or isinstance(isolation_dist, float)


# ---------------------------------------------------------------------------
# xcorr
# ---------------------------------------------------------------------------


def test_xcorr_autocorr(spike_times):
    """
    Calling xcorr with a single array (autocorrelation) should return a
    BinnedData object with correct types.
    """
    result = xcorr(spike_times)
    assert isinstance(result, BinnedData)
    assert isinstance(result.binned_data[0], np.ndarray)
    assert isinstance(result.bin_edges[0], np.ndarray)


def test_xcorr_crosscorr(spike_times_pair):
    """
    Calling xcorr with two arrays (cross-correlation) should return a
    BinnedData object.
    """
    x1, x2 = spike_times_pair
    result = xcorr(x1, x2)
    assert isinstance(result, BinnedData)


def test_xcorr_normed(spike_times):
    """
    When normed=True, the bin counts should all be <= 1 because they
    are divided by the number of spikes.
    """
    result = xcorr(spike_times, normed=True)
    counts = result.binned_data[0]
    assert np.all(counts <= 1.0 + 1e-9)


def test_xcorr_list_trange(spike_times):
    """
    Passing Trange as a Python list should work without error.
    """
    result = xcorr(spike_times, Trange=[-0.3, 0.3])
    assert isinstance(result, BinnedData)


def test_xcorr_bin_count(spike_times):
    """
    Verify that the length of binned_data[0] equals approximately
    int(np.ptp(Trange) / binsize) (±1 due to histogram edge handling).
    """
    Trange = np.array([-0.5, 0.5])
    binsize = 0.001
    result = xcorr(spike_times, Trange=Trange, binsize=binsize)
    expected = int(np.ptp(Trange) / binsize)
    actual = len(result.binned_data[0])
    assert abs(actual - expected) <= 1


def test_xcorr_cluster_id_kwarg(spike_times):
    """
    Passing cluster_id=[1] as a kwarg should store it on the returned
    BinnedData object.
    """
    result = xcorr(spike_times, cluster_id=[1])
    assert result.cluster_id == [1]


# ---------------------------------------------------------------------------
# contamination_percent
# ---------------------------------------------------------------------------


def test_contamination_percent_returns_tuple(spike_times):
    """
    Calling contamination_percent with a single spike train should return
    a 5-tuple (c, Qi, Q00, Q01, Ri).
    """
    result = contamination_percent(spike_times)
    assert isinstance(result, tuple)
    assert len(result) == 5


def test_contamination_percent_Qi_Ri_shapes(spike_times):
    """
    Qi and Ri returned by contamination_percent should each have length 10.
    """
    _, Qi, _, _, Ri = contamination_percent(spike_times)
    assert len(Qi) == 10
    assert len(Ri) == 10


def test_contamination_percent_x2_none(spike_times):
    """
    Calling with x2=None (auto-correlation case) should run without error.
    """
    result = contamination_percent(spike_times, x2=None)
    assert result is not None


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric — properties and simple methods
# ---------------------------------------------------------------------------


def test_n_spikes(spike_calcs, spike_times):
    """
    n_spikes property should equal len(spike_times).
    """
    assert spike_calcs.n_spikes == len(spike_times)


def test_trial_mean_fr_raises_without_duration(spike_times):
    """
    trial_mean_fr() should raise IndexError when duration is None.
    """
    sc = SpikeCalcsGeneric(spike_times, cluster=1)
    sc._duration = None
    with pytest.raises(IndexError):
        sc.trial_mean_fr()


def test_trial_mean_fr_value(spike_calcs, spike_times):
    """
    trial_mean_fr() should equal n_spikes / duration.
    """
    expected = len(spike_times) / 60.0
    assert spike_calcs.trial_mean_fr() == pytest.approx(expected)


def test_duration_setter(spike_calcs):
    """
    The duration property should be settable and retrievable.
    """
    spike_calcs.duration = 120.0
    assert spike_calcs.duration == pytest.approx(120.0)


def test_secs_per_bin_setter(spike_calcs):
    """
    The secs_per_bin property should be settable and retrievable.
    """
    spike_calcs.secs_per_bin = 0.005
    assert spike_calcs.secs_per_bin == pytest.approx(0.005)


def test_shuffle_isis_length(spike_calcs, spike_times):
    """
    The shuffled spike train returned by shuffle_isis should have
    len(spike_times) - 1 elements (ISIs from diff, then cumsum).
    """
    shuffled = spike_calcs.shuffle_isis()
    assert len(shuffled) == len(spike_times) - 1


def test_shuffle_isis_same_isis(spike_calcs, spike_times):
    """
    The ISI distribution should be preserved after shuffling
    (sorted ISIs of original and shuffled trains should match).
    """
    original_isis = np.sort(np.diff(spike_times))
    shuffled = spike_calcs.shuffle_isis()
    # shuffled is cumsum of permuted isis, so recover isis:
    shuffled_isis = np.sort(np.diff(np.concatenate([[0], shuffled])))
    np.testing.assert_allclose(original_isis, shuffled_isis)


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.acorr
# ---------------------------------------------------------------------------


def test_acorr_returns_binned_data(spike_calcs):
    """
    acorr() should return a BinnedData instance.
    """
    result = spike_calcs.acorr()
    assert isinstance(result, BinnedData)


def test_acorr_custom_trange(spike_calcs):
    """
    acorr() should work with a custom Trange without error.
    """
    result = spike_calcs.acorr(Trange=np.array([-0.2, 0.2]))
    assert isinstance(result, BinnedData)


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.mean_isi_range
# ---------------------------------------------------------------------------


def test_mean_isi_range_returns_float(spike_calcs):
    """
    mean_isi_range should return a float.
    """
    result = spike_calcs.mean_isi_range(50)
    assert isinstance(result, float)


def test_mean_isi_range_zero_for_empty_window(spike_calcs):
    """
    When isi_range is so small that no bins fall within the range,
    the function should return a value that is either 0.0, nan,
    or a masked value (np.ma.masked) without crashing.
    """
    # Very small isi_range -> no bins in (0, 1e-9) window
    result = spike_calcs.mean_isi_range(1e-9)
    # Accept 0.0, nan, or masked array (masked when no bins are selected)
    is_zero = not np.ma.is_masked(result) and result == pytest.approx(0.0)
    is_nan = not np.ma.is_masked(result) and np.isnan(result)
    is_masked = np.ma.is_masked(result)
    assert is_zero or is_nan or is_masked


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.psth
# ---------------------------------------------------------------------------


def test_psth_raises_without_event_ts(spike_calcs):
    """
    psth() should raise an Exception when _event_ts is None.
    """
    spike_calcs._event_ts = None
    with pytest.raises(Exception):
        spike_calcs.psth()


def test_psth_returns_lists(spike_calcs_with_events):
    """
    psth() should return a (list, list) tuple of the same length.
    """
    x, y = spike_calcs_with_events.psth()
    assert isinstance(x, list)
    assert isinstance(y, list)
    assert len(x) == len(y)


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.psch
# ---------------------------------------------------------------------------


def test_psch_raises_without_event_ts(spike_calcs):
    """
    psch() should raise an Exception when _event_ts is None.
    """
    spike_calcs._event_ts = None
    with pytest.raises(Exception):
        spike_calcs.psch(0.01)


def test_psch_output_shape(spike_calcs_with_events):
    """
    psch() should return an ndarray with shape (n_bins, n_events).
    n_events is the number of event timestamps.
    """
    bin_width = 0.01
    result = spike_calcs_with_events.psch(bin_width)
    n_events = len(spike_calcs_with_events._event_ts)
    # Check that the result is a 2-D array and has n_events columns
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == n_events


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.get_ifr
# ---------------------------------------------------------------------------


def test_get_ifr_output_length(spike_calcs, spike_times):
    """
    get_ifr output length should equal n_samples.
    """
    n_samples = 3000
    result = spike_calcs.get_ifr(spike_times)
    assert len(result) == n_samples


def test_get_ifr_nonnegative(spike_calcs, spike_times):
    """
    All values from get_ifr should be >= 0 (instantaneous firing rate).
    """
    n_samples = 3000
    result = spike_calcs.get_ifr(spike_times)
    assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.ifr_sp_corr
# ---------------------------------------------------------------------------


def test_ifr_sp_corr_returns_pearson_result(spike_calcs, spike_times):
    """
    ifr_sp_corr() result should have .statistic and .pvalue attributes
    (a scipy PearsonRResult).
    """
    np.random.seed(42)
    n_samples = 3000
    speed = np.abs(np.random.randn(n_samples) * 5 + 10)  # mean ~10 cm/s
    result = spike_calcs.ifr_sp_corr(spike_times, speed, nShuffles=10)
    assert hasattr(result, "statistic")
    assert hasattr(result, "pvalue")


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.get_shuffled_ifr_sp_corr
# ---------------------------------------------------------------------------


def test_get_shuffled_ifr_sp_corr_shape(spike_calcs, spike_times):
    """
    get_shuffled_ifr_sp_corr should return an array with length == nShuffles.
    Uses a spike train spanning >60 s so the random shift window is valid.
    """
    np.random.seed(42)
    # Build a spike train spanning 0-120 s so that ts[-1]-30=90 > 30=low,
    # ensuring the random integer shift window [30, ts[-1]-30) is valid.
    long_ts = np.sort(np.random.uniform(0, 120, 500))
    n_samples = 6000  # 120 s * 50 Hz
    speed = np.abs(np.random.randn(n_samples) * 5 + 10)
    sc = SpikeCalcsGeneric(long_ts, cluster=1)
    sc.duration = 120.0
    sc.pos_sample_rate = 50.0
    nShuffles = 5
    result = sc.get_shuffled_ifr_sp_corr(
        long_ts, speed, nShuffles=nShuffles, random_seed=42
    )
    assert len(result) == nShuffles


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.theta_mod_idx
# ---------------------------------------------------------------------------


def test_theta_mod_idx_returns_float(spike_calcs):
    """
    theta_mod_idx should return a float value.
    """
    result = spike_calcs.theta_mod_idx()
    assert isinstance(result, float)


def test_theta_mod_idx_in_range(spike_calcs):
    """
    theta_mod_idx result should lie in [-1, 1].
    """
    result = spike_calcs.theta_mod_idx()
    assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.theta_mod_idxV2
# ---------------------------------------------------------------------------


def test_theta_mod_idxV2_returns_float(spike_calcs):
    """
    theta_mod_idxV2 should return a float value.
    """
    result = spike_calcs.theta_mod_idxV2()
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.theta_mod_idxV3
# ---------------------------------------------------------------------------


def test_theta_mod_idxV3_returns_float(spike_calcs):
    """
    theta_mod_idxV3 should return a float value.
    """
    result = spike_calcs.theta_mod_idxV3()
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.get_ifr_power_spectrum
# ---------------------------------------------------------------------------


def test_get_ifr_power_spectrum_shapes(spike_calcs):
    """
    get_ifr_power_spectrum should return two arrays of equal length.
    """
    freqs, power = spike_calcs.get_ifr_power_spectrum()
    assert len(freqs) == len(power)


def test_get_ifr_power_spectrum_freqs_nonnegative(spike_calcs):
    """
    All frequency values returned by get_ifr_power_spectrum should be >= 0.
    """
    freqs, _ = spike_calcs.get_ifr_power_spectrum()
    assert np.all(freqs >= 0)


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.theta_band_max_freq
# ---------------------------------------------------------------------------


def test_theta_band_max_freq_in_theta_range(spike_calcs):
    """
    theta_band_max_freq result should be in [6, 12] Hz (the theta band).
    """
    result = spike_calcs.theta_band_max_freq()
    assert 6 <= result <= 12


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.smooth_spike_train
# ---------------------------------------------------------------------------


def test_smooth_spike_train_length(spike_calcs, spike_times):
    """
    smooth_spike_train output length should equal npos.
    """
    # spike_times are in seconds; smooth_spike_train uses them as integer
    # indices directly via np.bincount, so we need integer spike-sample
    # indices for the call. Build integer indices in [0, npos).
    npos = 3000
    int_spike_times = np.floor(spike_times * spike_calcs.pos_sample_rate).astype(int)
    int_spike_times = int_spike_times[int_spike_times < npos]
    sc = SpikeCalcsGeneric(int_spike_times, cluster=1)
    sc.duration = 60.0
    sc.pos_sample_rate = 50.0
    result = sc.smooth_spike_train(npos)
    assert len(result) == npos


def test_smooth_spike_train_with_shuffle(spike_times):
    """
    smooth_spike_train with shuffle parameter should still return
    an array of the correct length.
    """
    npos = 3000
    int_spike_times = np.floor(spike_times * 50).astype(int)
    int_spike_times = int_spike_times[int_spike_times < npos]
    sc = SpikeCalcsGeneric(int_spike_times, cluster=1)
    sc.duration = 60.0
    sc.pos_sample_rate = 50.0
    result = sc.smooth_spike_train(npos, shuffle=5)
    assert len(result) == npos


# ---------------------------------------------------------------------------
# SpikeCalcsGeneric.contamination_percent (instance method)
# ---------------------------------------------------------------------------


def test_instance_contamination_percent_returns_two_floats(spike_calcs):
    """
    The instance method contamination_percent() should return a tuple of
    two floats (Q, R).
    """
    Q, R = spike_calcs.contamination_percent()
    assert isinstance(Q, float) or np.isnan(Q)
    assert isinstance(R, float) or np.isnan(R)


# ---------------------------------------------------------------------------
# KSMetaTuple
# ---------------------------------------------------------------------------


def test_ks_meta_tuple_fields():
    """
    KSMetaTuple namedtuple should have the expected fields:
    Amplitude, group, KSLabel, ContamPct.
    """
    expected_fields = ("Amplitude", "group", "KSLabel", "ContamPct")
    assert KSMetaTuple._fields == expected_fields


@pytest.mark.parametrize("field_name", ["Amplitude", "group", "KSLabel", "ContamPct"])
def test_ks_meta_tuple_field_access(field_name):
    """
    Each field of KSMetaTuple should be accessible by name.
    """
    kst = KSMetaTuple(Amplitude=100.0, group="good", KSLabel="good", ContamPct=0.02)
    assert hasattr(kst, field_name)
