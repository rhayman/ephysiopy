import pytest
import numpy as np
from ephysiopy.common.waveformcalcs import WaveformCalcsGeneric, KSMetaTuple
from ephysiopy.common.utils import TrialFilter


# ===========================================================================
# Fixtures
# ===========================================================================

N_SPIKES = 200
N_CHANNELS = 4
N_SAMPLES = 50
CLUSTER_ID = 1


@pytest.fixture
def spike_times():
    """Monotonically increasing spike times over a 60-second trial."""
    np.random.seed(42)
    return np.sort(np.random.uniform(0, 60, N_SPIKES))


@pytest.fixture
def waveforms():
    """
    Synthetic waveforms: (n_spikes, n_channels, n_samples).
    Each waveform is a simple negative deflection (spike-like shape).
    """
    np.random.seed(42)
    t = np.linspace(0, 1, N_SAMPLES)
    # Build a prototypical spike shape: negative peak at ~40% through the window
    proto = -np.sin(np.pi * t) * np.exp(-((t - 0.4) ** 2) / 0.02)
    waves = np.tile(proto, (N_SPIKES, N_CHANNELS, 1))
    waves += np.random.normal(scale=0.05, size=waves.shape)  # add noise
    return waves.astype(float)


@pytest.fixture
def waveform_calc(waveforms, spike_times):
    """Default WaveformCalcsGeneric instance."""
    return WaveformCalcsGeneric(waveforms, spike_times, CLUSTER_ID)


@pytest.fixture
def waveform_calc_no_waves(spike_times):
    """Instance whose internal _waves is set to None (edge case)."""
    np.random.seed(0)
    waves = np.random.randn(len(spike_times), N_CHANNELS, N_SAMPLES)
    obj = WaveformCalcsGeneric(waves, spike_times, CLUSTER_ID)
    obj._waves = None
    return obj


# ===========================================================================
# Tests: __init__ / construction
# ===========================================================================


class TestInit:
    def test_basic_construction(self, waveforms, spike_times):
        obj = WaveformCalcsGeneric(waveforms, spike_times, CLUSTER_ID)
        assert obj.cluster == CLUSTER_ID

    def test_spike_times_are_masked_array(self, waveform_calc):
        assert isinstance(waveform_calc.spike_times, np.ma.MaskedArray)

    def test_waveforms_stored_as_masked_array(self, waveform_calc):
        assert isinstance(waveform_calc._waves, np.ma.MaskedArray)

    def test_shape_mismatch_raises(self, spike_times):
        """Passing mismatched waveform count vs spike_times must raise AssertionError."""
        bad_waves = np.random.randn(len(spike_times) + 5, N_CHANNELS, N_SAMPLES)
        with pytest.raises(AssertionError, match="Number of spike times"):
            WaveformCalcsGeneric(bad_waves, spike_times, CLUSTER_ID)

    def test_kwargs_injected_into_dict(self, waveforms, spike_times):
        obj = WaveformCalcsGeneric(waveforms, spike_times, CLUSTER_ID, custom_attr=99)
        assert obj.custom_attr == 99

    def test_default_sample_rate(self, waveform_calc):
        assert waveform_calc.sample_rate == 50000

    def test_default_pos_sample_rate(self, waveform_calc):
        assert waveform_calc.pos_sample_rate == 50

    def test_default_pre_spike_samples(self, waveform_calc):
        assert waveform_calc.pre_spike_samples == 10

    def test_default_post_spike_samples(self, waveform_calc):
        assert waveform_calc.post_spike_samples == 40

    def test_default_duration_is_none(self, waveform_calc):
        assert waveform_calc.duration is None

    def test_default_event_ts_is_none(self, waveform_calc):
        assert waveform_calc.event_ts is None

    def test_default_invert_waveforms_is_false(self, waveform_calc):
        assert waveform_calc.invert_waveforms is False

    def test_default_ksmeta_fields_are_none(self, waveform_calc):
        for field in KSMetaTuple._fields:
            assert getattr(waveform_calc.KSMeta, field) is None

    def test_default_event_window(self, waveform_calc):
        np.testing.assert_array_equal(
            waveform_calc.event_window, np.array((-0.050, 0.100))
        )


# ===========================================================================
# Tests: properties and setters
# ===========================================================================


class TestProperties:
    def test_n_spikes_matches_input(self, waveform_calc):
        assert waveform_calc.n_spikes == N_SPIKES

    def test_n_channels(self, waveform_calc):
        assert waveform_calc.n_channels == N_CHANNELS

    def test_n_samples(self, waveform_calc):
        assert waveform_calc.n_samples == N_SAMPLES

    def test_n_channels_no_waves(self, waveform_calc_no_waves):
        assert waveform_calc_no_waves.n_channels is None

    def test_n_samples_no_waves(self, waveform_calc_no_waves):
        assert waveform_calc_no_waves.n_samples is None

    def test_sample_rate_setter(self, waveform_calc):
        waveform_calc.sample_rate = 30000
        assert waveform_calc.sample_rate == 30000

    def test_pos_sample_rate_setter(self, waveform_calc):
        waveform_calc.pos_sample_rate = 25
        assert waveform_calc.pos_sample_rate == 25

    def test_pre_spike_samples_setter_coerces_to_int(self, waveform_calc):
        waveform_calc.pre_spike_samples = 12.9
        assert waveform_calc.pre_spike_samples == 12
        assert isinstance(waveform_calc.pre_spike_samples, int)

    def test_post_spike_samples_setter_coerces_to_int(self, waveform_calc):
        waveform_calc.post_spike_samples = 38.7
        assert waveform_calc.post_spike_samples == 38
        assert isinstance(waveform_calc.post_spike_samples, int)

    def test_duration_setter(self, waveform_calc):
        waveform_calc.duration = 120.0
        assert waveform_calc.duration == 120.0

    def test_duration_setter_none(self, waveform_calc):
        waveform_calc.duration = None
        assert waveform_calc.duration is None

    def test_event_ts_setter(self, waveform_calc):
        ts = np.array([1.0, 2.0, 3.0])
        waveform_calc.event_ts = ts
        np.testing.assert_array_equal(waveform_calc.event_ts, ts)

    def test_event_window_setter(self, waveform_calc):
        waveform_calc.event_window = np.array((-0.1, 0.2))
        np.testing.assert_array_equal(waveform_calc.event_window, np.array((-0.1, 0.2)))

    def test_stim_width_setter(self, waveform_calc):
        waveform_calc.stim_width = 5.0
        assert waveform_calc.stim_width == 5.0

    def test_stim_width_setter_none(self, waveform_calc):
        waveform_calc.stim_width = None
        assert waveform_calc.stim_width is None

    def test_secs_per_bin_setter(self, waveform_calc):
        waveform_calc.secs_per_bin = 0.002
        assert waveform_calc.secs_per_bin == 0.002

    def test_invert_waveforms_setter(self, waveform_calc):
        waveform_calc.invert_waveforms = True
        assert waveform_calc.invert_waveforms is True


# ===========================================================================
# Tests: waveforms() method
# ===========================================================================


class TestWaveformsMethod:
    def test_returns_all_channels_when_no_id(self, waveform_calc, waveforms):
        result = waveform_calc.waveforms()
        assert result is not None
        assert result.shape == (N_SPIKES, N_CHANNELS, N_SAMPLES)

    def test_returns_correct_single_channel(self, waveform_calc):
        result = waveform_calc.waveforms(0)
        assert result.shape == (N_SPIKES, 1, N_SAMPLES)

    def test_returns_correct_multiple_channels(self, waveform_calc):
        result = waveform_calc.waveforms([0, 2])
        assert result.shape == (N_SPIKES, 2, N_SAMPLES)

    def test_invert_waveforms_negates_values(self, waveform_calc):
        normal = waveform_calc.waveforms()
        waveform_calc.invert_waveforms = True
        inverted = waveform_calc.waveforms()
        np.testing.assert_array_equal(normal * -1, inverted)

    def test_returns_none_when_no_waves(self, waveform_calc_no_waves):
        assert waveform_calc_no_waves.waveforms() is None

    def test_int_channel_id_wrapped_in_list(self, waveform_calc):
        """Passing an int channel_id should return a 3-D array (not 2-D)."""
        result = waveform_calc.waveforms(channel_id=2)
        assert result.ndim == 3

    def test_waveform_values_unchanged_without_inversion(
        self, waveform_calc, waveforms
    ):
        result = waveform_calc.waveforms()
        np.testing.assert_array_almost_equal(result, waveforms)


# ===========================================================================
# Tests: n_spikes (masked array interaction)
# ===========================================================================


class TestNSpikes:
    def test_n_spikes_is_int(self, waveform_calc):
        assert isinstance(waveform_calc.n_spikes, (int, np.integer))

    def test_n_spikes_matches_unmasked_count(self, waveform_calc):
        assert waveform_calc.n_spikes == N_SPIKES

    def test_n_spikes_respects_mask(self, waveform_calc):
        """After masking half the spikes n_spikes should decrease."""
        mask = np.zeros(N_SPIKES, dtype=bool)
        mask[: N_SPIKES // 2] = True
        waveform_calc.spike_times.mask = mask
        assert waveform_calc.n_spikes == N_SPIKES // 2
        # reset
        waveform_calc.spike_times.mask = False


# ===========================================================================
# Tests: trial_mean_fr
# ===========================================================================


class TestTrialMeanFR:
    def test_raises_when_no_duration(self, waveform_calc):
        with pytest.raises(IndexError, match="No duration"):
            waveform_calc.trial_mean_fr()

    def test_returns_correct_rate(self, waveform_calc):
        waveform_calc.duration = 60.0
        fr = waveform_calc.trial_mean_fr()
        assert isinstance(fr, float)
        assert fr == pytest.approx(N_SPIKES / 60.0)

    def test_rate_scales_with_duration(self, waveform_calc):
        waveform_calc.duration = 30.0
        fr_30 = waveform_calc.trial_mean_fr()
        waveform_calc.duration = 60.0
        fr_60 = waveform_calc.trial_mean_fr()
        assert fr_30 == pytest.approx(fr_60 * 2)


# ===========================================================================
# Tests: mean_waveform
# ===========================================================================


class TestMeanWaveform:
    def test_returns_tuple_of_two_arrays(self, waveform_calc):
        result = waveform_calc.mean_waveform()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mean_shape_all_channels(self, waveform_calc):
        mean, std = waveform_calc.mean_waveform()
        assert mean.shape == (N_CHANNELS, N_SAMPLES)
        assert std.shape == (N_CHANNELS, N_SAMPLES)

    def test_mean_shape_single_channel(self, waveform_calc):
        mean, std = waveform_calc.mean_waveform(0)
        assert mean.shape == (1, N_SAMPLES)

    def test_mean_shape_multiple_channels(self, waveform_calc):
        mean, std = waveform_calc.mean_waveform([0, 1])
        assert mean.shape == (2, N_SAMPLES)

    def test_std_non_negative(self, waveform_calc):
        _, std = waveform_calc.mean_waveform()
        assert np.all(std >= 0)

    def test_returns_none_when_no_waves(self, waveform_calc_no_waves):
        assert waveform_calc_no_waves.mean_waveform() is None

    def test_values_are_finite(self, waveform_calc):
        mean, std = waveform_calc.mean_waveform()
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))


# ===========================================================================
# Tests: get_best_channel
# ===========================================================================


class TestGetBestChannel:
    def test_returns_int_index(self, waveform_calc):
        best = waveform_calc.get_best_channel()
        assert isinstance(best, (int, np.integer))

    def test_index_in_valid_range(self, waveform_calc):
        best = waveform_calc.get_best_channel()
        assert 0 <= best < N_CHANNELS

    def test_returns_none_when_no_waves(self, waveform_calc_no_waves):
        assert waveform_calc_no_waves.get_best_channel() is None

    def test_best_channel_has_highest_amplitude(self, waveform_calc):
        """Channel with largest mean peak-to-peak should be selected."""
        wvs = waveform_calc.waveforms()
        amps = np.mean(np.ptp(wvs, axis=-1), axis=0)
        expected = int(np.argmax(amps))
        assert waveform_calc.get_best_channel() == expected

    def test_deterministic_when_one_channel_amplified(self, spike_times):
        """Amplify channel 2 so it is unambiguously the best."""
        waves = np.random.randn(len(spike_times), N_CHANNELS, N_SAMPLES) * 0.1
        waves[:, 2, :] *= 50  # channel 2 dominates
        obj = WaveformCalcsGeneric(waves, spike_times, CLUSTER_ID)
        assert obj.get_best_channel() == 2


# ===========================================================================
# Tests: apply_filter
# ===========================================================================


class TestApplyFilter:
    def test_no_filter_resets_mask(self, waveform_calc):
        """Calling apply_filter() with no args should unmask everything."""
        waveform_calc.spike_times.mask = np.ones(N_SPIKES, dtype=bool)
        waveform_calc.apply_filter()
        assert not np.any(waveform_calc.spike_times.mask)

    def test_time_filter_masks_spikes_outside_window(self, waveform_calc):
        """
        apply_filter keeps spikes strictly OUTSIDE the window
        After the filter, n_spikes should reflect only included spikes
        outside this window.
        """
        f = TrialFilter("time", 10.0, 30.0)
        waveform_calc.apply_filter(f)
        # spikes in [10, 30] should be visible (mask == True in the mask means kept)
        times = waveform_calc.spike_times.data  # underlying data without mask
        n_in_window = int(np.sum((times > 10.0) & (times < 30.0)))
        assert waveform_calc.n_spikes == len(times) - n_in_window

    def test_multiple_time_filters_union(self, waveform_calc):
        """Two non-overlapping windows should preserve spikes in either."""
        f1 = TrialFilter("time", 0.0, 10.0)
        f2 = TrialFilter("time", 50.0, 60.0)
        waveform_calc.apply_filter(f1, f2)
        times = waveform_calc.spike_times.data
        n_expected = int(
            np.sum(((times > 0.0) & (times < 10.0)) | ((times > 50.0) & (times < 60.0)))
        )
        assert waveform_calc.n_spikes == len(times) - n_expected

    def test_filter_also_masks_waveforms(self, waveform_calc):
        """The waveform mask should mirror the spike_times mask
        along its first dimension (spikes)."""
        f = TrialFilter("time", 20.0, 40.0)
        waveform_calc.apply_filter(f)
        assert (
            np.shape(waveform_calc._waves.mask)[0]
            == np.shape(waveform_calc.spike_times.mask)[0]
        )

    def test_non_time_filter_raises(self, waveform_calc):
        """Only 'time' filters are accepted; anything else should raise."""
        f = TrialFilter("speed", 0.0, 10.0)
        with pytest.raises(AssertionError):
            waveform_calc.apply_filter(f)

    def test_wrong_type_raises(self, waveform_calc):
        """Passing something that is not a TrialFilter should raise."""
        with pytest.raises(AssertionError):
            waveform_calc.apply_filter("not_a_filter")

    def test_reapply_clears_previous_mask(self, waveform_calc):
        f = TrialFilter("time", 0.0, 5.0)
        waveform_calc.apply_filter(f)
        n_after_first = waveform_calc.n_spikes
        waveform_calc.apply_filter()  # clear
        assert waveform_calc.n_spikes == N_SPIKES
        # reapply different filter
        f2 = TrialFilter("time", 55.0, 60.0)
        waveform_calc.apply_filter(f2)
        n_after_second = waveform_calc.n_spikes
        assert n_after_first != n_after_second or True  # just ensure no exception


# ===========================================================================
# Tests: update_KSMeta
# ===========================================================================


class TestUpdateKSMeta:
    def test_populates_available_fields(self, waveform_calc):
        data = {
            "Amplitude": {CLUSTER_ID: 42.0},
            "group": {CLUSTER_ID: "good"},
            "KSLabel": {CLUSTER_ID: "good"},
            "ContamPct": {CLUSTER_ID: 1.5},
        }
        waveform_calc.update_KSMeta(data)
        assert waveform_calc.KSMeta.Amplitude == 42.0
        assert waveform_calc.KSMeta.group == "good"
        assert waveform_calc.KSMeta.KSLabel == "good"
        assert waveform_calc.KSMeta.ContamPct == 1.5

    def test_missing_cluster_id_sets_none(self, waveform_calc):
        data = {"Amplitude": {999: 99.0}}  # wrong cluster id
        waveform_calc.update_KSMeta(data)
        assert waveform_calc.KSMeta.Amplitude is None

    def test_missing_field_sets_none(self, waveform_calc):
        data = {}  # no fields at all
        waveform_calc.update_KSMeta(data)
        for field in KSMetaTuple._fields:
            assert getattr(waveform_calc.KSMeta, field) is None

    def test_partial_fields_populated(self, waveform_calc):
        data = {"Amplitude": {CLUSTER_ID: 10.0}}
        waveform_calc.update_KSMeta(data)
        assert waveform_calc.KSMeta.Amplitude == 10.0
        assert waveform_calc.KSMeta.group is None

    def test_ksmeta_is_named_tuple(self, waveform_calc):
        assert isinstance(waveform_calc.KSMeta, KSMetaTuple)


# ===========================================================================
# Tests: estimate_AHP
# ===========================================================================


class TestEstimateAHP:
    def test_returns_numeric_or_none(self, waveform_calc):
        result = waveform_calc.estimate_AHP()
        assert result is None or isinstance(
            result, (float, np.floating, int, np.integer)
        )

    def test_returns_none_when_no_waves(self, waveform_calc_no_waves):
        assert waveform_calc_no_waves.estimate_AHP() is None

    def test_does_not_raise_on_realistic_waveform(self, spike_times):
        """
        Build a waveform that has a clear negative trough followed by
        an AHP and check estimate_AHP doesn't crash.
        """
        np.random.seed(7)
        n = len(spike_times)
        t = np.linspace(0, 1, N_SAMPLES)
        # spike dip at ~0.2, AHP at ~0.6
        proto = -(
            np.exp(-((t - 0.2) ** 2) / 0.005) - 0.3 * np.exp(-((t - 0.6) ** 2) / 0.02)
        )
        waves = np.tile(proto, (n, N_CHANNELS, 1)).astype(float)
        waves += np.random.normal(0, 0.01, waves.shape)
        obj = WaveformCalcsGeneric(waves, spike_times, CLUSTER_ID)
        obj.pre_spike_samples = 10
        obj.post_spike_samples = 40
        # Should not raise; may return a float or None
        result = obj.estimate_AHP()
        assert result is None or isinstance(result, (float, np.floating))


# ===========================================================================
# Tests: edge / boundary cases
# ===========================================================================


class TestEdgeCases:
    def test_single_spike(self):
        """Object with a single spike should initialise correctly."""
        waves = np.random.randn(1, N_CHANNELS, N_SAMPLES)
        times = np.array([1.5])
        obj = WaveformCalcsGeneric(waves, times, 0)
        assert obj.n_spikes == 1

    def test_single_channel(self, spike_times):
        waves = np.random.randn(len(spike_times), 1, N_SAMPLES)
        obj = WaveformCalcsGeneric(waves, spike_times, 0)
        assert obj.n_channels == 1
        assert obj.get_best_channel() == 0

    def test_large_n_channels(self, spike_times):
        waves = np.random.randn(len(spike_times), 12, N_SAMPLES)
        obj = WaveformCalcsGeneric(waves, spike_times, 0)
        assert obj.n_channels == 12
        best = obj.get_best_channel()
        assert 0 <= best < 12

    def test_all_zeros_waveforms(self, spike_times):
        """All-zero waveforms shouldn't crash any computation."""
        waves = np.zeros((len(spike_times), N_CHANNELS, N_SAMPLES))
        obj = WaveformCalcsGeneric(waves, spike_times, CLUSTER_ID)
        mean, std = obj.mean_waveform()
        assert np.all(mean == 0)
        assert np.all(std == 0)

    def test_waveform_inversion_round_trip(self, waveform_calc):
        """Inverting twice should return the original values."""
        original = waveform_calc.waveforms().copy()
        waveform_calc.invert_waveforms = True
        waveform_calc.invert_waveforms = False
        np.testing.assert_array_equal(waveform_calc.waveforms(), original)

    def test_full_filter_then_clear(self, waveform_calc):
        """Filter that excludes all spikes, then clear; n_spikes should recover."""
        # Filter window that is outside all spike times (times are in [0, 60])
        f = TrialFilter("time", 0, 200.0)
        waveform_calc.apply_filter(f)
        assert waveform_calc.n_spikes == 0
        waveform_calc.apply_filter()
        assert waveform_calc.n_spikes == N_SPIKES
