"""
Comprehensive pytest test suite for the RateMap class
(src/ephysiopy/common/binning.py).

Relies on fixtures defined in conftest.py:
  - basic_xy        : (x, y) random-walk tuple
  - basic_PosCalcs  : PosCalcsGeneric instance (xyTS set, postprocesspos NOT yet called)
  - basic_BinnedData: BinnedData(XY, RATE, [100x100 array], [x_edges, y_edges])
"""

import numpy as np
import pytest

from ephysiopy.common.binning import RateMap
from ephysiopy.common.utils import BinnedData, VariableToBin, MapType


# ---------------------------------------------------------------------------
# Extra fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def standard_ratemap(basic_PosCalcs):
    """RateMap backed by a random walk; pos data fully post-processed."""
    P = basic_PosCalcs
    P.postprocesspos(tracker_params={"AxonaBadValue": 1023})
    return RateMap(P)


@pytest.fixture
def spk_weights(standard_ratemap):
    """Sparse spike-weight array (~5 % of positions have spikes)."""
    np.random.seed(42)
    n = len(standard_ratemap.pos_weights)
    w = np.ma.MaskedArray(np.zeros(n))
    idx = np.random.choice(n, size=max(1, int(n * 0.05)), replace=False)
    w[idx] = np.random.randint(1, 4, size=len(idx)).astype(float)
    return w


@pytest.fixture
def small_binned_data():
    """A 10×10 BinnedData used where basic_BinnedData's 100×100 is too slow."""
    x = np.linspace(-np.pi, np.pi, 10)
    y = np.linspace(-np.pi, np.pi, 10)
    xx, yy = np.meshgrid(x, y)
    r = np.sin(xx) * np.cos(yy)
    return BinnedData(
        VariableToBin.XY,
        MapType.RATE,
        [r],
        [x, y],
    )


# ---------------------------------------------------------------------------
# 1. Constructor & properties
# ---------------------------------------------------------------------------


class TestRateMapInit:
    """Tests for RateMap.__init__ and basic property accessors."""

    def test_init_defaults(self, standard_ratemap):
        R = standard_ratemap
        assert R.binsize == 3, "Default binsize should be 3"
        assert R.smooth_sz == 5, "Default smooth_sz should be 5"
        assert R.smoothingType == "gaussian", (
            "Default smoothingType should be 'gaussian'"
        )
        assert R.whenToSmooth == "before", "Default whenToSmooth should be 'before'"
        assert R.var2Bin == VariableToBin.XY, "Default var2Bin should be XY"
        assert R.mapType == MapType.RATE, "Default mapType should be RATE"

    def test_init_custom_params(self, basic_PosCalcs):
        basic_PosCalcs.postprocesspos(tracker_params={"AxonaBadValue": 1023})
        R = RateMap(basic_PosCalcs, binsize=5, smooth_sz=3)
        assert R.binsize == 5, "Custom binsize should be 5"
        assert R.smooth_sz == 3, "Custom smooth_sz should be 3"

    def test_xy_property(self, standard_ratemap):
        xy = standard_ratemap.xy
        assert xy.ndim == 2, "xy should be 2-D"
        assert xy.shape[0] == 2, "xy first dimension should be 2 (x and y)"

    def test_dir_property(self, standard_ratemap):
        d = standard_ratemap.dir
        assert d.ndim == 1, "dir should be 1-D"
        assert len(d) == standard_ratemap.xy.shape[1], (
            "dir length should match n_samples"
        )

    def test_speed_property(self, standard_ratemap):
        s = standard_ratemap.speed
        assert s.ndim == 1, "speed should be 1-D"
        assert len(s) == standard_ratemap.xy.shape[1], (
            "speed length should match n_samples"
        )

    def test_pos_times_property(self, standard_ratemap):
        pt = standard_ratemap.pos_times
        assert pt is not None, "pos_times should not be None"
        assert pt.ndim == 1, "pos_times should be 1-D"

    def test_ppm_property(self, standard_ratemap):
        # conftest creates PosCalcsGeneric with ppm=300
        assert standard_ratemap.ppm == 300, (
            "ppm should equal the value passed to PosCalcsGeneric"
        )

    def test_pos_weights_default(self, standard_ratemap):
        pw = standard_ratemap.pos_weights
        assert isinstance(pw, np.ma.MaskedArray), "pos_weights should be a MaskedArray"
        assert len(pw) == standard_ratemap.PosCalcs.npos, (
            "pos_weights length should equal npos"
        )
        assert np.all(pw.data == 1), "Default pos_weights data should be all ones"

    def test_pos_weights_setter(self, standard_ratemap):
        n = standard_ratemap.PosCalcs.npos
        custom = np.ma.MaskedArray(np.full(n, 2.0))
        standard_ratemap.pos_weights = custom
        assert np.all(standard_ratemap.pos_weights.data == 2.0), (
            "pos_weights setter did not persist"
        )

    def test_spike_weights_setter(self, standard_ratemap):
        n = standard_ratemap.PosCalcs.npos
        sw = np.ma.MaskedArray(np.ones(n))
        standard_ratemap.spike_weights = sw
        assert standard_ratemap.spike_weights is sw, (
            "spike_weights setter did not persist"
        )

    def test_binsize_setter(self, standard_ratemap):
        standard_ratemap.binsize = 5
        assert standard_ratemap.binsize == 5, "binsize setter failed"
        assert standard_ratemap.binedges is not None, (
            "bin edges should be recalculated after binsize change"
        )

    def test_smooth_sz_setter(self, standard_ratemap):
        standard_ratemap.smooth_sz = 7
        assert standard_ratemap.smooth_sz == 7, "smooth_sz setter failed"

    def test_smoothingType_setter(self, standard_ratemap):
        standard_ratemap.smoothingType = "boxcar"
        assert standard_ratemap.smoothingType == "boxcar", "smoothingType setter failed"

    def test_sample_to_bin_property(self, standard_ratemap):
        dummy = np.arange(10)
        standard_ratemap.sample_to_bin = dummy
        result = standard_ratemap.sample_to_bin
        np.testing.assert_array_equal(
            result, dummy, err_msg="sample_to_bin property round-trip failed"
        )


# ---------------------------------------------------------------------------
# 2. _calc_bin_edges
# ---------------------------------------------------------------------------


class TestCalcBinEdges:
    """Tests for RateMap._calc_bin_edges."""

    def test_xy_edges(self, standard_ratemap):
        edges = standard_ratemap._calc_bin_edges(binsize=3)
        assert isinstance(edges, (tuple, list)), "XY edges should be a tuple/list"
        assert len(edges) == 2, "XY edges should have 2 elements (y, x)"

    def test_dir_edges(self, standard_ratemap):
        standard_ratemap.var2Bin = VariableToBin.DIR
        edges = standard_ratemap._calc_bin_edges(binsize=6)
        # edges is a list of floats covering 0–360
        arr = np.asarray(edges)
        assert arr[0] >= 0, "DIR edges should start at 0"
        assert arr[-1] <= 360, "DIR edges should end at or before 360"

    def test_speed_edges(self, standard_ratemap):
        standard_ratemap.var2Bin = VariableToBin.SPEED
        edges = standard_ratemap._calc_bin_edges(binsize=5)
        arr = np.asarray(edges)
        assert arr[0] == 0, "SPEED edges should start at 0"

    def test_x_edges(self, standard_ratemap):
        standard_ratemap.var2Bin = VariableToBin.X
        edges = standard_ratemap._calc_bin_edges(binsize=3)
        assert isinstance(edges, np.ndarray), "X edges should be an ndarray"
        assert edges.ndim == 1, "X edges should be 1-D"

    def test_y_edges(self, standard_ratemap):
        standard_ratemap.var2Bin = VariableToBin.Y
        edges = standard_ratemap._calc_bin_edges(binsize=3)
        assert isinstance(edges, np.ndarray), "Y edges should be an ndarray"
        assert edges.ndim == 1, "Y edges should be 1-D"

    def test_custom_binsize_fewer_bins(self, standard_ratemap):
        standard_ratemap.var2Bin = VariableToBin.XY
        edges_small = standard_ratemap._calc_bin_edges(binsize=3)
        edges_large = standard_ratemap._calc_bin_edges(binsize=10)
        n_small = len(edges_small[0])
        n_large = len(edges_large[0])
        assert n_large <= n_small, "Larger binsize should produce fewer (or equal) bins"

    def test_binedges_set_on_ratemap(self, standard_ratemap):
        standard_ratemap.var2Bin = VariableToBin.XY
        standard_ratemap._calc_bin_edges()
        assert standard_ratemap.binedges is not None, (
            "binedges should be set after _calc_bin_edges"
        )


# ---------------------------------------------------------------------------
# 3. _getXYLimits
# ---------------------------------------------------------------------------


class TestGetXYLimits:
    """Tests for RateMap._getXYLimits."""

    def test_returns_min_max(self, standard_ratemap):
        x_lims, y_lims = standard_ratemap._getXYLimits()
        assert len(x_lims) == 2, "x_lims should have 2 elements"
        assert len(y_lims) == 2, "y_lims should have 2 elements"
        assert x_lims[0] < x_lims[1], "x min should be less than x max"
        assert y_lims[0] < y_lims[1], "y min should be less than y max"

    def test_sets_instance_lims(self, standard_ratemap):
        standard_ratemap._getXYLimits()
        assert standard_ratemap.x_lims is not None, "x_lims should be set"
        assert standard_ratemap.y_lims is not None, "y_lims should be set"


# ---------------------------------------------------------------------------
# 4. _bin_data
# ---------------------------------------------------------------------------


class TestBinData:
    """Tests for RateMap._bin_data."""

    def test_bin_1d_data(self, standard_ratemap):
        speed = standard_ratemap.speed
        bins = [np.linspace(0, np.nanmax(speed), 20)]
        w = np.ones(len(speed))
        result, edges = standard_ratemap._bin_data(speed, bins, w)
        assert isinstance(result, np.ndarray), "1-D binned result should be ndarray"
        assert result.ndim == 1, "1-D bin result should be 1-D"

    def test_bin_2d_data(self, standard_ratemap):
        xy = standard_ratemap.xy
        bin_edges = standard_ratemap.binedges  # set during __init__
        w = np.ones(xy.shape[1])
        result, edges = standard_ratemap._bin_data(xy, bin_edges, w)
        assert isinstance(result, np.ndarray), "2-D binned result should be ndarray"
        assert result.ndim == 2, "2-D bin result should be 2-D"

    def test_bin_data_none_weights(self, standard_ratemap):
        speed = standard_ratemap.speed
        bins = [np.linspace(0, np.nanmax(speed), 20)]
        result, _ = standard_ratemap._bin_data(speed, bins, None)
        assert isinstance(result, np.ndarray), (
            "None-weights bin should still return ndarray"
        )

    def test_bin_data_multi_weights(self, standard_ratemap):
        xy = standard_ratemap.xy
        n = xy.shape[1]
        bin_edges = standard_ratemap.binedges
        w2 = np.ones((2, n))
        result, _ = standard_ratemap._bin_data(xy, bin_edges, w2)
        assert isinstance(result, list), "Multi-weight binning should return a list"
        assert len(result) == 2, (
            "Multi-weight result should have one entry per weight row"
        )


# ---------------------------------------------------------------------------
# 5. get_map with VariableToBin.XY
# ---------------------------------------------------------------------------


class TestGetMapXY:
    """Tests for RateMap.get_map with XY binning."""

    def test_rate_map_returns_BinnedData(self, standard_ratemap, spk_weights):
        result = standard_ratemap.get_map(spk_weights, VariableToBin.XY, MapType.RATE)
        assert isinstance(result, BinnedData), "get_map(RATE) should return BinnedData"

    def test_rate_map_shape(self, standard_ratemap, spk_weights):
        result = standard_ratemap.get_map(spk_weights, VariableToBin.XY, MapType.RATE)
        assert len(result.binned_data) > 0, "binned_data should be non-empty"
        assert result.binned_data[0].ndim == 2, "XY rate map should be 2-D"

    def test_pos_map(self, standard_ratemap, spk_weights):
        result = standard_ratemap.get_map(spk_weights, VariableToBin.XY, MapType.POS)
        assert isinstance(result, BinnedData), "get_map(POS) should return BinnedData"
        assert result.map_type == MapType.POS, "map_type should be POS"

    def test_spk_map(self, standard_ratemap, spk_weights):
        result = standard_ratemap.get_map(spk_weights, VariableToBin.XY, MapType.SPK)
        assert isinstance(result, BinnedData), "get_map(SPK) should return BinnedData"

    def test_rate_map_no_smoothing(self, standard_ratemap, spk_weights):
        result = standard_ratemap.get_map(
            spk_weights, VariableToBin.XY, MapType.RATE, smoothing=False
        )
        assert isinstance(result, BinnedData), (
            "get_map(smoothing=False) should return BinnedData"
        )

    def test_rate_map_smooth_after(self, standard_ratemap, spk_weights):
        standard_ratemap.whenToSmooth = "after"
        result = standard_ratemap.get_map(spk_weights, VariableToBin.XY, MapType.RATE)
        assert isinstance(result, BinnedData), "smooth-after should return BinnedData"

    def test_adaptive_map(self, standard_ratemap, spk_weights):
        result = standard_ratemap.get_map(
            spk_weights, VariableToBin.XY, MapType.ADAPTIVE
        )
        assert isinstance(result, BinnedData), (
            "get_map(ADAPTIVE) should return BinnedData"
        )


# ---------------------------------------------------------------------------
# 6. get_map with other variable types
# ---------------------------------------------------------------------------


class TestGetMapVariants:
    """Tests for get_map with DIR, SPEED, X, Y variables."""

    @pytest.mark.parametrize(
        "var_type",
        [
            VariableToBin.DIR,
            VariableToBin.SPEED,
            VariableToBin.X,
            VariableToBin.Y,
        ],
    )
    def test_variable_rate_map(self, standard_ratemap, spk_weights, var_type):
        result = standard_ratemap.get_map(spk_weights, var_type, MapType.RATE)
        assert isinstance(result, BinnedData), (
            f"get_map({var_type.name}, RATE) should return BinnedData"
        )

    @pytest.mark.parametrize(
        "var_type",
        [
            VariableToBin.DIR,
            VariableToBin.SPEED,
            VariableToBin.X,
            VariableToBin.Y,
        ],
    )
    def test_variable_pos_map(self, standard_ratemap, spk_weights, var_type):
        result = standard_ratemap.get_map(spk_weights, var_type, MapType.POS)
        assert isinstance(result, BinnedData), (
            f"get_map({var_type.name}, POS) should return BinnedData"
        )


# ---------------------------------------------------------------------------
# 7. apply_mask
# ---------------------------------------------------------------------------


class TestApplyMask:
    """Tests for RateMap.apply_mask."""

    def test_apply_mask_zeros(self, standard_ratemap):
        n = standard_ratemap.PosCalcs.npos
        standard_ratemap.apply_mask(np.zeros(n, dtype=bool))
        assert not np.any(standard_ratemap.pos_weights.mask), (
            "All-zero mask should result in no masked positions"
        )

    def test_apply_mask_partial(self, standard_ratemap):
        n = standard_ratemap.PosCalcs.npos
        mask = np.zeros(n, dtype=bool)
        mask[:10] = True
        standard_ratemap.apply_mask(mask)
        assert np.sum(standard_ratemap.pos_weights.mask) == 10, (
            "Partial mask should mask exactly 10 positions"
        )


# ---------------------------------------------------------------------------
# 8. get_samples_when_spiking
# ---------------------------------------------------------------------------


class TestGetSamplesWhenSpiking:
    """Tests for RateMap.get_samples_when_spiking."""

    def test_raises_when_no_spike_weights(self, standard_ratemap):
        standard_ratemap._spike_weights = None
        with pytest.raises(ValueError, match="Spike weights not set"):
            standard_ratemap.get_samples_when_spiking()

    def test_returns_array_when_set(self, standard_ratemap, spk_weights):
        # get_map sets spike_weights internally; set sample first
        standard_ratemap.get_map(spk_weights, VariableToBin.XY, MapType.RATE)
        result = standard_ratemap.get_samples_when_spiking()
        assert result is not None, (
            "get_samples_when_spiking should return a non-None result"
        )


# ---------------------------------------------------------------------------
# 9. autoCorr2D / _autoCorr2D
# ---------------------------------------------------------------------------


class TestAutoCorr2D:
    """Tests for RateMap.autoCorr2D and _autoCorr2D."""

    def test_autoCorr2D_returns_BinnedData(self, standard_ratemap, small_binned_data):
        result = standard_ratemap.autoCorr2D(small_binned_data)
        assert isinstance(result, BinnedData), "autoCorr2D should return BinnedData"
        assert result.map_type == MapType.AUTO_CORR, "map_type should be AUTO_CORR"

    def test_autoCorr2D_output_has_data(self, standard_ratemap, small_binned_data):
        result = standard_ratemap.autoCorr2D(small_binned_data)
        assert len(result.binned_data) > 0, "autoCorr2D result should have binned_data"

    def test_autoCorr2D_output_shape(self, standard_ratemap, small_binned_data):
        result = standard_ratemap.autoCorr2D(small_binned_data)
        orig_shape = small_binned_data.binned_data[0].shape
        out_shape = result.binned_data[0].shape
        # autocorrelogram should be (2*m - 1) x (2*n - 1) — larger than input
        assert out_shape[0] >= orig_shape[0], (
            "autocorrelogram height should be >= input"
        )
        assert out_shape[1] >= orig_shape[1], "autocorrelogram width should be >= input"

    def test_internal_autoCorr2D(self, standard_ratemap):
        A = np.random.rand(8, 8)
        nodwell = np.isnan(A)
        result = standard_ratemap._autoCorr2D(A, nodwell)
        assert result.ndim == 2, "_autoCorr2D should return a 2-D array"


# ---------------------------------------------------------------------------
# 10. crossCorr2D / _crossCorr2D
# ---------------------------------------------------------------------------


class TestCrossCorr2D:
    """Tests for RateMap.crossCorr2D and _crossCorr2D."""

    def test_crossCorr2D_returns_BinnedData(self, standard_ratemap, small_binned_data):
        A = small_binned_data
        B = small_binned_data
        A_nd = np.isnan(A.binned_data[0])
        B_nd = np.isnan(B.binned_data[0])
        result = standard_ratemap.crossCorr2D(A, B, A_nd, B_nd)
        assert isinstance(result, BinnedData), "crossCorr2D should return BinnedData"

    def test_internal_crossCorr2D_shape(self, standard_ratemap):
        A = np.random.rand(8, 8)
        B = np.random.rand(8, 8)
        A_nd = np.isnan(A)
        B_nd = np.isnan(B)
        result = standard_ratemap._crossCorr2D(A, B, A_nd, B_nd)
        assert result.ndim == 2, "_crossCorr2D should return a 2-D array"

    def test_crossCorr2D_same_inputs(self, standard_ratemap, small_binned_data):
        A = small_binned_data
        A_nd = np.isnan(A.binned_data[0])
        result = standard_ratemap.crossCorr2D(A, A, A_nd, A_nd)
        assert len(result.binned_data) > 0, (
            "crossCorr2D with same input should produce data"
        )


# ---------------------------------------------------------------------------
# 11. getAdaptiveMap
# ---------------------------------------------------------------------------


class TestGetAdaptiveMap:
    """Tests for RateMap.getAdaptiveMap."""

    @pytest.fixture
    def pos_spk_pair(self):
        """Simple 10×10 pos and spk arrays for adaptive map tests."""
        np.random.seed(0)
        pos = np.abs(np.random.rand(10, 10)) + 0.01  # avoid all-zero rows
        spk = np.abs(np.random.rand(10, 10))
        return pos, spk

    def test_returns_three_arrays(self, standard_ratemap, pos_spk_pair):
        pos, spk = pos_spk_pair
        result = standard_ratemap.getAdaptiveMap(pos, spk, alpha=4)
        assert len(result) == 3, "getAdaptiveMap should return a 3-tuple"

    def test_output_shape_matches_input(self, standard_ratemap, pos_spk_pair):
        pos, spk = pos_spk_pair
        rate, spk_out, pos_out = standard_ratemap.getAdaptiveMap(pos.copy(), spk.copy())
        assert rate.shape == pos.shape, "Rate map shape should match input"
        assert spk_out.shape == spk.shape, "Smoothed spk shape should match input"
        assert pos_out.shape == pos.shape, "Smoothed pos shape should match input"

    def test_nans_where_unvisited(self, standard_ratemap, pos_spk_pair):
        pos, spk = pos_spk_pair
        pos[0, 0] = 0  # force one unvisited bin
        rate, _, _ = standard_ratemap.getAdaptiveMap(pos.copy(), spk.copy())
        assert np.isnan(rate[0, 0]), "Unvisited bin should be NaN in rate map"


# ---------------------------------------------------------------------------
# 12. tWinSAC
# ---------------------------------------------------------------------------


class TestTWinSAC:
    """Tests for RateMap.tWinSAC."""

    @pytest.fixture
    def twinsac_data(self, standard_ratemap):
        """Synthetic xy and spike indices for tWinSAC."""
        np.random.seed(7)
        n = standard_ratemap.PosCalcs.npos
        xy = standard_ratemap.xy.data  # (2, n) pixels
        # spike at ~10 % of positions
        spk_idx = np.sort(np.random.choice(n, size=max(2, n // 10), replace=False))
        return xy, spk_idx

    def test_twinsac_returns_BinnedData(self, standard_ratemap, twinsac_data):
        xy, spk_idx = twinsac_data
        result = standard_ratemap.tWinSAC(xy, spk_idx, ppm=300, winSize=5)
        assert isinstance(result, BinnedData), "tWinSAC should return BinnedData"

    def test_twinsac_output_has_data(self, standard_ratemap, twinsac_data):
        xy, spk_idx = twinsac_data
        result = standard_ratemap.tWinSAC(xy, spk_idx, ppm=300, winSize=5)
        assert len(result.binned_data) > 0, "tWinSAC binned_data should be non-empty"
        assert result.binned_data[0].ndim == 2, "tWinSAC map should be 2-D"


# ---------------------------------------------------------------------------
# 13. _calc_ego_angles
# ---------------------------------------------------------------------------


class TestCalcEgoAngles:
    """Tests for RateMap._calc_ego_angles."""

    def test_circle_arena_returns_angles_and_xy(self, standard_ratemap):
        angles, arena_xy = standard_ratemap._calc_ego_angles(arena_shape="circle")
        assert angles is not None, "Angles should not be None"
        assert arena_xy is not None, "Arena xy should not be None"
        assert angles.ndim == 2, "Angles should be 2-D (arena_pts × npos)"
        assert arena_xy.ndim == 2, "Arena xy should be 2-D"

    def test_angles_in_range(self, standard_ratemap):
        angles, _ = standard_ratemap._calc_ego_angles(arena_shape="circle")
        assert np.all(angles >= 0), "Ego angles should be >= 0"
        assert np.all(angles <= 2 * np.pi), "Ego angles should be <= 2π"

    def test_square_arena(self, standard_ratemap):
        angles, arena_xy = standard_ratemap._calc_ego_angles(arena_shape="square")
        assert angles is not None, "Square-arena angles should not be None"
        assert arena_xy is not None, "Square-arena xy should not be None"
        assert angles.ndim == 2, "Square-arena angles should be 2-D"
