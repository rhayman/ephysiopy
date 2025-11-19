"""
The main reason for this file is to do some position preprocessing
for linear track data

You can either use the x coordinate as the position that gets fed
into ephysiopy.common.fieldproperties.fieldprops or phi which is
the euclidean distance along the linear track

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle
from ephysiopy.common.utils import (
    VariableToBin,
    TrialFilter,
)
from ephysiopy.io.recording import AxonaTrial
from ephysiopy.common.fieldproperties import (
    FieldProps,
    RunProps,
)
from ephysiopy.common.phasecoding import (
    phasePrecession2D,
    RegressionResults,
)

MIN_THETA = 6
MAX_THETA = 10
# defines start/ end of theta cycle
MIN_ALLOWED_SPIKE_PHASE = np.pi
# percentile power below which theta cycles are rejected
MIN_POWER_THRESHOLD = 20  # percentile - total guess at the value
POS_SAMPLE_RATE = 50  # Hz
# SOME FIELD THRESHOLDS
FIELD_THRESHOLD_PERCENT = 50
FIELD_RATE_THRESHOLD = 0.5
MIN_FIELD_SIZE_IN_BINS = 3
MIN_SPIKES_PER_RUN = 3
# SOME RUN THRESHOLDS
MIN_RUN_LENGTH = 1  # seconds
EXCLUDE_SPEEDS = (0, 0.5)  # cm/s
# CONSTANTS FOR THE REGRESSION
N_PERMUTATIONS = 1000
ALPHA = 0.05
# The hypothesis to test:
# -1 = negatively correlated
#  0 = correlated in either direction
# +1 = positively correlated
HYPOTHESIS = 0
# Calculate confidence intervals via jackknife (True) or bootstrap (False)
CONF = True


def run_phase_analysis(
    trial: AxonaTrial, cluster: int, channel: int, **kwargs
) -> list[RegressionResults]:
    """
    Run the phase analysis on a linear track trial.

    Parameters
    ----------
    trial (AxonaTrial) - the trial
    cluster (int) - the cluster id
    channel (int) - the channel id
    kwargs (dict) - additional parameters to pass to the

    Returns
    -------
    list[RegressionResults] - list of RegressionResults
           ordered by the field id and the regression
           results for that field

    """
    # remove any pre-existing filters
    trial.apply_filter()
    PP = phasePrecession2D(
        trial,
        cluster,
        channel,
        regressors=["pos_d_normed_x"],
        method="clump_runs",
        **kwargs,
    )

    phase_pos, f_props, run_direction, n_fields, PP = fieldprops_phase_precession(
        PP, **kwargs
    )
    if phase_pos is None:
        return None

    plot = kwargs.get("plot", False)
    try:
        results = PP.do_correlation(phase_pos, plot=plot)
    except Exception:
        results = None

    if kwargs.get("return_pp", False):
        return results, PP

    return results


def fieldprops_phase_precession(P: phasePrecession2D, **kwargs):
    """
    Run the phase analysis on a linear track trial.

    Parameters
    ----------
    trial (AxonaTrial) - the trial
    cluster (int) - the cluster id
    channel (int) - the channel id
    kwargs (dict) - additional parameters to pass to the

    Returns
    -------
    dict - a dictionary with the field id as the key and the correlation
           results for that field as the value

    """
    run_direction = kwargs.get("run_direction", "e")
    # get the field properties for the linear track
    f_props = get_field_props_for_linear_track(
        P, direction=run_direction, **kwargs)
    if not f_props:
        return None, None, None, None, None
    # merge the LFP data with the field properties
    # each field in f_props will now have an LFPSegment instance
    # attached to it

    f_props = P.get_theta_props(f_props)

    phase_pos = P.get_phase_reg_per_field(f_props)

    return phase_pos, f_props, run_direction, len(phase_pos), P


def get_field_props_for_linear_track(
    P: phasePrecession2D,
    direction=None,
    var_type=VariableToBin.X,
    **kwargs,
) -> list[FieldProps]:
    """
    Get the field properties for a linear track trial.

    Filters the linear track data based on speed, direction (east
    or west; larger ranges than the usual 90degs are used), and position
    (masks the start and end 12cm of the track)

    Paraemters
    ----------
    trial (AxonaTrial) - the trial
    cluster (int) - the cluster id
    channel (int) - the channel id

    kwargs (dict) - additional parameters to pass to the
                    get_rate_map() function and additionally apply
                    filtering to the result of that (a BinnedData
                    instance). Rationale is that sometimes we might
                    want to limit the field extent according to
                    different criteria, e.g. field size in bins,
                    field rate threshold (mean, peak, etc.), etc.
    """
    P = apply_linear_track_filter(P, direction=direction, var_type=var_type)

    # partition the cells firing into distinct fields
    binned_data = P.trial.get_linear_rate_map(
        P.cluster, P.channel, var_type=var_type, **kwargs
    )
    # optionally apply a function to the binned data before
    # the field partitioning - e.g. smoothing or normalisation -
    # we might want this to act inside the fancy_partition function...
    if kwargs.get("ratemap_func", None):
        binned_data = kwargs["ratemap_func"](binned_data)

    field_props = P.get_pos_props(binned_data, var_type=var_type, **kwargs)

    return field_props


def apply_linear_track_filter(
    P: phasePrecession2D, direction=None, var_type=VariableToBin.X
):

    # filter the data for speed, direction and position
    # position filter is there to remove the start and end of the track
    speed_filter = TrialFilter("speed", EXCLUDE_SPEEDS[0], EXCLUDE_SPEEDS[1])
    if var_type.value == VariableToBin.PHI.value:
        min_pos = np.nanmin(P.trial.PosCalcs.phi.data)
        max_pos = np.nanmax(P.trial.PosCalcs.phi.data)
        pos_filt0 = TrialFilter("xrange", min_pos, min_pos + 6)
        pos_filt1 = TrialFilter("xrange", max_pos - 6, max_pos)
    if var_type.value == VariableToBin.X.value:
        min_pos = np.nanmin(P.trial.PosCalcs.xy[0].data)
        max_pos = np.nanmax(P.trial.PosCalcs.xy[0].data)
        pos_filt0 = TrialFilter("xrange", min_pos, min_pos + 6)
        pos_filt1 = TrialFilter("xrange", max_pos - 6, max_pos)
    if direction:
        # broaden the directional filters to 180 degrees
        # as runs are getting broken up
        if direction == "e" or direction == "east":
            # filter out east direction - LEAVING onyl west
            # dir_filter0 = TrialFilter("dir", 270, 90)
            dir_filter0 = TrialFilter("dir", "e")
        elif direction == "w" or direction == "west":
            # filter out west direction - LEAVING only east
            # dir_filter0 = TrialFilter("dir", 90, 270)
            dir_filter0 = TrialFilter("dir", "w")

        n = TrialFilter("dir", "n")
        s = TrialFilter("dir", "s")
        P.trial.apply_filter(pos_filt0, pos_filt1,
                             speed_filter, dir_filter0, n, s)
    else:
        P.trial.apply_filter(
            pos_filt0,
            pos_filt1,
            speed_filter,
        )
    return P


def get_run_direction(run: RunProps):
    df = run.xy[0][0] - run.xy[0][-1]
    if df > 0:
        return "w"
    return "e"


def add_normalised_run_position(f_props: list[FieldProps]) -> list[FieldProps]:
    """
    Adds the normalised run position to each run through a field in field_props
    where the run x position is normalised with respect to the
    field x position limits and the run direction (east or west)
    """
    for field in f_props:
        xmin = np.min(field.xy[0])
        xmax = np.max(field.xy[0])
        fp_e = np.linspace(-1, 1, 1000)
        fp_w = np.linspace(1, -1, 1000)
        xp = np.linspace(xmin, xmax, 1000)
        for run in field.runs:
            if get_run_direction(run) == "e":
                x_nrmd = np.interp(run.xy[0], xp, fp_e)
            else:
                x_nrmd = np.interp(run.xy[0], xp, fp_w)
            run.normalised_position = x_nrmd
    return f_props


def plot_linear_runs(f_props: list[FieldProps], var: str = "speed", **kwargs):
    """
    Plots the runs through the field(s) on a linear track
    as a sort of raster plot with each run as a separate line on
    the y-axis with ticks for each spike occurring on each run.
    For each run the height of
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    edges = f_props[0].binned_data.bin_edges[0]
    rate = f_props[0].binned_data.binned_data[0]

    ax.plot(
        edges[1::],
        rate,
        color="gray",
        linestyle="-",
    )

    axTrans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for f in f_props:

        rect = Rectangle(
            (edges[f.slice[0].start], 0),
            width=edges[f.slice[0].stop] - edges[f.slice[0].start],
            height=1,
            alpha=0.8,
            color="lightblue",
            transform=axTrans,
        )
        ax.add_patch(rect)

        inc = 1 / len(f.runs)
        for i, r in enumerate(f.runs):
            spike_x = np.ma.compressed(r.spiking_var("xy"))
            spike_ymax = np.ones_like(spike_x) * (inc * (i + 1))
            spike_ymin = np.zeros_like(spike_x) * (inc * (i + 1))
            ax.vlines(
                x=spike_x,
                ymin=spike_ymin,
                ymax=spike_ymax,
                # colors="k",
                transform=axTrans,
            )
