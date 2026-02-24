import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.collections import RegularPolyCollection
import warnings
import skimage
import copy
from scipy import ndimage
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy import spatial
from scipy import stats
import skimage.filters as skifilters
import scipy.signal as signal
from astropy.convolution import Gaussian1DKernel as gk1d
from astropy.convolution import Gaussian2DKernel as gk2d
from astropy.convolution import interpolate_replace_nans
from ephysiopy.common.fieldproperties import FieldProps, RunProps
from ephysiopy.common.utils import (
    BinnedData,
    MapType,
    VariableToBin,
    blur_image,
    bwperim,
)
from ephysiopy.visualise.plotting import _stripAx


# Some functions to extract and filter runs from field properties


def sort_fields_by_attr(field_props: list[FieldProps], attr="area", reverse=True):
    """
    Sorts the fields in the list by attribute

    Notes
    -----
    In the default case will sort by area, largest first
    """
    fp = sorted(field_props, key=lambda x: getattr(x, attr), reverse=reverse)
    return fp


def get_all_phase(field_props: list[FieldProps]) -> np.ndarray:
    """
    Get all the phases from the field properties

    Parameters
    ----------
    field_props : list of FieldProps
        The field properties to search through

    Returns
    -------
    np.ndarray
        An array of all the phases from all runs in all fields
    """
    phases = []
    for field in field_props:
        phases.extend(field.compressed_phase)
    return np.array(phases)


def get_run_times(field_props: list[FieldProps]) -> list:
    """
    Get the run start and stop times in seconds for all runs
    through all fields in the field_props list
    """
    run_times = []
    for field in field_props:
        for run in field.runs:
            sample_rate = run.sample_rate
            run_times.append(
                (run.slice.start / sample_rate, run.slice.stop / sample_rate)
            )
    return run_times


def get_run(field_props: list[FieldProps], run_num: int) -> RunProps:
    """
    Get a specific run from the field properties

    Parameters
    ----------
    field_props : list of FieldProps
        The field properties to search through
    run_num : int
        The run number to search for

    Returns
    -------
    RunProps
        The run properties for the specified run number
    """
    for field in field_props:
        for run in field.runs:
            if run.label == run_num:
                return run
    raise ValueError(f"Run {run_num} not found in field properties.")


def filter_runs(
    field_props: list[FieldProps],
    attributes,
    ops,
    vals,
    **kwargs,
) -> list[FieldProps]:
    """
    Filter out runs that are too short, too slow or have too few spikes

    Parameters
    ----------
    field_props : list of FieldProps
    attributes : list of str
        attributes of RunProps to filter on
    ops : list of str
        operations to use for filtering. Supported operations are
        np.less and np.greater
    vals : list of float
        values to filter on
    Returns
    -------
    list of FieldProps

    Notes
    -----
    this modifies the input list

    Example
    -------
    >> field_props = filter_runs(field_props, ['n_spikes'], [np.greater], [5])

    field_props now only contains runs with more than 5 spikes
    """
    assert [hasattr(RunProps, attr) for attr in attributes]

    for field in field_props:
        for attr, op, val in zip(attributes, ops, vals):
            # parse the op string...
            s = str(op).split("'")[1]
            print(f"Filtering runs for {attr} {s} than {val}...")

            field.runs = list(
                filter(lambda run: op(getattr(run, attr), val), field.runs)
            )

    return field_props


def filter_for_speed(
    field_props: list[FieldProps], min_speed: float
) -> list[FieldProps]:
    """
    Mask for low speeds across the list of fields / runs

    Parameters
    ----------
    field_props : list of FieldProps
        The field properties to filter
    min_speed : float
        The minimum speed to keep a run

    Returns
    -------
    list of FieldProps
        The filtered field properties
    """
    print(f"Masking data with speeds less than {min_speed} cm/s...")

    for field in field_props:
        for run in field.runs:
            orig_mask = run.mask
            # note logical inversion as we want to mask for speeds below
            # the min speed
            speed_mask = run.speed <= min_speed
            run.mask = np.logical_or(speed_mask, orig_mask)

    return field_props


def infill_ratemap(rmap: np.ndarray) -> np.ndarray:
    """
    The ratemaps used in the phasePrecession2D class are a) super smoothed and
    b) very large i.e. the bins per cm is low. This
    results in firing fields that have lots of holes (nans) in them. We want to
    smooth over these holes so we can construct measures such as the expected
    rate in a given bin whilst also preserving whatever 'geometry' of the
    environment exists in the ratemap as a result of where position has been
    sampled. That is, if non-sampled positions are designated with nans, we
    want to smooth over those that in theory could have been sampled and keep
    those that never could have been.

    Parameters
    ----------
    rmap : np.ndarray
        The ratemap to be filled

    Returns
    -------
    np.ndarray
        The filled ratemap
    """
    outline = np.isfinite(rmap)
    mask = ndi.binary_fill_holes(outline)
    rmap = np.ma.MaskedArray(rmap, np.invert(mask))
    rmap[np.invert(mask)] = 0
    if rmap.ndim == 1:
        k = gk1d(stddev=1)
    elif rmap.ndim == 2:
        k = gk2d(x_stddev=1)
    output = interpolate_replace_nans(rmap, k)
    output[np.invert(mask)] = np.nan
    return output


def get_peak_coords(rmap, labels):
    """
    Get the peak coordinates of the firing fields in the ratemap
    """
    fieldId, _ = np.unique(labels, return_index=True)
    if np.any(labels == 0):
        fieldId = fieldId[1::]
    else:
        fieldId = fieldId[0::]
    peakCoords = np.array(
        ndi.maximum_position(rmap, labels=labels, index=fieldId)
    ).astype(int)
    return peakCoords


def simple_partition(
    binned_data: BinnedData, rate_threshold_prc: int = 200, **kwargs
) -> tuple[np.ndarray]:
    """
    Simple partitioning of fields based on mean firing rate. Only
    returns a single field (the highest firing rate field) per
    binned_data instance

    The default is to limit to fields that have a mean firing rate
    greater than twice the mean firing rate of the entire
    ratemap

    Parameters
    ----------
    binned_data : BinnedData
        an instance of ephysiopy.common.utils.BinnedData
    rate_threshold_prc : int
        removes pixels in a field that fall below this percent of
        the mean firing rate

    Returns
    -------
    tuple of np.ndarray
        peaksXY - The xy coordinates of the peak rates in
        the highest firing field
        peaksRate - The peak rates in peaksXY
        labels - An array of the labels corresponding to the highest firing field
        rmap_filled - The filled ratemap of the tetrode / cluster

    Notes
    -----
    This is a simple method to partition fields that only returns
    a single field - the one with the highest mean firing rate.
    """
    rmap = np.atleast_2d(binned_data.binned_data[0])
    mean_rate = np.nanmean(rmap)
    rate_threshold = mean_rate * (rate_threshold_prc / 100)
    above_thresh = rmap >= rate_threshold
    labels, n_fields = ndi.label(above_thresh)
    labels = np.atleast_2d(labels)
    if n_fields == 0:
        return None, None, None, None
    field_props = skimage.measure.regionprops(labels, intensity_image=rmap)
    # get the field with the highest mean firing rate
    mean_rates = [fp.intensity_mean for fp in field_props]
    max_field_idx = np.argmax(mean_rates)
    max_field = field_props[max_field_idx]
    # create output arrays
    peaksXY = np.array(
        [[max_field.centroid_weighted[1]], [max_field.centroid_weighted[0]]]
    )
    peaksRate = np.array([max_field.intensity_max])
    output_labels = np.zeros_like(labels)
    output_labels[max_field.coords[:, 0], max_field.coords[:, 1]] = 1
    rmap_filled = infill_ratemap(rmap)

    if binned_data.binned_data[0].ndim == 1:
        peaksXY = np.ravel(peaksXY)
        peaksRate = np.ravel(peaksRate)
        output_labels = np.ravel(output_labels)
        rmap_filled = np.ravel(rmap_filled)

    return peaksXY, peaksRate, output_labels, rmap_filled


def fancy_partition(
    binned_data: BinnedData,
    field_threshold_percent: int | float = 50,
    area_threshold_percent: float = 10,
) -> tuple[np.ndarray, ...]:
    """
    Another partitioning method

    Parameters
    ----------
    binned_data - BinnedData

    field_threshold_percent - int | float
        pixels below this are set to zero and ignored

    area_threshold_percent - float
        the expected minimum size of a receptive field
    """
    rmap_filled = infill_ratemap(binned_data.binned_data[0])

    # Only label pixels above field_threshold_percent
    mn = np.nanmean(rmap_filled)
    threshold = mn * (field_threshold_percent / 100)
    rmap_to_label = rmap_filled > threshold

    # return here if nothing is above the threshold
    if not np.any(rmap_to_label):
        return None, None, None, None

    labels = skimage.measure.label(rmap_to_label, background=0)

    mean_env_len = np.mean([np.mean(e) for e in binned_data.bin_edges])
    min_size = mean_env_len * (area_threshold_percent / 100)

    block_size = int(min_size) if int(min_size) % 2 == 1 else int(min_size) + 1

    # Locally threshold the ratemap with a gaussian and a block
    # size approximately equal to the minimum expected size of a
    # receptive field. This is to get the markers for the watershed algorithm.
    sm = skifilters.threshold_local(
        rmap_filled, block_size=block_size, method="gaussian", param=3
    )

    sm[rmap_to_label < threshold] = 0

    peakCoords = get_peak_coords(rmap_filled, labels)

    if len(binned_data.bin_edges) == 1:
        # if there is only one bin edge then we are dealing with a 1D ratemap
        be = binned_data.bin_edges[0]
        peaksXY = be[peakCoords[:, 0]]
        peakRates = rmap_filled[peakCoords[:, 0]]
        peakLabels = labels[peakCoords[:, 0]]
        peaksXY = peaksXY[peakLabels - 1]
        peaksRate = peakRates[peakLabels - 1]
        return peaksXY, peaksRate, labels, rmap_filled

    else:
        # if there are two bin edges then we are dealing with a 2D ratemap
        ye, xe = binned_data.bin_edges
        peaksXY = np.vstack((xe[peakCoords[:, 1]], ye[peakCoords[:, 0]]))
        peakRates = rmap_filled[peakCoords[:, 0], peakCoords[:, 1]]
        peakLabels = labels[peakCoords[:, 0], peakCoords[:, 1]]
        peaksXY = peaksXY[:, peakLabels - 1]
        peaksRate = peakRates[peakLabels - 1]
        return peaksXY, peaksRate, labels, rmap_filled


"""
These methods differ from MapCalcsGeneric in that they are mostly
concerned with treating rate maps as images as opposed to using
the spiking information contained within them. They therefore mostly
deals with spatial rate maps of place and grid cells.
"""


def get_mean_resultant(ego_boundary_map: np.ndarray) -> np.complex128 | float:
    """
    Calculates the mean resultant vector of a boundary map in egocentric coordinates

    Parameters
    ----------
    ego_boundary_map : np.ndarray
        The egocentric boundary map

    Returns
    -------
    float
        The mean resultant vector of the egocentric boundary map

    Notes
    -----
    See Hinman et al., 2019 for more details

    """
    if np.nansum(ego_boundary_map) == 0:
        return np.nan
    m, n = ego_boundary_map.shape
    angles = np.linspace(0, 2 * np.pi, n)
    MR = np.nansum(np.nansum(ego_boundary_map, 0) * np.power(np.e, angles * 1j)) / (
        n * m
    )
    return MR


def get_mean_resultant_length(ego_boundary_map: np.ndarray, **kwargs) -> float:
    """
    Calculates the length of the mean resultant vector of a
    boundary map in egocentric coordinates

    Parameters
    ----------
    ego_boundary_map : np.ndarray
        The egocentric boundary map

    Returns
    -------
    float
        The length of the mean resultant vector of the egocentric boundary map

    Notes
    -----
    See Hinman et al., 2019 for more details

    """
    MR = get_mean_resultant(ego_boundary_map, **kwargs)
    return np.abs(MR)


def get_mean_resultant_angle(ego_boundary_map: np.ndarray, **kwargs) -> float:
    """
    Calculates the angle of the mean resultant vector of a
    boundary map in egocentric coordinates

    Parameters
    ----------
    ego_boundary_map : np.ndarray
        The egocentric boundary map

    Returns
    -------
    float
        The angle mean resultant vector of the egocentric boundary map

    Notes
    -----
    See Hinman et al., 2019 for more details

    """
    MR = get_mean_resultant(ego_boundary_map, **kwargs)
    return np.rad2deg(np.arctan2(np.imag(MR), np.real(MR)))


def field_lims(A):
    """
    Returns a labelled matrix of the ratemap A.
    Uses anything greater than the half peak rate to select as a field.
    Data is heavily smoothed.

    Parameters
    ----------
    A : BinnedData
        A BinnedData instance containing the ratemap

    Returns
    -------
    np.ndarray
        The labelled ratemap
    """
    Ac = A.binned_data[0]
    nan_idx = np.isnan(Ac)
    Ac[nan_idx] = 0
    h = int(np.max(Ac.shape) / 2)
    sm_rmap = blur_image(A, h, ftype="gaussian").binned_data[0]
    thresh = np.max(sm_rmap.ravel()) * 0.2  # select area > 20% of peak
    distance = ndi.distance_transform_edt(sm_rmap > thresh)
    peak_idx = skimage.feature.peak_local_max(
        distance, exclude_border=False, labels=sm_rmap > thresh
    )
    mask = np.zeros_like(distance, dtype=bool)
    mask[tuple(peak_idx.T)] = True
    label = ndi.label(mask)[0]
    w = watershed(image=-distance, markers=label, mask=sm_rmap > thresh)
    label = ndi.label(w)[0]
    return label


def limit_to_one(A, prc=50, min_dist=5):
    """
    Processes a multi-peaked ratemap and returns a matrix
    where the multi-peaked ratemap consist of a single peaked field that is
    a) not connected to the border and b) close to the middle of the
    ratemap

    Parameters
    ----------
    A : np.ndarray
        The ratemap
    prc : int
        The percentage of the peak rate to threshold the ratemap at
    min_dist : int
        The minimum distance between peaks

    Returns
    -------
    tuple
        RegionProperties of the fields (list of RegionProperties)
        The single peaked ratemap (np.ndarray)
        The index of the field (int)

    """
    Ac = A.copy()
    Ac[np.isnan(A)] = 0
    # smooth Ac more to remove local irregularities
    n = ny = 5
    x, y = np.mgrid[-n : n + 1, -ny : ny + 1]
    g = np.exp(-(x**2 / float(n) + y**2 / float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode="same")
    # remove really small values
    Ac[Ac < 1e-10] = 0
    Ac_r = skimage.exposure.rescale_intensity(
        Ac, in_range="image", out_range=(0, 1000)
    ).astype(np.int32)
    peak_idx = skimage.feature.peak_local_max(
        Ac_r, min_distance=min_dist, exclude_border=False
    )
    peak_mask = np.zeros_like(Ac, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True
    peak_labels = skimage.measure.label(peak_mask, connectivity=2)
    field_labels = watershed(image=Ac * -1, markers=peak_labels)
    nFields = np.max(field_labels)
    sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
    labelled_sub_field_mask = np.zeros_like(sub_field_mask)
    sub_field_props = skimage.measure.regionprops(field_labels, intensity_image=Ac)
    sub_field_centroids = []
    sub_field_size = []

    for sub_field in sub_field_props:
        tmp = np.zeros(Ac.shape).astype(bool)
        tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
        tmp2 = Ac > sub_field.max_intensity * (prc / float(100))
        sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
        labelled_sub_field_mask[sub_field.label - 1, np.logical_and(tmp2, tmp)] = (
            sub_field.label
        )
        sub_field_centroids.append(sub_field.centroid)
        sub_field_size.append(sub_field.area)  # in bins
    sub_field_mask = np.sum(sub_field_mask, 0)
    middle = np.round(np.array(A.shape) / 2)
    normd_dists = sub_field_centroids - middle
    field_dists_from_middle = np.hypot(normd_dists[:, 0], normd_dists[:, 1])
    central_field_idx = np.argmin(field_dists_from_middle)
    central_field = np.squeeze(labelled_sub_field_mask[central_field_idx, :, :])
    # collapse the labelled mask down to an 2d array
    labelled_sub_field_mask = np.sum(labelled_sub_field_mask, 0)
    # clear the border
    cleared_mask = skimage.segmentation.clear_border(central_field)
    # check we've still got stuff in the matrix or fail
    if ~np.any(cleared_mask):
        print("No fields were detected away from edges so nothing returned")
        return None, None, None
    else:
        central_field_props = sub_field_props[central_field_idx]
    return central_field_props, central_field, central_field_idx


def border_score(
    A,
    B=None,
    shape="square",
    fieldThresh=0.3,
    circumPrc=0.2,
    binSize=3.0,
    minArea=200,
):
    """

    Calculates a border score totally dis-similar to that calculated in
    Solstad et al (2008)

    Parameters
    ----------
    A : np.ndarray
        the ratemap
    B : np.ndarray, default None
        This should be a boolean mask where True (1)
        is equivalent to the presence of a border and False (0)
        is equivalent to 'open space'. Naievely this will be the
        edges of the ratemap but could be used to take account of
        boundary insertions/ creations to check tuning to multiple
        environmental boundaries. Default None: when the mask is
        None then a mask is created that has 1's at the edges of the
        ratemap i.e. it is assumed that occupancy = environmental
        shape
    shape : str, default 'square'
        description of environment shape. Currently
        only 'square' or 'circle' accepted. Used to calculate the
        proportion of the environmental boundaries to examine for
        firing
    fieldThresh : float, default 0.3
        Between 0 and 1 this is the percentage
        amount of the maximum firing rate
        to remove from the ratemap (i.e. to remove noise)
    circumPrc : float, default 0.2
        The percentage amount of the circumference
        of the environment that the field needs to be to count
        as long enough to make it through
    binSize : float, default 3.0
        bin size in cm
    minArea : float, default 200
        min area for a field to be considered

    Returns
    -------
    float
        the border score

    Notes
    -----
    If the cell is a border cell (BVC) then we know that it should
    fire at a fixed distance from a given boundary (possibly more
    than one). In essence this algorithm estimates the amount of
    variance in this distance i.e. if the cell is a border cell this
    number should be small. This is achieved by first doing a bunch of
    morphological operations to isolate individual fields in the
    ratemap (similar to the code used in phasePrecession.py - see
    the fancy_partition method therein). These partitioned fields are then
    thinned out (using skimage's skeletonize) to a single pixel
    wide field which will lie more or less in the middle of the
    (highly smoothed) sub-field. It is the variance in distance from the
    nearest boundary along this pseudo-iso-line that is the boundary
    measure

    Other things to note are that the pixel-wide field has to have some
    minimum length. In the case of a circular environment this is set to
    20% of the circumference; in the case of a square environment markers
    this is at least half the length of the longest side
    """
    # need to know borders of the environment so we can see if a field
    # touches the edges, and the perimeter length of the environment
    # deal with square or circles differently
    borderMask = np.zeros_like(A)
    A_rows, A_cols = np.shape(A)
    if "circle" in shape:
        radius = np.max(np.array(np.shape(A))) / 2.0
        dist_mask = skimage.morphology.disk(radius)
        if np.shape(dist_mask) > np.shape(A):
            dist_mask = dist_mask[1 : A_rows + 1, 1 : A_cols + 1]
        tmp = np.zeros([A_rows + 2, A_cols + 2])
        tmp[1:-1, 1:-1] = dist_mask
        dists = ndi.distance_transform_bf(tmp)
        dists = dists[1:-1, 1:-1]
        borderMask = np.logical_xor(dists <= 0, dists < 2)
        # open up the border mask a little
        borderMask = skimage.morphology.binary_dilation(
            borderMask, skimage.morphology.disk(1)
        )
    elif "square" in shape:
        borderMask[0:3, :] = 1
        borderMask[-3:, :] = 1
        borderMask[:, 0:3] = 1
        borderMask[:, -3:] = 1
        tmp = np.zeros([A_rows + 2, A_cols + 2])
        dist_mask = np.ones_like(A)
        tmp[1:-1, 1:-1] = dist_mask
        dists = ndi.distance_transform_bf(tmp)
        # remove edges to make same shape as input ratemap
        dists = dists[1:-1, 1:-1]
    A[~np.isfinite(A)] = 0
    # get some morphological info about the fields in the ratemap
    # start image processing:
    # get some markers
    # NB I've tried a variety of techniques to optimise this part and the
    # best seems to be the local adaptive thresholding technique which)
    # smooths locally with a gaussian - see the skimage docs for more
    idx = A >= np.nanmax(np.ravel(A)) * fieldThresh
    A_thresh = np.zeros_like(A)
    A_thresh[idx] = A[idx]

    # label these markers so each blob has a unique id
    labels, nFields = ndi.label(A_thresh)
    # remove small objects
    min_size = int(minArea / binSize) - 1
    skimage.morphology.remove_small_objects(labels, min_size=min_size, connectivity=2)
    labels = skimage.segmentation.relabel_sequential(labels)[0]
    nFields = np.nanmax(labels)
    if nFields == 0:
        return np.nan

    # Iterate over the labelled parts of the array labels calculating
    # how much of the total circumference of the environment edge it
    # covers

    fieldAngularCoverage = np.zeros([1, nFields]) * np.nan
    fractionOfPixelsOnBorder = np.zeros([1, nFields]) * np.nan
    fieldsToKeep = np.zeros_like(A).astype(bool)
    for i in range(1, nFields + 1):
        fieldMask = np.logical_and(labels == i, borderMask)

        # check the angle subtended by the fieldMask
        if np.nansum(fieldMask.astype(int)) > 0:
            s = skimage.measure.regionprops(
                fieldMask.astype(int), intensity_image=A_thresh
            )[0]
            x = s.coords[:, 0] - (A_cols / 2.0)
            y = s.coords[:, 1] - (A_rows / 2.0)
            subtended_angle = np.rad2deg(np.ptp(np.arctan2(x, y)))
            if subtended_angle > (360 * circumPrc):
                pixelsOnBorder = np.count_nonzero(fieldMask) / float(
                    np.count_nonzero(labels == i)
                )
                fractionOfPixelsOnBorder[:, i - 1] = pixelsOnBorder
                if pixelsOnBorder > 0.5:
                    fieldAngularCoverage[0, i - 1] = subtended_angle

            fieldsToKeep = np.logical_or(fieldsToKeep, labels == i)

    # Check the fields are big enough to qualify (minArea)
    # returning nan if not
    def fn(val):
        return np.count_nonzero(val)

    field_sizes = ndi.labeled_comprehension(
        A, labels, range(1, nFields + 1), fn, float, 0
    )
    field_sizes /= binSize
    if not np.any(field_sizes) > (minArea / binSize):
        warnings.warn("No fields bigger than the minimum size found")
        return np.nan

    fieldAngularCoverage = fieldAngularCoverage / 360.0
    rateInField = A[fieldsToKeep]
    # normalize firing rate in the field to sum to 1
    rateInField = rateInField / np.nansum(rateInField)
    dist2WallInField = dists[fieldsToKeep]
    Dm = np.dot(dist2WallInField, rateInField)
    if "circle" in shape:
        Dm = Dm / radius
    elif "square" in shape:
        Dm = Dm / (np.nanmax(np.shape(A)) / 2.0)
    borderScore = (fractionOfPixelsOnBorder - Dm) / (fractionOfPixelsOnBorder + Dm)
    return np.nanmax(borderScore)


def _get_field_labels(A: np.ndarray, **kwargs) -> tuple:
    """
    Returns a labeled version of A after finding the peaks
    in A and finding the watershed basins from the markers
    found from those peaks. Used in field_props() and
    grid_field_props()

    Parameters
    ----------
    A : np.ndarray
        The array to process
    **kwargs
        min_distance (float, optional): The distance in bins between fields to
        separate the regions of the image
        clear_border (bool, optional): Input to skimage.feature.peak_local_max.
        The number of pixels to ignore at the edge of the image
    """
    clear_border = True
    if "clear_border" in kwargs:
        clear_border = kwargs.pop("clear_border")

    min_distance = 1
    if "min_distance" in kwargs:
        min_distance = kwargs.pop("min_distance")

    A[~np.isfinite(A)] = -1
    A[A < 0] = -1
    Ac_r = skimage.exposure.rescale_intensity(
        A, in_range="image", out_range=(0, 1000)
    ).astype(np.int32)
    peak_coords = skimage.feature.peak_local_max(
        Ac_r, min_distance=min_distance, exclude_border=clear_border
    )
    peaksMask = np.zeros_like(A, dtype=bool)
    peaksMask[tuple(peak_coords.T)] = True
    peaksLabel, _ = ndi.label(peaksMask)
    ws = watershed(image=-1 * A, markers=peaksLabel)
    return peak_coords, ws


def plot_field_props(field_props: list[FieldProps]):
    """
    Plots the fields in the list of FieldProps

    Parameters
    ----------
    list of FieldProps
    """
    fig = plt.figure()
    subfigs = fig.subfigures(
        2,
        2,
    )
    ax = subfigs[0, 0].subplots(1, 1)
    # ax = fig.add_subplot(221)
    fig.canvas.manager.set_window_title("Field partitioning and runs")
    outline = np.isfinite(field_props[0]._intensity_image)
    outline = ndimage.binary_fill_holes(outline)
    outline = np.ma.masked_where(np.invert(outline), outline)
    outline_perim = bwperim(outline)
    outline_idx = np.nonzero(outline_perim)
    bin_edges = field_props[0].binned_data.bin_edges
    outline_xy = bin_edges[1][outline_idx[1]], bin_edges[0][outline_idx[0]]
    ax.plot(outline_xy[0], outline_xy[1], "k.", ms=1)
    # PLOT 1
    cmap_arena = matplotlib.colormaps["tab20c_r"].resampled(1)
    ax.pcolormesh(bin_edges[1], bin_edges[0], outline_perim, cmap=cmap_arena)
    # Runs through fields in global x-y coordinates
    max_field_label = np.max([f.label for f in field_props])
    cmap = matplotlib.colormaps["Set1"].resampled(max_field_label)
    [
        [
            ax.plot(r.xy[0], r.xy[1], color=cmap(f.label - 1), label=f.label - 1)
            for r in f.runs
        ]
        for f in field_props
    ]
    # plot the perimeters of the field(s)
    [
        ax.plot(
            f.global_perimeter_coords[0],
            f.global_perimeter_coords[1],
            "k.",
            ms=1,
        )
        for f in field_props
    ]
    [ax.plot(f.xy_at_peak[0], f.xy_at_peak[1], "ko", ms=2) for f in field_props]
    norm = matplotlib.colors.Normalize(1, max_field_label)
    tick_locs = np.linspace(1.5, max_field_label - 0.5, max_field_label)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        ticks=tick_locs,
    )
    cbar.set_ticklabels(list(map(str, [f.label for f in field_props])))
    # ratemaps are plotted with origin in top left so invert y axis
    ax.invert_yaxis()
    ax.set_aspect("equal")
    _stripAx(ax)
    # PLOT 2
    # Runs on the unit circle on a per field basis as it's too confusing to
    # look at all of them on a single unit circle
    n_rows = 2
    n_cols = np.ceil(len(field_props) / n_rows).astype(int)

    ax1 = np.ravel(subfigs[0, 1].subplots(n_rows, n_cols))
    [
        ax1[f.label - 1].plot(
            f.pos_xy[0],
            f.pos_xy[1],
            color=cmap(f.label - 1),
            lw=0.5,
            zorder=1,
        )
        for f in field_props
    ]
    [
        a.add_artist(
            matplotlib.patches.Circle((0, 0), 1, fc="none", ec="lightgrey", zorder=3),
        )
        for a, _ in zip(ax1, field_props)
    ]
    [a.set_xlim(-1, 1) for a in ax1]
    [a.set_ylim(-1, 1) for a in ax1]
    [a.set_title(f.label) for a, f in zip(ax1, field_props)]
    [a.set_aspect("equal") for a in ax1]
    [_stripAx(a) for a in ax1]

    # PLOT 3
    # The runs through the fields coloured by the distance of each xy coord in
    # the field to the peak and angle of each point on the perimeter to
    # the peak
    dist_cmap = matplotlib.colormaps["jet_r"]
    angular_cmap = matplotlib.colormaps["hsv"]
    im = np.zeros_like(field_props[0]._intensity_image).astype(int) * np.nan
    for f in field_props:
        sub_im = f.image * np.nan
        idx = np.nonzero(f.bw_perim)
        # the angles made by the perimeter to the field peak
        sub_im[idx[0], idx[1]] = f.perimeter_angle_from_peak
        im[f.slice] = sub_im
    ax2 = subfigs[1, 0].subplots(1, 1)
    # distances as collections of Rectangles
    distances = np.concatenate(
        [f.xy_dist_to_peak / f.xy_dist_to_peak.max() for f in field_props]
    )
    face_colours = dist_cmap(distances)
    offsets = np.concatenate([f.xy.T for f in field_props])
    rects = RegularPolyCollection(
        numsides=4,
        rotation=0,
        facecolors=face_colours,
        edgecolors=face_colours,
        offsets=offsets,
        offset_transform=ax2.transData,
    )
    ax2.add_collection(rects)
    ax2.pcolormesh(bin_edges[1], bin_edges[0], im, cmap=angular_cmap)
    _stripAx(ax2)

    ax2.invert_yaxis()
    ax2.set_aspect("equal")
    degs_norm = matplotlib.colors.Normalize(0, 360)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=angular_cmap, norm=degs_norm),
        ax=ax2,
    )
    [ax2.plot(f.xy_at_peak[0], f.xy_at_peak[1], "ko", ms=2) for f in field_props]
    # PLOT 4
    # The smoothed ratemap - maybe make this the first sub plot
    ax3 = subfigs[1, 1].subplots(1, 1)
    # smooth the ratemap a bunch
    rmap_to_plot = copy.copy(field_props[0]._intensity_image)
    rmap_to_plot = infill_ratemap(rmap_to_plot)
    ax3.pcolormesh(bin_edges[1], bin_edges[0], rmap_to_plot)
    # add the field labels to the ratemap plot
    [
        ax3.text(f.xy_at_peak[0], f.xy_at_peak[1], str(f.label), ha="left", va="bottom")
        for f in field_props
    ]

    ax3.invert_yaxis()
    ax3.set_aspect("equal")
    _stripAx(ax3)


# TODO: This needs moving and /or renaming due to conflicts with fieldproperties stuff
def field_props(
    A,
    min_dist=5,
    neighbours=2,
    prc=50,
    plot=False,
    ax=None,
    tri=False,
    verbose=True,
    **kwargs,
):
    """
    Returns a dictionary of properties of the field(s) in a ratemap A

    Args:
        A (array_like): a ratemap (but could be any image)
        min_dist (float): the separation (in bins) between fields for measures
            such as field distance to make sense. Used to
            partition the image into separate fields in the call to
            feature.peak_local_max
        neighbours (int): the number of fields to consider as neighbours to
            any given field. Defaults to 2
        prc (float): percent of fields to consider
        ax (matplotlib.Axes): user supplied axis. If None a new figure window
        is created
        tri (bool): whether to do Delaunay triangulation between fields
            and add to plot
        verbose (bool): dumps the properties to the console
        plot (bool): whether to plot some output - currently consists of the
            ratemap A, the fields of which are outline in a black
            contour. Default False

    Returns:
        result (dict): The properties of the field(s) in the input ratemap A
    """

    from skimage.measure import find_contours
    from sklearn.neighbors import NearestNeighbors

    nan_idx = np.isnan(A)
    Ac = A.copy()
    Ac[np.isnan(A)] = 0
    # smooth Ac more to remove local irregularities
    n = ny = 5
    x, y = np.mgrid[-n : n + 1, -ny : ny + 1]
    g = np.exp(-(x**2 / float(n) + y**2 / float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode="same")

    peak_idx, field_labels = _get_field_labels(Ac, **kwargs)

    nFields = np.max(field_labels)
    if neighbours > nFields:
        print(
            "neighbours value of {0} > the {1} peaks found".format(neighbours, nFields)
        )
        print("Reducing neighbours to number of peaks found")
        neighbours = nFields
    sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
    sub_field_props = skimage.measure.regionprops(field_labels, intensity_image=Ac)
    sub_field_centroids = []
    sub_field_size = []

    for sub_field in sub_field_props:
        tmp = np.zeros(Ac.shape).astype(bool)
        tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
        tmp2 = Ac > sub_field.max_intensity * (prc / float(100))
        sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
        sub_field_centroids.append(sub_field.centroid)
        sub_field_size.append(sub_field.area)  # in bins
    sub_field_mask = np.sum(sub_field_mask, 0)
    contours = skimage.measure.find_contours(sub_field_mask, 0.5)
    # find the nearest neighbors to the peaks of each sub-field
    nbrs = NearestNeighbors(n_neighbors=neighbours, algorithm="ball_tree").fit(peak_idx)
    distances, _ = nbrs.kneighbors(peak_idx)
    mean_field_distance = np.mean(distances[:, 1:neighbours])

    nValid_bins = np.sum(~nan_idx)
    # calculate the amount of out of field firing
    A_non_field = np.zeros_like(A) * np.nan
    A_non_field[~sub_field_mask.astype(bool)] = A[~sub_field_mask.astype(bool)]
    A_non_field[nan_idx] = np.nan
    out_of_field_firing_prc = (
        np.count_nonzero(A_non_field > 0) / float(nValid_bins)
    ) * 100
    Ac[np.isnan(A)] = np.nan
    # get some stats about the field ellipticity
    ellipse_ratio = np.nan
    _, central_field, _ = limit_to_one(A, prc=50)

    contour_coords = find_contours(central_field, 0.5)
    from skimage.measure import EllipseModel

    E = EllipseModel()
    E.estimate(contour_coords[0])
    ellipse_axes = E.params[2:4]
    ellipse_ratio = np.min(ellipse_axes) / np.max(ellipse_axes)

    """ using the peak_idx values calculate the angles of the triangles that
    make up a delaunay tesselation of the space if the calc_angles arg is
    in kwargs
    """
    if "calc_angs" in kwargs.keys():
        angs = calc_angs(peak_idx)
    else:
        angs = None

    props = {
        "Ac": Ac,
        "Peak_rate": np.nanmax(A),
        "Mean_rate": np.nanmean(A),
        "Field_size": np.mean(sub_field_size),
        "Pct_bins_with_firing": (np.sum(sub_field_mask) / nValid_bins) * 100,
        "Out_of_field_firing_prc": out_of_field_firing_prc,
        "Dist_between_fields": mean_field_distance,
        "Num_fields": float(nFields),
        "Sub_field_mask": sub_field_mask,
        "Smoothed_map": Ac,
        "field_labels": field_labels,
        "Peak_idx": peak_idx,
        "angles": angs,
        "contours": contours,
        "ellipse_ratio": ellipse_ratio,
    }

    if verbose:
        print(
            "\nPercentage of bins with firing: {:.2%}".format(
                np.sum(sub_field_mask) / nValid_bins
            )
        )
        print(
            "Percentage out of field firing: {:.2%}".format(
                np.count_nonzero(A_non_field > 0) / float(nValid_bins)
            )
        )
        print(f"Peak firing rate: {np.nanmax(A)} Hz")
        print(f"Mean firing rate: {np.nanmean(A)} Hz")
        print(f"Number of fields: {nFields}")
        print(f"Mean field size: {np.mean(sub_field_size)} cm")
        print(f"Mean inter-peak distance between fields: {mean_field_distance} cm")
    return props


def calc_angs(points):
    """
    Calculates the angles for all triangles in a delaunay tesselation of
    the peak points in the ratemap
    """

    # calculate the lengths of the sides of the triangles
    tri = spatial.Delaunay(points)
    angs = []
    for s in tri.simplices:
        A = tri.points[s[1]] - tri.points[s[0]]
        B = tri.points[s[2]] - tri.points[s[1]]
        C = tri.points[s[0]] - tri.points[s[2]]
        for e1, e2 in ((A, -B), (B, -C), (C, -A)):
            num = np.dot(e1, e2)
            denom = np.linalg.norm(e1) * np.linalg.norm(e2)
            angs.append(np.arccos(num / denom) * 180 / np.pi)
    return np.array(angs).T


def kl_spatial_sparsity(pos_map: BinnedData) -> float:
    """
    Calculates the spatial sampling of an arena by comparing the
    observed spatial sampling to an expected uniform one using kl divergence

    Data in pos_map should be unsmoothed (not checked) and the MapType should
    be POS (checked)

    Parameters
    ----------
    pos_map : BinnedData
        The position map

    Returns
    -------
    float
        The spatial sparsity of the position map
    """
    assert pos_map.map_type == MapType.POS
    return kldiv_dir(np.ravel(pos_map.binned_data[0]))


def spatial_sparsity(rate_map: np.ndarray, pos_map: np.ndarray) -> float:
    """
    Calculates the spatial sparsity of a rate map as defined by
    Markus et al (1994)

    For example, a sparsity score of 0.10 indicates that the cell fired on
    10% of the maze surface

    Parameters
    ----------
    rate_map : np.ndarray
        The rate map
    pos_map : np.ndarray
        The occupancy map

    Returns
    -------
    float
        The spatial sparsity of the rate map

    References
    ----------
    Markus, E.J., Barnes, C.A., McNaughton, B.L., Gladden, V.L. &
    Skaggs, W.E. Spatial information content and reliability of
    hippocampal CA1 neurons: effects of visual input. Hippocampus
    4, 410–421 (1994).

    """
    p_i = pos_map / np.nansum(pos_map)
    sparsity = np.nansum(p_i * rate_map) ** 2 / np.nansum(p_i * rate_map**2)
    return sparsity


def coherence(smthd_rate, unsmthd_rate):
    """
    Calculates the coherence of receptive field via correlation of smoothed
    and unsmoothed ratemaps

    Parameters
    ----------
    smthd_rate : np.ndarray
        The smoothed rate map
    unsmthd_rate : np.ndarray
        The unsmoothed rate map

    Returns
    -------
    float
        The coherence of the rate maps
    """
    smthd = smthd_rate.ravel()
    unsmthd = unsmthd_rate.ravel()
    si = ~np.isnan(smthd)
    ui = ~np.isnan(unsmthd)
    idx = ~(~si | ~ui)
    coherence = np.corrcoef(unsmthd[idx], smthd[idx])
    return coherence[1, 0]


def kldiv_dir(polarPlot: np.ndarray) -> float:
    """
    Returns a kl divergence for directional firing: measure of directionality.
    Calculates kl diveregence between a smoothed ratemap (probably should be
    smoothed otherwise information theoretic measures
    don't 'care' about position of bins relative to one another) and a
    pure circular distribution.
    The larger the divergence the more tendancy the cell has to fire when the
    animal faces a specific direction.

    Parameters
    ----------
    polarPlot np.ndarray
        The binned and smoothed directional ratemap

    Returns
    -------
    float
        The divergence from circular of the 1D-array
        from a uniform circular distribution
    """

    __inc = 0.00001
    polarPlot = np.atleast_2d(polarPlot)
    polarPlot[np.isnan(polarPlot)] = __inc
    polarPlot[polarPlot == 0] = __inc
    normdPolar = polarPlot / float(np.nansum(polarPlot))
    nDirBins = polarPlot.shape[1]
    compCirc = np.ones_like(polarPlot) / float(nDirBins)
    X = np.arange(0, nDirBins)
    kldivergence = kldiv(np.atleast_2d(X), normdPolar, compCirc)
    return kldivergence


def kldiv(
    X: np.ndarray, pvect1: np.ndarray, pvect2: np.ndarray, variant: str = ""
) -> float:
    """
    Calculates the Kullback-Leibler or Jensen-Shannon divergence between
    two distributions.

    Parameters
    ----------
    X : np.ndarray
        Vector of M variable values
    P1, P2 : np.ndarray
        Length-M vectors of probabilities representing distribution 1 and 2
    variant : str, default 'sym'
        If 'sym', returns a symmetric variant of the
        Kullback-Leibler divergence, given by [KL(P1,P2)+KL(P2,P1)]/2
        If 'js', returns the Jensen-Shannon divergence, given by
        [KL(P1,Q)+KL(P2,Q)]/2, where Q = (P1+P2)/2

    Returns
    -------
    float
        The Kullback-Leibler divergence or Jensen-Shannon divergence

    Notes
    -----
    The Kullback-Leibler divergence is given by:

    .. math:: KL(P1(x),P2(x)) = sum_[P1(x).log(P1(x)/P2(x))]

    If X contains duplicate values, there will be an warning message,
    and these values will be treated as distinct values.  (I.e., the
    actual values do not enter into the computation, but the probabilities
    for the two duplicate values will be considered as probabilities
    corresponding to two unique values.).
    The elements of probability vectors P1 and P2 must
    each sum to 1 +/- .00001.

    This function is taken from one on the Mathworks file exchange

    See Also
    --------
    Cover, T.M. and J.A. Thomas. "Elements of Information Theory," Wiley,
    1991.

    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """

    if len(np.unique(X)) != len(np.sort(X)):
        warnings.warn(
            "X contains duplicate values. Treated as distinct values.", UserWarning
        )
    if (
        not np.equal(np.shape(X), np.shape(pvect1)).all()
        or not np.equal(np.shape(X), np.shape(pvect2)).all()
    ):
        raise ValueError("Inputs are not the same size")
    if (np.abs(np.sum(pvect1) - 1) > 0.00001) or (np.abs(np.sum(pvect2) - 1) > 0.00001):
        print(f"Probabilities sum to {np.abs(np.sum(pvect1))} for pvect1")
        print(f"Probabilities sum to {np.abs(np.sum(pvect2))} for pvect2")
        warnings.warn("Probabilities dont sum to 1.", UserWarning)
    if variant:
        if variant == "js":
            logqvect = np.log2((pvect2 + pvect1) / 2)
            KL = 0.5 * (
                np.nansum(pvect1 * (np.log2(pvect1) - logqvect))
                + np.sum(pvect2 * (np.log2(pvect2) - logqvect))
            )
            return float(KL)
        elif variant == "sym":
            KL1 = np.nansum(pvect1 * (np.log2(pvect1) - np.log2(pvect2)))
            KL2 = np.nansum(pvect2 * (np.log2(pvect2) - np.log2(pvect1)))
            KL = (KL1 + KL2) / 2
            return float(KL)
        else:
            warnings.warn("Last argument not recognised", UserWarning)
    KL = np.nansum(pvect1 * (np.log2(pvect1) - np.log2(pvect2)))
    return float(KL)


def skaggs_info(ratemap: np.ndarray, dwelltimes: np.ndarray, **kwargs):
    """
    Calculates Skaggs information measure

    Parameters
    ----------
    ratemap, dwelltimes :np.ndarray
        The binned up ratemap and dwelltimes. Must be the same size

    Returns
    -------
    float
        Skaggs information score in bits spike

    Notes
    -----
    The ratemap data should have undergone adaptive binning as per
    the original paper. See getAdaptiveMap() in binning class

    The estimate of spatial information in bits per spike:

    .. math:: I = sum_{x} p(x).r(x).log(r(x)/r)
    """
    sample_rate = kwargs.get("sample_rate", 50)

    dwelltimes = dwelltimes / sample_rate  # assumed sample rate of 50Hz
    if ratemap.ndim > 1:
        ratemap = np.reshape(ratemap, (np.prod(np.shape(ratemap)), 1))
        dwelltimes = np.reshape(dwelltimes, (np.prod(np.shape(dwelltimes)), 1))
    duration = np.nansum(dwelltimes)
    meanrate = np.nansum(ratemap * dwelltimes) / duration
    if meanrate <= 0.0:
        bits_per_spike = np.nan
        return bits_per_spike
    p_x = dwelltimes / duration
    p_r = ratemap / meanrate
    dum = p_x * ratemap
    ind = np.nonzero(dum)[0]
    bits_per_spike = np.nansum(dum[ind] * np.log2(p_r[ind]))
    bits_per_spike = bits_per_spike / meanrate
    return bits_per_spike


def grid_field_props(A: BinnedData, maxima="centroid", allProps=True, **kwargs):
    """
    Extracts various measures from a spatial autocorrelogram

    Parameters
    ----------
    A : BinnedData
        object containing the spatial autocorrelogram (SAC) in
            A.binned_data[0]
    maxima (str, optional): The method used to detect the peaks in the SAC.
            Legal values are 'single' and 'centroid'. Default 'centroid'
    allProps : bool default=True
        Whether to return a dictionary that contains the attempt to fit
        an ellipse around the edges of the central size peaks. See below

    Returns
    -------
    dict
        Measures of the SAC.
        Keys include:
            * gridness score
            * scale
            * orientation
            * coordinates of the peaks (nominally 6) closest to SAC centre
            * a binary mask around the extent of the 6 central fields
            * values of the rotation procedure used to calculate gridness
            * ellipse axes and angle (if allProps is True and the it worked)

    Notes
    -----
    The output from this method can be used as input to the show() method
    of this class.
    When it is the plot produced will display a lot more informative.
    The coordinate system internally used is centred on the image centre.

    See Also
    --------
    ephysiopy.common.binning.autoCorr2D()
    """
    """
    Assign the output dictionary now as we want to return immediately if
    the input is bad
    """
    dictKeys = (
        "gridscore",
        "scale",
        "orientation",
        "closest_peak_coords",
        "dist_to_centre",
        "ellipse_axes",
        "ellipse_angle",
        "ellipseXY",
        "circleXY",
        "rotationArr",
        "rotationCorrVals",
    )

    outDict = dict.fromkeys(dictKeys, np.nan)

    A_tmp = A.binned_data[0].copy()

    if np.all(np.isnan(A_tmp)):
        warnings.warn("No data in SAC - returning nans in measures dict")
        outDict["dist_to_centre"] = np.atleast_2d(np.array([0, 0]))
        outDict["scale"] = 0
        outDict["closest_peak_coords"] = np.atleast_2d(np.array([0, 0]))
        return outDict

    A_tmp[~np.isfinite(A_tmp)] = -1
    A_tmp[A_tmp <= 0] = -1
    A_sz = np.array(np.shape(A_tmp))
    # [STAGE 1] find peaks & identify 7 closest to centre
    min_distance = np.ceil(np.min(A_sz / 2) / 8.0).astype(int)
    min_distance = kwargs.get("min_distance", min_distance)

    _, _, field_labels, _ = fancy_partition(
        A, field_threshold_percent=10, field_rate_threshold=0.001
    )
    # peak_idx, field_labels = _get_field_labels(A_tmp, neighbours=7, **kwargs)
    # a fcn for the labeled_comprehension function that returns
    # linear indices in A where the values in A for each label are
    # greater than half the max in that labeled region

    def fn(val, pos):
        return pos[val > (np.max(val) / 2)]

    nLbls = np.max(field_labels)
    indices = ndi.labeled_comprehension(
        A_tmp, field_labels, np.arange(0, nLbls), fn, np.ndarray, 0, True
    )
    # turn linear indices into coordinates
    coords = [np.unravel_index(i, A_sz) for i in indices]
    half_peak_labels = np.zeros(shape=A_sz)
    for peak_id, coord in enumerate(coords):
        xc, yc = coord
        half_peak_labels[xc, yc] = peak_id

    # Get some statistics about the labeled regions
    lbl_range = np.arange(0, nLbls)
    peak_coords = ndi.maximum_position(A.binned_data[0], half_peak_labels, lbl_range)
    peak_coords = np.array(peak_coords)
    # Now convert the peak_coords to the image centre coordinate system
    x_peaks, y_peaks = peak_coords.T
    x_peaks_ij = A.bin_edges[0][x_peaks]
    y_peaks_ij = A.bin_edges[1][y_peaks]
    peak_coords = np.array([x_peaks_ij, y_peaks_ij]).T
    # Get some distance and morphology measures
    peak_dist_to_centre = np.hypot(peak_coords[:, 0], peak_coords[:, 1])
    closest_peak_idx = np.argsort(peak_dist_to_centre)
    central_peak_label = closest_peak_idx[0]
    closest_peak_idx = closest_peak_idx[1 : np.min((7, len(closest_peak_idx) - 1))]
    # closest_peak_idx should now the indices of the labeled 6 peaks
    # surrounding the central peak at the image centre
    scale = np.median(peak_dist_to_centre[closest_peak_idx])
    orientation = np.nan
    orientation = __grid_orientation(peak_coords, closest_peak_idx)

    xv, yv = np.meshgrid(A.bin_edges[0], A.bin_edges[1], indexing="ij")
    xv = xv[:-1, :-1]  # remove last row and column
    yv = yv[:-1:, :-1]  # remove last row and column
    dist_to_centre = np.hypot(xv, yv)
    # get the max distance of the half-peak width labeled fields
    # from the centre of the image
    max_dist_from_centre = 0
    for peak_id, _coords in enumerate(coords):
        if peak_id in closest_peak_idx:
            xc, yc = _coords
            if np.any(xc) and np.any(yc):
                xc = A.bin_edges[0][xc]
                yc = A.bin_edges[1][yc]
                d = np.max(np.hypot(xc, yc))
                if d > max_dist_from_centre:
                    max_dist_from_centre = d

    # Set the outer bits and the central region of the SAC to nans
    # getting ready for the correlation procedure
    dist_to_centre[np.abs(dist_to_centre) > max_dist_from_centre] = 0
    dist_to_centre[half_peak_labels == central_peak_label] = 0
    dist_to_centre[dist_to_centre != 0] = 1
    dist_to_centre = dist_to_centre.astype(bool)
    sac_middle = A.binned_data[0].copy()
    sac_middle[~dist_to_centre] = np.nan

    if "step" in kwargs.keys():
        step = kwargs.pop("step")
    else:
        step = 30
    try:
        gridscore, rotationCorrVals, rotationArr = gridness(sac_middle, step=step)
    except Exception:
        gridscore, rotationCorrVals, rotationArr = np.nan, np.nan, np.nan

    if allProps:
        # attempt to fit an ellipse around the outer edges of the nearest
        # peaks to the centre of the SAC. First find the outer edges for
        # the closest peaks using a ndimages labeled_comprehension
        try:

            def fn2(val, pos):
                xc, yc = np.unravel_index(pos, A_sz)
                xc = xc - np.floor(A_sz[0] / 2)
                yc = yc - np.floor(A_sz[1] / 2)
                idx = np.argmax(np.hypot(xc, yc))
                return xc[idx], yc[idx]

            ellipse_coords = ndi.labeled_comprehension(
                A.binned_data[0],
                half_peak_labels,
                closest_peak_idx,
                fn2,
                tuple,
                0,
                True,
            )

            ellipse_fit_coords = np.array([(x, y) for x, y in ellipse_coords])
            from skimage.measure import EllipseModel

            E = EllipseModel()
            E.estimate(ellipse_fit_coords)
            im_centre = E.params[0:2]
            ellipse_axes = E.params[2:4]
            ellipse_angle = E.params[-1]
            ellipseXY = E.predict_xy(np.linspace(0, 2 * np.pi, 50), E.params)

            # get the min containing circle given the eliipse minor axis
            from skimage.measure import CircleModel

            _params = [im_centre, np.min(ellipse_axes)]
            circleXY = CircleModel().predict_xy(
                np.linspace(0, 2 * np.pi, 50), params=_params
            )
        except (TypeError, ValueError):  # non-iterable x and y
            ellipse_axes = None
            ellipse_angle = (None, None)
            ellipseXY = None
            circleXY = None

    # collect all the following keywords into a dict for output
    closest_peak_coords = np.array(peak_coords)[closest_peak_idx]

    # Assign values to the output dictionary created at the start
    for thiskey in outDict.keys():
        outDict[thiskey] = locals()[thiskey]
        # neat trick: locals is a dict holding all locally scoped variables
    return outDict


def __grid_orientation(peakCoords, closestPeakIdx):
    """
    Calculates the orientation angle of a grid field.

    The orientation angle is the angle of the first peak working
    counter-clockwise from 3 o'clock

    Parameters
    ----------
    peakCoords : np.ndarray
        The peak coordinates as pairs of xy
    closestPeakIdx : np.ndarray
        A 1D array of the indices in peakCoords
        of the peaks closest to the centre of the SAC

    Returns
    -------
    float
        The first value in an array of the angles of
        the peaks in the SAC working counter-clockwise from a line
        extending from the middle of the SAC to 3 o'clock.
    """
    if len(peakCoords) < 3 or closestPeakIdx.size == 0:
        return np.nan
    else:
        from ephysiopy.common.utils import polar

        peaks = peakCoords[closestPeakIdx]
        theta = polar(peaks[:, 1], -peaks[:, 0], deg=1)[1]
        return np.sort(theta.compress(theta >= 0))[0]


def gridness(image, step=30) -> tuple:
    """
    Calculates the gridness score in a grid cell SAC.

    The data in `image` is rotated in `step` amounts and
    each rotated array is correlated with the original.
    The maximum of the values at 30, 90 and 150 degrees
    is the subtracted from the minimum of the values at 60, 120
    and 180 degrees to give the grid score.

    Parameters
    ----------
    image : np.ndarray
        The spatial autocorrelogram
    step : int, default=30
        The amount to rotate the SAC in each step of the
        rotational correlation procedure

    Returns
    -------
    3-tuple
        The gridscore, the correlation values at each
        `step` and the rotational array

    Notes
    -----
    The correlation performed is a Pearsons R. Some rescaling of the
    values in `image` is performed following rotation.

    See Also
    --------
    skimage.transform.rotate : for how the rotation of `image` is done
    skimage.exposure.rescale_intensity : for the resscaling following
    rotation
    """
    # TODO: add options in here for whether the full range of correlations
    # are wanted or whether a reduced set is wanted (i.e. at the 30-tuples)
    from collections import OrderedDict

    rotationalCorrVals = OrderedDict.fromkeys(np.arange(0, 181, step), np.nan)
    rotationArr = np.zeros(len(rotationalCorrVals)) * np.nan
    # autoCorrMiddle needs to be rescaled or the image rotation falls down
    # as values are cropped to lie between 0 and 1.0
    in_range = (np.nanmin(image), np.nanmax(image))
    out_range = (0, 1)
    import skimage

    autoCorrMiddleRescaled = skimage.exposure.rescale_intensity(
        image, in_range=in_range, out_range=out_range
    )
    origNanIdx = np.isnan(autoCorrMiddleRescaled.ravel())
    gridscore = np.nan
    try:
        for idx, angle in enumerate(rotationalCorrVals.keys()):
            rotatedA = skimage.transform.rotate(
                autoCorrMiddleRescaled, angle=angle, cval=np.nan, order=3
            )
            # ignore nans
            rotatedNanIdx = np.isnan(rotatedA.ravel())
            allNans = np.logical_or(origNanIdx, rotatedNanIdx)
            # get the correlation between the original and rotated images
            rotationalCorrVals[angle] = stats.pearsonr(
                autoCorrMiddleRescaled.ravel()[~allNans], rotatedA.ravel()[~allNans]
            )[0]
            rotationArr[idx] = rotationalCorrVals[angle]
    except Exception:
        return gridscore, rotationalCorrVals, rotationArr
    gridscore = np.min((rotationalCorrVals[60], rotationalCorrVals[120])) - np.max(
        (rotationalCorrVals[150], rotationalCorrVals[30], rotationalCorrVals[90])
    )
    return gridscore, rotationalCorrVals, rotationArr


def __deform_SAC(A, circleXY=None, ellipseXY=None):
    """
    Deforms an elliptical SAC to be circular

    Parameters
    ----------
    A : np.ndarray
        The SAC
    circleXY : np.ndarray, default=None
        The xy coordinates defining a circle.
    ellipseXY : np.ndarray, default=None
        The xy coordinates defining an ellipse.

    Returns
    -------
    np.ndarray
        The SAC deformed to be more circular

    See Also
    --------
    ephysiopy.common.ephys_generic.FieldCalcs.grid_field_props
    skimage.transform.AffineTransform
    skimage.transform.warp
    skimage.exposure.rescale_intensity
    """
    if circleXY is None or ellipseXY is None:
        SAC_stats = grid_field_props(A)
        circleXY = SAC_stats["circleXY"]
        ellipseXY = SAC_stats["ellipseXY"]
        # The ellipse detection stuff might have failed, if so
        # return the original SAC
        if circleXY is None:
            warnings.warn("Ellipse detection failed. Returning original SAC")
            return A

    tform = skimage.transform.AffineTransform()
    tform.estimate(ellipseXY, circleXY)

    """
    the transformation algorithms used here crop values < 0 to 0. Need to
    rescale the SAC values before doing the deformation and then rescale
    again so the values assume the same range as in the unadulterated SAC
    """
    A[np.isnan(A)] = 0
    SACmin = np.nanmin(A.flatten())
    SACmax = np.nanmax(A.flatten())  # should be 1 if autocorr
    AA = A + 1
    deformedSAC = skimage.transform.warp(
        AA / np.nanmax(AA.flatten()), inverse_map=tform.inverse, cval=0
    )
    return skimage.exposure.rescale_intensity(deformedSAC, out_range=(SACmin, SACmax))


def __get_circular_regions(A: np.ndarray, **kwargs) -> tuple:
    """
    Returns a list of images which are expanding circular
    regions centred on the middle of the image out to the
    image edge and the radii used to create them.
    Used for calculating the grid score of each
    image to find the one with the max grid score.

    Parameters
    ----------
    A : np.ndarray
        The SAC

    **kwargs
        min_radius (int): The smallest radius circle to start with

    Returns
    -------
    tuple
        A list of images which are circular sub-regions of the
        original SAC and a list of the radii used to create them
    """
    from skimage.measure import CircleModel, grid_points_in_poly

    min_radius = kwargs.get("min_radius", 5)

    centre = tuple([d // 2 for d in np.shape(A)])
    max_radius = min(tuple(np.subtract(np.shape(A), centre)))
    t = np.linspace(0, 2 * np.pi, 51)
    circle = CircleModel()

    result = []
    radii = []
    for radius in range(min_radius, max_radius):
        circle.params = [*centre, radius]
        xy = circle.predict_xy(t)
        mask = grid_points_in_poly(np.shape(A), xy)
        im = A.copy()
        im[~mask] = np.nan
        result.append(im)
        radii.append(radius)
    return result, radii


def get_basic_gridscore(A: np.ndarray, **kwargs) -> float:
    """
    Calculates the grid score of a spatial autocorrelogram

    Parameters
    ----------
    A : np.ndarray
        The spatial autocorrelogram

    Returns
    -------
    float
        The grid score of the SAC

    """
    return gridness(A, **kwargs)[0]


def get_expanding_circle_gridscore(A: np.ndarray, **kwargs):
    """
    Calculates the gridscore for each circular sub-region of image A
    where the circles are centred on the image centre and expanded to
    the edge of the image. The maximum of the get_basic_gridscore() for
    each of these circular sub-regions is returned as the gridscore

    Parameters
    ----------
    A : np.ndarray
        The SAC

    Returns
    -------
    float
        The maximum grid score of the circular sub
        regions of the SAC
    """

    images, _ = __get_circular_regions(A, **kwargs)
    gridscores = [gridness(im)[0] for im in images]
    return max(gridscores)


def get_deformed_sac_gridscore(A: np.ndarray) -> float:
    """
    Deforms a non-circular SAC into a circular SAC (circular meaning
    the ellipse drawn around the edges of the 6 nearest peaks to the
    SAC centre) and returns get_basic_griscore() calculated on the
    deformed (or re-formed?!) SAC

    Parameters
    ----------
    A : np.ndarray
        The SAC

    Returns
    -------
    float
        The gridscore of the deformed SAC
    """
    deformed_SAC = __deform_SAC(A)
    return gridness(deformed_SAC)[0]


def get_thigmotaxis_score(xy: np.ndarray, shape: str = "circle") -> float:
    """
    Returns a score which is the ratio of the time spent in the inner
    portion of an environment to the time spent in the outer portion.
    The portions are allocated so that they have equal area.

    Parameters
    ----------
    xy : np.ndarray
        The xy coordinates of the animal's position. 2 x nsamples
    shape :str, default='circle'
        The shape of the environment. Legal values are 'circle' and 'square'

    Returns
    -------
    float
        Values closer to 1 mean more time was spent in the inner portion of the environment.
        Values closer to -1 mean more time in the outer portion of the environment.
        A value of 0 indicates the animal spent equal time in both portions of the
        environment.
    """
    # centre the coords to get the max distance from the centre
    xc, yc = np.nanmin(xy, -1) + (np.nanmax(xy, -1) - np.nanmin(xy, -1)) / 2
    xy = xy - np.array([[xc], [yc]])
    n_pos = np.shape(xy)[1]
    inner_mask = np.zeros((n_pos), dtype=bool)
    if shape == "circle":
        outer_radius = np.nanmax(np.hypot(xy[0], xy[1]))
        inner_radius = outer_radius / np.sqrt(2)
        inner_mask = np.less(np.hypot(xy[0], xy[1]), inner_radius, out=inner_mask)
    elif shape == "square":
        width, height = np.nanmax(xy, -1) - np.nanmin(xy, -1)
        inner_width = width / np.sqrt(2)
        inner_height = height / np.sqrt(2)
        x_gap = (width - inner_width) / 2
        y_gap = (height - inner_height) / 2
        x_mask = (xy[0] > np.nanmin(xy[0]) + x_gap) & (xy[0] < np.nanmax(xy[0]) - x_gap)
        y_mask = (xy[1] > np.nanmin(xy[1]) + y_gap) & (xy[1] < np.nanmax(xy[1]) - y_gap)
        inner_mask = np.logical_and(x_mask, y_mask, out=inner_mask)
    return (np.count_nonzero(inner_mask) - np.count_nonzero(~inner_mask)) / n_pos


def fast_overdispersion(
    rmap: BinnedData, xy: np.ndarray, spikes: np.ndarray, **kws
) -> float:
    """
    Calculates the overdispersion of a spatial ratemap

    Parameters
    ----------
    rmap : BinnedData
        The spatial ratemap
    xy : np.ndarray
        The xy data
    spikes : np.ndarray
        The spike times binned into in samples. Length
        will be equal to duration in seconds * sample_rate
    **kws
        sample_rate : int, default=50
            The sample rate of the position data
        window : int, default=5
            The window in seconds over which to calculate the overdispersion

    Returns
    -------
    float
        The overdispersion of the spatial ratemap
    """
    spikes = np.ravel(spikes)
    sample_rate = kws.get("sample_rate", 50)
    window = kws.get("window", 5)  # seconds
    window_samples = window * sample_rate
    n_windows = int(np.floor(len(spikes) / window_samples))
    if n_windows < 2:
        raise ValueError("Not enough data to calculate overdispersion")
    ratemap = rmap.binned_data[0]
    # get the bins in the ratemap that correspond to the position data
    if rmap.variable.value == VariableToBin.X.value:
        xbins = np.digitize(xy[0], rmap.bin_edges[0][:-1])
    if rmap.variable.value == VariableToBin.XY.value:
        xbins = np.digitize(xy[0], rmap.bin_edges[1][:-1])
        ybins = np.digitize(xy[1], rmap.bin_edges[0][:-1])

    # now loop over the windows and calculate the overdispersion
    Z2 = np.zeros(n_windows) * np.nan
    for w in range(n_windows):
        start_idx = w * window_samples
        end_idx = start_idx + window_samples
        xi = xbins[start_idx:end_idx] - 1
        if rmap.variable.value == VariableToBin.X.value:
            exptd_rate = ratemap[xi]
        if rmap.variable.value == VariableToBin.XY.value:
            yi = ybins[start_idx:end_idx] - 1
            exptd_rate = ratemap[yi, xi]
        exptd_spikes = np.sum(exptd_rate / sample_rate)
        observed_spikes = np.sum(spikes[start_idx:end_idx])
        Z2[w] = (observed_spikes - exptd_spikes) / np.sqrt(exptd_spikes)

    return np.nanvar(Z2), Z2
