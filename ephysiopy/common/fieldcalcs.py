import numpy as np
from scipy import signal
from scipy import ndimage
from scipy import spatial
from scipy import stats
import skimage
import warnings
from skimage.segmentation import watershed
from ephysiopy.common.utils import blur_image, BinnedData, MapType
from ephysiopy.common.binning import RateMap
from ephysiopy.common.ephys_generic import PosCalcsGeneric


"""
These methods differ from MapCalcsGeneric in that they are mostly
concerned with treating rate maps as images as opposed to using
the spiking information contained within them. They therefore mostly
deals with spatial rate maps of place and grid cells.
"""


def get_mean_resultant(ego_boundary_map: np.ndarray, **kwargs) -> float:
    """
    Calculates the mean resultant vector of a boundary map in egocentric coordinates

    See Hinman et al., 2019 for more details

    Args:
        ego_boundary_map (np.ndarray): The egocentric boundary map

    Returns:
        float: The mean resultant vector of the egocentric boundary map
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
    MR = get_mean_resultant(ego_boundary_map, **kwargs)
    return np.abs(MR)


def get_mean_resultant_angle(ego_boundary_map: np.ndarray, **kwargs) -> float:
    MR = get_mean_resultant(ego_boundary_map, **kwargs)
    return np.rad2deg(np.arctan2(np.imag(MR), np.real(MR)))


# def getCentreBearingCurve(rmap: RateMap, pos: PosCalcsGeneric) -> np.ndarray:
#     pass


def field_lims(A):
    """
    Returns a labelled matrix of the ratemap A.
    Uses anything greater than the half peak rate to select as a field.
    Data is heavily smoothed.

    Args:
        A (BinnedData): A BinnedData instance containing the ratemap

    Returns:
        label (np.array): The labelled ratemap
    """
    Ac = A.binned_data[0]
    nan_idx = np.isnan(Ac)
    Ac[nan_idx] = 0
    h = int(np.max(Ac.shape) / 2)
    sm_rmap = blur_image(A, h, ftype="gaussian").binned_data[0]
    thresh = np.max(sm_rmap.ravel()) * 0.2  # select area > 20% of peak
    distance = ndimage.distance_transform_edt(sm_rmap > thresh)
    peak_idx = skimage.feature.peak_local_max(
        distance, exclude_border=False, labels=sm_rmap > thresh
    )
    mask = np.zeros_like(distance, dtype=bool)
    mask[tuple(peak_idx.T)] = True
    label = ndimage.label(mask)[0]
    w = watershed(image=-distance, markers=label, mask=sm_rmap > thresh)
    label = ndimage.label(w)[0]
    return label


def limit_to_one(A, prc=50, min_dist=5):
    """
    Processes a multi-peaked ratemap (ie grid cell) and returns a matrix
    where the multi-peaked ratemap consist of a single peaked field that is
    a) not connected to the border and b) close to the middle of the
    ratemap
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
    peak_idx = skimage.feature.peak_local_max(
        Ac, min_distance=min_dist, exclude_border=False
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


def global_threshold(A, prc=50, min_dist=5):
    """
    Globally thresholds a ratemap and counts number of fields found
    """
    Ac = A.copy()
    Ac[np.isnan(A)] = 0
    n = ny = 5
    x, y = np.mgrid[-n : n + 1, -ny : ny + 1]
    g = np.exp(-(x**2 / float(n) + y**2 / float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode="same")
    maxRate = np.nanmax(np.ravel(Ac))
    Ac[Ac < maxRate * (prc / float(100))] = 0
    peak_idx = skimage.feature.peak_local_max(
        Ac, min_distance=min_dist, exclude_border=False
    )
    peak_mask = np.zeros_like(Ac, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True
    peak_labels = skimage.measure.label(peak_mask, connectivity=2)
    field_labels = watershed(image=Ac * -1, markers=peak_labels)
    nFields = np.max(field_labels)
    return nFields


def local_threshold(A, prc=50, min_dist=5):
    """
    Locally thresholds a ratemap to take only the surrounding prc amount
    around any local peak
    """
    Ac = A.copy()
    nanidx = np.isnan(Ac)
    Ac[nanidx] = 0
    # smooth Ac more to remove local irregularities
    n = ny = 5
    x, y = np.mgrid[-n : n + 1, -ny : ny + 1]
    g = np.exp(-(x**2 / float(n) + y**2 / float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode="same")
    peak_idx = skimage.feature.peak_local_max(
        Ac, min_distance=min_dist, exclude_border=False
    )
    peak_mask = np.zeros_like(Ac, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True
    peak_labels = skimage.measure.label(peak_mask, connectivity=2)
    field_labels = watershed(image=Ac * -1, markers=peak_labels)
    nFields = np.max(field_labels)
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
    A_out = np.zeros_like(A)
    A_out[sub_field_mask.astype(bool)] = A[sub_field_mask.astype(bool)]
    A_out[nanidx] = np.nan
    return A_out


def border_score(
    A,
    B=None,
    shape="square",
    fieldThresh=0.3,
    smthKernSig=3,
    circumPrc=0.2,
    binSize=3.0,
    minArea=200,
    debug=False,
):
    """

    Calculates a border score totally dis-similar to that calculated in
    Solstad et al (2008)

    Args:
        A (array_like): Should be the ratemap
        B (array_like): This should be a boolean mask where True (1)
            is equivalent to the presence of a border and False (0)
            is equivalent to 'open space'. Naievely this will be the
            edges of the ratemap but could be used to take account of
            boundary insertions/ creations to check tuning to multiple
            environmental boundaries. Default None: when the mask is
            None then a mask is created that has 1's at the edges of the
            ratemap i.e. it is assumed that occupancy = environmental
            shape
        shape (str): description of environment shape. Currently
            only 'square' or 'circle' accepted. Used to calculate the
            proportion of the environmental boundaries to examine for
            firing
        fieldThresh (float): Between 0 and 1 this is the percentage
            amount of the maximum firing rate
            to remove from the ratemap (i.e. to remove noise)
        smthKernSig (float): the sigma value used in smoothing the ratemap
            (again!) with a gaussian kernel
        circumPrc (float): The percentage amount of the circumference
            of the environment that the field needs to be to count
            as long enough to make it through
        binSize (float): bin size in cm
        minArea (float): min area for a field to be considered
        debug (bool): If True then some plots and text will be output

    Returns:
        float: the border score

    Notes:
        If the cell is a border cell (BVC) then we know that it should
        fire at a fixed distance from a given boundary (possibly more
        than one). In essence this algorithm estimates the amount of
        variance in this distance i.e. if the cell is a border cell this
        number should be small. This is achieved by first doing a bunch of
        morphological operations to isolate individual fields in the
        ratemap (similar to the code used in phasePrecession.py - see
        the partitionFields method therein). These partitioned fields are then
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
        dists = ndimage.distance_transform_bf(tmp)
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
        dists = ndimage.distance_transform_bf(tmp)
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
    labels, nFields = ndimage.label(A_thresh)
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

    Args:
        A (np.ndarray): The array to process
        min_distance (float, optional): The distance in bins between fields to
        separate the regions of the image
        clear_border (bool, optional): Input to skimage.feature.peak_local_max.
        The number of
            pixels to ignore at the edge of the image
    """
    clear_border = True
    if "clear_border" in kwargs:
        clear_border = kwargs.pop("clear_border")

    min_distance = 1
    if "min_distance" in kwargs:
        min_distance = kwargs.pop("min_distance")

    A[~np.isfinite(A)] = -1
    A[A < 0] = -1

    peak_coords = skimage.feature.peak_local_max(
        A, min_distance=min_distance, exclude_border=clear_border
    )
    peaksMask = np.zeros_like(A, dtype=bool)
    peaksMask[tuple(peak_coords.T)] = True
    peaksLabel, nLbls = ndimage.label(peaksMask)
    ws = watershed(image=-1 * A, markers=peaksLabel)
    return peak_coords, ws


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
    Calculates a measure of spatial sampling of an arena by comparing the
    given spatial sampling to a uniform one using kl divergence

    Data in pos_map should be unsmoothed (not checked) and the MapType should
    be POS (checked)
    """
    assert pos_map.map_type == MapType.POS
    return kldiv_dir(np.ravel(pos_map.binned_data[0]))


def spatial_sparsity(rate_map: np.ndarray, pos_map: np.ndarray) -> float:
    """
    Calculates the spatial sparsity of a rate map as defined by
    Markus et al (1994)

    For example, a sparsity score of 0.10 indicates that the cell fired on
    10% of the maze surface

    Args:
        rate_map (np.ndarray): The rate map
        pos_map (np.ndarray): The occupancy map

    Returns:
        float: The spatial sparsity of the rate map

    References:
        Markus, E.J., Barnes, C.A., McNaughton, B.L., Gladden, V.L. &
        Skaggs, W.E. Spatial information content and reliability of
        hippocampal CA1 neurons: effects of visual input. Hippocampus
        4, 410–421 (1994).

    """
    p_i = pos_map / np.nansum(pos_map)
    sparsity = np.nansum(p_i * rate_map) ** 2 / np.nansum(p_i * rate_map**2)
    return sparsity


def coherence(smthd_rate, unsmthd_rate):
    """calculates coherence of receptive field via correlation of smoothed
    and unsmoothed ratemaps
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

    Args:
        polarPlot (1D-array): The binned and smoothed directional ratemap

    Returns:
        klDivergence (float): The divergence from circular of the 1D-array
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

    Args:
        X (array_like): Vector of M variable values
        P1 (array_like): Length-M vector of probabilities representing
        distribution 1
        P2 (array_like): Length-M vector of probabilities representing
        distribution 2
        sym (str, optional): If 'sym', returns a symmetric variant of the
            Kullback-Leibler divergence, given by [KL(P1,P2)+KL(P2,P1)]/2
        js (str, optional): If 'js', returns the Jensen-Shannon divergence,
        given by
            [KL(P1,Q)+KL(P2,Q)]/2, where Q = (P1+P2)/2

    Returns:
        float: The Kullback-Leibler divergence or Jensen-Shannon divergence

    Notes:
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

    See Also:
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
        warnings.warn("Probabilities don" "t sum to 1.", UserWarning)
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


def skaggs_info(ratemap, dwelltimes, **kwargs):
    """
    Calculates Skaggs information measure

    Args:
        ratemap (array_like): The binned up ratemap
        dwelltimes (array_like): Must be same size as ratemap

    Returns:
        bits_per_spike (float): Skaggs information score

    Notes:
        THIS DATA SHOULD UNDERGO ADAPTIVE BINNING
        See getAdaptiveMap() in binning class

        Returns Skaggs et al's estimate of spatial information
        in bits per spike:

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

    Args:
        A: BinnedData object containing the spatial autocorrelogram (SAC) in
            A.binned_data[0]
        maxima (str, optional): The method used to detect the peaks in the SAC.
            Legal values are 'single' and 'centroid'. Default 'centroid'
        allProps (bool, optional): Whether to return a dictionary that
        contains the attempt to fit an ellipse around the edges of the
        central size peaks. See below
            Default True

    Returns:
        props (dict): A dictionary containing measures of the SAC.
        Keys include:
            * gridness score
            * scale
            * orientation
            * coordinates of the peaks (nominally 6) closest to SAC centre
            * a binary mask around the extent of the 6 central fields
            * values of the rotation procedure used to calculate gridness
            * ellipse axes and angle (if allProps is True and the it worked)

    Notes:
        The output from this method can be used as input to the show() method
        of this class.
        When it is the plot produced will display a lot more informative.
        The coordinate system internally used is centred on the image centre.

    See Also:
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

    peak_idx, field_labels = _get_field_labels(A_tmp, neighbours=7, **kwargs)
    # a fcn for the labeled_comprehension function that returns
    # linear indices in A where the values in A for each label are
    # greater than half the max in that labeled region

    def fn(val, pos):
        return pos[val > (np.max(val) / 2)]

    nLbls = np.max(field_labels)
    indices = ndimage.labeled_comprehension(
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
    peak_coords = ndimage.maximum_position(
        A.binned_data[0], half_peak_labels, lbl_range
    )
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
    orientation = grid_orientation(peak_coords, closest_peak_idx)

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

            ellipse_coords = ndimage.labeled_comprehension(
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


def grid_orientation(peakCoords, closestPeakIdx):
    """
    Calculates the orientation angle of a grid field.

    The orientation angle is the angle of the first peak working
    counter-clockwise from 3 o'clock

    Args:
        peakCoords (array_like): The peak coordinates as pairs of xy
        closestPeakIdx (array_like): A 1D array of the indices in peakCoords
        of the peaks closest to the centre of the SAC

    Returns:
        peak_orientation (float): The first value in an array of the angles of
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


def gridness(image, step=30):
    """
    Calculates the gridness score in a grid cell SAC.

    Briefly, the data in `image` is rotated in `step` amounts and
    each rotated array is correlated with the original.
    The maximum of the values at 30, 90 and 150 degrees
    is the subtracted from the minimum of the values at 60, 120
    and 180 degrees to give the grid score.

    Args:
        image (array_like): The spatial autocorrelogram
        step (int, optional): The amount to rotate the SAC in each step of the
        rotational correlation procedure

    Returns:
        gridmeasures (3-tuple): The gridscore, the correlation values at each
        `step` and the rotational array

    Notes:
        The correlation performed is a Pearsons R. Some rescaling of the
        values in `image` is performed following rotation.

    See Also:
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


def deform_SAC(A, circleXY=None, ellipseXY=None):
    """
    Deforms a SAC that is non-circular to be more circular

    Basically a blatant attempt to improve grid scores, possibly
    introduced in a paper by Matt Nolan...

    Args:
        A (array_like): The SAC
        circleXY (array_like, optional): The xy coordinates defining a circle.
        Default None.
        ellipseXY (array_like, optional): The xy coordinates defining an
        ellipse. Default None.

    Returns:
        deformed_sac (array_like): The SAC deformed to be more circular

    See Also:
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


def get_circular_regions(A: np.ndarray, **kwargs) -> list:
    """
    Returns a list of images which are expanding circular
    regions centred on the middle of the image out to the
    image edge. Used for calculating the grid score of each
    image to find the one with the max grid score. Based on
    some Moser paper I can't recall.

    Args:
        A (np.ndarray): The SAC

    Keyword Args:
        min_radius (int): The smallest radius circle to start with
    """
    from skimage.measure import CircleModel, grid_points_in_poly

    min_radius = 5
    if "min_radius" in kwargs.keys():
        min_radius = kwargs["min_radius"]

    centre = tuple([d // 2 for d in np.shape(A)])
    max_radius = min(tuple(np.subtract(np.shape(A), centre)))
    t = np.linspace(0, 2 * np.pi, 51)
    circle = CircleModel()

    result = []
    for radius in range(min_radius, max_radius):
        circle.params = [*centre, radius]
        xy = circle.predict_xy(t)
        mask = grid_points_in_poly(np.shape(A), xy)
        im = A.copy()
        im[~mask] = np.nan
        result.append(im)
    return result


def get_basic_gridscore(A: np.ndarray, **kwargs):
    return gridness(A, **kwargs)[0]


def get_expanding_circle_gridscore(A: np.ndarray, **kwargs):
    """
    Calculates the gridscore for each circular sub-region of image A
    where the circles are centred on the image centre and expanded to
    the edge of the image. The maximum of the get_basic_gridscore() for
    each of these circular sub-regions is returned as the gridscore
    """

    images = get_circular_regions(A, **kwargs)
    gridscores = [gridness(im) for im in images]
    return max(gridscores)


def get_deformed_sac_gridscore(A: np.ndarray):
    """
    Deforms a non-circular SAC into a circular SAC (circular meaning
    the ellipse drawn around the edges of the 6 nearest peaks to the
    SAC centre) and returns get_basic_griscore() calculated on the
    deformed (or re-formed?!) SAC
    """
    deformed_SAC = deform_SAC(A)
    return gridness(deformed_SAC)


def get_thigmotaxis_score(xy: np.ndarray, shape: str = "circle") -> float:
    """
    Returns a score which is the ratio of the time spent in the inner
    portion of an environment to the time spent in the outer portion.
    The portions are allocated so that they have equal area.

    Args:
        xy (np.ndarray): The xy coordinates of the animal's position. 2 x nsamples
        shape (str): The shape of the environment. Legal values are 'circle'
        and 'square'. Default 'circle'

    Returns:
    thigmoxtaxis_score (float): Values closer to 1 indicate the
    animal spent more time in the inner portion of the environment. Values closer to -1
    indicates the animal spent more time in the outer portion of the environment.
    A value of 0 indicates the animal spent equal time in both portions of the
    environment.
    """
    # centre the coords to get the max distance from the centre
    xc, yc = np.min(xy, -1) + np.ptp(xy, -1) / 2
    xy = xy - np.array([[xc], [yc]])
    n_pos = np.shape(xy)[1]
    inner_mask = np.zeros((n_pos), dtype=bool)
    if shape == "circle":
        outer_radius = np.max(np.hypot(xy[0], xy[1]))
        inner_radius = outer_radius / np.sqrt(2)
        inner_mask = np.less(np.hypot(xy[0], xy[1]), inner_radius, out=inner_mask)
    elif shape == "square":
        width, height = np.ptp(xy, -1)
        inner_width = width / np.sqrt(2)
        inner_height = height / np.sqrt(2)
        x_gap = (width - inner_width) / 2
        y_gap = (height - inner_height) / 2
        x_mask = (xy[0] > np.min(xy[0]) + x_gap) & (xy[0] < np.max(xy[0]) - x_gap)
        y_mask = (xy[1] > np.min(xy[1]) + y_gap) & (xy[1] < np.max(xy[1]) - y_gap)
        inner_mask = np.logical_and(x_mask, y_mask, out=inner_mask)
    return (np.count_nonzero(inner_mask) - np.count_nonzero(~inner_mask)) / n_pos
