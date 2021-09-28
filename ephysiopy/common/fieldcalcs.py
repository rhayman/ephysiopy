import numpy as np
from scipy import signal
from scipy import ndimage
from scipy import spatial
from scipy import stats
import skimage
import warnings
from skimage.segmentation import watershed
from ephysiopy.common.utils import blurImage

"""
These methods differ from MapCalcsGeneric in that they are mostly
concerned with treating rate maps as images as opposed to using
the spiking information contained within them. They therefore mostly
deals with spatial rate maps of place and grid cells.
"""


def field_lims(A):
    """
    Returns a labelled matrix of the ratemap A.
    Uses anything >
    than the half peak rate to select as a field. Data is heavily smoothed
    Parameters
    ----------
    A: np.array
        The ratemap
    Returns
    -------
    label: np.array
        The labelled ratemap
    """
    nan_idx = np.isnan(A)
    A[nan_idx] = 0
    h = int(np.max(A.shape) / 2)
    sm_rmap = blurImage(A, h, ftype='gaussian')
    thresh = np.max(sm_rmap.ravel()) * 0.2  # select area > 20% of peak
    distance = ndimage.distance_transform_edt(sm_rmap > thresh)
    mask = skimage.feature.peak_local_max(
        distance, indices=False,
        exclude_border=False,
        labels=sm_rmap > thresh)
    label = ndimage.label(mask)[0]
    w = watershed(
        image=-distance, markers=label,
        mask=sm_rmap > thresh)
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
    x, y = np.mgrid[-n:n+1, -ny:ny+1]
    g = np.exp(-(x**2/float(n) + y**2/float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode='same')
    # remove really small values
    Ac[Ac < 1e-10] = 0
    peak_mask = skimage.feature.peak_local_max(
        Ac, min_distance=min_dist,
        exclude_border=False,
        indices=False)
    peak_labels = skimage.measure.label(peak_mask, connectivity=2)
    field_labels = watershed(
        image=-Ac, markers=peak_labels)
    nFields = np.max(field_labels)
    sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
    labelled_sub_field_mask = np.zeros_like(sub_field_mask)
    sub_field_props = skimage.measure.regionprops(
        field_labels, intensity_image=Ac)
    sub_field_centroids = []
    sub_field_size = []

    for sub_field in sub_field_props:
        tmp = np.zeros(Ac.shape).astype(bool)
        tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
        tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
        sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(
            tmp2, tmp)
        labelled_sub_field_mask[
            sub_field.label-1, np.logical_and(tmp2, tmp)] = sub_field.label
        sub_field_centroids.append(sub_field.centroid)
        sub_field_size.append(sub_field.area)  # in bins
    sub_field_mask = np.sum(sub_field_mask, 0)
    middle = np.round(np.array(A.shape) / 2)
    normd_dists = sub_field_centroids - middle
    field_dists_from_middle = np.hypot(
        normd_dists[:, 0], normd_dists[:, 1])
    central_field_idx = np.argmin(field_dists_from_middle)
    central_field = np.squeeze(
        labelled_sub_field_mask[central_field_idx, :, :])
    # collapse the labelled mask down to an 2d array
    labelled_sub_field_mask = np.sum(labelled_sub_field_mask, 0)
    # clear the border
    cleared_mask = skimage.segmentation.clear_border(central_field)
    # check we've still got stuff in the matrix or fail
    if ~np.any(cleared_mask):
        print(
            'No fields were detected away from edges so nothing returned')
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
    x, y = np.mgrid[-n:n+1, -ny:ny+1]
    g = np.exp(-(x**2/float(n) + y**2/float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode='same')
    maxRate = np.nanmax(np.ravel(Ac))
    Ac[Ac < maxRate*(prc/float(100))] = 0
    peak_mask = skimage.feature.peak_local_max(
        Ac, min_distance=min_dist,
        exclude_border=False,
        indices=False)
    peak_labels = skimage.measure.label(peak_mask, connectivity=2)
    field_labels = watershed(
        image=-Ac, markers=peak_labels)
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
    x, y = np.mgrid[-n:n+1, -ny:ny+1]
    g = np.exp(-(x**2/float(n) + y**2/float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode='same')
    peak_mask = skimage.feature.peak_local_max(
        Ac, min_distance=min_dist, exclude_border=False,
        indices=False)
    peak_labels = skimage.measure.label(peak_mask, connectivity=2)
    field_labels = watershed(
        image=-Ac, markers=peak_labels)
    nFields = np.max(field_labels)
    sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
    sub_field_props = skimage.measure.regionprops(
        field_labels, intensity_image=Ac)
    sub_field_centroids = []
    sub_field_size = []

    for sub_field in sub_field_props:
        tmp = np.zeros(Ac.shape).astype(bool)
        tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
        tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
        sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(
            tmp2, tmp)
        sub_field_centroids.append(sub_field.centroid)
        sub_field_size.append(sub_field.area)  # in bins
    sub_field_mask = np.sum(sub_field_mask, 0)
    A_out = np.zeros_like(A)
    A_out[sub_field_mask.astype(bool)] = A[sub_field_mask.astype(bool)]
    A_out[nanidx] = np.nan
    return A_out


def border_score(
        A, B=None, shape='square', fieldThresh=0.3, smthKernSig=3,
        circumPrc=0.2, binSize=3.0, minArea=200, debug=False):
    """
    Calculates a border score totally dis-similar to that calculated in
    Solstad et al (2008)

    Parameters
    ----------
    A : array_like
        Should be the ratemap
    B : array_like
        This should be a boolean mask where True (1)
        is equivalent to the presence of a border and False (0)
        is equivalent to 'open space'. Naievely this will be the
        edges of the ratemap but could be used to take account of
        boundary insertions/ creations to check tuning to multiple
        environmental boundaries. Default None: when the mask is
        None then a mask is created that has 1's at the edges of the
        ratemap i.e. it is assumed that occupancy = environmental
        shape
    shape : str
        description of environment shape. Currently
        only 'square' or 'circle' accepted. Used to calculate the
        proportion of the environmental boundaries to examine for
        firing
    fieldThresh : float
        Between 0 and 1 this is the percentage
        amount of the maximum firing rate
        to remove from the ratemap (i.e. to remove noise)
    smthKernSig : float
        the sigma value used in smoothing the ratemap
        (again!) with a gaussian kernel
    circumPrc : float
        The percentage amount of the circumference
        of the environment that the field needs to be to count
        as long enough to make it through
    binSize : float
        bin size in cm
    minArea : float
        min area for a field to be considered
    debug : bool
        If True then some plots and text will be output

    Returns
    -------
    float : the border score

    Notes
    -----
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
    if 'circle' in shape:
        radius = np.max(np.array(np.shape(A))) / 2.0
        dist_mask = skimage.morphology.disk(radius)
        if np.shape(dist_mask) > np.shape(A):
            dist_mask = dist_mask[1:A_rows+1, 1:A_cols+1]
        tmp = np.zeros([A_rows + 2, A_cols + 2])
        tmp[1:-1, 1:-1] = dist_mask
        dists = ndimage.morphology.distance_transform_bf(tmp)
        dists = dists[1:-1, 1:-1]
        borderMask = np.logical_xor(dists <= 0, dists < 2)
        # open up the border mask a little
        borderMask = skimage.morphology.binary_dilation(
            borderMask, skimage.morphology.disk(1))
    elif 'square' in shape:
        borderMask[0:3, :] = 1
        borderMask[-3:, :] = 1
        borderMask[:, 0:3] = 1
        borderMask[:, -3:] = 1
        tmp = np.zeros([A_rows + 2, A_cols + 2])
        dist_mask = np.ones_like(A)
        tmp[1:-1, 1:-1] = dist_mask
        dists = ndimage.morphology.distance_transform_bf(tmp)
        # remove edges to make same shape as input ratemap
        dists = dists[1:-1, 1:-1]
    A[np.isnan(A)] = 0
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
    skimage.morphology.remove_small_objects(
        labels, min_size=min_size, connectivity=2, in_place=True)
    labels = skimage.segmentation.relabel_sequential(labels)[0]
    nFields = np.max(labels)
    if nFields == 0:
        return np.nan
    # Iterate over the labelled parts of the array labels calculating
    # how much of the total circumference of the environment edge it
    # covers

    fieldAngularCoverage = np.zeros([1, nFields]) * np.nan
    fractionOfPixelsOnBorder = np.zeros([1, nFields]) * np.nan
    fieldsToKeep = np.zeros_like(A)
    for i in range(1, nFields+1):
        fieldMask = np.logical_and(labels == i, borderMask)

        # check the angle subtended by the fieldMask
        if np.sum(fieldMask.astype(int)) > 0:
            s = skimage.measure.regionprops(
                fieldMask.astype(int), intensity_image=A_thresh)[0]
            x = s.coords[:, 0] - (A_cols / 2.0)
            y = s.coords[:, 1] - (A_rows / 2.0)
            subtended_angle = np.rad2deg(np.ptp(np.arctan2(x, y)))
            if subtended_angle > (360 * circumPrc):
                pixelsOnBorder = np.count_nonzero(
                    fieldMask) / float(np.count_nonzero(labels == i))
                fractionOfPixelsOnBorder[:, i-1] = pixelsOnBorder
                if pixelsOnBorder > 0.5:
                    fieldAngularCoverage[0, i-1] = subtended_angle

            fieldsToKeep = np.logical_or(fieldsToKeep, labels == i)
    fieldAngularCoverage = (fieldAngularCoverage / 360.)
    rateInField = A[fieldsToKeep]
    # normalize firing rate in the field to sum to 1
    rateInField = rateInField / np.nansum(rateInField)
    dist2WallInField = dists[fieldsToKeep]
    Dm = np.dot(dist2WallInField, rateInField)
    if 'circle' in shape:
        Dm = Dm / radius
    elif 'square' in shape:
        Dm = Dm / (np.max(np.shape(A)) / 2.0)
    borderScore = (fractionOfPixelsOnBorder-Dm) / (
        fractionOfPixelsOnBorder+Dm)
    return np.max(borderScore)


def _get_field_labels(A: np.ndarray, **kwargs) -> tuple:
    '''
    Returns a labeled version of A after finding the peaks
    in A and finding the watershed basins from the markers
    found from those peaks. Used in field_props() and
    grid_field_props()

    Parameters
    -----------------
    A : np.ndarray
    Valid kwargs:
    min_distance : float
        The distance in bins between fields to separate the regions
        of the image
    clear_border : bool
        Input to skimage.feature.peak_local_max. The number of
        pixels to ignore at the edge of the image
    '''
    clear_border = True
    if 'clear_border' in kwargs:
        clear_border = kwargs.pop('clear_border')
        
    min_distance = 1
    if 'min_distance' in kwargs:
        min_distance = kwargs.pop('min_distance')

    A[~np.isfinite(A)] = -1
    A[A < 0] = -1

    peak_coords = skimage.feature.peak_local_max(
        A, min_distance=min_distance,
        exclude_border=clear_border)
    peaksMask = np.zeros_like(A, dtype=bool)
    peaksMask[tuple(peak_coords.T)] = True
    peaksLabel, nLbls = ndimage.label(peaksMask)
    ws = watershed(image=-A, markers=peaksLabel)
    return peak_coords, ws


def field_props(
        A, min_dist=5, neighbours=2, prc=50,
        plot=False, ax=None, tri=False, verbose=True, **kwargs):
    """
    Returns a dictionary of properties of the field(s) in a ratemap A

    Parameters
    ----------
    A : array_like
        a ratemap (but could be any image)
    min_dist : float
        the separation (in bins) between fields for measures
        such as field distance to make sense. Used to
        partition the image into separate fields in the call to
        feature.peak_local_max
    neighbours : int
        the number of fields to consider as neighbours to
        any given field. Defaults to 2
    prc : float
        percent of fields to consider
    ax : matplotlib.Axes
        user supplied axis. If None a new figure window is created
    tri : bool
        whether to do Delaunay triangulation between fields
        and add to plot
    verbose : bool
        dumps the properties to the console
    plot : bool
        whether to plot some output - currently consists of the
        ratemap A, the fields of which are outline in a black
        contour. Default False

    Returns
    -------
    result : dict
        The properties of the field(s) in the input ratemap A
    """

    from skimage.measure import find_contours
    from sklearn.neighbors import NearestNeighbors

    nan_idx = np.isnan(A)
    Ac = A.copy()
    Ac[np.isnan(A)] = 0
    # smooth Ac more to remove local irregularities
    n = ny = 5
    x, y = np.mgrid[-n:n+1, -ny:ny+1]
    g = np.exp(-(x**2/float(n) + y**2/float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode='same')

    peak_idx, field_labels = _get_field_labels(Ac, **kwargs)

    nFields = np.max(field_labels)
    if neighbours > nFields:
        print('neighbours value of {0} > the {1} peaks found'.format(
            neighbours, nFields))
        print('Reducing neighbours to number of peaks found')
        neighbours = nFields
    sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
    sub_field_props = skimage.measure.regionprops(
        field_labels, intensity_image=Ac)
    sub_field_centroids = []
    sub_field_size = []

    for sub_field in sub_field_props:
        tmp = np.zeros(Ac.shape).astype(bool)
        tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
        tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
        sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(
            tmp2, tmp)
        sub_field_centroids.append(sub_field.centroid)
        sub_field_size.append(sub_field.area)  # in bins
    sub_field_mask = np.sum(sub_field_mask, 0)
    contours = skimage.measure.find_contours(sub_field_mask, 0.5)
    # find the nearest neighbors to the peaks of each sub-field
    nbrs = NearestNeighbors(n_neighbors=neighbours,
                            algorithm='ball_tree').fit(peak_idx)
    distances, _ = nbrs.kneighbors(peak_idx)
    mean_field_distance = np.mean(distances[:, 1:neighbours])

    nValid_bins = np.sum(~nan_idx)
    # calculate the amount of out of field firing
    A_non_field = np.zeros_like(A) * np.nan
    A_non_field[~sub_field_mask.astype(bool)] = A[
        ~sub_field_mask.astype(bool)]
    A_non_field[nan_idx] = np.nan
    out_of_field_firing_prc = (np.count_nonzero(
        A_non_field > 0) / float(nValid_bins)) * 100
    Ac[np.isnan(A)] = np.nan
    """
    get some stats about the field ellipticity
    """
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
    if 'calc_angs' in kwargs.keys():
        angs = calc_angs(peak_idx)
    else:
        angs = None

    props = {
        'Ac': Ac, 'Peak_rate': np.nanmax(A), 'Mean_rate': np.nanmean(A),
        'Field_size': np.mean(sub_field_size),
        'Pct_bins_with_firing': (np.sum(
            sub_field_mask) / nValid_bins) * 100,
        'Out_of_field_firing_prc': out_of_field_firing_prc,
        'Dist_between_fields': mean_field_distance,
        'Num_fields': float(nFields),
        'Sub_field_mask': sub_field_mask,
        'Smoothed_map': Ac,
        'field_labels': field_labels,
        'Peak_idx': peak_idx,
        'angles': angs,
        'contours': contours,
        'ellipse_ratio': ellipse_ratio}

    if verbose:
        print('\nPercentage of bins with firing: {:.2%}'.format(
            np.sum(sub_field_mask) / nValid_bins))
        print('Percentage out of field firing: {:.2%}'.format(
            np.count_nonzero(A_non_field > 0) / float(nValid_bins)))
        print('Peak firing rate: {:.3} Hz'.format(np.nanmax(A)))
        print('Mean firing rate: {:.3} Hz'.format(np.nanmean(A)))
        print('Number of fields: {0}'.format(nFields))
        print('Mean field size: {:.5} cm'.format(np.mean(sub_field_size)))
        print('Mean inter-peak distance between \
            fields: {:.4} cm'.format(mean_field_distance))
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
            angs.append(np.arccos(num/denom) * 180 / np.pi)
    return np.array(angs).T


def corr_maps(map1, map2, maptype='normal'):
    """
    correlates two ratemaps together ignoring areas that have zero sampling
    """
    if map1.shape > map2.shape:
        map2 = skimage.transform.resize(map2, map1.shape, mode='reflect')
    elif map1.shape < map2.shape:
        map1 = skimage.transform.resize(map1, map2.shape, mode='reflect')
    map1 = map1.flatten()
    map2 = map2.flatten()
    if 'normal' in maptype:
        valid_map1 = np.logical_or((map1 > 0), ~np.isnan(map1))
        valid_map2 = np.logical_or((map2 > 0), ~np.isnan(map2))
    elif 'grid' in maptype:
        valid_map1 = ~np.isnan(map1)
        valid_map2 = ~np.isnan(map2)
    valid = np.logical_and(valid_map1, valid_map2)
    r = np.corrcoef(map1[valid], map2[valid])
    return r[1][0]


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


def kldiv_dir(polarPlot):
    """
    Returns a kl divergence for directional firing: measure of
    directionality.
    Calculates kl diveregence between a smoothed ratemap (probably
    should be smoothed otherwise information theoretic measures
    don't 'care' about position of bins relative to
    one another) and a pure circular distribution.
    The larger the divergence the more tendancy the cell has to fire
    when the animal faces a specific direction.

    Parameters
    ----------
    polarPlot: 1D-array
        The binned and smoothed directional ratemap

    Returns
    -------
    klDivergence: float
        The divergence from circular of the 1D-array from a
        uniform circular distribution
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


def kldiv(X, pvect1, pvect2, variant=None):
    """
    Calculates the Kullback-Leibler or Jensen-Shannon divergence between
    two distributions.

    kldiv(X,P1,P2) returns the Kullback-Leibler divergence between two
    distributions specified over the M variable values in vector X.
    P1 is a length-M vector of probabilities representing distribution 1;
    P2 is a length-M vector of probabilities representing distribution 2.
        Thus, the probability of value X(i) is P1(i) for distribution 1 and
    P2(i) for distribution 2.

    The Kullback-Leibler divergence is given by:

    .. math:: KL(P1(x),P2(x)) = sum_[P1(x).log(P1(x)/P2(x))]

    If X contains duplicate values, there will be an warning message,
    and these values will be treated as distinct values.  (I.e., the
    actual values do not enter into the computation, but the probabilities
    for the two duplicate values will be considered as probabilities
    corresponding to two unique values.).
    The elements of probability vectors P1 and P2 must
    each sum to 1 +/- .00001.

    kldiv(X,P1,P2,'sym') returns a symmetric variant of the
    Kullback-Leibler divergence, given by [KL(P1,P2)+KL(P2,P1)]/2

    kldiv(X,P1,P2,'js') returns the Jensen-Shannon divergence, given by
    [KL(P1,Q)+KL(P2,Q)]/2, where Q = (P1+P2)/2.  See the Wikipedia article
    for "Kullback–Leibler divergence".  This is equal to 1/2 the so-called
    "Jeffrey divergence."

    See Also
    --------
    Cover, T.M. and J.A. Thomas. "Elements of Information Theory," Wiley,
    1991.

    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Notes
    -----
    This function is taken from one on the Mathworks file exchange
    """

    if len(np.unique(X)) != len(np.sort(X)):
        warnings.warn(
            'X contains duplicate values. Treated as distinct values.',
            UserWarning)
    if not np.equal(
        np.shape(X), np.shape(pvect1)).all() or not np.equal(
            np.shape(X), np.shape(pvect2)).all():
        raise ValueError("Inputs are not the same size")
    if (np.abs(
        np.sum(pvect1) - 1) > 0.00001) or (np.abs(
            np.sum(pvect2) - 1) > 0.00001):
        warnings.warn('Probabilities don''t sum to 1.', UserWarning)
    if variant:
        if variant == 'js':
            logqvect = np.log2((pvect2 + pvect1) / 2)
            KL = 0.5 * (np.nansum(pvect1 * (np.log2(pvect1) - logqvect)) +
                        np.sum(pvect2 * (np.log2(pvect2) - logqvect)))
            return KL
        elif variant == 'sym':
            KL1 = np.nansum(pvect1 * (np.log2(pvect1) - np.log2(pvect2)))
            KL2 = np.nansum(pvect2 * (np.log2(pvect2) - np.log2(pvect1)))
            KL = (KL1 + KL2) / 2
            return KL
        else:
            warnings.warn('Last argument not recognised', UserWarning)
    KL = np.nansum(pvect1 * (np.log2(pvect1) - np.log2(pvect2)))
    return KL


def skaggs_info(ratemap, dwelltimes, **kwargs):
    """
    Calculates Skaggs information measure

    Parameters
    ----------
    ratemap : array_like
        The binned up ratemap
    dwelltimes: array_like
        Must be same size as ratemap

    Returns
    -------
    bits_per_spike : float
        Skaggs information score

    Notes
    -----
    THIS DATA SHOULD UNDERGO ADAPTIVE BINNING
    See adaptiveBin in binning class above

    Returns Skaggs et al's estimate of spatial information
    in bits per spike:

    .. math:: I = sum_{x} p(x).r(x).log(r(x)/r)

    """
    if 'sample_rate' in kwargs:
        sample_rate = kwargs['sample_rate']
    else:
        sample_rate = 50

    dwelltimes = dwelltimes / sample_rate  # assumed sample rate of 50Hz
    if ratemap.ndim > 1:
        ratemap = np.reshape(
            ratemap, (np.prod(np.shape(ratemap)), 1))
        dwelltimes = np.reshape(
            dwelltimes, (np.prod(np.shape(dwelltimes)), 1))
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


def grid_field_props(
        A, maxima='centroid',  allProps=True,
        **kwargs):
    """
    Extracts various measures from a spatial autocorrelogram

    Parameters
    ----------
    A : array_like
        The spatial autocorrelogram (SAC)
    maxima : str, optional
        The method used to detect the peaks in the SAC.
        Legal values are 'single' and 'centroid'. Default 'centroid'
    allProps : bool, optional
        Whether to return a dictionary that contains the attempt to fit an
        ellipse around the edges of the central size peaks. See below
        Default True

    Returns
    -------
    props : dict
        A dictionary containing measures of the SAC. Keys include:
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

    See Also
    --------
    ephysiopy.common.binning.autoCorr2D()

    """
    A_tmp = A.copy()
    A_tmp[~np.isfinite(A)] = -1
    A_tmp[A_tmp <= 0] = -1
    A_sz = np.array(np.shape(A))
    # [STAGE 1] find peaks & identify 7 closest to centre
    if 'min_distance' in kwargs:
        min_distance = kwargs.pop('min_distance')
    else:
        min_distance = np.ceil(np.min(A_sz / 2) / 8.).astype(int)
    
    peak_idx, field_labels = _get_field_labels(
        A_tmp, neighbours=7, **kwargs)
    # a fcn for the labeled_comprehension function that returns
    # linear indices in A where the values in A for each label are
    # greater than half the max in that labeled region

    def fn(val, pos):
        return pos[val > (np.max(val)/2)]
    nLbls = np.max(field_labels)
    indices = ndimage.labeled_comprehension(
        A_tmp, field_labels, np.arange(0, nLbls), fn, np.ndarray, 0, True)
    # turn linear indices into coordinates
    coords = [np.unravel_index(i, np.shape(A)) for i in indices]
    half_peak_labels = np.zeros_like(A)
    for peak_id, coord in enumerate(coords):
        xc, yc = coord
        half_peak_labels[xc, yc] = peak_id

    # Get some statistics about the labeled regions
    # fieldPerim = bwperim(half_peak_labels)
    lbl_range = np.arange(0, nLbls)
    # meanRInLabel = ndimage.mean(A, half_peak_labels, lbl_range)
    # nPixelsInLabel = np.bincount(np.ravel(half_peak_labels.astype(int)))
    # sumRInLabel = ndimage.sum_labels(A, half_peak_labels, lbl_range)
    # maxRInLabel = ndimage.maximum(A, half_peak_labels, lbl_range)
    peak_coords = ndimage.maximum_position(
        A, half_peak_labels, lbl_range)

    # Get some distance and morphology measures
    centre = np.floor(np.array(np.shape(A))/2)
    centred_peak_coords = peak_coords - centre
    peak_dist_to_centre = np.hypot(
        centred_peak_coords.T[0],
        centred_peak_coords.T[1]
        )
    closest_peak_idx = np.argsort(peak_dist_to_centre)
    central_peak_label = closest_peak_idx[0]
    closest_peak_idx = closest_peak_idx[1:np.min((7, len(closest_peak_idx)-1))]
    # closest_peak_idx should now the indices of the labeled 6 peaks
    # surrounding the central peak at the image centre
    scale = np.median(peak_dist_to_centre[closest_peak_idx])
    orientation = np.nan
    orientation = grid_orientation(
        centred_peak_coords, closest_peak_idx)

    central_pt = peak_coords[central_peak_label]
    x = np.linspace(-central_pt[0], central_pt[0], A_sz[0])
    y = np.linspace(-central_pt[1], central_pt[1], A_sz[1])
    xv, yv = np.meshgrid(x, y, indexing='ij')
    dist_to_centre = np.hypot(xv, yv)
    # get the max distance of the half-peak width labeled fields
    # from the centre of the image
    max_dist_from_centre = 0
    for peak_id, _coords in enumerate(coords):
        if peak_id in closest_peak_idx:
            xc, yc = _coords
            if np.any(xc) and np.any(yc):
                xc = xc - np.floor(A_sz[0]/2)
                yc = yc - np.floor(A_sz[1]/2)
                d = np.max(np.hypot(xc, yc))
                if d > max_dist_from_centre:
                    max_dist_from_centre = d
    
    # Set the outer bits and the central region of the SAC to nans
    # getting ready for the correlation procedure
    dist_to_centre[np.abs(dist_to_centre) > max_dist_from_centre] = 0
    dist_to_centre[half_peak_labels == central_peak_label] = 0
    dist_to_centre[dist_to_centre != 0] = 1
    dist_to_centre = dist_to_centre.astype(bool)
    sac_middle = A.copy()
    sac_middle[~dist_to_centre] = np.nan

    if 'step' in kwargs.keys():
        step = kwargs.pop('step')
    else:
        step = 30
    try:
        gridscore, rotationCorrVals, rotationArr = gridness(
            sac_middle, step=step)
    except Exception:
        gridscore, rotationCorrVals, rotationArr = np.nan, np.nan, np.nan

    im_centre = central_pt

    if allProps:
        # attempt to fit an ellipse around the outer edges of the nearest
        # peaks to the centre of the SAC. First find the outer edges for
        # the closest peaks using a ndimages labeled_comprehension
        try:
            def fn2(val, pos):
                xc, yc = np.unravel_index(pos, A_sz)
                xc = xc - np.floor(A_sz[0]/2)
                yc = yc - np.floor(A_sz[1]/2)
                idx = np.argmax(np.hypot(xc, yc))
                return xc[idx], yc[idx]
            ellipse_coords = ndimage.labeled_comprehension(
                A, half_peak_labels, closest_peak_idx, fn2, tuple, 0, True)
        
            ellipse_fit_coords = np.array([(x, y) for x, y in ellipse_coords])
            from skimage.measure import EllipseModel
            E = EllipseModel()
            E.estimate(ellipse_fit_coords)
            im_centre = E.params[0:2]
            ellipse_axes = E.params[2:4]
            ellipse_angle = E.params[-1]
            ellipseXY = E.predict_xy(np.linspace(0, 2*np.pi, 50), E.params)
        
            # get the min containing circle given the eliipse minor axis
            from skimage.measure import CircleModel
            _params = im_centre
            _params.append(np.min(ellipse_axes))
            circleXY = CircleModel().predict_xy(
                np.linspace(0, 2*np.pi, 50), params=_params)
        except (TypeError, ValueError): #  non-iterable x and y i.e. ellipse coords fail
            ellipse_axes = None
            ellipse_angle = (None, None)
            ellipseXY = None
            circleXY = None
        
    # collect all the following keywords into a dict for output
    closest_peak_coords = np.array(peak_coords)[closest_peak_idx]
    dictKeys = (
        'gridscore', 'scale', 'orientation', 'closest_peak_coords',
        'dist_to_centre', 'ellipse_axes',
        'ellipse_angle', 'ellipseXY', 'circleXY', 'im_centre',
        'rotationArr', 'rotationCorrVals')
    outDict = dict.fromkeys(dictKeys, np.nan)
    for thiskey in outDict.keys():
        outDict[thiskey] = locals()[thiskey]
        # neat trick: locals is a dict holding all locally scoped variables
    return outDict


def grid_orientation(peakCoords, closestPeakIdx):
    """
    Calculates the orientation angle of a grid field.

    The orientation angle is the angle of the first peak working
    counter-clockwise from 3 o'clock

    Parameters
    ----------
    peakCoords : array_like
        The peak coordinates as pairs of xy
    closestPeakIdx : array_like
        A 1D array of the indices in peakCoords of the peaks closest
        to the centre of the SAC

    Returns
    -------
    peak_orientation : float
        The first value in an array of the angles of the peaks in the SAC
        working counter-clockwise from a line extending from the
        middle of the SAC to 3 o'clock.
    """
    if len(peakCoords) < 3 or closestPeakIdx.size == 0:
        return np.nan
    else:
        from ephysiopy.common.utils import polar
        # Assume that the first entry in peakCoords is
        # the central peak of the SAC
        peaks = peakCoords[closestPeakIdx]
        peaks = peaks - peakCoords[closestPeakIdx[0]]
        theta = polar(
            peaks[:, 1],
            -peaks[:, 0], deg=1)[1]
        return np.sort(theta.compress(theta >= 0))[0]


def gridness(image, step=30):
    """
    Calculates the gridness score in a grid cell SAC.

    Briefly, the data in `image` is rotated in `step` amounts and
    each rotated array is correlated with the original.
    The maximum of the values at 30, 90 and 150 degrees
    is the subtracted from the minimum of the values at 60, 120
    and 180 degrees to give the grid score.

    Parameters
    ----------
    image : array_like
        The spatial autocorrelogram
    step : int, optional
        The amount to rotate the SAC in each step of the rotational
        correlation procedure

    Returns
    -------
    gridmeasures : 3-tuple
        The gridscore, the correlation values at each `step` and
        the rotational array

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
    rotationalCorrVals = OrderedDict.fromkeys(
        np.arange(0, 181, step), np.nan)
    rotationArr = np.zeros(len(rotationalCorrVals)) * np.nan
    # autoCorrMiddle needs to be rescaled or the image rotation falls down
    # as values are cropped to lie between 0 and 1.0
    in_range = (np.nanmin(image), np.nanmax(image))
    out_range = (0, 1)
    import skimage
    autoCorrMiddleRescaled = skimage.exposure.rescale_intensity(
        image, in_range, out_range)
    origNanIdx = np.isnan(autoCorrMiddleRescaled.ravel())
    for idx, angle in enumerate(rotationalCorrVals.keys()):
        rotatedA = skimage.transform.rotate(
            autoCorrMiddleRescaled, angle=angle, cval=np.nan, order=3)
        # ignore nans
        rotatedNanIdx = np.isnan(rotatedA.ravel())
        allNans = np.logical_or(origNanIdx, rotatedNanIdx)
        # get the correlation between the original and rotated images
        rotationalCorrVals[angle] = stats.pearsonr(
            autoCorrMiddleRescaled.ravel()[~allNans],
            rotatedA.ravel()[~allNans])[0]
        rotationArr[idx] = rotationalCorrVals[angle]
    gridscore = np.min(
        (
            rotationalCorrVals[60],
            rotationalCorrVals[120])) - np.max(
            (
                rotationalCorrVals[150],
                rotationalCorrVals[30],
                rotationalCorrVals[90]))
    return gridscore, rotationalCorrVals, rotationArr


def deform_SAC(A, circleXY=None, ellipseXY=None):
    """
    Deforms a SAC that is non-circular to be more circular

    Basically a blatant attempt to improve grid scores, possibly
    introduced in a paper by Matt Nolan...

    Parameters
    ----------
    A : array_like
        The SAC
    circleXY : array_like
        The xy coordinates defining a circle. Default None.
    ellipseXY : array_like
        The xy coordinates defining an ellipse. Default None.

    Returns
    -------
    deformed_sac : array_like
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
        circleXY = SAC_stats['circleXY']
        ellipseXY = SAC_stats['ellipseXY']
        # The ellipse detection stuff might have failed, if so
        # return the original SAC
        if circleXY is None:
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
        AA / np.nanmax(AA.flatten()), inverse_map=tform.inverse, cval=0)
    return skimage.exposure.rescale_intensity(
        deformedSAC, out_range=(SACmin, SACmax))
