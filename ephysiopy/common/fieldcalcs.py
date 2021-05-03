import numpy as np
from scipy import signal
from scipy import ndimage
from scipy import spatial
from scipy import stats
import skimage
import warnings
from skimage.morphology import watershed
from ephysiopy.common.utils import blurImage, bwperim

"""
These methods differ from MapCalcsGeneric in that they are mostly
concerned with treating rate maps as images as opposed to using
the spiking information contained within them. They therefore mostly
deals with spatial rate maps of place and grid cells.
"""


def getFieldLims(A):
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
    w = skimage.morphology.watershed(
        -distance, label,
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
    peak_labels = skimage.measure.label(peak_mask, 8)
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
    peak_labels = skimage.measure.label(peak_mask, 8)
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
    peak_labels = skimage.measure.label(peak_mask, 8)
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


def getBorderScore(
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


def get_field_props(
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
    if 'clear_border' in kwargs.keys():
        clear_border = True
    else:
        clear_border = False
    peak_idx = skimage.feature.peak_local_max(
        Ac, min_distance=min_dist,
        exclude_border=clear_border)
    if neighbours > len(peak_idx):
        print('neighbours value of {0} > the {1} peaks found'.format(
            neighbours, len(peak_idx)))
        print('Reducing neighbours to number of peaks found')
        neighbours = len(peak_idx)
    peak_mask = skimage.feature.peak_local_max(
        Ac, min_distance=min_dist, exclude_border=clear_border)
    peak_labels = np.zeros_like(Ac, dtype=bool)
    peak_labels[tuple(peak_mask.T)] = True
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
    _, central_field, _ = limit_to_one(A, prc=50)
    if central_field is None:
        ellipse_ratio = np.nan
    else:
        contour_coords = find_contours(central_field, 0.5)
        a = _fit_ellipse(
            contour_coords[0][:, 0], contour_coords[0][:, 1])
        ellipse_axes = _ellipse_axis_length(a)
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
    if r.any():
        return r[1][0]
    else:
        return np.nan


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


def skaggsInfo(ratemap, dwelltimes, **kwargs):
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


def getGridFieldMeasures(
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
    if 'min_distance' in kwargs.keys():
        min_distance = kwargs.pop('min_distance')
    else:
        min_distance = np.ceil(np.min(A_sz / 2) / 8.).astype(int)
    import skimage.feature
    peak_idx = skimage.feature.peak_local_max(
        A_tmp, min_distance=min_distance,
        exclude_border=False)
    peaksMask = np.zeros_like(A, dtype=bool)
    peaksMask[tuple(peak_idx.T)] = True
    import skimage
    peaksLabel = skimage.measure.label(peaksMask, connectivity=2)
    if maxima == 'centroid':
        S = skimage.measure.regionprops(peaksLabel)
        xyCoordPeaks = np.fliplr(
            np.array([(x['Centroid'][1], x['Centroid'][0]) for x in S]))
    elif maxima == 'single':
        xyCoordPeaks = np.fliplr(np.rot90(
            np.array(np.nonzero(
                peaksLabel))))  # flipped so xy instead of yx
    # Convert so the origin is at the centre of the SAC
    centralPoint = np.ceil(A_sz/2).astype(int)
    xyCoordPeaksCentral = xyCoordPeaks - centralPoint
    # calculate distance of peaks from centre and find 7 closest
    # NB one is central peak - dealt with later
    peaksDistToCentre = np.hypot(
        xyCoordPeaksCentral[:, 1], xyCoordPeaksCentral[:, 0])
    orderOfClose = np.argsort(peaksDistToCentre)
    # Get id and coordinates of closest peaks1
    # NB closest peak at index 0 will be centre
    closestPeaks = orderOfClose[0:np.min((7, len(orderOfClose)))]
    closestPeaksCoord = xyCoordPeaks[closestPeaks, :]
    closestPeaksCoord = np.floor(closestPeaksCoord).astype(int)
    # [Stage 2] Expand peak pixels into the surrounding half-height region
    # 2a find the inverse drainage bin for each peak
    fieldsLabel = watershed(image=-A_tmp, markers=peaksLabel)
    # 2b. Work out what threshold to use in each drainage-basin
    nZones = np.max(fieldsLabel.ravel())
    fieldIDs = fieldsLabel[
        closestPeaksCoord[:, 0], closestPeaksCoord[:, 1]]
    thresholds = np.ones((nZones, 1)) * np.inf
    # set thresholds for each sub-field at half-maximum
    thresholds[fieldIDs - 1, 0] = A[
        closestPeaksCoord[:, 0], closestPeaksCoord[:, 1]] / 2
    fieldsMask = np.zeros((A.shape[0], A.shape[1], nZones))
    for field in fieldIDs:
        sub = fieldsLabel == field
        fieldsMask[:, :, field-1] = np.logical_and(
            sub, A > thresholds[field-1])
        # TODO: the above step can fragment a sub-field in
        # poorly formed SACs
        # need to deal with this...perhaps by only retaining
        # the largest  sub-sub-field
        labelled_sub_field = skimage.measure.label(
            fieldsMask[:, :, field-1], connectivity=2)
        sub_props = skimage.measure.regionprops(labelled_sub_field)
        if len(sub_props) > 1:
            distFromCentre = []
            for s in range(len(sub_props)):
                centroid = sub_props[s]['Centroid']
                distFromCentre.append(
                    np.hypot(centroid[0]-A_sz[1], centroid[1]-A_sz[0]))
            idx = np.argmin(distFromCentre)
            tmp = np.zeros_like(A)
            tmp[
                sub_props[idx]['Coordinates'][:, 0],
                sub_props[idx]['Coordinates'][:, 1]] = 1
            fieldsMask[:, :, field-1] = tmp.astype(bool)
    fieldsMask = np.max(fieldsMask, 2).astype(bool)
    fieldsLabel[~fieldsMask] = 0
    fieldPerim = bwperim(fieldsMask)
    fieldsLabel = fieldsLabel.astype(int)
    # [Stage 3] Calculate a couple of metrics based on the closest peaks
    # Find the (mean) autoCorr value at the closest peak pixels
    nPixelsInLabel = np.bincount(fieldsLabel.ravel())
    sumRInLabel = np.bincount(fieldsLabel.ravel(), weights=A.ravel())
    meanRInLabel = sumRInLabel[closestPeaks+1] / nPixelsInLabel[
        closestPeaks+1]
    # get scale of grid
    closestPeakDistFromCentre = peaksDistToCentre[closestPeaks[1:]]
    scale = np.median(closestPeakDistFromCentre.ravel())
    # get orientation
    try:
        orientation = getGridOrientation(
            xyCoordPeaksCentral, closestPeaks)
    except Exception:
        orientation = np.nan
    # calculate gridness
    # THIS STEP MASKS THE MIDDLE AND OUTER PARTS OF THE SAC
    #
    # crop to the central region of the image and remove central peak
    x = np.linspace(-centralPoint[0], centralPoint[0], A_sz[0])
    y = np.linspace(-centralPoint[1], centralPoint[1], A_sz[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    dist2Centre = np.hypot(xx, yy)
    maxDistFromCentre = np.nan
    if len(closestPeaks) >= 7:
        maxDistFromCentre = np.max(dist2Centre[fieldsMask])
    if np.logical_or(
        np.isnan(
            maxDistFromCentre), maxDistFromCentre >
            np.min(np.floor(A_sz/2))):
        maxDistFromCentre = np.min(np.floor(A_sz/2))
    gridnessMaskAll = dist2Centre <= maxDistFromCentre
    centreMask = fieldsLabel == fieldsLabel[
        centralPoint[0], centralPoint[1]]
    gridnessMask = np.logical_and(gridnessMaskAll, ~centreMask)
    W = np.ceil(maxDistFromCentre).astype(int)
    autoCorrMiddle = A.copy()
    autoCorrMiddle[~gridnessMask] = np.nan
    autoCorrMiddle = autoCorrMiddle[
        -W + centralPoint[0]:W + centralPoint[0],
        -W+centralPoint[1]:W+centralPoint[1]]
    # crop the edges of the middle if there are rows/ columns of nans
    if np.any(np.all(np.isnan(autoCorrMiddle), 1)):
        autoCorrMiddle = np.delete(
            autoCorrMiddle, np.nonzero((np.all(
                np.isnan(autoCorrMiddle), 1)))[0][0], 0)
    if np.any(np.all(np.isnan(autoCorrMiddle), 0)):
        autoCorrMiddle = np.delete(
            autoCorrMiddle, np.nonzero((np.all(
                np.isnan(autoCorrMiddle), 0)))[0][0], 1)
    if 'step' in kwargs.keys():
        step = kwargs.pop('step')
    else:
        step = 30
    try:  # HWPD
        gridness, rotationCorrVals, rotationArr = getGridness(
            autoCorrMiddle, step=step)
    except Exception:  # HWPD
        gridness, rotationCorrVals, rotationArr = np.nan, np.nan, np.nan
    # attempt to fit an ellipse to the closest peaks
    if allProps:
        try:
            a = _fit_ellipse(
                closestPeaksCoord[1:, 0], closestPeaksCoord[1:, 1])
            im_centre = _ellipse_center(a)
            ellipse_axes = _ellipse_axis_length(a)
            ellipse_angle = _ellipse_angle_of_rotation(a)
#            ang =  ang + np.pi
            ellipseXY = _getellipseXY(
                ellipse_axes[0], ellipse_axes[1], ellipse_angle, im_centre)
            # get the min containing circle given the eliipse minor axis
            circleXY = _getcircleXY(
                im_centre, np.min(ellipse_axes))
        except Exception:
            im_centre = centralPoint
            ellipse_angle = None
            ellipse_axes = (None, None)
            ellipseXY = None
            circleXY = None
    else:
        ellipseXY = None
        circleXY = None
        ellipse_axes = None
        ellipse_angle = None
        im_centre = centralPoint
    # collect all the following keywords into a dict for output
    dictKeys = (
        'gridness', 'scale', 'orientation', 'closestPeaksCoord',
        'gridnessMaskAll', 'gridnessMask', 'ellipse_axes',
        'ellipse_angle', 'ellipseXY', 'circleXY', 'im_centre',
        'rotationArr', 'rotationCorrVals')
    outDict = dict.fromkeys(dictKeys, np.nan)
    for thiskey in outDict.keys():
        outDict[thiskey] = locals()[thiskey]
        # neat trick: locals is a dict holding all locally scoped variables
    return outDict


def getGridOrientation(peakCoords, closestPeakIdx):
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
    if len(closestPeakIdx) == 1:
        return np.nan
    else:
        from ephysiopy.common.utils import polar
        # Assume that the first entry in peakCoords is
        # the central peak of the SAC
        peaks = peakCoords[closestPeakIdx[1::]]
        peaks = peaks - peakCoords[closestPeakIdx[0]]
        theta = polar(
            peaks[:, 1],
            -peaks[:, 0], deg=1)[1]
        return np.sort(theta.compress(theta > 0))[0]


def getGridness(image, step=30):
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


def deformSAC(A, circleXY=None, ellipseXY=None):
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
    ephysiopy.common.ephys_generic.FieldCalcs.getGridFieldMeasures
    skimage.transform.AffineTransform
    skimage.transform.warp
    skimage.exposure.rescale_intensity
    """
    if circleXY is None or ellipseXY is None:
        SAC_stats = getGridFieldMeasures(A)
        circleXY = SAC_stats['circleXY']
        ellipseXY = SAC_stats['ellipseXY']
        # The ellipse detection stuff might have failed, if so
        # return the original SAC
        if circleXY is None:
            return A

    if circleXY.shape[0] == 2:
        circleXY = circleXY.T
    if ellipseXY.shape[0] == 2:
        ellipseXY = ellipseXY.T

    tform = skimage.transform.AffineTransform()
    try:
        tform.estimate(ellipseXY, circleXY)
    except np.linalg.LinAlgError:  # failed to converge
        print("Failed to estimate ellipse. Returning original SAC")
        return A

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


def _findPeakExtent(A, peakID, peakCoord):
    """
    Finds extent of field that belongs to each peak.

    The extent is defined as the area that falls under the half-height.

    Parameters
    ----------
    A : array_like
        The SAC
    peakID : array_like
        I think this is a list of the peak identities i.e. [1, 2, 3 etc]
    peakCoord : array_like
        xy coordinates into A that contain the full peaks

    Returns
    -------
    out : 2-tuple
        Consisting of the labelled peaks and their labelled perimeters
    """
    peakLabel = np.zeros((A.shape[0], A.shape[1]))
    perimeterLabel = np.zeros_like(peakLabel)

    # define threshold to use - currently this is half-height
    halfHeight = A[peakCoord[1], peakCoord[0]] * .5
    aboveHalfHeightLabel = ndimage.label(
        A > halfHeight, structure=np.ones((3, 3)))[0]
    peakIDTmp = aboveHalfHeightLabel[peakCoord[1], peakCoord[0]]
    peakLabel[aboveHalfHeightLabel == peakIDTmp] = peakID
    perimeterLabel[bwperim(aboveHalfHeightLabel == peakIDTmp)] = peakID
    return peakLabel, perimeterLabel


def _getcircleXY(centre, radius):
    """
    Calculates xy coordinate pairs that define a circle

    Parameters
    ----------
    centre : array_like
        The xy coordinate of the centre of the circle
    radius : int
        The radius of the circle

    Returns
    -------
    circ : array_like
        100 xy coordinate pairs that describe the circle
    """
    npts = 100
    t = np.linspace(0+(np.pi/4), (2*np.pi)+(np.pi/4), npts)
    r = np.repeat(radius, npts)
    x = r * np.cos(t) + centre[1]
    y = r * np.sin(t) + centre[0]
    return np.array((x, y))


def _getellipseXY(a, b, ang, im_centre):
    """
    Calculates xy coordinate pairs that define an ellipse

    Parameters
    ----------
    a, b : float
        The major and minor axes of the ellipse respectively
    ang : float
        The angle of orientation of the ellipse
    im_centre : array_like
        The xy coordinate of the centre of the ellipse

    Returns
    -------
    ellipse : array_like
        100 xy coordinate pairs that describe the ellipse
    """
    pts = 100
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    theta = np.linspace(0, 2*np.pi, pts)
    X = a*np.cos(theta)*cos_a - sin_a*b*np.sin(theta) + im_centre[1]
    Y = a*np.cos(theta)*sin_a + cos_a*b*np.sin(theta) + im_centre[0]
    return np.array((X, Y))


def _fit_ellipse(x, y):
    """
    Does a best fits of an ellipse to the x/y coordinates provided

    Parameters
    ----------
    x, y : array_like
        The x and y coordinates

    Returns
    -------
    a : array_like
        The xy coordinate pairs that fit
    """
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


def _ellipse_center(a):
    """
    Finds the centre of an ellipse

    Parameters
    ----------
    a : array_like
        The values that describe the ellipse; major, minor axes etc

    Returns
    -------
    xy_centre : array_like
        The xy coordinates of the centre of the ellipse
    """
    b, c, d, f, _, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0 = (c*d-b*f)/num
    y0 = (a*f-b*d)/num
    return np.array([x0, y0])


def _ellipse_angle_of_rotation(a):
    """
    Finds the angle of rotation of an ellipse

    Parameters
    ----------
    a : array_like
        The values that describe the ellipse; major, minor axes etc

    Returns
    -------
    angle : array_like
        The angle of rotation of the ellipse
    """
    b, c, _, _, _, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def _ellipse_axis_length(a):
    """
    Finds the axis length of an ellipse

    Parameters
    ----------
    a : array_like
        The values that describe the ellipse; major, minor axes etc

    Returns
    -------
    axes_length : array_like
        The length of the major and minor axes (I think)
    """
    b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    _up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1 = (b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2 = (b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1 = np.sqrt(_up/np.abs(down1))
    res2 = np.sqrt(_up/np.abs(down2))
    return np.array([res1, res2])
