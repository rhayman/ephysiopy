import numpy as np
import astropy.convolution as cnv


def get_z_score(x: np.ndarray,
                mean=None,
                sd=None,
                axis=0) -> np.ndarray:
    '''
    Calculate the z-scores for array x based on the mean
    and standard deviation in that sample, unless stated
    '''
    if mean is None:
        mean = np.mean(x, axis=axis)
    if sd is None:
        sd = np.std(x, axis=axis)
    return (x - mean) / sd


def mean_norm(x: np.ndarray, mn=None, axis=0) -> np.ndarray:
    if mn is None:
        mn = np.mean(x, axis)
    x = (x - mn) / (np.max(x, axis) - np.min(x, axis))
    return x


def min_max_norm(x: np.ndarray, min=None, max=None, axis=0) -> np.ndarray:
    if min is None:
        min = np.min(x, axis)
    if max is None:
        max = np.max(x, axis)
    return (x - min) / (max - min)


def flatten_list(list_to_flatten: list) -> list:
    return [item for sublist in list_to_flatten for item in sublist]


def smooth(x, window_len=9, window='hanning'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    Args:
        x (array_like): The input signal.
        window_len (int): The length of the smoothing window.
        window (str): The type of window from 'flat', 'hanning', 'hamming', 
            'bartlett', 'blackman'. 'flat' window will produce a moving average 
            smoothing.

    Returns:
        out (array_like): The smoothed signal.

    Example:
        >>> t=linspace(-2,2,0.1)
        >>> x=sin(t)+randn(len(t))*0.1
        >>> y=smooth(x)

    See Also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
        numpy.convolve, scipy.signal.lfilter

    Notes:
        The window parameter could be the window itself if an array instead of
        a string.
    """

    if isinstance(x, list):
        x = np.array(x)

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if len(x) < window_len:
        print("length of x: ", len(x))
        print("window_len: ", window_len)
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x

    if (window_len % 2) == 0:
        window_len = window_len + 1

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', \
                'hamming', 'bartlett', 'blackman'")

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    y = cnv.convolve(x, w/w.sum(), normalize_kernel=False, boundary='extend')
    # return the smoothed signal
    return y


def blurImage(im, n, ny=None, ftype='boxcar', **kwargs):
    """
    Smooths a 2D image by convolving with a filter.

    Args:
        im (array_like): The array to smooth.
        n, ny (int): The size of the smoothing kernel.
        ftype (str): The type of smoothing kernel.
            Either 'boxcar' or 'gaussian'.

    Returns:
        res (array_like): The smoothed vector with shape the same as im.
    """
    if 'stddev' in kwargs.keys():
        stddev = kwargs.pop('stddev')
    else:
        stddev = 5
    n = int(n)
    if not ny:
        ny = n
    else:
        ny = int(ny)
    ndims = im.ndim
    if 'box' in ftype:
        if ndims == 1:
            g = cnv.Box1DKernel(n)
        elif ndims == 2:
            g = cnv.Box2DKernel(n)
        elif ndims == 3:  # mutlidimensional binning
            g = cnv.Box2DKernel(n)
            g = np.atleast_3d(g).T
    elif 'gaussian' in ftype:
        if ndims == 1:
            g = cnv.Gaussian1DKernel(stddev, x_size=n)
        if ndims == 2:
            g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=ny)
        if ndims == 3:
            g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=ny)
            g = np.atleast_3d(g).T
    return cnv.convolve(im, g, boundary='extend')


def count_to(n):
    """
    This function is equivalent to hstack((arange(n_i) for n_i in n)).
    It seems to be faster for some possible inputs and encapsulates
    a task in a function.

    Example:
        Given n = [0, 0, 3, 0, 0, 2, 0, 2, 1],
        the result would be [0, 1, 2, 0, 1, 0, 1, 0].
    """
    if n.ndim != 1:
        raise Exception("n is supposed to be 1d array.")

    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1]+1, dtype=int)
    ret[n_cumsum[n_mask]] -= n[n_mask]
    ret[0] -= 1
    return np.cumsum(ret)[:-1]


def repeat_ind(n: np.array):
    """
    Examples:
        >>> n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        >>> res = repeat_ind(n)
        >>> res = [2, 2, 2, 5, 5, 7, 7, 8]

    The input specifies how many times to repeat the given index.
    It is equivalent to something like this:

        hstack((zeros(n_i,dtype=int)+i for i, n_i in enumerate(n)))

    But this version seems to be faster, and probably scales better.
    At any rate, it encapsulates a task in a function.
    """
    if n.ndim != 1:
        raise Exception("n is supposed to be 1d array.")

    res = [[idx]*a for idx, a in enumerate(n) if a != 0]
    return np.concatenate(res)


def rect(r, w, deg=False):
    """
    Convert from polar (r,w) to rectangular (x,y)
    x = r cos(w)
    y = r sin(w)
    """
    # radian if deg=0; degree if deg=1
    if deg:
        w = np.pi * w / 180.0
    return r * np.cos(w), r * np.sin(w)


def polar(x, y, deg=False):
    """
    Converts from rectangular coordinates to polar ones.

    Args:
        x, y (array_like, list_like): The x and y coordinates.
        deg (int): Radian if deg=0; degree if deg=1.

    Returns:
        p (array_like): The polar version of x and y.
    """
    if deg:
        return np.hypot(x, y), 180.0 * np.arctan2(y, x) / np.pi
    else:
        return np.hypot(x, y), np.arctan2(y, x)


def bwperim(bw, n=4):
    """
    Finds the perimeter of objects in binary images.

    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.

    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.

    Args:
        bw (array_like): A black-and-white image.
        n (int, optional): Connectivity. Must be 4 or 8. Default is 8.

    Returns:
        perim (array_like): A boolean image.
    """

    if n not in (4, 8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows, cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows, cols))
    south = np.zeros((rows, cols))
    west = np.zeros((rows, cols))
    east = np.zeros((rows, cols))

    north[:-1, :] = bw[1:, :]
    south[1:, :] = bw[:-1, :]
    west[:, :-1] = bw[:, 1:]
    east[:, 1:] = bw[:, :-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west == bw) & \
          (east == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:] = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:] = bw[:-1, :-1]
        south_west[1:, :-1] = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw
