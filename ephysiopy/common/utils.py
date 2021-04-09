import numpy as np
from scipy.signal import boxcar
from scipy.signal import convolve


def smooth(x, window_len=9, window='hanning'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x : array_like
        the input signal
    window_len : int
        The length of the smoothing window
    window : str
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman'
        'flat' window will produce a moving average smoothing.

    Returns
    -------
    out : The smoothed signal

    Example
    -------
    >>> t=linspace(-2,2,0.1)
    >>> x=sin(t)+randn(len(t))*0.1
    >>> y=smooth(x)

    See Also
    --------
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array
    instead of a string
    """

    if isinstance(x, list):
        x = np.array(x)

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
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
    from astropy.convolution import convolve
    y = convolve(x, w/w.sum(), normalize_kernel=False, boundary='extend')
    # return the smoothed signal
    return y


def blur_image(im, n, ny=None, ftype='boxcar'):
    """ blurs the image by convolving with a filter ('gaussian' or
        'boxcar') of
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    n = int(n)
    if not ny:
        ny = n
    else:
        ny = int(ny)
    #  keep track of nans
    nan_idx = np.isnan(im)
    im[nan_idx] = 0
    if ftype == 'boxcar':
        if np.ndim(im) == 1:
            g = boxcar(n) / float(n)
        elif np.ndim(im) == 2:
            g = boxcar([n, ny]) / float(n)
    elif ftype == 'gaussian':
        x, y = np.mgrid[-n:n+1, -ny:ny+1]
        g = np.exp(-(x**2/float(n) + y**2/float(ny)))
        if np.ndim(im) == 1:
            g = g[n, :]
        g = g / g.sum()
    improc = convolve(im, g, mode='same')
    improc[nan_idx] = np.nan
    return improc


def count_to(self, n):
    """By example:

        #    0  1  2  3  4  5  6  7  8
        n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        res = [0, 1, 2, 0, 1, 0, 1, 0]

    That is it is equivalent to something like this :

        hstack((arange(n_i) for n_i in n))

    This version seems quite a bit faster, at least for some
    possible inputs, and at any rate it encapsulates a task
    in a function.
    """
    if n.ndim != 1:
        raise Exception("n is supposed to be 1d array.")

    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1]+1, dtype=int)
    ret[n_cumsum[n_mask]] -= n[n_mask]
    ret[0] -= 1
    return np.cumsum(ret)[:-1]


def repeat_ind(self, n):
    """
    Examples
    --------
    >>> n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
    >>> res = repeat_ind(n)
    >>> res = [2, 2, 2, 5, 5, 7, 7, 8]

    That is the input specifies how many times to repeat the given index.

    It is equivalent to something like this :

        hstack((zeros(n_i,dtype=int)+i for i, n_i in enumerate(n)))

    But this version seems to be faster, and probably scales better, at
    any rate it encapsulates a task in a function.
    """
    if n.ndim != 1:
        raise Exception("n is supposed to be 1d array.")

    n_mask = n.astype(bool)
    n_inds = np.nonzero(n_mask)[0]
    # take diff and leave 0th value in place
    n_inds[1:] = n_inds[1:]-n_inds[:-1]
    n_cumsum = np.empty(len(n)+1, dtype=int)
    n_cumsum[0] = 0
    n_cumsum[1:] = np.cumsum(n)
    ret = np.zeros(n_cumsum[-1], dtype=int)
    # note that n_mask is 1 element shorter than n_cumsum
    ret[n_cumsum[n_mask]] = n_inds
    return np.cumsum(ret)


def rect(r, w, deg=0):
    """
    Convert from polar (r,w) to rectangular (x,y)
    x = r cos(w)
    y = r sin(w)
    """
    # radian if deg=0; degree if deg=1
    if deg:
        w = np.pi * w / 180.0
    return r * np.cos(w), r * np.sin(w)


def polar(x, y, deg=0):
    """
    Converts from rectangular coordinates to polar ones

    Parameters
    ----------
    x, y : array_like, list_like
        The x and y coordinates
    deg : int
        radian if deg=0; degree if deg=1

    Returns
    -------
    p : array_like
        The polar version of x and y
    """
    if deg:
        return np.hypot(x, y), 180.0 * np.arctan2(y, x) / np.pi
    else:
        return np.hypot(x, y), np.arctan2(y, x)


def spiral(self, X, Y):
    """
    Given an array of shape X x Y this returns the coordinates needed to step
    out from the centre of the array to the edge in a spiral fashion:

    See Also
    --------
    See http://stackoverflow.com/questions/398299/looping-in-a-spiral?rq=1
    for original code and question/ solution(s)
    """
    x = 0
    y = 0
    dx = 0
    dy = -1
    x_out = []
    y_out = []
    for i in range(max(X, Y)**2):
        x_out.append(x)
        y_out.append(y)
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy

    return np.array(x_out), np.array(y_out)


def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)

    Find the perimeter of objects in binary images.

    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.

    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.

    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)

    Returns
    -------
      perim : A boolean image
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
