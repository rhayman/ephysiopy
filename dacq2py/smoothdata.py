import numpy as np

def smooth(x,window_len=9,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    
    if type(x) == type([]):
        x = np.array(x)

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x

    if (window_len % 2) == 0:
        window_len = window_len + 1

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    from astropy.convolution import convolve
    y=convolve(x, w/w.sum(),normalize_kernel=False, boundary='extend')
    # return the smoothed signal
    return y

def adc_interp(signal, window='hanning', step_size=None):
    '''
        this function returns signal after trying to figure out the underlying
    analog signal that has been quantized (perhaps signifigantly) by an
    analog to digital converter.
        in this algorithm, only 'ambiguous' data are included in the convolution, 
    or considered in the interpolation.  To be considered ambiguous, 
    data points must be within +/- 1 quantized step_size of the actual 
    signal value.
    inputs:
        signal
        window_len=9   : the size of the window over which to average
                            "ambiguous" data.
    returns:
        interpolated_signal
    '''
    if step_size is None:
        # figure out the quantization step size.
        s = list(set(signal))
        s.sort()
        diffs = np.diff(s)
        step_size = np.median(diffs)
    
    window_lens = [129,17,11,9,5]
    step_sizes  = np.array([1.1, 1.4, 3.0, 6.5, 6.5])*step_size
    for window_len, step_size in zip(window_lens, step_sizes):
        print(window_len,step_size)
        # create a simple smooth normalized window to do weighted averages
        win = np.zeros(2*window_len, np.float64)
        win[window_len] = 1.0
        win = smooth(win, window_len=window_len, window=window)
        isig = []
        # how many points to go forward and backward
        fore = window_len/2
        aft  = window_len - fore 
        for i in xrange(len(signal)):
            ambiguous_data_pts = []
            awin = []
            for j in xrange(max(0,i-aft),min(len(signal),i+fore)):
                dj = j-i
                dwin = dj+window_len
                if abs(signal[j] - signal[i]) <= step_size*1.1:
                    ambiguous_data_pts.append(signal[j]*win[dwin])
                    awin.append(win[dwin])
            isig.append(np.sum(ambiguous_data_pts/np.sum(awin)))
        signal = np.array(isig)
    return np.array(isig)
    
    
    

