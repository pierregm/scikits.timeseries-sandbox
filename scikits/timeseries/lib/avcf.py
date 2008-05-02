"""
Auto/Cross-correlation/covariance functions for time series

:author: Pierre GF Gerard-Marchant
:contact: pierregm_at_uga_edu
:date: $Date: 2006-12-20 03:24:40 -0500 (Wed, 20 Dec 2006) $
:version: $Id: timeseries.py 55 2006-12-20 08:24:40Z backtopop $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: backtopop $)"
__version__ = '1.0'
__revision__ = "$Revision: 55 $"
__date__     = '$Date: 2006-12-20 03:24:40 -0500 (Wed, 20 Dec 2006) $'

__all__ = ['acf','acf_std','avf',
           'ccf','cvf',
           'lags',
           'pacf']

import numpy as np
from numpy import ndarray
from numpy.linalg import solve as linalgsolve

from numpy.fft import fft, ifft, rfft, irfft
from scipy.stats.distributions import norm

import numpy.ma as ma
from numpy.ma import masked_array, masked, dot

import scikits.timeseries as ts
from scikits.timeseries import TimeSeries, TimeSeriesCompatibilityError

from scipy.linalg import solve, toeplitz


#..............................................................................
"""Note about N.correlate:
Consider two series $x$ and $y$, both of size $n$.
The autocorrelation is defined as: $\gamma_{xy}[j] = \sum_{i}{x[i] y[i+j]}$,
where $j$ is the lag.
Positive lags:
N.correlate(y,x)[n//2+j] = N.correlate(x,y)[n//2-j] = gamma[j] 
"""

def lags(x, y=None):
    """Returns the lags at which a cross-correlation is computed.
    
:Parameters:
    `x` : Sequence
        First time series.
    `y` : Sequence *[None]*
        Second series. If `None`, uses `x`.
    `axis` : integer *[None]*
        Axis along which to compute (0 for rows, 1 for cols).
        If None, the array is flattened first.
    """
    nx = len(np.asarray(x))
    if y is None:
        L = np.concatenate([np.arange(nx), np.arange(-nx+1,0)])
    else:
        ny = len(np.asarray(y))
        n = min(nx,ny)
        L = np.concatenate([ np.arange(n), np.arange(-(nx+ny-n)+1,0) ])
    return L

    
#...............................................................................

def cvf(x,y,periodogram=True):
    """Computes the cross-covariance function of two series x and y.
The computations are performed on anomalies (deviations from average).
Gaps in the series are filled first, anomalies are then computed and missing
values filled with 0.
If x and y are valid TimeSeries object, they are aligned so that their starting
and ending point match.

    The crosscovariance at lag k, $\hat{R_{x,y}}(k)$, of 2 series {x_1,...,x_n} 
and {y_1,...,y_n} with mean 0 is defined as:
\hat{R_{x,y}(k) = \sum_{t=1}^{n-k}{x_t y_{t+k}} / \sum_{t=1}^{n-k}{a_t b_{t+k}}
where x_k (y_k) is set to 0 if x_k (y_k) is initially masked, where a_k = 1 if
x_k is not masked, a_k = 0 if x_k is masked, b_k = 1 if y_k is not masked and
b_k = 0 if y_k is masked.

If the optional parameter `periodogram` is True, the denominator of the previous
expression is $\sum_{t=1}^{n-k}{a_t a_{t+k}} + k$.
 
Parameters
----------
    x : sequence
        Input data. 
    y : sequence
        Input data. 
        If y is longer than x, it is truncated to match the length of x.
        If y is shorter than x, x is truncated.
    periodogram : {True, False} optional
        Whether to return a periodogram or a standard estimate of the autocovariance.
        
Returns
-------
    cvf : ma.array
        Cross-covariance at lags [0,1,...,n,n-1,...,-1]
    
    """
    #
    x = ma.array(x, copy=False, subok=True, dtype=float)
    y = ma.array(y, copy=False, subok=True, dtype=float)
    if (x.ndim > 1) or (y.ndim > 1):
        raise ValueError("Input arrays should be 1D! (got %iD-%iD)" % \
                         (x.ndim, y.ndim))
    # Make sure the series have the same size .............
    if isinstance(x, TimeSeries):
        if not isinstance(y, TimeSeries):
            raise TypeError("The second input is NOT a valid TimeSeries")
        (x,y) = ts.align_series(x,y)
    elif isinstance(y, TimeSeries) and not isinstance(x, TimeSeries):
        raise TypeError("The first input is NOT a valid TimeSeries")
    else:
        if len(y) > len(x):
            y = y[:len(x)]
        else:
            x = x[:len(y)]
    # Get the masks .......................................
    mx = np.logical_not(ma.getmaskarray(x)).astype(int)
    my = np.logical_not(ma.getmaskarray(y)).astype(int)
    # Get the anomalies ...................................
    x = x.anom().filled(0).view(ndarray)
    y = y.anom().filled(0).view(ndarray)
    n = len(x)
    cvf_ = np.correlate(x, y, 'full')
    dnm_ = np.correlate(mx, my, 'full')
    if periodogram:
        dnm_ += np.concatenate([np.arange(n-1,0,-1), np.arange(n)])
    cvf_ /= dnm_
    return ma.fix_invalid(np.concatenate([cvf_[n-1:],cvf_[:n-1]]))


def ccf(x, y, periodogram=True):    
    """Computes the auto-correlation of the series x and y at different lags.
The computations are performed on anomalies (deviations from average).
Gaps in the series are filled first, anomalies are then computed and missing
values filled with 0.
If x and y are valid TimeSeries object, they are aligned so that their starting
and ending point match.
 
Parameters
----------
    x : sequence
        Input data. 
    y : sequence
        Input data. 
        If y is longer than x, it is truncated to match the length of x.
        If y is shorter than x, x is truncated.
    periodogram : {True, False} optional
        Whether to return a periodogram or a standard estimate of the autocovariance.
        
Returns
-------
    cvf : ma.array
        Cross-correlation at lags [0,1,...,n,n-1,...,-1]
    """
    ccf_ = ccf(x,y,periodogram)
    return ma.fix_invalid(ccf_/ccf_[0])


#..............................................................................

def avf(x, periodogram=True):
    """Computes the auto-covariance function of the series `x`.
The computations are performed on anomalies (deviations from average).
Gaps in the series are filled first, anomalies are then computed and missing
values filled with 0.
    The autocovariance at lag k, $\hat{R}(k)$, of a series {x_1,...,x_n} with
mean 0 is defined as:
\hat{R}(k) = \sum_{t=1}^{n-k}{y_t y_{t+k}} / \sum_{t=1}^{n-k}{a_t a_{t+k}}
where $y_k = x_k$ if $x_k$ is not masked and $y_k = 0$ if $x_k$ is masked, and
where $a_k = 1$ if $x_k$ is not masked and $a_k = 0$ of $x_k$ is masked.
If the optional parameter `periodogram` is True, the denominator of the previous
expression is $\sum_{t=1}^{n-k}{a_t a_{t+k}} + k$.
 
Parameters
----------
    x : sequence
        Input data. If x is a TimeSeries object, it is filled first.
    mode : {True, False} optional
        Whether to return a periodogram or a standard estimate of the autocovariance.
        
Returns
-------
    avf : ma.array
        Autocovariance at lags [0,1,...,n,n-1,...,-1]
        
    """
    x = ma.array(x, copy=False, subok=True, dtype=float)
    if x.ndim > 1:
        raise ValueError("The input array should be 1D only.")
    # make sure there's no gap in the data
    if isinstance(x, TimeSeries) and x.has_missing_dates():
        x = ts.fill_missing_dates(x)
    #
    m = np.logical_not(ma.getmaskarray(x)).astype(int)
    x = x.anom().filled(0).view(ndarray)
    n = len(x)
    #
    _avf = np.correlate(x,x,'full')
    denom = np.correlate(m,m,'full')
    if periodogram:
        denom += np.concatenate([np.arange(n-1,0,-1), np.arange(n)])
    _avf /= denom
    _avf = np.concatenate([_avf[n-1:],_avf[:n-1]])
    return ma.fix_invalid(_avf)


def acf(x, periodogram=True):
    """Computes the auto-correlation of the series `x` at different lags.
The computations are performed on anomalies (deviations from average).
Gaps in the series are filled first, anomalies are then computed and missing
values filled with 0.
 
 
Parameters
----------
    x : sequence
        Input data. If x is a TimeSeries object, it is filled first.
    mode : {True, False} optional
        Whether to return a periodogram or a standard estimate of the autocorrelation.
        
Returns
-------
    acf : ma.array
        Autocorrelation at lags [0,1,...,n,n-1,...,-1]

    """
    avf_ = avf(x,periodogram)
    return avf_/avf_[0]


def _acf(x, mode):
    """Computes the auto-correlation function of the time series x.
Note that the computations are performed on anomalies (deviations from average).
Gaps in the series are filled first, the anomalies are then computed and the missing
values filled with 0.
 
:Parameters:
    `x` : TimeSeries
        Time series.
    """
    x = ma.array(x, copy=False, subok=True, dtype=float)
    if x.ndim > 1:
        raise ValueError("The input array should be 1D only.")
    # make sure there's no gap in the data
    if isinstance(x, TimeSeries) and x.has_missing_dates():
        x = ts.fill_missing_dates(x)
    #
    m = np.logical_not(ma.getmaskarray(x)).astype(int)
    x = x.anom().filled(0).view(ndarray)
    xx = (x*x)
    n = len(x)
    #
    _avf = np.correlate(x,x,'full')[n-1:]
    if mode:
        dnm_ = np.fromiter((np.sum(x[k:]*x[:-k])/np.sum(m[k:]*xx[:-k])
                            for k in range(1,n)), 
                            dtype=float)
    else:
        dnm_ = np.fromiter((np.sum(x[k:]*x[:-k])/\
                            np.sqrt((m[k:]*xx[:-k]).sum() * (m[:-k]*xx[k:]).sum())
                            for k in range(1,n)), 
                           dtype=float)
    poslags = _avf[1:]/dnm_
    return ma.fix_invalid(np.concatenate([np.array([1.]), 
                                          poslags, 
                                          poslags[::-1]]))

##..............................................................................    
def acf_std(x, maxlag=None, periodogram=True,
            confidence=0.6826895, simplified=True, acf_cached=None):
    """Computes the approximate standard deviation of the autocorrelation 
coefficients.


Parameters
----------
    x : ndarray 
        Input data.
    maxlag : {None, int} optional
        Maximum lag beyond which the ACF coefficient can be considered as null.
    periodogram : {True, False}
        Whether to use a periodogram-like estimate of the ACF or not.
    confidence : {0.6826895, float} optional 
        Confidence level. The default value returns the standard deviation.
    simplified : {True, False} optional
        Whether to use a simplified or more complex approximation.
    acf_cached : {ndarray} optional
        Pre-computed acf coefficients.
        
Notes
-----
    When simplified is True, the standard error is computed as:
\begin{equation}
var[r_k] &\appr   \frac{1}{N} \left\{ 1 + 2 \sum_{j=1}^{+q}{ r_{j}^2 } \right\
\end{equation}
    Otherwise, it is computed as:
\begin{equation}
\begin{split}
var[r_k] &\approx 
    \frac{1}{N} \sum_{j=-\infty}^{+\infty}{ \left\{
        r_{j}^2 + r_{j+k} r_{j-k} - 4 r_{k} r_{j} r_{j-k} + 2 r_{j}^2 r_{k}^2
        \right\} \\
    \frac{1}{N} \sum_{j=-\infty}^{+\infty}{ \left\{
        r_{j}^2 [ 1 + 2 r_{k}^2] + r_{j+k} r_{j-k} - 4 r_{k} r_{j} r_{j-k}
        \right\} 
\end{split}
\end{equation}

References
----------
    Hippel & McLeod 1994: Time series modeling.
    """
    if acf_cached is None:
        acfx = acf(x,periodogram)
    else:
        acfx = acf_cached
    n = x.size
    r_i = acfx[:n]
    rr_i =  (r_i)**2
    # Artifically set the ACF coefficients to 0 beyond lag maxlag
    if maxlag > 0:
        rr_i[maxlag:] = 0
    # Compute the variance of the ACF coeffs
    if simplified:
        var_i = 1 + 2*rr_i.cumsum()
    else:
        var_i = (1 + 2 * rr_i) * rr_i.sum()
        cov_ = np.correlate(r_i,r_i,'full')[n-1:]
        var_i[:n//2] = cov_[::2]
        var_i -= (4*r_i*cov_)
    var_i /= float(n)
    var_i[0] = 0
    #....
    std_i =  np.sqrt(var_i)
    std_i = np.concatenate([std_i, std_i[n-1:0:-1]])
    #....
    if confidence < 0.5:
        confidence = 1.-confidence
    thresh = norm.isf((1.-confidence)/2.)
    std_i *= thresh
    return std_i

##..............................................................................
def pacf(x, periodogram=True, lagmax=None):
    """Computes the partial autocorrelation function of series `x` along 
    the given axis.

:Parameters:
    x : 1D array
        Time series.
    periodogram : {True, False} optional
        Whether to use a periodogram-like estimate of the ACF or not.        
    lagmax : {None, int} optional
        Maximum lag. If None, the maximum lag is set to n/4+1, with n the series
        length.        
    """    
    acfx = acf(x, periodogram)[:,None]
    #
    if lagmax is None:
        n = len(x) // 4 + 1
    else:
        n = min(lagmax, len(x))
    #
    arkf = np.zeros((n,n),float)
    arkf[1,1] = acfx[1,0]
    for k in range(2,n):
        res = solve(toeplitz(acfx[:k]), acfx[1:k+1]).squeeze()
        arkf[k,1:k+1] = res
    return arkf.diagonal()
