"""
.. currentmodule:: scikits.timeseries.lib

The :mod:`lib.moving_funcs` submodule contains an
assortment of functions for doing moving window calculations such as moving
averages, moving standard deviation, moving correlation, moving median, etc.

These functions are implemented in C and are much more efficient than what is
possible using a brute force approach in pure Python.


Moving windows
--------------

.. autosummary::
   :toctree: generated/

   mov_average
   mov_average_expw
   mov_corr
   mov_cov
   mov_max
   mov_mean
   mov_median
   mov_min
   mov_std
   mov_sum
   mov_var


Centered moving windows
-----------------------

.. autosummary::
   :toctree: generated/

   cmov_average
   cmov_mean
   cmov_window



"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author$)"
__revision__ = "$Revision$"
__date__     = '$Date$'

__all__ = ['mov_sum', 'mov_median', 'mov_min', 'mov_max',
           'mov_average', 'mov_mean', 'mov_average_expw',
           'mov_std', 'mov_var', 'mov_cov', 'mov_corr',
           'cmov_average', 'cmov_mean', 'cmov_window'
           ]

import numpy as np
from numpy import bool_, float_, sqrt
narray = np.array

import numpy.ma as ma
from numpy.ma import MaskedArray, nomask, getmask, getmaskarray, masked
marray = ma.array

from scikits.timeseries.cseries import \
    MA_mov_sum, MA_mov_median, MA_mov_min, MA_mov_max, MA_mov_average_expw


_doc_parameters = dict(
data="""data : array-like
        Input data, as a sequence or (subclass of) ndarray.
        Masked arrays and TimeSeries objects are also accepted.
        The input array should be 1D or 2D at most.
        If the input array is 2D, the function is applied on each column.""",
span="""span : int
        Size of the filtering window.""",
dtype="""dtype : dtype, optional
        Data type of the result.""",
ddof="""ddof : {0, integer}, optional
        Means Delta Degrees of Freedom.
        The divisor used in calculations of variance or standard deviation is
        ``N-ddof``.
        For biased estimates of the variance/standard deviation, use ``ddof=0``.
        For unbiased estimates, use ``ddof=1``.""",
x="""x : array-like
        First array to be included in the calculation.
        x must be a ndarray or a subclass of ndarray, such as MaskedArray or
        TimeSeries objects. In that case, the type is saved.""",
y="""y : array-like
        Second array to be included in the calculation.
        x must be a ndarray or a subclass of ndarray, such as MaskedArray or
        TimeSeries objects. In that case, the type is saved.""",
movfuncresults="""Returns
    -------
    result
        The result is always a masked array (preserves subclass attributes).
        The result at index i uses values from ``[i-span:i+1]``, and will be masked
        for the first ``span`` values.
        The result will also be masked at i if any of the input values in the slice
        ``[i-span:i+1]`` are masked.""",
movfuncexpwresults="""Returns
    -------
    result
        The result is always a masked array (preserves subclass attributes)."""
)


def _process_result_dict(orig_data, result_dict):
    "process the results from the c function"

    rarray = result_dict['array']
    rmask = result_dict.get('mask', ma.nomask)

    # makes a copy of the appropriate type
    data = orig_data.astype(rarray.dtype).copy()
    data.flat = rarray.ravel()
    if not hasattr(data, '__setmask__'):
        data = data.view(MaskedArray)
    data.__setmask__(rmask)
    return data

def _moving_func(data, cfunc, kwargs):

    data = ma.fix_invalid(data)
    data = ma.array(data.filled(0), mask=data._mask)

    if data.ndim == 1:
        kwargs['array'] = data
        result_dict = cfunc(**kwargs)
        return _process_result_dict(data, result_dict)

    elif data.ndim == 2:
        for i in range(data.shape[-1]):
            kwargs['array'] = data[:,i]
            result_dict = cfunc(**kwargs)

            if i == 0:
                rtype = result_dict['array'].dtype
                result = data.astype(rtype)
                print data.dtype, result.dtype

            rmask = result_dict.get('mask', ma.nomask)

            curr_col = marray(result_dict['array'], mask=rmask, copy=False)
            result[:,i] = curr_col

        return result

    else:
        raise ValueError, "Data should be at most 2D"

#...............................................................................
def _mov_sum(data, span, dtype=None, type_num_double=False):
    """
    Helper function for calculating moving sum.
    Resulting dtype can be determined in one of two ways.
    See C-code for more details.
    """
    kwargs = {'span':span, 'type_num_double':type_num_double}
    if dtype is not None:
        kwargs['dtype'] = dtype
    return _moving_func(data, MA_mov_sum, kwargs)



def mov_sum(data, span, dtype=None):
    """
    Calculates the moving sum of a series.

    Parameters
    ----------
    %(data)s
    %(span)s
    %(dtype)s

    %(movfuncresults)s
    """ % _doc_parameters

    return _mov_sum(data, span, dtype=dtype)



def mov_median(data, span, dtype=None):
    """
    Calculates the moving median of a series.

    Parameters
    ----------
    %(data)s
    %(span)s
    %(dtype)s

    %(movfuncresults)s
    """ % _doc_parameters

    kwargs = {'span':span}
    if dtype is not None:
        kwargs['dtype'] = dtype

    return _moving_func(data, MA_mov_median, kwargs)



def mov_min(data, span, dtype=None):
    """
    Calculates the moving minimum of a series.

    Parameters
    ----------
    %(data)s
    %(span)s
    %(dtype)s

    %(movfuncresults)s
    """ % _doc_parameters

    kwargs = {'span':span}
    if dtype is not None:
        kwargs['dtype'] = dtype

    return _moving_func(data, MA_mov_min, kwargs)



def mov_max(data, span, dtype=None):
    """
    Calculates the moving max of a series.

    Parameters
    ----------
    %(data)s
    %(span)s
    %(dtype)s

    %(movfuncresults)s
    """ % _doc_parameters

    kwargs = {'span':span}
    if dtype is not None:
        kwargs['dtype'] = dtype

    return _moving_func(data, MA_mov_max, kwargs)



def mov_average(data, span, dtype=None):
    """Calculates the moving average of a series.

    Parameters
    ----------
    %(data)s
    %(span)s
    %(dtype)s

    %(movfuncresults)s
    """ % _doc_parameters
    return _mov_sum(data, span, dtype=dtype, type_num_double=True)/span
mov_mean = mov_average



def mov_var(data, span, dtype=None, ddof=0):
    """
    Calculates the moving variance of a 1-D array.

    Parameters
    ----------
    %(data)s
    %(span)s
    %(dtype)s
    %(ddof)s

    %(movfuncresults)s
    """ % _doc_parameters
    return _mov_cov(data, data, span, ddof, dtype=dtype)



def mov_std(data, span, dtype=None, ddof=0):
    """
    Calculates the moving standard deviation of a 1-D array.

    Parameters
    ----------
    %(data)s
    %(span)s
    %(dtype)s
    %(ddof)s

    %(movfuncresults)s
    """ % _doc_parameters
    return sqrt(mov_var(data, span, dtype=dtype, ddof=ddof))



def _mov_cov(x, y, span, ddof, dtype=None):
    # helper function
    denom = span - ddof

    x = ma.asanyarray(x)
    y = ma.asanyarray(y)

    sum_prod = _mov_sum(x*y, span, dtype=dtype, type_num_double=True)
    sum_x = _mov_sum(x, span, dtype=dtype, type_num_double=True)
    sum_y = _mov_sum(y, span, dtype=dtype, type_num_double=True)

    return sum_prod/denom - (sum_x * sum_y) / (span*denom)


def mov_cov(x, y, span, bias=0, dtype=None):
    """
    Calculates the moving covariance of two 1-D arrays.

    Parameters
    ----------
    %(x)s
    %(y)s
    %(span)s
    %(dtype)s

    %(movfuncresults)s
    """ % _doc_parameters

    if bias==0:
        ddof = 1
    else:
        ddof = 0

    return _mov_cov(x, y, span, ddof, dtype=dtype)
#...............................................................................
def mov_corr(x, y, span, dtype=None):
    """
    Calculates the moving correlation of two 1-D arrays.

    Parameters
    ----------
    %(x)s
    %(y)s
    %(span)s
    %(dtype)s

    %(movfuncresults)s
    """ % _doc_parameters

    sum_x = _mov_sum(x, span, dtype=dtype, type_num_double=True)
    sum_y = _mov_sum(y, span, dtype=dtype, type_num_double=True)

    sum_prod = _mov_sum(x*y, span, dtype=dtype, type_num_double=True)
    _covar = sum_prod/span - (sum_x * sum_y) / (span ** 2)

    sum_prod = _mov_sum(x**2, span, dtype=dtype, type_num_double=True)
    _stddev_x = sqrt(sum_prod/span - (sum_x ** 2) / (span ** 2))

    sum_prod = _mov_sum(y**2, span, dtype=dtype, type_num_double=True)
    _stddev_y = sqrt(sum_prod/span - (sum_y ** 2) / (span ** 2))

    return _covar / (_stddev_x * _stddev_y)



def mov_average_expw(data, span, tol=1e-6, dtype=None):
    """
    Calculates the exponentially weighted moving average of a series.

    Parameters
    ----------
    %(data)s
    span : int
        Time periods. The smoothing factor is 2/(span + 1)
    tol : {1e-6, float}, optional
        Tolerance for the definition of the mask. When data contains masked
        values, this parameter determines what points in the result should be
        masked. Values in the result that would not be "significantly"
        impacted (as determined by this parameter) by the masked values are
        left unmasked.
    %(dtype)s

    %(movfuncexpwresults)s
    """ % _doc_parameters

    kwargs = {'span':span}
    if dtype is not None:
        kwargs['dtype'] = dtype
    result = _moving_func(data, MA_mov_average_expw, kwargs)
    mask = getattr(data, '_mask', ma.nomask)

    if mask is not ma.nomask:
        _unmasked = np.logical_not(mask).astype(float_)
        marker = 1.0 - MA_mov_average_expw(array=_unmasked, span=span)['array']
        result._mask = np.where(marker > tol, True, mask)

    return result



def cmov_window(data, span, window_type):
    """
    Applies a centered moving window of type ``window_type`` and size ``span``
    on the data.

    Parameters
    ----------
    %(data)s
    %(span)s
    window_type : {string/tuple/float}
        Window type (see Notes)

    Returns
    -------
    A (subclass of) MaskedArray.
    Noting ``k=span//2``, the ``k`` first and ``k`` last data are always masked.
    If ``data`` has a missing value at position ``i``, then the result has
    missing values in the interval ``[i-k:i+k+1]``.

    Notes
    -----
    The recognized window types are:

    * ``boxcar``
    * ``triang``
    * ``blackman``
    * ``hamming``
    * ``bartlett``
    * ``parzen``
    * ``bohman``
    * ``blackmanharris``
    * ``nuttall``
    * ``barthann``
    * ``kaiser`` (needs beta)
    * ``gaussian`` (needs std)
    * ``general_gaussian`` (needs power, width)
    * ``slepian`` (needs width).

    If the window requires special parameters, the ``window_type`` argument
    should be a tuple with the first argument the string name of the window,
    and the next arguments the needed parameters.
    If ``window_type`` is a floating point number, it is interpreted as the beta
    parameter of the ``kaiser`` window.

    Warnings
    --------
    Only ``boxcar`` has been thoroughly tested so far...

    """ % _doc_parameters

    from scipy.signal import convolve, get_window

    data = marray(data, copy=True, subok=True)
    if data._mask is nomask:
        data._mask = np.zeros(data.shape, bool_)
    window = get_window(window_type, span, fftbins=False)
    (n, k) = (len(data), span//2)
    #
    if data.ndim == 1:
        data._data.flat = convolve(data._data, window)[k:n+k] / float(span)
        data._mask[:] = ((convolve(getmaskarray(data), window) > 0)[k:n+k])
    elif data.ndim == 2:
        for i in range(data.shape[-1]):
            _data = data._data[:,i]
            _data.flat = convolve(_data, window)[k:n+k] / float(span)
            data._mask[:,i] = (convolve(data._mask[:,i], window) > 0)[k:n+k]
    else:
        raise ValueError, "Data should be at most 2D"
    data._mask[:k] = data._mask[-k:] = True
    return data



def cmov_average(data, span):
    """
    Computes the centered moving average of size ``span`` on the data.

    Parameters
    ----------
    %(data)s
    %(span)s

    Returns
    -------
    A (subclass of) MaskedArray.
    Noting ``k=span//2``, the ``k`` first and ``k`` last data are always masked.
    If ``data`` has a missing value at position ``i``, then the result has
    missing values in the interval ``[i-k:i+k+1]``.
    """ % _doc_parameters
    return cmov_window(data, span, 'boxcar')

cmov_mean = cmov_average

