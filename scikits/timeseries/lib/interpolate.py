"""
The :mod:`scikits.timeseries.lib.interpolate` submodule contains functions for
interpolating and filling missing values in MaskedArray's and TimeSeries
objects.
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author$)"
__revision__ = "$Revision$"
__date__     = '$Date$'

import numpy as np
import numpy.ma as ma
from numpy.ma import masked, nomask, getmask, getdata
from numpy.ma.extras import flatnotmasked_edges

marray = ma.array

__all__ = ['forward_fill', 'backward_fill', 'interp_masked1d',
          ]

#####---------------------------------------------------------------------------
#---- --- Functions for filling in masked values in a masked array ---
#####---------------------------------------------------------------------------
def forward_fill(marr, maxgap=None):
    """
    Forward fills masked values in a 1-d array when there are less ``maxgap``
    consecutive masked values.

    Parameters
    ----------
    marr : MaskedArray
        Series to fill
    maxgap : {int}, optional
        Maximum gap between consecutive masked values.
        If ``maxgap`` is not specified, all masked values are forward-filled.


    Examples
    --------
    >>> x = ma.arange(20)
    >>> x[(x%5)!=0] = ma.masked
    >>> print x
    [0 -- -- -- -- 5 -- -- -- -- 10 -- -- -- -- 15 -- -- -- --]
    >>> print forward_fill(x)
    [0 0 0 0 0 5 5 5 5 5 10 10 10 10 10 15 15 15 15 15]

    """
    # !!!: We should probably port that to C.
    # Initialization ..................
    if np.ndim(marr) > 1:
        raise ValueError,"The input array should be 1D only!"
    a = ma.array(marr, copy=True)
    amask = getmask(a)
    if amask is nomask or a.size == 0:
        return a
    #
    adata = getdata(a)
    # Get the indices of the masked values (except a[0])
    idxtofill = amask[1:].nonzero()[0] + 1
    currGap = 0
    if maxgap is not None:
        previdx = -1
        for i in idxtofill:
            if i != previdx + 1:
                currGap = 0
            currGap += 1
            if currGap <= maxgap and not amask[i-1]:
                adata[i] = adata[i-1]
                amask[i] = False
                previdx = i
            else:
                amask[i-maxgap:i] = True
    else:
        for i in idxtofill:
            if not amask[i-1]:
                adata[i] = adata[i-1]
                amask[i] = False
    return a


def backward_fill(marr, maxgap=None):
    """
    Backward fills masked values in a 1-d array when there are less than ``maxgap``
    consecutive masked values.


    Parameters
    ----------
    marr : MaskedArray
        Series to fill
    maxgap : {int}, optional
        Maximum gap between consecutive masked values.
        If ``maxgap`` is not specified, all masked values are backward-filled.

    Examples
    --------
    >>> x = ma.arange(20)
    >>> x[(x%5)!=0] = ma.masked
    >>> print x
    [0 -- -- -- -- 5 -- -- -- -- 10 -- -- -- -- 15 -- -- -- --]
    >>> print backward_fill(x)
    [0 5 5 5 5 5 10 10 10 10 10 15 15 15 15 15 0 0 0 0]


    """
    return forward_fill(marr[::-1], maxgap=maxgap)[::-1]


def interp_masked1d(marr, kind='linear'):
    """

    Interpolates masked values in an array according to the given method.

    Parameters
    ----------
    marr : MaskedArray
        Array to fill
    kind : {'constant', 'linear', 'cubic', quintic'}, optional
        Type of interpolation

    """
    if np.ndim(marr) > 1:
        raise ValueError("array must be 1 dimensional!")
    #
    marr = marray(marr, copy=True)
    if getmask(marr) is nomask:
        return marr
    #
    unmaskedIndices = (~marr._mask).nonzero()[0]
    if unmaskedIndices.size < 2:
        return marr
    #
    kind = kind.lower()
    if kind == 'constant':
        return forward_fill(marr)
    try:
        k = {'linear' : 1,
             'cubic' : 3,
             'quintic' : 5}[kind.lower()]
    except KeyError:
        raise ValueError("Unsupported interpolation type.")

    first_unmasked, last_unmasked = flatnotmasked_edges(marr)

    vals = marr.data[unmaskedIndices]

    from scipy.interpolate import fitpack
    tck = fitpack.splrep(unmaskedIndices, vals, k=k)

    maskedIndices = marr._mask.nonzero()[0]
    interpIndices = maskedIndices[(maskedIndices > first_unmasked) & \
                                  (maskedIndices < last_unmasked)]
    marr[interpIndices] = fitpack.splev(interpIndices, tck).astype(marr.dtype)
    return marr
