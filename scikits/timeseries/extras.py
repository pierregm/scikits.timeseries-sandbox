"""
Extras functions for time series.

:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com
:version: $Id: extras.py 3822 2008-01-12 10:06:39Z matthew.brett@gmail.com $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: matthew.brett@gmail.com $)"
__version__ = '1.0'
__revision__ = "$Revision: 3822 $"
__date__     = '$Date: 2008-01-12 05:06:39 -0500 (Sat, 12 Jan 2008) $'


import numpy
from numpy.ma import masked

import const as _c
from tseries import TimeSeries

__all__ = ['isleapyear', 'count_missing', 'accept_atmost_missing', 'guess_freq']

#..............................................................................
def isleapyear(year):
    """Returns true if year is a leap year.

:Input:
    year : integer / sequence
        A given (list of) year(s).
    """
    year = numpy.asarray(year)
    return numpy.logical_or(year % 400 == 0,
                            numpy.logical_and(year % 4 == 0, year % 100 > 0))

#..............................................................................
def count_missing(series):
    """Returns the number of missing data per period.


Notes
-----
This function is designed to return the actual number of missing values when
a series has been converted from one frequency to a smaller frequency.

For example, converting a 12-month-long daily series to months will yield
a (12x31) array, with missing values in February, April, June...
count_missing will discard these extra missing values.
    """
    if not isinstance(series, TimeSeries):
        raise TypeError, "The input data should be a valid TimeSeries object! "\
                         "(got %s instead)" % type(series)
    if series.ndim == 1:
        return len(series) - series.count()
    elif series.ndim != 2:
        raise NotImplementedError
    #
    missing =  series.shape[-1] - series.count(axis=-1)
    period = series.shape[-1]
    freq = series.freq
    if (period == 366) and (freq//_c.FR_ANN == 1):
        # row: years, cols: days
        missing -= ~isleapyear(series.year)
    elif period == 31 and (freq//_c.FR_MTH == 1):
        months = series.months
        # row: months, cols: days
        missing[numpy.array([m in [4,6,9,11] for m in months])] -= 1
        isfeb = (months == 2)
        missing[isfeb] -= 2
        missing[isfeb & ~isleapyear(series.year)] -= 1
    elif period == 92 and (freq//_c.FR_QTR == 1):
        # row: quarters, cold:days
        months = series.months
        if freq in (_c.FR_QTREJAN, _c.FR_QTRSJAN, _c.FR_QTREAPR, _c.FR_QTRSAPR,
                    _c.FR_QTREOCT, _c.FR_QTRSOCT, _c.FR_QTREOCT, _c.FR_QTRSOCT):
            isfeb = (months == 4)
            missing[isfeb] -= 2
        elif freq in (_c.FR_QTREFEB, _c.FR_QTRSFEB, _c.FR_QTREMAY, _c.FR_QTRSMAY,
                      _c.FR_QTREAUG, _c.FR_QTRSAUG, _c.FR_QTRENOV, _c.FR_QTRSNOV):
            missing[numpy.array([m in [2,11] for m in months])] -= 1
            isfeb = (months == 2)
        elif freq in (_c.FR_QTREMAR, _c.FR_QTRSMAR, _c.FR_QTREJUN, _c.FR_QTRSJUN,
                      _c.FR_QTRESEP, _c.FR_QTRSSEP, _c.FR_QTREDEC, _c.FR_QTRSDEC):
            missing[numpy.array([m in [3,6] for m in months])] -= 1
            isfeb = (months == 3)
        missing[isfeb & ~isleapyear(series.year)] -= 1
    elif period not in (12,7):
        raise NotImplementedError, "Not yet implemented for that frequency..."
    return missing

#.............................................................................
def accept_atmost_missing(series, max_missing, strict=False):
    """Masks the rows of the series that contains more than max_missing missing data.
    Returns a new masked series.

:Inputs:
    series : TimeSeries
        Input time series.
    max_missing : float
        Number of maximum acceptable missing values per row (if larger than 1),
        or maximum acceptable percentage of missing values (if lower than 1).
    strict : boolean *[False]*
        Whether the
    """
    series = numpy.array(series, copy=True, subok=True)
    if not isinstance(series, TimeSeries):
        raise TypeError, "The input data should be a valid TimeSeries object! "\
                         "(got %s instead)" % type(series)
    # Find the number of missing values ....
    missing = count_missing(series)
    # Transform an acceptable percentage in a number
    if max_missing < 1:
        max_missing = numpy.round(max_missing * series.shape[-1],0)
    #
    series.unshare_mask()
    if strict:
        series[missing > max_missing] = masked
    else:
        series[missing >= max_missing] = masked
    return series


def guess_freq(dates):
    """Tries to estimate the frequency of a list of dates or datetime objects
by checking the steps between consecutive dates. The steps should be in days.
Returns a frequency code.
"""
    # To do: consolidate currently separate logic for dates being datetime
    # objects vs timeseries dates or ints

    if type(dates[0]) is dt.datetime:
        sorted_dates = numpy.sort(dates)
        ddif = numpy.diff(sorted_dates)
        dset = set(ddif)
        try:
            dset.remove(dt.timedelta(0))
        except:
            pass
        res = min(dset)
        if getattr(res, 'seconds', 0) >= 1:
            fcode = _c.FR_SEC
        elif getattr(res, 'seconds', 0) >= 60:
            fcode = _c.FR_MIN
        elif getattr(res, 'seconds', 0) >= 60*60:
            fcode = _c.FR_HR
        elif getattr(res, 'day', 0) >= 1:
            fcode = _c.FR_DAY           
        elif getattr(res, 'day', 0) >= 7:
            fcode = _c.FR_WK
        elif getattr(res, 'month', 0) >= 1:
            fcode = _c.FR_MTH
        elif getattr(res, 'month', 0) >= 3:
            fcode = _c.FR_QTR
        elif getattr(res, 'year', 0) >= 1:
            fcode = _c.FR_ANN
        else:
            warnings.warn("Unable to estimate the frequency! %s" % res.__str__())
            fcode = _c.FR_UND
    else:
        ddif = numeric.asarray(numpy.diff(dates))
        ddif.sort()
        if ddif.size == 0:
            fcode = _c.FR_UND
        elif ddif[0] == ddif[-1] == 1.:
            fcode = _c.FR_DAY
        elif (ddif[0] == 1.) and (ddif[-1] == 3.):
            fcode = _c.FR_BUS
        elif (ddif[0] > 3.) and  (ddif[-1] == 7.):
            fcode = _c.FR_WK
        elif (ddif[0] >= 28.) and (ddif[-1] <= 31.):
            fcode = _c.FR_MTH
        elif (ddif[0] >= 90.) and (ddif[-1] <= 92.):
            fcode = _c.FR_QTR
        elif (ddif[0] >= 365.) and (ddif[-1] <= 366.):
            fcode = _c.FR_ANN
        elif numpy.abs(24.*ddif[0] - 1) <= 1e-5 and \
             numpy.abs(24.*ddif[-1] - 1) <= 1e-5:
            fcode = _c.FR_HR
        elif numpy.abs(1440.*ddif[0] - 1) <= 1e-5 and \
             numpy.abs(1440.*ddif[-1] - 1) <= 1e-5:
            fcode = _c.FR_MIN
        elif numpy.abs(86400.*ddif[0] - 1) <= 1e-5 and \
             numpy.abs(86400.*ddif[-1] - 1) <= 1e-5:
            fcode = _c.FR_SEC
        else:
            warnings.warn("Unable to estimate the frequency! %.3f<>%.3f" %\
                          (ddif[0], ddif[-1]))
            fcode = _c.FR_UND

    return fcode