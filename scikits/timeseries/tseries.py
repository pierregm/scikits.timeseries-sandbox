"""
The :class:`TimeSeries` class provides a base for the definition of time series.
A time series is defined here as the combination of two arrays:

- an array storing the time information
  (as a :class:`~scikits.timeseries.tdates.DateArray` instance);
- an array storing the data (as a :class:`MaskedArray` instance.)

These two classes were liberally adapted from :class:`MaskedArray` class.


:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com
"""

#!!!: * Allow different lengths for data and dates to handle 2D data more easily
#!!!:    In that case, just make sure that the data is (n,rows,cols) where n is the nb of dates
#!!!: * Add some kind of marker telling whether we are 1D or nD:
#!!!:    That could be done by checking the ratio series.size/series._dates.size
#!!!: * Disable some of the tests on date compatibility if we are nD
#!!!: * Adapt reshaping to preserve the first dimension: that goes for squeeze

__author__ = "Pierre GF Gerard-Marchant & Matt Knox"
__revision__ = "$Revision$"
__date__     = '$Date$'

import sys

import numpy as np
from numpy import bool_, complex_, float_, int_, object_, dtype,\
    ndarray, recarray
import numpy.core.umath as umath
from numpy.core.records import fromarrays as recfromarrays

from numpy import ma
from numpy.ma import MaskedArray, MAError, masked, nomask, \
    filled, getmask, getmaskarray, hsplit, make_mask_none, mask_or, make_mask, \
    masked_array

import tdates
from tdates import \
    DateError, FrequencyDateError, InsufficientDateError, Date, DateArray, \
    date_array, now, check_freq, check_freq_str, nodates

import const as _c
import cseries

__all__ = ['TimeSeries','TimeSeriesCompatibilityError','TimeSeriesError',
           'adjust_endpoints', 'align_series', 'align_with','aligned','asrecords',
           'compressed', 'concatenate', 'convert',
           'day','day_of_year',
           'empty_like',
           'fill_missing_dates','first_unmasked_val','flatten',
           'hour',
           'last_unmasked_val',
           'minute','month',
           'pct',
           'quarter',
           'second','split','stack',
           'time_series','tofile','tshift','masked','nomask',
           'week','weekday',
           'year',
           ]


def _unmasked_val(marray, x):
    "helper function for first_unmasked_val and last_unmasked_val"
    try:
        assert(marray.ndim == 1)
    except AssertionError:
        raise ValueError("array must have ndim == 1")

    idx = ma.extras.flatnotmasked_edges(marray)
    if idx is None:
        return masked
    return marray[idx[x]]

def first_unmasked_val(marray):
    """
    Retrieve the first unmasked value in a 1d MaskedArray.

    Parameters
    ----------
    marray : {MaskedArray}
        marray must be 1 dimensional.

    Returns
    -------
    val : {singleton of type marray.dtype}
        first unmasked value in marray. If all values in marray are masked,
        the function returns the numpy.ma.masked constant
"""
    return _unmasked_val(marray, 0)

def last_unmasked_val(marray):
    """
    Retrieve the last unmasked value in a 1d MaskedArray.

    Parameters
    ----------
    marray : {MaskedArray}
        marray must be 1 dimensional.

    Returns
    -------
    val : {singleton of type marray.dtype}
        last unmasked value in marray. If all values in marray are masked,
        the function returns the numpy.ma.masked constant
"""
    return _unmasked_val(marray, 1)

#### -------------------------------------------------------------------------
#--- ... TimeSeriesError class ...
#### -------------------------------------------------------------------------
class TimeSeriesError(Exception):
    "Class for TS related errors."
    def __init__ (self, value=None):
        "Creates an exception."
        self.value = value
    def __str__(self):
        "Calculates the string representation."
        return str(self.value)
    __repr__ = __str__


class TimeSeriesCompatibilityError(TimeSeriesError):
    """
    Defines the exception raised when series are incompatible.
    Incompatibility can arise from:

    * Inconsistent frequency;
    * Inconsistent starting dates;
    * Inconsistent size and/or shape.

    """
    def __init__(self, mode, first, second):
        if mode == 'freq':
            msg = "Incompatible time steps! (%s <> %s)"
        elif mode == 'start_date':
            msg = "Incompatible starting dates! (%s <> %s)"
        elif mode in ('size', 'shape'):
            msg = "Incompatible sizes! (%s <> %s)"
        else:
            msg = "Incompatibility !  (%s <> %s)"
        msg = msg % (first, second)
        TimeSeriesError.__init__(self, msg)

#???: Should we go crazy and add some new exceptions ?
#???: TimeSeriesShapeCompatibilityError
#???: TimeSeriesStepCompatibilityError


def _timeseriescompat(a, b, raise_error=True):
    """
    Checks the date compatibility of two TimeSeries object.
    Returns True if everything's fine, or raises an exception.
    """
    #!!!: We need to use _varshape to simplify the analysis
    # Check the frequency ..............
    (afreq, bfreq) = (getattr(a,'freq',None), getattr(b, 'freq', None))
    if afreq != bfreq:
        if raise_error:
            raise TimeSeriesCompatibilityError('freq', afreq, bfreq)
        return False
    # Make sure a.freq is not None
    if afreq is None:
        return True
    # Check the dates ...................
    (astart, bstart) = (getattr(a, 'start_date'), getattr(b, 'start_date'))
    if astart != bstart:
        if raise_error:
            raise TimeSeriesCompatibilityError('start_date', astart, bstart)
        return False
    # Check the time steps ..............
    asteps = getattr(a,'_dates',a).get_steps()
    bsteps = getattr(b,'_dates',b).get_steps()
    step_diff = (asteps != bsteps)
    if (step_diff is True) or \
       (hasattr(step_diff, "any") and step_diff.any()):
        if raise_error:
            raise TimeSeriesCompatibilityError('time_steps', asteps, bsteps)
        return False
    elif a.shape != b.shape:
        if raise_error:
            raise TimeSeriesCompatibilityError('size', "1: %s" % str(a.shape),
                                                       "2: %s" % str(b.shape))
        return False
    return True

def _timeseriescompat_multiple(*series):
    """Checks the date compatibility of multiple TimeSeries objects.
    Returns True if everything's fine, or raises an exception. Unlike
    the binary version, all items must be TimeSeries objects."""

    defsteps = series[0]._dates.get_steps()

    (freqs, start_dates, steps, shapes) = \
                                zip(*[(s.freq,
                                       s.start_date,
                                       (s._dates.get_steps() != defsteps).any(),
                                       s.shape) for s in series])
    # Check the frequencies ................
    freqset = set(freqs)
    if len(set(freqs)) > 1:
        err_items = tuple(freqset)
        raise TimeSeriesCompatibilityError('freq', err_items[0], err_items[1])
    # Check the strting dates ..............
    startset = set(start_dates)
    if len(startset) > 1:
        err_items = tuple(startset)
        raise TimeSeriesCompatibilityError('start_dates',
                                           err_items[0], err_items[1])
    # Check the steps ......................
    if max(steps) == True:
        bad_index = [x for (x, val) in enumerate(steps) if val][0]
        raise TimeSeriesCompatibilityError('time_steps',
                                           defsteps,
                                           series[bad_index]._dates.get_steps())
    # Check the shapes .....................
    shapeset = set(shapes)
    if len(shapeset) > 1:
        err_items = tuple(shapeset)
        raise TimeSeriesCompatibilityError('size',
                                           "1: %s" % str(err_items[0].shape),
                                           "2: %s" % str(err_items[1].shape))
    return True


def get_varshape(data, dates):
    """
    Checks the compatibility of dates and data.

    Parameters
    ----------
    data : array-like
        Array of data
    dates : Date, DateArray
        Sequence of dates

    Returns
    -------
    varshape : tuple
        A tuple indicating the shape of the data at any date.

    Raises
    ------
    A TimeSeriesCompatibilityError exception is raised if something goes
    wrong.

    """

    dshape = data.shape
    dates = np.array(dates, copy=False, ndmin=1)
    tshape = dates.shape
    err_args = ('shape', "data: %s" % str(dshape), "dates: %s" % str(tshape))
    # Same size: all is well
    #???: The (not dates.size) is introduced to deal with tsmasked
    if (not dates.size) or (dates.size == data.size):
        return ()
    # More dates than data: not good
    if (dates.size > data.size) or (data.ndim == 1):
        raise TimeSeriesCompatibilityError(*err_args)
    #....................
    dcumulshape = np.cumprod(dshape).tolist()
    try:
        k = dcumulshape.index(dates.size)
    except ValueError:
        raise TimeSeriesCompatibilityError(*err_args)
    else:
        return dshape[k+1:]


def _getdatalength(data):
    "Estimates the length of a series (size/nb of variables)."
    if np.ndim(data) >= 2:
        return np.asarray(np.shape(data))[:-1].prod()
    else:
        return np.size(data)

def _compare_frequencies(*series):
    """Compares the frequencies of a sequence of series.

Returns the common frequency, or raises an exception if series have different
frequencies.
"""
    unique_freqs = np.unique([x.freqstr for x in series])
    try:
        common_freq = unique_freqs.item()
    except ValueError:
        raise TimeSeriesError, \
            "All series must have same frequency! (got %s instead)" % \
            unique_freqs
    return common_freq



##### ------------------------------------------------------------------------
##--- ... Time Series ...
##### ------------------------------------------------------------------------
class _tsmathmethod(object):
    """
    Defines a wrapper for arithmetic array methods (add, mul...).
    When called, returns a new TimeSeries object, with the new series the result
    of the method applied on the original series. The `_dates` part remains
    unchanged.
    """
    def __init__ (self, methodname):
        self.__name__ = methodname
        self.__doc__ = getattr(MaskedArray, methodname).__doc__
        self.obj = None

    def __get__(self, obj, objtype=None):
        "Gets the calling object."
        self.obj = obj
        return self

    def __call__ (self, other, *args):
        "Execute the call behavior."
        instance = self.obj
        if isinstance(other, TimeSeries):
            compat = _timeseriescompat(instance, other, raise_error=False)
        else:
            compat = True
        func = getattr(super(TimeSeries, instance), self.__name__)
        if compat:
            result = np.array(func(other, *args), subok=True).view(type(instance))
            result._dates = instance._dates
        else:
            other_ = getattr(other, '_series', other)
            result_ = func(other_, *args)
            result = getattr(result_, '_series', result_)
        return result


class _tsarraymethod(object):
    """
    Defines a wrapper for basic array methods.
    When called, returns a new TimeSeries object, with the new series the result
    of the method applied on the original series.
    If `ondates` is True, the same operation is performed on the `_dates`.
    If `ondates` is False, the `_dates` part remains unchanged.
    """
    def __init__ (self, methodname, ondates=False):
        self.__name__ = methodname
        self.__doc__ = getattr(MaskedArray, methodname).__doc__
        self._ondates = ondates
        self.obj = None

    def __get__(self, obj, objtype=None):
        self.obj = obj
        return self

    def __call__ (self, *args, **kwargs):
        "Execute the call behavior."
        _name = self.__name__
        instance = self.obj
        _series = ndarray.__getattribute__(instance, '_series')
        _dates = ndarray.__getattribute__(instance, '_dates')
        func_series = getattr(_series, _name)
        result = func_series(*args, **kwargs).view(type(instance))
        if self._ondates:
            result._dates = getattr(_dates, _name)(*args, **kwargs)
        else:
            result._dates = _dates
        return result


class _tsaxismethod(object):
    """
    Defines a wrapper for array methods working on an axis (mean...).

    When called, returns a ndarray, as the result of the method applied on the
    series.
"""
    def __init__ (self, methodname):
        """abfunc(fillx, filly) must be defined.
           abinop(x, filly) = x for all x to enable reduce.
        """
        self.__name__ = methodname
        self.__doc__ = getattr(MaskedArray, methodname).__doc__
        self.obj = None

    def __get__(self, obj, objtype=None):
        self.obj = obj
        return self

    def __call__ (self, *args, **params):
        "Execute the call behavior."
        (_dates, _series) = (self.obj._dates, self.obj._series)
        func = getattr(_series, self.__name__)
        result = func(*args, **params)
        if _dates.size == _series.size:
            return result
        else:
            try:
                axis = params.get('axis', args[0])
                if axis in [-1, _series.ndim-1]:
                    result = result.view(type(self.obj))
                    result._dates = _dates
            except IndexError:
                pass
            return result




class TimeSeries(MaskedArray, object):
    """
    Base class for the definition of time series.

    Parameters
    ----------
    data : {array_like}
        Data portion of the array.
        Any data that is valid for constructing a MaskedArray can be used here.
    dates : {DateArray}
        A `DateArray` instance.
    **optional_parameters:
        All the parameters recognized by `MaskedArray` are also recognized by
        TimeSeries.

    Notes
    -----
    It is recommended to use the :func:`time_series` function for construction,
    as it is more flexible and convenient.

    See Also
    --------
    numpy.ma.MaskedArray
        ndarray with support for missing data.
    scikits.timeseries.DateArray
    """
    def __new__(cls, data, dates, mask=nomask, dtype=None, copy=False,
                fill_value=None, subok=True, keep_mask=True, hard_mask=False,
                **options):

        maparms = dict(copy=copy, dtype=dtype, fill_value=fill_value,
                       subok=subok, keep_mask=keep_mask, hard_mask=hard_mask)
        _data = MaskedArray.__new__(cls, data, mask=mask, **maparms)

        # Get the data .......................................................
        if not subok or not isinstance(_data,TimeSeries):
            _data = _data.view(cls)
        if _data is masked:
            assert(np.size(dates)==1)
            return _data.view(cls)
        # Check that the dates and data are compatible in shape.
        _data._varshape = get_varshape(_data,dates)
        # Set the dates
        _data._dates = dates
        # Make sure the data is properly sorted.
        #!!!: WE SHOULD TEST THAT MORE
        if dates._unsorted is not None:
            idx = dates._unsorted
            _data = _data[idx]
            _data._dates._unsorted = None
        return _data

    def __array_finalize__(self,obj):
        self._varshape = getattr(obj, '_varshape', ())
        MaskedArray.__array_finalize__(self, obj)

    def _update_from(self, obj):
        _dates = getattr(self, '_dates', nodates)
        newdates = getattr(obj, '_dates', nodates)
        # Only update the dates if we don't have any
        if not getattr(_dates, 'size', 0):
            self.__setdates__(newdates)
        MaskedArray._update_from(self, obj)

    def view(self, dtype=None, type=None):
        try:
            output = MaskedArray.view(self, dtype=dtype, type=type)
        except ValueError:
            output = MaskedArray.view(self, dtype)
        if isinstance(output, TimeSeries):
            if output.dtype.fields:
                output._varshape = ()
            else:
                fields = self.dtype.fields
                if fields:
                    output._varshape = (len(fields),)
        return output

    def _get_series(self):
        "Returns the series as a regular masked array."
        if self._mask.ndim == 0 and self._mask:
            return masked
        return self.view(MaskedArray)
    _series = property(fget=_get_series)

    def _index_checker(self, indx):
        if isinstance(indx, int):
            return (indx, indx, False)
        _dates = self._dates
        if isinstance(indx, basestring):
            if indx in (self.dtype.names or ()):
                return (indx, slice(None, None, None), False)
            try:
                indx = _dates.date_to_index(Date(_dates.freq, string=indx))
            except IndexError:
                # Trap the exception: we need the traceback
                exc_info = sys.exc_info()
                msg = "Invalid field or date '%s'" % indx
                raise IndexError(msg), None, exc_info[2]
            return (indx, indx, False)
        if isinstance(indx, (Date, DateArray)):
            indx = self._dates.date_to_index(indx)
            return (indx, indx, False)
        if isinstance(indx, slice):
            indx = slice(self._slicebound_checker(indx.start),
                         self._slicebound_checker(indx.stop),
                         indx.step)
            return (indx, indx, False)

        if isinstance(indx, tuple):
            if not self._varshape:
                return (indx, indx, False)
            else:
                return (indx, indx[0], False)
        return (indx, indx, True)

    def _slicebound_checker(self, bound):

        if bound is None or isinstance(bound, int):
            return bound

        _dates = self._dates
        if isinstance(bound, str):
            bound = Date(_dates.freq, string=bound)
        if not isinstance(bound, Date):
            raise ValueError(
                "invalid object used in slice: %s" % repr(bound))
        if bound.freq != _dates.freq:
            raise TimeSeriesCompatibilityError('freq',
                                               _dates.freq, bound.freq)
        # this allows for slicing with dates outside the end points of the
        # series and slicing on series with missing dates
        return np.sum(self._dates < bound)

    def __getitem__(self, indx):
        """x.__getitem__(y) <==> x[y]
Returns the item described by i. Not a copy.
        """
        (sindx, dindx, recheck) = self._index_checker(indx)
        _data = ndarray.__getattribute__(self, '_data')
        _mask = ndarray.__getattribute__(self, '_mask')
        _dates = ndarray.__getattribute__(self, '_dates')
        try:
            output = _data.__getitem__(sindx)
        except IndexError:
            # We don't need to recheck the index: just raise an exception
            if not recheck:
                raise
            # Maybe the index is a list of Dates ?
            try:
                indx = _dates.date_to_index(indx)
            except (IndexError, ValueError):
                # Mmh, is it a list of dates as strings ?
                try:
                    indx = _dates.date_to_index(date_array(indx,
                                                           freq=_dates.freq))
                except (IndexError, ValueError, DateError):
                    exc_info = sys.exc_info()
                    msg = "Invalid index or date '%s'" % indx
                    raise IndexError(msg), None, exc_info[2]
                else:
                    output = _data.__getitem__(indx)
                    sindx = dindx = indx
            else:
                output = _data.__getitem__(indx)
                sindx = dindx = indx
        # Don't find the date if it's not needed......
        if not getattr(output,'ndim', False):
            # A record ................
            if isinstance(output, np.void):
                mask = _mask[sindx]
                if mask.view((bool,len(mask.dtype))).any():
                    output = masked_array(output, mask=mask)
                else:
                    return output
            elif _mask is not nomask and _mask[sindx]:
                return masked
            return output
        # Get the date................................
        newdates = _dates.__getitem__(dindx)
        if not getattr(newdates, 'shape', 0):
            # No dates ? Output a MaskedArray
            newseries = output.view(MaskedArray)
        else:
            # Some dates: output a TimeSeries
            newseries = output.view(type(self))
            newseries._dates = newdates
        # Update some info from self (fill_value, _basedict...)
        MaskedArray._update_from(newseries, self)
        # Fix the fill_value if we were accessing a field of a flexible array
        if isinstance(sindx, basestring):
            _fv = self._fill_value
            if _fv is not None and not np.isscalar(_fv):
                 newseries._fill_value = _fv[sindx]
            newseries._isfield = True
        # Fix the mask
        if _mask is not nomask:
            newseries._mask = _mask[sindx]
            newseries._sharedmask = True
        return newseries

    def __setitem__(self, indx, value):
        """x.__setitem__(i, y) <==> x[i]=y
Sets item described by index. If value is masked, masks those locations.
"""
        (sindx, dindx, recheck) = self._index_checker(indx)
        _dates = ndarray.__getattribute__(self, '_dates')
        try:
            MaskedArray.__setitem__(self, sindx, value)
        except IndexError:
            # We don't need to recheck the index: just raise an exception
            if not recheck:
                raise
            # Maybe the index is a list of Dates ?
            try:
                indx = _dates.date_to_index(indx)
            except (IndexError, ValueError):
                # Mmh, is it a list of dates as strings ?
                try:
                    indx = _dates.date_to_index(date_array(indx,
                                                           freq=_dates.freq))
                except (IndexError, ValueError, DateError):
                    exc_info = sys.exc_info()
                    msg = "Invalid index or date '%s'" % indx
                    raise IndexError(msg), None, exc_info[2]
                else:
                    MaskedArray.__setitem__(self, indx, value)
            else:
                MaskedArray.__setitem__(self, indx, value)

    def __setattr__(self, attr, value):
        if attr in ['_dates','dates']:
            return self.__setdates__(value)
        elif attr == 'shape':
            if self._varshape:
                err_msg = "Reshaping a nV/nD series is not implemented yet !"
                raise NotImplementedError(err_msg)
            else:
                self._dates.shape = value
        return ndarray.__setattr__(self, attr, value)


    def __setdates__(self, value):
        # Make sure it's a DateArray
        if not isinstance(value, (Date, DateArray)):
            err_msg = "The input dates should be a valid Date or "\
                      "DateArray object (got %s instead)" % type(value)
            raise TypeError(err_msg)
        # Skip if dates is nodates (or empty)\
        if value is nodates or not getattr(value,'size',0):
            return super(TimeSeries, self).__setattr__('_dates', value)
        # Make sure it has the proper size
        tsize = getattr(value, 'size', 1)
        # Check the _varshape
        varshape = self._varshape
        if not varshape:
            # We may be using the default: retry
            varshape = self._varshape = get_varshape(self, value)
        # Get the data length (independently of the nb of variables)
        dsize = self.size // int(np.prod(varshape))
        if tsize != dsize:
            raise TimeSeriesCompatibilityError("size",
                                               "data: %s" % dsize,
                                               "dates: %s" % tsize)
        elif not varshape:
            # The data is 1D
            value = value.reshape(self.shape)
#            value.shape = self.shape
        return super(TimeSeries, self).__setattr__('_dates', value)

    dates = property(fget=lambda self:self._dates,
                     fset=__setdates__)

    #......................................................
    def __str__(self):
        """Returns a string representation of self (w/o the dates...)"""
        return str(self._series)
    def __repr__(self):
        """Calculates the repr representation, using masked for fill if it is
enabled. Otherwise fill with fill value.
"""
        desc = """\
timeseries(
 %(data)s,
           dates =
 %(time)s,
           freq  = %(freq)s)
"""
        desc_short = """\
timeseries(%(data)s,
           dates = %(time)s,
           freq  = %(freq)s)
"""
        if np.size(self._dates) > 2 and self.isvalid():
            timestr = "[%s ... %s]" % (str(self._dates[0]),str(self._dates[-1]))
        else:
            timestr = str(self.dates)

        if self.ndim <= 1:
            return desc_short % {'data': str(self._series),
                                 'time': timestr,
                                 'freq': self.freqstr, }
        return desc % {'data': str(self._series),
                       'time': timestr,
                       'freq': self.freqstr, }
    #............................................
    __add__ = _tsmathmethod('__add__')
    __radd__ = _tsmathmethod('__add__')
    __sub__ = _tsmathmethod('__sub__')
    __rsub__ = _tsmathmethod('__rsub__')
    __pow__ = _tsmathmethod('__pow__')
    __mul__ = _tsmathmethod('__mul__')
    __rmul__ = _tsmathmethod('__mul__')
    __div__ = _tsmathmethod('__div__')
    __rdiv__ = _tsmathmethod('__rdiv__')
    __truediv__ = _tsmathmethod('__truediv__')
    __rtruediv__ = _tsmathmethod('__rtruediv__')
    __floordiv__ = _tsmathmethod('__floordiv__')
    __rfloordiv__ = _tsmathmethod('__rfloordiv__')
    __eq__ = _tsmathmethod('__eq__')
    __ne__ = _tsmathmethod('__ne__')
    __lt__ = _tsmathmethod('__lt__')
    __le__ = _tsmathmethod('__le__')
    __gt__ = _tsmathmethod('__gt__')
    __ge__ = _tsmathmethod('__ge__')

    copy = _tsarraymethod('copy', ondates=True)
    compress = _tsarraymethod('compress', ondates=True)
    ravel = _tsarraymethod('ravel', ondates=True)
    cumsum = _tsarraymethod('cumsum',ondates=False)
    cumprod = _tsarraymethod('cumprod',ondates=False)
    anom = _tsarraymethod('anom',ondates=False)

    sum = _tsaxismethod('sum')
    prod = _tsaxismethod('prod')
    mean = _tsaxismethod('mean')
    var = _tsaxismethod('var')
    std = _tsaxismethod('std')
    all = _tsaxismethod('all')
    any = _tsaxismethod('any')


    def reshape(self, *newshape, **kwargs):
        """

    Returns a time series containing the data of a, but with a new shape.
    The result is a view to the original array; if this is not possible,
    a ValueError is raised.

    Parameters
    ----------
    shape : shape tuple or int
       The new shape should be compatible with the original shape. If an
       integer, then the result will be a 1D array of that length.
    order : {'C', 'F'}, optional
        Determines whether the array data should be viewed as in C
        (row-major) order or FORTRAN (column-major) order.

    Returns
    -------
    reshaped_array : array
        A new view to the timeseries.

    Warnings
    --------
    The `._dates` part is reshaped, but the order is NOT ensured.

        """
        # 1D series : reshape the dates as well
        if not self._varshape:
            kwargs.update(order=kwargs.get('order','C'))
            result = MaskedArray.reshape(self, *newshape, **kwargs)
            result._dates = ndarray.reshape(self._dates, *newshape, **kwargs)
        # nV/nD series: raise an exception for now (
        else:
            err_msg = "Reshaping a nV/nD series is not implemented yet !"
            raise NotImplementedError(err_msg)
            #!!!: We could also not do anything...
        return result

    #.........................................................................
    def ids (self):
        """Return the ids of the data, dates and mask areas"""
        return (id(self._series), id(self.dates),)
    #.........................................................................
    @property
    def series(self):
        """Returns the series."""
        return self._series
    @property
    def freq(self):
        """Returns the corresponding frequency (as an integer)."""
        return self._dates.freq
    @property
    def freqstr(self):
        """Returns the corresponding frequency (as a string)."""
        return self._dates.freqstr
    @property
    def day(self):
        """Returns the day of month for each date in self._dates."""
        return self._dates.day
    @property
    def weekday(self):
        """Returns the day of week for each date in self._dates."""
        return self._dates.weekday
    @property
    def day_of_year(self):
        """Returns the day of year for each date in self._dates."""
        return self._dates.day_of_year
    @property
    def month(self):
        """Returns the month for each date in self._dates."""
        return self._dates.month
    @property
    def quarter(self):
        """Returns the quarter for each date in self._dates."""
        return self._dates.quarter
    @property
    def year(self):
        """Returns the year for each date in self._dates."""
        return self._dates.year
    @property
    def second(self):
        """Returns the second for each date in self._dates."""
        return self._dates.second
    @property
    def minute(self):
        """Returns the minute for each date in self._dates."""
        return self._dates.minute
    @property
    def hour(self):
        """Returns the hour for each date in self._dates."""
        return self._dates.hour
    @property
    def week(self):
        """Returns the week for each date in self._dates."""
        return self._dates.week

    days = day
    weekdays = weekday
    yeardays = day_of_year
    months = month
    quarters = quarter
    years = year
    seconds = second
    minutes = minute
    hours = hour
    weeks = week

    @property
    def start_date(self):
        """Returns the first date of the series."""
        _dates = self._dates
        dsize = _dates.size
        if dsize == 0:
            return None
        elif dsize == 1:
            return _dates[0]
        else:
            return Date(self.freq, _dates.flat[0])
    @property
    def end_date(self):
        """Returns the last date of the series."""
        _dates = self._dates
        dsize = _dates.size
        if dsize == 0:
            return None
        elif dsize == 1:
            return _dates[-1]
        else:
            return Date(self.freq, _dates.flat[-1])

    def isvalid(self):
        """Returns whether the series has no duplicate/missing dates."""
        return self._dates.isvalid()

    def has_missing_dates(self):
        """Returns whether there's a date gap in the series."""
        return self._dates.has_missing_dates()

    def isfull(self):
        """Returns whether there's no date gap in the series."""
        return self._dates.isfull()

    def has_duplicated_dates(self):
        """Returns whether there are duplicated dates in the series."""
        return self._dates.has_duplicated_dates()

    def date_to_index(self, date):
        """Returns the index corresponding to a given date, as an integer."""
        return self._dates.date_to_index(date)
    #.....................................................
    def asfreq(self, freq, relation="END"):
        """
    Converts the dates portion of the TimeSeries to another frequency.

    The resulting TimeSeries will have the same shape and dimensions
    as the original series (unlike the :meth:`convert` method).

    Parameters
    ----------
    freq : {freq_spec}
    relation : {'END', 'START'} (optional)

    Returns
    -------
    A new TimeSeries with the :attr:`.dates` :class:`DateArray` at the
    specified frequency (the :meth`.asfreq` method of the :attr:`.dates`
    property will be called).
    The data in the resulting series will be a VIEW of the original series.

    Notes
    -----
    The parameters are the exact same as for
    :meth:`~scikit.timeseries.DateArray.asfreq`. Please see the docstring for
    that method for details on the parameters and how the actual conversion is
    performed.

    """
        if freq is None: return self

        return TimeSeries(self._series,
                          dates=self._dates.asfreq(freq, relation=relation))
    #.....................................................
    def transpose(self, *axes):
        if self._dates.size == self.size:
            result = MaskedArray.transpose(self, *axes)
            result._dates = self._dates.transpose(*axes)
        else:
            errmsg = "Operation not permitted on multi-variable series"
            if (len(axes)==0) or axes[0] != 0:
                raise TimeSeriesError, errmsg
            else:
                result = MaskedArray.transpose(self, *axes)
                result._dates = self._dates
        return result
    transpose.__doc__ = np.transpose.__doc__

    def split(self):
        """Split a multi-dimensional series into individual columns."""
        if self.ndim == 1:
            return [self]
        else:
            n = self.shape[1]
            arr = hsplit(self, n)[0]
            return [self.__class__(np.squeeze(a),
                                   self._dates,
                                   **_attrib_dict(self)) for a in arr]

    def filled(self, fill_value=None):
        """
    Returns an array of the same class as `_data`,  with masked values
    filled with `fill_value`. Subclassing is preserved.

    Parameters
    ----------
    fill_value : {None, singleton of type self.dtype}, optional
        The value to fill in masked values with.
        If `fill_value` is None, uses ``self.fill_value``.

    """
        result = self._series.filled(fill_value=fill_value).view(type(self))
        result._dates = self._dates
        return result

    def tolist(self):
        """
    Returns the dates and data portion of the TimeSeries "zipped" up in
    a list of standard python objects (eg. datetime, int, etc...).

    """
        if self.ndim > 0:
            return zip(self.dates.tolist(), self.series.tolist())
        else:
            return self.series.tolist()

    #......................................................
    # Pickling
    def __getstate__(self):
        """

    Returns the internal state of the TimeSeries, for pickling purposes.
        """
    #    raise NotImplementedError,"Please use timeseries.archive/unarchive instead."""
        state = (1,
                 self.shape,
                 self.dtype,
                 self.flags.fnc,
                 self._data.tostring(),
                 getmaskarray(self).tostring(),
                 self._fill_value,
                 self._dates.shape,
                 np.asarray(self._dates).tostring(),
                 self.freq,
                 )
        return state
    #
    def __setstate__(self, state):
        """

    Restores the internal state of the TimeSeries, for pickling purposes.
    `state` is typically the output of the ``__getstate__`` output, and is a 5-tuple:

        - class name
        - a tuple giving the shape of the data
        - a typecode for the data
        - a binary string for the data
        - a binary string for the mask.
        """
        (ver, shp, typ, isf, raw, msk, flv, dsh, dtm, frq) = state
        MaskedArray.__setstate__(self, (ver, shp, typ, isf, raw, msk, flv))
        _dates = self._dates
        _dates.__setstate__((dsh, dtype(int_), isf, dtm))
        _dates.freq = frq
        _dates._cachedinfo.update(dict(full=None, hasdups=None, steps=None,
                                       toobj=None, toord=None, tostr=None))
#
    def __reduce__(self):
        """Returns a 3-tuple for pickling a MaskedArray."""
        return (_tsreconstruct,
                (self.__class__, self._baseclass,
                 self.shape, self._dates.shape, self.dtype, self._fill_value),
                self.__getstate__())

def _tsreconstruct(genclass, baseclass, baseshape, dateshape, basetype, fill_value):
    """Internal function that builds a new TimeSeries from the information stored
    in a pickle."""
    #    raise NotImplementedError,"Please use timeseries.archive/unarchive instead."""
    _series = ndarray.__new__(baseclass, baseshape, basetype)
    _dates = ndarray.__new__(DateArray, dateshape, int_)
    _mask = ndarray.__new__(ndarray, baseshape, bool_)
    return genclass.__new__(genclass, _series, dates=_dates, mask=_mask,
                            dtype=basetype, fill_value=fill_value)

def _attrib_dict(series, exclude=[]):
    """this function is used for passing through attributes of one
time series to a new one being created"""
    result = {'fill_value':series.fill_value}
    return dict(filter(lambda x: x[0] not in exclude, result.iteritems()))


##### --------------------------------------------------------------------------
##--- ... Additional methods ...
##### --------------------------------------------------------------------------

def _extrema(self, method, axis=None,fill_value=None):
    "Private function used by max/min"
    (_series, _dates) = (self._series, self._dates)
    func = getattr(_series, method)
    idx = func(axis,fill_value)
    # 1D series .......................
    if (_dates.size == _series.size):
        if axis is None:
            return self.ravel()[idx]
        else:
            return self[idx]
    # nD series .......................
    else:
        if axis is None:
            idces = np.unravel_index(idx, _series.shape)
            result = time_series(_series[idces], dates=_dates[idces[0]])
        else:
            _shape = _series.shape
            _dates = np.repeat(_dates,np.prod(_shape[1:])).reshape(_shape)
            _s = np.rollaxis(_series,axis,0)[idx]
            _d = np.rollaxis(_dates,axis,0)[idx]
            _s = np.choose(idx, np.rollaxis(_series,axis,0))
            _d = np.choose(idx, np.rollaxis(_dates,axis,0))
            result = time_series(_s, dates=_d)
        return result

def _max(self, axis=None, fill_value=None):
    """Return the maximum of self along the given axis.
    Masked values are filled with fill_value.

    Parameters
    ----------
    axis : int, optional
        Axis along which to perform the operation.
        If None, applies to a flattened version of the array.
    fill_value : {var}, optional
        Value used to fill in the masked values.
        If None, use the the output of maximum_fill_value().
    """
    return _extrema(self,'argmax',axis,fill_value)
TimeSeries.max = _max

def _min(self,axis=None,fill_value=None):
    """Return the minimum of self along the given axis.
    Masked values are filled with fill_value.

    Parameters
    ----------
    axis : int, optional
        Axis along which to perform the operation.
        If None, applies to a flattened version of the array.
    fill_value : {var}, optional
        Value used to fill in the masked values.
        If None, use the the output of minimum_fill_value().
    """
    return _extrema(self, 'argmin',axis,fill_value)
TimeSeries.min = _min


#.......................................


class _tsblockedmethods(object):
    """Defines a wrapper for array methods that should be temporarily disabled.
    """
    def __init__ (self, methodname):
        """abfunc(fillx, filly) must be defined.
           abinop(x, filly) = x for all x to enable reduce.
        """
        self._name = methodname
        self.obj = None
    #
    def __get__(self, obj, objtype=None):
        self.obj = obj
        return self
    #
    def __call__ (self, *args, **params):
        raise NotImplementedError

TimeSeries.swapaxes = _tsarraymethod('swapaxes', ondates=True)

#####---------------------------------------------------------------------------
#---- --- Definition of functions from the corresponding methods ---
#####---------------------------------------------------------------------------
class _frommethod(object):
    """Defines functions from existing MaskedArray methods.
:ivar _methodname (String): Name of the method to transform.
    """
    def __init__(self, methodname):
        self.__name__ = methodname
        self.__doc__ = self.getdoc()
    def getdoc(self):
        "Returns the doc of the function (from the doc of the method)."
        try:
            return getattr(TimeSeries, self.__name__).__doc__
        except:
            return "???"
    #
    def __call__ (self, caller, *args, **params):
        if hasattr(caller, self.__name__):
            method = getattr(caller, self.__name__)
            # If method is not callable, it's a property, and don't call it
            if hasattr(method, '__call__'):
                return method.__call__(*args, **params)
            return method
        method = getattr(np.asarray(caller), self.__name__)
        try:
            return method(*args, **params)
        except SystemError:
            return getattr(np,self.__name__).__call__(caller, *args, **params)
#............................
weekday = _frommethod('weekday')
day_of_year = _frommethod('day_of_year')
week = _frommethod('week')
year = _frommethod('year')
quarter = _frommethod('quarter')
month = _frommethod('month')
day = _frommethod('day')
hour = _frommethod('hour')
minute = _frommethod('minute')
second = _frommethod('second')

split = _frommethod('split')

#
##### ---------------------------------------------------------------------------
#---- ... Additional methods ...
##### ---------------------------------------------------------------------------
def tofile(series, fileobject, format=None,
           separator=" ", linesep='\n', precision=5,
           suppress_small=False, keep_open=False):
    """
    Writes the TimeSeries to a file. The series should be 2D at most.

    Parameters
    ----------
    series : TimeSeries
        The array to write.
    fileobject
        An open file object or a string to a valid filename.
    format : {None, string}, optional
        Format string for the date.
        If None, uses the default date format.
    separator : {string}, optional
        Separator to write between elements of the array.
    linesep : {string}, optional
        Separator to write between rows of array.
    precision : {integer}, optional
        Number of digits after the decimal place to write.
    suppress_small : {boolean}, optional
        Whether on-zero to round small numbers down to 0.0
    keep_open : {boolean}, optional
        Whether to close the file or to return the open file.

    Returns
    -------
    file : file object
        The open file (if ``keep_open`` is non-zero).
    """

    try:
        import scipy.io
    except ImportError:
        raise ImportError("scipy is required for the tofile function/method")

    (_dates, _data) = (series._dates, series._series)
    optpars = dict(separator=separator,linesep=linesep,precision=precision,
                   suppress_small=suppress_small,keep_open=keep_open)
    if _dates.size == _data.size:
        # 1D version
        tmpfiller = ma.empty((_dates.size,2), dtype=np.object_)
        _data = _data.reshape(-1)
        tmpfiller[:,1:] = ma.atleast_2d(_data).T
    else:
        sshape = list(_data.shape)
        sshape[-1] += 1
        tmpfiller = ma.empty(sshape, dtype=np.object_)
        tmpfiller[:,1:] = _data
    #
    if format is None:
        tmpfiller[:,0] = _dates.ravel().tostring()
    else:
        tmpfiller[:,0] = [_.strftime(format) for _ in _dates.ravel()]
    return scipy.io.write_array(fileobject, tmpfiller, **optpars)


TimeSeries.tofile = tofile

#............................................
def asrecords(series):
    """
    Returns the time series as a recarray.

    Fields are ``_dates``, ``_data`` and ``_mask``.

    """
    desctype = [('_dates',int_), ('_series',series.dtype), ('_mask', bool_)]
    flat = series.ravel()
    _dates = np.asarray(flat._dates)
    if flat.size > 0:
        return recfromarrays([_dates, flat._data, getmaskarray(flat)],
                             dtype=desctype,
                             shape = (flat.size,),
                             )
    else:
        return recfromarrays([[], [], []], dtype=desctype,
                             shape = (flat.size,),
                             )
TimeSeries.asrecords = asrecords

def flatten(series):
    """
    Flattens a (multi-) time series to 1D series.

    """
    shp_ini = series.shape
    # Already flat time series....
    if len(shp_ini) == 1:
        return series
    # Folded single time series ..
    newdates = series._dates.ravel()
    if series._dates.size == series._series.size:
        newshape = (series._series.size,)
    else:
        newshape = (np.asarray(shp_ini[:-1]).prod(), shp_ini[-1])
    newseries = series._series.reshape(newshape)
    return time_series(newseries, newdates)

TimeSeries.flatten = flatten

##### -------------------------------------------------------------------------
#---- --- TimeSeries constructor ---
##### -------------------------------------------------------------------------
def time_series(data, dates=None, start_date=None, freq=None, mask=nomask,
                dtype=None, copy=False, fill_value=None, keep_mask=True,
                hard_mask=False):
    """
    Creates a TimeSeries object.

    Parameters
    ----------
    data : array_like
        Data portion of the array. Any data that is valid for constructing a
        MaskedArray can be used here. May also be a TimeSeries object.
    dates : {None, DateArray}, optional
        A sequence of dates corresponding to each entry.
        If None, the dates will be constructed as a DateArray with the same
        length as ``data``, starting at ``start_date`` with frequency ``freq``.
    start_date : {Date}, optional
        Date corresponding to the first entry of the data (index 0).
        This parameter must be a valid Date object, and is mandatory if ``dates``
        is None and if ``data`` has a length greater or equal to 1.
    freq : {freq_spec}, optional
        A valid frequency specification, as a string or an integer.
        This parameter is mandatory if ``dates`` is None.

    Notes
    -----
    * All other parameters that are accepted by the :func:`numpy.ma.array`
      function in the :mod:`numpy.ma` module are also accepted by this function.
    * The date portion of the time series must be specified in one of the
      following ways:

       * specify a TimeSeries object for the ``data`` parameter.
       * pass a DateArray for the ``dates`` parameter.
       * specify a start_date (a continuous DateArray will be automatically
         constructed for the dates portion).
       * specify just a frequency (for TimeSeries of size zero).

    """
    maparms = dict(copy=copy, dtype=dtype, fill_value=fill_value, subok=True,
                   keep_mask=keep_mask, hard_mask=hard_mask,)
    data = masked_array(data, mask=mask, **maparms)

    freq = check_freq(freq)

    if dates is None:
        _dates = getattr(data, '_dates', None)
    elif isinstance(dates, (Date, DateArray)):
        _dates = date_array(dates)
    elif isinstance(dates, (tuple, list, ndarray)):
        _dates = date_array(dlist=dates, freq=freq)
    else:
        _dates = date_array([], freq=freq)

    if _dates is not None:
        # Make sure _dates has the proper freqncy
        if (freq != _c.FR_UND) and (_dates.freq != freq):
            _dates = _dates.asfreq(freq)
    else:
        dshape = data.shape
        if len(dshape) > 0:
            length = dshape[0]
            _dates = date_array(start_date=start_date, freq=freq, length=length)
        else:
            _dates = date_array([], freq=freq)

    if _dates._unsorted is not None:
        idx = _dates._unsorted
        data = data[idx]
        _dates._unsorted = None
    return TimeSeries(data=data, dates=_dates,
                      copy=copy, dtype=dtype,
                      fill_value=fill_value, keep_mask=keep_mask,
                      hard_mask=hard_mask,)

##### --------------------------------------------------------------------------
#---- ... Additional functions ...
##### --------------------------------------------------------------------------

def compressed(series):
    """
    Suppresses missing values from a time series.
    Returns a TimeSeries object.

    """
    if series._mask is nomask:
        return series
    if series.ndim == 1:
        keeper = ~(series._mask)
    elif series.ndim == 2:
        _dates = series._dates
        _series = series._series
        # Both dates and data are 2D: ravel first
        if _dates.ndim == 2:
            series = series.ravel()
            keeper = ~(series._mask)
        # 2D series w/ only one date : return a new series ....
        elif _dates.size == 1:
            result = _series.compressed().view(type(series))
            result._dates = series.dates
            return result
        # a 2D series: suppress the rows (dates are in columns)
        else:
            keeper = ~(series._mask.any(-1))
    else:
        raise NotImplementedError
    return series[keeper]
TimeSeries.compressed = compressed
#...............................................................................
def adjust_endpoints(a, start_date=None, end_date=None):
    """
    Returns a TimeSeries going from `start_date` to `end_date`.
    If `start_date` and `end_date` both fall into the initial range of dates,
    the new series is **NOT** a copy but a simple slice of the initial series.

    Parameters
    ----------
    a : TimeSeries
        TimeSeries object whose dates must be adjusted
    start_date : Date object
        New starting date. If None, the current starting date is used.
    end_date : Date object
        New ending date. If None, the current ending date is used.
    """
    # Series validity tests .....................
    if not isinstance(a, TimeSeries):
        raise TypeError,"Argument should be a valid TimeSeries object!"
    if a.freq == 'U':
        raise TimeSeriesError, \
            "Cannot adjust a series with 'Undefined' frequency."
    if not a._dates.isvalid():
        raise TimeSeriesError, \
            "Cannot adjust a series with missing or duplicated dates."
    # Flatten the series if needed ..............
    a = a.flatten()
    shp_flat = a.shape
    # Dates validity checks .,...................
    msg = "%s should be a valid Date object! (got %s instead)"
    if a.dates.size >= 1:
        (dstart, dend) = a.dates[[0,-1]]
    else:
        (dstart, dend) = (None, None)
    # Skip the empty series case
    if dstart is None and (start_date is None or end_date is None):
        raise TimeSeriesError, "Both start_date and end_date must be specified"+\
                               " to adjust endpoints of a zero length series!"
    #....
    if start_date is None:
        start_date = dstart
        start_lag = 0
    else:
        if not isinstance(start_date, Date):
            raise TypeError, msg % ('start_date', type(start_date))
        if dstart is not None:
            start_lag = start_date - dstart
        else:
            start_lag = start_date
    #....
    if end_date is None:
        end_date = dend
        end_lag = 0
    else:
        if not isinstance(end_date, Date):
            raise TypeError, msg % ('end_date', type(end_date))
        if dend is not None:
            end_lag = end_date - dend
        else:
            end_lag = end_date
    # Check if the new range is included in the old one
    if start_lag >= 0:
        if end_lag == 0:
            return a[start_lag:]
        elif end_lag < 0:
            return a[start_lag:end_lag]
    # Create a new series .......................
    newdates = date_array(start_date=start_date, end_date=end_date)

    newshape = list(shp_flat)
    newshape[0] = len(newdates)
    newshape = tuple(newshape)

    newseries = np.empty(newshape, dtype=a.dtype).view(type(a))
    #!!!: Here, we may wanna use something else than MaskType
    newseries.__setmask__(np.ones(newseries.shape, dtype=bool_))
    newseries._dates = newdates
    newseries._update_from(a)
    if dstart is not None:
        start_date = max(start_date, dstart)
        end_date = min(end_date, dend) + 1
        newseries[start_date:end_date] = a[start_date:end_date]
    return newseries
#.....................................................
def align_series(*series, **kwargs):
    """
    Aligns several TimeSeries, so that their starting and ending dates match.
    Series are resized and filled with masked values accordingly. The resulting
    series have no missing dates (ie. ``series.isvalid() == True`` for each of the
    resulting series).

    The function accepts two extras parameters:
    - `start_date` forces the series to start at that given date,
    - `end_date` forces the series to end at that given date.
    By default, `start_date` and `end_date` are set respectively to the smallest
    and largest dates of the series.
    """
    if len(series) < 2:
        return series
    unique_freqs = np.unique([x.freqstr for x in series])
    common_freq = _compare_frequencies(*series)

    # if any of the series have missing dates, fill them in first
    filled_series = []
    for ser in series:
        if ser.isvalid():
            filled_series.append(ser)
        else:
            filled_series.append(ser.fill_missing_dates())

    start_date = kwargs.pop('start_date',
                            min([x.start_date for x in filled_series
                                     if x.start_date is not None]))
    if isinstance(start_date,str):
        start_date = Date(common_freq, string=start_date)
    end_date = kwargs.pop('end_date',
                          max([x.end_date for x in filled_series
                                   if x.end_date is not None]))
    if isinstance(end_date,str):
        end_date = Date(common_freq, string=end_date)

    return [adjust_endpoints(x, start_date, end_date) for x in filled_series]
aligned = align_series

#.....................................................
def align_with(*series):
    """
    Aligns several TimeSeries to the first of the list, so that their
    starting and ending dates match.
    Series are resized and filled with masked values accordingly.
    """
    if len(series) < 2:
        return series
    dates = series[0]._dates[[0,-1]]
    if len(series) == 2:
        return adjust_endpoints(series[-1], dates[0], dates[-1])
    return [adjust_endpoints(x, dates[0], dates[-1]) for x in series[1:]]


#....................................................................
def _convert1d(series, freq, func, position, *args, **kwargs):
    "helper function for `convert` function"
    if not isinstance(series,TimeSeries):
        raise TypeError("The argument should be a valid TimeSeries!")
    # Check the frequencies ..........................
    to_freq = check_freq(freq)
    from_freq = series.freq
    # Don't do anything if not needed
    if from_freq == to_freq:
        return series
    if from_freq == _c.FR_UND:
        err_msg = "Cannot convert a series with UNDEFINED frequency."
        raise TimeSeriesError(err_msg)
    if to_freq == _c.FR_UND:
        err_msg = "Cannot convert a series to UNDEFINED frequency."
        raise TimeSeriesError(err_msg)
    # Check the validity of the series .....
    if not series.isvalid():
        err_msg = "Cannot adjust a series with missing or duplicated dates."
        raise TimeSeriesError(err_msg)

    # Check the position parameter..........
    position = position.upper()
    if position not in ('END','START'):
        err_msg = "Invalid value for position argument: (%s). "\
                  "Should be in ['END','START']," % str(position)
        raise ValueError(err_msg)

    start_date = series._dates[0]

    if series.size == 0:
        return TimeSeries(series, freq=to_freq,
                          start_date=start_date.asfreq(to_freq))

    tmpdata = series._series.filled()
    tmpmask = getmaskarray(series)

    if (tmpdata.size // series._dates.size) > 1:
        raise TimeSeriesError("convert works with 1D data only !")

    cdictresult = cseries.TS_convert(tmpdata, from_freq, to_freq, position,
                                     int(start_date), tmpmask)
    start_date = Date(freq=to_freq, value=cdictresult['startindex'])
    tmpdata = masked_array(cdictresult['values'], mask=cdictresult['mask'])

    if tmpdata.ndim == 2:
        if func is None:
            newvarshape = tmpdata.shape[1:]
        else:
            tmpdata = ma.apply_along_axis(func, -1, tmpdata, *args, **kwargs)
            newvarshape = ()
    elif tmpdata.ndim == 1:
        newvarshape = ()

    newdates = date_array(start_date=start_date,
                          length=len(tmpdata),
                          freq=to_freq)

    newseries = tmpdata.view(type(series))
    newseries._varshape = newvarshape
    newseries._dates = newdates
    newseries._update_from(series)
    return newseries

def convert(series, freq, func=None, position='END', *args, **kwargs):
    """
    Converts a series from one frequency to another, by manipulating both the
    `data` and `dates` attributes.

    If the input series has any missing dates, it will first be filled in with
    masked values prior to doing the conversion.

    Parameters
    ----------
    series : TimeSeries
        Series to convert. Skip this parameter if you are calling this as
        a method of the TimeSeries object instead of the module function.
    freq : freq_spec
        Frequency to convert the TimeSeries to. Accepts any valid frequency
        specification (string or integer)
    func : function, optional
        When converting a series to a lower frequency, you can use the
        ``func`` parameter to perform a calculation on each period of values
        to aggregate results.
        For example, when converting a daily series to a monthly series, use
        :func:`numpy.ma.mean` to get a series of monthly averages. If you wish
        to get the first or last value from a period, use the functions
        :func:`scikits.timeseries.first_unmasked_val` and
        :func:`scikits.timeseries.last_unmasked_val`.
        If ``func`` is not given, the output series group the points of the
        initial series that share the same new date. For example, if the
        initial series has a daily frequency and is 1D, the output series is
        2D.
    position : {'END', 'START'}, optional
        When converting a series to a higher frequency, use this parameter to
        determine where the points should fall in the new period.
        For example, when converting a monthly series to daily, specifying
        position='START' will cause the values to fall on the first day of
        each month (with all other values being masked).
    *args : {extra arguments for func parameter}, optional
        Additional mandatory parameters of the ``func`` function.
    **kwargs : {extra keyword arguments for func parameter}, optional
        Additional optional keyword parameters of the ``func`` function.

    Returns
    -------
    converted_series
        A new TimeSeries a thte given frequency, without any missing nor duplicated
        dates

    """
    #!!!: Raise some kind of proper exception if the underlying dtype will mess things up
    #!!!: For example, mean on string array...

    if series.ndim > 2 or series.ndim == 0:
        raise ValueError(
            "only series with ndim == 1 or ndim == 2 may be converted")

    if series.has_duplicated_dates():
        raise TimeSeriesError("The input series must not have duplicated dates!")

    if series.has_missing_dates():
        # can only convert continuous time series, so fill in missing dates
        series = fill_missing_dates(series)

    if series.ndim == 1:
        obj = _convert1d(series, freq, func, position, *args, **kwargs)
    elif series.ndim == 2:
        base = _convert1d(series[:,0], freq, func, position, *args, **kwargs)
        obj = ma.column_stack([_convert1d(m,freq,func,position,
                                          *args, **kwargs)._series
                               for m in series.split()]).view(type(series))
        obj._dates = base._dates
        if func is None:
            shp = obj.shape
            ncols = base.shape[-1]
            obj.shape = (shp[0], shp[-1]//ncols, ncols)
            obj = np.swapaxes(obj,1,2)

    return obj
TimeSeries.convert = convert

#...............................................................................
def tshift(series, nper, copy=True):
    """
    Returns a series of the same size as `series`, with the same `start_date`
    and `end_date`, but values shifted by `nper`.

    Parameters
    ----------
    series : TimeSeries
        TimeSeries object to shift. Ignore this parameter if calling this as a
        method.
    nper : int
        Number of periods to shift. Negative numbers shift values to the right,
        positive to the left.
    copy : {True, False}, optional
        copies the data if True, returns a view if False.

    Example
    -------
    >>> series = time_series([0,1,2,3], start_date=Date(freq='A', year=2005))
    >>> series
    timeseries(data  = [0 1 2 3],
               dates = [2005 ... 2008],
               freq  = A-DEC)
    >>> tshift(series, -1)
    timeseries(data  = [-- 0 1 2],
               dates = [2005 ... 2008],
               freq  = A-DEC)
    >>> pct_change = 100 * (series/series.tshift(-1, copy=False) - 1)

    """
    newdata = masked_array(np.empty(series.shape, dtype=series.dtype),
                           mask=True)
    if copy:
        inidata = series._series.copy()
    else:
        inidata = series._series
    if nper < 0:
        nper = max(-len(series), nper)
        newdata[-nper:] = inidata[:nper]
    elif nper > 0:
        nper = min(len(series), nper)
        newdata[:-nper] = inidata[nper:]
    else:
        newdata = inidata
    newseries = newdata.view(type(series))
    newseries._dates = series._dates
    newseries._update_from(series)
    return newseries
TimeSeries.tshift = tshift
#...............................................................................
def _get_type_num_double(dtype):
    """
    Private used to force dtypes upcasting in certain functions
    (eg. int -> float in pct function).
    Adapted from function of the same name in the C source code.
    """
    if dtype.num  < np.dtype('f').num:
        return np.dtype('d')
    return dtype
#...............................................................................
def pct(series, nper=1):
    """
    Returns the rolling percentage change of the series.

    Parameters
    ----------
    series : {TimeSeries}
        TimeSeries object to to calculate percentage chage for. Ignore this
        parameter if calling this as a method.
    nper : {int}
        Number of periods for percentage change.

    Notes
    -----
    Series of integer types will be upcast
    1.0 == 100% in result

    Examples
    --------
    >>> series = time_series([2.,1.,2.,3.], start_date=Date(freq='A', year=2005))
    >>> series.pct()
    timeseries([-- -0.5 1.0 0.5],
               dates = [2005 ... 2008],
               freq  = A-DEC)
    >>> series.pct(2)
    timeseries([-- -- 0.0 2.0],
               dates = [2005 ... 2008],
               freq  = A-DEC)

    """
    _dtype = _get_type_num_double(series.dtype)
    if _dtype != series.dtype:
        series = series.astype(_dtype)
    newdata = masked_array(np.empty(series.shape, dtype=series.dtype),
                           mask=True)
    if nper < newdata.size:
        newdata[nper:] = (series._series[nper:]/series._series[:-nper] - 1)
    newseries = newdata.view(type(series))
    newseries._dates = series._dates
    newseries._update_from(series)
    return newseries
TimeSeries.pct = pct
#...............................................................................
def fill_missing_dates(data, dates=None, freq=None, fill_value=None):
    """
    Finds and fills the missing dates in a time series. The data
    corresponding to the initially missing dates are masked, or filled to
    `fill_value`.

    Parameters
    ----------
    data : {TimeSeries, ndarray}
        Initial array of data.
    dates : {DateArray} (optional)
        Initial array of dates. Specify this if you are passing a plain ndarray
        for the data instead of a :class:`TimeSeries`.
    freq : {freq_spec} (optional)
        Frequency of result. If not specified, the initial frequency is used.
    fill_value : {scalar of type data.dtype} (optional)
        Default value for missing data. If Not specified, the data are just
        masked.

    """
    # Check the frequency ........
    orig_freq = freq
    freq = check_freq(freq)
    if orig_freq is not None and freq == _c.FR_UND:
        freqstr = check_freq_str(freq)
        raise ValueError,\
              "Unable to define a proper date resolution (found %s)." % freqstr
    # Check the dates .............
    if dates is None:
        if not isinstance(data, TimeSeries):
            raise InsufficientDateError
        dates = data._dates
    else:
        if not isinstance(dates, DateArray):
            dates = DateArray(dates, freq)
    dflat = dates.asfreq(freq).ravel()
    if not dflat.has_missing_dates():
        if isinstance(data, TimeSeries):
            return data
        data = data.view(TimeSeries)
        data._dates = dflat
        return data
    # Check the data ..............
    if isinstance(data, MaskedArray):
        datad = data._data
        datam = data._mask
        if isinstance(data, TimeSeries):
            datat = type(data)
            datas = data._varshape
        else:
            datat = TimeSeries
            datas = ()
    else:
        datad = np.asarray(data)
        datam = nomask
        datat = TimeSeries
    # Check whether we need to flatten the data
    if data.ndim > 1:
        if (not datas):
            datad.shape = -1
        elif dflat.size != len(datad):
            err_msg = "fill_missing_dates is not yet implemented for nD series!"
            raise NotImplementedError(err_msg)
    # ...and now, fill it ! ......
    (tstart, tend) = dflat[[0,-1]]
    newdates = date_array(start_date=tstart, end_date=tend)
    (osize, nsize) = (dflat.size, newdates.size)
    #.............................
    # Get the steps between consecutive data.
    delta = dflat.get_steps()-1
    gap = delta.nonzero()
    slcid = np.r_[[0,], np.arange(1,osize)[gap], [osize,]]
    oldslc = np.array([slice(i,e)
                       for (i,e) in np.broadcast(slcid[:-1],slcid[1:])])
    addidx = delta[gap].astype(int_).cumsum()
    newslc = np.r_[[oldslc[0]],
                   [slice(i+d,e+d) for (i,e,d) in \
                           np.broadcast(slcid[1:-1],slcid[2:],addidx)]
                     ]
    #.............................
    # Just a quick check
    vdflat = np.asarray(dflat)
    vnewdates = np.asarray(newdates)
    for (osl,nsl) in zip(oldslc,newslc):
        assert np.equal(vdflat[osl],vnewdates[nsl]).all(),\
            "Slicing mishap ! Please check %s (old) and %s (new)" % (osl,nsl)
    #.............................
    newshape = list(datad.shape)
    newshape[0] = nsize
    newdatad = np.empty(newshape, data.dtype)
    #!!!: HERE, newdatam should call make_mask_none(newshape, mdtype) for records
    newdatam = np.ones(newshape, bool_)
    #....
    if datam is nomask:
        for (new,old) in zip(newslc,oldslc):
            newdatad[new] = datad[old]
            newdatam[new] = False
    else:
        for (new,old) in zip(newslc,oldslc):
            newdatad[new] = datad[old]
            newdatam[new] = datam[old]
    if fill_value is None:
        fill_value = getattr(data, 'fill_value', None)
    newdata = ma.masked_array(newdatad, mask=newdatam, fill_value=fill_value)
    _data = newdata.view(datat)
    _data._dates = newdates
    return _data
TimeSeries.fill_missing_dates = fill_missing_dates
#..............................................................................
def stack(*series):
    """
    Performs a column_stack on the data from each series, and the
    resulting series has the same dates as each individual series. The series
    must have compatible dates (same starting and ending dates, same frequency).

    Parameters
    ----------
    series : the series to be stacked
"""
    _timeseriescompat_multiple(*series)
    return time_series(ma.column_stack(series), series[0]._dates,
                       **_attrib_dict(series[0]))

def concatenate(series, axis=0, remove_duplicates=True, fill_missing=False):
    """
    Joins series together.

    The series are joined in chronological order.
    Duplicated dates are handled with the `remove_duplicates` parameter.
    If `remove_duplicate` is False, duplicated dates are saved.
    Otherwise, only the first occurence of the date is conserved.


    Parameters
    ----------
    series : {sequence}
        Sequence of time series to join
    axis : {0, None, int}, optional
        Axis along which to join
    remove_duplicates : {False, True}, optional
        Whether to remove duplicated dates.
    fill_missing : {False, True}, optional
        Whether to fill the missing dates with missing values.

    Example
    -------
    >>> a = time_series([1,2,3], start_date=now('D'))
    >>> b = time_series([10,20,30], start_date=now('D')+1)
    >>> c = concatenate((a,b))
    >>> c._series
    masked_array(data = [ 1  2  3 30],
          mask = False,
          fill_value=999999)

    """
    # Get the common frequency, raise an error if incompatibility
    common_f = _compare_frequencies(*series)
    # Concatenate the order of series
    sidx = np.concatenate([np.repeat(i,len(s))
                          for (i,s) in enumerate(series)], axis=axis)
    # Concatenate the dates and data
    ndates = np.concatenate([s._dates for s in series], axis=axis)
    ndata = ma.concatenate([s._series for s in series], axis=axis)
    # Resort the data chronologically
    norder = ndates.argsort(kind='mergesort')
    ndates = ndates[norder]
    ndata = ndata[norder]
    sidx = sidx[norder]
    #
    if not remove_duplicates:
        ndates = date_array(ndates, freq=common_f)
        result = time_series(ndata, dates=ndates)
    else:
        # Find the original dates
        orig = np.concatenate([[True],(np.diff(ndates) != 0)])
        result = time_series(ndata.compress(orig),
                             dates=ndates.compress(orig),freq=common_f)
    if fill_missing:
        result = fill_missing_dates(result)
    return result
#...............................................................................
def empty_like(series):
    """
    Returns an empty series with the same dtype, mask and dates as series.
    """
    result = np.empty_like(series).view(type(series))
    result._dates = series._dates
    result._mask = series._mask.copy()
    return result

################################################################################
