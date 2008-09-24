"""
Classes definition for the support of individual dates and array of dates.

:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com

"""

#!!!: Implement DateArray in C (Cython ?)

__author__ = "Pierre GF Gerard-Marchant & Matt Knox"
__revision__ = "$Revision$"
__date__     = '$Date$'

import datetime as dt

import itertools
import warnings
import types

import numpy as np
from numpy import bool_, float_, int_, object_, ndarray
import numpy.core.numerictypes as ntypes
from numpy.core.numerictypes import generic

import numpy.ma as ma

from parser import DateFromString, DateTimeFromString

import const as _c
import cseries

# initialize python callbacks for C code
cseries.set_callback_DateFromString(DateFromString)
cseries.set_callback_DateTimeFromString(DateTimeFromString)

from cseries import Date, now, check_freq, check_freq_str, get_freq_group,\
                    DateCalc_Error, DateCalc_RangeError

__all__ = ['ArithmeticDateError',
           'Date', 'DateArray', 'DateCalc_Error', 'DateCalc_RangeError',
           'DateError',
           'FrequencyDateError',
           'InsufficientDateError',
           'check_freq', 'check_freq_str', 'convert_to_float',
           'date_array', 'day', 'day_of_year',
           'get_freq_group',
           'hour',
           'minute', 'month',
           'nodates', 'now',
           'period_break', 'prevbusday',
           'quarter',
           'second',
           'weekday',
           'week',
           'year',
          ]

#####---------------------------------------------------------------------------
#---- --- Date Exceptions ---
#####---------------------------------------------------------------------------
class DateError(Exception):
    """
    Defines a generic DateArrayError.
    """
    def __init__ (self, value=None):
        "Creates an exception."
        self.value = value
    def __str__(self):
        "Calculates the string representation."
        return str(self.value)
    __repr__ = __str__

class InsufficientDateError(DateError):
    """
    Defines the exception raised when there is not enough information
    to create a Date object.
    """
    def __init__(self, msg=None):
        if msg is None:
            msg = "Insufficient parameters given to create a date at the given frequency"
        DateError.__init__(self, msg)

class FrequencyDateError(DateError):
    """
    Defines the exception raised when the frequencies are incompatible.
    """
    def __init__(self, msg, freql=None, freqr=None):
        msg += " : Incompatible frequencies!"
        if not (freql is None or freqr is None):
            msg += " (%s<>%s)" % (freql, freqr)
        DateError.__init__(self, msg)

class ArithmeticDateError(DateError):
    """
    Defines the exception raised when dates are used in arithmetic expressions.
    """
    def __init__(self, msg=''):
        msg += " Cannot use dates for arithmetics!"
        DateError.__init__(self, msg)


#####---------------------------------------------------------------------------
#---- --- Functions ---
#####---------------------------------------------------------------------------

def prevbusday(day_end_hour=18, day_end_min=0):
    """
    Returns the previous business day (Monday-Friday) at business frequency.

    Parameters
    ----------
    day_end_hour : {18, int}, optional
        Hour of the end of a business day.
    day_end_min : {0, int}, optional
        Minutes of the end of a business day.

    Notes
    -----
    If it is currently Saturday or Sunday, then the preceding Friday will be
    returned. If it is later than the specified ``day_end_hour`` and ``day_end_min``,
    ``now('Business')`` will be returned.
    Otherwise, ``now('Business')-1`` will be returned.

    """
    tempDate = dt.datetime.now()
    dateNum = tempDate.hour + float(tempDate.minute)/60
    checkNum = day_end_hour + float(day_end_min)/60
    if dateNum < checkNum and tempDate.weekday() < 5:
        return now(_c.FR_BUS) - 1
    else:
        return now(_c.FR_BUS)

#####---------------------------------------------------------------------------
#---- --- DateArray ---
#####---------------------------------------------------------------------------
ufunc_dateOK = ['add','subtract',
                'equal','not_equal','less','less_equal', 'greater','greater_equal',
                'isnan']

class _datearithmetics(object):
    """
    Defines a wrapper for arithmetic methods.
    Instead of directly calling a ufunc, the corresponding method of the `._data`
    object is called instead.
    If `asdates` is True, a DateArray object is returned , else a regular ndarray
    is returned.
    """
    def __init__ (self, methodname, asdates=True):
        """
    Parameters
    ----------
    methodname : string
        Method name.
    asdates : {True, False}
        Whether to return a DateArray object (True) or a regular ndarray.
        """
        self.methodname = methodname
        self._asdates = asdates
        self.__doc__ = getattr(methodname, '__doc__')
        self.obj = None
    #
    def __get__(self, obj, objtype=None):
        self.obj = obj
        return self
    #
    def __call__ (self, other, *args, **kwargs):
        "Execute the call behavior."
        instance = self.obj
        freq = instance.freq
        if 'context' not in kwargs:
            kwargs['context'] = 'DateOK'
        method = getattr(super(DateArray,instance), self.methodname)
        other_val = other
        if isinstance(other, DateArray):
            if other.freq != freq:
                raise FrequencyDateError("Cannot operate on dates", \
                                         freq, other.freq)
        elif isinstance(other, Date):
            if other.freq != freq:
                raise FrequencyDateError("Cannot operate on dates", \
                                         freq, other.freq)
            other_val = other.value
        elif isinstance(other, ndarray):
            if other.dtype.kind not in ['i','f']:
                raise ArithmeticDateError
        if self._asdates and not isinstance(other, (DateArray, Date)):
            return instance.__class__(method(other_val, *args),
                                      freq=freq)
        else:
            return method(other_val, *args)



class DateArray(ndarray):
    """
    Defines a ndarray of dates, as ordinals.

    When viewed globally (array-wise), ``DateArray`` is an array of integers.
    When viewed element-wise, ``DateArray`` is a sequence of dates.
    For example, a test such as :

    >>> DateArray(...) == value

    will be valid only if value is an integer, not a :class:`Date` object.
    However, a loop such as :

    >>> for d in DateArray(...):

    accesses the array element by element. Therefore, `d` is a :class:`Date` object.
    """
    def __new__(cls, dates=None, freq=None, copy=False):
        # Get the frequency ......
        if freq is None:
            _freq = getattr(dates, 'freq', _c.FR_UND)
        else:
            _freq = check_freq(freq)
        # Get the dates ..........
        _dates = np.array(dates, copy=copy, dtype=int_, subok=1)
        if _dates.ndim == 0:
            _dates.shape = (1,)
        _dates = _dates.view(cls)
        _dates.freq = _freq
        _dates._unsorted = None
        return _dates

    def __array_wrap__(self, obj, context=None):
        if context is None:
            return self
        elif context[0].__name__ not in ufunc_dateOK:
            raise ArithmeticDateError, "(function %s)" % context[0].__name__

    def __array_finalize__(self, obj):
        self.freq = getattr(obj, 'freq', _c.FR_UND)
        self._unsorted = getattr(obj,'_unsorted',None)
        self._cachedinfo = dict(toobj=None, tostr=None, toord=None,
                                steps=None, full=None, hasdups=None)
        if hasattr(obj,'_cachedinfo'):
            self._cachedinfo.update(obj._cachedinfo)
        return

    def __getitem__(self, indx):
        reset_full = True
        # Determine what kind of index is used
        if isinstance(indx, Date):
            indx = self.find_dates(indx)
            reset_full = False
        elif isinstance(indx, slice):
            pass
        elif np.asarray(indx).dtype.kind == 'O':
            try:
                indx = self.find_dates(indx)
            except AttributeError:
                pass

        # Select the data
        r = ndarray.__getitem__(self, indx)
        # Select the corresponding unsorted indices (if needed)
        if self._unsorted is not None:
            unsorted = self._unsorted[indx]
            #!!!: And then what do we do w/ unsorted ???
        # Case 1. A simple integer
        if isinstance(r, (generic, int)):
            return Date(self.freq, value=r)
        elif hasattr(r, 'ndim') and r.ndim == 0:
            # need to check if it has a ndim attribute for situations
            # like when the datearray is the data for a maskedarray
            # or some other subclass of ndarray with wierd getitem
            # behaviour
            return Date(self.freq, value=r.item())
        else:
            if hasattr(r, '_cachedinfo'):

                _cache = r._cachedinfo
                _cache.update(dict([(k,_cache[k][indx])
                                    for k in ('toobj', 'tostr', 'toord')
                                    if _cache[k] is not None]))
                _cache['steps'] = None
                if reset_full:
                    _cache['full'] = None
                    _cache['hasdups'] = None
            return r

    def __repr__(self):
        return ndarray.__repr__(self)[:-1] + \
               ",\n          freq='%s')" % self.freqstr


    def __contains__(self, date):
        """For checking if a single date (or equivalent integer value) is
        contained in the DateArray.
        """
        if isinstance(date, Date) and date.freq != self.freq:
            raise ValueError(
                "expected date of frequency '%s' but got date of frequency "\
                "'%s'" % (self.freqstr, date.freqstr))
        datenum = np.array(date, dtype=self.dtype)
        if datenum.ndim != 0:
            raise ValueError("Cannot check containment of multiple dates")
        return datenum in self.view(np.ndarray)

    #......................................................
    __add__ = _datearithmetics('__add__', asdates=True)
    __radd__ = _datearithmetics('__add__', asdates=True)
    __sub__ = _datearithmetics('__sub__', asdates=True)
    __rsub__ = _datearithmetics('__rsub__', asdates=True)
    __le__ = _datearithmetics('__le__', asdates=False)
    __lt__ = _datearithmetics('__lt__', asdates=False)
    __ge__ = _datearithmetics('__ge__', asdates=False)
    __gt__ = _datearithmetics('__gt__', asdates=False)
    __eq__ = _datearithmetics('__eq__', asdates=False)
    __ne__ = _datearithmetics('__ne__', asdates=False)
    #......................................................
    @property
    def freqstr(self):
        "Returns the frequency string code."
        return check_freq_str(self.freq)
    @property
    def day(self):
        "Returns the day of month."
        return self.__getdateinfo__('D')
    @property
    def weekday(self):
        "Returns the day of week."
        return self.__getdateinfo__('W')
    @property
    def day_of_year(self):
        "Returns the day of year."
        return self.__getdateinfo__('R')
    @property
    def month(self):
        "Returns the month."
        return self.__getdateinfo__('M')
    @property
    def quarter(self):
        "Returns the quarter."
        return self.__getdateinfo__('Q')
    @property
    def year(self):
        "Returns the year."
        return self.__getdateinfo__('Y')
    @property
    def qyear(self):
        """
    For quarterly frequency dates, returns the year corresponding to the
    year end (start) month. When using QTR or QTR-E based quarterly
    frequencies, this is the fiscal year in a financial context.

    For non-quarterly dates, this simply returns the year of the date.

    """

        return self.__getdateinfo__('F')
    @property
    def second(self):
        "Returns the seconds."
        return self.__getdateinfo__('S')
    @property
    def minute(self):
        "Returns the minutes."
        return self.__getdateinfo__('T')
    @property
    def hour(self):
        "Returns the hour."
        return self.__getdateinfo__('H')
    @property
    def week(self):
        "Returns the week."
        return self.__getdateinfo__('I')

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

    def __getdateinfo__(self, info):
        return np.asarray(cseries.DA_getDateInfo(np.asarray(self),
                                                 self.freq, info,
                                                 int(self.isfull())),
                               dtype=int_)
    __getDateInfo = __getdateinfo__
    #.... Conversion methods ....................
    #
    def tovalue(self):
        "Converts the dates to integer values."
        return np.asarray(self)
    #
    def toordinal(self):
        "Converts the dates from values to ordinals."
        # Note: we better try to cache the result
        if self._cachedinfo['toord'] is None:
            if self.freq == _c.FR_UND:
                diter = (d.value for d in self)
            else:
                diter = (d.toordinal() for d in self)
            toord = np.fromiter(diter, dtype=float_)
            self._cachedinfo['toord'] = toord
        return self._cachedinfo['toord']
    #
    def tolist(self):
        """Returns a hierarchical python list of standard datetime objects."""
        _result = np.empty(self.shape, dtype=np.object_)
        _result.flat = [d.datetime for d in self.ravel()]
        return _result.tolist()
    #
    def tostring(self):
        "Converts the dates to strings."
        # Note: we better cache the result
        if self._cachedinfo['tostr'] is None:
            firststr = str(self[0])
            if self.size > 0:
                ncharsize = len(firststr)
                tostr = np.fromiter((str(d) for d in self),
                                    dtype='|S%i' % ncharsize)
            else:
                tostr = firststr
            self._cachedinfo['tostr'] = tostr
        return self._cachedinfo['tostr']
    #
    def asfreq(self, freq=None, relation="END"):
        """

    Converts the dates to another frequency.

    Parameters
    ----------
    freq : {freq_spec}
        Frequency into which :class:`DateArray` must be converted.
        Accepts any valid frequency specification (string or integer)
    relation : {"END", "START"} (optional)
        Applies only when converting a lower frequency :class:`Date` to a higher
        frequency :class:`Date`, or when converting a weekend :class:`Date` to a business
        frequency :class:`Date`. Valid values are 'START' and 'END' (or just 'S' and
        'E' for brevity if you wish).

        For example, if converting a monthly date to a daily date, specifying
        'START' ('END') would result in the first (last) day in the month.

    """
        # Note: As we define a new object, we don't need caching
        if freq is None or freq == _c.FR_UND:
            return self
        tofreq = check_freq(freq)
        if tofreq == self.freq:
            return self

        relation = relation.upper()

        if relation not in ('START', 'END', 'S', 'E'):
            raise ValueError(
                "invalid specification for 'relation' parameter: %s" % \
                relation)

        fromfreq = self.freq
        if fromfreq == _c.FR_UND:
            fromfreq = _c.FR_DAY
        new = cseries.DA_asfreq(np.asarray(self), fromfreq, tofreq, relation[0])
        return DateArray(new, freq=freq)

    def find_dates(self, *dates):
        "Returns the indices corresponding to given dates, as an array."

        #http://aspn.activestate.com/ASPN/Mail/Message/python-tutor/2302348
        def flatten_sequence(iterable):
            """Flattens a compound of nested iterables."""
            itm = iter(iterable)
            for elm in itm:
                if hasattr(elm,'__iter__') and not isinstance(elm, basestring):
                    for f in flatten_sequence(elm):
                        yield f
                else:
                    yield elm

        def flatargs(*args):
            "Flattens the arguments."
            if not hasattr(args, '__iter__'):
                return args
            else:
                return flatten_sequence(args)

        ifreq = self.freq
        c = np.zeros(self.shape, bool_)
        for d in flatargs(*dates):
            if d.freq != ifreq:
                d = d.asfreq(ifreq)
            c += (self == d.value)
        c = c.nonzero()
        if np.size(c) == 0:
            raise IndexError, "Date out of bounds!"
        return c

    def date_to_index(self, dates):
        "Returns the index corresponding to one given date, as an integer."
        vals = self.view(ndarray)
        if isinstance(dates, Date):
            _val = dates.value
            if _val not in vals:
                raise IndexError("Date '%s' is out of bounds" % dates)
            if self.isvalid():
                return _val - vals[0]
            else:
                return np.where(vals == _val)[0][0]

        _dates = DateArray(dates, freq=self.freq)
        if self.isvalid():
            indx = (_dates.view(ndarray) - vals[0])
            err_cond = (indx < 0) | (indx > self.size)
            if err_cond.any():
                err_indx = np.compress(err_cond, _dates)[0]
                err_msg = "Date '%s' is out of bounds '%s' <= date <= '%s'"
                raise IndexError(err_msg % (err_indx, self[0], self[-1]))
            return indx
        vals = vals.tolist()
        indx = np.array([vals.index(d) for d in _dates.view(ndarray)])

        return indx

    def get_steps(self):
        """
    Returns the time steps between consecutive dates.
    The steps have the same unit as the frequency of the series.
    """
        if self._cachedinfo['steps'] is None:
            _cached = self._cachedinfo
            val = np.asarray(self).ravel()
            if val.size > 1:
                steps = val[1:] - val[:-1]
                if _cached['full'] is None:
                    _cached['full'] = (steps.max() == 1)
                if _cached['hasdups'] is None:
                    _cached['hasdups'] = (steps.min() == 0)
            else:
                _cached['full'] = True
                _cached['hasdups'] = False
                steps = np.array([], dtype=int_)
            self._cachedinfo['steps'] = steps
        return self._cachedinfo['steps']

    def has_missing_dates(self):
        "Returns whether the DateArray have missing dates."
        if self._cachedinfo['full'] is None:
            steps = self.get_steps()
        return not(self._cachedinfo['full'])

    def isfull(self):
        "Returns whether the :class:`DateArray` has no missing dates."
        if self._cachedinfo['full'] is None:
            steps = self.get_steps()
        return self._cachedinfo['full']

    def has_duplicated_dates(self):
        "Returns whether the :class:`DateArray` has duplicated dates."
        if self._cachedinfo['hasdups'] is None:
            steps = self.get_steps()
        return self._cachedinfo['hasdups']

    def isvalid(self):
        "Returns whether the :class:`DateArray` is valid: no missing/duplicated dates."
        return  (self.isfull() and not self.has_duplicated_dates())
    #......................................................
    @property
    def start_date(self):
        "Returns the first date of the array."
        if self.size:
            return self[0]
        return None

    @property
    def end_date(self):
        "Returns the last date of the array."
        if self.size:
            return self[-1]
        return None

def fill_missing_dates(dates, freq=None):
    """
    Finds and fills the missing dates in a :class:`DateArray`.

    Parameters
    ----------
    dates : {DateArray}
        Initial array of dates.
    freq : {freq_spec}, optional
        Frequency of result. 
        If not specified, the frequency of the input DateArray is used.
    """
    # Check the frequency ........
    orig_freq = freq
    freq = check_freq(freq)
    if orig_freq is not None and freq == _c.FR_UND:
        freqstr = check_freq_str(freq)
        raise ValueError,\
              "Unable to define a proper date resolution (found %s)." % freqstr
    # Check the dates .............
    if not isinstance(dates, DateArray):
        raise ValueError("expected DateArray, got %s" % type(dates))

    if freq != dates.freq:
        dates = dates.asfreq(freq)

    if dates.ndim != 1:
        dates = dates.ravel()
    if not dates.has_missing_dates():
        return dates

    # ...and now, fill it ! ......
    (tstart, tend) = dates[[0,-1]]
    return date_array(start_date=tstart, end_date=tend)

DateArray.fill_missing_dates = fill_missing_dates

nodates = DateArray([])

#####---------------------------------------------------------------------------
#---- --- DateArray functions ---
#####---------------------------------------------------------------------------
def _listparser(dlist, freq=None):
    "Constructs a DateArray from a list."
    dlist = np.array(dlist, copy=False, ndmin=1)

    # Case #1: dates as strings .................
    if dlist.dtype.kind in 'SU':
        #...construct a list of dates
        dvals = [Date(freq, string=s).value for s in dlist]
        dlist = np.array(dvals, copy=False, ndmin=1)

    # Make sure that the list is sorted (save the original order if needed)
    idx = dlist.argsort()
    if (idx[1:] - idx[:-1] < 0).any():
        dlist = dlist[idx]
    else:
        idx = None

    # Case #2: dates as numbers .................
    if dlist.dtype.kind in 'if':
        #...hopefully, they are values
        dates = dlist
    # Case #3: dates as objects .................
    elif dlist.dtype.kind == 'O':
        template = dlist[0]
        #...as Date objects
        if isinstance(template, Date):
            dates = np.fromiter((d.value for d in dlist), int_)
            if freq in (_c.FR_UND, None): freq = template.freq
        #...as mx.DateTime objects
        elif hasattr(template,'absdays'):
            dates = [Date(freq, datetime=m) for m in dlist]
        #...as datetime objects
        elif hasattr(template, 'toordinal'):
            dates = [Date(freq, datetime=d) for d in dlist]
    #
    result = DateArray(dates, freq)
    result._unsorted = idx
    return result

def date_array(dlist=None, start_date=None, end_date=None, length=None,
               freq=None):
    """
    Factory function for constructing a DateArray.

    Parameters
    ----------
    dlist : {sequence, DateArray}, optional
        A list of dates, integer representations of dates for a given
        frequency, datetime objects, or an existing DateArray.
        If `dlist` is a list of dates, the `freq` parameter must also be given.
    start_date : {Date}, optional
        First date of a continuous DateArray.
        This parameter is used only if `dlist` is None. In that case, an ending
        date (`end_date`) or the length of the array must be given.
        The frequency of the output will be the frequency of this parameter.
    end_date : {Date} (optional)
        Last date of the output. Specify this parameter or `length`
        in combination with `start_date` for a continuous DateArray.
    length : {int} (optional)
        Length of the output. Specify this parameter or
        `end_date` in combination with `start_date` for a continuous DateArray.

    Returns
    -------
    A :class:`DateArray` object.
    """
    freq = check_freq(freq)
    # Case #1: we have a list ...................
    if dlist is not None:
        # Already a DateArray....................
        if isinstance(dlist, DateArray):
            if (freq != _c.FR_UND) and (dlist.freq != check_freq(freq)):
                return dlist.asfreq(freq)
            else:
                return dlist
        # Make sure it's a sequence, else that's a start_date
        if hasattr(dlist,'__len__'):
            return _listparser(dlist, freq)
        elif start_date is not None:
            if end_date is not None:
                dmsg = "What starting date should be used ? '%s' or '%s' ?"
                raise DateError, dmsg % (dlist, start_date)
            else:
                (start_date, end_date) = (dlist, start_date)
        else:
            start_date = dlist
    # Case #2: we have a starting date ..........
    if start_date is None:
        if length == 0:
            return DateArray([], freq=freq)
        raise InsufficientDateError
    if not isinstance(start_date, Date):
        dmsg = "Starting date should be a valid Date instance! "
        dmsg += "(got '%s' instead)" % type(start_date)
        raise DateError, dmsg
    # Check if we have an end_date
    if end_date is None:
        if length is None:
            length = 1
    else:
        if not isinstance(end_date, Date):
            raise DateError, "Ending date should be a valid Date instance!"
        length = int(end_date - start_date) + 1
    dlist = np.arange(length, dtype=int_)
    dlist += start_date.value
    if freq == _c.FR_UND:
        freq = start_date.freq
    return DateArray(dlist, freq=freq)

#####---------------------------------------------------------------------------
#---- --- Definition of functions from the corresponding methods ---
#####---------------------------------------------------------------------------
class _frommethod(object):
    """Defines functions from existing MaskedArray methods.
:ivar _methodname (String): Name of the method to transform.
    """
    def __init__(self, methodname):
        self._methodname = methodname
        self.__doc__ = self.getdoc()
    def getdoc(self):
        "Returns the doc of the function (from the doc of the method)."
        try:
            return getattr(DateArray, self._methodname).__doc__
        except AttributeError:
            return "???"
    #
    def __call__(self, caller, *args, **params):
        if hasattr(caller, self._methodname):
            method = getattr(caller, self._methodname)
            # If method is not callable, it's a property, and don't call it
            if hasattr(method, '__call__'):
                return method.__call__(*args, **params)
            return method
        method = getattr(np.asarray(caller), self._methodname)
        try:
            return method(*args, **params)
        except SystemError:
            return getattr(np,self._methodname).__call__(caller, *args, **params)
#............................
weekday = _frommethod('weekday')
week = _frommethod('week')
day_of_year = _frommethod('day_of_year')
year = _frommethod('year')
quarter = _frommethod('quarter')
month = _frommethod('month')
day = _frommethod('day')
hour = _frommethod('hour')
minute = _frommethod('minute')
second = _frommethod('second')


def period_break(dates, period):
    """Returns the indices where the given period changes.

    Parameters
    ----------
    dates : DateArray
        Array of dates to monitor.
    period : string
        Name of the period to monitor.
    """
    current = getattr(dates, period)
    previous = getattr(dates-1, period)
    return (current - previous).nonzero()[0]


def convert_to_float(datearray, ofreq):
    """
    Convert a :class:`~scikits.timeseries.DateArray` from a ndarray of integer
    at the given frequency to a ndarray of float at a lower frequency.

    Parameters
    ----------
    datearray : DateArray
        Input :class:`~scikits.timeseries.DateArray` to convert.
    ofreq : var
        Valid frequency specifier.

    Notes
    -----
    This function is currently restricted to conversion between annual (``'A'``),
    quarterly (``'Q'``), monthly (``'M'``) and daily (``'D'``) frequencies only.
    """
    if not isinstance(datearray, DateArray):
        raise TypeError("The input should be a valid DateArray instance !"\
                        " (got '%s' instead)" % type(datearray))
    errmsg = "Not implemented for the frequencies ('%s', '%s')"
    #
    freqdict = dict([(f, check_freq(f)) for f in ('A','Q','M','D')])
    ifreq = datearray.freq
    ofreq = check_freq(ofreq)
    errmsg = "Not implemented for the frequencies ('%s', '%s')" % \
             (check_freq_str(ifreq), check_freq_str(ofreq))
    if ifreq < ofreq:
        output = datearray.asfreq(ofreq).tovalue().astype(float)
    elif ifreq == ofreq:
        output = datearray.tovalue().astype(float)
    # Quarterly.........
    elif (ifreq >= freqdict['Q']) and (ifreq < freqdict['M']):
        if (ofreq >= freqdict['A']) and (ofreq < freqdict['Q']):
            output = datearray.years.astype(float) + (datearray.quarters -1.)/4.
    # Monthly...........
    elif ifreq == freqdict['M']:
        #... to annual
        if (ofreq >= freqdict['A']) and (ofreq < freqdict['Q']):
            output = datearray.years.astype(float) + (datearray.months - 1)/12.
        else:
            raise NotImplementedError(errmsg)
    # Daily ............
    elif ifreq == freqdict['D']:
        # ... to annual
        if (ofreq >= freqdict['A']) and (ofreq < freqdict['Q']):
            output = datearray.asfreq('A')
            output = output.tovalue().astype(float) + \
                     (datearray.yeardays-1.) / output.yeardays.astype(float)
        # ... to quarterly
        elif (ofreq >= freqdict['Q']) and (ofreq < freqdict['M']):
            raise NotImplementedError
        # ... to monthly
        elif ofreq == freqdict['M']:
            output = datearray.asfreq('M')
            output = output.tovalue().astype(float) + \
                     (datearray.days-1.) / output.days.astype(float)
        # ... to other
        else:
            raise NotImplementedError(errmsg)
    # Less than daily
    elif ifreq > freqdict['D']:
        raise NotImplementedError(errmsg)
    else:
        raise NotImplementedError(errmsg)
    return output
