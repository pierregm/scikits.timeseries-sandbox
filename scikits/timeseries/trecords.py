# pylint: disable-msg=W0201, W0212
"""
Support for multi-variable time series, through masked recarrays.

:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com
:version: $Id: trecords.py 3822 2008-01-12 10:06:39Z matthew.brett@gmail.com $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: matthew.brett@gmail.com $)"
__version__ = '1.0'
__revision__ = "$Revision: 3822 $"
__date__     = '$Date: 2008-01-12 05:06:39 -0500 (Sat, 12 Jan 2008) $'


import sys

import numpy as np
from numpy import bool_, complex_, float_, int_, str_, object_, \
    ndarray, chararray, recarray
import numpy.core.numerictypes as ntypes
import numpy.core.umath as umath
from numpy.core.records import find_duplicate, format_parser, record, \
    fromarrays as recfromarrays

import numpy.ma as ma
from numpy.ma import MaskedArray, MAError, \
     default_fill_value, masked_print_option, masked, nomask, \
     getmask, getmaskarray, make_mask, make_mask_none, mask_or, filled

import numpy.ma.mrecords
from numpy.ma.mrecords import _checknames, \
     _guessvartypes, openfile, MaskedRecords, mrecarray, addfield, \
     fromrecords as mrecfromrecords, fromarrays as mrecfromarrays

from tseries import TimeSeries, time_series, _getdatalength
from tdates import Date, DateArray, date_array

_byteorderconv = numpy.core.records._byteorderconv
_typestr = ntypes._typestr

reserved_fields = numpy.ma.mrecords.reserved_fields + ['_dates']

import warnings

__all__ = [
'TimeSeriesRecords','time_records',
'fromarrays','fromrecords','fromtextfile',
]

def _getformats(data):
    """Returns the formats of each array of arraylist as a comma-separated
    string."""
    if isinstance(data, record):
        return ",".join([desc[1] for desc in data.dtype.descr])

    formats = ''
    for obj in data:
        obj = np.asarray(obj)
        formats += _typestr[obj.dtype.type]
        if issubclass(obj.dtype.type, ntypes.flexible):
            formats += `obj.itemsize`
        formats += ','
    return formats[:-1]


def _getdates(dates=None, newdates=None, length=None, freq=None,
              start_date=None):
    """Determines new dates (private function not meant to be used)."""
    if dates is None:
        if newdates is not None:
            if not hasattr(newdates, 'freq'):
                newdates = date_array(dlist=newdates, freq=freq)
        else:
            newdates = date_array(start_date=start_date, length=length,
                                  freq=freq)
    elif not hasattr(dates, 'freq'):
        newdates = date_array(dlist=dates, freq=freq)
    else:
        newdates = dates
    return newdates


class TimeSeriesRecords(TimeSeries, MaskedRecords, object):
    """

:IVariables:
    - `__localfdict` : Dictionary
        Dictionary of local fields (`f0_data`, `f0_mask`...)
    - `__globalfdict` : Dictionary
        Dictionary of global fields, as the combination of a `_data` and a `_mask`.
        (`f0`)
    """
    _defaultfieldmask = nomask
    _defaulthardmask = False
    def __new__(cls, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False,
                mask=nomask, hard_mask=False, fill_value=None, keep_mask=True,
                copy=False,
                dates=None, freq='U', start_date=None, observed=None,
                **options):
        _data = mrecarray.__new__(cls, shape, dtype=dtype, buf=buf, offset=offset,
                                  strides=strides, formats=formats,
                                  byteorder=byteorder, aligned=aligned,
                                  mask=mask, hard_mask=hard_mask, copy=copy,
                                  keep_mask=keep_mask, fill_value=fill_value,
                                  )
        #
        newdates = _getdates(dates, length=len(_data),
                             start_date=start_date,freq=freq)
        _data._dates = newdates
        _data._observed = observed
        cls._defaultfieldmask = _data._fieldmask
        #
        return _data

    def __array_finalize__(self,obj):
        _dates = getattr(obj,'_dates',DateArray([]))
        self.__dict__.update(_dates=_dates,
                             _observed=getattr(obj,'_observed',None),
                             _names = self.dtype.names)
        MaskedRecords.__array_finalize__(self, obj)
        return
    #......................................................
    def _getdata(self):
        "Returns the data as a recarray."
        return self.view(recarray)
    _data = property(fget=_getdata)

    def _getseries(self):
        "Returns the data as a MaskedRecord array."
        return self.view(mrecarray)
    _series = property(fget=_getseries)
    #......................................................
    def __getattribute__(self, attr):
        getattribute = MaskedRecords.__getattribute__
        _dict = getattribute(self,'__dict__')
        if attr in _dict.get('_names',[]):
            obj = getattribute(self,attr).view(TimeSeries)
            obj._dates = _dict['_dates']
            return obj
        return getattribute(self,attr)

    #......................................................
    def __getitem__(self, indx):
        """Returns all the fields sharing the same fieldname base.
    The fieldname base is either `_data` or `_mask`."""
        _localdict = self.__dict__
        # We want a field ........
        if indx in self.dtype.names:
            obj = self._data[indx].view(TimeSeries)
            obj._dates = _localdict['_dates']
            obj._mask = make_mask(_localdict['_fieldmask'][indx])
            return obj
        # We want some elements ..
        (sindx, dindx) = self._TimeSeries__checkindex(indx)
        obj = np.array(self._data[sindx], copy=False, subok=True).view(type(self))
        obj.__dict__.update(_dates=_localdict['_dates'][dindx],
                            _fill_value=_localdict['_fill_value'])
        obj._fieldmask = np.array(_localdict['_fieldmask'][sindx]).view(recarray)
        return obj

    def __getslice__(self, i, j):
        """Returns the slice described by [i,j]."""
        _localdict = self.__dict__
        (si, di) = super(TimeSeriesRecords, self)._TimeSeries__checkindex(i)
        (sj, dj) = super(TimeSeriesRecords, self)._TimeSeries__checkindex(j)
        newdata = self._data[si:sj].view(type(self))
        newdata.__dict__.update(_dates=_localdict['_dates'][di:dj],
                                _mask=_localdict['_fieldmask'][si:sj])
        return newdata

    def __setslice__(self, i, j, value):
        """Sets the slice described by [i,j] to `value`."""
        self.view(mrecarray).__setslice__(i,j,value)
        return

    #......................................................
    def __str__(self):
        """x.__str__() <==> str(x)
Calculates the string representation, using masked for fill if it is enabled.
Otherwise, fills with fill value.
        """
        if self.size > 1:
            mstr = ["(%s)" % ",".join([str(i) for i in s])
                    for s in zip(*[getattr(self,f) for f in self.dtype.names])]
            return "[%s]" % ", ".join(mstr)
        else:
            mstr = ["%s" % ",".join([str(i) for i in s])
                    for s in zip([getattr(self,f) for f in self.dtype.names])]
            return "(%s)" % ", ".join(mstr)

    def __repr__(self):
        """x.__repr__() <==> repr(x)
Calculates the repr representation, using masked for fill if it is enabled.
Otherwise fill with fill value.
        """
        _names = self.dtype.names
        _dates = self._dates
        if np.size(_dates) > 2 and self._dates.isvalid():
            timestr = "[%s ... %s]" % (str(_dates[0]),str(_dates[-1]))
        else:
            timestr = str(_dates)
        fmt = "%%%is : %%s" % (max([len(n) for n in _names])+4,)
        reprstr = [fmt % (f,getattr(self,f)) for f in self.dtype.names]
        reprstr.insert(0,'TimeSeriesRecords(')
        reprstr.extend([fmt % ('dates', timestr),
                        fmt % ('    fill_value', self._fill_value),
                         '               )'])
        return str("\n".join(reprstr))
    #.............................................
    def copy(self):
        "Returns a copy of the argument."
        copied = MaskedRecords.copy(self)
        copied._dates = self._dates.copy()
        return copied
    #.............................................
    def convert(self, freq, func=None, position='END', *args, **kwargs):
        """Converts a series to a frequency. Private function called by convert

    Parameters
    ----------
    series : TimeSeries
        the series to convert. Skip this parameter if you are calling this as
        a method of the TimeSeries object instead of the module function.
    freq : freq_spec
        Frequency to convert the TimeSeries to. Accepts any valid frequency
        specification (string or integer)
    func : {None,function}, optional
        When converting to a lower frequency, func is a function that acts on
        one date's worth of data. func should handle masked values appropriately.
        If func is None, then each data point in the resulting series will a
        group of data points that fall into the date at the lower frequency.

        For example, if converting from monthly to daily and you wanted each
        data point in the resulting series to be the average value for each
        month, you could specify numpy.ma.average for the 'func' parameter.
    position : {'END', 'START'}, optional
        When converting to a higher frequency, position is 'START' or 'END'
        and determines where the data point is in each period. For example, if
        going from monthly to daily, and position is 'END', then each data
        point is placed at the end of the month.
    *args : {extra arguments for func parameter}, optional
        if a func is specified that requires additional parameters, specify
        them here.
    **kwargs : {extra keyword arguments for func parameter}, optional
        if a func is specified that requires additional keyword parameters,
        specify them here.
    
        """
        kwargs.update(func=func, position=position)
        field_names = self.dtype.names
        by_field = [self[f].convert(freq,**kwargs) for f in field_names]
        output = fromarrays(by_field, 
                            dates=by_field[0].dates,
                            names=field_names)
        output.fill_value = self.fill_value
        return output
trecarray = TimeSeriesRecords


#####---------------------------------------------------------------------------
#---- --- Constructors ---
#####---------------------------------------------------------------------------

def time_records(mrecord, dates=None):
    trecords = np.array(mrecord, subok=True).view(trecarray)
    trecords._dates = dates
    return trecords

#!!!: * The docstrings of the following functions need some serious work ;)
#!!!: * We should try to have a list of TimeSeries sufficient to build a record...
#!!!:   without having to precise a list of dates...
#!!!:   > check the compatibility of dates
#!!!:   > try to adjust endpoints if needed
#!!!:   > if one of the series is not a TimeSeries, keep going.

def fromarrays(arraylist, dates=None, start_date=None, freq='U',
               fill_value=None,
               dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None,):
    """Creates a mrecarray from a (flat) list of masked arrays.

    Parameters
    ----------
    arraylist : array_like
        A list of (masked) arrays. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : numeric.dtype
        Data type descriptor.
    shape : integer
        Number of records. If None, shape is defined from the shape of the
        first array in the list.
    formats : sequence
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : sequence
        Sequence of the names of each field.
    titles : sequence
      (Description to write)
    aligned : boolean
      (Description to write, not used anyway)
    byteorder: boolean
      (Description to write, not used anyway)
    fill_value : sequence
        Sequence of data to be used as filling values.

    Notes
    -----
    Lists of tuples should be preferred over lists of lists for faster processing.
    """
    _array = mrecfromarrays(arraylist, dtype=dtype, shape=shape, formats=formats,
                            names=names, titles=titles, aligned=aligned,
                            byteorder=byteorder, fill_value=fill_value)
    _array = _array.view(trecarray)
    _array._dates = _getdates(dates, length=len(_array),
                              start_date=start_date,freq=freq)
    return _array


#..............................................................................
def fromrecords(reclist, dates=None, freq=None, start_date=None,
                fill_value=None, mask=nomask,
                dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None):
    """Creates a MaskedRecords from a list of records.

    The data in the same field can be heterogeneous, they will be promoted
    to the highest data type.  This method is intended for creating
    smaller record arrays.  If used to create large array without formats
    defined, it can be slow.

    If formats is None, then this will auto-detect formats. Use a list of
    tuples rather than a list of lists for faster processing.
    """
    _data = mrecfromrecords(reclist, dtype=dtype, shape=shape, formats=formats,
                            names=names, titles=titles, aligned=aligned,
                            byteorder=byteorder, mask=mask)
    _dtype = _data.dtype
    # Check the names for a '_dates' .................
    newdates = None
    _names = list(_dtype.names)
    reserved = [n for n in _names if n.lower() in ['dates', '_dates']]
    if len(reserved) > 0:
        newdates = _data[reserved[-1]]
        [_names.remove(n) for n in reserved]
        _dtype = np.dtype([t for t in _dtype.descr \
                                    if t[0] not in reserved ])
        _data = mrecfromarrays([_data[n] for n in _names], dtype=_dtype)
    #
    result = _data.view(trecarray)
    if dates is None:
        dates = getattr(reclist, '_dates', None)
    result._dates = _getdates(dates=dates, newdates=newdates, length=len(_data),
                              freq=freq, start_date=start_date)
    #
    return result


def fromtextfile(fname, delimitor=None, commentchar='#', missingchar='',
                 dates_column=None, varnames=None, vartypes=None,
                 dates=None):
    """Creates a TimeSeriesRecords from data stored in the file `filename`.

:Parameters:
    - `filename` : file name/handle
      Handle of an opened file.
    - `delimitor` : Character *None*
      Alphanumeric character used to separate columns in the file.
      If None, any (group of) white spacestring(s) will be used.
    - `commentchar` : String *['#']*
      Alphanumeric character used to mark the start of a comment.
    - `missingchar` : String *['']*
      String indicating missing data, and used to create the masks.
    - `dates_column` : Integer *[None]*
      Position of the columns storing dates. If None, a position will be
      estimated from the variable names.
    - `varnames` : Sequence *[None]*
      Sequence of the variable names. If None, a list will be created from
      the first non empty line of the file.
    - `vartypes` : Sequence *[None]*
      Sequence of the variables dtypes. If None, the sequence will be estimated
      from the first non-commented line.


    Ultra simple: the varnames are in the header, one line"""
    # Try to open the file ......................
    f = openfile(fname)
    # Get the first non-empty line as the varnames
    while True:
        line = f.readline()
        firstline = line[:line.find(commentchar)].strip()
        _varnames = firstline.split(delimitor)
        if len(_varnames) > 1:
            break
    if varnames is None:
        varnames = _varnames
    # Get the data ..............................
    _variables = ma.asarray([line.strip().split(delimitor) for line in f
                             if line[0] != commentchar and len(line) > 1])
    (nvars, nfields) = _variables.shape
    # Check if we need to get the dates..........
    if dates_column is None:
        dates_column = [i for (i,n) in enumerate(list(varnames))
                            if n.lower() in ['_dates','dates']]
    elif isinstance(dates_column,(int,float)):
        if dates_column > nfields:
            raise ValueError,\
                  "Invalid column number: %i > %i" % (dates_column, nfields)
        dates_column = [dates_column,]
    if len(dates_column) > 0:
        cols = range(nfields)
        [cols.remove(i) for i in dates_column]
        newdates = date_array(_variables[:,dates_column[-1]])
        _variables = _variables[:,cols]
        varnames = [varnames[i] for i in cols]
        if vartypes is not None:
            vartypes = [vartypes[i] for i in cols]
        nfields -= len(dates_column)
    else:
        newdates = None
    # Try to guess the dtype ....................
    if vartypes is None:
        vartypes = _guessvartypes(_variables[0])
    else:
        vartypes = [np.dtype(v) for v in vartypes]
        if len(vartypes) != nfields:
            msg = "Attempting to %i dtypes for %i fields!"
            msg += " Reverting to default."
            warnings.warn(msg % (len(vartypes), nfields))
            vartypes = _guessvartypes(_variables[0])
    # Construct the descriptor ..................
    mdescr = [(n,f) for (n,f) in zip(varnames, vartypes)]
    mfillv = [ma.default_fill_value(f) for f in vartypes]
    # Get the data and the mask .................
    # We just need a list of masked_arrays. It's easier to create it like that:
    _mask = (_variables.T == missingchar)
    _datalist = [ma.array(a,mask=m,dtype=t,fill_value=f)
                 for (a,m,t,f) in zip(_variables.T, _mask, vartypes, mfillv)]
    #
    newdates = _getdates(dates=dates, newdates=newdates, length=nvars,
                         freq=None, start_date=None)

    # Sort the datalist according to newdates._unsorted
    idx = newdates._unsorted
    _sorted_datalist = [a[idx] for a in _datalist]
    return fromarrays(_sorted_datalist, dates=newdates, dtype=mdescr)


