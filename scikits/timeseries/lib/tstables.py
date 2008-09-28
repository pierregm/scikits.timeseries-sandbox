"""
.. currentmodule:: scikits.timeseries.lib.tstables

:mod:`~scikits.timeseries.lib.tstables`
=======================================

Interface between :mod:`scikits.timeseries` and :mod:`PyTables`.
:mod:`PyTables` is a Python package "for managing hierarchical datasets and 
designed to efficiently and easily cope with extremely large amounts of data".
:mod:`PyTables` is available at this `URL <http://www.pytables.org/moin>`_.


The :mod:`~scikits.timeseries.lib.tstables` module defines two new objects,
:class:`MaskedTable` and :class:`TimeSeriesTable` designed to store MaskedArrays
and TimeSeries objects as tables.
It also introduces new methods to the :class:`tables.File` object.


Classes
-------

.. class:: MaskedTable

   Based on :class:`tables.Table`
   
   Stores a :class:`~numpy.ma.MaskedArray` into a table.
   The initial :class:`~numpy.ma.MaskedArray` is transformed into a flexible-type
   standard ndarray before storage:

   * if the initial :class:`~numpy.ma.MaskedArray` has no named fields (standard
     type), the resulting ndarray has two named fields: ``_data`` and ``_mask``.
     The ``_data`` field has the same type as the original array, while the
     ``_mask`` field is always boolean.
     Note that a mask is always created, even if the initial
     :class:`~numpy.ma.MaskedArray` has no missing values.

   * if the initial :class:`~numpy.ma.MaskedArray` has named fields (flexible
     type), the resulting ndarray will have as many fields as the initial array,
     with the same name.
     Each field will have two nested subfields, ``_data`` and ``_mask``.
     The ``_data`` subfield will have the same dtype as the original field, 
     while the ``_mask`` subfield will always be boolean.
     For example, if the dtype of the initial array is:
     
        >>> input_dtype = [('a', int), ('b', float), ('c', '|S3')]
     
     The resulting ndarray will have a dtype:
     
        >>> output_dtype = [('a', [('_data', '<i8'), ('_mask', '|b1')]),
        ...                 ('b', [('_data', '<f8'), ('_mask', '|b1')]), 
        ...                 ('c', [('_data', '|S3'), ('_mask', '|b1')])]

   When a :class:`MaskedTable` is read, it is automatically retransformed into a
   :class:`~numpy.ma.MaskedArray`.

   Additional information about the :class:`~numpy.ma.MaskedArray` (such as 
   ``_baseclass``, ``_fill_value``...) is stored into a dictionary named
   ``special_attrs``.
   This information can be accessed through the :attr:`attr` attribute of the
   table.

   .. automethod:: MaskedTable.read



.. class:: TimeSeriesTable
   
   Based on :class:`MaskedTable`
   
   Stores a :class:`~scikits.timeseries.TimeSeries` object into a table.
   The initial :class:`~scikits.timeseries.TimeSeries` is transformed into a
   flexible-type standard ndarray before storage:

   * if the initial :class:`~scikits.timeseries.TimeSeries` has no named fields
     (standard dtype), the resulting ndarray has three named fields: ``_dates``,
     ``_data`` and ``_mask``.
     The ``_data`` field has the same type as the original array, while the
     ``_mask`` field is always boolean.
     The ``_dates`` field is always integer.
     Note that a mask is always created, even if the initial
     :class:`~scikits.timeseries.TimeSeries` has no missing values.

   * if the initial :class:`~scikits.timeseries.TimeSeries` has named fields
     (flexible type), the resulting ndarray will have the same fields as the
     initial array, with the same name, with the addition of an extra field
     ``_dates``.
     Each field (except ``_dates``) will have two nested subfields, ``_data``
     and ``_mask``.
     The ``_data`` subfield will have the same dtype as the original field, 
     while the ``_mask`` subfield will always be boolean.
     The ``_dates`` field is always integer.


   Additional information about the :class:`~scikits.timeseries.TimeSeries`
   (such as ``_baseclass``, ``_fill_value``, or the frequency attribute of the 
   ``_dates``...) is stored into a dictionary named ``special_attrs``.
   This information can be accessed through the :attr:`attr` attribute of the
   table.


   To create a :class:`TimeSeriesTable`, just use the
   :meth:`File.createTimeSeriesTable` method of a standard :class:`tables.File`
   object.


   .. automethod:: TimeSeriesTable.read


Methods
-------

.. method:: tables.File.createMaskedTable(where, name, maskedarray, title="", filters=None, expectedrows=10000, chunkshape=None, byteorder=None, createparents=False`)

   Use this method to create a new :class:`MaskedTable` object.
   This method accepts the same input parameters as the standard
   :meth:`tables.File.createTable`.



.. method:: tables.File.createTimeSeriesTable(where, name, series, title="", filters=None, expectedrows=10000, chunkshape=None, byteorder=None, createparents=False)

   Use this method to create a new :class:`TimeSeriesTable` object.
   This method accepts the same input parameters as the standard
   :meth:`tables.File.createTable`.


"""



import itertools

import numpy as np
import numpy.ma as ma

from numpy.ma import MaskedArray, masked

from scikits.timeseries import TimeSeries, DateArray, time_series

import tables
from tables import File, Table
from tables.file import _checkfilters
from tables.parameters import EXPECTED_ROWS_TABLE


_doc_parameters = dict(
mareturn="""
    Depending on the value of the ``field`` parameter, the method returns either
    * a ndarray, if ``field=='_data'`` or if ``field=='_mask'``;
    * a :class:`~numpy.ma.MaskedArray`, if ``field`` is None or a valid field.
""",
tsreturn="""
    Depending on the value of the ``field`` parameter, the method returns:
    
    * a :class:`~scikits.timeseries.TimeSeries`, if ``field`` is None or a valid
      field;
    * a :class:`~scikits.timeseries.DateArray`, if ``field=='_dates'``;
    * a ndarray, if ``field=='_data'`` or if ``field=='_mask'``;
    * a :class:`~numpy.ma.MaskedArray`, if ``field=='_series'``.
""",
readinput="""
    start : {None, int}, optional
        Index of the first record to read.
        If None, records will be read starting from the very first one.
    stop : {None, int}, optional
        Index of the last record to read.
        If None, records will be read until the very last one.
    step : {None, int}, optional
        Increment between succesive records to read.
        If None, all the records between ``start`` and ``stop`` will be read.
    field : {None, str}, optional
        Name of the field to read.
        If None, all the fields from each record are read.
""",
readcoordinateinput="""
    coords : sequence
        A sequence of integers, corresponding to the indices of the rows to
        retrieve
    field : {None, str}, optional
        Name of the field to read.
        If None, all the fields from each record are read.
""",
)



def _tabulate_masked_array(a):
    """
    Private function called by tabulate for flexible-dtype MaskedArrays.
    """
    basedtype = a.dtype
    basenames = basedtype.names
    if basenames is None:
        recshape = tuple(a.shape[1:])
        if recshape != ():
            a = a.reshape(len(a),-1)
            nrep = a.shape[1]
            pseudodtype = [('_data',(basedtype,nrep)), ('_mask',(bool,nrep))]
        else:
            pseudodtype = [('_data',basedtype),('_mask',bool)]
        pseudo = itertools.izip(ma.filled(a), ma.getmaskarray(a))
    else:
        pseudodtype = [(fname,[('_data',ftype), ('_mask',bool)])
                       for (fname,ftype) in basedtype.descr]
        fields = [a[f] for f in basenames]
        pseudo = itertools.izip(*[zip(ma.filled(f).flat,ma.getmaskarray(f).flat)
                                 for f in fields])
    return np.fromiter(pseudo, dtype=pseudodtype)


def _tabulate_time_series(a):
    """
    Private function called by tabulate for flexible-dtype TimeSeries.
    """
    basedtype = a.dtype
    basenames = basedtype.names
    if basenames is None:
        _varshape = a._varshape
        if _varshape != ():
            pseudodtype = [('_dates', int),
                           ('_data',(basedtype, _varshape)),
                           ('_mask',(bool,_varshape))]
        else:
            pseudodtype = [('_dates', int),
                           ('_data', basedtype),
                           ('_mask', bool)]
        pseudo = itertools.izip(a._dates, a.filled(), ma.getmaskarray(a),)
    else:
        pseudodtype = [('_dates', int)]
        pseudodtype.extend([(fname,[('_data',ftype), ('_mask',bool)])
                            for (fname,ftype) in basedtype.descr])
        fields = [a[f] for f in basenames]
        pseudo = itertools.izip(a._dates,
                                *[zip(f.filled().flat, ma.getmaskarray(f).flat)
                                  for f in fields])
    return np.fromiter(pseudo, dtype=pseudodtype)


def tabulate(a):
    """
    Transforms a MaskedArray into a flexible ndarray, for easier insertion.
    If the initial masked array has no named fields, the fields of the output
    are ``_data`` and ``_mask``.
    If the initial array has named fields, the fields of the output are nested:
    the names of the fields are the same as the original ones; each field has
    two sub-fields, ``_data`` and ``_mask``.
    
    Parameters
    ----------
    a : MaskedArray
        Input array.
    
    See Also
    --------
    numpy.ma.MaskedArray.torecords
        Transforms a masked-array into a flexible-type ndarray with only two fields,
        ``_data`` and ``_mask``.
    
    Returns
    -------
    output : ndarray
        A ndarray with flexible dtype.
    """
    a = np.asanyarray(a)
    if isinstance(a, TimeSeries):
        return _tabulate_time_series(a)
    else:
        return _tabulate_masked_array(a)



class MaskedTable(Table):
    _c_classId = 'MaskedTable'
    def __init__(self, parentNode, name, description=None,
                 title="", filters=None,
                 expectedrows=EXPECTED_ROWS_TABLE,
                 chunkshape=None, byteorder=None, _log=True):
        new = description is None
        if not new:
            maskedarray = ma.asanyarray(description)
            description = tabulate(maskedarray)
        Table.__init__(self, parentNode, name, 
                       description=description, title=title,
                       filters=filters,
                       expectedrows=expectedrows,
                       chunkshape=chunkshape, byteorder=byteorder,
                       _log=_log)
        if not new:
            self.attrs.special_attrs = self._update_special_attrs(maskedarray)
        return


    def _update_special_attrs(self, a):
        #
        special_attrs = {}
        special_attrs.update(dict(_baseclass=a._baseclass,
                                  _fill_value=a._fill_value,
                                  _hardmask=a._hardmask,
                                  _optinfo=a._optinfo))
        special_attrs['recshape'] = tuple(a.shape[1:])
        return special_attrs


    def _get_dtype(self):
        #
        _v_dtype = self._v_dtype
        try:
            dtypedict = dict(_v_dtype.descr)
        except ValueError:
            dtypedict = dict([(_[0], _[1]) for _ in _v_dtype.descr])
        #
        field_names = list(_v_dtype.names)
        #
        if field_names == ['_data', '_mask']:
            return np.dtype(dtypedict['_data'])
        else:
            return np.dtype([(name, dict(dtypedict[name])['_data'])
                             for name in field_names])


    def _reader(self, meth, *args, **kwargs):
        """
    Private function that retransforms the output of Table.read and equivalent
    to the proper type.
        """
        data = meth(self, *args, **kwargs)
        special_attrs = getattr(self.attrs, 'special_attrs', {})
        fill_value = special_attrs.get('_fill_value', None)
        #
        ndtype = self._get_dtype()
        field_names = ndtype.names
        field = kwargs.get('field', None)
        #
        if field in ['_data','_mask']:
            output = data
        else:
            if (field_names is None) or (field in field_names):
                output = ma.array(data['_data'], mask=data['_mask'])
            else:
                output = ma.empty(data.shape, dtype=ndtype)
                for name in field_names:
                    current = data[name]
                    output[name] = ma.array(current['_data'],
                                            mask=current['_mask'])
            # Reset some attributes..................
            output._baseclass = special_attrs.get('_baseclass', np.ndarray)
            fill_value = special_attrs.get('_fill_value', None)
            if (field is not None) and (fill_value is not None):
                output.fill_value = fill_value[field]
            else:
                output.fill_value = fill_value
            output._hardmask = special_attrs.get('_hardmask', False)
            output._optinfo = special_attrs.get('_optinfo', {})
        recshape = special_attrs.get('recshape',())
        if recshape != ():
            output.shape = tuple([-1,]+list(recshape))
        return output


    def read(self, start=None, stop=None, step=None, field=None):
        """
    Reads the current :class:`MaskedTable`.
    
    Returns
    -------
    %(mareturn)s
    
    Parameters
    ----------
    %(readinput)s
        """
        args = ()
        kwargs = dict(field=field, start=start, stop=stop, step=step)
        return self._reader(Table.read, *args, **kwargs)
    read.__doc__ = ((read.__doc__ or '') % _doc_parameters) or None


    def readCoordinates(self, coords, field=None):
        """
    Reads a set of rows given their coordinates.

    Parameters
    ----------
    %(readcoordinateinput)s
    
    Returns
    -------
    %(mareturn)s
        """
        args = (coords,)
        kwargs = dict(field=field)
        return self._reader(Table.readCoordinates, *args, **kwargs)
    readCoordinates.__doc__ = ((readCoordinates.__doc__ or '') % _doc_parameters) or None

    def append(self, rows):
        """
        """
        rows = tabulate(rows)
        Table.append(self, rows)


class TimeSeriesTable(MaskedTable):
    """
    """
    _c_classId = 'TimeSeriesTable'
    def __init__(self,parentNode, name, description=None,
                 title="", filters=None,
                 expectedrows=EXPECTED_ROWS_TABLE,
                 chunkshape=None, byteorder=None, _log=True):
        """
        """
        MaskedTable.__init__(self, parentNode, name, description=description,
                             title=title, filters=filters,
                             expectedrows=expectedrows,
                             chunkshape=chunkshape, byteorder=byteorder,
                             _log=_log)


    def _update_special_attrs(self, a):
        #
        special_attrs = MaskedTable._update_special_attrs(self, a)
        special_attrs['recshape'] = tuple(a._varshape)
        special_attrs['freq'] = a._dates.freq
        return special_attrs


    def _get_dtype(self):
        #
        _v_dtype = self._v_dtype
        try:
            dtypedict = dict(_v_dtype.descr)
        except ValueError:
            dtypedict = dict([(_[0], _[1]) for _ in _v_dtype.descr])
        dtypedict.pop('_dates',None)
        #
        field_names = list(_v_dtype.names)
        field_names.remove('_dates')
        #
        if field_names == ['_data', '_mask']:
            return np.dtype(dtypedict['_data'])
        else:
            return np.dtype([(name, dict(dtypedict[name])['_data'])
                             for name in field_names])


#    def read(self, start=None, stop=None, step=None, field=None):
#        """
#    Reads the current :class:`TimeSeriesTable`.
#    
#    Returns
#    -------
#    
#    Depending on the value of the ``field`` parameter, the method returns:
#    
#    * a :class:`~scikits.timeseries.TimeSeries`, if ``field`` is None or a valid
#      field;
#    * a :class:`~scikits.timeseries.DateArray`, if ``field=='_dates'``;
#    * a ndarray, if ``field=='_data'`` or if ``field=='_mask'``;
#    * a :class:`~numpy.ma.MaskedArray`, if ``field=='_series'``.
#    
#    
#    Parameters
#    ----------
#    start : {None, int}, optional
#        Index of the first record to read.
#        If None, records will be read starting from the very first one.
#    stop : {None, int}, optional
#        Index of the last record to read.
#        If None, records will be read until the very last one.
#    step : {None, int}, optional
#        Increment between succesive records to read.
#        If None, all the records between ``start`` and ``stop`` will be read.
#    field : {None, str}, optional
#        Name of the field to read.
#        If None, all the fields from each record are read.
#        The argument should be one of the field of the series, or one of the
#        following: ``'_data','_mask','_dates','_series'``.
#        """
##        data = Table.read(self, start=start, stop=stop, step=step,
##                          field=field)
#        special_attrs = getattr(self.attrs, 'special_attrs', {})
#        fill_value = special_attrs.get('_fill_value', None)
#        baseclass = special_attrs.get('_baseclass', np.ndarray)
#        #
#        position_keywords = dict(start=start, stop=stop, step=step)
#        #
#        ndtype = self._get_dtype()
#        field_names = ndtype.names
#        
#        # Case 1. : Global read .................
#        if field is None:
#            data = Table.read(self, **position_keywords)
#            dates = DateArray(data['_dates'],
#                              freq=special_attrs.get('freq','U'))
#            if field_names is None:
#                output = time_series(data['_data'],
#                                     dates = dates,
#                                     mask=data['_mask'])
#            else:
#                output = ma.empty(data.shape, dtype=ndtype).view(TimeSeries)
#                for name in field_names:
#                    current = data[name]
#                    output[name] = ma.array(current['_data'],
#                                            mask=current['_mask'])
#                output._dates = dates
#            # Reset some attributes..................
#            output._baseclass = baseclass
#            output.fill_value = fill_value
#            output._hardmask = special_attrs.get('_hardmask', False)
#            output._optinfo = special_attrs.get('_optinfo', {})
#        # Case 2. Partial reads..................
#        elif field in ['_dates','_data','_mask']:
#            output = Table.read(self, field=field, **position_keywords)
#        # Case 3. The series as a masked array
#        elif field == '_series':
#            # Special case: read the table, but keep it as MaskedArray
#            data = Table.read(self, field=None, **position_keywords)
#            if field_names is None:
#                output = ma.array(data['_data'], mask=data['_mask'])
#            else:
#                output = ma.empty(data.shape, dtype=ndtype)
#                for name in field_names:
#                    current = data[name]
#                    output[name] = ma.array(current['_data'],
#                                            mask=current['_mask'])
#            output.fill_value = fill_value
#            output._baseclass = baseclass
#            output._hardmask = special_attrs.get('_hardmask', False)
#            output._optinfo = special_attrs.get('_optinfo', {})
#        # Case 4. Field read ....................
#        elif field in field_names:
#            data = Table.read(self, field=field, **position_keywords)
#            dates = Table.read(self, field='_dates', **position_keywords)
#            dates = DateArray(dates, freq=special_attrs.get('freq','U'))
#            # Get the data part
#            output = time_series(data['_data'],
#                                 dates=dates,
#                                 mask=data['_mask'],)
#            output._baseclass = baseclass
#            if fill_value is not None:
#                output.fill_value = fill_value[field]
#            output._hardmask = special_attrs.get('_hardmask', False)
#            output._optinfo = special_attrs.get('_optinfo', {})
#        else:
#            raise KeyError("Unable to process field '%s'" % field)
#        return output


    def _reader(self, meth, *args, **kwargs):
        """
    Private function that retransforms the output of Table.read and equivalent
    to the proper type.
        """
        # Get some of the special attributes
        special_attrs = getattr(self.attrs, 'special_attrs', {})
        fill_value = special_attrs.get('_fill_value', None)
        baseclass = special_attrs.get('_baseclass', np.ndarray)
        # Get the dtype, in particular, the names of the fields.
        ndtype = self._get_dtype()
        field_names = ndtype.names
        
        field = kwargs.get('field', None)
        # Case 1. : Global read .................
        if field is None:
            data = meth(self, *args, **kwargs)
            dates = DateArray(data['_dates'],
                              freq=special_attrs.get('freq','U'))
            if field_names is None:
                output = time_series(data['_data'],
                                     dates = dates,
                                     mask=data['_mask'])
            else:
                output = ma.empty(data.shape, dtype=ndtype).view(TimeSeries)
                for name in field_names:
                    current = data[name]
                    output[name] = ma.array(current['_data'],
                                            mask=current['_mask'])
                output._dates = dates
            # Reset some attributes..................
            output._baseclass = baseclass
            output.fill_value = fill_value
            output._hardmask = special_attrs.get('_hardmask', False)
            output._optinfo = special_attrs.get('_optinfo', {})
        # Case 2. Partial reads..................
        elif field in ['_dates','_data','_mask']:
            output = meth(self, *args, **kwargs)
        # Case 3. The series as a masked array
        elif field == '_series':
            # Special case: read the table, but keep it as MaskedArray
            kwargs['field'] = None
            data = meth(self, *args, **kwargs)
            if field_names is None:
                output = ma.array(data['_data'], mask=data['_mask'])
            else:
                output = ma.empty(data.shape, dtype=ndtype)
                for name in field_names:
                    current = data[name]
                    output[name] = ma.array(current['_data'],
                                            mask=current['_mask'])
            output.fill_value = fill_value
            output._baseclass = baseclass
            output._hardmask = special_attrs.get('_hardmask', False)
            output._optinfo = special_attrs.get('_optinfo', {})
        # Case 4. Field read ....................
        elif field in field_names:
            data = meth(self, *args, **kwargs)
            kwargs['field'] = '_dates'
            dates = meth(self, *args, **kwargs)
            dates = DateArray(dates, freq=special_attrs.get('freq','U'))
            # Get the data part
            output = time_series(data['_data'],
                                 dates=dates,
                                 mask=data['_mask'],)
            output._baseclass = baseclass
            if fill_value is not None:
                output.fill_value = fill_value[field]
            output._hardmask = special_attrs.get('_hardmask', False)
            output._optinfo = special_attrs.get('_optinfo', {})
        else:
            raise KeyError("Unable to process field '%s'" % field)
        return output


    def readCoordinates(self, coords, field=None):
        """
    Reads a set of rows given their coordinates.

    Parameters
    ----------
    %(readcoordinateinput)s
    
    Returns
    -------
    %(tsreturn)s
        """
        args = (coords,)
        kwargs = dict(field=field)
        return self._reader(Table.readCoordinates, *args, **kwargs)
    readCoordinates.__doc__ = ((readCoordinates.__doc__ or '') % _doc_parameters) or None


    def read(self, start=None, stop=None, step=None, field=None):
        """
    Reads the current :class:`TimeSeriesTable`.
    
    Returns
    -------
    %(tsreturn)s
    
    Parameters
    ----------
    %(readcoordinateinput)s
        """
        args = ()
        kwargs = dict(field=field, start=start, stop=stop, step=step)
        return self._reader(Table.read, *args, **kwargs)
    read.__doc__ = ((read.__doc__ or '') % _doc_parameters) or None


#-- File extensions -----------------------------------------------------------                       

def createMaskedTable(self, where, name, maskedarray, title="",
                      filters=None, expectedrows=10000,
                      chunkshape=None, byteorder=None,
                      createparents=False):
    """
    Creates a :class:`MaskedTable` from a masked array.

    Parameters
    ----------
    where : Group
        Location of the table.
    name : string
        Name of the table.
    maskedarray : MaskedArray
        Masked array to store
    title : {'', string}, optional
        Title of the table
    """
    parentNode = self._getOrCreatePath(where, createparents)
    _checkfilters(filters)
    return MaskedTable(parentNode, name, maskedarray,
                       title=title, filters=filters, 
                       expectedrows=expectedrows,
                       chunkshape=chunkshape, byteorder=byteorder)
File.createMaskedTable = createMaskedTable



def createTimeSeriesTable(self, where, name, series, title="",
                          filters=None, expectedrows=10000,
                          chunkshape=None, byteorder=None,
                          createparents=False):
    """
    Creates a :class:`TimeSeriesTable` from
    a :class:`~scikits.timeseries.TimeSeries` object.

    Parameters
    ----------
    where : Group
        Location of the table.
    name : string
        Name of the table.
    series : TimeSeries
        Series to store
    """
    parentNode = self._getOrCreatePath(where, createparents)
    _checkfilters(filters)
    return TimeSeriesTable(parentNode, name, series, title=title,
                           filters=filters, expectedrows=expectedrows,
                           chunkshape=chunkshape, byteorder=byteorder)
File.createTimeSeriesTable = createTimeSeriesTable


################################################################################

