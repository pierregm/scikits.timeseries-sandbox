.. sectionauthor:: Matt Knox <mattknox.ca@hotmail.com>

.. currentmodule:: scikits.timeseries

.. highlight:: python
   :linenothreshold: 5


.. _Pytables: http://www.pytables.org
.. _HDF5: http://www.hdfgroup.org/HDF5/


=========
Databases
=========

Two approaches for working with :class:`TimeSeries` objects are supported with
the :mod:`scikits.timeseries` module.
The first approach consists in using standard relational databases. 
The second is to use the PyTables_ package.
Both approaches have their pros and cons.

Generally speaking, if concurrent access to a database is required, or if 
an interface with an online server is needed (for example, if time series 
charts must be generated in real-time for a web site), relational databases 
are probably more convenient.

Alternatively, if the only objective is to store data for a specific project, 
without having to deal with the administrative complexities of a relational 
database, then PyTables_ may be enough.

It is recommended to experiment with both approaches in order to find which 
one is the most performant or the most adequate to any application.



Relational Databases
====================

.. sectionauthor:: Matt Knox <mattknox.ca@hotmail.com>

Storing and retrieving time series from standard relational databases is very
simple once you know a few tricks.
For these examples, we use the `ceODBC <http://ceodbc.sourceforge.net/>`_ 
database module which I have found to be more reliable and faster than the 
`pyodbc <http://code.google.com/p/pyodbc/>`_ module.
However, I *think* these examples should work with the pyodbc module as well.

SQL Server 2005 Express edition is the database used in the examples. Other
standard relational databases should also work, but I have not personally
verified it.

Note that it is also possible to use this approach with sqlite databases using
the sqlite3 module included with the standard python distribution. The below
code will need to be modified a bit to work with sqlite, but the basic approach
remains the same.

A database called "test" is assumed to have been created already along with a
table called "test_table" described by the following query:

.. code-block:: mysql
   :linenos:

   CREATE TABLE  test_table (
       [date] [datetime] NULL,
       [value] [decimal](18, 6) NULL
   )


Example
-------


.. code-block:: python
   :linenos:

   import ceODBC as odbc
   import scikits.timeseries as ts

   test_series = ts.time_series(range(50), start_date=ts.now('b'))

   # lets mask one value just to make things interesting
   test_series[5] = ts.masked

   conn = odbc.Connection(
   "Driver={SQL Native Client};Server=localhost;Database=test;Uid=userid;Pwd=password;")
   crs = conn.cursor()

   # start with an empty table for these examples
   crs.execute("DELETE FROM test_table")

   # convert series to list of (datetime, value) tuples which can be interpreted
   # by the database module. Note that masked values will get converted to None
   # with the tolist method. None gets translated to NULL when inserted into the
   # database.
   _tslist = test_series.tolist()

   # insert time series data
   crs.executemany("""
       INSERT INTO test_table
       ([date], [value]) VALUES (?, ?)
   """, _tslist)

   # Read the data back out of the database.
   # Explicitly cast data of type decimal to float for reading purposes,
   # otherwise you will get decimal objects for your result.
   crs.execute("""
       SELECT
           [date],
           CAST(ISNULL([value], 999) AS float) as vals, -- convert NULL's to 999
           (CASE
               WHEN [value] is NULL THEN 1
               ELSE 0
           END) AS mask -- retrieve a mask column
       FROM test_table
       ORDER BY [date] ASC
   """)

   # zip(*arg) converts row based results to column based results. This is the
   # crucial trick needed for easily reading time series data from a relational
   # database with Python
   _dates, _values, _mask = zip(*crs.fetchall())

   _series = ts.time_series(_values, dates=_dates, mask=_mask, freq='B')

   # commit changes to the database
   conn.commit()
   conn.close()



PyTables (HDF5)
===============

.. sectionauthor:: Pierre Gerard-Marchant <pierregmcode@gmail.com>
.. currentmodule:: scikits.timeseries.lib.tstables

PyTables_ is basically a Python interface to HDF5_, a portable format designed 
to manage large and complex hierarchical datasets.

PyTables_ is build on top of HDF5_ and numpy and provides a convenient 
interface to directly read ndarrays from HDF5 tables, or write ndarrays to 
HDF5 tables.

The :mod:`scikits.timeseries.lib.tstables` sub-module defines two new objects, 
:class:`MaskedTable` and :class:`TimeSeriesTable`.
These classes are designed to store :class:`~numpy.ma.MaskedArray` and 
:class:`~scikits.timeseries.TimeSeries` as tables in PyTables_.
The module also introduces new methods to the :class:`tables.File` object.



Classes
-------

.. class:: MaskedTable

   Based on :class:`tables.Table`
   
   Stores a :class:`~numpy.ma.MaskedArray` into a table.
   The initial :class:`~numpy.ma.MaskedArray` is transformed into a structured
   flexible-type :class:`~numpy.ndarray` before storage:

   * If  the initial  :class:`~numpy.ma.MaskedArray`  has  no named  fields
     (standard type), the resulting  ndarray has two  named fields: ``_data``
     and ``_mask``.
     The ``_data`` field has the  same type as the original array, while
     the ``_mask`` field  is always boolean. 
     Note that a  mask is *always* created, even if the  initial 
     :class:`~numpy.ma.MaskedArray` has no missing values.
   * If the initial :class:`~numpy.ma.MaskedArray` object has named fields
     (structured array), the resulting :class:`~numpy.ndarray` has as many
     fields as the initial array, with the same name.
     Each field of the result has two nested sub-fields, ``_data`` and
     ``_mask``.
     The ``_data`` subfield has the same dtype as the original field, while
     the ``_mask`` subfield will always be boolean.

     For example, if the dtype of the initial array is:
     
        >>> input_dtype = [('a', int), ('b', float), ('c', '|S3')]
     
     The resulting ndarray will have a dtype:
     
        >>> output_dtype = [('a', [('_data', int),   ('_mask', bool)]),
        ...                 ('b', [('_data', float), ('_mask', bool)]), 
        ...                 ('c', [('_data', '|S3'), ('_mask', bool)])]

   When a :class:`MaskedTable` is read, it is automatically retransformed into
   a :class:`~numpy.ma.MaskedArray`.

   Additional information about the :class:`~numpy.ma.MaskedArray` (such as
   :attr:`~numpy.ma.MaskedArray.baseclass`,
   :attr:`~numpy.ma.MaskedArray.fill_value`...) is stored into a dictionary
   named  ``special_attrs``.
   This information can be accessed through the :attr:`~tables.Table.attr`
   attribute of the table.

   .. automethod:: MaskedTable.read


______

.. class:: TimeSeriesTable
   
   Based on :class:`MaskedTable`
   
   Stores a :class:`~scikits.timeseries.TimeSeries` object into a table.
   The initial :class:`~scikits.timeseries.TimeSeries` is transformed into a
   structured :class:`~numpy.ndarray` before storage:

   * If the initial :class:`~scikits.timeseries.TimeSeries` has no named 
     fields (standard dtype), the resulting ndarray has three named fields:
     ``_dates``, ``_data`` and ``_mask``.
     The ``_data`` field has the same :attr:`dtype` as the original object.   
     The ``_mask`` field is always boolean.
     The ``_dates`` field is always integer, and corresponds to the integer
     representation of the underlying :class:`~scikits.timeseries.DateArray`.

     Note that a mask is always created, even if the initial
     :class:`~scikits.timeseries.TimeSeries` has no missing values.

   * If the initial :class:`~scikits.timeseries.TimeSeries` has named fields
     (flexible type), the resulting ndarray has the same named fields as the
     initial array, with the addition of an extra field ``_dates``.
     The ``_dates`` field is always integer, and corresponds to the integer 
     representation of the underlying :class:`~scikits.timeseries.DateArray`.
     Each other field is composed of two nested sub-fields, ``_data`` and
     ``_mask``.
     The ``_data`` subfield has the same dtype as the original field, 
     while the ``_mask`` subfield is always boolean.


   Additional information about the :class:`~scikits.timeseries.TimeSeries`
   (such as :attr:`~scikits.timeseries.TimeSeries.baseclass`,
   :attr:`~scikits.timeseries.TimeSeries.fill_value`, or
   the :attr:`~scikits.timeseries.TimeSeries.dates` frequency...) is stored
   into a dictionary named ``special_attrs``.
   This information can be accessed through the :attr:`~tables.Table.attr`
   attribute of the table.


   To create a :class:`TimeSeriesTable`, just use the
   :meth:`File.createTimeSeriesTable` method of a standard
   :class:`tables.File` object.


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


