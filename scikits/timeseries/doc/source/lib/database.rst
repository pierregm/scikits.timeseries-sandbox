.. sectionauthor:: Matt Knox <mattknox.ca@hotmail.com>

.. highlight:: python
   :linenothreshold: 5


=========
Databases
=========

There are two approaches for working with TimeSeries objects that are
supported with the scikits.timeseries module. One is using standard relational
databases and the other is using the PyTables (http://www.pytables.org). Both
approaches have their pros and cons.

Generally speaking, if you need concurrent access to the database and plan to
do web based things (generate time series charts on a web site) you will
probably be better off using a relational database.

If you are doing research on your own and need simple storage for a specific
project without all the administrative complexities of a relational database
then PyTables may suit your needs very well.

If performance is critical for your application, I recommend expirementing with
both approaches to see what suits your particular project better.

Relational Databases
====================

Storing and retrieving time series from standard relational databases is very
simple once you know a few tricks.
For these examples, we use the ceODBC database module (http://ceodbc.sourceforge.net/) which I have found to be more
reliable and faster than the pyodbc module.
However, I *think* these examples
should work with the pyodbc module as well.

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

If you are not familiar with PyTables, I recommend doing some background
reading first (http://www.pytables.org). PyTables is basically a python
interface to HDF5 databases. It supports all the standard numpy data types and
you can read and write numpy arrays directly using PyTables.

The :mod:`scikits.timeseries.lib.tstables` sub-module includes classes to support
working with TimeSeries and MaskedArray objects in PyTables.

Module Details
--------------

.. automodule:: scikits.timeseries.lib.tstables
