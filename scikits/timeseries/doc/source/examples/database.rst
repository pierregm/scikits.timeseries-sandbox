.. sectionauthor:: Matt Knox <mattknox.ca@hotmail.com>

.. highlight:: python
   :linenothreshold: 5


Databases
=========

Storing and retrieving time series from standard relational databases is very
simple once you know a few tricks.
For these examples, we use the ceODBC database module (http://ceodbc.sourceforge.net/) which I have found to be more
reliable and faster than the pyodbc module.
However, I *think* these examples
should work with the pyodbc module as well.

SQL Server 2005 Express edition is the database used in the examples. Other
standard relational databases should also work, but I have not personally
verified it.

A database called "test" is assumed to have been created already along with a
table called "test_table" described by the following query:[[BR]][[BR]]

.. code-block:: mysql
   :linenos:
   
   CREATE TABLE  test_table (
                             [date] [datetime] NULL,
                             [value] [decimal](18, 6) NULL
                             )


If you have verified these examples to work with other databases and python
db modules, it would be greatly appreciated if you could add a note to the
wiki.


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
                   """,
                   _tslist
                   )
   
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


