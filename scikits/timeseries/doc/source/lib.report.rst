.. currentmodule:: scikits.timeseries.lib.reportlib
.. moduleauthor:: Matt Knox <mattknox.ca@hotmail.com>

=======
Reports
=======

The :mod:`~scikits.timeseries.lib.reportlib` sub-module provides the :class:`Report` class for generating text based time series reports.



The :class:`Report` class
=========================

.. autoclass:: Report


Examples
========

The following variables will be used throughout the examples

   >>> import scikits.timeseries.lib.reportlib as rl
   >>> ser1 = ts.time_series(np.random.uniform(-100,100,10),
   ...                       start_date=ts.now('b')-5)
   >>> ser2 = ts.time_series(np.random.uniform(-100,100,10),
   ...                       start_date=ts.now('b'))
   >>> strings = ['some string', 'another string',
   ...            'yet another, string', 'final string']
   >>> ser3 = ts.time_series(strings,
   ...                       start_date=ts.now('b'),
   ...                       dtype=np.string_)
   >>> dArray = ts.date_array(start_date=ts.now('b'), length=3)


Example 1: Basic report
-----------------------

   >>> basicReport = rl.Report(ser1, ser2, ser3)
   >>> basicReport()
   29-Jan-2007 | -95.4554568525 |             -- | --
   30-Jan-2007 |  8.58356086571 |             -- | --
   31-Jan-2007 |  41.6353000447 |             -- | --
   01-Feb-2007 | -70.4674570816 |             -- | --
   02-Feb-2007 |  2.98803489327 |             -- | --
   05-Feb-2007 | -21.6474414786 |  -77.750560056 | some string
   06-Feb-2007 |  84.3212422071 |  56.2238118715 | another string
   07-Feb-2007 |  23.5664556686 |  64.2491772743 | yet another, string
   08-Feb-2007 |  34.8778728662 | -39.4734173695 | final string
   09-Feb-2007 | -64.0545308092 | -83.7175337221 | --
   12-Feb-2007 |             -- |  52.4958419122 | --
   13-Feb-2007 |             -- |   7.1396171176 | --
   14-Feb-2007 |             -- | -57.7688749366 | --
   15-Feb-2007 |             -- |  71.2844695721 | --
   16-Feb-2007 |             -- |  87.1665936067 | --


Example 2: csv report for excel
-------------------------------

   >>> mycsv = open('mycsv.csv', 'w')
   >>> strfmt = lambda x: '"'+str(x)+'"'
   >>> fmt_func = [None, None, strfmt]
   >>> csvReport = rl.Report(ser1, ser2, ser3, fmt_func=fmt_func,
   ...                       mask_rep='#N/A', delim=',',
   ...                       fixed_width=False)
   >>> csvReport() # output to sys.stdout
   29-Jan-2007,67.4086881661,#N/A,#N/A
   30-Jan-2007,-78.8405461996,#N/A,#N/A
   31-Jan-2007,10.0559754743,#N/A,#N/A
   01-Feb-2007,-71.149716374,#N/A,#N/A
   02-Feb-2007,-46.055865283,#N/A,#N/A
   05-Feb-2007,35.9105419931,85.1744316431,"some string"
   06-Feb-2007,2.93015788615,-87.0634270731,"another string"
   07-Feb-2007,-49.0774248826,-91.4854233865,"yet another, string"
   08-Feb-2007,94.8175754225,36.587114053,"final string"
   09-Feb-2007,-88.9474880802,37.3563788938,#N/A
   12-Feb-2007,#N/A,21.1325367724,#N/A
   13-Feb-2007,#N/A,72.2437957896,#N/A
   14-Feb-2007,#N/A,37.2619438419,#N/A
   15-Feb-2007,#N/A,-87.1465826319,#N/A
   16-Feb-2007,#N/A,63.5556895555,#N/A
   >>> csvReport(output=mycsv) # output to file



Example 3: HTML report
----------------------

   >>> numfmt = lambda x: '%.2f' % x
   >>> fmt_func = [numfmt, numfmt, None]
   >>> footer_func = [ma.sum, ma.sum, None]
   >>> footer_label = "Total"
   >>> htmlReport = rl.Report(ser1, ser2, ser3)
   >>> htmlReport.set_options(prefix='<tr><td>',
   ...                        delim='</td><td>',
   ...                        postfix='</td></tr>')
   >>> htmlReport.set_options(wrap_func=rl.wrap_onspace(10,nls='<BR>'))
   >>> htmlReport.set_options(fmt_func=fmt_func)
   >>> htmlReport.set_options(footer_label=footer_label,
   ...                        footer_func=footer_func,
   ...                        footer_char='')
   >>> htmlReport.set_options(dates=dArray)
   >>> htmlReport() # output to sys.stdout
   <tr><td>05-Feb-2007</td><td> 91.66</td><td>-99.21</td><td>some<BR>string           </td></tr>
   <tr><td>06-Feb-2007</td><td>-68.84</td><td> 30.50</td><td>another<BR>string        </td></tr>
   <tr><td>07-Feb-2007</td><td> 93.53</td><td> 90.46</td><td>yet<BR>another,<BR>string</td></tr>
   <tr><td>Total      </td><td>116.36</td><td> 21.75</td><td>                         </td></tr>



Example 4: Extra Options
------------------------

   >>> basicReport = rl.Report(ser1, ser2, ser3, dates=dArray)


Output report with a header
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, a line of dashes will separate the header and the first row of data.
Optionally, you can specify a label for the Date column as well (so a list with
four entries instead of three like this example), If you wish to get rid of the
separator line, or use a different character, specify: ``header_char=''``

   >>> basicReport(header_row=['col 1', 'col 2', 'col 3'])
               |          col 1 |          col 2 | col 3
   ------------------------------------------------------------------
   06-Feb-2007 |  2.59583929443 | -96.2110139217 | some string
   07-Feb-2007 | -24.1064434097 |  86.0387977626 | another string
   08-Feb-2007 | -21.6432010416 |  4.83754030508 | yet another, string


Change column justification for the report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can specify a single string (*'right'*, *'left'*, or *'center'*) and this
will impact all columns, or you can specify a list of strings
(optionally including the Date column, which is *'left'* by default)

   >>> basicReport(justify=['left', 'left', 'right'])
   06-Feb-2007 | 2.59583929443  | -96.2110139217 |         some string
   07-Feb-2007 | -24.1064434097 | 86.0387977626  |      another string
   08-Feb-2007 | -21.6432010416 | 4.83754030508  | yet another, string


Change formatting of Date column
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   >>> basicReport(datefmt='%d')
   06 |  2.59583929443 | -96.2110139217 | some string
   07 | -24.1064434097 |  86.0387977626 | another string
   08 | -21.6432010416 |  4.83754030508 | yet another, string


Add a separater line between each row
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   >>> basicReport(row_char='-')
   06-Feb-2007 |  2.59583929443 | -96.2110139217 | some string
   -------------------------------------------------------------------
   07-Feb-2007 | -24.1064434097 |  86.0387977626 | another string
   -------------------------------------------------------------------
   08-Feb-2007 | -21.6432010416 |  4.83754030508 | yet another, string


Report different series
~~~~~~~~~~~~~~~~~~~~~~~

Notice that the other options set remain intact (ie. dates=dArray)

   >>> basicReport(ser1)
   06-Feb-2007 |  2.59583929443
   07-Feb-2007 | -24.1064434097
   08-Feb-2007 | -21.6432010416


Specify column widths
~~~~~~~~~~~~~~~~~~~~~

Just as in the header and justify options, you can specify a single value to
affect all columns, or a list which optionally includes a specification for the
Date column.
Specify ``-1`` to auto-size a column

   >>> basicReport(col_width=[20, 20, -1])
   06-Feb-2007 |        2.59583929443 |       -96.2110139217 | some string
   07-Feb-2007 |       -24.1064434097 |        86.0387977626 | another string
   08-Feb-2007 |       -21.6432010416 |        4.83754030508 | yet another, string
