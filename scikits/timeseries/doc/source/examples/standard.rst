.. currentmodule:: scikits.timeseries

:class:`~tdates.Date` objects
=============================

Even if you have no use for time series in general, you may still find the 
:class:`~tdates.Date` class contained in the module quite useful. 
A :class:`~tdates.Date` object combines some date and/or time related information 
with a given frequency.
You can picture the frequency as the unit into which the date is expressed.
For example, we can create dates in the following manner:

   >>> # The following imports are assumed throughout the documentation
   >>> import numpy as np
   >>> import numpy.ma as ma
   >>> import datetime
   >>> import scikits.timeseries as ts
   >>>
   >>> D = ts.Date(freq='D', year=2007, month=1, day=1)
   >>> M = ts.Date(freq='M', year=2007, month=1)
   >>> Y = ts.Date(freq='A', year=2007)


Observe that you only need to specify as much information as is relevant to the
frequency.
The importance of the frequency will become clearer later on.

.. note::
   A more technical note: :class:`~tdates.Date` objects are internally stored as integers.
   The conversion to integers and back is controlled by the frequency. 
   In the example above, the internal representation of the three objects ``D``, 
   ``M`` and ``Y`` are ``732677``, ``24073`` and ``2007``, respectively.



Construction of a :class:`~tdates.Date` object
----------------------------------------------

Several options are available to construct a :class:`~tdates.Date` object explicitly.
In each case, the ``frequency`` argument must be given.
Valid frequency specifications are given in the Frequencies_ section below.

* Give appropriate values to any of the ``year``, ``month``, ``day``, ``quarter``,
  ``hour``, ``minute``, ``second`` arguments.

   >>> ts.Date(freq='Q',year=2004,quarter=3)
   <Q : 2004Q3>
   >>> ts.Date(freq='D',year=2001,month=1,day=1)
   <D : 01-Jan-2001>

* Use the ``string`` keyword.
  
   >>> ts.Date('D', string='2007-01-01')
   <D : 01-Jan-2007>

* Use the ``datetime`` keyword with an existing :class:`datetime.datetime` object.

   >>> ts.Date('D', datetime=datetime.datetime.now())

* Use the ``value`` keyword and provide an integer representation of the date.
   
   >>> ts.Date('D', value=732677)
   <D : 01-Jan-2007>



Frequencies
-----------

For any functions or class constructors taking a frequency argument, the frequency
can be specified in one of two ways:

* using a valid string representation of the frequency, 
* using the integer frequency constants.

The constants can be found in the timeseries.const sub-module.
The following table lists the frequency constants and their valid string aliases.

.. tabularcolumns:: |l|l|

===============  =====================================================================================================================
 CONSTANT         String aliases (case insensitive)
===============  =====================================================================================================================
[1]_
 ``FR_ANN``      ``A``, ``Y``, ``ANNUAL``, ``ANNUALLY``, ``YEAR``, ``YEARLY`` 
 ``FR_ANNDEC``   ``A-DEC``, ``A-December``, ``Y-DEC``, ``ANNUAL-DEC``, etc... (annual frequency with December year end, equivalent to FR_ANN) 
 ``FR_ANNNOV``   ``A-NOV``, ``A-NOVEMBER``, ``Y-NOVEMBER``, ``ANNUAL-NOV``, etc... (annual frequency with November year end) 
 ``FR_ANNOCT``   ``A-OCT``, ``A-OCTOBER``, ``Y-OCTOBER``, ``ANNUAL-OCT``, etc... (annual frequency with October year end) 
 ``FR_ANNSEP``   ``A-SEP``, ``A-SEPTEMBER``, ``Y-SEPTEMBER``, ``ANNUAL-SEP``, etc... (annual frequency with September year end) 
 ``FR_ANNAUG``   ``A-AUG``, ``A-AUGUST``, ``Y-AUGUST``, ``ANNUAL-AUG``, etc... (annual frequency with August year end) 
 ``FR_ANNJUL``   ``A-JUL``, ``A-JULY``, ``Y-JULY``, ``ANNUAL-JUL``, etc... (annual frequency with July year end) 
 ``FR_ANNJUN``   ``A-JUN``, ``A-JUNE``, ``Y-JUNE``, ``ANNUAL-JUN``, etc... (annual frequency with June year end) 
 ``FR_ANNMAY``   ``A-MAY``, ``Y-MAY``, ``YEARLY-MAY``, ``ANNUAL-MAY``, etc... (annual frequency with May year end) 
 ``FR_ANNAPR``   ``A-APR``, ``A-APRIL``, ``Y-APRIL``, ``ANNUAL-APR``, etc... (annual frequency with April year end) 
 ``FR_ANNMAR``   ``A-MAR``, ``A-MARCH``, ``Y-MARCH``, ``ANNUAL-MAR``, etc... (annual frequency with March year end) 
 ``FR_ANNFEB``   ``A-FEB``, ``A-FEBRUARY``, ``Y-FEBRUARY``, ``ANNUAL-FEB``, etc... (annual frequency with February year end) 
 ``FR_ANNJAN``   ``A-JAN``, ``A-JANUARY``, ``Y-JANUARY``, ``ANNUAL-JAN``, etc... (annual frequency with January year end) 

 [2]_ 
 ``FR_QTR``      ``Q``, ``QUARTER``, ``QUARTERLY`` 
 ``FR_QTREDEC``  ``Q-DEC``, ``QTR-December``, ``QUARTERLY-DEC``, etc... (quarterly frequency with December year end, equivalent to FR_QTR) 
 ``FR_QTRENOV``  ``Q-NOV``, ``QTR-NOVEMBER``, ``QUARTERLY-NOV``, etc... (quarterly frequency with November year end) 
 ``FR_QTREOCT``  ``Q-OCT``, ``QTR-OCTOBER``, ``QUARTERLY-OCT``, etc... (quarterly frequency with October year end) 
 ``FR_QTRESEP``  ``Q-SEP``, ``QTR-SEPTEMBER``, ``QUARTERLY-SEP``, etc... (quarterly frequency with September year end) 
 ``FR_QTREAUG``  ``Q-AUG``, ``QTR-AUGUST``, ``QUARTERLY-AUG``, etc... (quarterly frequency with August year end) 
 ``FR_QTREJUL``  ``Q-JUL``, ``QTR-JULY``, ``QUARTERLY-JUL``, etc... (quarterly frequency with July year end) 
 ``FR_QTREJUN``  ``Q-JUN``, ``QTR-JUNE``, ``QUARTERLY-JUN``, etc... (quarterly frequency with June year end) 
 ``FR_QTREMAY``  ``Q-MAY``, ``QTR-MAY``, ``QUARTERLY-MAY``, etc... (quarterly frequency with May year end) 
 ``FR_QTREAPR``  ``Q-APR``, ``QTR-APRIL``, ``QUARTERLY-APR``, etc... (quarterly frequency with April year end) 
 ``FR_QTREMAR``  ``Q-MAR``, ``QTR-MARCH``, ``QUARTERLY-MAR``, etc... (quarterly frequency with March year end) 
 ``FR_QTREFEB``  ``Q-FEB``, ``QTR-FEBRUARY``, ``QUARTERLY-FEB``, etc... (quarterly frequency with February year end) 
 ``FR_QTREJAN``  ``Q-JAN``, ``QTR-JANUARY``, ``QUARTERLY-JAN``, etc... (quarterly frequency with January year end) 

 [3]_ 
 ``FR_QTRSDEC``  ``Q-S-DEC``, ``QTR-S-December``, etc... (quarterly frequency with December year end) 
 ``FR_QTRSNOV``  ``Q-S-NOV``, ``QTR-S-NOVEMBER``, etc... (quarterly frequency with November year end) 
 ``FR_QTRSOCT``  ``Q-S-OCT``, ``QTR-S-OCTOBER``, etc... (quarterly frequency with October year end) 
 ``FR_QTRSSEP``  ``Q-S-SEP``, ``QTR-S-SEPTEMBER``, etc... (quarterly frequency with September year end) 
 ``FR_QTRSAUG``  ``Q-S-AUG``, ``QTR-S-AUGUST``, etc... (quarterly frequency with August year end) 
 ``FR_QTRSJUL``  ``Q-S-JUL``, ``QTR-S-JULY``, etc... (quarterly frequency with July year end) 
 ``FR_QTRSJUN``  ``Q-S-JUN``, ``QTR-S-JUNE``, etc... (quarterly frequency with June year end) 
 ``FR_QTRSMAY``  ``Q-S-MAY``, ``QTR-S-MAY``, etc... (quarterly frequency with May year end) 
 ``FR_QTRSAPR``  ``Q-S-APR``, ``QTR-S-APRIL``, etc... (quarterly frequency with April year end) 
 ``FR_QTRSMAR``  ``Q-S-MAR``, ``QTR-S-MARCH``, etc... (quarterly frequency with March year end) 
 ``FR_QTRSFEB``  ``Q-S-FEB``, ``QTR-S-FEBRUARY``, etc... (quarterly frequency with February year end) 
 ``FR_QTRSJAN``  ``Q-S-JAN``, ``QTR-S-JANUARY``, etc... (quarterly frequency with January year end) 

 ``FR_MTH``      ``M``, ``MONTH``, ``MONTHLY`` 
 ``FR_WK``       ``W``, ``WEEK``, ``WEEKLY`` 
 ``FR_WKSUN``    ``W-SUN``, ``WEEK-SUNDAY``, ``WEEKLY-SUN``, etc... (weekly frequency with Sunday being the last day of the week, equivalent to FR_WK) 
 ``FR_WKSAT``    ``W-SAT``, ``WEEK-SATURDAY``, ``WEEKLY-SUN``, etc... (weekly frequency with Saturday being the last day of the week) 
 ``FR_WKFRI``    ``W-FRI``, ``WEEK-FRIDAY``, ``WEEKLY-FRI``, etc... (weekly frequency with Friday being the last day of the week) 
 ``FR_WKTHU``    ``W-THU``, ``WEEK-THURSDAY``, ``WEEKLY-THU``, etc... (weekly frequency with Thursday being the last day of the week) 
 ``FR_WKWED``    ``W-WED``, ``WEEK-WEDNESDAY``, ``WEEKLY-WED``, etc... (weekly frequency with Wednesday being the last day of the week) 
 ``FR_WKTUE``    ``W-TUE``, ``WEEK-TUESDAY``, ``WEEKLY-TUE``, etc... (weekly frequency with Tuesday being the last day of the week) 
 ``FR_WKMON``    ``W-MON``, ``WEEK-MONDAY``, ``WEEKLY-MON``, etc... (weekly frequency with Monday being the last day of the week) 
 ``FR_BUS``      ``B``, ``BUSINESS``, ``BUSINESSLY`` 
 ``FR_DAY``      ``D``, ``DAY``, ``DAILY`` 
 ``FR_HR``       ``H``, ``HOUR``, ``HOURLY`` 
 ``FR_MIN``      ``T``, ``MINUTE``, ``MINUTELY`` 
 ``FR_SEC``      ``S``, ``SECOND``, ``SECONDLY`` 
 ``FR_UND``      ``U``, ``UNDEF``, ``UNDEFINED`` 
===============  =====================================================================================================================

_[1] For annual frequencies,
     ``year`` is determined by where the last month of the year falls.
_[2] For the following quarterly frequencies, 
     ``year`` is determined by where the last quarter of the current group of quarters **ENDS**.
_[3] For these quarterly frequencies,
     ``year`` is determined by where the first quarter of the current group of quarters **STARTS**.


Convenience functions
---------------------

 * :func:`~tdates.now` 
   Get the current Date at a specified frequency
 * :func:`~tdates.prevbusday`
   Get the previous business day, determined by a specified cut off time.
   See the function's docstring for more details.


Manipulating dates
------------------

You can convert a :class:`~tdates.Date` object from one frequency to another with the 
:meth:`~tdates.Date.asfreq` method.
When converting to a higher frequency (for example, from monthly to daily),
you may optionally specify the "relation" parameter with the value ``"START"`` or
``"END"`` (default is ``"END"``).
Note that if you convert a daily :class:`~tdates.Date` to a monthly frequency and back 
to a daily one, you will lose your day information in the process 
(similarly for converting any higher frequency to a lower one):

   >>> D = ts.Date('D', year=2007, month=12, day=31)
   >>> D.asfreq('M')
   <M: Dec-2006>
   >>> D.asfreq('M').asfreq('D', relation="START")
   <D: 01-Dec-2006>
   >>> D.asfreq('M').asfreq('D', relation="END")
   <D: 31-Dec-2006>


You can add and subtract integers from a :class:`~tdates.Date` object to get a new 
:class:`~tdates.Date`` object.
The frequency of the new object is the same as the original one.
For example:

   >>> yesterday = ts.now('D') - 1
   >>> infivemonths = ts.now('M') + 5


You can also subtract a :class:`~tdates.Date` from another :class:`~tdates.Date` of the same 
frequency to determine the number of periods between the two dates.

   >>> Y = ts.Date('A', year=2007)
   >>> days_in_year = Y.asfreq('D', relation='END') - Y.asfreq('D', relation='START') + 1
   >>> days_in_year
   365


Some other methods worth mentioning are:

* :meth:`~tdates.Date.toordinal`
  Converts an object to the equivalent proleptic gregorian date.
* :meth:`~tdates.Date.tostring`
  Converts an object to the corresponding string.


Formatting Dates as String
--------------------------

To output a date as a string, you can simply cast it to a string (call 
:func:`str` on it) and a default output format for that frequency will be used,
or you can use the :meth:`~tdates.Date.strfmt` method for explicit control. 
The :meth:`~tdates.Date.strfmt` method of the Date class takes one argument: a format string.
This behaves in essentially the same manner as the ``strftime`` function in the 
standard python time module and accepts the same directives, plus several 
additional directives outlined below.

* '%q'
  The ''quarter'' of the date.
* '%f'
  Year without century as a decimal number [00,99]. 
  The ''year'' in this case is the year of the date determined by the year for the current quarter.
  This is the same as '%y' unless the Date is one of the quarterly frequencies. 
  In financial terms, this is the 'fiscal year'.
* '%F'
  Year with century as a decimal number.
  The ''year'' in this case is the year of the date determined by the year for the current quarter.
  This is the same as %Y unless the Date is one of the quarterly frequencies.
  In financial terms, this is the 'fiscal year'.


Examples
~~~~~~~~

   >>> a = ts.Date(freq='q-jul', year=2006, quarter=1)
   >>> a.strfmt("%F-Q%q")
   '2006-Q1'
   >>> a.strfmt("%b-%Y") # this will output the last month in the quarter for this date
   'Oct-2005'
   >>> b = ts.Date(freq='d', year=2006, month=4, day=25)
   >>> b.strfmt("%d-%b-%Y")
   '25-Apr-2006'


________________________________________________________________________________


:class:`~tdates.DateArray` objects
==========================

A :class:`~tdates.DateArray` object is  are simply a :class:`numpy.ndarray` of 
:class:`~tdates.Date` objects.
They accept the same methods as a :class:`Date` object, with the addition of:

* :meth:`~tdates.DateArray.tovalue`
  Converts the array to an array of integers.
  Each integer is the internal representation of the corresponding date.
* :meth:`~tdates.DateArray.has_missing_dates`
  Outputs a boolean on whether some dates are missing or not.
* :meth:`~tdates.DateArray.has_duplicated_dates`
  Outputs a boolean on whether some dates are duplicated or not.


Construction
------------

To construct a :class:`~tdates.DateArray` object, you can use the factory function 
:func:`~tdates.date_array` (preferred), or call the class directly. 
See the ``__doc__`` strings of :func:`~tdates.date_array` and :class:`~tdates.DateArray` 
for a more detailed presentation of the available parameters.


_______________________________________________________________________________


:class:`~tseries.TimeSeries`
============================

A :class:`~tseries.TimeSeries` object is the combination of three ndarrays:

* :attr:`dates`: A :class:`~tdates.DateArray` object.
* :attr:`data` : A :class:`numpy.ndarray`.
* :attr:`mask` : A boolean :class:`numpy.ndarray`, indicating missing or invalid data.

These three arrays can be accessed as attributes of a :class:`~tseries.TimeSeries` object.
Another very useful attribute is :attr:`~tseries.series`, that gives you the possibility
to directly access :attr:`~tseries.TimeSeries.data` and 
:attr:`~tseries.TimeSeries.mask` as a masked array.


Construction
------------

To construct a :class:`~tseries.TimeSeries`, you can use the factory function 
:func:`~tseries.time_series` (preferred) or call the class directly.
See the ``__doc__`` strings of this function and this class for more details 
on the input parameters.
Even if it is recommended to use the factory function :func:`~tseries.time_series`,
you can still use the class constructor if you need to bypass some of the overhead 
associated with the additional flexibility of the factory function.

As an example, let us construct a series of 600 random elements, starting 600 
business days ago, at  a business daily frequency:

   >>> data = np.random.uniform(-100,100,600)
   >>> today = ts.now('B')
   >>> series = ts.time_series(data, dtype=np.float_, freq='B', start_date=today-600)

We can check that ``series.dates`` is a :class:`~tdates.DateArray` object and that 
``series.series`` is a :class:`numpy.ma.MaskedArray` object.

   >>> isinstance(series.dates, ts.DateArray)
   True
   >>> isinstance(series.series, ma.MaskedArray)
   True


So, if you are already familiar with the :mod:`numpy.ma` module, using the
:mod:`~scikits.timeseries` package should be straightforward.
Just keep in mind that another attribute is always present, :attr:`dates`.


Dates and Data compatibility
----------------------------

The example we just introduced corresponds to the simplest case of only one
variable indexed in time.
In that case, the :class:`~tdates.DateArray` object should have the same size as the 
:attr:`~tseries.TimeSeries.data` part.
In our example, the length of the :class:`~tdates.DateArray` was automatically adjusted 
to match the data length, and we have ``DateArray.size == series.size``.

However, it is often convenient to use series with multiple variables.
A simple representation of this kind of data is a matrix, with as many rows as
actual observations and as many columns as variables.
In that case, the :class:`~tdates.DateArray` object should have the same length as the
number of rows.
More generally, ``DateArray.size`` should be equal to ``series.shape[0]``.

When a :class:`~tseries.TimeSeries` is created from a multi-dimensional ``data`` and a
single starting date, it is assumed that the data consists of several variables: 
the length of the :class:`~tdates.DateArray` is then adjusted to match ``len(data)``.
However, you can force the length of the :class:`~tdates.DateArray`
with the ``length`` optional parameter.

For example, let us consider the case of an array of (50 x 12) points, 
corresponding to 50 years of monthly data.

   >>> data = np.random.uniform(-1,1,50*12).reshape(50,12)

We may want to consider each month independently from the others:
in that case, we want an annual series of 50 observations, each observation 
consisting of 12 variables.
We define the time series as:

   >>> newseries = ts.time_series(data, start_date=ts.now('Y')-50)
   >>> newseries._dates.size
   50

But we can also consider the series as monthly data.
We could even ravel the initial data, or force the length of the :class:`~tdates.DateArray`:

   >>> newseries = ts.time_series(data, start_date=ts.now('M')-600, length=600)
   >>> newseries._dates.size
   600

Now, let us consider the case of a (5x10x10) array.
For example, each (10x10) slice could be a raster map, or a picture.
The following code defines a daily series of 5 maps:

   >>> data = np.random.uniform(-1,1,5*10*10).reshape(5,10,10)
   >>> newseries = ts.time_series(data, start_date=ts.now('D'))


Indexing
--------

Elements of a :class:`~tseries.TimeSeries` can be accessed just like with regular ndarrays.
Thus,

   >>> series[0]

outputs the first element, while

   >>> series[-30:]

outputs the last 30 elements.

But you can also use a date:

   >>> thirtydaysago = today - 30
   >>> series[thirtydaysago:]

or even a string...

   >>> series[thirtydaysago.tostring():]

or a sequence/ndarray of integers...

   >>> series[[0,-1]]

.. note::
   This latter is quite useful: it gives you the first and last data of your series.


In a similar way, setting elements of a :class:`~tseries.TimeSeries` works seamlessly.
Let us set negative values to zero...

   >>> series[series<0] = 0

... and the values falling on Fridays to 100

   >>> series[series.weekday == 4] = 100


We can also index on multiple criteria.
We will create a temporary array of 'weekdays' to avoid recomputing the weekdays
multiple times. 
Here we will set all Wednesday and Fridays to 100.

   >>> weekdays = series.weekday
   >>> series[(weekdays == 2)  (weekdays == 4)] = 100

You should keep in mind that :class:`~tseries.TimeSeries` are basically :class:`numpy.ma.MaskedArrays`.
If some data of an array are masked, you will not be able to use this array as index,
you will have to fill it first.


Missing Observations (aka masked values)
----------------------------------------

Missing observations are handled in exactly the same way as with masked arrays.
If you are familiar with masked arrays, then there is nothing new to learn. 
Please see the main numpy documentation for additional info on masked arrays.


Operations on TimeSeries
------------------------

If you work with only one :class:`~tseries.TimeSeries`, you can use the :mod:`numpy.ma` 
commands to process the data. For example:

   >>> series_log = ma.log(series)

Note that invalid values (negative, in that case), are automatically masked.
Note also that you could use the standard numpy version of the function instead,
however the :meth:`reduce` and :meth:`accumulate` methods of some ufuncs 
(such as :func:`add` or :func:`multiply`) will only function properly with the 
:mod:`numpy.ma` versions.

.. note::

   The reason is that the methods of the numpy.ufuncs will not know how to
   properly ignore masked values for such operations.


When working with multiple series, only series of the same frequency, size and 
starting date can be used in basic operations. 
The function :func:`~tseries.align_series` (or its alias :func:`~tseries.aligned`) forces 
series to have matching starting and ending dates.
By default, the starting date will be set to the smallest starting date of the
series, and the ending date to the largest.

Let's construct a list of months, starting on Jan 2005 and ending on Dec 2006,
with a gap from Oct 2005 to Jan 2006.

   >>> mlist_1 = ['2005-%02i' % i for i in range(1,10)]
   >>> mlist_1 += ['2006-%02i' % i for i in range(2,13)]
   >>> mdata_1 = np.arange(len(mlist_1))
   >>> mser_1 = ts.time_series(mdata_1, mlist_1, freq='M')

Let us check whether there are some duplicated dates (no):

   >>> mser_1.has_duplicated_dates()
   False

...or missing dates (yes):

   >>> mser_1.has_missing_dates()
   True

Let us construct a second monthly series, this time without missing dates:

   >>> mlist_2 = ['2004-%02i' % i for i in range(1,13)]
   >>> mlist_2 += ['2005-%02i' % i for i in range(1,13)]
   >>> mser_2 = ts.time_series(np.arange(len(mlist_2)), mlist_2, freq='M')


We cannot perform binary operations on these two series (such as adding them
together) because the dates of the series do not line up.
Thus, we need to align them first.

   >>> (malg_1, malg_2) = ts.align_series(mser_1, mser_2)

Now we can add the two series.
Only the data that fall on dates common to the original, non-aligned series will
be actually added, the others will be masked. 
After all, we are adding masked arrays.

   >>> mser_3 = malg_1 + malg_2

We could have filled the initial series first (replace masked values with a 
specified value):

   >>> mser_3 = malg_1.filled(0) + malg_2.filled(0)

When aligning the series, we could have forced the series to start/end at some 
given dates:

   >>> (malg_1,malg_2) = align_series(mser_1_filled, mser2,
   ...                                start_date='2004-06', end_date='2006-06')


Time Shifting Operations
------------------------
Calculating things like rate of change, or difference in a :class:`~tseries.TimeSeries`
can be done most easily using a special method called :meth:`~tseries.TimeSeries.tshift`.

Suppose we want to calculate a Year over Year rate of return for a monthly time 
series. One might initially try to do something along the lines of...

   >>> YoY_change = 100*(mser[12:]/mser[:-12] - 1)

This will give you the correct numerical result, but since ``mser[12:]`` and 
``mser[:-12]`` have different start and end dates, the result will be forced to 
a plain :class:`numpy.ma.MaskedArray`.
Also, it will not be the same shape as your original input series, 
which may also be inconvenient.
To get around these issues, use the :meth:`tshift` method instead.

   >>> YoY_change = 100*(mser/mser.tshift(-12, copy=False) - 1)

``mser.tshift(-12, copy=False)`` returns a series with the ``same start_date`` 
and ``end_date`` as ``mser``, but values shifted to the right by 12 periods. 
Note that this will result in 12 masked values at the start of the resulting series. 
By default :meth:`~tseries.TimeSeries.tshift` copies any data it uses from the original series, 
but for situations like the example above you may want to avoid that.


TimeSeries Frequency Conversion
-------------------------------

To convert a :class:`~tseries.TimeSeries` to another frequency, 
use the :meth:`~tseries.TimeSeries.convert` method or is function equivalent.
The optional argument ``func`` must be a function that acts on a 
1D masked array and returns a scalar.

   >>> mseries = series.convert('M',func=ma.average)

If ``func`` is None (the default value), the convert method/function returns a 
2D array, where each row corresponds to the new frequency, and the columns to 
the original data. 
In our example, :meth:`convert` will return a 2D array with 23 columns, as there
are at most 23 business days per month.

   >>> mseries_default = series.convert('M')


When converting from a lower frequency to a higher frequency, an extra argument 
``position`` is used to determine the placement of values in the resulting series.
The value of the argument is either ``'START'`` or ``'END'`` (``'END'`` by default).
This will yield a series with a lot of masked values.
To fill in these masked values, see the section `Interpolating Masked Values <interpolating>`_ below.

.. warning::
   Be careful not to confuse th two methods :meth:`asfreq` and :meth:`convert`.
   * :meth:`~tseries.TimeSeries.asfreq` simply takes every date 
     in the :attr:~tseries.TimeSeries.`dates` attribute of the 
     :class:`~tseries.TimeSeries` instance and changes it to the specified frequency, so the 
     resulting series will have the same shape as the original series.
   * :meth:`TimeSeries.convert` is a more complicated function that takes a series with no 
     missing or duplicated dates and creates a series at the new frequency with 
     no missing or duplicated dates and intelligently places the data from the 
     original series into appropriate points in the new series.


.. _interpolating
Interpolating Masked Values
---------------------------

The :mod:`~scikits.timeseries.lib.interpolate` sub-module contains several functions for
filling in masked values in an array.
Currently this includes:

* :func:`~lib.interpolate.interp_masked1d`
* :func:`~lib.interpolate.foward_fill`
* :func:`~lib.interpolate.backward_fill`

Let us take a monthly :class:`~tseries.TimeSeries` , convert it to business frequency,
and then interpolate the resulting masked values.

   >>> import scikits.timeseries.lib.interpolate as itp
   >>> mser = ts.time_series(np.arange(12, dtype=np.float_), start_date=ts.now('M'))
   >>> bser = mser.convert("B", position='END')
   >>> bser_ffill = itp.forward_fill(bser, maxgap=30)
   >>> bser_bfill = itp.backward_fill(bser)
   >>> bser_linear = itp.interp_masked1d(bser, kind='linear')

The optional ``maxgap`` parameter for :func:`~lib.interpolate.forward_fill` and 
:func:`backward_fill` will ensure that if there are more than ``maxgap``
consecutive masked values, they will not be filled. 
Using ``maxgap=30`` like in our above example will ensure that missing months 
from our original monthly series are not filled in.
