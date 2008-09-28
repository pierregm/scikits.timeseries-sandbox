.. currentmodule:: scikits.timeseries

:class:`~TimeSeries` objects
====================================

A :class:`~TimeSeries` object is the combination of three ndarrays:

* :attr:`dates`: A :class:`~DateArray` object.
* :attr:`data` : A :class:`numpy.ndarray`.
* :attr:`mask` : A boolean :class:`numpy.ndarray`, indicating missing or invalid data.

These three arrays can be accessed as attributes of a :class:`~TimeSeries` object.
Another very useful attribute is :attr:`~TimeSeries.series`, that gives you the possibility
to directly access :attr:`~TimeSeries.data` and
:attr:`~TimeSeries.mask` as a masked array.

As :class:`TimeSeries` objects subclass :class:`~numpy.ma.MaskedArrays`, they
inherit all their attributes and methods, as well as the attributes and methods
of regular ndarrays.

Construction
------------

To construct a :class:`~TimeSeries`, you can use the factory function
:func:`~time_series` (preferred) or call the class directly.

.. autofunction:: time_series

Even if it is recommended to use the factory function :func:`~time_series`,
you can still use the class constructor if you need to bypass some of the overhead
associated with the additional flexibility of the factory function.

As an example, let us construct a series of 600 random elements, starting 600
business days ago, at  a business daily frequency:

   >>> data = np.random.uniform(-100,100,600)
   >>> today = ts.now('B')
   >>> series = ts.time_series(data, dtype=np.float_, freq='B', start_date=today-600)

We can check that ``series.dates`` is a :class:`~DateArray` object and that
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
In that case, the :class:`~DateArray` object should have the same size as the
:attr:`~TimeSeries.data` part.
In our example, the length of the :class:`~DateArray` was automatically adjusted
to match the data length, and we have ``DateArray.size == series.size``.

However, it is often convenient to use series with multiple variables.
A simple representation of this kind of data is a matrix, with as many rows as
actual observations and as many columns as variables.
In that case, the :class:`~DateArray` object should have the same length as the
number of rows.
More generally, ``DateArray.size`` should be equal to ``series.shape[0]``.

When a :class:`~TimeSeries` is created from a multi-dimensional ``data`` and a
single starting date, it is assumed that the data consists of several variables:
the length of the :class:`~DateArray` is then adjusted to match ``len(data)``.
However, you can force the length of the :class:`~DateArray`
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
We could even ravel the initial data, or force the length of the :class:`~DateArray`:

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

Elements of a :class:`~TimeSeries` can be accessed just like with regular ndarrays.
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


In a similar way, setting elements of a :class:`~TimeSeries` works seamlessly.
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

You should keep in mind that :class:`~TimeSeries` are basically :class:`numpy.ma.MaskedArrays`.
If some data of an array are masked, you will not be able to use this array as index,
you will have to fill it first.


Missing Observations (aka masked values)
----------------------------------------

Missing observations are handled in exactly the same way as with masked arrays.
If you are familiar with masked arrays, then there is nothing new to learn.
Please see the main numpy documentation for additional info on masked arrays.


Operations on TimeSeries
------------------------

If you work with only one :class:`~TimeSeries`, you can use the :mod:`numpy.ma`
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
The function :func:`~align_series` (or its alias :func:`~aligned`)
forces series to have matching starting and ending dates.
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
Calculating things like rate of change, or difference in a :class:`~TimeSeries`
can be done most easily using a special method called :meth:`~TimeSeries.tshift`.

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
By default :meth:`~TimeSeries.tshift` copies any data it uses from the
original series, but for situations like the example above you may want to avoid
that.


TimeSeries Frequency Conversion
-------------------------------

The following method converts :class:`TimeSeries` from one frequency to another.

.. method:: TimeSeries.asfreq(freq)

   Returns a series whose :attr:`dates` has been converted to the new frequency ``freq``.
   The :attr:`series` part remains unchanged.
   Therefore, when converting to a lower frequency, the new series will have
   duplicated dates, whereas when converting to a higher frequency, the new series
   will have missing dates.


.. automethod:: TimeSeries.convert

.. warning::
   Be careful not to confuse the two methods :meth:`asfreq` and :meth:`convert`.

   * :meth:`~TimeSeries.asfreq` simply takes every date
     in the :attr:`~TimeSeries.dates` attribute of the
     :class:`~TimeSeries` instance and changes it to the specified frequency,
     so the resulting series will have the same shape as the original series.
   * :meth:`~TimeSeries.convert` is a more complicated function
     that takes a series with no missing nor duplicated dates and creates a series
     at the new frequency with no missing nor duplicated dates and intelligently
     places the data from the original series into appropriate points in the new
     series.

Attributes
----------

   .. attribute:: data

      Returns a view of a :class:`TimeSeries` as a :class:`~numpy.ndarray`.
      This attribute is read-only and cannot be directly set.

   .. attribute:: mask

      Returns the mask of the object, as a boolean ndarray with the same shape
      as :attr:`data`, or as the special value ``nomask`` (equivalent to ``False``).
      This attribute is writable and can be modified.

   .. attribute:: series

      Returns a view of a :class:`TimeSeries` as a :class:`~numpy.ma.MaskedArray`.
      This attribute is read-only and cannot be directly set

   .. attribute:: dates

      Returns the :class:`DateArray` object of the dates of the series.
      This attribute is writable and can be modified.


In addition, all the attributes of the :class:`DateArray` :attr:`dates` attributes
are directly accessible by :class:`TimeSeries`.


Methods
-------

The following methods access information about the :attr:`dates` attribute:

.. automethod:: TimeSeries.isfull
.. automethod:: TimeSeries.has_missing_dates
.. automethod:: TimeSeries.has_duplicated_dates
.. automethod:: TimeSeries.isvalid


.. automethod:: TimeSeries.date_to_index


Format conversions
~~~~~~~~~~~~~~~~~~

The following methods transform the :class:`TimeSeries` to different formats.

.. automethod:: TimeSeries.tolist
.. automethod:: TimeSeries.tofile
.. automethod:: TimeSeries.asrecords


.. automethod:: TimeSeries.split



Functions
---------

.. autofunction:: adjust_endpoints
.. autofunction:: align_series
.. autofunction:: align_with

.. autofunction:: tshift
.. autofunction:: pct
.. autofunction:: fill_missing_dates

.. autofunction:: empty_like



Exceptions
----------

.. autoexception:: TimeSeriesError
   :show-inheritance:
.. autoexception:: TimeSeriesCompatibilityError
   :show-inheritance:
