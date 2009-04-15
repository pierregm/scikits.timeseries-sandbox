.. currentmodule:: scikits.timeseries

.. _timeseries_indexing:

Indexing and slicing
--------------------

Accessing basic elements
~~~~~~~~~~~~~~~~~~~~~~~~

Elements of a :class:`~TimeSeries` can be accessed just like their regular 
:class:`~numpy.ma.MaskedArray` counterparts, using an integer, a sequence of
integers, an array of integers or booleans, a slice, or a string corresponding 
to a field name (if the series has a structured :attr:`dtype`).
In addition, a :class:`TimeSeries` can be indexed by:

   * a :class:`Date` object;
   * a sequence of :class:`Date` objects;
   * a :class:`DateArray` object;
   * a string representing a date;
   * a sequence of strings representing dates.

In the last two cases, the strings are converted to a :class:`Date` object 
with the same frequency as the :class:`TimeSeries` instance.


If a single element is accessed on a :class:`TimeSeries` with a standard 
:attr:`dtype` (without named fields), the result is either **(i)** a scalar 
if the element is not masked, or **(ii)** the constant :const:`~numpy.ma.masked` 
if the corresponding :attr:`mask` element is :const:`True`.

Accessing a single element of a :class:`TimeSeries` with a structured data-type 
returns either **(i)** a :const:`numpy.void` object if the element is not masked 
or **(ii)** a :class:`~numpy.ma.MaskedArray` otherwise.

Accessing several elements at once always returns a :class:`TimeSeries` object, 
with the same frequency as the input.


Examples
''''''''

   >>> monthly_start = ts.Date('M', '2005-01')
   >>> series = ts.time_series(np.arange(12), 
   ...                         start_date=monthly_start)

Accessing a single element with an integer:
   >>> series[0]
   0

Accessing a single element with a Date:
   >>> series[monthly_start+6]
   6

Accessing a single element with a string:
   >>> series['2005-06-15']
   5

Accessing several elements with a sequence of integers:
   >>> series[[0, -1]]
   timeseries([ 0 11],
      dates = [Jan-2005 Dec-2005],
      freq  = M)

Accessing several elements with a slice:
   >>> series[-3:]
   timeseries([ 9 10 11],
      dates = [Oct-2005 ... Dec-2005],
      freq  = M)

Accessing elements with an array of booleans:
   >>> condition = (series < 7)
   >>> series[condition]
   timeseries([0 1 2 3 4 5 6],
      dates = [Jan-2005 ... Jul-2005],
      freq  = M)


Additional functions
''''''''''''''''''''

Two functions provide a convenient way to access the first and last valid
(not masked) elements of a :class:`TimeSeries`:

.. autosummary::
   :toctree: generated/

   first_unmasked_val
   last_unmasked_val





Setting elements
~~~~~~~~~~~~~~~~

Individual or multiple elements of a :class:`~TimeSeries` can be set the same
way as for :class:`~numpy.ma.MaskedArray`.

To mask one or several elements of a :class:`~TimeSeries`, they must be set 
to the :const:`~numpy.ma.masked` constant.

Examples
'''''''''
   >>> series
   timeseries([ 0  1  2  3  4  5  6  7  8  9 10 11],
      dates = [Jan-2005 ... Dec-2007],
      freq  = M)

Masking the first and last elements of the series
   >>> series[[0, -1]] = ma.masked
   >>> series
   timeseries([-- 1 2 3 4 5 6 7 8 9 10 --],
      dates = [Jan-2005 ... Dec-2005],
      freq  = M)

Setting the elements falling on the second quarter to 0
   >>> series[series.quarter == 2] = 0
   >>> series
   timeseries([-- 1 2 0 0 0 6 7 8 9 10 --],
      dates = [Jan-2005 ... Dec-2005],
      freq  = M)


