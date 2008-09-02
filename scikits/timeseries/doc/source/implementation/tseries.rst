.. currentmodule:: scikits.timeseries.tseries

:mod:`tseries`
==============

.. automodule:: scikits.timeseries.tseries
   :synopsis: Define a subclass of ndarray with support of masked data and time-indexing


:class:`TimeSeries` object
--------------------------

.. class:: TimeSeries
 
   Bases: :class:`numpy.ma.MaskedArray`.

   :class:`TimeSeries` objects are ndarrays objects that support masked data and
   special time indexing.


As :class:`TimeSeries` objects subclass :class:`~numpy.ma.MaskedArrays`, they
inherit all their attributes and methods, as well as the attributes and methods
of regular ndarrays.
In addition, :class:`TimeSeries` objects have some specific attributs and methods.


Attributes
~~~~~~~~~~

   .. attribute:: series

      Returns a view of a :class:`TimeSeries` as a :class:`~numpy.ma.MaskedArrays`.

   .. attribute:: data

      Returns a view of a :class:`TimeSeries` as a :class:`~numpy.ndarray`.

   .. attribute:: dates

      Returns the :class:`DateArray` object of the dates of the series.

   .. attribute:: start_date

      Returns the first date of the series, as a :class:`~scikits.timeseries.tdates.Date` object.

   .. attribute:: end_date

      Returns the last date of the series, as a :class:`~scikits.timeseries.tdates.Date` object.

   .. attribute:: freq

      Returns the frequency of the series, as an integer.

   .. attribute:: freqstr

      Returns the frequency of the series, as a string.

   .. attribute:: year, years

      Returns the year for each date of the array, as an integer.

   .. attribute:: quarter, quarters

      Returns the quarter for each date of the series, as an integer between 1 and 4.

   .. attribute:: month, months

      Returns the month for each date of the series, as an integer between 1 and 12.

   .. attribute:: week, weeks

      Returns the week for each date of the series, as an integer between 1 and 53.

   .. attribute:: day, days

      Returns the day of month for each date of the series, as an integer between 1 and 31.

   .. attribute:: weekday, weekdays

      Returns the day of the week for each date of the series, as an integer between 0 and 6.

   .. attribute:: day_of_year, yeardays

      Returns the day of year for each date of the series, as an integer between 1 and 366.

   .. attribute:: hour, hours

      Returns the hour for each date of the series, as an integer between 0 and 23.

   .. attribute:: minute, minutes

      Returns the minute for each date of the series, as an integer between 0 and 59.

   .. attribute:: second, seconds

      Returns the second of each date of the series, as an integer between 0 and 23.


Methods
~~~~~~~

The following methods access information about the :attr:`dates` attribute:

.. automethod:: TimeSeries.isfull
.. automethod:: TimeSeries.has_missing_dates
.. automethod:: TimeSeries.has_duplicated_dates
.. automethod:: TimeSeries.isvalid


.. automethod:: TimeSeries.date_to_index


The following method converts :class:`TimeSeries` from one frequency to another.

.. automethod:: TimeSeries.asfreq
.. automethod:: TimeSeries.convert


The following methods transform the :class:`TimeSeries` to different formats.

.. automethod:: TimeSeries.tolist
.. automethod:: TimeSeries.tofile
.. automethod:: TimeSeries.asrecords


.. automethod:: TimeSeries.split

Functions
---------

.. autofunction:: time_series

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