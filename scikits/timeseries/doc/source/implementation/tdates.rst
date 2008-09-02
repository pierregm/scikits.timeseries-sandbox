.. currentmodule:: scikits.timeseries.tdates

==============================================
:class:`Dates` and :class:`DaterArray` objects
==============================================

.. automodule:: scikits.timeseries.tdates



:class:`Date` objects
=====================

.. autoclass:: Date
   :members:



:class:`DateArray` objects
==========================

.. autoclass:: DateArray


Attributes
----------

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


Information methods
-------------------

The following methods give some information about the distribution of dates.
These methods do not need any argument.

.. automethod:: DateArray.get_steps
.. automethod:: DateArray.isfull
.. automethod:: DateArray.has_missing_dates
.. automethod:: DateArray.has_duplicated_dates
.. automethod:: DateArray.isvalid


Indexing methods
----------------

These methods allow the user to access specific elements of a :class:`DateArray`
by directly giving a date instead of an integer.

.. automethod:: DateArray.date_to_index
.. automethod:: DateArray.find_dates


Conversion methods
------------------

These methods transform a :class:`DateArray` to other formats.

.. automethod:: DateArray.tovalue
.. automethod:: DateArray.toordinal
.. automethod:: DateArray.tostring
.. automethod:: DateArray.tolist



Functions
=========

.. autofunction:: prevbusday
.. autofunction:: fill_missing_dates
.. autofunction:: period_break



Exceptions
==========

.. autoexception:: DateError
.. autoexception:: ArithmeticDateError
.. autoexception:: FrequencyDateError
.. autoexception:: InsufficientDateError

