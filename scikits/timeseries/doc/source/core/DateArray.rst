.. currentmodule:: scikits.timeseries

===========================
:class:`DateArray` objects
===========================

.. autoclass:: DateArray

Construction
============

To construct a :class:`~DateArray` object, you can use the factory function
:func:`~date_array` (preferred), or call the class directly.

.. autofunction:: date_array

Attributes
==========

Frequency information
---------------------

.. attribute:: DateArray.freq

   Returns the frequency of the series, as an integer.
   This attribute is read-only.

.. attribute:: DateArray.freqstr

   Returns the frequency of the series, as a string.
   This attribute is read-only.


Date information
----------------

The following attributes give some information about the dates.
They are read-only.

+--------------------------------------+---------------------------------------+-------+
| Name                                 | Description                           | Range |
+======================================+=======================================+=======+
| Integer ndarrays, with the same size as the :class:`DateArray` object.               |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.year        | Year                                  | ...   |
| .. attribute:: DateArray.years       |                                       |       |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.qyear       | Quarter Year (1)                      | ...   |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.quarter     | Quarter                               | 1-4   |
| .. attribute:: DateArray.quarters    |                                       |       |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.month       | Month                                 | 1-12  |
| .. attribute:: DateArray.months      |                                       |       |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.week        | Week                                  | 1-53  |
| .. attribute:: DateArray.weeks       |                                       |       |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.day         | Day of the month                      | 1-31  |
| .. attribute:: DateArray.days        |                                       |       |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.day_of_week | Day of the week, starting Monday      | 0-6   |
| .. attribute:: DateArray.weekdays    |                                       |       |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.day_of_year | Day of the year, starting in Jan-01   | 1-366 |
| .. attribute:: DateArray.yeardays    |                                       |       |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.hour        | Hour                                  | 00-23 |
| .. attribute:: DateArray.hours       |                                       |       |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.minute      | Minute                                | 00-59 |
| .. attribute:: DateArray.minutes     |                                       |       |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.second      | Seconds                               | 00-59 |
| .. attribute:: DateArray.seconds     |                                       |       |
+--------------------------------------+---------------------------------------+-------+
| Single :class:`Date` objects                                                         |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.start_date  | First date of the series              |       |
+--------------------------------------+---------------------------------------+-------+
| .. attribute:: DateArray.end_date    | Last date of the series               |       |
+--------------------------------------+---------------------------------------+-------+


Methods
=======

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



.. _datearray-functions :

Functions
=========

.. autofunction:: fill_missing_dates
.. autofunction:: period_break



.. _date-exceptions :

Exceptions
==========

.. autoexception:: DateError
.. autoexception:: ArithmeticDateError
.. autoexception:: FrequencyDateError
.. autoexception:: InsufficientDateError
