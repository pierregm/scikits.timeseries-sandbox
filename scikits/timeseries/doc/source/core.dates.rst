.. currentmodule:: scikits.timeseries

.. index::
   single: Date


=====================
:class:`Date` objects
=====================

Even if you have no use for time series in general, you may still find the :class:`Date` class contained in the module quite useful.


.. class:: Date(freq=frequency, *args)

   Defines an object that combines some date- and/or time-related information with a given frequency.
   The frequency can be pictured as the unit into which the date is expressed.

   In practice, :class:`Date` objects are stored internally as integers.
   The conversion to integers and back is controlled by the frequency.
   For example, a :class:`Date` object with a daily frequency corresponds to the gregorian proleptic date, the number of days since January 1st, 1AD; a :class:`Date` with a monthly frequency corresponds to the number of months since January 1AD, and so forth.

   Details about the creation of a :class:`Date` object are presented in a following `section <dates_creation>`_.



.. _dates_creation:

Construction of a :class:`Date` object
======================================

Several options are available to construct a :class:`Date` object explicitly.
In each case, the frequency must be specified with the :keyword:`freq` argument.
Valid frequency specifications are listed in the :ref:`date_frequencies` section.

* Give appropriate values to any of the :keyword:`year`, :keyword:`month`, :keyword:`day`, :keyword:`quarter`, :keyword:`hour`, :keyword:`minute` or :keyword:`second` arguments.

   >>> ts.Date(freq='Q',year=2004,quarter=3)
   <Q : 2004Q3>
   >>> ts.Date(freq='D',year=2001,month=1,day=1)
   <D : 01-Jan-2001>

  Note that there is only need to specify as much information as is relevant depending on the frequency.
  In the example above, there was no need to give a :keyword:`day` input parameter with a quarterly frequency ``'Q'``.

* Use the :keyword:`string` keyword.
  This method uses a modified version of the :mod:`mx.DateTime` parser sub-module. More information is available in its `documentation <http://www.egenix.com/products/python/mxBase/mxDateTime/mxDateTime.pdf>`_.

   >>> ts.Date('D', string='2007-01-01')
   <D : 01-Jan-2007>

* Use the :keyword:`datetime` keyword with an existing :class:`datetime.datetime` or :class:`datetime.date` object.

   >>> ts.Date('D', datetime=datetime.datetime(2007, 1, 1, 17, 0))
   <D : 01-Jan-2007>

* Use the :keyword:`value` keyword and provide an integer representation of the date.

   >>> ts.Date('D', value=732677)
   <D : 01-Jan-2007>
   >>> ts.Date('M', value=(12*2008+6))
   <M : Jun-2009>
   >>> ts.Date('Y', value=2007)
   <A-DEC : 2008>



Manipulating :class:`Date` objects
==================================


Arithmetic operations
---------------------

Integers can be added to or subtracted from a :class:`Date` object to get a new :class:`Date` object.
The frequency of the result is the same as the frequency of the :class:`Date` input::

   >>> yesterday = ts.now('D') - 1
   >>> infivemonths = ts.now('M') + 5


A :class:`Date` object can also be subtracted from another :class:`Date` of the same frequency to calculate the number of periods between the two dates.

   >>> Y = ts.Date('A', year=2007)
   >>> Y_beg = Y.asfreq('D', relation='START')
   >>> Y_end = Y.asfreq('D', relation='END')
   >>> days_in_year = (Y_end - Y_beg + 1)
   >>> days_in_year
   365

If the two :class:`Date` objects have different frequencies, a :exc:`ValueError` exception is raised.


Comparison operations
---------------------

:class:`Date` objects can be compared to integers or other :class:`Date` objects of the same frequency, using the basic Python comparison operators (``==``, ``<``, ``>``, ``<=``, ``>=``, ``!=``) or their method equivalents.

If two :class:`Date` objects of different frequency are compared, a :exc:`ValueError` exception is raised.



Formatting :class:`Dates` as strings
------------------------------------

A :class:`Date` object can simply be casted to a string with the Python function :func:`str`.
A default output format for that frequency will then be used.

.. tip:: Here we should list the formats...

Alternatively, the :meth:`~Date.strftime` method can be used for explicit control.


Attributes
==========

Base attributes
---------------

.. attribute:: value

   Returns the integer coding the :class:`Date` object.
   This attribute is read-only.

   >>> ts.Date('D','2001-01-01').value
   730486


.. attribute:: datetime

   Returns the object as a :class:`datetime.datetime` object.
   This attribute is read-only.

   >>> ts.Date('D','2001-01-01').datetime
   datetime.datetime(2001, 1, 1, 0, 0)



Date information
----------------

The following attributes are read-only, with an integer type.

.. tabularcolumns:: |l|l|c|

+------------------------------+----------------------------------+---------+
| Name                         | Description                      |  Range  |
+==============================+==================================+=========+
| .. attribute:: year          | Year                             |   ...   |
+------------------------------+----------------------------------+---------+
| .. attribute:: qyear         | Quarter Year (1)                 |   ...   |
+------------------------------+----------------------------------+---------+
| .. attribute:: quarter       | Quarter                          | [1-4]   |
+------------------------------+----------------------------------+---------+
| .. attribute:: month         | Month                            | [1-12]  |
+------------------------------+----------------------------------+---------+
| .. attribute:: week          | Week number                      | [1-53]  |
+------------------------------+----------------------------------+---------+
| .. attribute:: day           | Day of the month                 | [1-31]  |
+------------------------------+----------------------------------+---------+
| .. attribute:: day_of_week   | Day of the week, starting Monday | [0-6]   |
| .. attribute:: weekday       |                                  |         |
+------------------------------+----------------------------------+---------+
| .. attribute:: day_of_year   | Day of the year, starting Jan-01 | [1-366] |
+------------------------------+----------------------------------+---------+
| .. attribute:: hour          | Hour                             | [00-23] |
+------------------------------+----------------------------------+---------+
| .. attribute:: minute        | Minute                           | [00-59] |
+------------------------------+----------------------------------+---------+
| .. attribute:: second        | Seconds                          | [00-59] |
+------------------------------+----------------------------------+---------+

.. note::

   (1)
      For :class:`Date` objects with a quarterly frequency, :attr:`qyear` returns the year corresponding to the ending (starting) month of the year. When using ``QTR`` or ``QTR-E`` based quarterly frequencies, it correponds the fiscal year in a financial context.
      For non-quarterly frequencies, this simply returns the corresponding year.



Frequency information
---------------------

.. attribute:: Date.freq

   Frequency of the :class:`Date` object, as an integer.
   This attribute is read-only.

   >>> ts.Date('D', '2001-01-01').freq
   6000


.. attribute:: Date.freqstr

   Frequency of the :class:`Date` object, as a string.
   This attribute is read-only.

   >>> ts.Date('D', '2001-01-01').freqstr
   'D'



Methods
=======

==============================  ===================================================
:meth:`Date.strftime` (format)  Returns a string representation of the instance,
                                according to :keyword:`format`.
:meth:`Date.toordinal` ()       Returns the proleptic Gregorian date as an ordinal.
:meth:`Date.asfreq` (freq)      Returns a new :class:`Date` object with frequency
                                :keyword:`freq`.
==============================  ===================================================


.. _date_function :

Convenience functions
=====================

Two convenience functions are provided to access the current date:

.. autosummary::

   scikits.timeseries.now
   scikits.timeseries.prevbusday

