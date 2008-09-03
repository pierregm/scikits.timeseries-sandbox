.. currentmodule:: scikits.timeseries.tdates

======================
:class:`Dates` objects
======================

.. automodule:: scikits.timeseries.tdates



.. _date_class:

:class:`Date` objects
=====================

.. class:: Date

   Defines a Date object, as the combination of a date and a frequency.
   Several options are available to construct a Date object explicitly:

   * Give appropriate values to the ``year``, ``month``, ``day``, ``quarter``,
     ``hours``, ``minutes``, ``seconds`` arguments.

     >>> Date(freq='Q',year=2004,quarter=3)
     >>> Date(freq='D',year=2001,month=1,day=1)

   * Use the ``string`` keyword. This method uses a modified version of the
     :mod:`mx.DateTime` parser submodule. More information is available in its
     `documentation <http://www.egenix.com/products/python/mxBase/mxDateTime/mxDateTime.pdf>`_.

     >>> Date('D', '2007-01-01')

   * Use the ``datetime`` keyword with an existing :class:`datetime.datetime` object.

     >>> Date('D', datetime=datetime.datetime.now())

   * Use the ``value`` keyword and provide an integer representation of the date.
   
     >>> ts.Date('D', value=732677)
     <D : 01-Jan-2007>
   

Attributes
----------

Base attributes
~~~~~~~~~~~~~~~

.. attribute:: value

   Returns the integer coding the :class:`Date` object.
   
   >>> Date('D','2001-01-01').value
   730486


.. attribute:: datetime

   Returns the object as a :class:`datetime.datetime` object.


Date information
~~~~~~~~~~~~~~~~

The following attributes are integers:

.. tabularcolumns:: |l|l|c|

+------------------------------+---------------------------------------+-------+
| Name                         | Description                           | Range |
+==============================+=======================================+=======+
| .. attribute:: year          | Year                                  | ...   |
+------------------------------+---------------------------------------+-------+
| .. attribute:: qyear         | Quarter Year (1)                      | ...   |
+------------------------------+---------------------------------------+-------+
| .. attribute:: quarter       | Quarter                               | 1-4   |
+------------------------------+---------------------------------------+-------+
| .. attribute:: month         | Month                                 | 1-12  |
+------------------------------+---------------------------------------+-------+
| .. attribute:: week          | Week                                  | 1-53  |
+------------------------------+---------------------------------------+-------+
| .. attribute:: day           | Day of the month                      | 1-31  |
+------------------------------+---------------------------------------+-------+
| .. attribute:: day_of_week   | Day of the week, starting Monday      | 0-6   |
| .. attribute:: weekday       |                                       |       |
+------------------------------+---------------------------------------+-------+
| .. attribute:: day_of_year   | Day of the year, starting in Jan-01   | 1-366 |
+------------------------------+---------------------------------------+-------+
| .. attribute:: hour          | Hour                                  | 00-23 |
+------------------------------+---------------------------------------+-------+
| .. attribute:: minute        | Minute                                | 00-59 |
+------------------------------+---------------------------------------+-------+
| .. attribute:: second        | Seconds                               | 00-59 |
+------------------------------+---------------------------------------+-------+

Notes

(1)  For :class:`Date` objects with a quarterly frequency dates, 
     :attr:`qyear` returns the year corresponding to the ending (starting) month
     of the year. When using ``QTR`` or ``QTR-E`` based quarterly frequencies,
     this is the fiscal year in a financial context.
     For non-quarterly dates, this simply returns the year of the date.


Frequency information
~~~~~~~~~~~~~~~~~~~~~

.. attribute:: Date.freq

   Frequency of the :class:`Date` object, as an integer.

   >>> now('D').freq
   6000


.. attribute:: Date.freqstr

   Frequency of the :class:`Date` object, as a string.

   >>> now('D').freqstr
   'D'


Methods
-------

.. method:: Date.toordinal()

   Returns the proleptic Gregorian ordinal corresponding to the current object.
   The first day of year 1 has a value of 1.

.. automethod:: Date.strfmt
.. method:: Date.strftime(format)

   Returns the string representation of the :class:`Date`, depending on the selected
   format.
   The formats recognized by the :func:`time.strftime` function of the standard
   Python distribution are recognized. Some additional directives are also recognized:
   
   * ``'%q'`` : Quarter of the date

     >>> Date('D', '2001-01-01').strftime('%q')
     '1'

   * ``'%f'`` : Last two digits of the :attr:`qyear` attribute.
   * ``'%F'`` : Four digit of the :attr:`qyear` attribute.

.. automethod:: Date.asfreq


.. _date_function : 

Functions
=========

Two convenience functions return :class:`Date` objects:

.. autofunction:: now
.. autofunction:: prevbusday



.. _date_frequencies:

Available Frequencies 
======================

.. tabularcolumns:: |l|l|

+----------------+--------+-----------------------------------------------------------------------+
| CONSTANT       |  Int   |  String aliases (case insensitive)                                    |
+================+========+=======================================================================+
| **Annual Frequencies**                                                                          |
+----------------+--------+-----------------------------------------------------------------------+
| For annual frequencies, the :attr:`~Date.year` is determined by where the last month            |
| of the year falls.                                                                              |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANN``     |  1000  | ``A``, ``Y``, ``ANNUAL``, ``ANNUALLY``, ``YEAR``, ``YEARLY``          |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNDEC``  |  1000  | ``A-DEC``, ``A-December``, ``Y-DEC``, ``ANNUAL-DEC``, etc...          |
|                |        | (annual frequency with December year end, equivalent to FR_ANN)       |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNNOV``  |  1011  | ``A-NOV``, ``A-NOVEMBER``, ``Y-NOVEMBER``, ``ANNUAL-NOV``, etc...     |
|                |        | (annual frequency with November year end)                             |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNOCT``  |  1010  | ``A-OCT``, ``A-OCTOBER``, ``Y-OCTOBER``, ``ANNUAL-OCT``, etc...       |
|                |        | (annual frequency with October year end)                              |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNSEP``  |  1009  | ``A-SEP``, ``A-SEPTEMBER``, ``Y-SEPTEMBER``, ``ANNUAL-SEP``, etc...   |
|                |        | (annual frequency with September year end)                            |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNAUG``  |  1008  | ``A-AUG``, ``A-AUGUST``, ``Y-AUGUST``, ``ANNUAL-AUG``, etc...         |
|                |        | (annual frequency with August year end)                               |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNJUL``  |  1007  | ``A-JUL``, ``A-JULY``, ``Y-JULY``, ``ANNUAL-JUL``, etc...             |
|                |        | (annual frequency with July year end)                                 |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNJUN``  |  1006  | ``A-JUN``, ``A-JUNE``, ``Y-JUNE``, ``ANNUAL-JUN``, etc...             |
|                |        | (annual frequency with June year end)                                 |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNMAY``  |  1005  | ``A-MAY``, ``Y-MAY``, ``YEARLY-MAY``, ``ANNUAL-MAY``, etc...          |
|                |        | (annual frequency with May year end)                                  |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNAPR``  |  1004  | ``A-APR``, ``A-APRIL``, ``Y-APRIL``, ``ANNUAL-APR``, etc...           |
|                |        | (annual frequency with April year end)                                |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNMAR``  |  1003  | ``A-MAR``, ``A-MARCH``, ``Y-MARCH``, ``ANNUAL-MAR``, etc...           |
|                |        | (annual frequency with March year end)                                |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNFEB``  |  1002  | ``A-FEB``, ``A-FEBRUARY``, ``Y-FEBRUARY``, ``ANNUAL-FEB``, etc...     |
|                |        | (annual frequency with February year end)                             |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_ANNJAN``  |  1001  | ``A-JAN``, ``A-JANUARY``, ``Y-JANUARY``, ``ANNUAL-JAN``, etc...       |
|                |        | (annual frequency with January year end)                              |
+----------------+--------+-----------------------------------------------------------------------+
+----------------+--------+-----------------------------------------------------------------------+
| **Quarterly frequencies**                                                                       |
+----------------+--------+-----------------------------------------------------------------------+
|For the following quarterly frequencies, :attr:`year` is determined by where the last            |
| quarter of the current group of quarters **ENDS**.                                              |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTR``     |  2000  | ``Q``, ``QUARTER``, ``QUARTERLY``                                     |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTREDEC`` |  2000  | ``Q-DEC``, ``QTR-December``, ``QUARTERLY-DEC``, etc...                |
|                |        | (quarterly frequency with December year end, equivalent to ``FR_QTR``)|
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRENOV`` |  2011  | ``Q-NOV``, ``QTR-NOVEMBER``, ``QUARTERLY-NOV``, etc...                |
|                |        | (quarterly frequency with November year end)                          |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTREOCT`` |  2010  | ``Q-OCT``, ``QTR-OCTOBER``, ``QUARTERLY-OCT``, etc...                 |
|                |        | (quarterly frequency with October year end)                           |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRESEP`` |  2009  | ``Q-SEP``, ``QTR-SEPTEMBER``, ``QUARTERLY-SEP``, etc...               |
|                |        | (quarterly frequency with September year end)                         |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTREAUG`` |  2008  | ``Q-AUG``, ``QTR-AUGUST``, ``QUARTERLY-AUG``, etc...                  |
|                |        | (quarterly frequency with August year end)                            |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTREJUL`` |  2007  | ``Q-JUL``, ``QTR-JULY``, ``QUARTERLY-JUL``, etc...                    |
|                |        | (quarterly frequency with July year end)                              |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTREJUN`` |  2006  | ``Q-JUN``, ``QTR-JUNE``, ``QUARTERLY-JUN``, etc...                    |
|                |        | (quarterly frequency with June year end)                              |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTREMAY`` |  2005  | ``Q-MAY``, ``QTR-MAY``, ``QUARTERLY-MAY``, etc...                     |
|                |        | (quarterly frequency with May year end)                               |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTREAPR`` |  2004  | ``Q-APR``, ``QTR-APRIL``, ``QUARTERLY-APR``, etc...                   |
|                |        | (quarterly frequency with April year end)                             |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTREMAR`` |  2003  | ``Q-MAR``, ``QTR-MARCH``, ``QUARTERLY-MAR``, etc...                   |
|                |        | (quarterly frequency with March year end)                             |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTREFEB`` |  2002  | ``Q-FEB``, ``QTR-FEBRUARY``, ``QUARTERLY-FEB``, etc...                |
|                |        | (quarterly frequency with February year end)                          |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTREJAN`` |  2001  | ``Q-JAN``, ``QTR-JANUARY``, ``QUARTERLY-JAN``, etc...                 |
|                |        | (quarterly frequency with January year end)                           |
+----------------+--------+-----------------------------------------------------------------------+
+----------------+--------+-----------------------------------------------------------------------+
|For the following quarterly frequencies, :attr:`year` is determined by where the last            |
| quarter of the current group of quarters **STARTS**.                                            |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSDEC`` |  2012  | ``Q-S-DEC``, ``QTR-S-December``, etc...                               |
|                |        | (quarterly frequency with December year end)                          |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSNOV`` |  2023  | ``Q-S-NOV``, ``QTR-S-NOVEMBER``, etc...                               |
|                |        | (quarterly frequency with November year end)                          |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSOCT`` |  2022  | ``Q-S-OCT``, ``QTR-S-OCTOBER``, etc...                                |
|                |        | (quarterly frequency with October year end)                           |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSSEP`` |  2021  | ``Q-S-SEP``, ``QTR-S-SEPTEMBER``, etc...                              |
|                |        | (quarterly frequency with September year end)                         |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSAUG`` |  2020  | ``Q-S-AUG``, ``QTR-S-AUGUST``, etc...                                 |
|                |        | (quarterly frequency with August year end)                            |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSJUL`` |  2019  | ``Q-S-JUL``, ``QTR-S-JULY``, etc...                                   |
|                |        | (quarterly frequency with July year end)                              |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSJUN`` |  2018  | ``Q-S-JUN``, ``QTR-S-JUNE``, etc...                                   |
|                |        | (quarterly frequency with June year end)                              |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSMAY`` |  2017  | ``Q-S-MAY``, ``QTR-S-MAY``, etc...                                    |
|                |        | (quarterly frequency with May year end)                               |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSAPR`` |  2016  | ``Q-S-APR``, ``QTR-S-APRIL``, etc...                                  |
|                |        | (quarterly frequency with April year end)                             |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSMAR`` |  2015  | ``Q-S-MAR``, ``QTR-S-MARCH``, etc...                                  |
|                |        | (quarterly frequency with March year end)                             |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSFEB`` |  2014  | ``Q-S-FEB``, ``QTR-S-FEBRUARY``, etc...                               |
|                |        | (quarterly frequency with February year end)                          |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_QTRSJAN`` |  2013  | ``Q-S-JAN``, ``QTR-S-JANUARY``, etc...                                |
|                |        | (quarterly frequency with January year end)                           |
+----------------+--------+-----------------------------------------------------------------------+
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_MTH``     |  3000  | ``M``, ``MONTH``, ``MONTHLY``                                         |
+----------------+--------+-----------------------------------------------------------------------+
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_WK``      |  4000  | ``W``, ``WEEK``, ``WEEKLY``                                           |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_WKSUN``   |  4000  | ``W-SUN``, ``WEEK-SUNDAY``, ``WEEKLY-SUN``, etc...                    |
|                |        | (weekly frequency with Sunday being the last day of the week)         |
|                |        | (equivalent to FR_WK)                                                 |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_WKSAT``   |  4006  | ``W-SAT``, ``WEEK-SATURDAY``, ``WEEKLY-SUN``, etc...                  |
|                |        | (weekly frequency with Saturday being the last day of the week)       |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_WKFRI``   |  4005  | ``W-FRI``, ``WEEK-FRIDAY``, ``WEEKLY-FRI``, etc...                    |
|                |        | (weekly frequency with Friday being the last day of the week)         |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_WKTHU``   |  4004  | ``W-THU``, ``WEEK-THURSDAY``, ``WEEKLY-THU``, etc...                  |
|                |        | (weekly frequency with Thursday being the last day of the week)       |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_WKWED``   |  4003  | ``W-WED``, ``WEEK-WEDNESDAY``, ``WEEKLY-WED``, etc...                 |
|                |        | (weekly frequency with Wednesday being the last day of the week)      |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_WKTUE``   |  4002  | ``W-TUE``, ``WEEK-TUESDAY``, ``WEEKLY-TUE``, etc...                   |
|                |        | (weekly frequency with Tuesday being the last day of the week)        |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_WKMON``   |  4001  | ``W-MON``, ``WEEK-MONDAY``, ``WEEKLY-MON``, etc...                    |
|                |        | (weekly frequency with Monday being the last day of the week)         |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_BUS``     |  5000  | ``B``, ``BUSINESS``, ``BUSINESSLY``                                   |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_DAY``     |  6000  | ``D``, ``DAY``, ``DAILY``                                             |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_HR``      |  7000  | ``H``, ``HOUR``, ``HOURLY``                                           |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_MIN``     |  8000  | ``T``, ``MINUTE``, ``MINUTELY``                                       |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_SEC``     |  9000  | ``S``, ``SECOND``, ``SECONDLY``                                       |
+----------------+--------+-----------------------------------------------------------------------+
| ``FR_UND``     | -10000 | ``U``, ``UNDEF``, ``UNDEFINED``                                       |
+----------------+--------+-----------------------------------------------------------------------+


