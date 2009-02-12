.. currentmodule:: scikits.timeseries

===========================
:class:`DateArray` objects
===========================

.. class:: DateArray(dates=None, freq=None, copy=None)
   
   Based on :class:`numpy.ndarray`.

   Defines a :class:`~numpy.ndarray` of dates, as ordinals.
   The :keyword:`dates` parameter must then be a sequence of :class:`Date` 
   objects or a sequence of integers.
   The :keyword:`freq` parameter must be a valid frequency specification, 
   as listed in the :ref:`date_frequencies` section.

When viewed globally (array-wise), a :class:`DateArray` object behaves as an 
:class:`~numpy.ndarray` of integers.
When viewed element-wise, :class:`DateArray` behaves as a standard sequence of 
:class:`Date` objects.

Thus, a test such as::

   >>> DateArray(...) == value

is valid only if ``value`` is an integer, not a :class:`Date` object.
In that case, the test returns a boolean :class:`~numpy.ndarray` that has
the same shape as the :class:`DateArray` object.

However, a loop such as :

   >>> for d in DateArray(...):

accesses the array element by element, and at each iteration, `d` is a 
 :class:`Date` object.


The only requirement for the :class:`Date` elements of a :class:`DateArray` is 
that they must have the same frequency.
Otherwise, they do not have to be in chronological order nor to be regularly 
spaced, and duplicated entries are also permitted.



Construction
============

A :class:`DateArray` object can be created simply by calling the class 
with the required inputs.
However, the recommended method is to use the :func:`~date_array` factory 
function.

.. autofunction:: date_array



.. _datearray_manipulation:

Manipulating :class:`DateArray` objects
=======================================

Accessing elements
------------------

As subclasses of :class:`~numpy.ndarray`, :class:`DateArray` objects follow 
the same rules for accessing elements through 
`indexing <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.
In addition, :class:`DateArray` objects can be indexed with one or several 
:class:`Date` objects.


When a single element of a :class:`DateArray` is accessed, 
the result is a :class:`Date` object with the same frequency as the input.
Otherwise, the result is a :class:`DateArray` with the same frequency 
as the input.
Note that when using a slice to access specific elements 
of a :class:`DateArray`, the result is always a :class:`DateArray`


   >>> dates = ts.date_array(start_date=ts.Date('M', '2001-01'), length=36)
   >>> # Accessing a single element with an integer
   >>> dates[0]
   <M : Jan-2001>
   >>> # Accessing a single element with a Date object
   >>> dates[ts.Date('M', '2002-01')]
   <M : Jan-2002>
   >>> # Using a slice to access a single element : the result is a DateArray
   >>> dates[-1:]
   DateArray([Dec-2003],
             freq='M')
   >>> # Accessing multiple elements with a list of integers
   >>> dates[[0, 12, 24]]
   DateArray([Jan-2001, Jan-2002, Jan-2003],
             freq='M')
   >>> # Accessing multiple elements with a list of Date objects
   >>> dates[[ts.Date('M', '2002-01'), ts.Date('M', '2003-01')]]
   DateArray([Jan-2002, Jan-2003],
             freq='M')
   



Operations
----------

Arithmetic operations
~~~~~~~~~~~~~~~~~~~~~

Arithmetic operations on :class:`DateArray` objects are limited to additions 
and subtractions.
Any other arithmetic operation will raise a :exc:`ArithmeticDateError` exception.

Adding (subtracting) a scalar or a sequence of scalars to (from) 
a :class:`DateArray` returns a :class:`DateArray` with the same frequency.
The shapes of the inputs must be compatible, as described in the 
`broadcasting <http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_ 
section of the Numpy 
`documentation <http://docs.scipy.org/doc/numpy/reference/>`_.

Adding (subtracting) two :class:`DateArray` or a :class:`DateArray` and 
a :class:`Date` is possible only if the two objects have the same frequency.
The result is then a :class:`~numpy.ndarray`.
If the inputs do not share the same frequency, 
a :exc:`FrequencyDateError` exception is raised.



Comparison operations
~~~~~~~~~~~~~~~~~~~~~

All the comparison operations involving the standard Python operators 
(``==``, ``<``, ``>``, ``<=``, ``>=``, ``!=``) or their method equivalents are 
supported between a :class:`DateArray` and:

   (i) a scalar, 
   (ii) a sequence (or array) of scalars, 
   (iii) a :class:`Date` object or 
   (iv) another :class:`DateArray`.

In the last two cases, the objects must have the same frequency, 
or a :exc:`FrequencyDateError` is raised.
If the other input is a sequence, its shape must be compatible with the shape 
of the :class:`DateArray`.

In every case, the result is then an array of booleans, with the same shape as 
the input.

In addition, it is possible to check whether a single :class:`Date` object or 
its equivalent integer value is contained in a :class:`DateArray` with the 
``in`` operator or its method equivalent :meth:`DateArray.__contains__`.



.. _datearray_attributes:

Attributes
==========

Because :class:`DateArray` objects are subclasses of :class:`~numpy.ndarray`, 
they inherit all its `attributes <http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#array-attributes>`_, 
as describe in the Numpy `documentation <http://docs.scipy.org/doc/numpy/reference/>`_.

In addition, they have the specific following attributes.


... inherited from :class:`~numpy.ndarray`
------------------------------------------

.. autosummary::
   :toctree: generated/

   DateArray.base
   DateArray.ctypes
   DateArray.dtype
   DateArray.flags

   DateArray.itemsize
   DateArray.nbytes
   DateArray.ndim
   DateArray.shape
   DateArray.size
   DateArray.strides

   DateArray.imag
   DateArray.real

   DateArray.flat
   DateArray.__array_priority__



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

+--------------------------------------+------------------------------------+---------+
| Name                                 | Description                        |  Range  |
+======================================+====================================+=========+
| Integer ndarrays, with the same size as the instance.                               |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.year        | Year                               |   ...   |
| .. attribute:: DateArray.years       |                                    |         |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.qyear       | Quarter Year (1)                   |   ...   |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.quarter     | Quarter                            | [1-4]   |
| .. attribute:: DateArray.quarters    |                                    |         |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.month       | Month                              | [1-12]  |
| .. attribute:: DateArray.months      |                                    |         |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.week        | Week                               | [1-53]  |
| .. attribute:: DateArray.weeks       |                                    |         |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.day         | Day of the month                   | [1-31]  |
| .. attribute:: DateArray.days        |                                    |         |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.day_of_week | Day of the week, starting Monday   | [0-6]   |
| .. attribute:: DateArray.weekdays    |                                    |         |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.day_of_year | Day of the year, starting Jan-01   | [1-366] |
| .. attribute:: DateArray.yeardays    |                                    |         |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.hour        | Hour                               | [00-23] |
| .. attribute:: DateArray.hours       |                                    |         |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.minute      | Minute                             | [00-59] |
| .. attribute:: DateArray.minutes     |                                    |         |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.second      | Seconds                            | [00-59] |
| .. attribute:: DateArray.seconds     |                                    |         |
+--------------------------------------+------------------------------------+---------+
| Single :class:`Date` objects, with the same frequency as the instance               |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.start_date  | First date of the series           |         |
|                                      | (in chronological order).          |         |
+--------------------------------------+------------------------------------+---------+
| .. attribute:: DateArray.end_date    | Last date of the series            |         |
|                                      | (in chronological order).          |         |
+--------------------------------------+------------------------------------+---------+



Private Attributes
------------------

.. warning::
   The following attributes are *private* and are not meant to be used directly 
   by a user.
   They are reported here for the sake of completeness.

.. attribute:: _cachedinfo

   The :attr:`_cachedinfo` attribute is a directory storing some information
   about the instance.
   It has the following keys:

   ``'chronidx'`` : *{None, ndarray}*
       If not None, a :class:`~numpy.ndarray` of integers corresponding
       to the indices sorting the instance in chronological order.
       If the :class:`DateArray` is already sorted chronologically, then
       `_cachedinfo['chronidx'] = np.array([], dtype=int)`

   ``'full'`` : *{True, False}*
       Whether there are no missing dates in the series.
   
   ``'hasdups'`` : *{True, False}*
       Whether there are duplicating dates in the series.

   ``'ischrono'`` : *{True, False}*
       Whether the instance is sorted in chronological order.

   ``'steps'`` : *{ndarray}*
      Time steps between consecutive dates (in chronological order).

   ``'toobj'`` : *{None, ndarray}*

   ``'toord'`` : *{None, ndarray}*
      :class:`~numpy.ndarray` of integers corresponding to the Gregorian
      proleptic dates.

   ``'tostr'`` : *{None, ndarray}*
     :class:`~numpy.ndarray` of strings corresponding to the string
     representations of the dates.


Methods
=======

Because :class:`DateArray` objects are subclasses of :class:`~numpy.ndarray`, 
they inherit their `methods <http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#array-methods>`_, 
as described in the Numpy documentation. 

In addition, they accept the following specific methods.


Information methods
-------------------

The following methods give some information about the distribution of dates.
They do not need any argument.


==========================================  ========================================================================================
.. method:: DateArray.get_steps             Returns the time steps between consecutive dates,
                                            in the same unit as the instance frequency.
.. method:: DateArray.has_missing_dates     Returns whether the instance has missing dates.
.. method:: DateArray.has_duplicated_dates  Returns whether the instance has duplicated dates.
.. method:: DateArray.is_full               Returns whether the instance has no missing dates.
.. method:: DateArray.is_valid              Returns whether the instance is valid (that there are no missing nor duplicated dates).
.. method:: Date.is_chronological           Returns whether the instance is sorted in chronological order.
==========================================  ========================================================================================



Indexing methods
----------------

These methods allow the user to access specific elements of a :class:`DateArray` 
by directly giving a date instead of an integer.

.. autosummary::
   :toctree: generated/
   
   DateArray.date_to_index
   DateArray.find_dates



Conversion methods
------------------

... to another object
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   
   DateArray.tovalue
   DateArray.tofloat
   DateArray.toordinal
   DateArray.tostring
   DateArray.tolist


... to another frequency
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DateArray.asfreq


Sorting methods
---------------

.. autosummary::
   :toctree: generated/

   DateArray.sort_chronologically



Functions
=========

.. autosummary::
   :toctree: generated/

   period_break
   convert_to_float

