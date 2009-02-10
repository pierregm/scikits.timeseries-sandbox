.. currentmodule:: scikits.timeseries

============================
:class:`~TimeSeries` objects
============================

.. class:: TimeSeries

   A subclass of :class:`~numpy.ma.MaskedArray` designed to manipulate time series.

   :Parameters:
       **data** : {array_like}
           Data portion of the array.
           Any data that is valid for constructing a :class:`~numpy.ma.MaskedArray` can be used here:
           
           * a sequence of objects (numbers, characters, objects);
           * a :class:`~numpy.ndarray` or one of its subclass.
             In particular, :class:`~numpy.ma.MaskedArray` and :class:`TimeSeries` are recognized.

       **dates** : {DateArray}
           A :class:`DateArray` instance storing the date information.

       **autosort** : {True, False}, optional
           Whether to sort the series in chronological order.

       **\*\*optional_parameters** :
           All the parameters recognized by :class:`~numpy.ma.MaskedArray` are also recognized by :class:`TimeSeries`.

   .. seealso::
      :class:`~numpy.ma.MaskedArray`



A :class:`~TimeSeries` object is the combination of three ndarrays:

* :attr:`dates`: A :class:`~DateArray` object.
* :attr:`data` : A :class:`~numpy.ndarray`.
* :attr:`mask` : A boolean :class:`~numpy.ndarray`, indicating missing or invalid data.

These three arrays can be accessed as attributes of a :class:`~TimeSeries` object.
Another very useful attribute is :attr:`series`, that gives the possibility to directly access :attr:`data` and :attr:`mask` as a masked array.

As :class:`TimeSeries` objects subclass :class:`~numpy.ma.MaskedArray`, they inherit all their attributes and methods, as well as the attributes and methods of regular ndarrays.



Attributes
==========

... specific to :class:`TimeSeries`
-----------------------------------

   .. attribute:: data

      Returns a view of a :class:`TimeSeries` as a :class:`~numpy.ndarray`.
      This attribute is read-only and cannot be directly set.


   .. attribute:: mask

      Returns the mask of the object, as a :class:`~numpy.ndarray`
      with the same shape as :attr:`data`, or as the special value
      :data:`~numpy.ma.nomask` (equivalent to :const:`False`).
      This attribute is writable and can be modified.

      If :attr:`data` has a standard :attr:`~numpy.dtype` (no named fields), 
      the :attr:`~numpy.dtype` of the mask is boolean.
      If :attr:`data` is a structured array with named fields, the mask
      has the same structure as the :attr:`data`\'s, but each field is 
      atomically boolean.

      In any case, a value of :const:`True` in the mask indicates that the
      corresponding value of the series is invalid.

   .. attribute:: series

      Returns a view of a :class:`TimeSeries` as a :class:`~numpy.ma.MaskedArray`.
      This attribute is read-only and cannot be directly set


   .. attribute:: dates

      Returns the :class:`DateArray` object of the dates of the series.
      This attribute is writable and can be modified.
      However, the size of the array must be zero or match either the size of the series or its length.


   .. attribute:: varshape

      Returns the number of equivalent variables for each date.
      If ``varshape == ()``, the series has only one variable and is called a 1V-series.


... direct access to the :attr:`dates` information
--------------------------------------------------

============================== ===========================
:attr:`~DateArray.freq`        :attr:`~DateArray.freqstr`
:attr:`~DateArray.year`        :attr:`~DateArray.years`    
:attr:`~DateArray.qyear`
:attr:`~DateArray.quarter`     :attr:`~DateArray.quarters`
:attr:`~DateArray.month`       :attr:`~DateArray.months`
:attr:`~DateArray.week`        :attr:`~DateArray.weeks`
:attr:`~DateArray.day`         :attr:`~DateArray.days`
:attr:`~DateArray.day_of_week` :attr:`~DateArray.weekdays`
:attr:`~DateArray.day_of_year` :attr:`~DateArray.yeardays`
:attr:`~DateArray.hour`        :attr:`~DateArray.hours`
:attr:`~DateArray.minute`      :attr:`~DateArray.minutes`
:attr:`~DateArray.second`      :attr:`~DateArray.seconds`
:attr:`~DateArray.start_date`  :attr:`~DateArray.end_date`
============================== ===========================


... inherited from :class:`~numpy.ma.MaskedArray`
-------------------------------------------------

.. autosummary::
   :toctree: generated/

   TimeSeries.fill_value
   TimeSeries.baseclass
   TimeSeries.recordmask
   TimeSeries.hardmask
   TimeSeries.sharedmask


... inherited from :class:`~numpy.ndarray`
------------------------------------------

.. autosummary::
   :toctree: generated/

   TimeSeries.base
   TimeSeries.ctypes
   TimeSeries.dtype
   TimeSeries.flags

   TimeSeries.itemsize
   TimeSeries.nbytes
   TimeSeries.ndim
   TimeSeries.shape
   TimeSeries.size
   TimeSeries.strides

   TimeSeries.imag
   TimeSeries.real

   TimeSeries.flat
   TimeSeries.__array_priority__



Construction
============

To construct a :class:`~TimeSeries` object, the simplest method is to directly call the class constructor with the proper parameters.

However, the recommended way is to use the :func:`~time_series` factory function.

.. autofunction:: time_series


.. note::

   By default, the series is automatically sorted in chronological order.
   This behavior can be overwritten by setting the  keyword ``autosort=False``.



Dates and data compatibility
----------------------------

The simplest example of a :class:`TimeSeries` consists in a series ``series`` of one variable, where a date is associated with each element of the array.
In that case, the :attr:`dates` attribute is a :class:`DateArray` with the same size as the underlying array.

For example, we can create a 4-element series:

   >>> first_date = ts.Date('D', '2009-01-01')
   >>> series = ts.time_series([1, 2, 3, 4], start_date=first_date)
   >>> series
   timeseries([1 2 3 4],
      dates = [01-Jan-2009 ... 04-Jan-2009],
      freq  = D)

Note that with the use of the :keyword:`start_date` keyword,  the size 
of the :attr:`dates` attribute is automatically adjusted by :func:`time_series` 
to match the size of the input data.

The :attr:`dates` can now be modified in place.
For example, they can be shifted by one week with the following command.

   >>> series.dates +=7
   >>> series
   timeseries([1 2 3 4],
      dates = [08-Jan-2009 ... 11-Jan-2009],
      freq  = D)

The dates can also be changed by setting the :attr:`dates` attribute to another  :class:`DateArray` object.
In that case, the size of the new dates must match the size of the series, or a :exc:`TimeSeriesCompatibilityError` is raised.
Setting the :attr:`dates` attribute to an object of a different type raises a :exc:`TypeError` exception.


It is often convenient to manipulate a series of several variables at once.
Once possibility is to use a `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ as input, 
as illustrated by the following example:

   >>>  series = ts.time_series(zip(np.random.normal(0, 1, 10),
   ...                              np.random.uniform(0, 1, 10)),
   ...                          dtype=[('norm', float), ('unif', float)],
   ...                          start_date=ts.Date('D', '2001-01-01'))

In this example, ``series`` consists of two fields (`'norm'` and `'unif'`).
Note that in this example, the two fields have the same type (:class:`~numpy.float`), but this is not a requirement.
Each field can be accessed as an independent :class:`TimeSeries` using ``series['norm']`` and ``series['unif']``.

In practice, each individual entry of ``series`` is a :class:`numpy.void` object.
The series as a whole behaves as a 1D masked array, as represented by the :attr:`shape` of the series: ``series.shape = (10,)``.
Because ``series`` is a 1D array, the size of ``series.dates`` must match ``series.size``.

Despite the convenience of this approach to manipulate multi-variable series, it presents a serious disadvantage: structured arrays are usually not recognized by standard numpy functions.

An alternative is then to represent a series as a two-dimensional array, using  columns as variables and rows as actual obervations.
In that case, all the variables must have the same type, and the size of the :attr:`dates` attibute must match the length of the series.

More generally, it is possible to create a multi-variable series as a nD array.
The corresponding dates must then satisfy the condition ``series.dates.size == series.shape[0]`` or a :exc:`TimeSeriesCompatibilityError` is raised.
The specific attribute :attr:`varshape` is then set to keep track of the number of variables.


For example, a series of 50 years of monthly data can be represented as a (600,)-array of observations at a monthly frequency, or as a (50,12)-array of observations at an annual frequency.

   >>> start - ts.Date('M', '2001-01')
   >>> data = np.random.uniform(-1, +1, 50*12).reshape(50, 12)
   >>> mseries = ts.time_series(data, start_date=start, length=50*12)
   >>> aseries = ts.time_series(data, start_date=start.asfreq('Y'), length=50)

Both series have the same shape, ``(50, 12)``, but ``mseries`` is a series of one variable, with ``mseries.varshape == ()``, while ``aseries`` is a series of 12 variables, ``aseries.varshape == (12,)``, each variable corresponding to a month.

   >>> (mseries.shape, mseries.varshape)
   ((50, 12), ())
   >>> (aseries.shape, aseries.varshape)
   ((50, 12), (12,))

Because ``aseries`` is basically a 2D array, we can easily compute annual and monthly means.
Thus, monthly means over the whole 50 years can be calculated at once with the :meth:`~scikits.timeseries.mean` method, using ``axis=0`` as parameter.
We can also compute the equivalent of 50 years of annual data using :meth:`~scikits.timeseries.mean` method, this time with ``axis=1``.

   >>> amean = aseries.mean(axis=1)
   >>> amean.shape = (50,)
   >>> mmean = aseries.mean(axis=0)
   >>> mmean.shape = (12,)


Another example of multi-variable series would be one year of daily (256x256) raster map.
This dataset can easily be represented as a (365,256,256)-array, and a corresponding series created with the following code::

   >>> data = np.random.uniform(-1, +1, 365*256*256).reshape(365, 256, 256)
   >>> newseries = ts.time_series(data, start_date=ts.now('D'))



Methods
=======

Date information
----------------

The following methods access information about the :attr:`dates` attribute:

===========================================  ========================================================================================
.. method:: TimeSeries.get_steps             Returns the time steps between consecutive dates,
                                             in the same unit as the instance frequency.
.. method:: TimeSeries.has_missing_dates     Returns whether the instance has missing dates.
.. method:: TimeSeries.has_duplicated_dates  Returns whether the instance has duplicated dates.
.. method:: TimeSeries.is_full               Returns whether the instance has no missing dates.
.. method:: TimeSeries.is_valid              Returns whether the instance is valid (that there are no missing nor duplicated dates).
.. method:: TimeSeries.is_chronological      Returns whether the instance is sorted in chronological order.
===========================================  ========================================================================================

.. autosummary::
   :toctree: generated/

   TimeSeries.date_to_index
   TimeSeries.sort_chronologically


Dates manipulation
------------------

.. autosummary::
   :toctree: generated/

   TimeSeries.adjust_endpoints
   TimeSeries.compressed
   TimeSeries.fill_missing_dates


Shape manipulation
------------------

For reshape, resize, and transpose, the single tuple argument may be
replaced with ``n`` integers which will be interpreted as an n-tuple.

.. autosummary::
   :toctree: generated/

   TimeSeries.flatten
   TimeSeries.ravel
   TimeSeries.reshape
   TimeSeries.resize
   TimeSeries.split
   TimeSeries.squeeze
   TimeSeries.swapaxes
   TimeSeries.transpose
   TimeSeries.T


Item selection and manipulation
-------------------------------

.. autosummary::
   :toctree: generated/

   TimeSeries.argmax
   TimeSeries.argmin
   TimeSeries.argsort
   TimeSeries.choose
   TimeSeries.compress
   TimeSeries.diagonal
   TimeSeries.fill
   TimeSeries.filled
   TimeSeries.item
   TimeSeries.nonzero
   TimeSeries.put
   TimeSeries.repeat
   TimeSeries.searchsorted
   TimeSeries.sort
   TimeSeries.take
   TimeSeries.tshift


Pickling and copy
-----------------

.. autosummary::
   :toctree: generated/

   TimeSeries.copy
   TimeSeries.dump
   TimeSeries.dumps


Calculations
------------

.. autosummary::
   :toctree: generated/

   TimeSeries.all
   TimeSeries.anom
   TimeSeries.any
   TimeSeries.clip
   TimeSeries.conj
   TimeSeries.conjugate
   TimeSeries.cumprod
   TimeSeries.cumsum
   TimeSeries.max
   TimeSeries.mean
   TimeSeries.min
   TimeSeries.pct
   TimeSeries.prod
   TimeSeries.product
   TimeSeries.ptp
   TimeSeries.round
   TimeSeries.std
   TimeSeries.sum
   TimeSeries.trace
   TimeSeries.var

..
	Arithmetic and comparison operations
	------------------------------------

	.. index:: comparison, arithmetic, operation, operator

	Comparison operators:
	~~~~~~~~~~~~~~~~~~~~~

	.. autosummary::
	   :toctree: generated/

	   TimeSeries.__lt__
	   TimeSeries.__le__
	   TimeSeries.__gt__
	   TimeSeries.__ge__
	   TimeSeries.__eq__
	   TimeSeries.__ne__

	Truth value of an array (:func:`bool()`):
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	.. autosummary::
	   :toctree: generated/

	   TimeSeries.__nonzero__


	Arithmetic:
	~~~~~~~~~~~

	.. autosummary::
	   :toctree: generated/

	   TimeSeries.__abs__
	   TimeSeries.__add__
	   TimeSeries.__radd__
	   TimeSeries.__sub__
	   TimeSeries.__rsub__
	   TimeSeries.__mul__
	   TimeSeries.__rmul__
	   TimeSeries.__div__
	   TimeSeries.__rdiv__
	   TimeSeries.__truediv__
	   TimeSeries.__rtruediv__
	   TimeSeries.__floordiv__
	   TimeSeries.__rfloordiv__
	   TimeSeries.__mod__
	   TimeSeries.__rmod__
	   TimeSeries.__divmod__
	   TimeSeries.__rdivmod__
	   TimeSeries.__pow__
	   TimeSeries.__rpow__
	   TimeSeries.__lshift__
	   TimeSeries.__rlshift__
	   TimeSeries.__rshift__
	   TimeSeries.__rrshift__
	   TimeSeries.__and__
	   TimeSeries.__rand__
	   TimeSeries.__or__
	   TimeSeries.__ror__
	   TimeSeries.__xor__
	   TimeSeries.__rxor__


	Arithmetic, in-place:
	~~~~~~~~~~~~~~~~~~~~~

	.. autosummary::
	   :toctree: generated/
   
	   TimeSeries.__iadd__
	   TimeSeries.__isub__
	   TimeSeries.__imul__
	   TimeSeries.__idiv__
	   TimeSeries.__itruediv__
	   TimeSeries.__ifloordiv__
	   TimeSeries.__imod__
	   TimeSeries.__ipow__
	   TimeSeries.__ilshift__
	   TimeSeries.__irshift__
	   TimeSeries.__iand__
	   TimeSeries.__ior__
	   TimeSeries.__ixor__


	Representation
	--------------

	.. autosummary::
	   :toctree: generated/

	   TimeSeries.__repr__
	   TimeSeries.__str__

	   TimeSeries.ids
	   TimeSeries.iscontiguous


	Special methods
	---------------

	For standard library functions:

	.. autosummary::
	   :toctree: generated/

	   TimeSeries.__copy__
	   TimeSeries.__deepcopy__
	   TimeSeries.__getstate__
	   TimeSeries.__reduce__
	   TimeSeries.__setstate__


	Basic customization:

	.. autosummary::
	   :toctree: generated/

	   TimeSeries.__new__
	   TimeSeries.__array__
	   TimeSeries.__array_wrap__


	Container customization: (see `Indexing <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_)

	.. autosummary::
	   :toctree: generated/

	   TimeSeries.__len__
	   TimeSeries.__getitem__
	   TimeSeries.__setitem__
	   TimeSeries.__delitem__
	   TimeSeries.__getslice__
	   TimeSeries.__setslice__
	   TimeSeries.__contains__



	Methods inherited from :class:`~numpy.ma.MaskedArray`
	-----------------------------------------------------

	Handling the mask
	~~~~~~~~~~~~~~~~~

	The following methods can be used to access information about the mask or to
	manipulate the mask.

	.. autosummary::
	   :toctree: generated/

	   TimeSeries.__setmask__

	   TimeSeries.harden_mask
	   TimeSeries.soften_mask
	   TimeSeries.unshare_mask
	   TimeSeries.shrink_mask


	Handling the :attr:`fill_value`
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	.. autosummary::
	   :toctree: generated/

	   TimeSeries.get_fill_value
	   TimeSeries.set_fill_value


	Counting the missing elements
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	.. autosummary::
	   :toctree: generated/

	   TimeSeries.count

