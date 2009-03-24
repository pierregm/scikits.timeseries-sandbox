.. currentmodule:: scikits.timeseries

.. _timeseries_operations:

Arithmetic and comparison operations
------------------------------------

The unary and binary operations defined in :mod:`numpy` or :mod:`numpy.ma`
can be directly applied to :class:`TimeSeries` objects.

.. note::
   If possible, it is recommended to use the :mod:`numpy.ma` function instead
   of its standard numpy counterpart.
   The reason is that the :meth:`reduce` and :meth:`accumulate` methods of
   most numpy.ufuncs (such as :func:`~numpy.add` or :func:`~numpy.multiply`)
   cannot properly handle masked values.



Unary operations
~~~~~~~~~~~~~~~~

When applied to a :class:`TimeSeries`, unary functions that operate on a whole
array at once (like :func:`numpy.sum`) return a scalar or
the :const:`numpy.ma.masked` constant, depending on whether some entries are
not invalid or if all the entries are masked.
If the function can be applied on an axis, the result is
a :class:`~numpy.ma.MaskedArray`.

Unary functions that operate element-wise on an array (like :func:`numpy.log`)
return a  new :class:`TimeSeries` object with the same dates and frequency
as the input, and whose :attr:`series` attribute is the result of the operation
on the input :attr:`series`.

Output values are masked if the corresponding input values are themselves
masked, or if they fall outside the validity domain of the operation.

Examples
''''''''

   >>> s = ts.time_series([-2, -1, 0, 1, 2, 3], mask=[0, 0, 0, 0, 1, 0]
   ...                    start_date=ts.Date('M','2001-01'))
   >>> logs = ma.log(s)
   >>> logs
   timeseries([-- -- -- 0.0 -- 1.09861228867],
      dates = [Jan-2001 ... Jun-2001],
      freq  = M)

.. note::
   In the previous example, ``logs[-2]`` is masked because ``s[-2]`` is
   itself masked; ``logs[:3]`` is masked because ``s[:3] <= 0``.



Binary operations
~~~~~~~~~~~~~~~~~

The binary operations defined in :mod:`numpy` or :mod:`numpy.ma` can also be
directly applied to :class:`TimeSeries` if the second input is a scalar,
a sequence, a :class:`~numpy.ndarray` or a :class:`~numpy.ma.MaskedArray`.
The standard `broadcasting <http://docs.scipy.org/doc/numpy/reference/ufuncs.html#index-121>`_ rules about shape compatibility apply.

When the second input is another :class:`TimeSeries` object, the two series
must satisfy the following conditions:

   * they must have the same frequency;
   * they must be sorted in chronological order;
   * they must have matching dates;
   * they must have the same :attr:`shape`.

Note that the two series may have duplicated and/or missing dates.

If any of these conditions is not satisfied, the result is a standard
:class:`MaskedArray`.
Otherwise, the result is a new :class:`TimeSeries` object, with the same
:attr:`dates` as the two inputs.

The function :func:`~align_series` (or its alias :func:`~aligned`) forces
series to have matching starting and ending dates.
By default, the starting date will be set to the smallest starting date
sof the series, and the ending date to the largest.



Examples
''''''''

Adding two series with compatible dates and frequency
   >>> a = ts.time_series([1, 2, 3],
   ...                    dates=[2001, 2002, 2003], freq='A')
   >>> b = ts.time_series([10, 20, 30],
   ...                    dates=[2001, 2002, 2003], freq='A')
   >>> a+b
   timeseries([11 22 33],
      dates = [2001 ... 2003],
      freq  = A-DEC)

Adding two series with incompatible dates
   >>> b = ts.time_series([1,2,3], dates=[2001,2001,2003], freq='A')
   >>> a+b
   masked_array(data = [11 22 33],
             mask = False,
       fill_value = 999999)
