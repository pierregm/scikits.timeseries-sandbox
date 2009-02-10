.. currentmodule:: scikits.timeseries


Manipulating :class:`TimeSeries` objects
========================================

Because :class:`TimeSeries` are subclasses of :class:`~numpy.ma.MaskedArray`, 
they behave as such for most operations (indexing, arithmetics, comparisons...), 
as described in the corresponding `section <http://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html>`_ of the Numpy documentation.

In the following, only the significant differences will be illustrated.


.. toctree::

   core.timeseries.element_access
   core.timeseries.operations
   core.timeseries.conversion
   core.timeseries.dates_manipulation

..
	Let's construct a list of months, starting on Jan 2005 and ending on Dec 2006, with a gap from Oct 2005 to Jan 2006.

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


	We cannot perform binary operations on these two series (such as adding them together) because the dates of the series do not line up.
	Thus, we need to align them first.

	   >>> (malg_1, malg_2) = ts.align_series(mser_1, mser_2)

	Now we can add the two series.
	Only the data that fall on dates common to the original, non-aligned series will be actually added, the others will be masked.
	After all, we are adding masked arrays.

	   >>> mser_3 = malg_1 + malg_2

	We could have filled the initial series first (replace masked values with a specified value):

	   >>> mser_3 = malg_1.filled(0) + malg_2.filled(0)

	When aligning the series, we could have forced the series to start/end at some given dates:

	   >>> (malg_1,malg_2) = align_series(mser_1_filled, mser2,
	   ...                                start_date='2004-06', end_date='2006-06')


