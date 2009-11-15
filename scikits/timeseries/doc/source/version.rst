.. currentmodule:: scikits.timeseries

***************
Version History
***************

Release 0.91.3 (Nov. 15, 2009)
==============================

Bug Fixes
---------

* general improvements for tsfromtxt
* accept datetime objects for 'value' positional arg in Date class
* fixes for compatibility with matplotlib 0.99.1
* fix problem with '%j' directive in strftime method
* fix problem with concatenate and 2-d series
* fixed crash in reportlib.Report class when fixed_width=False and a header_row
  were specified at same time

Release 0.91.2 (Aug. 23, 2009)
==============================

Bug Fixes
---------

* fixed mov_average_expw so that the mask of the input is not modified
* adjust_endpoints : allow the input starting and ending dates to be strings
* enhancements to guess_freq function, more frequencies supported
* fixes for compatibility with matplotlib 0.99.0


Release 0.91.1 (Apr. 15, 2009)
==============================

New Features
------------

* Calling the :meth:`~TimeSeries.convert` method (or the corresponding function)
  with an explicit `func` parameter should now be faster when the function
  `func` accepts an `axis` parameter.
* The functions :func:`first_unmasked_val` and :func:`last_unmasked_val`
  now accept an optional `axis` parameter (with a default of None).
* The display of dates for a series with undefined frequency has been
  simplified: dates are presented as integers.
* :func:`date_array` function now accepts strings for start_date and end_date
  parameters.

Bug Fixes
---------

* :meth:`~TimeSeries.max` / :meth:`~TimeSeries.min` methods now work for
  multi-dimension :class:`~TimeSeries` objects
* The method :meth:`~TimeSeries.fill_missing_dates` (or the corresponding
  function) now works with structured arrays.
* Methods and functions accepting an `axis` parameter should now return a
  :class:`TimeSeries` when possible.

Release 0.91.0 (Apr. 1, 2009)
=============================

* First public release.
