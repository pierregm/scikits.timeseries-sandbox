.. currentmodule:: scikits.timeseries

***************
Version History
***************


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
