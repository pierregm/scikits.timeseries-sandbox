.. currentmodule:: scikits.timeseries

***************
Version History
***************


Release 0.91.1 (Apr. 15, 2009)
==============================

* The display of dates for a series with undefined frequency has been simplified: dates are prsented as integers.
* Calling the :meth:`~TimeSeries.convert` method (or the corresponding function) with an explicit `func` parameter should now be faster when the function `func` accepts an `axis` parameter.
* Methods and functions accepting an `axis` parameter should now return a :class:`TimeSeries` when possible.
* The functions :func:`first_unmasked_values` and :func:`last_unmasked_values` now accept an optional `axis` parameter (with a default of None).
* The method :meth:`~TimeSeries.fill_missing_dates` (or the corresponding function) now works with structured arrays.


Release 0.91.0 (Apr. 1, 2009)
=============================

* First public release.
