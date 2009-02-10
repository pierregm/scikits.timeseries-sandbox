.. currentmodule:: scikits.timeseries


************
Introduction
************

Conventions
===========

By convention, the following imports are assumed throughout the documentation::

   >>> import numpy as np
   >>> import numpy.ma as ma
   >>> import datetime
   >>> import scikits.timeseries as ts

Input variables and keywords are represented in :keyword`this type`. 




Introduction
============

The :mod:`scikits.timeseries` module provides classes and functions for
manipulating, reporting, and plotting time series of various frequencies.
The focus is on convenient data access and manipulation while leveraging
the existing mathematical functionality in numpy and scipy.

If the following scenarios sound familiar to you, then you will likely find
the :mod:`scikits.timeseries` module useful:

* Compare many time series with different ranges of data (eg. stock prices);
* Create time series plots with intelligently spaced axis labels;
* Convert a daily time series to monthly by taking the average value during
  each month;
* Work with data that has missing values;
* Determine the last business day of the previous month/quarter/year for
  reporting purposes;
* Compute a moving standard deviation *efficiently*.

These are just some of the scenarios that are made very simple with the
:mod:`scikits.timeseries` module.



Overview
========

The :mod:`scikits.timeseries` module is desined to manipulate time series.


As an example, let us construct a series of 600 random elements, starting 600
business days ago, at  a business daily frequency:

   >>> data = np.random.uniform(-100,100,600)
   >>> today = ts.now('B')
   >>> series = ts.time_series(data, dtype=np.float_, freq='B', start_date=today-600)

We can check that ``series.dates`` is a :class:`~DateArray` object and that
``series.series`` is a :class:`~numpy.ma.MaskedArray` object.

   >>> isinstance(series.dates, ts.DateArray)
   True
   >>> isinstance(series.series, ma.MaskedArray)
   True


So, if you are already familiar with the :mod:`numpy.ma` module, using the
:mod:`scikits.timeseries` package should be straightforward.
.. Just keep in mind that another attribute is always present, :attr:`dates`.

