Introduction
============

The :mod:`scikits.timeseries` module provides classes and functions for
manipulating, reporting, and plotting time series of various frequencies.
The focus is on convenient data access and manipulation while leveraging
the existing mathematical functionality in numpy and scipy.

The :mod:`scikits.timeseries` module originally grew out of a need to manipulate
financial and economic data of business day and monthly frequencies. The need
to compare monthly data with business frequency data led to the creation of
a series of frequency conversion algorithms (implemented in C for extra
speed) for reshaping the series. The initial version was released as a scipy 
"sandbox" module. Pierre Gerard-Marchant took the original prototype and adapted
it to be based on the :class:`numpy.ma.MaskedArray` class for handling missing data
in order to work with climate data.

If the following scenarios sound familiar to you, then you will likely find
the scikits.timeseries module useful:

* Compare many time series with different ranges of data (eg. stock prices)
* Determine the last business day of the previous month/quarter/year for
  reporting purposes
* Create time series plots with intelligently spaced axis labels
* Convert a daily time series to monthly by taking the average value during
  each month
* Compute a moving standard deviation *efficiently*.

These are just some of the scenarios that are made very simple with the
scikits.timeseries module.


.. toctree::

   license
