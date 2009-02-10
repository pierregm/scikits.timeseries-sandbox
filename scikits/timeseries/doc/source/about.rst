About
-----


History
~~~~~~~

The :mod:`scikits.timeseries` module was originally developed by Matt Knox to manipulate financial and economic data of daily (Monday-Friday) and monthly frequencies and to compare data series of differing frequencies. 
Matt created a large number of frequency conversion algorithms (implemented in C for extra speed) for reshaping the series.
The initial version was released winter 2006 as a module in the :mod:`SciPy` sandbox.

Pierre Gerard-Marchant rewrote the original prototype late December 2006 and adapted it to be based on the :class:`numpy.ma.MaskedArray` class for handling missing data in order to work with environmental time series.


