************
Introduction
************

About
=====

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


**********
Installing
**********

Dependencies
============

Requirements
------------

In order to use the `scikits.timeseries` package, the following external
packages must be installed beforehand:

Python_ 2.4 or later.
   Please note that Python_ 3 is not supported yet.

setuptools_
   `scikits` is a namespace package, and as a result every `scikit` requires
   setuptools_ to be installed to function properly.

Numpy_ 1.3.0 or later.
   Numpy_ is a library to manipulate large arrays of numerical data.
   Version 1.3.0 provides improved support to `numpy.ma.MaskedArray` objects
   with structured datatype.

.. _Python: http://www.python.org/download/
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _Numpy: http://www.scipy.org/Download


Optional
--------

The following packages are strongly recommended:

SciPy_ 0.7.0 or later:
   SciPy_ is a set of Numpy_\-based tools for engineering and scientific applications.
   Some of the `scikits.timeseries.lib` sub-modules (`lib.interpolate`,
   `lib.moving_funcs`...) use SciPy_ interpolation and signal functions.

matplotlib_ 0.98.0 or later:
   matplotlib_ is a Python 2D plotting library.
   `scikits.timeseries` includes some extensions to matplotlib_ to plot
   time series.

PyTables_ 2.0 or later:
   PyTables_ is a package for managing hierarchical datasets, using
   the `HDF5 <http://www.hdfgroup.org/HDF5/>`_ format.
   `scikits.timeseries` provides support to store time series with missing data.

.. _SciPy: http://www.scipy.org/Download
.. _matplotlib: http://matplotlib.sourceforge.net
.. _PyTables: http://www.pytables.org



Download
========

You can download source code and windows installers from the
`sourceforge project page <https://sourceforge.net/project/showfiles.php?group_id=239604>`_.

For svn repository access::

   svn co http://svn.scipy.org/svn/scikits/trunk/timeseries timeseries


Installation
============

To install on windows, it is recommend that you use the pre-built installers
from the
`sourceforge project page <https://sourceforge.net/project/showfiles.php?group_id=239604>`_.

To install the `scikits.timeseries` package directly from source, run the
command::

    python setup.py install
