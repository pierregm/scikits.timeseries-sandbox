.. _installing:

**********
Installing
**********

Dependencies
============

Requirements
------------

In order to use the :mod:`scikits.timeseries` package, the following external
packages must be installed beforehand:

Python_ 2.4 or later.
   Please note that Python_ 3 is not supported yet.

setuptools_
   :mod:`scikits` is a namespace package, and as a result every :mod:`scikit` requires setuptools_ to be installed to function properly.

Numpy_ 1.3.0 or later.
   Numpy_ is a library to manipulate large arrays of numerical data.
   Version 1.3.0 provides improved support to :class:`~numpy.ma.MaskedArray` objects with structured datatype.

.. _Python: http://www.python.org/download/
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _Numpy: http://www.scipy.org/Download


Optional
--------

The following packages are strongly recommended:

SciPy_ 0.7.0 or later:
   SciPy_ is a set of Numpy_\-based tools for engineering and scientific applications.
   Some of the :mod:`~scikits.timeseries.lib` sub-modules (:mod:`~scikits.timeseries.lib.interpolate`,   :mod:`~scikits.timeseries.lib.moving_funcs`...) use SciPy_ interpolation and signal functions.

matplotlib_ 0.98.0 or later:
   matplotlib_ is a Python 2D plotting library.
   :mod:`scikits.timeseries` includes some extensions to matplotlib_ to plot time series.

PyTables_ 2.0 or later:
   PyTables_ is a package for managing hierarchical datasets, using the `HDF5 <http://www.hdfgroup.org/HDF5/>`_ format.
   :mod:`scikits.timeseries` provides support to store time series with missing data.

.. _SciPy: http://www.scipy.org/Download
.. _matplotlib: http://matplotlib.sourceforge.net
.. _PyTables: http://www.pytables.org



Download
========

The module includes a setup script which you can use in the standard python
manner to compile the C code. If you have difficulty installing, please ask
for assistance on the
`scipy-user mailing list <http://projects.scipy.org/mailman/listinfo/scipy-user>`_.

The :mod:`scikits.timeseries` module itself is currently available through subversion only.
You can download the latest version of the source files by checking out the repository with the command::

   svn co http://svn.scipy.org/svn/scikits/trunk/timeseries timeseries

This command will create a :file:`timeseries` folder in the current directory.
On Windows, you can also use a SVN client such as `Tortoise SVN <http://tortoisesvn.net/>`_.


Installation
============

To install the :mod:`scikits.timeseries` package, run the command::

    python setup.py install

in the directory you checked out the source code to.
If you are using Windows and are having trouble compiling the module,
please see the following page in the cookbook:
`Compiling Extensions on Windows <http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW>`_

We do plan to begin doing official releases and distributing windows binaries in the not too distant future.
In the mean time, please bear with us.
