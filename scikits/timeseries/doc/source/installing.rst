.. _installing:

**********
Installing
**********

Dependencies
============

Requirements
------------

In order to use the :mod:`scikits.timeseries` package, the following external
packages must be installed before hand:

* `Python <http://www.python.org/download/>`_ 2.4 or later (but not Python 3).

* `setuptools <http://pypi.python.org/pypi/setuptools>`_: scikits is a
  namespace package, and as a result every scikit requires ``setuptools`` to be
  installed to function properly.

* `Numpy <http://www.scipy.org/Download>`_ 1.2.1 or later

Optional
--------

* `Scipy <http://www.scipy.org/Download>`_ 0.7.0 or later:
  Some of the lib sub-modules (:mod:`~scikits.timeseries.lib.interpolate`,
  :mod:`~scikits.timeserires.lib.moving_funcs`) make use of Scipy functions.
* `matplotlib <http://matplotlib.sourceforge.net>`_ 0.98.0 or later:
  matplotlib is required for time series plotting.


Setup
=====

The module includes a setup script which you can use in the standard python
manner to compile the C code. If you have difficulty installing, please ask
for assistance on the
`scipy-user mailing list <http://projects.scipy.org/mailman/listinfo/scipy-user>`_.

The :mod:`scikits.timeseries` module itself is currently only through subversion.
You can download the latest sources by checking out the repository with the
command:
::

   svn co http://svn.scipy.org/svn/scikits/trunk/timeseries timeseries

This command will create a :file:`timeseries` folder in the current directory.
On Windows, you can also use a SVN client such as `Tortoise SVN <http://tortoisesvn.net/>`_.

To install the :mod:`scikits.timeseries` package, run the command
::

    python setup.py install

in the directory you checked out the source code to.
If you are using Windows and are having trouble compiling the module,
please see the following page in the cookbook:
`Compiling Extensions on Windows <http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW>`_

We do plan to begin doing official releases and distributing windows binaries
in the not too distant future. In the mean time, please bear with us.
