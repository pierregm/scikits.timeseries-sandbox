.. _installing:

**********
Installing
**********

Dependencies
============

**Requirements**

In order to use the TimeSeries package, you will first need install the following
external packages:

* `Python 2.4 or later <http://www.python.org/download/>`__ (but not Python 3)

* `setuptools <http://pypi.python.org/pypi/setuptools>`__: scikits is a
  namespace package, and as a result every scikit requires setuptools to be
  installed to function properly.

* `numpy 1.1.0 or later <http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103>`__

**Optional**

* `scipy <http://sourceforge.net/project/showfiles.php?group_id=27747>`__:
  Some of the lib sub-modules (interpolate, moving_funcs) make use of scipy
  functions.
* `matplotlib 0.98.0 or later <http://matplotlib.sourceforge.net>`__:
  matplotlib is required for time series plotting.

Setup
=====

The module includes a setup script which you can use in the standard python
manner to compile the C code. If you have difficulty installing, please ask
for assistance on the
`scipy-user mailing list <http://projects.scipy.org/mailman/listinfo/scipy-user>`__.

The timeseries module itself is currently only through subversion
(http://svn.scipy.org/svn/scikits/trunk/timeseries). To install it, run::

    python setup.py install

in the directory you checked out the source code to. If you are using Windows
and are having trouble compiling the module, please see the following page in
the cookbook:
`Compiling Extensions on Windows <http://www.scipy.org/Cookbook/CompilingExtensionsOnWindowsWithMinGW>`__

The current plan is to begin doing official releases and distributing windows
binaries once an official release of numpy has been made which includes the
new version of masked array. In the mean time, please bear with us.
