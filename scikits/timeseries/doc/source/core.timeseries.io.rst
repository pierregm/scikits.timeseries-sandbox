.. currentmodule:: scikits.timeseries


.. _timeseries_input:

Loading a :class:`TimeSeries` from a text file
----------------------------------------------

The :mod:`scikits.timeseries` package provides a convenient way to create
a :class:`TimeSeries` object from the information stored in an ASCII file.

.. autofunction:: tsfromtxt



Saving a :class:`TimeSeries` to a text file
-------------------------------------------

Several options are available to store a :class:`TimeSeries` object to an
ASCII file.

A first possibility is to use the :func:`numpy.savetxt` function on a structured
array, after having converted the object with the :meth:`~TimeSeries.toflex`
method.

Another possibility is to use the :class:`~lib.reportlib.Report` class,
described in the :ref:`scikits_timeseries_report` section.
