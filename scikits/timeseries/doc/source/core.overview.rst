.. currentmodule:: scikits.timeseries

Overview
========

Note, the following imports are assumed throughout the documentation::

   >>> import numpy as np
   >>> import numpy.ma as ma
   >>> import datetime
   >>> import scikits.timeseries as ts

There are three core objects central to the :mod:`scikits.timeseries` package.
They are :class:`~Date`, :class:`~DateArray`, and :class:`~TimeSeries`. The
sections that follow will go into detail on each of these classes. Understanding
how these classes work and interact with each other is key to understanding the
:mod:`scikits.timeseries` package as a whole.

The :class:`~TimeSeries` class is essentially just a subclass of
:class:`~numpy.ma.MaskedArray` with a date/time dimension attached to it
(using a :class:`~DateArray`). Thus, it is helpful to have a solid understanding
of how MaskedArray's behave before diving into the :mod:`scikits.timeseries`
package. Documentation for the :mod:`numpy.ma` module can be found at
`<http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_

As an example, let us construct a series of 600 random elements, starting 600
business days ago, at a business day frequency:

   >>> data = np.random.uniform(-100,100,600)
   >>> today = ts.now('B')
   >>> series = ts.time_series(data, dtype=np.float32, freq='B', start_date=today-600)

We can check that ``series.dates`` is a :class:`~DateArray` object and that
``series.series`` is a :class:`~numpy.ma.MaskedArray` object.

   >>> isinstance(series.dates, ts.DateArray)
   True
   >>> isinstance(series.series, ma.MaskedArray)
   True
