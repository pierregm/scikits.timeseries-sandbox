.. currentmodule:: scikits.timeseries


.. _timeseries_dates_manipulation:

Dates manipulation
------------------

.. autosummary::
   :toctree: generated/

   TimeSeries.adjust_endpoints
   TimeSeries.compressed
   TimeSeries.fill_missing_dates


.. _timeseries_shifts:

Shifts and alignments
---------------------


Aligning series
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   adjust_endpoints
   align_series
   aligned



Shifting series
~~~~~~~~~~~~~~~

Rates of change or differences in a :class:`~TimeSeries` can be calculated 
easily using  the :meth:`~TimeSeries.tshift` method.

As an illustration, let us suppose we need to calculate a term-to-term 
return rate for a monthly series ``mser``.
We could initially try something along the lines of

   >>> mser = ts.time_series(np.arange(1., 13.), 
   ...                       start_date=ts.Date('M','2001-01'))
   >>> tot_change = 100 * (mser[3:]/mser[:-3] - 1)
   >>> tot_change
   masked_array(data = [ 300.          150.          100.           
                          75.           60.           50.
                          42.85714286   37.5          33.33333333],
                mask = False,
          fill_value = 1e+20)

This gives the correct numerical result, but as ``mser[3:]`` and ``mser[:-3]`` 
have different starting and ending dates, the result is forced to a plain 
:class:`numpy.ma.MaskedArray`.
Moreover, ``yoy_change`` does not have the same size as the original input 
series, which may be inconvenient.

A solution consists in using the :meth:`~TimeSeries.tshift` method.

   >>> yoy_change = 100 * (mser/mser.tshift(-3, copy=False) - 1)

The command ``mser.tshift(-3, copy=False)`` returns a series with the same 
dates as ``mser``, but with values shifted 3 periods to the right.
The first 3 values of the resulting series are masked.

.. note::
   By default, the :meth:`~TimeSeries.tshift` copies the original series.
   For simpler cases as the one above, a copy may be avoided by using the 
   ``copy=False``.

.. autosummary::
   :toctree: generated/

   TimeSeries.tshift

