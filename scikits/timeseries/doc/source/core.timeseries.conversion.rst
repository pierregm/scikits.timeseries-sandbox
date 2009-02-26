.. currentmodule:: scikits.timeseries


.. _timeseries_conversion:

Conversion of :class:`TimeSeries`
---------------------------------

... to other types
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   TimeSeries.__float__
   TimeSeries.__hex__
   TimeSeries.__int__
   TimeSeries.__long__
   TimeSeries.__oct__

   TimeSeries.view
   TimeSeries.astype
   TimeSeries.byteswap


... to other objects
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   TimeSeries.tofile
   TimeSeries.toflex
   TimeSeries.tolist
   TimeSeries.torecords
   TimeSeries.tostring


... to other frequencies
~~~~~~~~~~~~~~~~~~~~~~~~

The following methods converts :class:`TimeSeries` from one frequency to another.

.. autosummary::
   :toctree: generated/

   TimeSeries.asfreq
   TimeSeries.convert


.. warning::
   Be careful not to confuse the two methods :meth:`~TimeSeries.asfreq` 
   and :meth:`~TimeSeries.convert`.

   * :meth:`~TimeSeries.asfreq` simply takes every date
     in the :attr:`~TimeSeries.dates` attribute of the
     :class:`~TimeSeries` instance and changes it to the specified frequency,
     so the resulting series will have the same shape as the original series.

   * :meth:`~TimeSeries.convert` is a more complicated function
     that takes a series with no missing nor duplicated dates and creates a series
     at the new frequency with no missing nor duplicated dates and intelligently
     places the data from the original series into appropriate points in the new
     series.


.. autosummary::
   :toctree: generated/

   convert
   scikits.timeseries.extras.convert_to_annual
