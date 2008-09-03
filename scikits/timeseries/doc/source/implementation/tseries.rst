.. currentmodule:: scikits.timeseries.tseries

===========================
:class:`TimeSeries` objects
===========================

.. automodule:: scikits.timeseries.tseries
   :synopsis: Define a subclass of ndarray with support of masked data and time-indexing


.. class:: TimeSeries
 
   Bases: :class:`numpy.ma.MaskedArray`.

   :class:`TimeSeries` objects are ndarrays objects that support masked data and
   special time indexing.


As :class:`TimeSeries` objects subclass :class:`~numpy.ma.MaskedArrays`, they
inherit all their attributes and methods, as well as the attributes and methods
of regular ndarrays.
:class:`TimeSeries` objects have also their own specific attributes and methods.



Attributes
----------

   .. attribute:: series

      Returns a view of a :class:`TimeSeries` as a :class:`~numpy.ma.MaskedArrays`.

   .. attribute:: data

      Returns a view of a :class:`TimeSeries` as a :class:`~numpy.ndarray`.

   .. attribute:: dates

      Returns the :class:`DateArray` object of the dates of the series.

In addition, all the attributes of the :class:`DateArray` :attr:`dates` attributes
are directly accessible by :class:`TimeSeries`.



Methods
=======

The following methods access information about the :attr:`dates` attribute:

.. automethod:: TimeSeries.isfull
.. automethod:: TimeSeries.has_missing_dates
.. automethod:: TimeSeries.has_duplicated_dates
.. automethod:: TimeSeries.isvalid


.. automethod:: TimeSeries.date_to_index


Frequency conversions
---------------------

The following method converts :class:`TimeSeries` from one frequency to another.

.. method:: TimeSeries.asfreq(freq)

   Returns a series whose :attr:`dates` has been converted to the new frequency ``freq``.
   The :attr:`series` part remains unchanged.
   Therefore, when converting to a lower frequency, the new series will have
   duplicated dates, whereas when converting to a higher frequency, the new series
   will have missing dates.


.. automethod:: TimeSeries.convert


Format conversions
------------------

The following methods transform the :class:`TimeSeries` to different formats.

.. automethod:: TimeSeries.tolist
.. automethod:: TimeSeries.tofile
.. automethod:: TimeSeries.asrecords


.. automethod:: TimeSeries.split



Functions
=========

.. autofunction:: time_series

.. autofunction:: adjust_endpoints
.. autofunction:: align_series
.. autofunction:: align_with

.. autofunction:: tshift
.. autofunction:: pct
.. autofunction:: fill_missing_dates

.. autofunction:: empty_like



Exceptions
==========

.. autoexception:: TimeSeriesError
   :show-inheritance:
.. autoexception:: TimeSeriesCompatibilityError
   :show-inheritance: