.. currentmodule:: scikits.timeseries.lib.moving_funcs


=======================
Moving Window Functions
=======================


The :mod:`lib.moving_funcs` submodule contains an assortment of functions for
doing moving window calculations such as moving averages, moving standard
deviation, moving correlation, moving median, etc.

These functions are implemented in C and are much more efficient than what is
possible using a brute force approach in pure Python.


Moving windows
--------------

.. autosummary::
   :toctree: generated/

   mov_average
   mov_average_expw
   mov_corr
   mov_cov
   mov_max
   mov_mean
   mov_median
   mov_min
   mov_std
   mov_sum
   mov_var


Centered moving windows
-----------------------

.. autosummary::
   :toctree: generated/

   cmov_average
   cmov_mean
   cmov_window
