.. currentmodule:: scikits.timeseries

========
Plotting
========


The :mod:`lib.plotlib` submodule makes it relatively simple to produce
time series plots using matplotlib_.
It relieves the user from the burden of having to setup appropriately spaced and
formatted tick labels.

If you have never used matplotlib_, you should first go through the tutorial on
the matplotlib web-site before following the examples below.


.. _matplotlib: http://matplotlib.sourceforge.net


Examples
========

Adaptation of date_demo2.py from matplotlib tutorial
----------------------------------------------------

.. plot:: yahoo.py
   :include-source:


Monthly Data with an exponential moving average
-----------------------------------------------

.. plot:: expmave.py
   :include-source:

Separate scales for left and right axis
---------------------------------------

.. plot:: sepaxis.py
   :include-source:


Sample plots at various levels of zoom
--------------------------------------

The following charts show daily data being plotted at varying length date ranges.
This demonstrates the dynamic nature of the axis labels.
With interactive plotting, labels will be updated dynamically as you scroll and zoom.

.. plot:: zoom1.py

.. plot:: zoom2.py

.. plot:: zoom3.py

.. plot:: zoom4.py
