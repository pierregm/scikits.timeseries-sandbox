import datetime
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo
import scikits.timeseries as ts
import scikits.timeseries.lib.plotlib as tpl

date1 = datetime.date(2002, 1, 5)
date2 = datetime.date(2003, 12, 1)

# retrieve data from yahoo.
quotes = quotes_historical_yahoo('INTC', date1, date2)

"""
The dates from the yahoo quotes module get returned as integers, which happen
to correspond to the integer representation of 'DAILY' frequency dates in the
scikits.timeseries module. So create a DateArray of daily dates,  then convert
this to business day frequency afterwards.
"""
dates = ts.date_array([q[0] for q in quotes], freq='DAILY').asfreq('BUSINESS')
opens = [q[1] for q in quotes]
raw_series = ts.time_series(opens, dates)

"""
`fill_missing_dates` will insert masked values for any missing data points.
Note that you could plot the series without doing this, but it would cause
missing values to be linearly interpolated rather than left empty in the plot.
"""
series = ts.fill_missing_dates(raw_series)
fig = tpl.tsfigure()
fsp = fig.add_tsplot(111)
fsp.tsplot(series, '-')

"""
Add grid lines at start of each quarter. Grid lines appear at the major tick
marks by default (which, due to the dynamic nature of the ticks for time
series plots, cannot be guaranteed to be at quarter start). So if you want
grid lines to appear at specific intervals, you must first specify xticks
explicitly.
"""
dates = series.dates
quarter_starts = dates[dates.quarter != (dates-1).quarter]
fsp.set_xticks(quarter_starts.tovalue())
fsp.grid()
plt.show()
