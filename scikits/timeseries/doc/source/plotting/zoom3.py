import numpy as np
import matplotlib.pyplot as plt
import scikits.timeseries as ts
import scikits.timeseries.lib.plotlib as tpl

# generate some random data
num_points = 250
data = np.cumprod(1 + np.random.normal(0, 1, num_points)/100)
series = ts.time_series(data, start_date=ts.now('d')-num_points)
fig = tpl.tsfigure()
fsp = fig.add_tsplot(111)
fsp.tsplot(series, '-')
fsp.set_xlim(int(series.start_date), int(series.end_date))
fsp.set_title('%i days' % num_points)
plt.show()
