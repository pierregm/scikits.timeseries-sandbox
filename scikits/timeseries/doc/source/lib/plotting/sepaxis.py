import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scikits.timeseries as ts
import scikits.timeseries.lib.plotlib as tpl

# generate some random data
data1 = np.cumprod(1 + np.random.normal(0, 1, 300)/100)
data2 = np.cumprod(1 + np.random.normal(0, 1, 300)/100)*100
start_date = ts.Date(freq='M', year=1982, month=1)
series1 = ts.time_series(data1, start_date=start_date-50)
series2 = ts.time_series(data2, start_date=start_date)
fig = tpl.tsfigure()
fsp = fig.add_tsplot(111)
# plot series on left axis
fsp.tsplot(series1, 'b-', label='<- left series')
fsp.set_ylim(ma.min(series1.series), ma.max(series1.series))
# create right axis
fsp_right = fsp.add_yaxis(position='right', yscale='log')
# plot series on right axis
fsp_right.tsplot(series2, 'r-', label='-> right series')
fsp_right.set_ylim(ma.min(series2.series), ma.max(series2.series))
# setup legend
fsp.legend(
    (fsp.lines[-1], fsp_right.lines[-1]),
    (fsp.lines[-1].get_label(), fsp_right.lines[-1].get_label()),
)
plt.show()
