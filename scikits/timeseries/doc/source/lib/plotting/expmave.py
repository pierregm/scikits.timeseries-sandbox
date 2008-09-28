import numpy as np
import matplotlib.pyplot as plt
import scikits.timeseries as ts
import scikits.timeseries.lib.plotlib as tpl
from scikits.timeseries.lib.moving_funcs import mov_average_expw

# generate some random data
data = np.cumprod(1 + np.random.normal(0, 1, 300)/100)
series = ts.time_series(data,
                       start_date=ts.Date(freq='M', year=1982, month=1))
fig = tpl.tsfigure()
fsp = fig.add_tsplot(111)
fsp.tsplot(series, '-', mov_average_expw(series, 40), 'r--')
plt.show()
