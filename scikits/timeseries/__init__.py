"""TimeSeries

:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com
:version: $Id$
"""


__author__ = "Pierre GF Gerard-Marchant  & Matt Knox ($Author$)"
__revision__ = "$Revision$"
__date__     = '$Date$'


try:
    import const as _c
except ImportError:
    msg = """
    Error importing the frequency constant module.
    You should not try to import scikits.timeseries from its source directory.
    Please exit the scikits.timeseries source tree and relaunch the Python
    interpreter from there.
    """
    raise ImportError(msg)


import const
import tdates
from tdates import *
import tseries
from tseries import *
import trecords
from trecords import *

from extras import tsfromtxt

from scikits.timeseries.version import __version__

__all__ = ['_c', 'const', 'tdates','tseries','trecords']
__all__.extend(tdates.__all__)
__all__.extend(tseries.__all__)
__all__.extend(trecords.__all__)
#__all__ += ['tsfromtxt']

from numpy.testing import Tester
test = Tester("scikits.timeseries").test
