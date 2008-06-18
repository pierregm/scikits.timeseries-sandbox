"""TimeSeries

:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com
:version: $Id$
"""


__author__ = "Pierre GF Gerard-Marchant  & Matt Knox ($Author$)"
#__version__ = '0.67'
__revision__ = "$Revision$"
__date__     = '$Date$'

import const
import tdates
from tdates import *
import tseries
from tseries import *
import trecords
from trecords import *

from scikits.timeseries.version import __version__

__all__ = ['const', 'tdates','tseries','trecords']
__all__ += tdates.__all__
__all__ += tseries.__all__
__all__ += trecords.__all__
