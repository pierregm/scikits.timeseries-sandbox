"""TimeSeries

:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com
:version: $Id: __init__.py 3327 2007-09-19 00:51:14Z mattknox_ca $
"""


__author__ = "Pierre GF Gerard-Marchant  & Matt Knox ($Author: mattknox_ca $)"
__version__ = '1.0'
__revision__ = "$Revision: 3327 $"
__date__     = '$Date: 2007-09-18 20:51:14 -0400 (Tue, 18 Sep 2007) $'

import const
import tdates
from tdates import *
import tseries
from tseries import *
import trecords
from trecords import *

from scikits.timeseries.version import version

__all__ = ['const', 'tdates','tseries','trecords']
__all__ += tdates.__all__
__all__ += tseries.__all__
__all__ += trecords.__all__
