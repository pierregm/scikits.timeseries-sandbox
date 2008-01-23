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
import dates
from dates import *
import tseries
from tseries import *
import trecords
from trecords import *

import report
from report import *

import lib
from lib import filters, interpolate, moving_funcs

__all__ = ['const', 'dates','tseries','trecords','report','filters',
           'interpolate', 'moving_funcs']
__all__ += dates.__all__
__all__ += tseries.__all__
__all__ += trecords.__all__
__all__ += report.__all__
