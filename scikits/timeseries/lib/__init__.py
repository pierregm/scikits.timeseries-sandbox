"""
:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com
:version: $Id$
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author$)"
__revision__ = "$Revision$"
__date__     = '$Date$'


import interpolate
from interpolate import *
import moving_funcs
from moving_funcs import *

__all__ = []
__all__ += interpolate.__all__
__all__ += moving_funcs.__all__
