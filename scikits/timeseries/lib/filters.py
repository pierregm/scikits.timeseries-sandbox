"""

A collection of filters for timeseries

:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com
:version: $Id: filters.py 3822 2008-01-12 10:06:39Z matthew.brett@gmail.com $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: matthew.brett@gmail.com $)"
__version__ = '1.0'
__revision__ = "$Revision: 3822 $"
__date__     = '$Date: 2008-01-12 05:06:39 -0500 (Sat, 12 Jan 2008) $'

from moving_funcs import mov_average_expw, cmov_average, cmov_mean, \
                         cmov_window

__all__ = ['mov_average_expw'
           'cmov_average', 'cmov_mean', 'cmov_window']
