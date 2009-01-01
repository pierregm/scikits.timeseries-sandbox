"""
:author: Pierre Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu & mattknox_ca_at_hotmail_dot_com
:version: $Id: test_interpolate.py 3836 2008-01-15 13:09:03Z matthew.brett@gmail.com $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: matthew.brett@gmail.com $)"
__revision__ = "$Revision: 3836 $"
__date__     = '$Date: 2008-01-15 08:09:03 -0500 (Tue, 15 Jan 2008) $'

import numpy as np
import numpy.ma as ma
from numpy.ma import masked_array, masked

from numpy.testing import *
from numpy.ma.testutils import assert_equal, assert_almost_equal

import scikits.timeseries as ts
from scikits.timeseries.lib.interpolate import \
     backward_fill, forward_fill, interp_masked1d

class TestFuncs(TestCase):

    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.mask = [1,0,1,0,0,1,1,0,0,0]
        self.data = np.arange(10)
        self.test_array = masked_array(self.data, mask=self.mask)

    def test_backward_fill (self):
        result = masked_array(self.data, mask=self.mask)
        result[0] = 1
        result[2] = 3

        assert_equal(backward_fill(self.test_array, maxgap=1), result)

        result[5] = 7
        result[6] = 7

        assert_equal(backward_fill(self.test_array), result)

    def test_forward_fill_old(self):
        result = masked_array(self.data, mask=self.mask)
        result[2] = 1

        assert_equal(forward_fill(self.test_array, maxgap=1), result)

        result[5] = 4
        result[6] = 4

        assert_equal(forward_fill(self.test_array), result)

    def test_interp_fill(self):
        result_lin = masked_array(self.data).astype(float)
        result_lin[0] = masked
        test = interp_masked1d(self.test_array.astype(float), kind='linear')
        assert_almost_equal(test, result_lin)

    def test_forward_fill(self):
        x = ma.arange(20)
        x[(x%5 != 0)] = masked
        # Test forward_fill w/o gaps, starting unmasked
        test = forward_fill(x)
        assert_equal(test, [ 0, 0, 0, 0, 0, 5, 5, 5, 5, 5,
                            10,10,10,10,10,15,15,15,15,15])
        # Test forward_fill w/ gaps, starting unmasked
        test = forward_fill(x, 3)
        assert_equal(test, x)
        assert_equal(test._mask, x._mask)
        # Test forward_fill w/ gaps, starting unmasked
        x[[3,4]] = (3,4)
        test = forward_fill(x, 3)
        assert_equal(test, [ 0, 0, 0, 3, 4, 5, 5, 5, 5, 5,
                            10,10,10,10,10,15,15,15,15,15,])
        assert_equal(test._mask,[0,0,0,0,0,0,1,1,1,1,
                                 0,1,1,1,1,0,1,1,1,1,])
        # Test forward_fill w/o gaps, starting masked
        x[[0,3,4]] = masked
        test = forward_fill(x)
        assert_equal(test, [ 0, 0, 0, 0, 0, 5, 5, 5, 5, 5,
                            10,10,10,10,10,15,15,15,15,15])
        assert_equal(test._mask, [1,1,1,1,1,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0,])
        # Test forward_fill w/ gaps, starting masked
        test = forward_fill(x,3)
        assert_equal(test, [ 0, 0, 0, 0, 0, 5, 5, 5, 5, 5,
                            10,10,10,10,10,15,15,15,15,15])
        assert_equal(test._mask, [1,1,1,1,1,0,1,1,1,1,
                                  0,1,1,1,1,0,1,1,1,1,])

    def test_forward_fill_non_contiguous(self):
        # this used to fail at some point in the past (based on a failure
        # I encountered with an older version), but seems to work now.

        x = ts.time_series(ma.arange(10), start_date=ts.now('m'))
        x = x.adjust_endpoints(start_date=x.start_date-3)
        x[-3:] = ts.masked
        test = forward_fill(x)

        # first three points masked
        assert_equal(test.mask[:3].sum(), 3)

        # last three points == 6
        assert_equal((test[-3:] == 6).sum(), 3)

###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    run_module_suite()
