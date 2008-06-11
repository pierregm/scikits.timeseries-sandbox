# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for MaskedArray & subclassing.

:author: Pierre Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu & mattknox_ca_at_hotmail_dot_com
:version: $Id: test_filters.py 2819 2007-03-03 23:00:20Z pierregm $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: pierregm $)"
__version__ = '1.0'
__revision__ = "$Revision: 2819 $"
__date__     = '$Date: 2007-03-03 18:00:20 -0500 (Sat, 03 Mar 2007) $'

#!!!: Switch to the import numpy as np, numpy.ma as ma convention
#!!!: Get rid of numpy.core.numeric
#!!!: Use ma.array instead of MaskedArray

import numpy as N
import numpy.core.numeric as numeric

from scipy.testing import *

from numpy.ma.testutils import *

import numpy.ma as MA
from numpy.ma import MaskedArray, masked

import scikits.timeseries as TS
from scikits.timeseries import time_series, now

from scikits.timeseries.lib import moving_funcs as MF

class TestCMovAverage(TestCase):

    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.data = numeric.arange(25)
        self.maskeddata = MaskedArray(self.data)
        self.maskeddata[10] = masked
    #
    def test_onregulararray(self):
        data = self.data
        for width in [3,5,7]:
            k = (width-1)/2
            ravg = MF.cmov_average(data,width)
            assert(isinstance(ravg, MaskedArray))
            assert_equal(ravg, data)
            assert_equal(ravg._mask, [1]*k+[0]*(len(data)-2*k)+[1]*k)
    #
    def test_onmaskedarray(self):
        data = self.maskeddata
        for width in [3,5,7]:
            k = (width-1)/2
            ravg = MF.cmov_average(data,width)
            assert(isinstance(ravg, MaskedArray))
            assert_equal(ravg, data)
            m = N.zeros(len(data), N.bool_)
            m[:k] = m[-k:] = m[10-k:10+k+1] = True
            assert_equal(ravg._mask, m)
    #
    def test_ontimeseries(self):
        data = time_series(self.maskeddata, start_date=now('D'))
        for width in [3,5,7]:
            k = (width-1)/2
            ravg = MF.cmov_average(data,width)
            assert(isinstance(ravg, MaskedArray))
            assert_equal(ravg, data)
            m = N.zeros(len(data), N.bool_)
            m[:k] = m[-k:] = m[10-k:10+k+1] = True
            assert_equal(ravg._mask, m)
            assert_equal(ravg._dates, data._dates)
    #
    def tests_onmultitimeseries(self):
        maskeddata = MaskedArray(N.random.random(75).reshape(25,3))
        maskeddata[10] = masked
        data = time_series(maskeddata, start_date=now('D'))
        for width in [3,5,7]:
            k = (width-1)/2
            ravg = MF.cmov_average(data,width)
            assert(isinstance(ravg, MaskedArray))
            assert_almost_equal(ravg[18]._series.squeeze(), 
                                data[18-k:18+k+1]._series.mean(0))
            m = N.zeros(data.shape, N.bool_)
            m[:k] = m[-k:] = m[10-k:10+k+1] = True
            assert_equal(ravg._mask, m)
            assert_equal(ravg._dates, data._dates)



class TestMovFuncs(TestCase):

    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.data = numeric.arange(25)
        self.maskeddata = MaskedArray(self.data)
        self.maskeddata[10] = masked
        self.func_pairs = [
            (MF.mov_average, MA.mean),
            (MF.mov_median, MA.median),
            ((lambda x, span : MF.mov_std(x, span)), MA.std)]
    #
    def test_onregulararray(self):
        data = self.data
        for Mfunc, Nfunc in self.func_pairs:
            for k in [3,4,5]:
                result = Mfunc(data, k)
                assert(isinstance(result, MaskedArray))
                for x in range(len(data)-k+1):
                    assert_almost_equal(result[x+k-1], Nfunc(data[x:x+k]))
                assert_equal(result._mask, [1]*(k-1)+[0]*(len(data)-k+1))

    #
    def test_onmaskedarray(self):
        data = self.maskeddata

        for Mfunc, Nfunc in self.func_pairs:
            for k in [3,4,5]:
                result = Mfunc(data, k)
                assert(isinstance(result, MaskedArray))
                for x in range(len(data)-k+1):
                    if result[x+k-1] is not MA.masked:
                        assert_almost_equal(result[x+k-1], Nfunc(data[x:x+k]))
                result_mask = N.array([1]*(k-1)+[0]*(len(data)-k+1))
                result_mask[10:10+k] = 1
                assert_equal(result._mask, result_mask)

    #
    def test_ontimeseries(self):

        data = time_series(self.maskeddata, start_date=now('D'))

        for Mfunc, Nfunc in self.func_pairs:
            for k in [3,4,5]:
                result = Mfunc(data, k)
                assert(isinstance(result, MaskedArray))
                for x in range(len(data)-k+1):
                    if result[x+k-1] is not TS.tsmasked:
                        assert_almost_equal(
                                N.asarray(result[x+k-1]),
                                N.asarray(Nfunc(data[x:x+k])))
                result_mask = N.array([1]*(k-1)+[0]*(len(data)-k+1))
                result_mask[10:10+k] = 1
                assert_equal(result._mask, result_mask)
                assert_equal(result._dates, data._dates)

    def test_cov(self):
        # test that covariance of series with itself is equal to variance
        data = self.maskeddata
        for bias in [1, 0]:
            cov = MF.mov_cov(data, data, 3, bias=bias)
            var = MF.mov_var(data, 3, ddof=1-bias)
            assert_equal(cov, var)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    nose.run(argv=['', __file__])
