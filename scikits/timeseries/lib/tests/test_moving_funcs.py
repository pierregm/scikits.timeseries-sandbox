# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for MaskedArray & subclassing.

:author: Pierre Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu & mattknox_ca_at_hotmail_dot_com
:version: $Id: test_filters.py 2819 2007-03-03 23:00:20Z pierregm $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: pierregm $)"
__revision__ = "$Revision: 2819 $"
__date__     = '$Date: 2007-03-03 18:00:20 -0500 (Sat, 03 Mar 2007) $'


import numpy as np

from numpy.testing import *
from numpy.ma.testutils import *

import numpy.ma as ma
from numpy.ma import MaskedArray, masked

import scikits.timeseries as ts
from scikits.timeseries.lib import moving_funcs as mf



class TestCMovAverage(TestCase):
    """
    Testing Centered Moving Average
    """

    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.data = np.arange(25)
        self.maskeddata = ma.array(self.data)
        self.maskeddata[10] = masked
    #
    def test_onregulararray(self):
        "Test cmov_average on a basic ndarray."
        data = self.data
        for width in [3, 5, 7]:
            k = (width-1)/2
            ravg = mf.cmov_average(data, width)
            self.failUnless(isinstance(ravg, MaskedArray))
            assert_equal(ravg, data)
            assert_equal(ravg._mask, [1]*k+[0]*(len(data)-2*k)+[1]*k)
    #
    def test_onmaskedarray(self):
        "Test cmov_average on a MaskedArray."
        data = self.maskeddata
        for width in [3, 5, 7]:
            k = (width-1)/2
            ravg = mf.cmov_average(data, width)
            self.failUnless(isinstance(ravg, MaskedArray))
            assert_equal(ravg, data)
            m = np.zeros(len(data), bool)
            m[:k] = m[-k:] = m[10-k:10+k+1] = True
            assert_equal(ravg._mask, m)
    #
    def test_ontimeseries(self):
        "Test cmov_average on a 1D TimeSeries."
        data = ts.time_series(self.maskeddata, start_date=ts.now('D'))
        for width in [3, 5, 7]:
            k = (width-1)/2
            ravg = mf.cmov_average(data, width)
            self.failUnless(isinstance(ravg, MaskedArray))
            assert_equal(ravg, data)
            m = np.zeros(len(data), bool)
            m[:k] = m[-k:] = m[10-k:10+k+1] = True
            assert_equal(ravg._mask, m)
            assert_equal(ravg._dates, data._dates)
    #
    def tests_onmultitimeseries(self):
        "Test cmov_average on a nD TimeSeries."
        maskeddata = MaskedArray(np.random.random(75).reshape(25, 3))
        maskeddata[10] = masked
        data = ts.time_series(maskeddata, start_date=ts.now('D'))
        for width in [3, 5, 7]:
            k = (width-1)/2
            ravg = mf.cmov_average(data, width)
            self.failUnless(isinstance(ravg, MaskedArray))
            #!!!: __getitem__ used to return a TimeSeries, now returns an array
#            assert_almost_equal(ravg[18]._series.squeeze(),
#                                data[18-k:18+k+1]._series.mean(0))
            assert_almost_equal(ravg[18].squeeze(),
                                data[18-k:18+k+1]._series.mean(0))
            m = np.zeros(data.shape, bool)
            m[:k] = m[-k:] = m[10-k:10+k+1] = True
            assert_equal(ravg._mask, m)
            assert_equal(ravg._dates, data._dates)



class TestMovFuncs(TestCase):
    """
    Test suite for moving functions
    """

    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.data = np.arange(25)
        self.maskeddata = MaskedArray(self.data)
        self.maskeddata[10] = masked
        self.func_pairs = [
            (mf.mov_average, ma.mean),
            (mf.mov_median, ma.median),
            ((lambda x, span : mf.mov_std(x, span)), ma.std)]
    #
    def test_onregulararray(self):
        "Tests the moving functions on regular ndarrays."
        data = self.data
        for mfunc, nfunc in self.func_pairs:
            for k in [3, 4, 5]:
                result = mfunc(data, k)
                self.failUnless(isinstance(result, MaskedArray))
                for x in range(len(data)-k+1):
                    assert_almost_equal(result[x+k-1], nfunc(data[x:x+k]))
                assert_equal(result._mask, [1]*(k-1)+[0]*(len(data)-k+1))

    #
    def test_onmaskedarray(self):
        "Tests the moving functions on MaskedArrays."
        data = self.maskeddata
        for Mfunc, Nfunc in self.func_pairs:
            for k in [3, 4, 5]:
                result = Mfunc(data, k)
                self.failUnless(isinstance(result, MaskedArray))
                for x in range(len(data)-k+1):
                    if result[x+k-1] is not ma.masked:
                        assert_almost_equal(result[x+k-1], Nfunc(data[x:x+k]))
                result_mask = np.array([1]*(k-1)+[0]*(len(data)-k+1))
                result_mask[10:10+k] = 1
                assert_equal(result._mask, result_mask)

    #
    def test_ontimeseries(self):
        "Test the moving functions on a 1D TimeSeries"
        data = ts.time_series(self.maskeddata, start_date=ts.now('D'))
        for Mfunc, Nfunc in self.func_pairs:
            for k in [3, 4, 5]:
                result = Mfunc(data, k)
                self.failUnless(isinstance(result, MaskedArray))
                for x in range(len(data)-k+1):
                    if result[x+k-1] is not ma.masked:
                        assert_almost_equal(
                                np.asarray(result[x+k-1]),
                                np.asarray(Nfunc(data[x:x+k])))
                result_mask = np.array([1]*(k-1)+[0]*(len(data)-k+1))
                result_mask[10:10+k] = 1
                assert_equal(result._mask, result_mask)
                assert_equal(result._dates, data._dates)

    def test_cov(self):
        "Test that  the covariance of series with itself is equal to variance"
        data = self.maskeddata
        for bias in [1, 0]:
            cov = mf.mov_cov(data, data, 3, bias=bias)
            var = mf.mov_var(data, 3, ddof=1-bias)
            assert_equal(cov, var)


    def test_on_list(self):
        "Test the moving functions on lists"
        data = self.data.tolist()
        for mfunc, nfunc in self.func_pairs:
            for k in [3, 4, 5]:
                result = mfunc(data, k)
                self.failUnless(isinstance(result, MaskedArray))
                for x in range(len(data)-k+1):
                    assert_almost_equal(result[x+k-1], nfunc(data[x:x+k]))
                assert_equal(result._mask, [1]*(k-1)+[0]*(len(data)-k+1))


    def test_w_nans(self):
        "Test the moving functions on series w/ NaNs"
        #
        data = np.random.rand(20)
        control = mf.mov_average(data, 3)
        # Test that we get the right result
        a = ma.array(data, mask=False)
        assert_almost_equal(control, mf.mov_average(a, 3))
        # Test with a masked element
        a.data[10] = 0
        a.mask[10] = True
        assert_almost_equal(control, mf.mov_average(a, 3))
        # Test w/ a masked element and an underlying nan
        a[10] = np.nan
        assert_almost_equal(control, mf.mov_average(a, 3))

    def test_masked_exclude(self):
        "Make sure that masked values are not allowed to cause FP overflow"
        # ensure masked values are not allowed to cause floating point
        # overflow problems
        ser_a = ma.array(range(150), dtype=np.float32)
        ser_a[:50] = ma.masked

        ser_b = ser_a.copy()
        # insert a very high value in the middle of the masked section of the
        # array. This should have no impact on the final result because it is
        # surround by masked values
        ser_b[45] = 5.0e34

        res_a = mf.mov_sum(ser_a, 60)
        res_b = mf.mov_sum(ser_b, 60)
        assert_almost_equal(res_a, res_b)


    def test_mov_average_expw(self):
        "Test mov_average_expw"
        ser_a = ma.array(range(150), dtype=np.float32)
        ser_b = ser_a.copy()
        # set extreme value to ensure masked values are not impacting result
        ser_b[0] = 5e34
        ser_b[0] = ma.masked
        #
        res_a = mf.mov_average_expw(ser_a, 10)
        res_b = mf.mov_average_expw(ser_b, 10, tol=1e-6)
        assert_almost_equal(res_a[-1], 144.5)
        assert_almost_equal(res_a, res_b)

        # with specified tolerance value, should be 69 masked values in res_b
        assert_equal(np.sum(res_b.mask), 69)


    def test_mov_average_expw_mask(self):
        "Make sure that mov_average_expw doesn't modify the initial mask"
        N = 256
        series = ts.time_series(np.random.rand(N),
                                start_date=ts.Date('D','2008-01-01'))
        series[96: 128] = ma.masked
        controlmask = np.zeros(N, dtype=bool)
        controlmask[96: 128] = True
        #
        test = mf.mov_average_expw(series, 16)
        assert_not_equal(test.mask, series.mask)
        assert_equal(series.mask, controlmask)



#------------------------------------------------------------------------------
if __name__ == "__main__":
    run_module_suite()
