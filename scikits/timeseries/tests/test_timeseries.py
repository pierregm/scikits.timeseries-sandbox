# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for MaskedArray.
Adapted from the original test_ma by Pierre Gerard-Marchant

:author: Pierre Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu & mattknox_ca_at_hotmail_dot_com
:version: $Id: test_timeseries.py 3836 2008-01-15 13:09:03Z matthew.brett@gmail.com $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: matthew.brett@gmail.com $)"
__revision__ = "$Revision: 3836 $"
__date__     = '$Date: 2008-01-15 08:09:03 -0500 (Tue, 15 Jan 2008) $'

import numpy as np
from numpy import bool_, complex_, float_, int_, object_
from numpy.testing import *

import numpy.ma as ma
from numpy.ma import MaskedArray, masked, nomask
from numpy.ma.testutils import *

import scikits.timeseries as ts

from scikits.timeseries import \
    TimeSeries, TimeSeriesError, TimeSeriesCompatibilityError, \
    tseries, Date, date_array, now, time_series, \
    adjust_endpoints, align_series, align_with, \
    concatenate, fill_missing_dates, find_duplicated_dates, \
    remove_duplicated_dates, split, stack

get_varshape = tseries.get_varshape
_timeseriescompat_multiple = tseries._timeseriescompat_multiple


#------------------------------------------------------------------------------

class TestCreation(TestCase):
    "Base test class for MaskedArrays."

    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        dlist = ['2007-01-%02i' % i for i in range(1, 16)]
        dates = date_array(dlist, freq='D')
        data = ma.array(np.arange(15), mask=[1, 0, 0, 0, 0]*3)
        self.d = (dlist, dates, data)

    def test_fromlist (self):
        "Test the creation of a TimeSeries w/ a list of dates as input dates."
        (dlist, dates, data) = self.d
        series = time_series(data, dlist, freq='D')
        self.failUnless(isinstance(series, TimeSeries))
        assert_equal(series.mask, [1, 0, 0, 0, 0]*3)
        assert_equal(series.series, data)
        assert_equal(series.dates, dates)
        assert_equal(series.freqstr, 'D')

    def test_fromrange (self):
        "Test the creation of a TimeSeries w/ a starting date."
        (dlist, dates, data) = self.d
        series = time_series(data, start_date=dates[0])
        self.failUnless(isinstance(series, TimeSeries))
        assert_equal(series.mask, [1, 0, 0, 0, 0]*3)
        assert_equal(series.series, data)
        assert_equal(series.dates, dates)
        assert_equal(series.freqstr, 'D')

    def test_fromseries (self):
        "Test the creation of a TimeSeries w/ a time series as input data."
        (dlist, dates, data) = self.d
        series = time_series(data, dlist, freq='D')
        dates = dates+15
        series = time_series(series, dates)
        self.failUnless(isinstance(series, TimeSeries))
        assert_equal(series.mask, [1, 0, 0, 0, 0]*3)
        assert_equal(series.series, data)
        assert_equal(series.dates, dates)
        assert_equal(series.freqstr, 'D')


    def test_fromdatearray(self):
        "Tests the creation of a series with a DateArray as input data."
        (_, dates, _) = self.d
        data = dates
        #
        series = time_series(data, dates)
        self.failUnless(isinstance(series, TimeSeries))
        assert_equal(series.dates, dates)
        assert_equal(series.data, data)
        assert_equal(series.freqstr, 'D')
        #
        series[5] = masked
        # ensure that series can be represented by a string after masking a value
        # (there was a bug before that prevented this from working when using a
        # DateArray for the data)
        strrep = str(series)


    def test_datafromlist(self):
        "Test the creation of a series w/ a list as input data."
        (_, dates, _) = self.d
        data = list(range(15))
        series = time_series(data, dates)
        assert_equal(series._data.size, 15)


    def test_unsorted(self):
        "Tests that the data are properly sorted along the dates."
        dlist = ['2007-01-%02i' % i for i in (3, 2, 1)]
        data = [10, 20, 30]
        series = time_series(data, dlist, freq='D')
        assert_equal(series.data, [30, 20, 10])
        #
        dates = date_array(dlist, freq='D')
        series = TimeSeries(data, dates)
        assert_equal(series.data, [30, 20, 10])
        #
        series = time_series(data, dlist, freq='D', mask=[1, 0, 0])
        assert_equal(series.mask, [0, 0, 1])
        #
        data = ma.array([10, 20, 30],mask=[1, 0, 0])
        series = time_series(data, dlist, freq='D')
        assert_equal(series._mask,[0, 0, 1])

    def test_unsorted_w_datearray(self):
        "Tests that the data are properly sorted along the dates."
        dlist = ['2007-01-%02i' % i for i in (3, 2, 1)]
        data = [10, 20, 30]
        dates = date_array(dlist, freq='D')
        self.failUnless(dates._unsorted is not None)
        #
        series = time_series(data, dates=dates)
        assert_equal(series.data, [30, 20, 10])
        self.failUnless(dates._unsorted is not None)
        self.failUnless(series.dates._unsorted is None)
        #
        series = time_series(data, dates=dates)
        assert_equal(series.data, [30, 20, 10])
        self.failUnless(series.dates._unsorted is None)


    def test_setdates(self):
        "Tests setting the dates of a series."
        (dlist, dates, data) = self.d
        reference = time_series(data, dates=dates)
        # Set with a DateArray: that should work
        test_series = data.view(TimeSeries)
        test_series.dates = dates
        assert_equal(test_series.dates, reference.dates)

    def test_setdates_asndarray(self):
        "Tests setting the dates as a ndarray."
        (dlist, dates, data) = self.d
        test_series = data.view(TimeSeries)
        # Set with a ndarray: that shouldn't work
        test_dates = np.array(dates, copy=False, subok=False)
        try:
            test_series._dates = test_dates
        except TypeError:
            pass
        else:
            err_msg = "Dates shouldn't be set as basic ndarrays."
            raise TimeSeriesError(err_msg)

    def test_setdates_asdate(self):
        "Tests setting the dates as a Date"
        (dlist, dates, data) = self.d
        series = data.view(TimeSeries)
        try:
            series.dates = ts.now('D')
        except TypeError:
            pass
        else:
            err_msg = "Dates shouldn't be set as a Date objects."
            raise TimeSeriesError(err_msg)

    def test_setdates_with_incompatible_size(self):
        "Tests setting the dates w/ a DateArray of incompatible size"
        (dlist, dates, data) = self.d
        series = data.view(TimeSeries)
        try:
            series.dates = dates[:len(dates)//2]
        except ts.TimeSeriesCompatibilityError:
            pass
        else:
            err_msg = "Dates size should match the input."
            raise TimeSeriesError(err_msg)


    def test_setdates_with_autoreshape(self):
        "Tests the automatic reshaping of dates."
        (dlist, dates, data) = self.d
        reference = time_series(data, dates=dates)
        test_series = data.view(TimeSeries)
        # Set with a datearray w/ a different size than expected: should fail
        test_dates = dates[:-1]
        try:
            test_series.dates = test_dates
        except TimeSeriesCompatibilityError:
            pass
        else:
            err_msg = "Dates should have a size compatible with data"
            raise TimeSeriesError(err_msg)
        # Set w/ a date of a different shape: should work, but the shape changes
        test_dates = dates.reshape(-1, 1)
        test_series._dates = test_dates
        assert_equal(test_series.dates, reference.dates)
        assert_equal(test_series.dates.shape, test_series.shape)

        test_dates = np.array(dates, copy=False, subok=True, ndmin=2)
        test_series._dates = test_dates
        assert_equal(test_series.dates, reference.dates)
        assert_equal(test_series.dates.shape, test_series.shape)

    def test_setdates_unsorted_basic(self):
        "Test automatic sorting when setting dates - 1D case."
        dates = date_array([ts.Date('D',
                                    '2001-01-%02i' % _) for _ in (4, 3, 2, 1)])
        a = np.array((4, 3, 2, 1), dtype=float)
        series = a.view(ts.TimeSeries)
        assert_equal(series.dates, [])
        assert_equal(series, (4, 3, 2, 1))
        #
        series._dates = dates
        series.sort_chronologically()
        assert_equal(series, (1, 2, 3, 4))

    def test_setdates_unsorted_reshaped(self):
        "Test automatic sorting when setting dates - 1D case reshaped to nD."
        dates = date_array([ts.Date('D',
                                    '2001-01-%02i' % _) for _ in (4, 3, 2, 1)])
        a = np.array([[4., 3.],[2., 1.]], dtype=float)
        series = a.view(TimeSeries)
        series._dates = dates
        series.sort_chronologically()
        assert_equal(series, [[1., 2.], [3., 4.]])

    def test_setdates_unsorted_2D(self):
        "Test automatic sorting when setting dates - 1D case reshaped to nD."
        dates = date_array([ts.Date('D',
                                    '2001-01-%02i' % _) for _ in (4, 3, 2, 1)])
        a = np.arange(12).reshape(4, 3)
        series = a.view(TimeSeries)
        series._dates = dates
        series.sort_chronologically()
        assert_equal(series, [[ 9., 10., 11.],
                              [ 6.,  7.,  8.],
                              [ 3.,  4.,  5.],
                              [ 0.,  1.,  2.]])


    def test_copy(self):
        "Tests the creation of a timeseries with copy=True"
        dlist = ['2007-01-%02i' % i for i in range(1, 16)]
        dates = date_array(dlist, freq='D')
        data = ma.array(np.arange(15), mask=[1, 0, 0, 0, 0]*3)
        series = time_series(data, dates)
        assert_equal(series.dates.ctypes.data, dates.ctypes.data)
        assert_equal(series.data.ctypes.data, data.data.ctypes.data)
        assert_equal(series.mask.ctypes.data, data.mask.ctypes.data)
        #
        series = time_series(data, dates, copy=True)
        assert_not_equal(series.dates.ctypes.data, dates.ctypes.data)
        assert_not_equal(series.data.ctypes.data, data.data.ctypes.data)
        assert_not_equal(series.mask.ctypes.data, data.mask.ctypes.data)


    def test_using_length(self):
        "Test using the `length` parameter of time_series."
        start = ts.Date('M', '1955-01')
        data = np.random.uniform(0, 1, 50*12).reshape(50, 12)
        # Default : the dates should be (50,)
        series = ts.time_series(data, start_date=start)
        assert_equal(series.shape, (50, 12))
        assert_equal(series.dates.shape, (50,))
        assert_equal(series.varshape, (12,))
        # Forcing dates to be 2D
        series = ts.time_series(data, start_date=start, length=600)
        assert_equal(series.shape, (50, 12))
        assert_equal(series.dates.shape, (50, 12))
        assert_equal(series.varshape, ())
        # Forcing dates to 1D
        series = ts.time_series(data, start_date=start, length=50)
        assert_equal(series.shape, (50, 12))
        assert_equal(series.dates.shape, (50,))
        assert_equal(series.varshape, (12,))
        # Make sure we raise an exception if something goes wrong....
        try:
            series = ts.time_series(data, start_date=start, length=100)
        except ts.TimeSeriesCompatibilityError:
            pass
        else:
            errmsg = "The should not be dates/data compatibility in this case."
            raise TimeSeriesCompatibilityError(errmsg)


    def test_varshape(self):
        "Test some corner case of varshape"
        test = ts.time_series(np.ones((10, 2)), start_date=ts.now('d'))
        assert_equal(test.varshape, (2,))
        #
        test = ts.time_series(np.ones((10, 1)), start_date=ts.now('d'))
        assert_equal(test.varshape, (1,))
        #
        test = ts.time_series(np.ones((10,)), start_date=ts.now('d'))
        assert_equal(test.varshape, ())


#------------------------------------------------------------------------------

class TestArithmetics(TestCase):
    "Some basic arithmetic tests"

    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        dlist = ['2007-01-%02i' % i for i in range(1, 16)]
        dates = date_array(dlist, freq='D')
        data = ma.array(np.arange(15), mask=[1, 0, 0, 0, 0]*3)
        self.d = (time_series(data, dlist, freq='D'), data)

    def test_intfloat(self):
        "Test arithmetic timeseries/integers"
        (series, data) =self.d
        #
        nseries = series+1
        self.failUnless(isinstance(nseries, TimeSeries))
        assert_equal(nseries.mask, [1, 0, 0, 0, 0]*3)
        assert_equal(nseries.series, data+1)
        assert_equal(nseries.dates, series.dates)
        #
        nseries = series-1
        self.failUnless(isinstance(nseries, TimeSeries))
        assert_equal(nseries.mask, [1,0,0,0,0]*3)
        assert_equal(nseries.series, data-1)
        assert_equal(nseries.dates, series.dates)
        #
        nseries = series*1
        self.failUnless(isinstance(nseries, TimeSeries))
        assert_equal(nseries.mask, [1, 0, 0, 0, 0]*3)
        assert_equal(nseries.series, data*1)
        assert_equal(nseries.dates, series.dates)
        #
        nseries = series/1.
        self.failUnless(isinstance(nseries, TimeSeries))
        assert_equal(nseries.mask, [1, 0, 0, 0, 0]*3)
        assert_equal(nseries.series, data/1.)
        assert_equal(nseries.dates, series.dates)

    def test_intfloat_inplace(self):
        "Test int/float arithmetics in place."
        (series, data) =self.d
        nseries = series.astype(float_)
        idini = id(nseries)
        data = data.astype(float_)
        #
        nseries += 1.
        self.failUnless(isinstance(nseries, TimeSeries))
        assert_equal(nseries.mask, [1, 0, 0, 0, 0]*3)
        assert_equal(nseries.series, data + 1.)
        assert_equal(nseries.dates, series.dates)
        assert_equal(id(nseries),idini)
        #
        nseries -= 1.
        self.failUnless(isinstance(nseries, TimeSeries))
        assert_equal(nseries.mask, [1, 0, 0, 0, 0]*3)
        assert_equal(nseries.series, data)
        assert_equal(nseries.dates, series.dates)
        assert_equal(id(nseries),idini)
        #
        nseries *= 2.
        self.failUnless(isinstance(nseries, TimeSeries))
        assert_equal(nseries.mask, [1,0,0,0,0]*3)
        assert_equal(nseries.series, data*2.)
        assert_equal(nseries.dates, series.dates)
        assert_equal(id(nseries),idini)
        #
        nseries /= 2.
        self.failUnless(isinstance(nseries, TimeSeries))
        assert_equal(nseries.mask, [1, 0, 0, 0, 0]*3)
        assert_equal(nseries.series, data)
        assert_equal(nseries.dates, series.dates)
        assert_equal(id(nseries), idini)

    def test_updatemask(self):
        "Checks modification of mask."
        (series, data) =self.d
        assert_equal(series.mask, [1, 0, 0, 0, 0]*3)
        series.mask = nomask
        self.failUnless(not series.mask.any())
        self.failUnless(not series.series.mask.any())
        #series._series.mask = [1,0,0]*5
        series.mask = [1, 0, 0]*5
        assert_equal(series.mask, [1, 0, 0]*5)
        assert_equal(series.series.mask, [1, 0, 0]*5)
        series[2] = masked
        assert_equal(series.mask, [1, 0, 1] + [1, 0, 0]*4)
        assert_equal(series.series.mask, [1, 0, 1] + [1, 0, 0]*4)

    def test_ismasked(self):
        "Checks checks on masked"
        (series, data) =self.d
        self.failUnless(series._series[0] is masked)
        #!!!:... and of course, masked doesn't have a _series attribute
#        self.failUnless(series[0]._series is masked)


    def test_incompatible_dates(self):
        """
    Test operations on two series with same dimensions but incompatible dates
        """
        (series, data) =self.d
        a, b = series[1:], series[:-1]
        result = a + b
        self.failUnless(not isinstance(result, TimeSeries))
        assert_equal(result.ndim, a.ndim)
        assert_equal(result.size, a.size)


#------------------------------------------------------------------------------

class TestGetitem(TestCase):
    "Some getitem tests"

    def setUp(self):
        dates = date_array(['2007-01-%02i' % i for i in range(1, 16)], freq='D')
        data1D = ma.array(np.arange(15), mask=[1, 0, 0, 0, 0]*3, dtype=float_)
        data3V = ma.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]],
                          mask=[[1, 0, 0,], [0, 0, 0], [0, 0, 1]])
        data2D = ma.array(np.random.rand(60).reshape(3, 4, 5))
        for i in range(3):
            data2D[i,i,i] = masked
        #.........................
        series1D = time_series(data1D, dates, freq='D')
        series3V = time_series(data3V, dates[:len(data3V)], freq='D')
        series2D = time_series(data2D, dates[:len(data2D)], freq='D')
        self.info = locals()
        del(self.info['i'])
        self.__dict__.update(self.info)
        return


    def test_with_integers(self):
        # 1D series ..............
        (series1D, data1D) = (self.series1D, self.data1D)
        self.failUnless(series1D[0] is masked)
        test = series1D[-1]
        assert_equal(test, data1D[-1])
        self.failUnless(not isinstance(test, TimeSeries))
        # nV series ..............
        (series3V, data3V) = (self.series3V, self.data3V)
        test = series3V[-1]
        assert_equal(test, data3V[-1])
        assert_equal(test.mask, [0, 0, 1])
        self.failUnless(not isinstance(test, TimeSeries))
        # 2D series ..............
        (series2D, data2D) = (self.series2D, self.data2D)
        test = series2D[-1]
        assert_equal(test, data2D[-1].squeeze())
        self.failUnless(not isinstance(test, TimeSeries))


    def test_with_slices(self):
        "Tests __getitem__ w/ slices."
        def _wslice(series, data, dates):
            test = series[1:2]
            self.failUnless(isinstance(test, TimeSeries))
            assert_equal(test._varshape, series._varshape)
            assert_equal(test.series, data[1:2])
            assert_equal(test.dates, dates[1:2])
            assert_equal(test.mask, data.mask[1:2])
            assert_equal(test.freq, dates.freq)
            #
            test = series[:3]
            self.failUnless(isinstance(test, TimeSeries))
            test_series = test.series
            assert_equal(test_series.data, data[:3].data)
            assert_equal(test_series.mask, data[:3].mask)
            assert_equal(test.dates, dates[:3])
        #.....
        dates = self.dates
        (series1D, data1D) = (self.series1D, self.data1D)
        _wslice(series1D, data1D, dates)
        (series3V, data3V) = (self.series3V, self.data3V)
        _wslice(series3V, data3V, dates)
        (series2D, data2D) = (self.series2D, self.data2D)
        _wslice(series2D, data2D, dates)


    def test_with_slices_on_nD(self):
        (series3V, data3V) = (self.series3V, self.data3V)
        #
        test = series3V[0, :]
        self.failUnless(not isinstance(test, TimeSeries))
        assert_equal(test, data3V[0, :])
        assert_equal(test.mask, data3V[0, :].mask)
        #
        test = series3V[:,0]
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal(test, data3V[:,0])
        assert_equal(test.mask, data3V[:,0].mask)
        assert_equal(test._varshape, ())
        assert_equal(test.dates, series3V.dates)
        #
        (series2D, data2D) = (self.series2D, self.data2D)
        test = series2D[0]
        self.failUnless(not isinstance(test, TimeSeries))
        assert_equal(test.shape, (4, 5))
        assert_equal(test, data2D[0])
        #
        test = series2D[:, :, 0]
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal(test, series2D.data[:, :, 0])
        assert_equal(test.dates, series2D.dates)


    def test_with_list(self):
        "Tests __getitem__ w/ list."
        def _wlist(series, data, dates):
            test = series[[0, 1, -1]]
            control = data[[0, 1, -1]]
            self.failUnless(isinstance(test, TimeSeries))
            assert_equal(test.series, control)
            assert_equal(test.mask, control.mask)
            assert_equal(test.dates, dates[[0, 1, -1]])
        #.....
        dates = self.dates
        (series1D, data1D) = (self.series1D, self.data1D)
        _wlist(series1D, data1D, dates)
        (series3V, data3V) = (self.series3V, self.data3V)
        _wlist(series3V, data3V, dates[:3])
        (series2D, data2D) = (self.series2D, self.data2D)
        _wlist(series2D, data2D, dates[:3])


    def test_with_dates(self):
        "Tests __getitem__ w/ dates."
        def _wdates(series, data, dates):
            # Single date
            test = series[dates[0]]
            assert_equal(test, data[0])
            assert_equal(test.mask, data[0].mask)
            self.failUnless(not isinstance(test, TimeSeries))
            # Multiple dates as a date_array
            test = series[dates[[0, -1]]]
            assert_equal(test, data[[0, -1]])
            self.failUnless(isinstance(test, TimeSeries))
            assert_equal(test.dates, dates[[0, -1]])
            # Multiple dates as a list
            test = series[[dates[0], dates[-1]]]
            assert_equal(test, data[[0, -1]])
            self.failUnless(isinstance(test, TimeSeries))
            # Multiple dates as a slice
            dslice = slice(dates[1], None, None)
            test = series[dslice]
            assert_equal(test, data[1:])
            self.failUnless(isinstance(test, TimeSeries))
        #.....
        dates = self.dates
        (series1D, data1D) = (self.series1D, self.data1D)
        _wdates(series1D, data1D, dates)
        (series3V, data3V) = (self.series3V, self.data3V)
        _wdates(series3V, data3V, dates[:3])
        (series2D, data2D) = (self.series2D, self.data2D)
        _wdates(series2D, data2D, dates[:3])


    def test_slicing_with_dates(self):
        "Tests __getitem__ w/ date based slices"
        def _testslice(series):
            sd, ed = series.start_date, series.end_date
            # full range of series
            assert_equal(series, series[sd:ed+1])
            # exclude first and last point of series
            assert_equal(series[1:-1], series[sd+1:ed])
            # slice with dates beyond the start and end dates
            assert_equal(series, series[sd-10:ed+10])
            # slice with dates before the series start date
            assert_equal(series[0:0], series[sd-10:sd-5])
        #.....
        series = self.series1D
        _testslice(series)
        # Now try slicing on a series with missing dates
        series = series[::2]
        _testslice(series)


    def test_with_dates_as_str(self):
        "Test using a string corresponding to a date as index."
        def _wdates(series, data):
            date = self.dates[0].strfmt("%Y-%m-%d")
            # Single date
            test = series[date]
            assert_equal(test, data[0])
            assert_equal(test.mask, data[0].mask)
            self.failUnless(not isinstance(test, TimeSeries))
        #.....
        (series1D, data1D) = (self.series1D, self.data1D)
        _wdates(series1D, data1D)
        (series3V, data3V) = (self.series3V, self.data3V)
        _wdates(series3V, data3V)
        (series2D, data2D) = (self.series2D, self.data2D)
        _wdates(series2D, data2D)
        #
        test = series1D[['2007-01-01', '2007-01-15']]
        control = series1D[[0, -1]]
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.dates, control.dates)


    def test_on1D_reshaped(self):
        trick = time_series(self.data1D.reshape(3,5),
                            dates=self.dates.reshape(3,5), freq='D')
        test = trick[0, 0]
        self.failUnless(not isinstance(test, TimeSeries))
        self.failUnless(test is masked)
        #
        test = trick[-1, -1]
        self.failUnless(not isinstance(test, TimeSeries))
        assert_equal(test, 14)
        #
        test = trick[0]
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal(test._varshape, ())
        assert_equal(test, trick.series[0])
        assert_equal(test.dates, trick.dates[0])


    def test_wtimeseries(self):
        "Tests getitem w/ TimeSeries as index"
        series1D = self.series1D
        # Testing a basic condition on data
        cond = (series1D < 8).filled(False)
        dseries = series1D[cond]
        assert_equal(dseries.data, [1, 2, 3, 4, 6, 7])
        assert_equal(dseries.dates, series1D.dates[[1, 2, 3, 4, 6, 7]])
        assert_equal(dseries.mask, nomask)
        # Testing a basic condition on dates
        series1D[series1D.dates < Date('D', string='2007-01-06')] = masked
        assert_equal(series1D[:5].series.mask, [1, 1, 1, 1, 1])


    def test_on2d(self):
        "Tests getitem on a 2D series"
        (a,b,d) = ([1,2,3],[3,2,1], date_array(now('M'), length=3))
        ser_x = time_series(np.column_stack((a, b)), dates=d)
        assert_equal(ser_x[0,0], time_series(a[0], d[0]))
        assert_equal(ser_x[0,:], (a[0], b[0]))
        assert_equal(ser_x[:,0], time_series(a, d))
        assert_equal(ser_x[:,:], ser_x)


    def test_slicing_and_keeping_additional_attributes(self):
        series1D = self.series1D
        series1D.fill_value = -9999
        series1D._basedict['info'] = '???'
        piece = series1D[:5]
        assert_equal(piece._fill_value, -9999)
        assert_equal(piece[:5]._basedict['info'], '???')


#------------------------------------------------------------------------------

class TestSetItem(TestCase):
    #
    def setUp(self):
        dlist = ['2007-01-%02i' % i for i in range(1, 6)]
        dates = date_array(dlist, freq='D')
        data = ma.array(np.arange(5), mask=[1, 0, 0, 0, 0], dtype=float)
        self.series = time_series(data, dates)
        self.dates = dates
    #
    def test_with_integers(self):
        "Tests setitem with integers"
        series = self.series
        series[0] = 1
        assert_equal(series.data, [1, 1, 2, 3, 4])
        assert_equal(series.mask, [0, 0, 0, 0, 0])
        series[0] = masked
        assert_equal(series.data, [1, 1, 2, 3, 4])
        assert_equal(series.mask, [1, 0, 0, 0, 0])
        try:
            series[10] = -999
        except IndexError:
            pass
    #
    def test_with_dates(self):
        "Test setitem w/ dates"
        (series, dates) = (self.series, self.dates)
        #
        last_date = dates[-1]
        series[last_date] = 5
        assert_equal(series.data, [0, 1, 2, 3, 5])
        assert_equal(series.mask, [1, 0, 0, 0, 0])
        #
        last_date += 10
        try:
            series[last_date] = -999
        except IndexError:
            pass
        # With dates as string
        series['2007-01-01'] = 5
        assert_equal(series.data, [5, 1, 2, 3, 5])
        assert_equal(series.mask, [0, 0, 0, 0, 0])

        # test for bug fixed in r1203
        x, y = ts.now('b'), ts.now('b')+1
        a = ts.time_series([1], start_date=x)
        b = ts.time_series([4, 5], start_date=x)
        b[x:y] = a[x:y]
        assert_equal(b[0], 1)


    def test_with_datearray(self):
        "Test setitem w/ a date_array"
        (series, dates) = (self.series, self.dates)
        # Test with date array
        series[dates[[0, -1]]] = 0
        assert_equal(series.data, [0, 1, 2, 3, 0])
        assert_equal(series.mask, [0, 0, 0, 0, 0])
        # Test with date as list ofstring
        series[['2007-01-01','2007-01-02']] = 10
        assert_equal(series.data, [10, 10, 2, 3, 0])
        assert_equal(series.mask, [ 0,  0, 0, 0, 0])


#------------------------------------------------------------------------------

class TestTimeSeriesMethods(TestCase):

    def setUp(self):
        dates = date_array(['2007-01-%02i' % i for i in (1, 2, 3)], freq='D')
        data1D = ma.array([1, 2, 3], mask=[1, 0, 0,])
        data3V = ma.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]],
                          mask=[[1, 0, 0,],[0, 0, 0], [0, 0, 1]])
        data2D = np.random.rand(60).reshape(3,4,5)
        series1D = time_series(data1D, dates, freq='D')
        series3V = time_series(data3V, dates, freq='D')
        series2D = time_series(data2D, dates, freq='D')
        self.info = locals()
        del(self.info['i'])
        return


    def test_toflex_1D(self):
        "Test conversion to records on 1D series"
        series = ts.time_series([1, 2, 3],
                                start_date=ts.Date('M', '2001-01-01'),
                                mask=[0, 1, 0])
        ndtype = [('_dates', int), ('_data', int), ('_mask', bool)]
        control = np.array([(24001, 1, False),
                            (24002, 2, True),
                            (24003, 3, False)], dtype=ndtype)
        test = series.toflex()
        assert_equal(test, control)

    def test_toflex_2D(self):
        "Test torecords on 2D series"
        series = ts.time_series([[1, 1], [2, 2], [3, 3]],
                                start_date=ts.Date('M', '2001-01-01'),
                                mask=[[0, 1], [0, 0], [1, 0]])
        ndtype = [('_dates', int),
                  ('_data', (int, (2,))),
                  ('_mask', (bool, (2,)))]
        control = np.array([(24001, [1, 1], [False, True]),
                            (24002, [2, 2], [False, False]),
                            (24003, [3, 3], [True, False])], dtype=ndtype)
        test = series.torecords()
        assert_equal_records(test, control)

    def test_toflex_structured(self):
        "Test torecords on structured array"
        series = ts.time_series([(1, 1), (2, 2), (3, 3)],
                                start_date=ts.Date('M', '2001-01-01'),
                                mask=[(0, 1), (0, 0), (1, 0)],
                                dtype=[('a', int), ('b', float)])
        ndtype = [('_dates', int),
                  ('_data', [('a', int), ('b', float)]),
                  ('_mask', [('a', bool), ('b', bool)])]
        control = np.array([(24001, (1, 1), (False, True)),
                            (24002, (2, 2), (False, False)),
                            (24003, (3, 3), (True, False))], dtype=ndtype)
        test = series.torecords()
        assert_equal_records(test, control)


    def test_reshape_1D(self):
        "Test reshape on data w/ 1 variables"
        start = ts.Date('M', '2001-01')
        series = ts.time_series([1, 2, 3, 4], mask=[0, 0, 1, 0],
                                start_date=start)
        test = series.reshape(2, 2)
        control = ts.time_series([[1, 2], [3, 4]], mask=[[0, 0], [1, 0]],
                                 dates=ts.date_array(start_date=start,
                                                     length=4).reshape(2,2))
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.dates, control.dates)
        assert_equal(test.varshape, series.varshape)
        #
        test = series.copy()
        test.shape = (2, 2)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.dates, control.dates)
        assert_equal(test.varshape, series.varshape)


    def test_reshape_1V(self):
        "Test reshape on series w/ 2 variables"
        series = ts.time_series([[1, 2], [3, 4]],
                                mask=[[0, 0], [1, 0]],
                                start_date=ts.Date('M', '2001-01'))
        test = series.reshape((-1, 1))
        control = ts.time_series([[[1, 2]], [[3, 4]]],
                                 mask=[[[0, 0]], [[1, 0]]],
                                 dates=series.dates.reshape((-1, 1)))
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.dates, control.dates)
        assert_equal(test.varshape, control.varshape)
        #
        test = series.reshape((1, -1, 1))
        control = ts.time_series([[[[1, 2]], [[3, 4]]]],
                                 mask= [[[[0, 0]], [[1, 0]]]],
                                 dates=series.dates.reshape((1, -1, 1)))
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.dates, control.dates)


    def test_reshaping_1D(self):
        "Tests the reshaping of a 1D series."
        series1D = self.info['series1D']
        newshape = (3,1)
        test1D = series1D.reshape(newshape)
        assert_equal(test1D.shape, newshape)
        assert_equal(test1D.series.shape, newshape)
        assert_equal(test1D.dates.shape, newshape)
        assert_equal(test1D.varshape, series1D.varshape)
        # Make sure we haven't propagated the new shape
        self.failUnless(test1D.shape != series1D.shape)
        self.failUnless(test1D.dates.shape != series1D.dates.shape)
        # Using .shape
        test1D = series1D.copy()
        test1D.shape = newshape
        assert_equal(test1D.shape, newshape)
        assert_equal(test1D.series.shape, newshape)
        assert_equal(test1D.dates.shape, newshape)
        self.failUnless(series1D.dates.shape != newshape)
        assert_equal(test1D.varshape, series1D.varshape)
        # Using multiple args
        test1D = series1D.reshape(*newshape)
        assert_equal(test1D.shape, newshape)
        assert_equal(test1D.varshape, series1D.varshape)


    def test_reshape_batch(self):
        "Test a succession of reshape"
        a = ts.time_series([1,2,3], start_date=ts.now('D'))
        test = a.reshape(-1, 1)
        assert_equal(test.shape, (3, 1))
        assert_equal(test.varshape, ())
        test = a.reshape(-1, 1).reshape(-1)
        assert_equal(test.shape, (3,))
        assert_equal(test.varshape, ())


    def test_reshaping_2D(self):
        "Tests the reshaping of a nV/nD series."
        series3V = self.info['series3V']
        newshape = (1, 3, 3)
        try:
            test3V = series3V.reshape(newshape)
            assert_equal(test3V.shape, newshape)
            assert_equal(test3V.series.shape, newshape)
            assert_equal(test3V.dates.shape, (1, 3))
            assert_equal(test3V.varshape, series3V.varshape)
        except NotImplementedError:
            pass
        else:
            raise Exception("Reshaping nV/nD series should be implemented!")
        # Using .shape
        try:
            test3V = series3V.copy()
            test3V.shape = newshape
            assert_equal(test3V.shape, newshape)
            assert_equal(test3V.series.shape, newshape)
            assert_equal(test3V.dates.shape, (1, 3))
            assert_equal(test3V.varshape, series3V.varshape)
        except NotImplementedError:
            pass
        else:
            raise Exception("Reshaping nV/nD series should be implemented!")


    def test_ravel_1D(self):
        "Test .ravel on 1D data"
        series = ts.time_series([1, 2, 3, 4],
                                mask=[0, 0, 1, 0],
                                start_date=ts.Date('M','2009-01'))
        test = series.ravel()
        assert_equal(test, series)
        assert_equal(test.mask, series.mask)
        assert_equal(test.dates, series.dates)
        assert_equal(test.varshape, series.varshape)

    def test_ravel_1V(self):
        "Test .ravel on nD/1V data"
        dates = ts.date_array(start_date=ts.Date('M', '2009-01'),
                              length=4)
        series = ts.time_series([[1, 2], [3, 4]],
                                mask=[[0, 0], [1, 0]],
                                dates=dates)
        test = series.ravel()
        assert_equal(test.data, series.data.ravel())
        assert_equal(test.mask, series.mask.ravel())
        assert_equal(test.dates, series.dates.ravel())
        assert_equal(test.varshape, series.varshape)
        assert_equal(test.varshape, ())

    def test_ravel_2V(self):
        "Test .ravel on 2V data"
        series = ts.time_series([[1, 2], [3, 4]],
                                mask=[[0, 0], [1, 0]],
                                start_date=ts.Date('M', '2009-01'),)
        test = series.ravel()
        assert_equal(test.data, series.data)
        assert_equal(test.mask, series.mask)
        assert_equal(test.dates, series.dates)
        assert_equal(test.varshape, series.varshape)
        #
        dates = ts.date_array(start_date=ts.Date('M', '2009-01'),
                              length=2)
        series = ts.time_series([[[1, 2]], [[3, 4]]],
                                mask=[[[0, 0]], [[1, 0]]],
                                dates=dates.reshape(1, 2))
        test = series.ravel()
        assert_equal(test.data, [[1, 2], [3, 4]])
        assert_equal(test.mask, [[0, 0], [1, 0]])
        assert_equal(test.dates, series.dates.ravel())
        assert_equal(test.varshape, (2,))


#------------------------------------------------------------------------------

class TestFunctions(TestCase):
    "Some getitem tests"
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        dlist = ['2007-01-%02i' % i for i in range(1, 16)]
        dates = date_array(dlist, freq='D')
        data = ma.array(np.arange(15), mask=[1, 0, 0, 0, 0]*3)
        self.d = (time_series(data, dates), data, dates)
    #
    def test_adjustendpoints(self):
        "Tests adjust_endpoints"
        (series, data, dates) = self.d
        dseries = adjust_endpoints(series, series.dates[0], series.dates[-1])
        assert_equal(dseries, series)
        dseries = adjust_endpoints(series, series.dates[3], series.dates[-3])
        assert_equal(dseries, series[3:-2])
        dseries = adjust_endpoints(series,
                                   end_date=Date('D', string='2007-01-31'))
        assert_equal(dseries.size, 31)
        assert_equal(dseries._mask, np.r_[series.mask, [1]*16])
        dseries = adjust_endpoints(series,
                                   end_date=Date('D', string='2007-01-06'))
        assert_equal(dseries.size, 6)
        assert_equal(dseries, series[:6])
        dseries = adjust_endpoints(series,
                                   start_date=Date('D', string='2007-01-06'),
                                   end_date=Date('D', string='2007-01-31'))
        assert_equal(dseries.size, 26)
        assert_equal(dseries._mask, np.r_[series.mask[5:], [1]*16])
    #
    def test_adjustendpoints_withdatestrings(self):
        "Tests adjust_endpoints w/ string dates"
        (series, data, dates) = self.d
        dseries = adjust_endpoints(series, end_date='2007-01-31')
        assert_equal(dseries.size, 31)
        assert_equal(dseries._mask, np.r_[series.mask, [1]*16])
        dseries = adjust_endpoints(series, end_date='2007-01-06')
        assert_equal(dseries.size, 6)
        assert_equal(dseries, series[:6])
        dseries = adjust_endpoints(series,
                                   start_date='2007-01-06',
                                   end_date='2007-01-31')
        assert_equal(dseries.size, 26)
        assert_equal(dseries._mask, np.r_[series.mask[5:], [1]*16])
    #
    def test_alignseries(self):
        "Tests align_series & align_with"
        (series, data, dates) = self.d
        #
        empty_series = time_series([], freq='d')
        a, b = align_series(series, empty_series)
        assert_equal(a.start_date, b.start_date)
        assert_equal(a.end_date, b.end_date)
        #
        aseries = time_series(data, dates+10)
        bseries = time_series(data, dates-10)
        (a, b) = align_with(series, aseries, bseries)
        assert_equal(a.dates, series.dates)
        assert_equal(b.dates, series.dates)
        assert_equal(a[-5:], series[:5])
        assert_equal(b[:5], series[-5:])
    #
    def test_tshift(self):
        "Test tshift function"
        series = self.d[0]
        shift_negative = series.tshift(-1)
        result_data = [999] + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        result_mask = [  1] + [1, 0, 0, 0, 0, 1, 0, 0, 0, 0,  1,  0,  0,  0]
        shift_negative_result = time_series(result_data,
                                            dates=series.dates,
                                            mask=result_mask)

        shift_positive = series.tshift(1)
        result_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] + [999]
        result_mask = [0, 0, 0, 0, 1, 0, 0, 0, 0,  1,  0,  0,  0,  0] + [  1]
        shift_positive_result = time_series(result_data,
                                            dates=series.dates,
                                            mask=result_mask)

        assert_array_equal(shift_negative, shift_negative_result)
        assert_array_equal(shift_positive, shift_positive_result)
    #
    def test_split(self):
        """Test the split function."""
        ms = time_series(np.arange(62).reshape(31,2),
                         start_date=Date(freq='d', year=2005, month=7, day=1))
        d1,d2 = split(ms)
        assert_array_equal(d1.data, ms.data[:,0])
        assert_array_equal(d1.dates, ms.dates)
        assert_array_equal(d2.data, ms.data[:,1])

        series = self.d[0]
        ss = split(series)[0]
        assert_array_equal(series, ss)


    def test_convert(self):
        """Test convert function

Just check basic functionality. The details of the actual
date conversion algorithms already tested by asfreq in the
test_dates test suite.
        """
        June2005M = Date(freq='M', year=2005, month=6)
        lowFreqSeries = time_series(np.arange(10), start_date=June2005M)
        # Conversion to same frequency
        assert_array_equal(lowFreqSeries, lowFreqSeries.convert("M"))
        # Conversion to higher frequency - position=START
        lowToHigh_start = lowFreqSeries.convert('B', position='START')
        assert_equal(lowToHigh_start.start_date,
                     June2005M.asfreq("B", relation="START"))
        assert_equal(lowToHigh_start.end_date,
                     (June2005M + 9).asfreq("B", relation="END"))
        assert_equal(lowToHigh_start.mask[0], False)
        assert_equal(lowToHigh_start.mask[-1], True)
        # Conversion to higher frequencyt - position=END
        lowToHigh_end = lowFreqSeries.convert('B', position='END')
        assert_equal(lowToHigh_end.start_date,
                     June2005M.asfreq("B", relation="START"))
        assert_equal(lowToHigh_end.end_date,
                     (June2005M + 9).asfreq("B", relation="END"))
        assert_equal(lowToHigh_end.mask[0], True)
        assert_equal(lowToHigh_end.mask[-1], False)
        # ensure that position argument is not case sensitive
        lowToHigh_start_lowercase = lowFreqSeries.convert('B', position='start')
        assert_array_equal(lowToHigh_start, lowToHigh_start_lowercase)
        #
        # Conversion to lower frequency
        June2005B = Date(freq='b', year=2005, month=6, day=1)
        highFreqSeries = time_series(np.arange(100), start_date=June2005B)
        highToLow = highFreqSeries.convert('M', func=None)
        assert_equal(highToLow.ndim, 2)
        assert_equal(highToLow.shape[1], 23)
        assert_equal(highToLow.start_date, June2005B.asfreq('M'))
        assert_equal(highToLow.end_date, (June2005B + 99).asfreq('M'))

    def test_convert_with_func(self):
        "Test convert w/ function on 1D series"
        mdata = ts.time_series(np.arange(24),
                               mask=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                               start_date=ts.Date('M', '2001-01'))
        test = mdata.convert('A', func=ts.last_unmasked_val)
        control = ts.time_series([7, 22], start_date=ts.Date('A', '2001'))
        assert_equal(test, control)


    def test_convert_nd_with_func(self):
        "Test convert w/ function on nD series"
        ndseries = time_series(np.arange(124).reshape(62, 2),
                               start_date=Date(freq='D', string='2005-07-01'))
        assert_equal(ndseries.convert('M', sum), [[930, 961], [2852, 2883]])





    def test_fill_missing_dates(self):
        """Test fill_missing_dates function"""
        _start = Date(freq='m', year=2005, month=1)
        _end = Date(freq='m', year=2005, month=4)
        #
        dates = date_array([_start, _end], freq='M')
        series = time_series([1, 2], dates)
        filled_ser = fill_missing_dates(series)
        #
        assert_equal(filled_ser.start_date, _start)
        assert_equal(filled_ser.end_date, _end)
        self.failUnless(filled_ser.is_full())
        self.failUnless(not filled_ser.has_duplicated_dates())
        assert_equal(filled_ser.size, _end - _start + 1)
        #
        data = np.arange(5*24).reshape(5,24)
        datelist = ['2007-07-0%i' % i for i in (1, 2, 3, 5, 6)]
        dates = date_array(datelist, freq='D')
        dseries = time_series(data, dates)
        ndates = date_array(start_date=dates[0], end_date=dates[-2])
        #
        fseries = fill_missing_dates(dseries)
        assert_equal(fseries.shape, (6, 24))
        assert_equal(fseries.mask[:,0], [0, 0, 0, 1, 0, 0])
        #
        fseries = fill_missing_dates(dseries[:, 0])
        assert_equal(fseries.shape, (6,))
        assert_equal(fseries.mask, [0, 0, 0, 1, 0, 0])
        #
        series = time_series(data.ravel()[:4].reshape(2, 2), dates=dates[:-1])
        fseries = fill_missing_dates(series)
        assert_equal(fseries.shape, (5,))
        assert_equal(fseries.mask, [0, 0, 0, 1, 0,])

    def test_fill_missing_dates_structured_arrays(self):
        "Test fill_missing_dates on structured arrays"
        ndtype = [('a', float), ('b', float)]
        series = ts.time_series([(1, 1), (2, 2), (3, 3),],
                                dtype=ndtype,
                                dates=['2001-%02i' % i for i in (1, 2, 6)],
                                freq='M')
        test = series.fill_missing_dates()
        control = ts.time_series([(1, 1), (2, 2), (0, 0),
                                  (0, 0), (0, 0), (3, 3),],
                                 mask=[False, False, True, True, True, False],
                                 dtype=ndtype,
                                 start_date=ts.Date('M', '2001-01'))
        assert_equal(test, control)
    #
    def test_fill_missing_dates_undefined(self):
        "Test fill_missing_dates on undefined frequencies."
        ndtype = [('a', float), ('b', float)]
        series = ts.time_series([(1, 1), (2, 2), (3, 3),],
                                dtype=ndtype,
                                dates=[1, 2, 6],
                                freq='U')
        test = series.fill_missing_dates()
        control = ts.time_series([(1, 1), (2, 2), (0, 0),
                                  (0, 0), (0, 0), (3, 3),],
                                 mask=[False, False, True, True, True, False],
                                 dtype=ndtype,
                                 start_date=ts.Date('U', 1))
        assert_equal(test, control)


    def test_pickling(self):
        "Tests pickling/unpickling"
        (series, data, dates) = self.d
        import cPickle
        series_pickled = cPickle.loads(series.dumps())
        assert_equal(series_pickled.dates, series.dates)
        assert_equal(series_pickled.data, series.data)
        assert_equal(series_pickled.mask, series.mask)
        #
        data = ma.array(np.matrix(range(10)).T, mask=[1, 0, 0, 0, 0]*2)
        dates = date_array(start_date=now('D'), length=10)
        series = time_series(data, dates=dates)
        series_pickled = cPickle.loads(series.dumps())
        assert_equal(series_pickled.dates, series.dates)
        assert_equal(series_pickled.data, series.data)
        assert_equal(series_pickled.mask, series.mask)
        self.failUnless(isinstance(series_pickled._data, np.matrix))
    #
    def test_pickling_memo(self):
        "Test the conservation of _optinfo"
        import cPickle
        control = ts.time_series(np.arange(10), start_date=ts.Date('A', 2001))
        control._optinfo['memo'] = "Control information"
        test = cPickle.loads(cPickle.dumps(control))
        assert_equal(test._dates, control._dates)
        assert_equal(test, control)
        assert_equal(test._optinfo, control._optinfo)


    def test_empty_timeseries(self):
        "Tests that empty TimeSeries are  handled properly"
        empty_ts = time_series([], freq='b')
        assert_array_equal(empty_ts, empty_ts + 1)
        assert_array_equal(empty_ts, empty_ts + empty_ts)
        assert_equal(empty_ts.start_date, None)
        assert_equal(empty_ts.end_date, None)

    def test__timeseriescompat_multiple(self):
        "Tests the compatibility of multiple time series."
        newyearsday = Date('D', '2005-01-01')
        aprilsfool = Date('D', '2005-04-01')

        seriesM_10 = time_series(np.arange(10),
                                 date_array(start_date=newyearsday.asfreq('M'),
                                            length=10))
        seriesD_10 = time_series(np.arange(10),
                                 date_array(start_date=newyearsday, length=10))
        seriesD_5 = time_series(np.arange(5),
                                date_array(start_date=newyearsday, length=5))
        seriesD_5_apr = time_series(np.arange(5),
                                    date_array(start_date=aprilsfool, length=5))

        self.failUnless(tseries._timeseriescompat_multiple(seriesM_10, seriesM_10, seriesM_10))

        exception = False
        try:
            tseries._timeseriescompat_multiple(seriesM_10, seriesD_10)
        except ts.TimeSeriesCompatibilityError:
            exception = True
        self.failUnless(exception)

        exception = False
        try:
            tseries._timeseriescompat_multiple(seriesD_5, seriesD_10)
        except ts.TimeSeriesCompatibilityError:
            exception = True
        self.failUnless(exception)

        exception = False
        try:
            tseries._timeseriescompat_multiple(seriesD_5, seriesD_5_apr)
        except ts.TimeSeriesCompatibilityError:
            exception = True
        self.failUnless(exception)

    def test_compressed(self):
        "Tests compress"
        dlist = ['2007-01-%02i' % i for i in range(1, 16)]
        dates = date_array(dlist, freq='D')
        data = ma.array(np.arange(15), mask=[1, 0, 0, 0, 0]*3, dtype=float_)
        series = time_series(data, dlist, freq='D')
        #
        keeper = np.array([0, 1, 1, 1, 1]*3, dtype=bool_)
        c_series = series.compressed()
        assert_equal(c_series.data, [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
        assert_equal(c_series.mask, nomask)
        assert_equal(c_series.dates, dates[keeper])
        #
        series_st = time_series(ma.column_stack((data, data[::-1])),
                                dates=dates)
        c_series = series_st.compressed()
        d = [1, 2, 3, 6, 7, 8, 11, 12, 13]
        assert_equal(c_series.data, np.c_[(d, list(reversed(d)))])
        assert_equal(c_series.mask, nomask)
        assert_equal(c_series.dates, dates[d])

    def test_concatenate(self):
        "Tests concatenate"
        dlist = ['2007-%02i' % i for i in range(1,6)]
        _dates = date_array(dlist, freq='M')
        data = ma.array(np.arange(5), mask=[1, 0, 0, 0, 0], dtype=float_)
        #
        ser_1 = time_series(data, _dates)
        ser_2 = time_series(data, dates=_dates+10)
        newseries = concatenate((ser_1, ser_2), fill_missing=True)
        assert_equal(newseries._series,[0, 1, 2, 3, 4,
                                        0, 0, 0, 0, 0,
                                        0, 1, 2, 3, 4])
        assert_equal(newseries._mask,[1, 0, 0, 0, 0] + [1]*5 + [1, 0, 0, 0, 0])
        assert ~(newseries.has_missing_dates())
         #
        ser_1 = time_series(data, _dates)
        ser_2 = time_series(data, dates=_dates+10)
        newseries = concatenate((ser_1, ser_2))
        assert_equal(newseries._data,[0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        assert_equal(newseries._mask,[1, 0, 0, 0, 0] + [1, 0, 0, 0, 0])
        assert newseries.has_missing_dates()
        #
        ser_2 = time_series(data, dates=_dates+3)
        newseries = concatenate((ser_1, ser_2))
        assert_equal(newseries._data, [0, 1, 2, 3, 4, 2, 3, 4])
        assert_equal(newseries._mask, [1, 0, 0, 0, 0, 0, 0, 0])
        #
        newseries = concatenate((ser_1, ser_1[::-1]))
        assert_equal(newseries, ser_1)
    #
    def test_maxmin(self):
        "Test min/max"
        series = time_series(np.arange(10), start_date=now('D'))
        smax = series.max()
        #!!!: Used to be a TimeSeries, now is only a scalar
#        self.failUnless(isinstance(smax, TimeSeries))
#        assert_equal(smax._dates, date_array(series._dates[-1]))
        self.failUnless(not isinstance(smax, TimeSeries))
        assert_equal(smax, 9)
        #
        smin = series.min()
        #!!!: Used to be a TimeSeries, now is only a scalar
#        self.failUnless(isinstance(smin, TimeSeries))
#        assert_equal(smin._dates, date_array(series._dates[0]))
        assert_equal(smin, 0)
        #
        series = time_series([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]],
                             start_date=now('D'))
        smax = series.max(0)
        assert_equal(smax.series, [9, 8, 7, 6, 5])
        assert_equal(smax.dates, date_array([series.dates[1]]*5))
        smax = series.max(1)
        assert_equal(smax.series, [4, 9])
        assert_equal(smax.dates, series.dates)
        smax = series.max()
        assert_equal(smax.series, [9])
        assert_equal(smax.dates, date_array(series.dates[1]))

        ser_m = ts.time_series(range(10), freq='M', start_date='2008-01-01')
        ser_q = ser_m.convert(freq='Q')
        mx = ser_q.max(-1)
        assert_equal(mx, ma.array([2, 5, 8, 9]))
        self.failUnless(isinstance(mx, TimeSeries))

    #
    def test_pct(self):
        series = time_series(np.arange(1, 10), start_date=now('D'))
        _pct = series.pct()
        assert_equal(_pct.dtype, np.dtype('d'))
        assert_equal(series.start_date, _pct.start_date)
        assert_equal(series.end_date, _pct.end_date)
        self.failUnless(_pct[0] is masked)
        assert_equal(_pct[1], 1.0)
        assert_equal(_pct[2], 0.5)

        series = ts.time_series([2.,1.,2.,3.],
                                start_date=ts.Date(freq='A', year=2005))

        # standard pct
        result = series.pct()
        assert_almost_equal(result,
                            ma.array([999, -0.5, 1.0, 0.5], mask=[1,0,0,0])
        )

        result = series.pct(2)
        assert_almost_equal(
            result,
            ma.array([999, 999, 0.0, 2.0], mask=[1,1,0,0])
        )

        # log pct
        result = series.pct_log()
        assert_almost_equal(
            result,
            ma.array(
                [999, -0.69314718056, 0.69314718056, 0.405465108108],
                mask=[1,0,0,0])
        )

        result = series.pct_log(2)
        assert_almost_equal(
            result,
            ma.array([999, 999, 0.0, 1.09861228867], mask=[1,1,0,0])
        )

        # symmetric pct
        result = series.pct_symmetric()
        assert_almost_equal(
            result,
            ma.array(
                [999, -0.666666666667, 0.666666666667, 0.4], mask=[1,0,0,0])
        )

        result = series.pct_symmetric(2)
        assert_almost_equal(
            result,
            ma.array([999, 999, 0.0, 1.0], mask=[1,1,0,0])
        )

    def test_find_duplicated_dates(self):
        "Test find_duplicated_dates"
        years = ['2000', '2001', '2002', '2003', '2003',
                 '2003', '2004', '2005', '2005', '2006']
        series = time_series(np.arange(len(years)), dates=years, freq='A')
        test = find_duplicated_dates(series)
        control = {Date('A', '2003'): (np.array([3, 4, 5]),),
                   Date('A', '2005'): (np.array([7, 8]),),}
        assert_equal(test, control)
    #
    def test_find_duplicated_dates_allduplicated(self):
        "Test find_duplicated_dates w/all duplicates"
        series = time_series([0, 1, 2, 3, 4],
                             dates=[2000, 2000, 2000, 2000, 2000], freq='A')
        test = find_duplicated_dates(series)
        control = {Date('A', '2000'): (np.array([0, 1, 2, 3, 4]),),}
        assert_equal(test, control)
    #
    def test_find_duplicated_dates_noduplicates(self):
        "Test find_duplicated_dates w/o duplicates"
        series = time_series(np.arange(5), start_date=Date('A', '2001'))
        test = find_duplicated_dates(series)
        assert_equal(test, {})


    def test_remove_duplicated_dates(self):
        "Test remove_duplicated_dates"
        years = ['2000', '2001', '2002', '2003', '2003',
                 '2003', '2004', '2005', '2005', '2006']
        series = time_series(np.arange(len(years)), dates=years, freq='A')
        test = remove_duplicated_dates(series)
        control = time_series([0, 1, 2, 3, 6, 7, 9],
                              start_date=Date('A', '2000'))
        assert_equal(test, control)
        assert_equal(test._dates, control._dates)
    #
    def test_remove_duplicated_dates_allduplicates(self):
        "Test remove_duplicated_dates w/ all duplicates"
        years = ['2000', '2000', '2000', '2000', '2000']
        series = time_series(np.arange(len(years)), dates=years, freq='A')
        test = remove_duplicated_dates(series)
        control = time_series([0,],
                              start_date=Date('A', '2000'))
        assert_equal(test, control)
        assert_equal(test._dates, control._dates)
    #
    def test_remove_duplicated_dates_noduplicates(self):
        "Test remove_duplicated_dates w/o duplicates"
        series = time_series(np.arange(5), start_date=Date('A', '2001'))
        test = remove_duplicated_dates(series)
        assert_equal(test, series)
        assert_equal(test._dates, series._dates)
    #
    def test_remove_duplicated_dates_nonchrono(self):
        "Test remove_duplicated_dates on non-chronological series"
        series = time_series([0, 1, 2, 3, 4, 5, 6],
                             dates=[2005, 2005, 2004, 2003, 2002, 2002, 2002],
                             freq='A',
                             autosort=False)
        test = remove_duplicated_dates(series)
        control = time_series([0, 2, 3, 4],
                              dates=[2005, 2004, 2003, 2002], freq='A',
                              autosort=True)
        assert_equal(test, control)
        assert_equal(test._dates, control._dates)




#------------------------------------------------------------------------------

class TestMisc(TestCase):


    def test_ma_ufuncs(self):
        a = time_series([-2, -1, 0, 1, 2], start_date=now('D'))
        z = ma.sqrt(a)
        self.failUnless(isinstance(z, TimeSeries))
        assert_equal(z, [1, 1, 0, 1, np.sqrt(2)])
        assert_equal(z.mask, [1, 1, 0, 0, 0])
        assert_equal(z.dates, a.dates)


    def test_emptylike(self):
        x = time_series([1, 2, 3, 4, 5], mask=[1, 0, 0, 0, 0],
                        start_date=now('D'))
        y = ts.empty_like(x)
        # Basic checks
        assert_equal(x.dtype, y.dtype)
        assert_equal(x.shape, y.shape)
        #
        y.flat = 0
        assert_equal(x.mask, [1,0,0,0,0])
        assert_equal(y.mask, nomask)
        #
        x.mask = nomask
        y = ts.empty_like(x)
        assert_equal(y.mask, nomask)


    def test_compatibility_shape(self):
        "Tests shape compatibility."
        data = np.arange(2*3*4*5,)
        dates = np.empty((2*3*4*5,))
        assert_equal(get_varshape(data, dates), ())
        #
        dates.shape = (2,3,4,5)
        assert_equal(get_varshape(data, dates), ())
        #
        dates = np.empty((2*3*4,))
        try:
            assert_equal(get_varshape(data, dates), None)
        except TimeSeriesCompatibilityError:
            pass
        #
        dates = np.empty((3*3*5,))
        try:
            assert_equal(get_varshape(data, dates), None)
        except TimeSeriesCompatibilityError:
            pass
        #
        data.shape = (2*3*4,5)
        dates = np.empty((2*3*4,))
        assert_equal(get_varshape(data, dates), (5,))
        data.shape = (2*3, 4*5)
        dates = np.empty((2*3*4,))
        try:
            assert_equal(get_varshape(data, dates), None)
        except TimeSeriesCompatibilityError:
            pass
        dates = np.empty((2*3,4))
        try:
            assert_equal(get_varshape(data, dates), None)
        except TimeSeriesCompatibilityError:
            pass
        data.shape = (2*3, 4, 5)
        dates = np.empty((2,))
        try:
            assert_equal(get_varshape(data, dates), None)
        except TimeSeriesCompatibilityError:
            pass
        dates = np.empty((2*3,))
        assert_equal(get_varshape(data, dates), (4, 5))
        # 1D
        start = ts.now('M')
        series = time_series(np.arange(60), start_date=start)
        assert_equal(series._varshape, ())
        # 2D (multi 1D series)
        series = time_series(np.arange(60).reshape(20, 3), start_date=start)
        assert_equal(series._varshape, (3,))
        # 3D (2D series)
        series = time_series(np.arange(60).reshape(5, 4, 3), start_date=start)
        assert_equal(series._varshape, (4, 3))


    def test_deepcopy(self):
        "Test deepcopy"
        from copy import deepcopy
        t = time_series([0,1,2], mask=[0,1,0], start_date=ts.now('D'))
        t_ = deepcopy(t)
        for attr in ('_data','_mask','_dates'):
            attrt = getattr(t, attr)
            attrt_ = getattr(t_, attr)
            assert_equal(attrt, attrt_)
            assert_not_equal(id(attrt), id(attrt_))
        t_.mask[1] = False
        assert_equal(t_.mask, [False, False, False])
        assert_equal(t.mask, [False, True, False])


    def test_firstlast_unmasked_vals(self):
        data = ma.array([[ 0,  1,  2,  3,  4],
                         [ 5,  6,  7,  8,  9],
                         [10, 11, 12, 13, 14],
                         [15, 16, 17, 18, 19],
                         [20, 21, 22, 23, 24]],
                        mask=[[0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 1],
                              [1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0]],)
        assert_equal(ts.first_unmasked_val(data, None), 0)
        assert_equal(ts.last_unmasked_val(data, None), 24)
        assert_equal(ts.first_unmasked_val(data, 0), [0, 1, 7, 3, 4])
        assert_equal(ts.last_unmasked_val(data, 0), [15, 16, 17, 23, 24])
        assert_equal(ts.first_unmasked_val(data, -1), [0, 5, 12, 15, 23])
        assert_equal(ts.last_unmasked_val(data, -1), [4, 7, 14, 19, 24])
        #
        data_ = data.data
        assert_equal(ts.first_unmasked_val(data_, None), 0)
        assert_equal(ts.last_unmasked_val(data_, None), 24)
        assert_equal(ts.first_unmasked_val(data_, 0), [0, 1, 2, 3, 4])
        assert_equal(ts.last_unmasked_val(data_, 0), [20, 21, 22, 23, 24])
        assert_equal(ts.first_unmasked_val(data_, -1), [0, 5, 10, 15, 20])
        assert_equal(ts.last_unmasked_val(data_, -1), [4, 9, 14, 19, 24])
        #
        data[-2] = ma.masked
        assert_equal(ts.first_unmasked_val(data, None), 0)
        assert_equal(ts.last_unmasked_val(data, None), 24)
        assert_equal(ts.first_unmasked_val(data, 0), [0, 1, 7, 3, 4])
        assert_equal(ts.last_unmasked_val(data, 0), [5, 6, 12, 23, 24])
        assert_equal(ts.first_unmasked_val(data, -1),
                     ma.array([0, 5, 12, -1, 23], mask=[0, 0, 0, 1, 0]))
        assert_equal(ts.last_unmasked_val(data, -1),
                     ma.array([4, 7, 14, -1, 24], mask=[0, 0, 0, 1, 0]))



#------------------------------------------------------------------------------

class TestGenericMethods(TestCase):
    #
    class SubTimeSeries(TimeSeries):
        pass
    #
    def setUp(self):
        self.methodlist = ('cumsum','cumprod','ravel')
        data = np.random.rand(10)
        self.series = time_series(data, start_date=ts.now('D')-len(data))
    #
    def test_generic_methods(self):
        "Tests some generic methods"
        series = self.series
        for method in self.methodlist:
            test = getattr(series, method).__call__()
            self.failUnless(isinstance(test, ts.TimeSeries))
            assert_equal(test, getattr(series.series, method).__call__())
            assert_equal(test.dates, series.dates)
    #
    def test_generic_methods_w_subclassing(self):
        "Tests generic method on subclasses of TimeSeries."
        subseries = self.series.view(self.SubTimeSeries)
        for method in self.methodlist:
            test = getattr(subseries, method).__call__()
            self.failUnless(isinstance(test, self.SubTimeSeries))
            assert_equal(test, getattr(subseries._series, method).__call__())
            assert_equal(test.dates, subseries.dates)
    #
    def test_generated_method(self):
        "Test that TimeSeries.method(s) gives the same result as s.method"
        series = ts.time_series(range(10), start_date=ts.now('D'))
        control = ts.time_series(np.cumsum(range(10)), start_date=ts.now('D'))
        #
        test = series.cumsum()
        assert_equal(test, control)
        #
        test = ts.TimeSeries.cumsum(series)
        assert_equal(test, control)
    #
    def test_generated_axismethod(self):
        "Test axismethod"
        series = ts.time_series(np.arange(9).reshape(3,3),
                                start_date=ts.now('D'))
        control = ts.time_series([1., 4., 7.], start_date=ts.now('D'))
        #
        test = series.mean(1)
        assert_equal(test, control)
        #
        test = ts.TimeSeries.mean(series, 1)
        assert_equal(test, control)
        test = ts.TimeSeries.mean(series, axis=1)
        assert_equal(test, control)
    #
    def test_axismethod(self):
        "Test axis method"
        series = ts.time_series(np.arange(9).reshape(3, 3),
                                start_date=ts.now('D'))
        control = ts.time_series([0, 60, 336], start_date=ts.now('D'))
        assert_equal(series.product(axis=-1), control)
        assert_equal(series.product(-1), control)
        assert_equal(series.prod(axis=-1), control)
        assert_equal(series.prod(-1), control)
        #
        control = ts.time_series([3, 12, 21], start_date=ts.now('D'))
        assert_equal(series.sum(axis=-1), control)
        assert_equal(series.sum(-1), control)



#------------------------------------------------------------------------------

class TestFlexibleType(TestCase):
    "Test flexible types"
    #
    def setUp(self):
        ndtype = [('a',float), ('b',float)]
        data = ma.array(zip(np.random.rand(10), np.arange(10)), dtype=ndtype)
        data.mask[0] = (0,1)
        data.mask[1] = (1,1)
        data.mask[-1] = (1,0)
        series = time_series(data, start_date=ts.Date('M', '2007-01'))
        self.data = (data, series)
    #
    def test_getitem_index(self):
        (data, series) = self.data
        test = series[0]
        self.failUnless(isinstance(test, MaskedArray))
        assert_equal(test, data[0])
        test = series[1]
        self.failUnless(isinstance(test, MaskedArray))
        assert_equal(test, data[1])
        test = series[2]
        self.failUnless(isinstance(test, np.void))
        assert_equal(test, data[2])
    #
    def test_getitem_dates(self):
        (data, series) = self.data
        test = series['2007-01']
        self.failUnless(isinstance(test, MaskedArray))
        assert_equal(test, data[0])
        test = series['2007-02']
        self.failUnless(isinstance(test, MaskedArray))
        assert_equal(test, data[1])
        test = series['2007-03']
        self.failUnless(isinstance(test, np.void))
        assert_equal(test, data[2])


#------------------------------------------------------------------------------

class TestViewTimeSeries(TestCase):
    #
    def setUp(self):
        (a, b) = (np.arange(10), np.random.rand(10))
        ndtype = [('a',np.float), ('b',np.float)]
        tarr = ts.time_series(np.array(zip(a,b), dtype=ndtype),
                              start_date=ts.now('M'))
        tarr.mask[3] = (False, True)
        self.data = (tarr, a, b)
    #
    def test_view_by_itself(self):
        (tarr, a, b) = self.data
        test = tarr.view()
        self.failUnless(isinstance(test, ts.TimeSeries))
        assert_equal_records(test, tarr)
        assert_equal_records(test.mask, tarr.mask)
        assert_equal(test.dates, tarr.dates)
    #
    def test_view_simple_dtype(self):
        (tarr, a, b) = self.data
        ntype = (np.float, 2)
        test = tarr.view(ntype)
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal(test, np.array(zip(a, b), dtype=np.float))
        self.failUnless(test[3,1] is ma.masked)
    #
    def test_view_flexible_type(self):
        (tarr, a, b) = self.data
        arr = tarr._series
        alttype = [('A',np.float), ('B',np.float)]
        test = tarr.view(alttype)
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal_records(test, arr.view(alttype))
        self.failUnless(test['B'][3] is masked)
        assert_equal(test.dtype, np.dtype(alttype))
        self.failUnless(test._fill_value is None)
    #
    def test_view(self):
        "Test view w/ flexible dtype"
        ndtype = [('a', float), ('b', float), ('c', float)]
        data = np.random.rand(15).reshape(-1, 3)
        fseries = time_series([tuple(_) for _ in data],
                              start_date=ts.now('D'), dtype=ndtype)
        dseries = time_series(data, start_date=ts.now('D'))
        #
        test = fseries.view((float,3))
        assert_equal(test._series, dseries)
        self.failUnless(test._varshape == (3,))
        #
        self.failUnless(dseries._varshape == (3,))
        test = dseries.view(ndtype).squeeze()
        assert_equal(test, fseries)
        self.failUnless(test._varshape == ())

###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    run_module_suite()
