# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for Date handling.

:author: Pierre Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknow_ca_at_hotmail_dot_com
:version: $Id: test_dates.py 3836 2008-01-15 13:09:03Z matthew.brett@gmail.com $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: matthew.brett@gmail.com $)"
__revision__ = "$Revision: 3836 $"
__date__ = '$Date: 2008-01-15 08:09:03 -0500 (Tue, 15 Jan 2008) $'

import types
import datetime

import numpy as np
from numpy.testing import *

from numpy import ma
from numpy.ma.testutils import assert_equal, assert_array_equal

import scikits.timeseries as ts
from scikits.timeseries import const as C, Date, DateArray, now, date_array
from scikits.timeseries.cseries import freq_dict
from scikits.timeseries.tdates import convert_to_float



class TestCreation(TestCase):
    "Base test class for MaskedArrays."

    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)

    def test_fromstrings(self):
        "Tests creation from list of strings"
        # A simple case: daily data
        dlist = ['2007-01-%02i' % i for i in range(1, 15)]
        dates = date_array(dlist, freq='D')
        assert_equal(dates.freqstr, 'D')
        self.failUnless(dates.is_full())
        self.failUnless(not dates.has_duplicated_dates())
        assert_equal(dates, 732677 + np.arange(len(dlist)))

        # Still daily data, that we force to month
        dates = date_array(dlist, freq='M')
        assert_equal(dates.freqstr, 'M')
        self.failUnless(not dates.is_full())
        self.failUnless(dates.has_duplicated_dates())
        assert_equal(dates, [24073] * len(dlist))

        # Now, for monthly data
        dlist = ['2007-%02i' % i for i in range(1, 13)]
        dates = date_array(dlist, freq='M')
        assert_equal(dates.freqstr, 'M')
        self.failUnless(dates.is_full())
        self.failUnless(not dates.has_duplicated_dates())
        assert_equal(dates, 24073 + np.arange(12))

        # quarterly date
        assert_equal(
            Date(freq='Q', string='2007-01'),
            Date(freq='Q', year=2007, quarter=1))

        # just a year
        assert_equal(Date(freq='A', string='2007'), Date(freq='A', year=2007))


    def test_fromstrings_sorting_bug(self):
        """regression test for previous bug with string dates getting sorted
        incorrectly"""
        dlist = ['5-jan-2005', '1-apr-2008', '3-may-2009']
        dvals = [Date(freq='d', string=x).value for x in dlist]
        dvals = np.array(dvals)
        dates = date_array(dlist, freq='d')
        assert_equal(dates, dvals)


    def test_from_startend_dates_strings(self):
        "Test creating from a starting & ending dates as strings"
        control = DateArray(np.arange(366) + 733042, freq='D')
        test = date_array('2008-01-01', '2008-12-31', freq='D')
        assert_equal(test, control)
        test = date_array(start_date='2008-01-01', end_date='2008-12-31',
                          freq='D')
        assert_equal(test, control)
        test = date_array(start_date='2008-12-31', end_date='2008-01-01',
                          freq='D')
        assert_equal(test, control)


    def test_fromstrings_wmissing(self):
        "Tests creation from list of strings w/ missing dates"
        dlist = ['2007-01-%02i' % i for i in (1, 2, 4, 5, 7, 8, 10, 11, 13)]
        #
        dates = date_array(dlist)
        assert_equal(dates.freqstr, 'U')
        self.failUnless(not dates.is_full())
        self.failUnless(not dates.has_duplicated_dates())
        assert_equal(dates.tovalue(), 732676 + np.array([1, 2, 4, 5, 7, 8, 10, 11, 13]))
        #
        ddates = date_array(dlist, freq='D')
        assert_equal(ddates.freqstr, 'D')
        self.failUnless(not ddates.is_full())
        self.failUnless(not ddates.has_duplicated_dates())
        #
        mdates = date_array(dlist, freq='M')
        assert_equal(mdates.freqstr, 'M')
        self.failUnless(not mdates.is_full())
        self.failUnless(mdates.has_duplicated_dates())


    def test_fromsobjects(self):
        "Tests creation from list of objects."
        #
        dlist = ['2007-01-%02i' % i for i in (1, 2, 4, 5, 7, 8, 10, 11, 13)]
        dates = date_array(dlist, freq='D')
        dobj = [d.datetime for d in dates]
        odates = date_array(dobj, freq='D')
        assert_equal(dates, odates)
        # check that frequency gets set when passing list of Date objects
        dlist = [Date(freq='M', year=2001, month=2), Date(freq='M', year=2001, month=3)]
        dates = date_array(dlist)
        assert_equal(dates.freq, dlist[0].freq)
        #
        dates = date_array(['2006-01'], freq='M')
        assert_equal(dates[0], ts.Date(freq='M', year=2006, month=1))


    def test_from_datetime_objects(self):
        "Test creation from a list of datetime.date or datetime.datetime objects."
        # test from datetime.date object
        _dt = ts.Date(freq='D', datetime=datetime.date(2007, 1, 1))
        _tsdt = ts.Date(freq='D', year=2007, month=1, day=1)
        assert_equal(_dt, _tsdt)
        # test from datetime.datetime object
        _dt = ts.Date(freq='D', datetime=datetime.datetime(2007, 1, 1, 0, 0, 0, 0))
        assert_equal(_dt, _tsdt)

        # try using the 'value' positional arg
        _dt = ts.Date('D', datetime.datetime(2007, 1, 1, 0, 0, 0, 0))
        assert_equal(_dt, _tsdt)


    def test_consistent_value(self):
        "Tests that values don't get mutated when constructing dates from a value"
        freqs = [x[0] for x in freq_dict.values() if x[0] != 'U']
        for f in freqs:
            _now = now(f)
            assert_equal(Date(freq=f, value=_now.value), _now)


    def test_shortcuts(self):
        "Tests some creation shortcuts. Because I'm lazy like that."
        # Dates shortcuts
        assert_equal(Date('D', '2007-01'), Date('D', string='2007-01'))
        assert_equal(Date('D', '2007-01'), Date('D', value=732677))
        assert_equal(Date('D', 732677), Date('D', value=732677))
        # DateArray shortcuts
        n = now('M')
        d = date_array(start_date=n, length=3)
        assert_equal(date_array(n, length=3), d)
        assert_equal(date_array(n, n + 2), d)


    def test_unsorted(self):
        "Test DateArray on unsorted data"
        dates = ts.DateArray([2001, 2007, 2003, 2002, 2001], freq='Y')
        assert(dates.has_duplicated_dates())
        assert(dates.has_missing_dates())
        assert_equal(dates.start_date.value, 2001)
        assert_equal(dates.end_date.value, 2007)
        #
        dates = ts.DateArray(['2001', '2007', '2003', '2002', '2001'], freq='Y')
        assert(dates.has_duplicated_dates())
        assert(dates.has_missing_dates())
        assert_equal(dates.start_date.value, 2001)
        assert_equal(dates.end_date.value, 2007)
        #
        dates = ts.DateArray([ts.Date('A', year=i)
                              for i in (2001, 2007, 2003, 2002, 2001)],
                             freq='Y')
        assert(dates.has_duplicated_dates())
        assert(dates.has_missing_dates())
        assert_equal(dates.start_date.value, 2001)
        assert_equal(dates.end_date.value, 2007)
        #
        dates = ts.date_array([2001, 2007, 2003, 2002, 2001], freq='Y')
        assert(dates.has_duplicated_dates())
        assert(dates.has_missing_dates())
        assert_equal(dates.start_date.value, 2001)
        assert_equal(dates.end_date.value, 2007)



class TestDateProperties(TestCase):
    "Test properties such as year, month, weekday, etc...."
    #
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)

    def test_properties_annually(self):
        "Test properties on DateArrays with annually frequency."
        a_date = Date(freq='A', year=2007)
        assert_equal(a_date.year, 2007)


    def test_properties_quarterly(self):
        "Test properties on DateArrays with daily frequency."
        q_date = Date(freq=C.FR_QTREDEC, year=2007, quarter=1)
        qedec_date = Date(freq=C.FR_QTREDEC, year=2007, quarter=1)
        qejan_date = Date(freq=C.FR_QTREJAN, year=2007, quarter=1)
        qejun_date = Date(freq=C.FR_QTREJUN, year=2007, quarter=1)
        qsdec_date = Date(freq=C.FR_QTREDEC, year=2007, quarter=1)
        qsjan_date = Date(freq=C.FR_QTREJAN, year=2007, quarter=1)
        qsjun_date = Date(freq=C.FR_QTREJUN, year=2007, quarter=1)
        #
        for x in range(3):
            for qd in (qedec_date, qejan_date, qejun_date,
                       qsdec_date, qsjan_date, qsjun_date):
                assert_equal((qd + x).qyear, 2007)
                assert_equal((qd + x).quarter, x + 1)


    def test_properties_monthly(self):
        "Test properties on DateArrays with daily frequency."
        m_date = Date(freq='M', year=2007, month=1)
        for x in range(11):
            m_date_x = m_date + x
            assert_equal(m_date_x.year, 2007)
            if 1 <= x + 1 <= 3:
                assert_equal(m_date_x.quarter, 1)
            elif 4 <= x + 1 <= 6:
                assert_equal(m_date_x.quarter, 2)
            elif 7 <= x + 1 <= 9:
                assert_equal(m_date_x.quarter, 3)
            elif 10 <= x + 1 <= 12:
                assert_equal(m_date_x.quarter, 4)
            assert_equal(m_date_x.month, x + 1)


    def test_properties_weekly(self):
        "Test properties on DateArrays with daily frequency."
        w_date = Date(freq='W', year=2007, month=1, day=7)
        #
        assert_equal(w_date.year, 2007)
        assert_equal(w_date.quarter, 1)
        assert_equal(w_date.month, 1)
        assert_equal(w_date.week, 1)
        assert_equal((w_date - 1).week, 52)


    def test_properties_daily(self):
        "Test properties on DateArrays with daily frequency."
        b_date = Date(freq='B', year=2007, month=1, day=1)
        #
        assert_equal(b_date.year, 2007)
        assert_equal(b_date.quarter, 1)
        assert_equal(b_date.month, 1)
        assert_equal(b_date.day, 1)
        assert_equal(b_date.weekday, 0)
        assert_equal(b_date.day_of_year, 1)
        #
        d_date = Date(freq='D', year=2007, month=1, day=1)
        #
        assert_equal(d_date.year, 2007)
        assert_equal(d_date.quarter, 1)
        assert_equal(d_date.month, 1)
        assert_equal(d_date.day, 1)
        assert_equal(d_date.weekday, 0)
        assert_equal(d_date.day_of_year, 1)


    def test_properties_hourly(self):
        "Test properties on DateArrays with hourly frequency."
        h_date = Date(freq='H', year=2007, month=1, day=1, hour=0)
        #
        assert_equal(h_date.year, 2007)
        assert_equal(h_date.quarter, 1)
        assert_equal(h_date.month, 1)
        assert_equal(h_date.day, 1)
        assert_equal(h_date.weekday, 0)
        assert_equal(h_date.day_of_year, 1)
        assert_equal(h_date.hour, 0)
        #
        harray = date_array(start_date=h_date, end_date=h_date + 3000)
        assert_equal(harray.week[0], h_date.week)
        assert_equal(harray.week[-1], (h_date + 3000).week)


    def test_properties_minutely(self):
        "Test properties on DateArrays with minutely frequency."
        t_date = Date(freq='T', year=2007, month=1, day=1, hour=0, minute=0)
        #
        assert_equal(t_date.quarter, 1)
        assert_equal(t_date.month, 1)
        assert_equal(t_date.day, 1)
        assert_equal(t_date.weekday, 0)
        assert_equal(t_date.day_of_year, 1)
        assert_equal(t_date.hour, 0)
        assert_equal(t_date.minute, 0)


    def test_properties_secondly(self):
        "Test properties on DateArrays with secondly frequency."
        s_date = Date(freq='T', year=2007, month=1, day=1,
                                       hour=0, minute=0, second=0)
        #
        assert_equal(s_date.year, 2007)
        assert_equal(s_date.quarter, 1)
        assert_equal(s_date.month, 1)
        assert_equal(s_date.day, 1)
        assert_equal(s_date.weekday, 0)
        assert_equal(s_date.day_of_year, 1)
        assert_equal(s_date.hour, 0)
        assert_equal(s_date.minute, 0)
        assert_equal(s_date.second, 0)



def dArrayWrap(date):
    "wrap a date into a DateArray of length 1"
    return date_array(start_date=date, length=1)

def noWrap(item):
    return item


class TestFreqConversion(TestCase):
    "Test frequency conversion of date objects"

    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.dateWrap = [(dArrayWrap, assert_array_equal),
                         (noWrap, assert_equal)]

    def test_conv_annual(self):
        "frequency conversion tests: from Annual Frequency"

        for dWrap, assert_func in self.dateWrap:
            date_A = dWrap(Date(freq='A', year=2007))

            date_AJAN = dWrap(Date(freq=C.FR_ANNJAN, year=2007))
            date_AJUN = dWrap(Date(freq=C.FR_ANNJUN, year=2007))
            date_ANOV = dWrap(Date(freq=C.FR_ANNNOV, year=2007))

            date_A_to_Q_start = dWrap(Date(freq='Q', year=2007, quarter=1))
            date_A_to_Q_end = dWrap(Date(freq='Q', year=2007, quarter=4))
            date_A_to_M_start = dWrap(Date(freq='M', year=2007, month=1))
            date_A_to_M_end = dWrap(Date(freq='M', year=2007, month=12))
            date_A_to_W_start = dWrap(Date(freq='W', year=2007, month=1, day=1))
            date_A_to_W_end = dWrap(Date(freq='W', year=2007, month=12, day=31))
            date_A_to_B_start = dWrap(Date(freq='B', year=2007, month=1, day=1))
            date_A_to_B_end = dWrap(Date(freq='B', year=2007, month=12, day=31))
            date_A_to_D_start = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_A_to_D_end = dWrap(Date(freq='D', year=2007, month=12, day=31))
            date_A_to_H_start = dWrap(Date(freq='H', year=2007, month=1, day=1,
                                      hour=0))
            date_A_to_H_end = dWrap(Date(freq='H', year=2007, month=12, day=31,
                                     hour=23))
            date_A_to_T_start = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                      hour=0, minute=0))
            date_A_to_T_end = dWrap(Date(freq='T', year=2007, month=12, day=31,
                                     hour=23, minute=59))
            date_A_to_S_start = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                      hour=0, minute=0, second=0))
            date_A_to_S_end = dWrap(Date(freq='S', year=2007, month=12, day=31,
                                     hour=23, minute=59, second=59))

            date_AJAN_to_D_end = dWrap(Date(freq='D', year=2007, month=1, day=31))
            date_AJAN_to_D_start = dWrap(Date(freq='D', year=2006, month=2, day=1))
            date_AJUN_to_D_end = dWrap(Date(freq='D', year=2007, month=6, day=30))
            date_AJUN_to_D_start = dWrap(Date(freq='D', year=2006, month=7, day=1))
            date_ANOV_to_D_end = dWrap(Date(freq='D', year=2007, month=11, day=30))
            date_ANOV_to_D_start = dWrap(Date(freq='D', year=2006, month=12, day=1))

            assert_func(date_A.asfreq('Q', "START"), date_A_to_Q_start)
            assert_func(date_A.asfreq('Q', "END"), date_A_to_Q_end)
            assert_func(date_A.asfreq('M', "START"), date_A_to_M_start)
            assert_func(date_A.asfreq('M', "END"), date_A_to_M_end)
            assert_func(date_A.asfreq('W', "START"), date_A_to_W_start)
            assert_func(date_A.asfreq('W', "END"), date_A_to_W_end)
            assert_func(date_A.asfreq('B', "START"), date_A_to_B_start)
            assert_func(date_A.asfreq('B', "END"), date_A_to_B_end)
            assert_func(date_A.asfreq('D', "START"), date_A_to_D_start)
            assert_func(date_A.asfreq('D', "END"), date_A_to_D_end)
            assert_func(date_A.asfreq('H', "START"), date_A_to_H_start)
            assert_func(date_A.asfreq('H', "END"), date_A_to_H_end)
            assert_func(date_A.asfreq('T', "START"), date_A_to_T_start)
            assert_func(date_A.asfreq('T', "END"), date_A_to_T_end)
            assert_func(date_A.asfreq('S', "START"), date_A_to_S_start)
            assert_func(date_A.asfreq('S', "END"), date_A_to_S_end)

            assert_func(date_AJAN.asfreq('D', "START"), date_AJAN_to_D_start)
            assert_func(date_AJAN.asfreq('D', "END"), date_AJAN_to_D_end)

            assert_func(date_AJUN.asfreq('D', "START"), date_AJUN_to_D_start)
            assert_func(date_AJUN.asfreq('D', "END"), date_AJUN_to_D_end)

            assert_func(date_ANOV.asfreq('D', "START"), date_ANOV_to_D_start)
            assert_func(date_ANOV.asfreq('D', "END"), date_ANOV_to_D_end)

            assert_func(date_A.asfreq('A'), date_A)


    def test_conv_quarterly(self):
        "frequency conversion tests: from Quarterly Frequency"

        for dWrap, assert_func in self.dateWrap:
            date_Q = dWrap(Date(freq='Q', year=2007, quarter=1))
            date_Q_end_of_year = dWrap(Date(freq='Q', year=2007, quarter=4))

            date_QEJAN = dWrap(Date(freq=C.FR_QTREJAN, year=2007, quarter=1))
            date_QEJUN = dWrap(Date(freq=C.FR_QTREJUN, year=2007, quarter=1))

            date_QSJAN = dWrap(Date(freq=C.FR_QTRSJAN, year=2007, quarter=1))
            date_QSJUN = dWrap(Date(freq=C.FR_QTRSJUN, year=2007, quarter=1))
            date_QSDEC = dWrap(Date(freq=C.FR_QTRSDEC, year=2007, quarter=1))

            date_Q_to_A = dWrap(Date(freq='A', year=2007))
            date_Q_to_M_start = dWrap(Date(freq='M', year=2007, month=1))
            date_Q_to_M_end = dWrap(Date(freq='M', year=2007, month=3))
            date_Q_to_W_start = dWrap(Date(freq='W', year=2007, month=1, day=1))
            date_Q_to_W_end = dWrap(Date(freq='W', year=2007, month=3, day=31))
            date_Q_to_B_start = dWrap(Date(freq='B', year=2007, month=1, day=1))
            date_Q_to_B_end = dWrap(Date(freq='B', year=2007, month=3, day=30))
            date_Q_to_D_start = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_Q_to_D_end = dWrap(Date(freq='D', year=2007, month=3, day=31))
            date_Q_to_H_start = dWrap(Date(freq='H', year=2007, month=1, day=1,
                                      hour=0))
            date_Q_to_H_end = dWrap(Date(freq='H', year=2007, month=3, day=31,
                                     hour=23))
            date_Q_to_T_start = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                      hour=0, minute=0))
            date_Q_to_T_end = dWrap(Date(freq='T', year=2007, month=3, day=31,
                                     hour=23, minute=59))
            date_Q_to_S_start = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                      hour=0, minute=0, second=0))
            date_Q_to_S_end = dWrap(Date(freq='S', year=2007, month=3, day=31,
                                     hour=23, minute=59, second=59))

            date_QEJAN_to_D_start = dWrap(Date(freq='D', year=2006, month=2, day=1))
            date_QEJAN_to_D_end = dWrap(Date(freq='D', year=2006, month=4, day=30))

            date_QEJUN_to_D_start = dWrap(Date(freq='D', year=2006, month=7, day=1))
            date_QEJUN_to_D_end = dWrap(Date(freq='D', year=2006, month=9, day=30))

            date_QSJAN_to_D_start = dWrap(Date(freq='D', year=2007, month=2, day=1))
            date_QSJAN_to_D_end = dWrap(Date(freq='D', year=2007, month=4, day=30))

            date_QSJUN_to_D_start = dWrap(Date(freq='D', year=2007, month=7, day=1))
            date_QSJUN_to_D_end = dWrap(Date(freq='D', year=2007, month=9, day=30))

            date_QSDEC_to_D_start = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_QSDEC_to_D_end = dWrap(Date(freq='D', year=2007, month=3, day=31))

            assert_func(date_Q.asfreq('A'), date_Q_to_A)
            assert_func(date_Q_end_of_year.asfreq('A'), date_Q_to_A)

            assert_func(date_Q.asfreq('M', "START"), date_Q_to_M_start)
            assert_func(date_Q.asfreq('M', "END"), date_Q_to_M_end)
            assert_func(date_Q.asfreq('W', "START"), date_Q_to_W_start)
            assert_func(date_Q.asfreq('W', "END"), date_Q_to_W_end)
            assert_func(date_Q.asfreq('B', "START"), date_Q_to_B_start)
            assert_func(date_Q.asfreq('B', "END"), date_Q_to_B_end)
            assert_func(date_Q.asfreq('D', "START"), date_Q_to_D_start)
            assert_func(date_Q.asfreq('D', "END"), date_Q_to_D_end)
            assert_func(date_Q.asfreq('H', "START"), date_Q_to_H_start)
            assert_func(date_Q.asfreq('H', "END"), date_Q_to_H_end)
            assert_func(date_Q.asfreq('T', "START"), date_Q_to_T_start)
            assert_func(date_Q.asfreq('T', "END"), date_Q_to_T_end)
            assert_func(date_Q.asfreq('S', "START"), date_Q_to_S_start)
            assert_func(date_Q.asfreq('S', "END"), date_Q_to_S_end)

            assert_func(date_QEJAN.asfreq('D', "START"), date_QEJAN_to_D_start)
            assert_func(date_QEJAN.asfreq('D', "END"), date_QEJAN_to_D_end)
            assert_func(date_QEJUN.asfreq('D', "START"), date_QEJUN_to_D_start)
            assert_func(date_QEJUN.asfreq('D', "END"), date_QEJUN_to_D_end)

            assert_func(date_QSJAN.asfreq('D', "START"), date_QSJAN_to_D_start)
            assert_func(date_QSJAN.asfreq('D', "END"), date_QSJAN_to_D_end)
            assert_func(date_QSJUN.asfreq('D', "START"), date_QSJUN_to_D_start)
            assert_func(date_QSJUN.asfreq('D', "END"), date_QSJUN_to_D_end)
            assert_func(date_QSDEC.asfreq('D', "START"), date_QSDEC_to_D_start)
            assert_func(date_QSDEC.asfreq('D', "END"), date_QSDEC_to_D_end)

            assert_func(date_Q.asfreq('Q'), date_Q)


    def test_conv_monthly(self):
        "frequency conversion tests: from Monthly Frequency"

        for dWrap, assert_func in self.dateWrap:
            date_M = dWrap(Date(freq='M', year=2007, month=1))
            date_M_end_of_year = dWrap(Date(freq='M', year=2007, month=12))
            date_M_end_of_quarter = dWrap(Date(freq='M', year=2007, month=3))
            date_M_to_A = dWrap(Date(freq='A', year=2007))
            date_M_to_Q = dWrap(Date(freq='Q', year=2007, quarter=1))
            date_M_to_W_start = dWrap(Date(freq='W', year=2007, month=1, day=1))
            date_M_to_W_end = dWrap(Date(freq='W', year=2007, month=1, day=31))
            date_M_to_B_start = dWrap(Date(freq='B', year=2007, month=1, day=1))
            date_M_to_B_end = dWrap(Date(freq='B', year=2007, month=1, day=31))
            date_M_to_D_start = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_M_to_D_end = dWrap(Date(freq='D', year=2007, month=1, day=31))
            date_M_to_H_start = dWrap(Date(freq='H', year=2007, month=1, day=1,
                                      hour=0))
            date_M_to_H_end = dWrap(Date(freq='H', year=2007, month=1, day=31,
                                     hour=23))
            date_M_to_T_start = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                      hour=0, minute=0))
            date_M_to_T_end = dWrap(Date(freq='T', year=2007, month=1, day=31,
                                     hour=23, minute=59))
            date_M_to_S_start = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                      hour=0, minute=0, second=0))
            date_M_to_S_end = dWrap(Date(freq='S', year=2007, month=1, day=31,
                                     hour=23, minute=59, second=59))

            assert_func(date_M.asfreq('A'), date_M_to_A)
            assert_func(date_M_end_of_year.asfreq('A'), date_M_to_A)
            assert_func(date_M.asfreq('Q'), date_M_to_Q)
            assert_func(date_M_end_of_quarter.asfreq('Q'), date_M_to_Q)

            assert_func(date_M.asfreq('W', "START"), date_M_to_W_start)
            assert_func(date_M.asfreq('W', "END"), date_M_to_W_end)
            assert_func(date_M.asfreq('B', "START"), date_M_to_B_start)
            assert_func(date_M.asfreq('B', "END"), date_M_to_B_end)
            assert_func(date_M.asfreq('D', "START"), date_M_to_D_start)
            assert_func(date_M.asfreq('D', "END"), date_M_to_D_end)
            assert_func(date_M.asfreq('H', "START"), date_M_to_H_start)
            assert_func(date_M.asfreq('H', "END"), date_M_to_H_end)
            assert_func(date_M.asfreq('T', "START"), date_M_to_T_start)
            assert_func(date_M.asfreq('T', "END"), date_M_to_T_end)
            assert_func(date_M.asfreq('S', "START"), date_M_to_S_start)
            assert_func(date_M.asfreq('S', "END"), date_M_to_S_end)

            assert_func(date_M.asfreq('M'), date_M)


    def test_conv_weekly(self):
        "frequency conversion tests: from Weekly Frequency"

        for dWrap, assert_func in self.dateWrap:
            date_W = dWrap(Date(freq='W', year=2007, month=1, day=1))

            date_WSUN = dWrap(Date(freq='W-SUN', year=2007, month=1, day=7))
            date_WSAT = dWrap(Date(freq='W-SAT', year=2007, month=1, day=6))
            date_WFRI = dWrap(Date(freq='W-FRI', year=2007, month=1, day=5))
            date_WTHU = dWrap(Date(freq='W-THU', year=2007, month=1, day=4))
            date_WWED = dWrap(Date(freq='W-WED', year=2007, month=1, day=3))
            date_WTUE = dWrap(Date(freq='W-TUE', year=2007, month=1, day=2))
            date_WMON = dWrap(Date(freq='W-MON', year=2007, month=1, day=1))

            date_WSUN_to_D_start = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_WSUN_to_D_end = dWrap(Date(freq='D', year=2007, month=1, day=7))
            date_WSAT_to_D_start = dWrap(Date(freq='D', year=2006, month=12, day=31))
            date_WSAT_to_D_end = dWrap(Date(freq='D', year=2007, month=1, day=6))
            date_WFRI_to_D_start = dWrap(Date(freq='D', year=2006, month=12, day=30))
            date_WFRI_to_D_end = dWrap(Date(freq='D', year=2007, month=1, day=5))
            date_WTHU_to_D_start = dWrap(Date(freq='D', year=2006, month=12, day=29))
            date_WTHU_to_D_end = dWrap(Date(freq='D', year=2007, month=1, day=4))
            date_WWED_to_D_start = dWrap(Date(freq='D', year=2006, month=12, day=28))
            date_WWED_to_D_end = dWrap(Date(freq='D', year=2007, month=1, day=3))
            date_WTUE_to_D_start = dWrap(Date(freq='D', year=2006, month=12, day=27))
            date_WTUE_to_D_end = dWrap(Date(freq='D', year=2007, month=1, day=2))
            date_WMON_to_D_start = dWrap(Date(freq='D', year=2006, month=12, day=26))
            date_WMON_to_D_end = dWrap(Date(freq='D', year=2007, month=1, day=1))

            date_W_end_of_year = dWrap(Date(freq='W', year=2007, month=12, day=31))
            date_W_end_of_quarter = dWrap(Date(freq='W', year=2007, month=3, day=31))
            date_W_end_of_month = dWrap(Date(freq='W', year=2007, month=1, day=31))
            date_W_to_A = dWrap(Date(freq='A', year=2007))
            date_W_to_Q = dWrap(Date(freq='Q', year=2007, quarter=1))
            date_W_to_M = dWrap(Date(freq='M', year=2007, month=1))

            if Date(freq='D', year=2007, month=12, day=31).weekday == 6:
                date_W_to_A_end_of_year = dWrap(Date(freq='A', year=2007))
            else:
                date_W_to_A_end_of_year = dWrap(Date(freq='A', year=2008))

            if Date(freq='D', year=2007, month=3, day=31).weekday == 6:
                date_W_to_Q_end_of_quarter = dWrap(Date(freq='Q', year=2007, quarter=1))
            else:
                date_W_to_Q_end_of_quarter = dWrap(Date(freq='Q', year=2007, quarter=2))

            if Date(freq='D', year=2007, month=1, day=31).weekday == 6:
                date_W_to_M_end_of_month = dWrap(Date(freq='M', year=2007, month=1))
            else:
                date_W_to_M_end_of_month = dWrap(Date(freq='M', year=2007, month=2))

            date_W_to_B_start = dWrap(Date(freq='B', year=2007, month=1, day=1))
            date_W_to_B_end = dWrap(Date(freq='B', year=2007, month=1, day=5))
            date_W_to_D_start = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_W_to_D_end = dWrap(Date(freq='D', year=2007, month=1, day=7))
            date_W_to_H_start = dWrap(Date(freq='H', year=2007, month=1, day=1,
                                      hour=0))
            date_W_to_H_end = dWrap(Date(freq='H', year=2007, month=1, day=7,
                                     hour=23))
            date_W_to_T_start = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                      hour=0, minute=0))
            date_W_to_T_end = dWrap(Date(freq='T', year=2007, month=1, day=7,
                                     hour=23, minute=59))
            date_W_to_S_start = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                      hour=0, minute=0, second=0))
            date_W_to_S_end = dWrap(Date(freq='S', year=2007, month=1, day=7,
                                     hour=23, minute=59, second=59))

            assert_func(date_W.asfreq('A'), date_W_to_A)
            assert_func(date_W_end_of_year.asfreq('A'), date_W_to_A_end_of_year)
            assert_func(date_W.asfreq('Q'), date_W_to_Q)
            assert_func(date_W_end_of_quarter.asfreq('Q'), date_W_to_Q_end_of_quarter)
            assert_func(date_W.asfreq('M'), date_W_to_M)
            assert_func(date_W_end_of_month.asfreq('M'), date_W_to_M_end_of_month)

            assert_func(date_W.asfreq('B', "START"), date_W_to_B_start)
            assert_func(date_W.asfreq('B', "END"), date_W_to_B_end)

            assert_func(date_W.asfreq('D', "START"), date_W_to_D_start)
            assert_func(date_W.asfreq('D', "END"), date_W_to_D_end)

            assert_func(date_WSUN.asfreq('D', "START"), date_WSUN_to_D_start)
            assert_func(date_WSUN.asfreq('D', "END"), date_WSUN_to_D_end)
            assert_func(date_WSAT.asfreq('D', "START"), date_WSAT_to_D_start)
            assert_func(date_WSAT.asfreq('D', "END"), date_WSAT_to_D_end)
            assert_func(date_WFRI.asfreq('D', "START"), date_WFRI_to_D_start)
            assert_func(date_WFRI.asfreq('D', "END"), date_WFRI_to_D_end)
            assert_func(date_WTHU.asfreq('D', "START"), date_WTHU_to_D_start)
            assert_func(date_WTHU.asfreq('D', "END"), date_WTHU_to_D_end)
            assert_func(date_WWED.asfreq('D', "START"), date_WWED_to_D_start)
            assert_func(date_WWED.asfreq('D', "END"), date_WWED_to_D_end)
            assert_func(date_WTUE.asfreq('D', "START"), date_WTUE_to_D_start)
            assert_func(date_WTUE.asfreq('D', "END"), date_WTUE_to_D_end)
            assert_func(date_WMON.asfreq('D', "START"), date_WMON_to_D_start)
            assert_func(date_WMON.asfreq('D', "END"), date_WMON_to_D_end)

            assert_func(date_W.asfreq('H', "START"), date_W_to_H_start)
            assert_func(date_W.asfreq('H', "END"), date_W_to_H_end)
            assert_func(date_W.asfreq('T', "START"), date_W_to_T_start)
            assert_func(date_W.asfreq('T', "END"), date_W_to_T_end)
            assert_func(date_W.asfreq('S', "START"), date_W_to_S_start)
            assert_func(date_W.asfreq('S', "END"), date_W_to_S_end)

            assert_func(date_W.asfreq('W'), date_W)


    def test_conv_business(self):
        "frequency conversion tests: from Business Frequency"

        for dWrap, assert_func in self.dateWrap:
            date_B = dWrap(Date(freq='B', year=2007, month=1, day=1))
            date_B_end_of_year = dWrap(Date(freq='B', year=2007, month=12, day=31))
            date_B_end_of_quarter = dWrap(Date(freq='B', year=2007, month=3, day=30))
            date_B_end_of_month = dWrap(Date(freq='B', year=2007, month=1, day=31))
            date_B_end_of_week = dWrap(Date(freq='B', year=2007, month=1, day=5))

            date_B_to_A = dWrap(Date(freq='A', year=2007))
            date_B_to_Q = dWrap(Date(freq='Q', year=2007, quarter=1))
            date_B_to_M = dWrap(Date(freq='M', year=2007, month=1))
            date_B_to_W = dWrap(Date(freq='W', year=2007, month=1, day=7))
            date_B_to_D = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_B_to_H_start = dWrap(Date(freq='H', year=2007, month=1, day=1,
                                      hour=0))
            date_B_to_H_end = dWrap(Date(freq='H', year=2007, month=1, day=1,
                                     hour=23))
            date_B_to_T_start = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                      hour=0, minute=0))
            date_B_to_T_end = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                     hour=23, minute=59))
            date_B_to_S_start = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                      hour=0, minute=0, second=0))
            date_B_to_S_end = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                     hour=23, minute=59, second=59))

            assert_func(date_B.asfreq('A'), date_B_to_A)
            assert_func(date_B_end_of_year.asfreq('A'), date_B_to_A)
            assert_func(date_B.asfreq('Q'), date_B_to_Q)
            assert_func(date_B_end_of_quarter.asfreq('Q'), date_B_to_Q)
            assert_func(date_B.asfreq('M'), date_B_to_M)
            assert_func(date_B_end_of_month.asfreq('M'), date_B_to_M)
            assert_func(date_B.asfreq('W'), date_B_to_W)
            assert_func(date_B_end_of_week.asfreq('W'), date_B_to_W)

            assert_func(date_B.asfreq('D'), date_B_to_D)

            assert_func(date_B.asfreq('H', "START"), date_B_to_H_start)
            assert_func(date_B.asfreq('H', "END"), date_B_to_H_end)
            assert_func(date_B.asfreq('T', "START"), date_B_to_T_start)
            assert_func(date_B.asfreq('T', "END"), date_B_to_T_end)
            assert_func(date_B.asfreq('S', "START"), date_B_to_S_start)
            assert_func(date_B.asfreq('S', "END"), date_B_to_S_end)

            assert_func(date_B.asfreq('B'), date_B)


    def test_conv_daily(self):
        "frequency conversion tests: from Business Frequency"

        for dWrap, assert_func in self.dateWrap:
            date_D = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_D_end_of_year = dWrap(Date(freq='D', year=2007, month=12, day=31))
            date_D_end_of_quarter = dWrap(Date(freq='D', year=2007, month=3, day=31))
            date_D_end_of_month = dWrap(Date(freq='D', year=2007, month=1, day=31))
            date_D_end_of_week = dWrap(Date(freq='D', year=2007, month=1, day=7))

            date_D_friday = dWrap(Date(freq='D', year=2007, month=1, day=5))
            date_D_saturday = dWrap(Date(freq='D', year=2007, month=1, day=6))
            date_D_sunday = dWrap(Date(freq='D', year=2007, month=1, day=7))
            date_D_monday = dWrap(Date(freq='D', year=2007, month=1, day=8))

            date_B_friday = dWrap(Date(freq='B', year=2007, month=1, day=5))
            date_B_monday = dWrap(Date(freq='B', year=2007, month=1, day=8))

            date_D_to_A = dWrap(Date(freq='A', year=2007))

            date_Deoq_to_AJAN = dWrap(Date(freq='A-JAN', year=2008))
            date_Deoq_to_AJUN = dWrap(Date(freq='A-JUN', year=2007))
            date_Deoq_to_ADEC = dWrap(Date(freq='A-DEC', year=2007))

            date_D_to_QEJAN = dWrap(Date(freq=C.FR_QTREJAN, year=2007, quarter=4))
            date_D_to_QEJUN = dWrap(Date(freq=C.FR_QTREJUN, year=2007, quarter=3))
            date_D_to_QEDEC = dWrap(Date(freq=C.FR_QTREDEC, year=2007, quarter=1))

            date_D_to_QSJAN = dWrap(Date(freq=C.FR_QTRSJAN, year=2006, quarter=4))
            date_D_to_QSJUN = dWrap(Date(freq=C.FR_QTRSJUN, year=2006, quarter=3))
            date_D_to_QSDEC = dWrap(Date(freq=C.FR_QTRSDEC, year=2007, quarter=1))

            date_D_to_M = dWrap(Date(freq='M', year=2007, month=1))
            date_D_to_W = dWrap(Date(freq='W', year=2007, month=1, day=7))

            date_D_to_H_start = dWrap(Date(freq='H', year=2007, month=1, day=1,
                                      hour=0))
            date_D_to_H_end = dWrap(Date(freq='H', year=2007, month=1, day=1,
                                     hour=23))
            date_D_to_T_start = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                      hour=0, minute=0))
            date_D_to_T_end = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                     hour=23, minute=59))
            date_D_to_S_start = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                      hour=0, minute=0, second=0))
            date_D_to_S_end = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                     hour=23, minute=59, second=59))

            assert_func(date_D.asfreq('A'), date_D_to_A)

            assert_func(date_D_end_of_quarter.asfreq('A-JAN'), date_Deoq_to_AJAN)
            assert_func(date_D_end_of_quarter.asfreq('A-JUN'), date_Deoq_to_AJUN)
            assert_func(date_D_end_of_quarter.asfreq('A-DEC'), date_Deoq_to_ADEC)

            assert_func(date_D_end_of_year.asfreq('A'), date_D_to_A)
            assert_func(date_D_end_of_quarter.asfreq('Q'), date_D_to_QEDEC)
            assert_func(date_D.asfreq(C.FR_QTREJAN), date_D_to_QEJAN)
            assert_func(date_D.asfreq(C.FR_QTREJUN), date_D_to_QEJUN)
            assert_func(date_D.asfreq(C.FR_QTREDEC), date_D_to_QEDEC)
            assert_func(date_D.asfreq(C.FR_QTRSJAN), date_D_to_QSJAN)
            assert_func(date_D.asfreq(C.FR_QTRSJUN), date_D_to_QSJUN)
            assert_func(date_D.asfreq(C.FR_QTRSDEC), date_D_to_QSDEC)
            assert_func(date_D.asfreq('M'), date_D_to_M)
            assert_func(date_D_end_of_month.asfreq('M'), date_D_to_M)
            assert_func(date_D.asfreq('W'), date_D_to_W)
            assert_func(date_D_end_of_week.asfreq('W'), date_D_to_W)

            assert_func(date_D_friday.asfreq('B'), date_B_friday)
            assert_func(date_D_saturday.asfreq('B', "START"), date_B_friday)
            assert_func(date_D_saturday.asfreq('B', "END"), date_B_monday)
            assert_func(date_D_sunday.asfreq('B', "START"), date_B_friday)
            assert_func(date_D_sunday.asfreq('B', "END"), date_B_monday)

            assert_func(date_D.asfreq('H', "START"), date_D_to_H_start)
            assert_func(date_D.asfreq('H', "END"), date_D_to_H_end)
            assert_func(date_D.asfreq('T', "START"), date_D_to_T_start)
            assert_func(date_D.asfreq('T', "END"), date_D_to_T_end)
            assert_func(date_D.asfreq('S', "START"), date_D_to_S_start)
            assert_func(date_D.asfreq('S', "END"), date_D_to_S_end)

            assert_func(date_D.asfreq('D'), date_D)

    def test_conv_hourly(self):
        "frequency conversion tests: from Hourly Frequency"

        for dWrap, assert_func in self.dateWrap:
            date_H = dWrap(Date(freq='H', year=2007, month=1, day=1, hour=0))
            date_H_end_of_year = dWrap(Date(freq='H', year=2007, month=12, day=31,
                                      hour=23))
            date_H_end_of_quarter = dWrap(Date(freq='H', year=2007, month=3, day=31,
                                         hour=23))
            date_H_end_of_month = dWrap(Date(freq='H', year=2007, month=1, day=31,
                                       hour=23))
            date_H_end_of_week = dWrap(Date(freq='H', year=2007, month=1, day=7,
                                      hour=23))
            date_H_end_of_day = dWrap(Date(freq='H', year=2007, month=1, day=1,
                                     hour=23))
            date_H_end_of_bus = dWrap(Date(freq='H', year=2007, month=1, day=1,
                                     hour=23))

            date_H_to_A = dWrap(Date(freq='A', year=2007))
            date_H_to_Q = dWrap(Date(freq='Q', year=2007, quarter=1))
            date_H_to_M = dWrap(Date(freq='M', year=2007, month=1))
            date_H_to_W = dWrap(Date(freq='W', year=2007, month=1, day=7))
            date_H_to_D = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_H_to_B = dWrap(Date(freq='B', year=2007, month=1, day=1))

            date_H_to_T_start = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                      hour=0, minute=0))
            date_H_to_T_end = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                     hour=0, minute=59))
            date_H_to_S_start = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                      hour=0, minute=0, second=0))
            date_H_to_S_end = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                     hour=0, minute=59, second=59))

            assert_func(date_H.asfreq('A'), date_H_to_A)
            assert_func(date_H_end_of_year.asfreq('A'), date_H_to_A)
            assert_func(date_H.asfreq('Q'), date_H_to_Q)
            assert_func(date_H_end_of_quarter.asfreq('Q'), date_H_to_Q)
            assert_func(date_H.asfreq('M'), date_H_to_M)
            assert_func(date_H_end_of_month.asfreq('M'), date_H_to_M)
            assert_func(date_H.asfreq('W'), date_H_to_W)
            assert_func(date_H_end_of_week.asfreq('W'), date_H_to_W)
            assert_func(date_H.asfreq('D'), date_H_to_D)
            assert_func(date_H_end_of_day.asfreq('D'), date_H_to_D)
            assert_func(date_H.asfreq('B'), date_H_to_B)
            assert_func(date_H_end_of_bus.asfreq('B'), date_H_to_B)

            assert_func(date_H.asfreq('T', "START"), date_H_to_T_start)
            assert_func(date_H.asfreq('T', "END"), date_H_to_T_end)
            assert_func(date_H.asfreq('S', "START"), date_H_to_S_start)
            assert_func(date_H.asfreq('S', "END"), date_H_to_S_end)

            assert_func(date_H.asfreq('H'), date_H)

    def test_conv_minutely(self):
        "frequency conversion tests: from Minutely Frequency"

        for dWrap, assert_func in self.dateWrap:
            date_T = dWrap(Date(freq='T', year=2007, month=1, day=1,
                          hour=0, minute=0))
            date_T_end_of_year = dWrap(Date(freq='T', year=2007, month=12, day=31,
                                      hour=23, minute=59))
            date_T_end_of_quarter = dWrap(Date(freq='T', year=2007, month=3, day=31,
                                         hour=23, minute=59))
            date_T_end_of_month = dWrap(Date(freq='T', year=2007, month=1, day=31,
                                       hour=23, minute=59))
            date_T_end_of_week = dWrap(Date(freq='T', year=2007, month=1, day=7,
                                      hour=23, minute=59))
            date_T_end_of_day = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                     hour=23, minute=59))
            date_T_end_of_bus = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                     hour=23, minute=59))
            date_T_end_of_hour = dWrap(Date(freq='T', year=2007, month=1, day=1,
                                      hour=0, minute=59))

            date_T_to_A = dWrap(Date(freq='A', year=2007))
            date_T_to_Q = dWrap(Date(freq='Q', year=2007, quarter=1))
            date_T_to_M = dWrap(Date(freq='M', year=2007, month=1))
            date_T_to_W = dWrap(Date(freq='W', year=2007, month=1, day=7))
            date_T_to_D = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_T_to_B = dWrap(Date(freq='B', year=2007, month=1, day=1))
            date_T_to_H = dWrap(Date(freq='H', year=2007, month=1, day=1, hour=0))

            date_T_to_S_start = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                      hour=0, minute=0, second=0))
            date_T_to_S_end = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                     hour=0, minute=0, second=59))

            assert_func(date_T.asfreq('A'), date_T_to_A)
            assert_func(date_T_end_of_year.asfreq('A'), date_T_to_A)
            assert_func(date_T.asfreq('Q'), date_T_to_Q)
            assert_func(date_T_end_of_quarter.asfreq('Q'), date_T_to_Q)
            assert_func(date_T.asfreq('M'), date_T_to_M)
            assert_func(date_T_end_of_month.asfreq('M'), date_T_to_M)
            assert_func(date_T.asfreq('W'), date_T_to_W)
            assert_func(date_T_end_of_week.asfreq('W'), date_T_to_W)
            assert_func(date_T.asfreq('D'), date_T_to_D)
            assert_func(date_T_end_of_day.asfreq('D'), date_T_to_D)
            assert_func(date_T.asfreq('B'), date_T_to_B)
            assert_func(date_T_end_of_bus.asfreq('B'), date_T_to_B)
            assert_func(date_T.asfreq('H'), date_T_to_H)
            assert_func(date_T_end_of_hour.asfreq('H'), date_T_to_H)

            assert_func(date_T.asfreq('S', "START"), date_T_to_S_start)
            assert_func(date_T.asfreq('S', "END"), date_T_to_S_end)

            assert_func(date_T.asfreq('T'), date_T)

    def test_conv_secondly(self):
        "frequency conversion tests: from Secondly Frequency"

        for dWrap, assert_func in self.dateWrap:
            date_S = dWrap(Date(freq='S', year=2007, month=1, day=1,
                          hour=0, minute=0, second=0))
            date_S_end_of_year = dWrap(Date(freq='S', year=2007, month=12, day=31,
                                      hour=23, minute=59, second=59))
            date_S_end_of_quarter = dWrap(Date(freq='S', year=2007, month=3, day=31,
                                         hour=23, minute=59, second=59))
            date_S_end_of_month = dWrap(Date(freq='S', year=2007, month=1, day=31,
                                       hour=23, minute=59, second=59))
            date_S_end_of_week = dWrap(Date(freq='S', year=2007, month=1, day=7,
                                      hour=23, minute=59, second=59))
            date_S_end_of_day = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                     hour=23, minute=59, second=59))
            date_S_end_of_bus = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                     hour=23, minute=59, second=59))
            date_S_end_of_hour = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                      hour=0, minute=59, second=59))
            date_S_end_of_minute = dWrap(Date(freq='S', year=2007, month=1, day=1,
                                        hour=0, minute=0, second=59))

            date_S_to_A = dWrap(Date(freq='A', year=2007))
            date_S_to_Q = dWrap(Date(freq='Q', year=2007, quarter=1))
            date_S_to_M = dWrap(Date(freq='M', year=2007, month=1))
            date_S_to_W = dWrap(Date(freq='W', year=2007, month=1, day=7))
            date_S_to_D = dWrap(Date(freq='D', year=2007, month=1, day=1))
            date_S_to_B = dWrap(Date(freq='B', year=2007, month=1, day=1))
            date_S_to_H = dWrap(Date(freq='H', year=2007, month=1, day=1,
                               hour=0))
            date_S_to_T = dWrap(Date(freq='T', year=2007, month=1, day=1,
                               hour=0, minute=0))

            assert_func(date_S.asfreq('A'), date_S_to_A)
            assert_func(date_S_end_of_year.asfreq('A'), date_S_to_A)
            assert_func(date_S.asfreq('Q'), date_S_to_Q)
            assert_func(date_S_end_of_quarter.asfreq('Q'), date_S_to_Q)
            assert_func(date_S.asfreq('M'), date_S_to_M)
            assert_func(date_S_end_of_month.asfreq('M'), date_S_to_M)
            assert_func(date_S.asfreq('W'), date_S_to_W)
            assert_func(date_S_end_of_week.asfreq('W'), date_S_to_W)
            assert_func(date_S.asfreq('D'), date_S_to_D)
            assert_func(date_S_end_of_day.asfreq('D'), date_S_to_D)
            assert_func(date_S.asfreq('B'), date_S_to_B)
            assert_func(date_S_end_of_bus.asfreq('B'), date_S_to_B)
            assert_func(date_S.asfreq('H'), date_S_to_H)
            assert_func(date_S_end_of_hour.asfreq('H'), date_S_to_H)
            assert_func(date_S.asfreq('T'), date_S_to_T)
            assert_func(date_S_end_of_minute.asfreq('T'), date_S_to_T)

            assert_func(date_S.asfreq('S'), date_S)

    def test_convert_to_float_daily(self):
        "Test convert_to_float on daily data"
        dbase = ts.date_array(start_date=ts.Date('D', '2007-01-01'),
                              end_date=ts.Date('D', '2008-12-31'),
                              freq='D')
        # D -> A
        test = convert_to_float(dbase, 'A')
        assert_equal(test, np.r_[2007 + np.arange(365) / 365.,
                                 2008 + np.arange(366) / 366.])
        # D -> M
        test = convert_to_float(dbase, 'M')
        control = []
        for (m, d) in zip(ts.date_array(length=24,
                                       start_date=ts.Date('M', '2007-01')),
                         [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                          31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, ]):
            control.extend([m.value + np.arange(d) / float(d)])
        control = np.concatenate(control)
        assert_equal(test, control)


    def test_convert_to_float_monthly(self):
        "Test convert_to_float on daily data"
        mbase = ts.date_array(start_date=ts.Date('M', '2007-01'),
                              end_date=ts.Date('M', '2009-01',),
                              freq='M')
        # M -> A
        test = convert_to_float(mbase, 'A')
        assert_equal(test, 2007 + np.arange(25) / 12.)
        # M -> Q


    def test_convert_to_float_quarterly(self):
        "Test convert_to_float on daily data"
        qbase = ts.date_array(start_date=ts.Date('Q', year=2007, quarter=1),
                              end_date=ts.Date('Q', year=2008, quarter=4),
                              freq='Q')
        # Q -> A
        test = convert_to_float(qbase, 'A')
        assert_equal(test, 2007 + np.arange(8) / 4.)



class TestMethods(TestCase):
    "Base test class for MaskedArrays."

    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)

    def test_getitem(self):
        "Tests getitem"
        dlist = ['2007-%02i' % i for i in range(1, 5) + range(7, 13)]
        mdates = date_array(dlist, freq='M')
        # Using an integer
        assert_equal(mdates[0].value, 24073)
        assert_equal(mdates[-1].value, 24084)
        # Using a date
        lag = mdates.find_dates(mdates[0])
        assert_equal(mdates[lag][0], mdates[0])
        lag = mdates.find_dates(Date('M', value=24080))
        assert_equal(mdates[lag][0], mdates[5])
        # Using several dates
        lag = mdates.find_dates(Date(freq='M', value=24073), Date(freq='M', value=24084))
        assert_equal(mdates[lag],
                     DateArray([mdates[0], mdates[-1]], freq='M'))
        assert_equal(mdates[[mdates[0], mdates[-1]]], mdates[lag])
        #
        assert_equal(mdates >= mdates[-4], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1])


    def test_getitem_solodates(self):
        "Check that using a Date as index returns a Date"
        dates = ts.date_array(start_date=Date('M', '2009-01'), length=4)
        test = dates[dates[-1]]
        self.failUnless(isinstance(test, Date))


    def test_add(self):
        dt1 = Date(freq='D', year=2008, month=1, day=1)
        dt2 = Date(freq='D', year=2008, month=1, day=2)
        assert_equal(dt1 + 1, dt2)
        assert_equal(dt1 + 1.0, dt2)
        assert_equal(dt1 + np.int16(1), dt2)
        assert_equal(dt1 + np.int32(1), dt2)
        assert_equal(dt1 + np.int64(1), dt2)
        assert_equal(dt1 + np.float32(1), dt2)
        assert_equal(dt1 + np.float64(1), dt2)
        #
        try:
            temp = dt1 + "str"
        except TypeError:
            pass
        else:
            raise RuntimeError("invalid add type passed")
        #
        try:
            temp = dt1 + dt2
        except TypeError:
            pass
        else:
            raise RuntimeError("invalid add type passed")


    def test_getsteps(self):
        "Tests the getsteps method"
        dlist = ['2007-01-%02i' % i for i in (1, 2, 3, 4, 8, 9, 10, 11, 12, 15)]
        ddates = date_array(dlist, freq='D')
        assert_equal(ddates.get_steps(), [1, 1, 1, 4, 1, 1, 1, 1, 3])


    def test_empty_datearray(self):
        empty_darray = DateArray([], freq='Business')
        assert_equal(empty_darray.is_full(), True)
        assert_equal(empty_darray.is_valid(), True)
        assert_equal(empty_darray.get_steps(), None)


    def test_cachedinfo(self):
        D = date_array(start_date=now('D'), length=5)
        Dstr = D.tostring()
        assert_equal(D.tostring(), Dstr)
        DL = D[[0, -1]]
        assert_equal(DL.tostring(), Dstr[[0, -1]])


    def test_date_to_index_valid(self):
        "Tests date_to_index"
        dates = date_array(['2007-01-%02i' % i for i in range(1, 16)], freq='D')
        choices = date_array(['2007-01-03', '2007-01-05', '2007-01-07', ] + \
                             ['2007-01-09', '2007-02-01'], freq='D')
        #
        chosen = dates.date_to_index(choices[0])
        assert_equal(chosen, 2)
        self.failUnless(isinstance(chosen, int))
        #
        chosen = dates.date_to_index(choices[:-1])
        assert_equal(chosen, [2, 4, 6, 8])
        self.failUnless(isinstance(chosen, np.ndarray))
        #
        try:
            assert_equal(dates.date_to_index(choices), [2, 4, 6, 8, -99])
        except (IndexError, ValueError):
            pass
        else:
            raise IndexError("An invalid indexed has been accepted !")

        # test behaviour of date_to_index on DateArray of length 1
        chosen = dates.date_to_index(choices[0:1])
        assert_equal(chosen, [2])


    def test_contains(self):
        dt = ts.now('d')
        darr = date_array(start_date=dt, length=5)
        self.failUnless(dt in darr)
        self.failUnless(dt - 1 not in darr)
        self.failUnless(dt.value in darr)
        self.failUnless((dt - 1).value not in darr)
        #
        try:
            ts.now('b') in darr
        except ValueError:
            pass
        else:
            raise RuntimeError("containment of wrong frequency permitted")


    def test_date_to_index_invalid(self):
        "Tests date_to_index"
        dates_invalid = date_array(['2007-01-%02i' % i for i in range(1, 11)] + \
                                   ['2007-01-%02i' % i for i in range(15, 20)],
                                   freq='D')
        choices = date_array(['2007-01-03', '2007-01-05', '2007-01-07', ] + \
                             ['2007-01-09', '2007-02-01'], freq='D')
        #
        chosen = dates_invalid.date_to_index(choices[0])
        assert_equal(chosen, 2)
        self.failUnless(isinstance(chosen, int))
        #
        chosen = dates_invalid.date_to_index(choices[:-1])
        assert_equal(chosen, [2, 4, 6, 8])
        self.failUnless(isinstance(chosen, np.ndarray))
        #
        try:
            assert_equal(dates_invalid.date_to_index(choices),
                         [2, 4, 6, 8, -99])
        except (IndexError, ValueError):
            pass
        else:
            raise IndexError("An invalid indexed has been accepted !")


    def test_argsort(self):
        "Test argsort"
        dates = date_array(['2001-03', '2001-02', '2001-01'],
                           freq='M', autosort=False)
        test = dates.argsort()
        self.failUnless(isinstance(test, np.ndarray))
        self.failUnless(not isinstance(test, DateArray))
        assert_equal(test, [2, 1, 0])

    def test_sort_wcached(self):

        "Test cache update w/ sorting"
        dates = ts.DateArray([2002, 2000, 2001, 2002], freq='A')
        assert_equal(dates.is_chronological(), False)
        assert_equal(dates.has_duplicated_dates(), True)
        # Sort w/ function
        sorted_dates = np.sort(dates)
        assert_equal(sorted_dates.is_chronological(), True)
        assert_equal(sorted_dates.has_duplicated_dates(), True)
        assert_equal(dates.is_chronological(), False)
        # Sort w/ method
        dates.sort()
        assert_equal(dates.is_chronological(), True)
        assert_equal(dates.has_duplicated_dates(), True)
        # Sort 2D data w/ function
        dates = ts.DateArray([[2002, 2000], [2001, 2002]], freq='A')
        assert_equal(dates.is_chronological(), False)
        assert_equal(dates.has_duplicated_dates(), True)
        sorted_dates = np.sort(dates)
        assert_equal(sorted_dates.is_chronological(), False)
        assert_equal(sorted_dates.has_duplicated_dates(), True)
        # Sort 2D data w/ method
        dates.sort()
        assert_equal(dates.is_chronological(), False)
        assert_equal(dates.has_duplicated_dates(), True)

    def test_minmax(self):
        "Test min and max on DateArrays"
        start_date = Date("M", "2001-01")
        dates = date_array(start_date=start_date, length=12)
        dates.shape = (2, 6)
        test = dates.min()
        assert_equal(test, start_date)
        assert(isinstance(test, Date))
        test = dates.max()
        assert(isinstance(test, Date))
        assert_equal(test, start_date + 11)
        #
        test = dates.min(axis=0)
        assert_equal(test, dates[0])
        assert(isinstance(test, DateArray))
        test = dates.max(axis=0)
        assert_equal(test, dates[-1])
        assert(isinstance(test, DateArray))


def test_pickling():
    "Tests pickling DateArrays"
    import cPickle
    base = date_array(start_date=ts.now('D'), length=7)
    target = cPickle.loads(cPickle.dumps(base))
    assert_equal(base.freq, target.freq)
    assert_equal(base, target)


###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    run_module_suite()
