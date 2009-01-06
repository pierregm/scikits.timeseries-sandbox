# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for MaskedArray.
Adapted from the original test_ma by Pierre Gerard-Marchant

:author: Pierre Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu & mattknox_ca_at_hotmail_dot_com
:version: $Id: test_extras.py 3836 2008-01-15 13:09:03Z matthew.brett@gmail.com $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: matthew.brett@gmail.com $)"
__revision__ = "$Revision: 3836 $"
__date__     = '$Date: 2008-01-15 08:09:03 -0500 (Tue, 15 Jan 2008) $'

import StringIO

import numpy as np
from numpy.testing import *
from numpy.ma import masked
from numpy.ma.testutils import assert_equal, assert_almost_equal

from scikits.timeseries import Date, TimeSeries, date_array, time_series
import scikits.timeseries.extras
from scikits.timeseries.extras import *

#..............................................................................
class TestMisc(TestCase):
    "Base test class for MaskedArrays."
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
    #
    def test_leapyear(self):
        leap = isleapyear([1900,1901,1902,1903,1904,2000,2001,2002,2003,2004])
        assert_equal(leap, [0,0,0,0,1,1,0,0,0,1])

#..............................................................................
class TestCountmissing(TestCase):
    #
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        data = time_series(np.arange(731),
                           start_date=Date(string='2003-01-01', freq='D'),
                           freq='D')
        self.data = data

    def test_count_missing(self):
        data = self.data
        assert_equal(count_missing(data), 0)
        assert_equal(count_missing(data.convert('A')), (0,0))
        assert_equal(count_missing(data.convert('M')), [0]*24)
        #
        series = data.copy()
        series[np.logical_not(data.day % 10)] = masked
        assert_equal(count_missing(series), 70)
        assert_equal(count_missing(series.convert('A')), (35,35))
        assert_equal(count_missing(series.convert('M')),
                     [3,2,3,3,3,3,3,3,3,3,3,3]*2)
        #
        series[series.day == 31] = masked
        assert_equal(count_missing(series), 84)
        assert_equal(count_missing(series.convert('A')), (42,42))
        assert_equal(count_missing(series.convert('M')),
                     [4,2,4,3,4,3,4,4,3,4,3,4]*2)
    #
    def test_accept_atmost_missing(self):
        series = self.data.copy()
        series[np.logical_not(self.data.day % 10)] = masked
        result = accept_atmost_missing(series.convert('M'),3,True)
        assert_equal(result._mask.all(-1), [0]*24)
        result = accept_atmost_missing(series.convert('M'),3,False)
        assert_equal(result._mask.all(-1), [1,0,1,1,1,1,1,1,1,1,1,1]*2)
        result = accept_atmost_missing(series.convert('M'),0.1,True)
        assert_equal(result._mask.all(-1), [0]*24)
        result = accept_atmost_missing(series.convert('A'),35,True)
        assert_equal(result._mask.all(-1), [0,0])
        result = accept_atmost_missing(series.convert('A'),35,False)
        assert_equal(result._mask.all(-1), [1,1])
        result = accept_atmost_missing(series.convert('A'),0.05,True)
        assert_equal(result._mask.all(-1), [1,1])



class TestFromTxt(TestCase):
    "Test tsfromtxt"
    #
    def test_nodateinfo(self):
        # No dates column specified: crash.
        "Test no date info"
        fcontent = StringIO.StringIO("""#
'Dates', 'One (S)','Two (I)','Three (F)','Four (M)','Five (-)','Six (C)'
'2007-01', 'strings',1,1.0,'mixed column',,1
'2007-02', 'with embedded "double quotes"',2,2.0,1.0,,1
'2007-03', 'strings',3,3.0E5,3,,1
'2007-05','strings',4,-1e-10,,,1
""")
        try:
            test = tsfromtxt(fcontent, delimiter=",", names="A,B,C,D,E,F,G")
        except TypeError:
            pass
    #
    def test_with_names(self):
        "Tests w/ names"
        fcontent = StringIO.StringIO("""#
'Dates', 'One (S)','Two (I)','Three (F)','Four (M)','Five (-)','Six (C)'
'2007-01', 'strings',1,1.0,'mixed column',,1
'2007-02', 'with embedded "double quotes"',2,2.0,1.0,,1
'2007-03', 'strings',3,3.0E5,3,,1
'2007-05','strings',4,-1e-10,,,1
""")
        test = tsfromtxt(fcontent, delimiter=",", datecols=0, skiprows=2,
                         names="A,B,C,D,E,F,G", freq='M')
        assert(isinstance(test, TimeSeries))
        dlist = ['2007-%02i' % i for i in (1,2,3,5)]
        assert_equal(test.dates.tovalue(), date_array(dlist,freq='M').tovalue())
        assert_equal(test.dtype.names, ['B','C','D','E','F','G'])
        assert_equal(test['G'], [1,1,1,1])
        assert_equal(test['F'].mask, [1,1,1,1])
        assert_equal(test['D'], [1,2,3.e+5,-1e-10])
    #
    def test_without_names(self):
        "Test w/o names"
        fcontent = StringIO.StringIO("""#
'Dates', 'One (S)','Two (I)','Three (F)','Four (M)','Five (-)','Six (C)'
'2007-01', 'strings',1,1.0,'mixed column',,1
'2007-02', 'with embedded "double quotes"',2,2.0,1.0,,1
'2007-03', 'strings',3,3.0E5,3,,1
'2007-05','strings',4,-1e-10,,,1
""")
        test = tsfromtxt(fcontent, delimiter=",", skiprows=1, names=True,
                         freq='M')
        assert(isinstance(test, TimeSeries))
        dlist = ['2007-%02i' % i for i in (1,2,3,5)]
        assert_equal(test.dates.tovalue(), date_array(dlist,freq='M').tovalue())
        assert_equal(test.dtype.names,
                     ['One_S', 'Two_I', 'Three_F', 'Four_M', 'Five_', 'Six_C'])
        assert_equal(test['Six_C'], [1,1,1,1])
        assert_equal(test['Five_'].mask, [1,1,1,1])
        assert_equal(test['Three_F'], [1,2,3.e+5,-1e-10])
    #
    def test_with_datecols(self):
        "Test two datecols"
        fcontent = StringIO.StringIO("""
year, month, A, B
2009, 01, 1, 1.
2009, 03, 3, 3.
""")
        dateconv = lambda y, m: Date("M", year=int(y), month=int(m))
        test = tsfromtxt(fcontent, delimiter=",", skiprows=1, names=True,
                         converters={'dates': dateconv}, datecols=(0,1))
        dates = date_array(['2009-01', '2009-03'], freq='M')
        assert_equal(test.dates.tovalue(), dates)
        assert_equal(test['A'], [1, 3])
        assert_equal(test['B'], [1., 3.])
        assert_equal(test.dtype, np.dtype([('A', int), ('B', float)]))



###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    run_module_suite()

