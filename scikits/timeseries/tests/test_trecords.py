# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for trecords.

:author: Pierre Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu  & mattknox_ca_at_hotmail_dot_com
:version: $Id: test_trecords.py 3836 2008-01-15 13:09:03Z matthew.brett@gmail.com $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: matthew.brett@gmail.com $)"
__revision__ = "$Revision: 3836 $"
__date__     = '$Date: 2008-01-15 08:09:03 -0500 (Tue, 15 Jan 2008) $'

import types

import numpy as np
import numpy.core.records as nr
import numpy.ma as ma
from numpy.ma import nomask
import numpy.ma.mrecords as mr
from numpy.ma.mrecords import addfield

from numpy.testing import *
from numpy.ma.testutils import assert_equal, assert_array_equal, assert_equal_records

import scikits.timeseries as ts
from scikits.timeseries.trecords import \
    TimeSeriesRecords, TimeSeries,\
    fromarrays, fromtextfile, fromrecords, \
    date_array, time_series, time_records


#..............................................................................
class TestTimeSeriesRecords(TestCase):
    "Base test class for MaskedArrays."
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.setup()

    def setup(self):
        "Generic setup"
        d = np.arange(5)
        m = ma.make_mask([1,0,0,1,1])
        base_d = np.r_[d,d[::-1]].reshape(2,-1).T
        base_m = np.r_[[m, m[::-1]]].T
        base = ma.array(base_d, mask=base_m)
        mrec = mr.fromarrays(base.T,)
        dlist = ['2007-%02i' % (i+1) for i in d]
        dates = date_array(dlist)
        mts = time_series(mrec,dates)
        rts = time_records(mrec,dates)
        self.data = [d, m, mrec, dlist, dates, mts, rts]

    def test_get_fields(self):
        "Tests fields retrieval"
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        self.failUnless(isinstance(rts['f0'], TimeSeries))
        self.failUnless(not isinstance(rts['f0'], TimeSeriesRecords))
        assert_equal(rts['f0']._dates, dates)
        assert_equal(rts['f0']._data, d)
        assert_equal(rts['f0']._mask, m)

    def test_get_index(self):
        "Tests index retrieval"
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        test = rts[0]
        self.failUnless(not (isinstance(test, TimeSeriesRecords)))
        assert_equal_records(test, rts._data[0])
        assert_equal_records(test, mrec._data[0])
        # We can't use assert_equal here, as it tries to convert the tuple into a singleton
#        self.failUnless(rts[0]._data.view(numpyndarray) == mrec[0])
        assert_equal_records(rts._data[0], mrec[0])
        assert_equal(rts.mask[0], mrec.mask[0])
        assert_equal(test.mask, rts.mask[0])

    def test_get_dates(self):
        "Tests date retrieval"
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        # Get item by dates w/ masked values
        test = rts['2007-01']
        self.failUnless(isinstance(test, ma.MaskedArray))
        self.failUnless(not isinstance(test, TimeSeries))
        assert_equal_records(test, mrec[0])
        # Get item by dates w/o masked values
        test = rts['2007-03']
        self.failUnless(isinstance(test, np.void))
        assert_equal_records(test, mrec[2])

    def test_get_field_asattribute(self):
        "Tests item retrieval"
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        self.failUnless(isinstance(rts.f0, TimeSeries))
        self.failUnless(not isinstance(rts[0], TimeSeriesRecords))
        assert_equal(rts.f0, time_series(d, dates=dates, mask=m))
        assert_equal(rts.f1, time_series(d[::-1], dates=dates, mask=m[::-1]))
        self.failUnless((rts._mask == nr.fromarrays([m, m[::-1]])).all())
        # Was _mask, now is recordmask
        assert_equal(rts.recordmask, np.r_[[m,m[::-1]]].all(0))
        assert_equal(rts.f0[1], rts[1].f0)

    def test_get_slice(self):
        "Tests item retrieval"
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        self.failUnless(isinstance(rts[:2], TimeSeriesRecords))
        assert_equal(rts[:2]._data.f0, mrec[:2].f0)
        assert_equal(rts[:2]._data.f1, mrec[:2].f1)
        assert_equal(rts[:2]._dates, dates[:2])


    def test_set(self):
        "Tests setting fields/attributes."
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        rts.f0._data[:] = 5
        assert_equal(rts['f0']._data, [5,5,5,5,5])
        rts.f0 = 1
        assert_equal(rts['f0']._data, [1]*5)
        assert_equal(ma.getmaskarray(rts['f0']), [0]*5)
        rts.f1 = ma.masked
        assert_equal(rts.f1.mask, [1]*5)
        assert_equal(ma.getmaskarray(rts['f1']), [1]*5)
        rts.mask = ma.masked
        assert_equal(ma.getmaskarray(rts['f1']), [1]*5)
        assert_equal(rts['f0']._mask, rts['f1']._mask)
        rts.mask = ma.nomask
        assert_equal(ma.getmaskarray(rts['f1']), [0]*5)
        assert_equal(rts['f0']._mask, rts['f1']._mask)

    def test_set_field_mask(self):
        "Tests setting fields/attributes."
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        rts.f0[1] = ma.masked
        assert_equal(rts.f0._mask, [1,1,0,1,1])

    def test_setslices(self):
        "Tests setting slices."
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        #
        try:
            rts[:2] = 5
        except TypeError:
            pass
        else:
            raise TypeError("Should have expected a readable buffer object.")
        rts[:2] = (5,5)
        assert_equal(rts.f0._data, [5,5,2,3,4])
        assert_equal(rts.f1._data, [5,5,2,1,0])
        assert_equal(rts.f0._mask, [0,0,0,1,1])
        assert_equal(rts.f1._mask, [0,0,0,0,1])
        #
        rts.harden_mask()
        try:
            rts[-2:] = (5,5)
            assert_equal(rts.f0._data, [5,5,2,3,4])
            assert_equal(rts.f1._data, [5,5,2,5,0])
            assert_equal(rts.f0._mask, [0,0,0,1,1])
            assert_equal(rts.f1._mask, [0,0,0,0,1])
        except NotImplementedError:
            pass
        except AssertionError:
            raise
        else:
            raise Exception("Flexible hard masks should be supported")

    def test_hardmask(self):
        "Test hardmask"
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        rts.harden_mask()
        self.failUnless(rts._hardmask)
        rts.mask = nomask
        # Was _mask, now is recordmask
        assert_equal(rts.recordmask, np.r_[[m,m[::-1]]].all(0))
        rts.soften_mask()
        self.failUnless(not rts._hardmask)
        rts.mask = nomask
        self.failUnless(rts['f1']._mask is nomask)
        assert_equal(rts['f0']._mask,rts['f1']._mask)

    def test_addfield(self):
        "Tests addfield"
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        rts = addfield(rts, ma.array(d+10, mask=m[::-1]))
        assert_equal(rts.f2, d+10)
        assert_equal(rts.f2._mask, m[::-1])

    def test_fromrecords(self):
        "Test from recarray."
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        nrec = nr.fromarrays(np.r_[[d,d[::-1]]])
        mrecfr = fromrecords(nrec.tolist(), dates=dates)
        assert_equal(mrecfr.f0, mrec.f0)
        assert_equal(mrecfr.dtype, mrec.dtype)
        #....................
        altrec = [tuple([d,]+list(r)) for (d,r) in zip(dlist,nrec)]
        mrecfr = fromrecords(altrec, names='dates,f0,f1')
        assert_equal(mrecfr.f0, mrec.f0)
        assert_equal(mrecfr.dtype, mrec.dtype)
        #....................
        tmp = time_records(rts._series[::-1], dates=rts.dates)
        mrecfr = fromrecords(tmp)
        assert_equal(mrecfr.f0, mrec.f0[::-1])
        #....................
        mrecfr = fromrecords(mrec.data, dates=dates, mask=m)
        assert_equal(mrecfr.recordmask, m)

    def test_fromtextfile(self):
        "Tests reading from a text file."
        fcontent = """#
'Dates', 'One (S)','Two (I)','Three (F)','Four (M)','Five (-)','Six (C)'
'2007-01', 'strings',1,1.0,'mixed column',,1
'2007-02', 'with embedded "double quotes"',2,2.0,1.0,,1
'2007-03', 'strings',3,3.0E5,3,,1
'2007-05','strings',4,-1e-10,,,1
"""
        import os
        from datetime import datetime
        import tempfile
        (tmp_fd,tmp_fl) = tempfile.mkstemp()
        os.write(tmp_fd, fcontent)
        os.close(tmp_fd)
        mrectxt = fromtextfile(tmp_fl, delimitor=',', varnames='ABCDEFG',
                               dates_column=0, skiprows=2)
        os.remove(tmp_fl)
        #
        dlist = ['2007-%02i' % i for i in (1,2,3,5)]
        self.failUnless(isinstance(mrectxt, TimeSeriesRecords))
        assert_equal(mrectxt._dates, date_array(dlist,'M'))
        assert_equal(mrectxt.dtype.names, ['B','C','D','E','F','G'])
        assert_equal(mrectxt.G, [1,1,1,1])
        assert_equal(mrectxt.F._mask, [1,1,1,1])
        assert_equal(mrectxt.D, [1,2,3.e+5,-1e-10])
    #
    def test_sorted(self):
        dates = [ts.Date('D',string='2007-01-%02i' % i) for i in (3,2,1)]
        (a,b) = zip(*[(3.,30), (2.,20), (1.,10),])
        ndtype = [('a', np.float), ('b',np.int)]
        controldates = date_array(dates, freq='D')
        series = time_series(zip(*(a,b)), dates, freq='D',dtype=ndtype)
        assert_equal(series._data.tolist(), [(1.,10), (2.,20), (3.,30)])
        assert_equal(series._dates, controldates)
        #
        trec = time_records(zip(*(a,b)), dates, freq='D',dtype=ndtype)
        assert_equal(trec._data.tolist(), [(1.,10), (2.,20), (3.,30)])
        assert_equal(trec._dates, controldates)
        assert_equal(trec['a'], [1., 2., 3.])
        assert_equal(trec.a, [1., 2., 3.])
        #
        trec = fromrecords(zip(a,b), dates, names=('a','b'))
        assert_equal(trec._data.tolist(), [(1.,10), (2.,20), (3.,30)])
        assert_equal(trec._dates, controldates)
        assert_equal(trec['a'], [1., 2., 3.])
        assert_equal(trec.a, [1., 2., 3.])
        #
        trec = fromarrays([a,b], dates, names=('a','b'))
        assert_equal(trec._data.tolist(), [(1.,10), (2.,20), (3.,30)])
        assert_equal(trec._dates, controldates)
        assert_equal(trec['a'], [1., 2., 3.])
        assert_equal(trec.a, [1., 2., 3.])


#..............................................................................
class TestTimeSeriesRecords_Functions(TestCase):
    "Base test class for MaskedArrays."
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.setup()
    #
    def setup(self):
        a = time_series(np.random.rand(24),
                        start_date=ts.now('M'))
        b = time_series(np.random.rand(24)*100, dtype=int,
                        start_date=ts.now('M'),)
#        c = time_series(["%02i" % _ for _ in np.arange(24)],
#                         start_date=ts.now('M'))
        c = time_series(np.arange(24),
                         start_date=ts.now('M'))
        trec = fromarrays([a,b,c], dates=a.dates, names='a,b,c')
        self.info = (a,b,c,trec)
    #
    def test_convert(self):
        #
        (a, b, c,trec) = self.info
        base = dict(a=a, b=b, c=c)
        a_trec = trec.convert('A',ma.mean)
        # Don't convert the dtype by themselves, that won't work...
        # ... as ma.mean will convert ints to floats, eg.
        assert_equal(a_trec.dtype.names, trec.dtype.names)
        for key in ('a','b','c'):
            assert_equal(a_trec[key], base[key].convert('A', ma.mean))


class TestViewTS(TestCase):
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
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal_records(test, tarr)
        assert_equal_records(test._mask, tarr._mask)
    #
    def test_view_simple_dtype(self):
        (tarr, a, b) = self.data
        ntype = (np.float, 2)
        test = tarr.view(ntype)
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal(test, np.array(zip(a,b), dtype=np.float))
        self.failUnless(test[3,1] is ma.masked)
    #
    def test_view_flexible_type(self):
        (tarr, a, b) = self.data
        arr = tarr._series
        alttype = [('A',np.float), ('B',np.float)]
        test = tarr.view(alttype)
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal_records(test, arr.view(alttype))
        self.failUnless(test['B'][3] is ma.masked)
        assert_equal(test.dtype, np.dtype(alttype))
        self.failUnless(test._fill_value is None)

###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    run_module_suite()

