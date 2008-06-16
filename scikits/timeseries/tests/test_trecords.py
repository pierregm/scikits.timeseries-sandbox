# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for trecords.

:author: Pierre Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu  & mattknox_ca_at_hotmail_dot_com
:version: $Id: test_trecords.py 3836 2008-01-15 13:09:03Z matthew.brett@gmail.com $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: matthew.brett@gmail.com $)"
__version__ = '1.0'
__revision__ = "$Revision: 3836 $"
__date__     = '$Date: 2008-01-15 08:09:03 -0500 (Tue, 15 Jan 2008) $'

import types

import numpy as np
import numpy.core.records as nr
import numpy.ma as ma
from numpy.ma import nomask
import numpy.ma.mrecords as mr
from numpy.ma.mrecords import addfield

from scipy.testing import *
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

    def test_get(self):
        "Tests fields retrieval"
        [d, m, mrec, dlist, dates, mts, rts] = self.data
        assert(isinstance(rts['f0'], TimeSeries))
        assert_equal(rts['f0']._dates, dates)
        assert_equal(rts['f0']._data, d)
        assert_equal(rts['f0']._mask, m)
        #
        assert(isinstance(rts[0], TimeSeriesRecords))
        assert_equal(rts._data[0], mrec._data[0])
        # We can't use assert_equal here, as it tries to convert the tuple into a singleton
#        assert(rts[0]._data.view(numpyndarray) == mrec[0])
        assert_equal_records(rts._data[0], mrec[0])
        assert_equal(rts._dates[0], dates[0])
        assert_equal(rts[0]._dates, dates[0])
        #
        assert(isinstance(rts['2007-01'], TimeSeriesRecords))
        assert_equal_records(rts['2007-01']._data, mrec[0])
        assert_equal(rts['2007-01']._dates, dates[0])
        #
        assert(isinstance(rts.f0, TimeSeries))
        assert_equal(rts.f0, time_series(d, dates=dates, mask=m))
        assert_equal(rts.f1, time_series(d[::-1], dates=dates, mask=m[::-1]))
        assert((rts._fieldmask == nr.fromarrays([m, m[::-1]])).all())
        # Was _mask, now is recordmask
        assert_equal(rts.recordmask, np.r_[[m,m[::-1]]].all(0))
        assert_equal(rts.f0[1], rts[1].f0)
        #
        assert(isinstance(rts[:2], TimeSeriesRecords))
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
        rts._mask = ma.masked
        assert_equal(ma.getmaskarray(rts['f1']), [1]*5)
        assert_equal(rts['f0']._mask, rts['f1']._mask)
        rts._mask = ma.nomask
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
        assert(rts._hardmask)
        rts._mask = nomask
        # Was _mask, now is recordmask
        assert_equal(rts.recordmask, np.r_[[m,m[::-1]]].all(0))
        rts.soften_mask()
        assert(not rts._hardmask)
        rts._mask = nomask
        assert(rts['f1']._mask is nomask)
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
        mrectxt = fromtextfile(tmp_fl, delimitor=',', varnames='ABCDEFG')
        os.remove(tmp_fl)
        #
        dlist = ['2007-%02i' % i for i in (1,2,3,5)]
        assert(isinstance(mrectxt, TimeSeriesRecords))
        assert_equal(mrectxt._dates, date_array(dlist,'M'))
        assert_equal(mrectxt.dtype.names, ['B','C','D','E','F','G'])
        assert_equal(mrectxt.G, [1,1,1,1])
        assert_equal(mrectxt.F._mask, [1,1,1,1])
        assert_equal(mrectxt.D, [1,2,3.e+5,-1e-10])


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
        

###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    nose.run(argv=['', __file__])
