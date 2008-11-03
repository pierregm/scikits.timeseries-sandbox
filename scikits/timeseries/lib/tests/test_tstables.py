"""
Test suite for tstables
"""

import os
import tempfile

import numpy as np
import numpy.ma as ma
import numpy.ma.mrecords as mr
from numpy.ma import MaskedArray, masked_array, masked

import scikits.timeseries as ts
from scikits.timeseries import TimeSeries

from numpy.testing import *
from numpy.ma.testutils import assert_equal, assert_equal_records

try:
    import tables
    has_tables = True
except ImportError:
    has_tables = False
import scikits.timeseries.lib.tstables as tstables
tabulate = tstables.tabulate




def common_ma_setup():
    data2D = ma.array([np.random.rand(25).reshape(5,5),
                       np.random.rand(25).reshape(5,5),
                       np.random.rand(25).reshape(5,5),
                       np.random.rand(25).reshape(5,5),
                       np.random.rand(25).reshape(5,5),],
                       mask=[np.random.rand(25).reshape(5,5)>.5,
                             np.random.rand(25).reshape(5,5)>.5,
                             np.random.rand(25).reshape(5,5)>.5,
                             np.random.rand(25).reshape(5,5)>.5,
                             np.random.rand(25).reshape(5,5)>.5,]
                      ) 
    data1D = ma.array(np.random.rand(25),
                      mask=np.random.rand(25)>0.9,
                      fill_value=-9999)
    dtype5R = [('a',float),('b',int),('c','|S3')]
    data5N = ma.array(zip(np.random.rand(5),
                          np.arange(5),
                          'ABCDE'),
                      dtype=dtype5R)
    data5R = mr.fromarrays([np.random.rand(5),
                            np.arange(5),
                            ('A','B','C','D','E')],
                           dtype=dtype5R)
    data5R._mask['a'][0]=True
    data5R._mask['b'][2]=True
    data5R._mask['c'][-1]=True
    return dict(data1D=data1D, 
                data2D=data2D,
                data5N=data5N,
                data5R=data5R)



class TestTabulate(TestCase):
    #
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.data = common_ma_setup()
    #
    def test_tabulate1D(self):
        "Tests the transformation from a 1D MaskedArray to..."
        # Test 1D
        data1D = self.data['data1D']
        _data1D = tabulate(data1D)
        assert_equal(_data1D['_data'], data1D.filled())
        assert_equal(_data1D['_mask'], data1D.mask)
    #
    def test_tabulate2D(self):
        "Tests the transformation from a nD series to..."
        data2D = self.data['data2D']
        _data2D = tabulate(data2D)
        assert_equal(_data2D['_data'], data2D.filled().reshape(len(data2D),-1))
        assert_equal(_data2D['_mask'], data2D._mask.reshape(len(data2D),-1))
    #
    def test_tabulateNV(self):
        "Tests the transformation of named variables to..."
        data5N = self.data['data5N']
        _data5N = tabulate(data5N)
        zipped5N = [((ma.filled(s['a']), ma.getmaskarray(s['a'])),
                     (ma.filled(s['b']), ma.getmaskarray(s['b'])),
                     (ma.filled(s['c']), ma.getmaskarray(s['c']))) 
                    for s in data5N]
        ndtype5N = [(fname,[('_data',ftype), ('_mask',bool)])
                    for (fname,ftype) in data5N.dtype.descr]
        data5N = np.array(zipped5N, dtype=ndtype5N)
        for key in ('a','b','c'):
            assert_equal_records(data5N[key].view(np.recarray), 
                                 _data5N[key].view(np.recarray))
        #
        _data5R = tabulate(self.data['data5N'])
        data5R = np.array(zipped5N, dtype=ndtype5N)
        for key in ('a','b','c'):
            assert_equal_records(data5R[key].view(np.recarray), 
                                 _data5R[key].view(np.recarray))



class TestMaskedTable(TestCase):
    #
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.data = common_ma_setup()
        self.file = tempfile.mktemp(".hdf5")
        self.fileh = tables.openFile(self.file,'a')
        self.populate()
    #
    def tearDown(self):
        if self.fileh.isopen:
            self.fileh.close()
        os.remove(self.file)
    #
    def populate(self):
        h5file = self.fileh
        for (key,data) in self.data.iteritems():
            table = h5file.createMaskedTable('/', key, data, "Example")
            h5file.flush()
    #
    def test_read1D(self):
        data1D = self.data['data1D']
        table = self.fileh.root.data1D
        tarray = table.read()
        
        assert_equal(tarray.dtype, data1D.dtype)
        assert_equal(tarray.filled(), data1D.filled())
        assert_equal(tarray._mask, data1D._mask)
        assert_equal(tarray.fill_value, data1D.fill_value)
        #
        tarray = table.read(1,5,2)
        control = data1D[1:5:2]
        assert_equal(tarray, control)
        #
        tarray = table.read(field='_data')
        assert_equal(tarray, data1D.filled())
    #
    def test_read2D(self):
        data2D = self.data['data2D']
        table = self.fileh.root.data2D
        tarray = table.read()
        assert_equal(tarray.dtype, data2D.dtype)
        assert_equal(tarray.fill_value, data2D.fill_value)
        assert_equal(tarray.filled(), data2D.filled())
        assert_equal(tarray._mask, data2D._mask)
        #
        tarray = table.read(1,5,2)
        control = (data2D.reshape(len(data2D),-1)[1:5:2]).reshape(-1,5,5)
        assert_equal(tarray.mask, control.mask)
        assert_equal(tarray.filled(), control.filled())
        #
        tarray = table.read(field='_data')
        assert_equal(tarray, data2D.filled())
    #
    def test_read5N(self):
        data5N = self.data['data5N']
        table = self.fileh.root.data5N
        tarray = table.read()
        assert_equal(tarray.dtype, data5N.dtype)
        for f in ('a','b','c'):
            assert_equal(tarray[f], data5N[f])
        assert_equal(tarray._mask, data5N._mask)
        assert_equal(tarray.fill_value, data5N.fill_value)
        #
        tarray = table.read(1,5,2)
        initial = data5N[1:5:2]
        for f in ('a','b','c'):
            assert_equal(tarray[f], initial[f])
        #
        tarray = table.read(field='a')
        assert_equal(tarray, data5N['a'])
        #
        data5R = self.data['data5R']
        table = self.fileh.root.data5R
        tarray = table.read(field='a')
        assert_equal(tarray, data5R['a'])
        assert_equal(tarray._mask, data5R['a']._mask)



def common_ts_setup():
    series2D = ts.time_series([np.random.rand(25).reshape(5,5),
                               np.random.rand(25).reshape(5,5),
                               np.random.rand(25).reshape(5,5),
                               np.random.rand(25).reshape(5,5),
                               np.random.rand(25).reshape(5,5),],
                              start_date=ts.now('M'),
                              mask=[np.random.rand(25).reshape(5,5)>.5,
                                    np.random.rand(25).reshape(5,5)>.5,
                                    np.random.rand(25).reshape(5,5)>.5,
                                    np.random.rand(25).reshape(5,5)>.5,
                                    np.random.rand(25).reshape(5,5)>.5,]
                             ) 
    series1D = ts.time_series(np.random.rand(25),
                              mask=np.random.rand(25)>0.7,
                              start_date=ts.now('M'),
                              fill_value=-999)
    series5V = ts.time_series(np.random.rand(25).reshape(5,5),
                              mask=np.random.rand(25).reshape(5,5)>0.7,
                              start_date=ts.now('M'))
    series5N = ts.time_series(zip(np.random.rand(5),
                                  np.random.rand(5),
                                  np.arange(5)),
                              start_date=ts.now('M'),
                              dtype=[('a',float),('b',float),('c',int)]
                              )
    return dict(series1D=series1D, 
                series5V=series5V,
                series2D=series2D,
                series5N=series5N)


class TestTimeSeriesTable(TestCase):
    #
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.data = common_ts_setup()
        self.file = tempfile.mktemp(".hdf5")
        self.fileh = tables.openFile(self.file,'a')
        self.populate()
    #
    def tearDown(self):
        if self.fileh.isopen:
            self.fileh.close()
        os.remove(self.file)
    #
    def populate(self):
        h5file = self.fileh
        for (key,data) in self.data.iteritems():
            table = h5file.createTimeSeriesTable('/', key, data, "Example")
            h5file.flush()
    #
    def test_read1D(self):
        series1D = self.data['series1D']
        table = self.fileh.root.series1D
        tarray = table.read()
        assert_equal(tarray.dtype, series1D.dtype)
        assert_equal(tarray, series1D)
        assert_equal(tarray._mask, series1D._mask)
        assert_equal(tarray._dates, series1D._dates)
        assert_equal(tarray.fill_value, series1D.fill_value)
        #
        tarray = table.read(1,5,2)
        assert_equal(tarray, series1D[1:5:2])
        #
        tarray = table.read(field='_data')
        assert_equal(tarray, series1D.filled())
        #
        tarray = table.read(field='_dates')
        assert_equal(tarray, series1D.dates)
    #
    def test_read2D(self):
        series2D = self.data['series2D']
        table = self.fileh.root.series2D
        tarray = table.read()
        assert_equal(tarray.dtype, series2D.dtype)
        assert_equal(tarray, series2D)
        assert_equal(tarray._mask, series2D._mask)
        assert_equal(tarray.fill_value, series2D.fill_value)
        #
        tarray = table.read(1,5,2)
        initial = series2D[1:5:2]
        assert_equal(tarray, initial._series.reshape(2,5,5))
        assert_equal(tarray._dates, initial._dates)
        #
        tarray = table.read(field='_series')
        assert_equal(tarray, series2D._series)
    #
    def test_read5N(self):
        series5N = self.data['series5N']
        table = self.fileh.root.series5N
        tarray = table.read()
        assert_equal(tarray.dtype, series5N.dtype)
        for f in ('a','b','c'):
            assert_equal(tarray[f], series5N[f])
        assert_equal(tarray._mask, series5N._mask)
        assert_equal(tarray.fill_value, series5N.fill_value)
        #
        tarray = table.read(1,5,2)
        initial = series5N[1:5:2]
        for f in ('a','b','c'):
            assert_equal(tarray[f], initial[f])
        #
        tarray = table.read(field='a')
        assert_equal(tarray, series5N['a'])
        #
        series5V = self.data['series5V']
        table = self.fileh.root.series5V
        tarray = table.read()
        assert_equal(tarray, series5V)
        assert_equal(tarray._mask, series5V._mask)


class TestSpecialAttrs(TestCase):
    #
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        self.marray = ma.array(np.random.rand(100).reshape(10,10),
                               mask = (np.random.rand(100).reshape(10,10) > 0.7),
                               fill_value=-999,
                               hard_mask=True)
        self.marray._optinfo['memo'] = "Some line of text"
        self.tseries = ts.time_series(self.marray,
                                      start_date=ts.now('D'))
        self.file = tempfile.mktemp(".hdf5")
        self.fileh = tables.openFile(self.file,'a')
        self.populate()
    #
    def tearDown(self):
        if self.fileh.isopen:
            self.fileh.close()
        os.remove(self.file)
    #
    def populate(self):
        h5file = self.fileh
        table = h5file.createMaskedTable('/', 'marray', self.marray, "")
        h5file.flush()
        table = h5file.createTimeSeriesTable('/', 'tseries', self.tseries, "")
        h5file.flush()
    #
    def test_specialattributes_maskedtable(self):
        data = self.marray
        table = self.fileh.root.marray
        test = table.read()
        assert_equal(test._optinfo, data._optinfo)
        assert_equal(test._hardmask, data._hardmask)
    #
    def test_specialattributes_timeseriestable(self):
        data = self.tseries
        table = self.fileh.root.tseries
        test = table.read()
        assert_equal(test._optinfo, data._optinfo)
        assert_equal(test._hardmask, data._hardmask)


class TestTableRead(TestCase):
    #
    def __init__(self, *args, **kwds):
        TestCase.__init__(self, *args, **kwds)
        series = ts.time_series(zip(np.random.rand(10),
                                    np.arange(10)),
                                start_date=ts.now('M'),
                                dtype=[('a',float),('b',int)])
        series.mask[0] = (0,1)
        series.mask[-1] = (1,0)
        self.tseries = series
        self.marray = series._series
        self.file = tempfile.mktemp(".hdf5")
        self.h5file = tables.openFile(self.file,'a')
        self.populate()
    #
    def tearDown(self):
        if self.h5file.isopen:
            self.h5file.close()
        os.remove(self.file)
    #
    def populate(self):
        h5file = self.h5file
        table = h5file.createMaskedTable('/', 'marray', self.marray, "")
        h5file.flush()
        table = h5file.createTimeSeriesTable('/', 'tseries', self.tseries, "")
        h5file.flush()
    #
    def test_tseries_read(self):
        "Test reading specific elements of a TimeSeriesTable"
        table = self.h5file.root.tseries
        series = self.tseries
        #
        test = table.read()
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal_records(test, series)
        #
        test = table.read(field='a')
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal(test, series['a'])
        #
        test = table.read(step=2)
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal(test, series[::2])
        #
        test = table.readCoordinates([1,2,3])
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal(test, series[[1,2,3]])
        #
        test = table.readCoordinates([1,2,3], field='a')
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal(test, series['a'][[1,2,3]])
    #
    def test_marray_read(self):
        "Test reading specific elements of a MaskedTable"
        table = self.h5file.root.marray
        data = self.marray
        #
        test = table.read()
        self.failUnless(isinstance(test, MaskedArray))
        assert_equal_records(test, data)
        #
        test = table.read(field='a')
        self.failUnless(isinstance(test, MaskedArray))
        assert_equal(test, data['a'])
        #
        test = table.read(step=2)
        self.failUnless(isinstance(test, MaskedArray))
        assert_equal(test, data[::2])
        #
        test = table.readCoordinates([1,2,3])
        self.failUnless(isinstance(test, MaskedArray))
        assert_equal(test, data[[1,2,3]])
        #
        test = table.readCoordinates([1,2,3], field='a')
        self.failUnless(isinstance(test, MaskedArray))
        assert_equal(test, data['a'][[1,2,3]])
    #
    def test_append_maskedarray(self):
        "Test appending to a MaskedTable"
        table = self.h5file.root.marray
        data = self.marray
        newdata = ma.array(zip(np.random.rand(3), np.arange(3)+10),
                           mask=[(0,0),(1,0),(0,1)],
                           dtype=data.dtype)
        table.append(newdata)
        test = table.read()
        self.failUnless(isinstance(test, MaskedArray))
        assert_equal_records(test, ma.mr_[data,newdata])
    #
    def test_append_timeseries(self):
        "Test appending to a MaskedTable"
        table = self.h5file.root.tseries
        tseries = self.tseries
        newdata = ts.time_series(zip(np.random.rand(3), np.arange(3)+10),
                           mask=[(0,0),(1,0),(0,1)],
                           dtype=tseries.dtype,
                           start_date=tseries.dates[-1]+1)
        table.append(newdata)
        test = table.read()
        self.failUnless(isinstance(test, TimeSeries))
        assert_equal_records(test, ts.concatenate((tseries,newdata)))
    #

###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    if has_tables:
        run_module_suite()
