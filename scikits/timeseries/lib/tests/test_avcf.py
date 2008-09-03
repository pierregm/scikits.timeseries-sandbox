"""
Test suite for the ACVF module

:author: Pierre GF Gerard-Marchant
:contact: pierregm_at_uga_edu
:date: $Date: 2006-12-20 03:24:40 -0500 (Wed, 20 Dec 2006) $
:version: $Id: timeseries.py 55 2006-12-20 08:24:40Z backtopop $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: backtopop $)"
__revision__ = "$Revision: 55 $"
__date__     = '$Date: 2006-12-20 03:24:40 -0500 (Wed, 20 Dec 2006) $'

import numpy as np
from numpy import nan
import numpy.ma as ma

from scikits.timeseries.lib.avcf import avf, acf, cvf, ccf, pacf

from numpy.testing import *
from numpy.ma.testutils import assert_almost_equal



class TestCorrel(TestCase):
    def __init__(self,*args,**kwargs):
        TestCase.__init__(self,*args,**kwargs)
        self.presidents = [nan, 87, 82, 75, 63, 50, 43, 32, 35, 60, 54, 55,
                           36, 39,nan,nan, 69, 57, 57, 51, 45, 37, 46, 39,
                           36, 24, 32, 23, 25, 32,nan, 32, 59, 74, 75, 60,
                           71, 61, 71, 57, 71, 68, 79, 73, 76, 71, 67, 75,
                           79, 62, 63, 57, 60, 49, 48, 52, 57, 62, 61, 66,
                           71, 62, 61, 57, 72, 83, 71, 78, 79, 71, 62, 74,
                           76, 64, 62, 57, 80, 73, 69, 69, 71, 64, 69, 62,
                           63, 46, 56, 44, 44, 52, 38, 46, 36, 49, 35, 44,
                           59, 65, 65, 56, 66, 53, 61, 52, 51, 48, 54, 49,
                           49, 61,nan,nan, 68, 44, 40, 27, 28, 25, 24, 24]
        self.mdeaths = [2134,1863,1877,1877,1492,1249,1280,1131,1209,1492,1621,
                        1846,2103,2137,2153,1833,1403,1288,1186,1133,1053,1347,
                        1545,2066,2020,2750,2283,1479,1189,1160,1113, 970, 999,
                        1208,1467,2059,2240,1634,1722,1801,1246,1162,1087,1013,
                         959,1179,1229,1655,2019,2284,1942,1423,1340,1187,1098,
                        1004, 970,1140,1110,1812,2263,1820,1846,1531,1215,1075,
                        1056, 975, 940,1081,1294,1341]
        self.fdeaths = [901, 689, 827, 677, 522, 406, 441, 393, 387, 582, 578,
                        666, 830, 752, 785, 664, 467, 438, 421, 412, 343, 440,
                        531, 771, 767,1141, 896, 532, 447, 420, 376, 330, 357,
                        445, 546, 764, 862, 660, 663, 643, 502, 392, 411, 348,
                        387, 385, 411, 638, 796, 853, 737, 546, 530, 446, 431,
                        362, 387, 430, 425, 679, 821, 785, 727, 612, 478, 429,
                        405, 379, 393, 411, 487, 574]
        self.mdeaths = ma.asarray(self.mdeaths)
        self.fdeaths = ma.asarray(self.fdeaths)

    #
    def test_avf_nomasked(self):
        "Tests avf - no masked values"
        n = 100
        x = np.random.rand(n)
        x = x - x.mean()
        # periodogram = True
        avfx = avf(x)
        assert_almost_equal(avfx[0], x.var(ddof=0))
        assert_almost_equal(avfx[:n],
                            np.r_[float(x.var()),
                                  np.fromiter(((x[k:]*x[:-k]).sum()/n
                                               for k in range(1,n)), float)])
        # periodogram = False
        avfx = avf(x,False)
        assert_almost_equal(avfx[0], x.var(ddof=0))
        assert_almost_equal(avfx[:n],
                            np.r_[float(x.var()),
                                  np.fromiter(((x[k:]*x[:-k]).mean()
                                               for k in range(1,n)), float)])
    #......................................................
    def test_avf_masked(self):
        presidents = ma.fix_invalid(self.presidents)
        # periodogram : True
        avfp = avf(presidents)
        assert_almost_equal(avfp[:21].round(2),
                            [241.74,185.75,159.63,116.92, 95.91, 60.36, 45.69,
                              34.97, 31.74, 10.91,  7.48,  1.32, 11.70,  7.71,
                              13.57,  4.16, -1.05, -9.76,-11.24,-15.67,-12.32])
        # preiodogram : False
        avfp = avf(presidents,0)
        pz = presidents.anom()
        mz = (~pz.mask).astype(int)
        assert_almost_equal(avfp[:21],
                            np.r_[[pz.var()],
                                  [(pz[k:]*pz[:-k]).sum()/(mz[k:]*mz[:-k]).sum()
                                   for k in range(1,21)]])
    #......................................................
    def test_cvf_nomasked(self):
        (mdeaths, fdeaths) = (self.mdeaths, self.fdeaths)
        cvfm = cvf(mdeaths,mdeaths)
        assert_almost_equal(avf(mdeaths), cvfm)
        cvfmf = cvf(mdeaths,fdeaths)
        assert_almost_equal(cvfmf[:16].round(),
                            [ 74941, 56473, 27961, -820,-29395,-47777,-52855,
                             -46871,-29427, -1390, 30090, 50403,55378, 49061,
                             27739,-260])
        assert_almost_equal(cvfmf[-15:].round(),
                            [  1156, 28067, 47243, 54365, 47740, 26100, -1883,
                             -29299,-46964,-52031,-46368,-26827,  1517, 31105,
                              57137,])
    #......................................................
    def test_cvf_masked(self):
        (mdeaths, fdeaths) = (self.mdeaths, self.fdeaths)
        (mdeaths_, fdeaths_) = (mdeaths.copy(), fdeaths.copy())
        mdeaths_[0] = fdeaths_[-1] = ma.masked
        cvfmf = cvf(mdeaths_,fdeaths_)
        assert_almost_equal(cvfmf[:16].round(4),
                            [ 73964.4731, 56471.7150, 27967.4713,  -795.2985,
                             -29350.6550,-47710.5560,-52768.6081,-46767.4760,
                             -29311.7816, -1280.9791, 30179.0824, 50463.3681,
                              55404.6871, 49071.1267, 27764.8371,  -219.5959])
        assert_almost_equal(cvfmf[-15:].round(4),
                            [   206.5786, 26763.5676, 46821.7325, 53326.3888,
                              47955.9913, 26600.5188, -2221.0872,-28578.8452,
                             -46765.5598,-52410.7168,-46279.6304,-27242.8170,
                                498.1511, 29551.1878, 57576.9793])
    #......................................................
    def test_pacf(self):
        mdeaths = self.mdeaths
        pacfm = pacf(mdeaths)
        assert_almost_equal(pacfm[:19].round(3),
                            [ 0.000, 0.757,-0.401,-0.238,-0.389,-0.174,-0.100,
                             -0.108, 0.045, 0.226, 0.332, 0.085, 0.034, 0.138,
                             -0.046,-0.059, 0.068, 0.148, 0.061])
        mdeaths_ = mdeaths.copy()
        mdeaths_[0] = ma.masked
        pacfm = pacf(mdeaths_)
        assert_almost_equal(pacfm[:19].round(3),
                            [ 0.000, 0.763,-0.445,-0.229,-0.359,-0.183,-0.132,
                             -0.095, 0.017, 0.217, 0.338, 0.043, 0.041, 0.166,
                             -0.099,-0.006, 0.114, 0.084, 0.038])

if __name__ == '__main__':
    run_module_suite()