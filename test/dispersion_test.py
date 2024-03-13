import pytest
import numpy as np
from scipy import constants as C
import modules.dispersion as dispersion

class TestSpotCheckSolids:
    # Most of the data points from https://refractiveindex.info
    l1 = 1.0e-6/(2*np.pi)
    w1 = C.c/l1
    def test_bk7(self):
        mat = dispersion.BK7(self.l1)
        data = [(.532e-6,1.5195),(.6328e-6,1.51502),(1.064e-6,1.5066)]
        for d in data:
            w = 2*np.pi*C.c/d[0]
            n = np.sqrt(1+mat.chi(w/self.w1))
            assert n == pytest.approx(d[1],1e-4)
    def test_germanium(self):
        mat = dispersion.Ge(self.l1)
        data = [(2e-6,4.1085),(5e-6,4.0158),(10e-6,4.0040)]
        for d in data:
            w = 2*np.pi*C.c/d[0]
            n = np.sqrt(1+mat.chi(w/self.w1))
            assert n == pytest.approx(d[1],1e-4)
    def test_znse(self):
        mat = dispersion.ZnSe(self.l1)
        data = [(2.5e-6,2.4362),(5e-6,2.4132),(10e-6,2.3923)]
        for d in data:
            w = 2*np.pi*C.c/d[0]
            n = np.sqrt(1+mat.chi(w/self.w1))
            assert n == pytest.approx(d[1],1e-2)
    def test_nacl(self):
        mat = dispersion.NaCl(self.l1)
        data = [(2.5e-6,1.5330),(5e-6,1.5240),(10e-6,1.5000)]
        for d in data:
            w = 2*np.pi*C.c/d[0]
            n = np.sqrt(1+mat.chi(w/self.w1))
            assert n == pytest.approx(d[1],1e-2)
    def test_kcl(self):
        mat = dispersion.KCl(self.l1)
        data = [(2.5e-6,1.4780),(5e-6,1.4720),(10e-6,1.4580)]
        for d in data:
            w = 2*np.pi*C.c/d[0]
            n = np.sqrt(1+mat.chi(w/self.w1))
            assert n == pytest.approx(d[1],1e-2)
