import pytest
import modules.photoionization_base as izbase
import numpy as np
from scipy import constants as C

class TestIonizationRates:

    def test_LLHydrogen(self):
        ionizer = izbase.Hydrogen(False,0.5,1.0,0,0,0,0)
        assert ionizer.Rate(0.1-ionizer.cutoff_field) == pytest.approx(0.0509,1e-3)
    def test_ADK(self):
        # test against line entry computation of ADK Eq. 21 correcting typo in exponential (n**4 -> n**3)
        au = izbase.AtomicUnits()
        Uion = au.energy_to_au(izbase.ip['He'][0])
        ionizer = izbase.ADK(True,Uion,1.0)
        assert ionizer.Rate(0.1-ionizer.cutoff_field) == pytest.approx(4.546e-7,1e-2)