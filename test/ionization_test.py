import pytest
import photoionization_base
import numpy as np
from scipy import constants as C

class TestIonizationRates:

    def test_LLHydrogen(self):
        ionizer = photoionization_base.Hydrogen(False,0.5,1.0,0,0,0,0)
        assert ionizer.Rate(0.1-ionizer.cutoff_field) == pytest.approx(0.0509,1e-3)
    def test_ADK(self):
        # test against line entry computation of ADK Eq. 21 correcting typo in exponential (n**4 -> n**3)
        au = photoionization_base.AtomicUnits()
        Uion = au.energy_to_au(photoionization_base.ip['He'][0])
        ionizer = photoionization_base.ADK(True,Uion,1.0)
        assert ionizer.Rate(0.1-ionizer.cutoff_field) == pytest.approx(4.546e-7,1e-2)