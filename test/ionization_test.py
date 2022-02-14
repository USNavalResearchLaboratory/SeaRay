import pytest
import photoionization_base

class TestIonization:

    def test_LLHydrogen(self):
        ionizer = photoionization_base.Hydrogen(False,0.5,1.0,0,0,0,0)
        assert ionizer.Rate(0.1-ionizer.cutoff_field) == pytest.approx(0.0509,1e-3)
