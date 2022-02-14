import pytest
import numpy as np
import init
import ray_kernel

class TestAction:
    bdx = 1e-4
    def test_initial_action(self):
        cl,args = init.setup_opencl(['rays.py','run'])
        xp = np.array([[
            [0,0,0,0,1,0,0,1],
            [0,-self.bdx,0,0,1,0,0,1],
            [0,+self.bdx,0,0,1,0,0,1],
            [0,0,-self.bdx,0,1,0,0,1],
            [0,0,+self.bdx,0,1,0,0,1],
            [0,0,0,-self.bdx,1,0,0,1],
            [0,0,0,+self.bdx,1,0,0,1],
        ]])
        vg = np.array([[
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
        ]])
        eikonal = np.array([[0,1,0,0]])
        assert ray_kernel.GetMicroAction(xp,eikonal,vg) == pytest.approx(8*self.bdx**3,1e-4)
