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
        ]]).astype(np.double)
        vg = np.array([[
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
        ]]).astype(np.double)
        eikonal = np.array([[0,1,0,0]]).astype(np.double)
        assert ray_kernel.GetMicroAction(xp,eikonal,vg) == pytest.approx(8*self.bdx**3,1e-4)
    def test_ballistic_propagation(self):
        cl,args = init.setup_opencl(['rays.py','run'])
        divAngle = 0.5
        sq = np.sin(divAngle)
        cq = np.cos(divAngle)
        ds = np.array([[6.0,6.0,6.0,6.0,6.0,6.0,6.0]]).astype(np.double)
        xp = np.array([[
            [0,0,0,0,1,0,0,1],
            [0,-self.bdx,0,0,1,-sq,0,cq],
            [0,+self.bdx,0,0,1,+sq,0,cq],
            [0,0,-self.bdx,0,1,0,-sq,cq],
            [0,0,+self.bdx,0,1,0,+sq,cq],
            [0,0,0,-self.bdx,1,0,0,1],
            [0,0,0,+self.bdx,1,0,0,1],
        ]]).astype(np.double)
        vg = np.array([[
            [1,0,0,1],
            [1,-sq,0,cq],
            [1,+sq,0,cq],
            [1,0,-sq,cq],
            [1,0,+sq,cq],
            [1,0,0,1],
            [1,0,0,1],
        ]]).astype(np.double)
        eikonal = np.array([[0,1,0,0]]).astype(np.double)
        assert ray_kernel.GetMicroAction(xp,eikonal,vg) == pytest.approx(8*self.bdx**3,1e-4)
        ray_kernel.FullStep(ds,xp,eikonal,vg)
        for i in range(1,4):
            assert xp[0,:,i] == pytest.approx(vg[0,:,i]*ds[0,:],1e-4)
        assert ray_kernel.GetMicroAction(xp,eikonal,vg) == pytest.approx(8*self.bdx**3,1e-4)
