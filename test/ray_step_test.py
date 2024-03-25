import pytest
import numpy as np

import sys
sys.path.append('modules')
import modules.init as init
import modules.ray_kernel as ray_kernel

class TestAction:
    # length of edge of regular tetrahedron
    a = 1e-4
    # the actual volume is a**3/6/2**0.5, following is the (v1xv2).v3 metric.
    vol = a**3/2/np.cos(np.pi/6)
    # the length of a spoke from center of any face to any vertex on that face.
    dr = a/(2*np.cos(np.pi/6))
    def test_initial_action(self):
        cl,args = init.setup_opencl(['rays.py','run'])
        dx = self.dr*np.cos(np.pi/3)
        dy = self.dr*np.sin(np.pi/3)
        dz = 2*self.dr
        xp = np.array([[
            [0,0,0,0,1,0,0,1],
            [0,self.dr,0,-dz,1,0,0,1],
            [0,-dx,dy,-dz,1,0,0,1],
            [0,-dx,-dy,-dz,1,0,0,1],
        ]]).astype(np.double)
        vg = np.array([[
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1],
            [1,0,0,1]
        ]]).astype(np.double)
        eikonal = np.array([[0,1,0,0]]).astype(np.double)
        assert ray_kernel.GetMicroAction(xp,eikonal,vg) == pytest.approx(self.vol,1e-4)
    def test_ballistic_propagation(self):
        cl,args = init.setup_opencl(['rays.py','run'])
        divAngle = 0.5
        sq = np.sin(divAngle)
        cq = np.cos(divAngle)
        ds = np.array([[6.0,6.0,6.0,6.0]]).astype(np.double)
        dx = self.dr*np.cos(np.pi/3)
        dy = self.dr*np.sin(np.pi/3)
        dz = 2*self.dr
        xp = np.array([[
            [0,0,0,0, 1,0,0,1],
            [0,self.dr,0,-dz, 1,sq,0,cq],
            [0,-dx,dy,-dz, 1,-sq,0,cq],
            [0,-dx,-dy,-dz, 1,0,-sq,cq]
        ]]).astype(np.double)
        vg = np.array([[
            [1,0,0,1],
            [1,sq,0,cq],
            [1,-sq,0,cq],
            [1,0,-sq,cq],
        ]]).astype(np.double)
        eikonal = np.array([[0,1,0,0]]).astype(np.double)
        assert ray_kernel.GetMicroAction(xp,eikonal,vg) == pytest.approx(self.vol,1e-4)
        ray_kernel.FullStep(ds,xp,eikonal,vg)
        for i in range(1,4):
            assert xp[0,:,i] == pytest.approx(vg[0,:,i]*ds[0,:],abs=dz)
        assert ray_kernel.GetMicroAction(xp,eikonal,vg) == pytest.approx(self.vol,1e-4)
