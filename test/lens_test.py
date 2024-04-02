import sys
sys.path.append("modules")
import pytest
import numpy as np
from scipy import constants as C
import modules.surface
import modules.volume
import modules.dispersion
import rays
import pathlib

out = pathlib.Path('out')
rb = 0.01
r00 = 100
a00 = 1
t00 = 1000
w00 = 1

sim = {
    'mks_length': 0.8e-6/(2*np.pi),
    'mks_time': 0.8e-6/(2*np.pi)/C.c,
    'message': 'test lenses'
}

sources = [
    {
        'rays': {
            'origin': (None,0,0,-1000),
            'euler angles': (0,0,0),
            'number': (1,2,2,None),
            'bundle radius': (None,) + (rb,)*3,
            'loading coordinates': 'cartesian',
            'bounds': (0.9,1.1) + (-3*r00,3*r00) + (-3*r00,3*r00) + (None,None)
        },
        'waves': [
            {
                'a0': (None,a00,0,None),
                'r0': (t00,r00,r00,t00),
                'k0': (w00,None,None,w00),
                'mode': (None,0,0,None),
                'basis': 'hermite'
            }
        ]
    }
]

diagnostics = {
    'suppress details': False,
    'clean old files': True,
    'orbit rays': (1,2,2,None),
    'base filename': 'out/test'
}

class TestLenses:

    def test_ideal_lens_at_focus(self):
        optics = [
            {
                'object': modules.surface.IdealLens('L1'),
                'euler angles': (0,0,0),
                'radius': 4*r00,
                'focal length': 100
            },
            {
                'object': modules.surface.EikonalProfiler('det'),
                'euler angles': (0,0,0),
                'size': (4*r00,4*r00),
                'origin': (None,0,0,100)
            }
        ]
        rays.run([],sim,sources,optics,diagnostics)
        xps = np.load(out / 'test_det_xps.npy')
        eiks = np.load(out / 'test_det_eiks.npy')
        assert xps.shape==(4,8)
        assert eiks.shape==(4,4)
        assert np.allclose(xps[:,1],0,0,1e-10)
        assert np.allclose(xps[:,2],0,0,1e-10)
        assert np.allclose(eiks[:,0],1100,.01,0)
    
    def test_ideal_lens_halfway_to_focus(self):
        optics = [
            {
                'object': modules.surface.IdealLens('L1'),
                'euler angles': (0,0,0),
                'radius': 4*r00,
                'focal length': 100
            },
            {
                'object': modules.surface.EikonalProfiler('det'),
                'euler angles': (0,0,0),
                'size': (4*r00,4*r00),
                'origin': (None,0,0,50)
            }
        ]
        rays.run([],sim,sources,optics,diagnostics)
        xp0 = np.load(out / 'test_xp0.npy')
        xps = np.load(out / 'test_det_xps.npy')
        eik0 = np.load(out / 'test_eikonal0.npy')
        eiks = np.load(out / 'test_det_eiks.npy')
        assert xp0.shape==(4,4,8)
        assert xps.shape==(4,8)
        assert eik0.shape==(4,4)
        assert eiks.shape==(4,4)
        assert np.allclose(xp0[:,0,1]/2,xps[:,1],1e-4,0)
        assert np.allclose(xp0[:,0,2]/2,xps[:,2],1e-4,0)
        amag0 = np.einsum("ij,ij->i",eik0[:,1:],eik0[:,1:])
        amag = np.einsum("ij,ij->i",eiks[:,1:],eiks[:,1:])
        assert np.allclose(amag0,amag/4,.01,0)
    
    def test_bk7_lens(self):
        bk7 = modules.dispersion.BK7(sim['mks_length'])
        lens_r = r00*5
        lens_R = r00*20
        lens_t = r00*2
        nrefr = np.sqrt(1+bk7.chi(1)[0])
        f = 1/(nrefr-1)/(1/lens_R + 1/lens_R - (nrefr-1)*lens_t/(nrefr*lens_R*lens_R))
        sim['message'] += "\nfocal length = " + str(f)
        optics = [
            {
                'object': modules.volume.SphericalLens('L1'),
                'dispersion inside': bk7,
                'dispersion outside': modules.dispersion.Vacuum(),
                'thickness': lens_t,
                'rcurv beneath': lens_R,
                'rcurv above': -lens_R,
                'aperture radius': lens_r,
                'euler angles': (0,0,0),
                'origin': (None,0,0,0),
            },
            {
                'object': modules.surface.EikonalProfiler('det'),
                'euler angles': (0,0,0),
                'size': (4*r00,4*r00),
                'origin': (None,0,0,f)
            }
        ]
        rays.run(["log=debug"],sim,sources,optics,diagnostics)
        xps = np.load(out / 'test_det_xps.npy')
        eiks = np.load(out / 'test_det_eiks.npy')
        assert xps.shape==(4,8)
        assert eiks.shape==(4,4)
        assert np.allclose(xps[:,1],0,0,r00/100)
        assert np.allclose(xps[:,2],0,0,r00/100)
        assert np.allclose(eiks[:,0],1000+f+lens_t*(nrefr-1),.01,0)
