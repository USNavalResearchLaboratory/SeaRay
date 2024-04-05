'''
Test Module for Prisms
----------------------
* Rays of various frequencies refract into the right angle
* Indirect test of refraction and dispersion
'''
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
mks_length = 0.8e-6/(2*np.pi)
rb = 0.01
r00 = 100
a00 = 1
t00 = 1000
w00 = 1
bk7 = modules.dispersion.BK7(mks_length)
prism_box = (None,2000,1000,1000)

def expected_output_angle(w):
    '''get expected angle of outgoing ray for frequency w,
    relative to the z-axis'''
    nrefr = np.sqrt(1+bk7.chi(w))
    q0 = np.arctan(0.5*prism_box[3]/prism_box[1]) # half angle of the prism
    q1i = -q0 # angle of incidence first surface
    q1r = np.arcsin(np.sin(q1i)/nrefr) # angle of refraction first surface
    q2i = q1r + 2*q0
    q2r = np.arcsin(np.sin(q2i)*nrefr)
    return -(q2r - q0)

sim = {
    'mks_length': mks_length,
    'mks_time': mks_length/C.c,
    'message': 'test prisms'
}

sources = [
    {
        'rays': {
            'origin': (None,0,0,-1000),
            'euler angles': (0,0,0),
            'number': (4,2,2,None),
            'bundle radius': (None,) + (rb,)*3,
            'loading coordinates': 'cartesian',
            'bounds': (0.5,1.5) + (-3*r00,3*r00) + (-3*r00,3*r00) + (None,None)
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

class TestPrisms:

    def test_isoceles(self):
        central_angle = expected_output_angle(1.0)[0]
        optics = [
            {
                'object': modules.volume.Prism('prism'),
                'size': prism_box[1:],
                'origin': (None,0,0,0),
                'euler angles': (0,0,0),
                'dispersion outside': modules.dispersion.Vacuum(),
                'dispersion inside': bk7
            },
            {
                'object': modules.surface.EikonalProfiler('det'),
                'euler angles': (0,0,0),
                'size': (2000,2000),
                'origin': (None,3000*np.sin(central_angle),0.0,3000*np.cos(central_angle))
            }
        ]
        rays.run([],sim,sources,optics,diagnostics)
        xps = np.load(out / 'test_det_xps.npy')
        eiks = np.load(out / 'test_det_eiks.npy')
        assert xps.shape==(16,8)
        assert eiks.shape==(16,4)
        output_angles = np.arctan(xps[:,5]/xps[:,7])
        no_angle = np.arctan(xps[:,6]/xps[:,7])
        expected_outputs = expected_output_angle(xps[:,4])
        assert np.allclose(no_angle,0,0,1e-10)
        assert np.allclose(output_angles,expected_outputs,0,1e-10)
    