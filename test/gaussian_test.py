import sys
sys.path.append("modules")
import pathlib
from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.volume as volume
import modules.input_tools as input_tools 
import rays

out = pathlib.Path('out')

mks_length = 0.8e-6 / (2*np.pi)
cm = 100*mks_length
mess = 'Processing input file...\n'
helper = input_tools.InputHelper(mks_length)

a00 = 1.0
w00 = 1.0
r00 = .01/cm
zR = 0.5*w00*r00**2

sim = {
    'mks_length': mks_length,
    'mks_time': mks_length/C.c,
    'message': 'gaussian test'
}

# Dummy sources, not used

sources = [
    {
        'rays': {
            'origin': (None,0,0,-10*zR),
            'euler angles': (0,0,0),
            'number': (2,2,2,None),
            'bundle radius': (None,) + (.001*r00,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': (0.9,1.1) + (0,3*r00) + (0,2*np.pi) + (None,None)
        },
        'waves': [
            {
                'a0': (None,a00,0,None),
                'r0': (r00,r00,r00,r00),
                'k0': (w00,None,None,w00),
                'mode': (None,0,0,None),
                'basis': 'hermite'
            },
        ]
    }
]

diagnostics = {
    'suppress details': False,
    'clean old files': True,
    'orbit rays': (2,2,2,None),
    'base filename': 'out/test'
}

def incoming_gaussian(w_nodes,x_nodes,y_nodes,dw00):
    f = lambda x,x0,dx: np.exp(-(x-x0)**2/dx**2)
    wfunc = f(w_nodes,w00,dw00)
    return a00*np.einsum('i,j,k',wfunc,f(x_nodes,0,r00),f(y_nodes,0,r00))

class TestParaxialGaussian:

    def test_H00_expansion(self):
        optics = []
        optics.append({})
        optics[-1]['object'] = volume.AnalyticBox('vacuum')
        optics[-1]['density reference'] = 1.0
        optics[-1]['density function'] = '1.0'
        optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(x.shape)
        optics[-1]['incoming wave'] = lambda w,x,y: incoming_gaussian(w,x,y,.01)
        optics[-1]['frequency band'] = (.95,1.05)
        optics[-1]['wave grid'] = (64,128,128,3)
        optics[-1]['wave coordinates'] = 'cartesian'
        optics[-1]['dispersion inside'] = dispersion.Vacuum()
        optics[-1]['dispersion outside'] = dispersion.Vacuum()
        optics[-1]['size'] = (36*r00,36*r00,4*zR)
        optics[-1]['origin'] = (None,0,0,0)
        optics[-1]['euler angles'] = (0.,0.,0.)
        optics[-1]['propagator'] = 'paraxial'
        optics[-1]['subcycles'] = 1
        rays.run([],sim,sources,optics,diagnostics)
        A = np.load(out / 'test_vacuum_paraxial_wave.npy')
        dom4d = np.load(out / 'test_vacuum_paraxial_plot_ext.npy')
        dw = (dom4d[1]-dom4d[0])/A.shape[0]
        dx = (dom4d[3]-dom4d[2])/A.shape[1]
        dy = (dom4d[5]-dom4d[4])/A.shape[2]
        w_nodes = np.linspace(dom4d[0] + 0.5*dw,dom4d[1] - 0.5*dw,A.shape[0])
        x_nodes = np.linspace(dom4d[2] + 0.5*dx,dom4d[3] - 0.5*dx,A.shape[1])
        y_nodes = np.linspace(dom4d[4] + 0.5*dy,dom4d[5] - 0.5*dy,A.shape[2])
        assert A.shape==(64,128,128,3)

        # first check the midway point
        xlineout = np.abs(A[32,:,64,1])
        wfac = np.exp(-(w_nodes[32]-w00)**2/.01**2)
        yfac = np.exp(-y_nodes[64]**2/r00**2/5)
        xlineout_expected = wfac*yfac*a00*np.exp(-x_nodes**2/r00**2/5) / 5**0.5
        assert np.allclose(xlineout,xlineout_expected,.02,.01)

        # now check the final point
        xlineout = np.abs(A[32,:,64,2])
        wfac = np.exp(-(w_nodes[32]-w00)**2/.01**2)
        yfac = np.exp(-y_nodes[64]**2/r00**2/17)
        xlineout_expected = wfac*yfac*a00*np.exp(-x_nodes**2/r00**2/17) / 17**0.5
        assert np.allclose(xlineout,xlineout_expected,.02,.01)

class TestUPPEGaussian:

    def test_H00_expansion(self):
        optics = []
        optics.append({})
        optics[-1]['object'] = volume.AnalyticBox('vacuum')
        optics[-1]['density reference'] = 1.0
        optics[-1]['density function'] = '1.0'
        optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(x.shape)
        optics[-1]['incoming wave'] = lambda w,x,y: incoming_gaussian(w,x,y,0.1)
        optics[-1]['frequency band'] = (0.0,2.0)
        optics[-1]['nonlinear band'] = (0.0,2.0)
        optics[-1]['wave grid'] = (65,128,128,3)
        optics[-1]['wave coordinates'] = 'cartesian'
        optics[-1]['dispersion inside'] = dispersion.Vacuum()
        optics[-1]['dispersion outside'] = dispersion.Vacuum()
        optics[-1]['size'] = (36*r00,36*r00,4*zR)
        optics[-1]['origin'] = (None,0,0,0)
        optics[-1]['euler angles'] = (0.,0.,0.)
        optics[-1]['propagator'] = 'uppe'
        optics[-1]['subcycles'] = 1
        optics[-1]['minimum step'] = 1
        rays.run([],sim,sources,optics,diagnostics)
        A = np.load(out / 'test_vacuum_uppe_wave.npy')
        dom4d = np.load(out / 'test_vacuum_uppe_plot_ext.npy')
        dw = (dom4d[1]-dom4d[0])/A.shape[0]
        dx = (dom4d[3]-dom4d[2])/A.shape[1]
        dy = (dom4d[5]-dom4d[4])/A.shape[2]
        w_nodes = np.linspace(dom4d[0] + 0.5*dw,dom4d[1] - 0.5*dw,A.shape[0])
        x_nodes = np.linspace(dom4d[2] + 0.5*dx,dom4d[3] - 0.5*dx,A.shape[1])
        y_nodes = np.linspace(dom4d[4] + 0.5*dy,dom4d[5] - 0.5*dy,A.shape[2])
        assert A.shape==(65,128,128,3)
        
        # first check the midway point
        xlineout = np.abs(A[32,:,64,1])
        wfac = np.exp(-(w_nodes[32]-w00)**2/.1**2)
        yfac = np.exp(-y_nodes[64]**2/r00**2/5)
        xlineout_expected = wfac*yfac*a00*np.exp(-x_nodes**2/r00**2/5) / 5**0.5
        assert np.allclose(xlineout,xlineout_expected,.02,.01)

        # now check the final point
        xlineout = np.abs(A[32,:,64,2])
        wfac = np.exp(-(w_nodes[32]-w00)**2/.1**2)
        yfac = np.exp(-y_nodes[64]**2/r00**2/17)
        xlineout_expected = wfac*yfac*a00*np.exp(-x_nodes**2/r00**2/17) / 17**0.5
        assert np.allclose(xlineout,xlineout_expected,.02,.01)