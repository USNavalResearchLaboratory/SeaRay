from scipy import constants as Cmks
import numpy as np
import modules.surface as surface
import modules.input_tools as input_tools 

# This illustrates an unstable off-axis cavity resonator.
# In order to fold the path, we position multiple copies of
# a mirror in the same position.  This allows the rays to
# effectively interact with the "same" surface more than once.

# Units and scales

mks_length = 10.3e-6 / (2*np.pi)
bundle_scale = 1e-4
cm = 100*mks_length
inch = 100*mks_length/2.54
ps = 1e12*mks_length/Cmks.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Setup pulse parameters

theta = -.01 # initial propagation direction
a00 = 1.0
w00 = 1.0
r00 = 0.05/cm # spot size of radiation
rb = r00*bundle_scale
t00 = 1/ps
band = (0.9,1.1)

# Resonator configuration

feedback_pos = (None,1/cm,0.0,100/cm)
feedback_angle = (np.pi/2,0.0035,-np.pi/2)
Rf = 0.5/cm
primary_pos = (None,0.0,0.0,0/cm)
primary_angle = (0.0,0.0,0.0)
Rp = 10/cm

# Set up dictionaries

sim = {}
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/Cmks.c
sim['message'] = mess

sources = [
    {
        'rays': {
            'origin': (None,0,0,-10/cm),
            'euler angles': helper.rot_zx(theta),
            'number': (1,8,1,None),
            'bundle radius': (None,) + (rb,)*3,
            'loading coordinates': 'cartesian',
            'bounds': band + (-2*r00,2*r00) + (-2*r00,2*r00) + (None,None)
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

optics.append({})
optics[-1]['object'] = surface.SphericalCap('M1')
optics[-1]['reflective'] = True
optics[-1]['radius of sphere'] = 2*50/cm
optics[-1]['radius of edge'] = Rf
optics[-1]['origin'] = feedback_pos
optics[-1]['euler angles'] = feedback_angle

optics.append({})
optics[-1]['object'] = surface.SphericalCap('M2')
optics[-1]['reflective'] = True
optics[-1]['radius of sphere'] = 2*150/cm
optics[-1]['radius of edge'] = Rp
optics[-1]['origin'] = primary_pos
optics[-1]['euler angles'] = primary_angle

optics.append({})
optics[-1]['object'] = surface.SphericalCap('M3')
optics[-1]['reflective'] = True
optics[-1]['radius of sphere'] = 2*50/cm
optics[-1]['radius of edge'] = Rf
optics[-1]['origin'] = feedback_pos
optics[-1]['euler angles'] = feedback_angle

optics.append({})
optics[-1]['object'] = surface.SphericalCap('M4')
optics[-1]['reflective'] = True
optics[-1]['radius of sphere'] = 2*150/cm
optics[-1]['radius of edge'] = Rp
optics[-1]['origin'] = primary_pos
optics[-1]['euler angles'] = primary_angle

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('det')
optics[-1]['size'] = (4/inch,4/inch)
optics[-1]['origin'] = (None,0,0,100/cm)
optics[-1]['euler angles'] = helper.rot_zx_deg(0)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (1,8,1,None)
diagnostics['base filename'] = 'out/test'
