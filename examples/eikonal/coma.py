from scipy import constants as C
import numpy as np
import modules.surface as surface
import modules.input_tools as input_tools 

# Off-axis focusing with spherical mirror, leading to coma aberration

# Suggested plotter command:
# python plotter.py out/test det=1,2/0,0
# Observe deformed spot offset from the axis.

# Units and scales

mks_length = 0.8e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
fs = 1e15*mks_length/C.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Control parameters

w00 = 1.0
theta = -5/deg # direction of propagation, 0 is +z
mirror_D = 100/mm
mirror_R = 1000/mm
r00 = 10/mm # spot size of radiation
t00 = 100/fs

# Derived parameters

f_num = (mirror_R/2)/(2*r00)
t00,band = helper.TransformLimitedBandwidth(w00,t00,4)
a00 = helper.InitialVectorPotential(w00,1.0,mirror_R/2,f_num)
rb = r00*bundle_scale
mess = mess + helper.ParaxialFocusMessage(w00,1.0,mirror_R/2,f_num)

# Set up dictionaries

sim = {}
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = mess

sources = [
    {
        'rays': {
            'origin': (None,0.0,0.0,-100/mm),
            'euler angles': helper.rot_zx(theta),
            'number': (1,128,128,None),
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

optics.append({})
optics[-1]['object'] = surface.SphericalCap('mirror')
optics[-1]['reflective'] = True
optics[-1]['radius of sphere'] = mirror_R
optics[-1]['radius of edge'] = mirror_D/2
optics[-1]['origin'] = (None,0,0,0)
optics[-1]['euler angles'] = (0.,np.pi,0.)

optics.append({})
optics[-1]['object'] = surface.FullWaveProfiler('det')
optics[-1]['size'] = (1/mm,1/mm,1/mm)
optics[-1]['wave grid'] = (1,1024,1024,1)
optics[-1]['distance to caustic'] = 10/mm
optics[-1]['origin'] = (None,43/mm,0,-490/mm)
optics[-1]['euler angles'] = (0.,np.pi,0.)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('terminus')
optics[-1]['size'] = (500/mm,500/mm)
optics[-1]['origin'] = (None,0,0,-1000/mm)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (1,4,4,None)
diagnostics['base filename'] = 'out/test'
