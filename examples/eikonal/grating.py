from scipy import constants as C
import numpy as np
import modules.surface as surface
import modules.input_tools as input_tools 

# Example of diffraction grating

# Suggested plotter command
# python plotter.py out/test o3d det=4,5

# Units and scales

mks_length = 0.8e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
inch = 100*mks_length/2.54
fs = 1e15*mks_length/C.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Setup pulse parameters

w00 = 1.0
theta = 0.0 # initial propagation angle
r00 = 3/mm # spot size of radiation
rb = r00*bundle_scale
t00,band = helper.TransformLimitedBandwidth(w00,'10 fs',1)
a00 = 1.0

# Work out angles for grating and sensor plane

m = 1.0 # diffracted order
g = 1000*mm # groove density
incidence_angle = 45/deg
central_diff_angle = np.arcsin(np.sin(incidence_angle)-2*np.pi*m*g/w00) # grating equation
total_angle = incidence_angle + central_diff_angle
central_direction = np.array([-np.sin(total_angle),0.0,-np.cos(total_angle)])
mess += 'diffraction angle = ' + str(central_diff_angle)

# Set up dictionaries

sim = {}
optics = []
diagnostics = {}

sources = [
    {
        'rays': {
            'origin': (None,0.0,0.0,-50/mm),
            'euler angles': helper.rot_zx(theta),
            'number': (64,64,16,None),
            'bundle radius': (None,) + (rb,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': band + (0,3*r00) + (0,2*np.pi) + (None,None)
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

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = mess

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('start')
optics[-1]['size'] = (1/inch,1/inch)
optics[-1]['origin'] = (None,0,0,-49/mm)

optics.append({})
optics[-1]['object'] = surface.Grating('G1')
optics[-1]['size'] = (2/inch,1/inch)
optics[-1]['diffracted order'] = m
optics[-1]['groove density'] = g
optics[-1]['origin'] = (None,0,0,0)
optics[-1]['euler angles'] = helper.rot_zx(-incidence_angle)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('det')
optics[-1]['size'] = (3/inch,3/inch)
optics[-1]['origin'] = (None,) + tuple(central_direction*50/mm)
optics[-1]['euler angles'] = helper.rot_zx(-total_angle)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (16,4,4,None)
diagnostics['base filename'] = 'out/test'
