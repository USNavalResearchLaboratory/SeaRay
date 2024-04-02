from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools 

# Example of dispersion through prism

# Suggested plotter command:
# python plotter.py out/test o3d det=4,5

mks_length = 0.8e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
inch = 100*mks_length/2.54
fs = 1e15*mks_length/C.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Setup pulse parameters

theta = 0.0
w00 = 1.0
r00 = 2/mm # spot size of radiation
rb = r00*bundle_scale
t00,band = helper.TransformLimitedBandwidth(w00,'1 fs',1)
a00 = 1.0

# Setup prism and predict central trajectory angle

material = dispersion.BK7(mks_length)
nrefr = np.sqrt(1+material.chi(w00)[0])
prism_box = (2/inch,1/inch,1/inch)
q0 = np.arctan(0.5*prism_box[2]/prism_box[0]) # half angle of the prism
q1i = -q0 # angle of incidence first surface
q1r = np.arcsin(np.sin(q1i)/nrefr) # angle of refraction first surface
q2i = q1r + 2*q0
q2r = np.arcsin(np.sin(q2i)*nrefr)
central_angle = -(q2r - q0)
central_direction = np.array([np.sin(central_angle),0.0,np.cos(central_angle)])

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
            'origin': (None,0,0,-50/mm),
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

optics.append({})
optics[-1]['object'] = volume.Prism('P1')
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['dispersion inside'] = material
optics[-1]['size'] = prism_box
optics[-1]['origin'] = (None,0,0,0)
optics[-1]['euler angles'] = (0.0,0.0,0.0)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('det')
optics[-1]['size'] = (2/inch,1/inch)
optics[-1]['origin'] = (None,) + tuple(central_direction*.05/mks_length)
optics[-1]['euler angles'] = helper.rot_zx(-central_angle)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (16,2,2,None)
diagnostics['base filename'] = 'out/test'
