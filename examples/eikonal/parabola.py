from scipy import constants as C
import numpy as np
import modules.surface as surface
import modules.input_tools as input_tools 

# Example showing 90 degree off-axis parabolic mirror
# Suggested plotter command:
# python plotter.py out/test det=1,2/0,0/0.1
# Verify spot size and intensity against preprocessing calculation

# Units and scales

mks_length = 0.8e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
fs = 1e15*mks_length/C.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Preprocessing calculations

w00 = 1.0
theta = 0.0 # direction of propagation, 0 is +z
par_f = 500/mm
y0 = 2*par_f
z0 = -2*par_f
r00 = 50/mm # spot size of radiation
t00 = 100/fs

# N.b. the effective focal length is y0, not par_f
f_num = y0/(2*r00)
rb = bundle_scale*r00
t00,band = helper.TransformLimitedBandwidth(w00,t00,4)
a00 = helper.InitialVectorPotential(w00,1.0,y0,f_num)
mess = mess + helper.ParaxialFocusMessage(w00,1.0,y0,f_num)

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
            'origin': (None,0,y0,z0),
            'euler angles': helper.rot_zx(theta),
            'number': (1,128,32,None),
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
optics[-1]['object'] = surface.Paraboloid('mirror')
optics[-1]['reflective'] = True
optics[-1]['focal length'] = par_f
optics[-1]['acceptance angle'] = 180/deg/1.8
optics[-1]['off axis angle'] = 0.
optics[-1]['euler angles'] = (0.,180/deg,0.)

optics.append({})
optics[-1]['object'] = surface.FullWaveProfiler('det')
optics[-1]['size'] = (.4/mm,.4/mm,.2/mm)
optics[-1]['wave grid'] = (1,2048,2048,1)
optics[-1]['distance to caustic'] = 1.25/mm
optics[-1]['origin'] = (None,0.,1.25/mm,0.)
optics[-1]['euler angles'] = (0.,90/deg,0.)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('terminus')
optics[-1]['size'] = (300/mm,300/mm)
optics[-1]['origin'] = (None,0.,-par_f,0.)
optics[-1]['euler angles'] = (0.,90/deg,0.)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (1,2,32,None)
diagnostics['base filename'] = 'out/test'
