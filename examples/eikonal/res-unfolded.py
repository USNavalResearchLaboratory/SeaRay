from scipy import constants as Cmks
import numpy as np
import modules.dispersion as dispersion
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools 

# Unfolded confocal unstable resonator using ideal lenses.

# Also illustrates coupling rays into and out of a paraxial wave region.
# The final focusing lens should put the waist midway through the wave zone.
# Note the rays have no information about the wave zone interior, and will appear as going "straight through".
# The rays that emerge should be thought of as "new rays".
# The new rays should have the correct amplitude, wavevector, and phase, given their (x,y) position.
# However accuracy tends to be stressed for marginal rays; increasing box size and resolution helps.

# Units and scales

mks_length = 10.3e-6 / (2*np.pi)
bundle_scale = 1e-2
cm = 100*mks_length
inch = 100*mks_length/2.54
ps = 1e12*mks_length/Cmks.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Setup pulse parameters

theta = 0.0 # initial propagation direction
a00 = 1.0
w00 = 1.0
r00 = 0.05/cm # spot size of radiation
rb = r00*bundle_scale
t00 = 1/ps
band = (0.9,1.1)

# Resonator configuration

Rf = 1/cm
Rp = 5/cm
ff = -50/cm
fp = 150/cm
L = 100/cm

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
            'origin': (None,0,0,-0.1/cm),
            'euler angles': helper.rot_zx(-theta),
            'number': (32,128,128,None),
            'bundle radius': (None,) + (rb,)*3,
            'loading coordinates': 'cartesian',
            'bounds': band + (-3*r00,3*r00) + (-3*r00,3*r00) + (None,None)
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
optics[-1]['object'] = surface.IdealLens('L1')
optics[-1]['focal length'] = -50/cm
optics[-1]['radius'] = Rf
optics[-1]['origin'] = helper.set_pos([None,0.0,0.0,0.0])

optics.append({})
optics[-1]['object'] = surface.IdealLens('L2')
optics[-1]['focal length'] = 150/cm
optics[-1]['radius'] = Rp
optics[-1]['origin'] = helper.move(0.0,0.0,L)

optics.append({})
optics[-1]['object'] = surface.IdealLens('L3')
optics[-1]['focal length'] = -50/cm
optics[-1]['radius'] = Rf
optics[-1]['origin'] = helper.move(0.0,0.0,L)


optics.append({})
optics[-1]['object'] = surface.IdealLens('L4')
optics[-1]['focal length'] = 150/cm
optics[-1]['radius'] = Rp
optics[-1]['origin'] = helper.move(0.0,0.0,L)

optics.append({})
optics[-1]['object'] = surface.IdealLens('final_focus')
optics[-1]['focal length'] = L/2
optics[-1]['radius'] = Rp
optics[-1]['origin'] = helper.move(0.0,0.0,L)

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('vacuum')
optics[-1]['density reference'] = 1.0
optics[-1]['density function'] = '1.0'
optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(x.shape)
optics[-1]['frequency band'] = band
optics[-1]['wave grid'] = (32,128,128,5)
optics[-1]['wave coordinates'] = 'cartesian'
optics[-1]['dispersion inside'] = dispersion.Vacuum()
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['size'] = (4/cm,4/cm,L)
optics[-1]['origin'] = helper.move(0.0,0.0,L/2+1/cm)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['propagator'] = 'paraxial'
optics[-1]['subcycles'] = 1
optics[-1]['full relaunch'] = True

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('det')
optics[-1]['size'] = (4/inch,4/inch)
optics[-1]['origin'] = helper.move(0.0,0.0,L+1/cm)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (1,4,4,None)
diagnostics['base filename'] = 'out/test'
