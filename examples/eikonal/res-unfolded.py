from scipy import constants as Cmks
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Unfolded confocal unstable resonator using ideal lenses.

# Also illustrates coupling rays into and out of a paraxial wave region.
# The final focusing lens should put the waist midway through the wave zone.
# Note the rays have no information about the wave zone interior, and will appear as going "straight through".
# The rays that emerge should be thought of as "new rays".
# The new rays should have the correct amplitude, wavevector, and phase, given their (x,y) position.
# However accuracy tends to be stressed for marginal rays; increasing box size and resolution helps.

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

theta = 0.0 # initial propagation direction
a00 = 1.0
w00 = 1.0
r00 = 0.05/cm # spot size of radiation
rb = r00*bundle_scale
t00 = 1/ps
band = (0.9,1.1)

# Resonator configuration

Rf = 0.5/cm
Rp = 2/cm
ff = -50/cm
fp = 150/cm
L = 100/cm

# Set up dictionaries

sim = {}
wave = []
ray = []
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/Cmks.c
sim['message'] = mess

ray.append({})
ray[-1]['number'] = (32,32,32,1)
ray[-1]['bundle radius'] = (rb,rb,rb,rb)
ray[-1]['loading coordinates'] = 'cartesian'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = band + (-2*r00,2*r00) + (-2*r00,2*r00) + (0.0,0.0)

wave.append({})
wave[-1]['a0'] = (0.0,a00*np.cos(theta),0.0,-a00*np.sin(theta)) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,w00*np.sin(theta),0.0,w00*np.cos(theta)) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = (0.0,0.0,0.0,-.1/cm)
wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = surface.IdealLens('L1')
optics[-1]['focal length'] = -50/cm
optics[-1]['radius'] = Rf
optics[-1]['origin'] = helper.set_pos([0.0,0.0,0.0])

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
diagnostics['orbit rays'] = (1,4,4,1)
diagnostics['base filename'] = 'out/test'
