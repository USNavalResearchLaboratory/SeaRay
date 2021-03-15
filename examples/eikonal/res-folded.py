from scipy import constants as Cmks
import numpy as np
import dispersion
import surface
import volume
import input_tools

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

theta = .01 # initial propagation direction
a00 = 1.0
w00 = 1.0
r00 = 0.05/cm # spot size of radiation
rb = r00*bundle_scale
t00 = 1/ps
band = (0.9,1.1)

# Resonator configuration

feedback_pos = (1/cm,0.0,100/cm)
feedback_angle = (np.pi/2,0.0035,-np.pi/2)
Rf = 0.5/cm
primary_pos = (0.0,0.0,0/cm)
primary_angle = (0.0,0.0,0.0)
Rp = 10/cm

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
ray[-1]['number'] = (1,8,1,1)
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
wave[-1]['focus'] = (0.0,0.0,0.0,-10/cm)
wave[-1]['supergaussian exponent'] = 2

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
optics[-1]['origin'] = (0.0,0.0,100/cm)
optics[-1]['euler angles'] = helper.rot_zx_deg(0)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (1,8,1,1)
diagnostics['base filename'] = 'out/test'
