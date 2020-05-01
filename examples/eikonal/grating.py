from scipy import constants as C
import numpy as np
import surface
import input_tools

# Example of diffraction grating

# Suggested plotter command
# python ray_plotter.py out/test o3d det=4,5

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
wave = []
ray = []
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = mess

ray.append({})
ray[-1]['number'] = (64,64,16,1)
ray[-1]['bundle radius'] = (rb,rb,rb,rb)
ray[-1]['loading coordinates'] = 'cylindrical'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)

wave.append({})
wave[-1]['a0'] = (0.0,a00*np.cos(theta),0.0,-a00*np.sin(theta)) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,w00*np.sin(theta),0.0,w00*np.cos(theta)) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = (0.0,0.0,0.0,-50/mm)
wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('start')
optics[-1]['size'] = (1/inch,1/inch)
optics[-1]['origin'] = (0.,0.,-49/mm)

optics.append({})
optics[-1]['object'] = surface.Grating('G1')
optics[-1]['size'] = (2/inch,1/inch)
optics[-1]['diffracted order'] = m
optics[-1]['groove density'] = g
optics[-1]['origin'] = (0.,0.,0.)
optics[-1]['euler angles'] = helper.rot_zx(-incidence_angle)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('det')
optics[-1]['size'] = (3/inch,3/inch)
optics[-1]['origin'] = tuple(central_direction*50/mm)
optics[-1]['euler angles'] = helper.rot_zx(-total_angle)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (16,4,4,1)
diagnostics['base filename'] = 'out/test'
