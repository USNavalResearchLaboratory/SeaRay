from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Example of dispersion through prism

# Suggested plotter command:
# python ray_plotter.py out/test o3d det=4,5

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
ray[-1]['box'] = band + (0.0,3*r00) + (0.0,2*np.pi) + (0.0,0.0)

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
optics[-1]['object'] = volume.Prism('P1')
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['dispersion inside'] = material
optics[-1]['size'] = prism_box
optics[-1]['origin'] = (0.,0.,0.)
optics[-1]['euler angles'] = (0.0,0.0,0.0)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('det')
optics[-1]['size'] = (2/inch,1/inch)
optics[-1]['origin'] = tuple(central_direction*.05/mks_length)
optics[-1]['euler angles'] = helper.rot_zx(-central_angle)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (16,2,2,1)
diagnostics['base filename'] = 'out/test'
