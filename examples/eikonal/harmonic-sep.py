from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools 

# Example of separating Nd:glass wavelength and its second harmonic in a Pellin-Broca prism.

# Suggested plotter command
# python plotter.py out/test o3d o31

# Units and scales

mks_length = 1.054e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
inch = 100*mks_length/2.54
fs = 1e15*mks_length/C.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Setup pulse parameters

theta = 0.0 # initial propagation angle
a00 = 1.0
w00 = 1.0
r00 = 0.1/mm # spot size of radiation
rb = r00*bundle_scale
t00 = 500/fs
band = (0.9,2.1)

# Setup prism

material = dispersion.BK7(mks_length)
nrefr = np.sqrt(1+material.chi(w00)[0])
prism_box = (1/inch,1/inch,0.75/inch)
refraction_angle = 30/deg
incidence_angle = np.arcsin(nrefr*np.sin(refraction_angle))
mess += 'TIR angle = ' + str(np.arcsin(1/nrefr)*deg) + ' deg\n'

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
ray[-1]['number'] = (64,1,4,None)
ray[-1]['bundle radius'] = (None,rb,rb,rb)
ray[-1]['loading coordinates'] = 'cylindrical'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = band + (0.0,3*r00) + (0.0,2*np.pi) + (None,None)

wave.append({}) # fundamental
wave[-1]['a0'] = (0.0,a00*np.cos(theta),0.0,-a00*np.sin(theta)) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,w00*np.sin(theta),0.0,w00*np.cos(theta)) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = (0.0,0.0,0.0,-31/mm)
wave[-1]['supergaussian exponent'] = 2

wave.append({}) # second harmonic
wave[-1]['a0'] = (0.0,a00*np.cos(theta),0.0,-a00*np.sin(theta)) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (2*w00,2*w00*np.sin(theta),0.0,2*w00*np.cos(theta)) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = (0.0,0.0,0.0,-31/mm)
wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = volume.PellinBroca('P1')
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['dispersion inside'] = material
optics[-1]['size'] = prism_box
optics[-1]['angle'] = refraction_angle
optics[-1]['origin'] = (-10/mm,0.,0.)
optics[-1]['euler angles'] = helper.rot_zx(incidence_angle)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('det')
optics[-1]['size'] = (1/inch,1/inch)
optics[-1]['origin'] = (100/mm,0.0,10/mm)
optics[-1]['euler angles'] = helper.rot_zx(-90/deg)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (16,1,4,None)
diagnostics['base filename'] = 'out/test'
