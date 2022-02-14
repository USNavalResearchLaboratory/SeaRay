from scipy import constants as Cmks
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Units and scales

mks_length = 12.0e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
cm = 100*mks_length
inch = 100*mks_length/2.54
fs = 1e15*mks_length/Cmks.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Setup pulse parameters

theta = 0/deg # initial propagation direction
a00 = 1.0
w00 = 1.0
r00 = 0.15/cm # spot size of radiation
rb = r00*bundle_scale
t00 = 20/fs
band = (0.5,2.0)

# Setup prism
# characteristic angle is ideally the refraction angle for Brewster incidence.

material = dispersion.KCl(mks_length)
prism_box = (1.273/inch,1/inch,2.25/inch)
characteristic_angle = 34.629/deg
#OPL_offset = 0.284/inch
nrefr = np.sqrt(1+material.chi(w00)[0])
mess += 'TIR angle = ' + str(np.arcsin(1/nrefr)*deg) + ' deg\n'

# The angle of incidence for a ray at the design wavelength
incidence_angle = np.arcsin(nrefr*np.sin(characteristic_angle))
# The actual angle of incidence can be otherwise (this can be viewed as tuning the wavelength)
prism_angle = -incidence_angle
# calculating midpoint in x1 direction for the rotated prism
temp_prism = volume.PellinBroca2('temp')
A,B,C,D = temp_prism.SideLengths(prism_box[0],prism_box[2],characteristic_angle)
Ac,Bc,Cc,Dc = temp_prism.SideCenters(A,B,C,D,characteristic_angle)
prism_displacement_x = -(Ac[0]*np.cos(prism_angle) - Ac[2]*np.sin(prism_angle))

# Lenses
lens_D = 1/inch
lens_t = 3/mm
lens_R1 = 1e6/mm
lens_R2 = 280.5/mm

# General layout
focus = (1e5/fs,0.0,0.0,0.0)

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
ray[-1]['number'] = (32,8,8,1)
ray[-1]['bundle radius'] = (rb,rb,rb,rb)
ray[-1]['loading coordinates'] = 'cylindrical'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = band + (0.0,1*r00) + (0.0,2*np.pi) + (0.0,0.0)

wave.append({})
wave[-1]['a0'] = (0.0,a00*np.cos(theta),0.0,-a00*np.sin(theta)) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,w00*np.sin(theta),0.0,w00*np.cos(theta)) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = focus
wave[-1]['supergaussian exponent'] = 2

# wave.append({})
# wave[-1]['a0'] = (0.0,a00*np.cos(theta),0.0,-a00*np.sin(theta)) # EM 4-potential (eA/mc^2) , component 0 not used
# wave[-1]['r0'] = (t00/5,r00,r00,t00/5) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
# wave[-1]['k0'] = (5*w00,5*w00*np.sin(theta),0.0,5*w00*np.cos(theta)) # 4-wavenumber: omega,kx,ky,kz
# # 0-component of focus is time at which pulse reaches focal point.
# # If time=0 use paraxial wave, otherwise use spherical wave.
# # Thus in the paraxial case the pulse always starts at the waist.
# wave[-1]['focus'] = focus
# wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = volume.SphericalLens('lens1')
optics[-1]['dispersion inside'] = dispersion.ZnSe(mks_length)
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['thickness'] = lens_t
optics[-1]['rcurv beneath'] = lens_R1
optics[-1]['rcurv above'] = -lens_R2
optics[-1]['aperture radius'] = lens_D/2
optics[-1]['origin'] = helper.set_pos([0.0,0.0,20.5/cm])
optics[-1]['euler angles'] = helper.rot_zx_deg(0)

optics.append({})
optics[-1]['object'] = volume.PellinBroca2('P1')
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['dispersion inside'] = material
optics[-1]['size'] = prism_box
optics[-1]['angle'] = characteristic_angle
optics[-1]['origin'] = helper.move(prism_displacement_x,0.0,8/cm)
optics[-1]['euler angles'] = helper.rot_zx(prism_angle+1.15/deg)

optics.append({})
optics[-1]['object'] = volume.SphericalLens('lens2')
optics[-1]['dispersion inside'] = dispersion.ZnSe(mks_length)
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['thickness'] = lens_t
optics[-1]['rcurv beneath'] = lens_R2
optics[-1]['rcurv above'] = lens_R1
optics[-1]['aperture radius'] = lens_D/2
optics[-1]['origin'] = helper.move(8/cm,0.0,-1.8/cm)
optics[-1]['euler angles'] = helper.rot_zx_deg(-90)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('det')
optics[-1]['size'] = (6/inch,2/inch)
optics[-1]['origin'] = helper.move(19.5/cm,0,2/cm)
optics[-1]['euler angles'] = helper.rot_zx_deg(270)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (8,2,4,1)
diagnostics['base filename'] = 'out/test'
