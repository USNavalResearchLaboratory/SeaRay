from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Example of aspheric lens focusing and dispersive effects
# The aspheric surface is modeled using a surface mesh
# IMPORTANT: frequency and azimuthal nodes in source and detector must match

# Suggested plotter command
# python ray_plotter.py out/test det=0,4/0,0,0
# Illustrates induced chirp via Wigner transform

# Units and scales

mks_length = 0.4e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
fs = 1e15*mks_length/C.c
w00 = 1.0
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Control parameters

theta = 0 # direction of propagation, 0 is +z
lens_D = 2.4/mm # lens diameter
lens_t = 1/mm # lens thickness
f = 2.5/mm # focal length
lens_R = 1.42/mm # lens radius of curvature, derived externally based on f (see extras/aspheric.py)
f_num = 8.0
t_pulse = 10/fs
material = dispersion.BK7(mks_length)

# Derived parameters

nrefr = np.sqrt(1+material.chi(w00)[0])
mess = mess + '  lens refractive index at {:.0f} nm = {:.3f}\n'.format(2*np.pi*mks_length*1e9,nrefr)
r00 = 0.5*f/f_num # spot size of radiation
rb = r00*bundle_scale
t00,band = helper.TransformLimitedBandwidth(w00,t_pulse,4)
a00 = helper.InitialVectorPotential(w00,1.0,f,f_num)
mess = mess + helper.ParaxialFocusMessage(w00,1.0,f,f_num)

# Set up dictionaries

sim = {}
wave = []
ray = {}
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = mess

ray['number'] = (64,128,4,1)
ray['bundle radius'] = (rb,rb,rb,rb)
ray['loading coordinates'] = 'cylindrical'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray['box'] = band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)

wave.append({})
wave[-1]['a0'] = (0.0,a00*np.cos(theta),0.0,-a00*np.sin(theta)) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,w00*np.sin(theta),0.0,w00*np.cos(theta)) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = (0.0,0.0,0.0,-3/mm)
wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('init')
optics[-1]['size'] = (.002/mks_length,.002/mks_length)
optics[-1]['origin'] = (0.,0.,-0.0029/mks_length)

optics.append({})
optics[-1]['object'] = volume.AsphericLens('lens')
optics[-1]['dispersion inside'] = material
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['thickness'] = lens_t
optics[-1]['rcurv beneath'] = lens_R
#optics[-1]['rcurv above'] = lens_R*10000,
optics[-1]['aperture radius'] = lens_D/2
# The mesh is spherical (polar,azimuth).
# Poor polar resolution will lead to inaccurate intensity
# Poor azimuthal resolution will lead to numerical astigmatism.
optics[-1]['mesh points'] = (256,128)
optics[-1]['conic constant'] = 0.0
optics[-1]['aspheric coefficients'] = (-.2861/f**3,1.034/f**5,-10/f**7,25.98/f**9,-31.99/f**11)
#optics[-1]['aspheric coefficients'] = (0.0,0.0,0.0,0.0,0.0)
optics[-1]['origin'] = (0.,0.,0.)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = surface.CylindricalProfiler('det')
optics[-1]['integrator'] = 'transform'
optics[-1]['frequency band'] = band
optics[-1]['size'] = (150e-6/mks_length,150e-6/mks_length,900e-6/mks_length)
optics[-1]['wave grid'] = (64,256,4,1)
optics[-1]['distance to caustic'] = 0.0005/mks_length
optics[-1]['origin'] = (0.,0.,0.002/mks_length)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('terminus')
optics[-1]['size'] = (.002/mks_length,.002/mks_length)
optics[-1]['origin'] = (0.,0.,0.003/mks_length)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (1,4,4,1)
diagnostics['base filename'] = 'out/test'
