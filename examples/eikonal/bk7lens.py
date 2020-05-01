from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Example of focusing with an extra-thick bi-convex spherical lens
# Illustrates difficulty of full wave reconstruction for highly aberrated beams
# Requires at least 16384^2 grid points for reasonable results; more tends to stress system memory.
# A trick that can be used is to put the eikonal plane after the focus and reverse the Helmholtz propagator,
# but this input file does not do that.

# Suggested plotter command:
# python ray_plotter.py out/test o3d
# The imperfection of the focus should be obvious

# Units and scales

mks_length = 0.8e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
fs = 1e15*mks_length/C.c
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Control parameters

w00 = 1.0
theta = 0 # direction of propagation, 0 is +z
lens_D = 100/mm
lens_R = 100/mm
lens_t = 30/mm
r00 = 10/mm # spot size of radiation
t_pulse = 100/fs
material = dispersion.BK7(mks_length)

# Derived parameters

nrefr = np.sqrt(1+material.chi(w00)[0])
mess = mess + '  Lens refractive index at {:.0f} nm = {:.3f}\n'.format(2*np.pi*mks_length*1e9,nrefr)
f = 1/(nrefr-1)/(1/lens_R + 1/lens_R - (nrefr-1)*lens_t/(nrefr*lens_R*lens_R))
mess = mess + '  thick lens focal length = {:.2f} meters\n'.format(f*mks_length)
f_num = f/(2*r00)
t00,band = helper.TransformLimitedBandwidth(w00,t_pulse,4)
a00 = helper.InitialVectorPotential(w00,1.0,f,f_num)
rb = r00*bundle_scale
mess = mess + helper.ParaxialFocusMessage(w00,1.0,f,f_num)

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
ray[-1]['number'] = (128,128,1)
ray[-1]['bundle radius'] = (rb,rb,rb,rb)
ray[-1]['loading coordinates'] = 'cartesian'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = (-3*r00,3*r00,-3*r00,3*r00,-2*t00,2*t00)

wave.append({})
wave[-1]['a0'] = (0.0,a00*np.cos(theta),0.0,-a00*np.sin(theta)) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,w00*np.sin(theta),0.0,w00*np.cos(theta)) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = (0.0,0.0,0.0,-100/mm)
wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = volume.SphericalLens('lens')
optics[-1]['dispersion inside'] = dispersion.BK7(mks_length)
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['thickness'] = lens_t
optics[-1]['rcurv beneath'] = lens_R
optics[-1]['rcurv above'] = -lens_R
optics[-1]['aperture radius'] = lens_D/2
optics[-1]['origin'] = (0.,0.,0.)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = surface.FullWaveProfiler('det')
optics[-1]['size'] = (.02/mks_length,.02/mks_length,.001/mks_length)
optics[-1]['wave grid'] = (1024,1024,1)
optics[-1]['distance to caustic'] = .057/mks_length
optics[-1]['origin'] = (0.,0.,0.05/mks_length)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('terminus')
optics[-1]['size'] = (.1/mks_length,.1/mks_length)
optics[-1]['origin'] = (0.,0.,.15/mks_length)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (4,4,1)
diagnostics['base filename'] = 'out/test'
