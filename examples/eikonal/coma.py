from scipy import constants as C
import numpy as np
import modules.surface as surface
import modules.input_tools as input_tools 

# Off-axis focusing with spherical mirror, leading to coma aberration

# Suggested plotter command:
# python plotter.py out/test det=1,2/0,0
# Observe deformed spot offset from the axis.

# Units and scales

mks_length = 0.8e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
fs = 1e15*mks_length/C.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Control parameters

w00 = 1.0
theta = 5/deg # direction of propagation, 0 is +z
mirror_D = 100/mm
mirror_R = 1000/mm
r00 = 10/mm # spot size of radiation
t00 = 100/fs

# Derived parameters

f_num = (mirror_R/2)/(2*r00)
t00,band = helper.TransformLimitedBandwidth(w00,t00,4)
a00 = helper.InitialVectorPotential(w00,1.0,mirror_R/2,f_num)
rb = r00*bundle_scale
mess = mess + helper.ParaxialFocusMessage(w00,1.0,mirror_R/2,f_num)

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
ray[-1]['number'] = (1,128,128,None)
ray[-1]['bundle radius'] = (None,rb,rb,rb)
ray[-1]['loading coordinates'] = 'cartesian'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = (0.9,1.1) + (-3*r00,3*r00) + (-3*r00,3*r00) + (None,None)

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
optics[-1]['object'] = surface.SphericalCap('mirror')
optics[-1]['reflective'] = True
optics[-1]['radius of sphere'] = mirror_R
optics[-1]['radius of edge'] = mirror_D/2
optics[-1]['origin'] = (0.,0.,0.)
optics[-1]['euler angles'] = (0.,np.pi,0.)

optics.append({})
optics[-1]['object'] = surface.FullWaveProfiler('det')
optics[-1]['size'] = (1/mm,1/mm,1/mm)
optics[-1]['wave grid'] = (1,1024,1024,1)
optics[-1]['distance to caustic'] = 10/mm
optics[-1]['origin'] = (43/mm,0.0,-490/mm)
optics[-1]['euler angles'] = (0.,np.pi,0.)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('terminus')
optics[-1]['size'] = (500/mm,500/mm)
optics[-1]['origin'] = (0.,0.,-1000/mm)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (1,4,4,None)
diagnostics['base filename'] = 'out/test'
