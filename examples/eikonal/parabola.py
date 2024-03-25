from scipy import constants as C
import numpy as np
import modules.surface as surface
import modules.input_tools as input_tools 

# Example showing 90 degree off-axis parabolic mirror
# Suggested plotter command:
# python plotter.py out/test det=1,2/0,0/0.1
# Verify spot size and intensity against preprocessing calculation

# Units and scales

mks_length = 0.8e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
fs = 1e15*mks_length/C.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Preprocessing calculations

w00 = 1.0
theta = 0.0 # direction of propagation, 0 is +z
par_f = 500/mm
y0 = 2*par_f
z0 = -2*par_f
r00 = 50/mm # spot size of radiation
t00 = 100/fs

# N.b. the effective focal length is y0, not par_f
f_num = y0/(2*r00)
rb = bundle_scale*r00
t00,band = helper.TransformLimitedBandwidth(w00,t00,4)
a00 = helper.InitialVectorPotential(w00,1.0,y0,f_num)
mess = mess + helper.ParaxialFocusMessage(w00,1.0,y0,f_num)

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
ray[-1]['number'] = (1,128,32,None)
ray[-1]['bundle radius'] = (None,rb,rb,rb)
ray[-1]['loading coordinates'] = 'cylindrical'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = band + (0.0,3*r00) + (0.0,2*np.pi) + (None,None)

wave.append({})
wave[-1]['a0'] = (0.0,a00*np.cos(theta),0.0,-a00*np.sin(theta)) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,w00*np.sin(theta),0.0,w00*np.cos(theta)) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = (0.0,0.0,y0,z0)
wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = surface.Paraboloid('mirror')
optics[-1]['reflective'] = True
optics[-1]['focal length'] = par_f
optics[-1]['acceptance angle'] = 180/deg/1.8
optics[-1]['off axis angle'] = 0.
optics[-1]['euler angles'] = (0.,180/deg,0.)

optics.append({})
optics[-1]['object'] = surface.FullWaveProfiler('det')
optics[-1]['size'] = (.4/mm,.4/mm,.2/mm)
optics[-1]['wave grid'] = (1,2048,2048,1)
optics[-1]['distance to caustic'] = 1.25/mm
optics[-1]['origin'] = (0.,1.25/mm,0.)
optics[-1]['euler angles'] = (0.,90/deg,0.)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('terminus')
optics[-1]['size'] = (300/mm,300/mm)
optics[-1]['origin'] = (0.,-par_f,0.)
optics[-1]['euler angles'] = (0.,90/deg,0.)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (1,2,32,None)
diagnostics['base filename'] = 'out/test'
