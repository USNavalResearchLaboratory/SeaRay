from scipy import constants as C
import numpy as np
from scipy.optimize import brentq
import dispersion
import surface
import volume
import input_tools

# Example input file for parabolic or quartic plasma lenses.
# The parabolic lens produces large area caustic surfaces in
# the half-space upstream of the paraxial focus.
# Therefore the eikonal plane is placed AFTER the paraxial focus
# and the fields in the wave zone are computed by propagating backwards.

# For the quartic lens, most of the caustic is after the paraxial focus.
# Therefore the wave zone calculation is done in the forward direction.

# Suggested plotter command:
# python ray_plotter.py out/test o31 det=1,2/0,0/0.01
# Note underfocused marginal rays, yet micron scale spot size.

# Units and scales

mks_length = 0.8e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
fs = 1e15*mks_length/C.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Preprocessing calculations
# Use thick lens theory to set up channel parameters for given focal length

theta = 0.0
w00 = 1.0
quartic_correction = True
f = 10/mm
f_num = 3.0
r00 = 0.5*f/f_num # spot size of radiation
Rlens = 10/mm
Lch = 2/mm
c0 = 0.01
h0 = np.sqrt(1-c0)
t = Lch/np.sqrt(1-c0)
Omega = brentq(lambda q : q*np.tan(q) - t/(f-Lch/2), 0.0, 0.999*np.pi/2) / t
c2 = Omega**2
c4 = 0.0
c6 = 0.0
eik_to_caustic = -1/mm
if quartic_correction:
	c4 = -Omega**4/4
	x0 = 100*r00
	c4 *= 1 + Lch**2*(0.33/x0**2 + 0.5*Omega**2/h0**2 + Omega**2)
	eik_to_caustic *= -1

t00,band = helper.TransformLimitedBandwidth(w00,'100 fs',4)
a00 = helper.InitialVectorPotential(w00,1.0,f,f_num)
mess = mess + helper.ParaxialFocusMessage(w00,1.0,f,f_num)
rb = bundle_scale*r00

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
ray[-1]['number'] = (512,4,1)
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
wave[-1]['focus'] = (0.0,0.0,0.0,-6/mm)
wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = volume.AnalyticCylinder('plasma')
optics[-1]['dispersion inside'] = dispersion.ColdPlasma()
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['radius'] = Rlens
optics[-1]['length'] = Lch
optics[-1]['origin'] = (0.,0.,0.)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['density function'] = str(c0)+'+'+str(c2)+'*r2+'+str(c4)+'*r2*r2'
optics[-1]['density lambda'] = lambda x,y,z,r2 : c0 + c2*r2 + c4*r2*r2
optics[-1]['dt'] = Lch/1000
# Use enough steps to make sure rays reach end of box.
# Too many steps is OK, SeaRay can adjust down automatically.
# Too few steps is not OK.
optics[-1]['steps'] = 1200
optics[-1]['subcycles'] = 10

optics.append({})
optics[-1]['object'] = surface.CylindricalProfiler('det')
optics[-1]['integrator'] = 'transform'
optics[-1]['size'] = (2/mm,2/mm,30e-3/mm)
optics[-1]['wave grid'] = (4096,4,1)
optics[-1]['distance to caustic'] = eik_to_caustic
optics[-1]['origin'] = (0.,0.,f - eik_to_caustic)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('terminus')
optics[-1]['size'] = (5/mm,5/mm)
optics[-1]['euler angles'] = (0.0,0.0,0.0)
optics[-1]['origin'] = (0.,0.,20/mm)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (4,4,1)
diagnostics['base filename'] = 'out/test'
