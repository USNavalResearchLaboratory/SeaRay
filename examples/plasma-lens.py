from scipy import constants as C
from scipy.optimize import brentq
import numpy as np
import dispersion
import surface
import volume

# Example input file for parabolic or quartic plasma lenses.
# The parabolic lens produces large area caustic surfaces in
# the half-space upstream of the paraxial focus.
# Therefore the eikonal plane is placed AFTER the paraxial focus
# and the fields in the wave zone are computed by propagating backwards.

# For the quartic lens, most of the caustic is after the paraxial focus.
# Therefore the wave zone calculation is done in the forward direction.

mks_length = 0.8e-6 / (2*np.pi)
sim = []
wave = []
ray = []
optics = []
diagnostics = []
mess = 'Processing input file...\n'

# Preprocessing calculations
# Use thick lens theory to set up channel parameters for given focal length

quartic_correction = True
f = 0.01/mks_length
Fnum = 3.0
r00 = 0.5*f/Fnum # spot size of radiation
t00 = 1e-6*C.c/mks_length # pulse width (not important)
Rlens = 0.01/mks_length
Lch = 0.002/mks_length
c0 = 0.01
h0 = np.sqrt(1-c0)
t = Lch/np.sqrt(1-c0)
Omega = brentq(lambda q : q*np.tan(q) - t/(f-Lch/2), 0.0, 0.999*np.pi/2) / t
c2 = Omega**2
c4 = 0.0
c6 = 0.0
eik_to_caustic = -0.002/mks_length
if quartic_correction:
	c4 = -Omega**4/4
	x0 = 100*r00
	c4 *= 1 + Lch**2*(0.33/x0**2 + 0.5*Omega**2/h0**2 + Omega**2)
	eik_to_caustic *= -1
a00 = 1e-3*Fnum

# Set up dictionaries

for i in range(1):

	sim.append({'mks_length' : mks_length ,
				'mks_time' : mks_length/C.c ,
				'message' : mess})

	wave.append({	# EM 4-potential (eA/mc^2) , component 0 not used
					'a0' : (0.0,1.0,0.0,0.0) ,
					# 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
					'r0' : (t00,r00,r00,t00) ,
					# 4-wavenumber: omega,kx,ky,kz
					'k0' : (1.0,0.0,0.0,1.0) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,0.0,-.006/mks_length),
					'pulse shape' : 'sech',
					'supergaussian exponent' : 8})

	ray.append({	'number' : (128,128,1),
					'loading coordinates' : 'cartesian',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : (-2*r00,2*r00,-2*r00,2*r00,-2*t00,2*t00)})

	optics.append([
		{	'object' : volume.PlasmaChannel('plasma'),
			'dispersion inside' : dispersion.ColdPlasma(),
			'dispersion outside' : dispersion.Vacuum(),
			'radius' : Rlens,
			'length' : Lch,
			'origin' : (0.,0.,0.),
			'euler angles' : (0.,0.,0.),
			'integrator' : 'symplectic',
			'radial coefficients' : (c0,c2,c4,c6),
			'dt' : Lch/1000,
			# Use enough steps to make sure rays reach end of box.
			# Too many steps is OK, SeaRay can adjust down automatically.
			# Too few steps is not OK.
			'steps' : 1200,
			'subcycles' : 10},

		{	'object' : surface.FullWaveProfiler('det'),
			'size' : (.0012/mks_length,.0012/mks_length,.001/mks_length),
			'grid points' : (2048,2048,1),
			'distance to caustic' : eik_to_caustic,
			'origin' : (0.,0.,f-eik_to_caustic)},

		{	'object' : surface.EikonalProfiler('det2'),
			'size' : (.01/mks_length,.01/mks_length),
			'grid points' : (128,128,1),
			'origin' : (0.,0.,2*f)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (5,5,1),
						'base filename' : 'out/test'})
