from scipy import constants as C
import numpy as np
from scipy.optimize import brentq
import dispersion
import surface
import volume

# Example input file for ray-in-cell propagation through ideal form plasma lens.
# The alternate test case creates a quartic lens on a grid.
# The ideal form lens data must be in ./aux.  Generate with synth-lens.py.

mks_length = 0.8e-6 / (2*np.pi)
sim = []
wave = []
ray = []
optics = []
diagnostics = []
mess = 'Processing input file...\n'

# Preprocessing calculations
# Use thick lens theory to set up channel parameters for given focal length

ideal_form = False
f = 0.01/mks_length
Rlens = 0.75*f
if ideal_form:
	Lch = 1.5*f
	lens_object = volume.AxisymmetricGrid('plasma')
else:
	Lch = 0.2*f
	lens_object = volume.TestGrid('plasma')
Fnum = 1.0
r00 = 0.5*f/Fnum # spot size of radiation
t00 = 1e-6*C.c/mks_length # pulse width (not important)
c0 = 0.01
h0 = np.sqrt(1-c0)
t = Lch/np.sqrt(1-c0)
Omega = brentq(lambda q : q*np.tan(q) - t/(f-Lch/2), 0.0, 0.999*np.pi/2) / t
c2 = Omega**2
c4 = -Omega**4/4
x0 = 100*r00
c4 *= 1 + Lch**2*(0.33/x0**2 + 0.5*Omega**2/h0**2 + Omega**2)
c6 = 0.0
eik_to_caustic = 0.001/mks_length
a00 = 1e-3*Fnum

# Set up dictionaries

for i in range(1):

	sim.append({'mks_length' : mks_length ,
				'mks_time' : mks_length/C.c ,
				'message' : mess})

	wave.append({	# EM 4-potential (eA/mc^2) , component 0 not used
					'a0' : (0.0,a00,0.0,0.0) ,
					# 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
					'r0' : (t00,r00,r00,t00) ,
					# 4-wavenumber: omega,kx,ky,kz
					'k0' : (1.0,0.0,0.0,1.0) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,0.0,-f),
					'pulse shape' : 'sech',
					'supergaussian exponent' : 8})

	ray.append({	'number' : (512,4,1),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : (0,1.4*r00,0.0,2*np.pi,-2*t00,2*t00)})

	optics.append([
		{	'object' : lens_object,
			'radial coefficients' : (c0,c2,c4,c6),
			'mesh points' : (400,400),
			'file' : 'extras/ideal-form.npy',
			'density multiplier' : 1.0,
			'dispersion inside' : dispersion.ColdPlasma(),
			'dispersion outside' : dispersion.Vacuum(),
			'radius' : Rlens,
			'length' : Lch,
			'origin' : (0.,0.,0.),
			'euler angles' : (0.,0.,0.),
			'integrator' : 'symplectic',
			'dt' : Lch/1000,
			'steps' : 1500,
			'subcycles' : 10},

		# {	'object' : surface.CylindricalProfiler('det'),
		# 	'integrator' : 'transform',
		# 	'size' : (.001/mks_length,.001/mks_length,.0001/mks_length),
		# 	'grid points' : (16000,4,256),
		# 	'distance to caustic' : eik_to_caustic,
		# 	'origin' : (0.,0.,f - eik_to_caustic)},

		{	'object' : surface.EikonalProfiler('det2'),
			'size' : (.01/mks_length,.01/mks_length),
			'grid points' : (128,128,1),
			'euler angles' : (0.0,0.0,0.0),
			'origin' : (0.,0.,.015/mks_length)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (12,4,1),
						'base filename' : 'out/test'})
