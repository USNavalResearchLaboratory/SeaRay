from scipy import constants as C
import numpy as np
from scipy.optimize import brentq
import dispersion
import ionization
import surface
import volume
import input_tools

# Example input file for 2 color THz generation via UPPE module.

mks_length = 0.8e-6 / (2*np.pi)
sim = []
wave = []
ray = []
optics = []
diagnostics = []
mess = 'Processing input file...\n'

# Preprocessing calculations

helper = input_tools.InputHelper(mks_length)

air = dispersion.Air(mks_length)
w00 = 1.0
r00 = 100e-6 / mks_length
a00 = helper.Wcm2_to_a0(3e13,0.8e-6)
#chi3 = 0.0
chi3 = helper.mks_n2_to_chi3(1.0,5e-19*1e-4)
mess = mess + '  a0 = ' + str(a00) + '\n'
mess = mess + '  chi3 = ' + str(chi3) + '\n'

# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,8.0)
t00,pulse_band = helper.TransformLimitedBandwidth(w00,'35 fs',1.0)

Lprop = .005/mks_length

# Set up dictionaries

for i in range(1):

	sim.append({'mks_length' : mks_length ,
				'mks_time' : mks_length/C.c ,
				'message' : mess})

	wave.append([
				{	# EM 4-potential (eA/mc^2) , component 0 not used
					'a0' : (0.0,a00,0.0,0.0) ,
					# 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
					'r0' : (t00,r00,r00,t00) ,
					# 4-wavenumber: omega,kx,ky,kz
					'k0' : (w00,0.0,0.0,w00) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,0.0,-1.0),
					'supergaussian exponent' : 2},

					# Superposition of the second harmonic
				{	'a0' : (0.0,0.4*a00,0.0,0.0),
					'r0' : (t00,r00,r00,t00),
					'k0' : (2*w00,0.0,0.0,2*w00),
					'focus' : (0.0,0.0,0.0,-1.0),
					'phase' : np.pi/2,
					'supergaussian exponent' : 2}
				])

	ray.append({	'number' : (2048,32,2,1),
					'bundle radius' : (.001*r00,.001*r00,.001*r00,.001*r00),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : band + (0.0,4*r00,0.0,2*np.pi,-2*t00,2*t00)})

	optics.append([
		{	'object' : surface.EikonalProfiler('init'),
			'frequency band' : (1-1e-6,1+1e-6),
			'size' : (6*r00,6*r00),
			'origin' : (0.,0.,-0.5),
			'euler angles' : (0.,0.,0.)},

		{	'object' : volume.TestGrid('air'),
			'propagator' : 'uppe',
			'ionizer' : ionization.ADK(0.5,1.0,2.7e25,mks_length),
			'wave coordinates' : 'cylindrical',
			'wave grid' : (2048,90,1,9),
			'radial coefficients' : (1.0,0.0,0.0,0.0),
			'frequency band' : band,
			'damping filter' : lambda w : 0.5*(1+np.tanh((w-0.05)/0.01)) * 0.5*(1+np.tanh((3.5-w)/0.01)),
			'mesh points' : (2,2,2),
			'subcycles' : 40,
			'density multiplier' : 1.0,
			'dispersion inside' : air,
			'dispersion outside' : dispersion.Vacuum(),
			'chi3' : chi3,
			'size' : (15*r00,15*r00,Lprop),
			'origin' : (0.,0.,Lprop/2),
			'euler angles' : (0.,0.,0.),
			'window speed' : air.GroupVelocityMagnitude(1.0)},

		{	'object' : surface.EikonalProfiler('stop'),
			'frequency band' : (1-1e-6,1+1e-6),
			'size' : (8*r00,8*r00),
			'origin' : (0.,0.,2*Lprop),
			'euler angles' : (0.,0.,0.)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (8,4,2,1),
						'base filename' : 'out/test'})