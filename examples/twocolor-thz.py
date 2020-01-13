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

prop_range = (-0.1/mks_length,0.0/mks_length)
L = prop_range[1]-prop_range[0]
# air = dispersion.HumidAir(mks_length,0.4,1e-3)
# air.add_opacity_region(40.0,0.05e-6,0.25e-6)
# air.add_opacity_region(5.0,13e-6,17e-6)
# air.add_opacity_region(40.0,100e-6,.001)
air = dispersion.Vacuum()
ionizer = ionization.ADK(0.5,1.0,2.7e25,mks_length,terms=4)
w00 = 1.0
r00 = 0.005 / mks_length
P0_mks = 7e-3 / 80e-15
I0_mks = 2*P0_mks/(np.pi*r00**2*mks_length**2)
a800 = helper.Wcm2_to_a0(I0_mks*1e-4,0.8e-6)
a400 = helper.Wcm2_to_a0(0.1*I0_mks*1e-4,0.4e-6)
chi3 = 0.0#helper.mks_n2_to_chi3(1.0,5e-19*1e-4)
mess = mess + '  a800 = ' + str(a800) + '\n'
mess = mess + '  a400 = ' + str(a400) + '\n'
mess = mess + '  chi3 = ' + str(chi3) + '\n'

# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,4.5)
t00,pulse_band = helper.TransformLimitedBandwidth(w00,'80 fs',1.0)

# Set up dictionaries

for i in range(1):

	sim.append({'mks_length' : mks_length ,
				'mks_time' : mks_length/C.c ,
				'message' : mess})

	wave.append([
				{	# EM 4-potential (eA/mc^2) , component 0 not used
					'a0' : (0.0,a800,0.0,0.0) ,
					# 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
					'r0' : (t00,r00,r00,t00) ,
					# 4-wavenumber: omega,kx,ky,kz
					'k0' : (w00,0.0,0.0,-w00) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,0.0,-0.95/mks_length),
					'supergaussian exponent' : 2},

					# Superposition of the second harmonic
				{	'a0' : (0.0,a400,0.0,0.0),
					'r0' : (t00,r00,r00,t00),
					'k0' : (2*w00,0.0,0.0,-2*w00),
					'focus' : (0.0,0.0,0.0,-0.95/mks_length),
					'phase' : np.pi/2,
					'supergaussian exponent' : 2}
				])

	ray.append({	'number' : (2049,64,2,1),
					'bundle radius' : (.001*r00,.001*r00,.001*r00,.001*r00),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)})

	optics.append([

		{	'object' : surface.SphericalCap('M1'),
			'reflective' : True,
			'radius of sphere' : 2/mks_length,
			'radius of edge' : .0125/mks_length,
			'origin' : (0.,0.,-1/mks_length),
			'euler angles' : (0.,0.,0.)},

		{	'object' : volume.AnalyticBox('air'),
			'propagator' : 'uppe',
			'ionizer' : ionizer,
			'wave coordinates' : 'cylindrical',
			'wave grid' : (2049,128,1,7),
			'density function' : 'exp(-4*x.s3*x.s3/'+str(L**2)+')',
			'density lambda' : lambda x,y,z,r2 : np.exp(-4*z**2/L**2),
			'density multiplier' : 1.0,
			'frequency band' : band,
			'subcycles' : 10,
			'dispersion inside' : air,
			'dispersion outside' : dispersion.Vacuum(),
			'chi3' : chi3,
			'size' : (6e-3/mks_length,6e-3/mks_length,prop_range[1]-prop_range[0]),
			'origin' : (0.,0.,(prop_range[0]+prop_range[1])/2),
			'euler angles' : (0.,0.,0.),
			'window speed' : air.GroupVelocityMagnitude(1.0)},

		{	'object' : surface.EikonalProfiler('stop'),
			'frequency band' : (0,3),
			'size' : (10*r00,10*r00),
			'origin' : (0.,0.,0.4/mks_length),
			'euler angles' : (0.,0.,0.)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (4,8,2,1),
						'base filename' : 'out/test'})
