from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Example input file for axisymmetric UPPE wave equation.
# Illustrates self focusing, self phase modulation, and group velocity dispersion in glass
# Effects are most easily observed using the interactive viewer

mks_length = 0.8e-6 / (2*np.pi)
sim = []
wave = []
ray = []
optics = []
diagnostics = []
mess = 'Processing input file...\n'

# Preprocessing calculations

helper = input_tools.InputHelper(mks_length)

glass = dispersion.BK7(mks_length)
# Suppress out of band frequencies
glass.add_opacity_region(2000.0,0.1e-6,0.6e-6)
glass.add_opacity_region(2000.0,1.2e-6,4e-6)
w00 = 1.0
r00 = 100e-6 / mks_length
a00 = helper.Wcm2_to_a0(6e12,0.8e-6)
chi3 = helper.mks_n2_to_chi3(1.5,1e-20)
mess = mess + '  a0 = ' + str(a00) + '\n'
mess = mess + '  chi3 = ' + str(chi3) + '\n'

# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,4.0)
t00,pulse_band = helper.TransformLimitedBandwidth(w00,'15 fs',1.0)

# Work out the dispersion length
vg1 = glass.GroupVelocityMagnitude(pulse_band[0])
vg2 = glass.GroupVelocityMagnitude(pulse_band[1])
Ldisp = t00 / np.abs(1/vg1 - 1/vg2)
Lprop = 4*Ldisp
mess = mess + '  Red speed = ' + str(vg1) + '\n'
mess = mess + '  Blue speed = ' + str(vg2) + '\n'
mess = mess + '  Dispersion length = ' + str(1e3*Ldisp*mks_length) + ' mm\n'
mess = mess + '  Propagation length = ' + str(1e3*Lprop*mks_length) + ' mm\n'

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
					'k0' : (w00,0.0,0.0,w00) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,0.0,-1.0),
					'supergaussian exponent' : 2})

	ray.append({	'number' : (1025,64,4,1),
					'bundle radius' : (.001*r00,.001*r00,.001*r00,.001*r00),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)})

	optics.append([
		{	'object' : surface.EikonalProfiler('start'),
			'size' : (6*r00,6*r00),
			'origin' : (0.,0.,-0.5),
			'euler angles' : (0.,0.,0.)},

		{	'object' : volume.AnalyticBox('glass'),
			'propagator' : 'uppe',
			'wave coordinates' : 'cylindrical',
			'wave grid' : (1025,64,1,9),
			'density function' : '1.0',
			'density lambda' : lambda x,y,z,r2 : np.ones(x.shape),
			'density multiplier' : 1.0,
			'frequency band' : band,
			'subcycles' : 1,
			'dispersion inside' : glass,
			'dispersion outside' : dispersion.Vacuum(),
			'chi3' : chi3,
			'size' : (6*r00,6*r00,Lprop),
			'origin' : (0.,0.,Lprop/2),
			'euler angles' : (0.,0.,0.),
			'window speed' : glass.GroupVelocityMagnitude(1.0)},

		{	'object' : surface.EikonalProfiler('stop'),
			'size' : (8*r00,8*r00),
			'origin' : (0.,0.,2*Lprop),
			'euler angles' : (0.,0.,0.)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (8,4,4,1),
						'base filename' : 'out/test'})
