from scipy import constants as C
import numpy as np
import dispersion
import surface
import input_tools

# Off-axis focusing with spherical mirror, leading to coma aberration

mks_length = 0.8e-6 / (2*np.pi)
sim = []
wave = []
ray = []
optics = []
diagnostics = []
mess = 'Processing input file...\n'

# Preprocessing calculations

helper = input_tools.InputHelper(mks_length)

w00 = 1.0
theta = 5*np.pi/180 # direction of propagation, 0 is +z
mirror_D = 0.1/mks_length
mirror_R = 1/mks_length
r00 = .01/mks_length # spot size of radiation
f_num = (mirror_R/2)/(2*r00)
t00,band = helper.TransformLimitedBandwidth(w00,'100 fs',4)
a00 = helper.InitialVectorPotential(w00,1.0,mirror_R/2,f_num)
mess = mess + helper.ParaxialFocusMessage(w00,1.0,mirror_R/2,f_num)

# Set up dictionaries

for i in range(1):

	sim.append({'mks_length' : mks_length ,
				'mks_time' : mks_length/C.c ,
				'message' : mess})

	wave.append({	# EM 4-potential (eA/mc^2) , component 0 not used
					'a0' : (0.0,a00*np.cos(theta),0.0,-a00*np.sin(theta)) ,
					# 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
					'r0' : (t00,r00,r00,t00) ,
					# 4-wavenumber: omega,kx,ky,kz
					'k0' : (w00,w00*np.sin(theta),0.0,w00*np.cos(theta)) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,0.0,-.1/mks_length),
					'supergaussian exponent' : 2})

	ray.append({	'number' : (128,128,1),
					'bundle radius' : (.001*r00,.001*r00,.001*r00,.001*r00),
					'loading coordinates' : 'cartesian',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : (-3*r00,3*r00,-3*r00,3*r00,-2*t00,2*t00)})

	optics.append([
		{	'object' : surface.SphericalCap('mirror'),
			'reflective' : True,
			'radius of sphere' : mirror_R,
			'radius of edge' : mirror_D/2,
			'origin' : (0.,0.,0.),
			'euler angles' : (0.,np.pi,0.)},

		{	'object' : surface.FullWaveProfiler('det'),
			'size' : (.001/mks_length,.001/mks_length,.001/mks_length),
			'grid points' : (1024,1024,1),
			'distance to caustic' : .01/mks_length,
			'origin' : (0.043/mks_length,0.0,-.49/mks_length),
			'euler angles' : (0.,np.pi,0.)},

		{	'object' : surface.EikonalProfiler('terminus'),
			'size' : (0.5/mks_length,0.5/mks_length),
			'origin' : (0.,0.,-1/mks_length)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (5,5,1),
						'base filename' : 'out/test'})
