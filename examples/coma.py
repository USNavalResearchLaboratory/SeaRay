from scipy import constants as C
import numpy as np
import dispersion
import surface

# Example showing off-axis focusing with spherical mirror

# The input file must do one thing:
#   create dictionaries sim[i], wave[i], ray[i], optics[i], diagnostics[i].
# Here, i is a list index, used to handle batch jobs.
# The dictionaries can be created using any means available in python
# SeaRay will only look at the dictionaries
# Best practice in post-processing is also to look only at the dictionaries

mks_length = 0.8e-6 / (2*np.pi)
sim = []
wave = []
ray = []
optics = []
diagnostics = []
mess = 'Processing input file...\n'

# Preprocessing calculations

mirror_D = 0.1/mks_length
mirror_R = 1/mks_length
r00 = .01/mks_length # spot size of radiation
t00 = 1e-6*C.c/mks_length # pulse width (not important)
theta = 5*np.pi/180 # direction of propagation, 0 is +z
f_num = (mirror_R/2)/(2*r00)
a00 = 1e-3*f_num

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
					'k0' : (1.0,np.sin(theta),0.0,np.cos(theta)) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,0.0,-.1/mks_length),
					'pulse shape' : 'sech',
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

		{	'object' : surface.EikonalProfiler('terminal'),
			'size' : (0.5/mks_length,0.5/mks_length),
			'origin' : (0.,0.,-1/mks_length)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (5,5,1),
						'base filename' : 'out/test'})
