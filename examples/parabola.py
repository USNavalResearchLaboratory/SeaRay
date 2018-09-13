from scipy import constants as C
import numpy as np
import dispersion
import surface

# Example showing off-axis focusing with parabolic mirror

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

par_f = 0.5/mks_length
y0 = 2*par_f
z0 = -2*par_f
r00 = .05/mks_length # spot size of radiation
t00 = 1e-6*C.c/mks_length # pulse width (not important)
theta = 0.0 # direction of propagation, 0 is +z

# Set up dictionaries

for i in range(1):

	sim.append({'mks_length' : mks_length ,
				'mks_time' : mks_length/C.c ,
				'message' : mess})

	wave.append({	# EM 4-potential (eA/mc^2) , component 0 not used
					'a0' : (0.0,np.cos(theta),0.0,-np.sin(theta)) ,
					# 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
					'r0' : (t00,r00,r00,t00) ,
					# 4-wavenumber: omega,kx,ky,kz
					'k0' : (1.0,np.sin(theta),0.0,np.cos(theta)) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,y0,z0),
					'pulse shape' : 'sech',
					'supergaussian exponent' : 2})

	ray.append({	'number' : (64,64,1),
					'bundle radius' : (.001*r00,.001*r00,.001*r00,.001*r00),
					'loading coordinates' : 'cartesian',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : (-3*r00,3*r00,-3*r00,3*r00,-2*t00,2*t00)})

	optics.append([
		{	'object' : surface.Paraboloid('mirror'),
			'reflective' : True,
			'focal length' : par_f,
			'acceptance angle' : np.pi/1.8,
			'off axis angle' : 0.,
			'euler angles' : (0.,np.pi,0.)},

		{	'object' : surface.FullWaveProfiler('det'),
			'size' : (300e-6/mks_length,300e-6/mks_length,.001/mks_length),
			'grid points' : (1024,1024,1),
			'distance to caustic' : .001/mks_length,
			'origin' : (0.,0.001/mks_length,0.),
			'euler angles' : (0.,np.pi/2,0.)},

		{	'object' : surface.EikonalProfiler('det2'),
			'size' : (0.3/mks_length,0.3/mks_length),
			'grid points' : (128,128,1),
			'origin' : (0.,-par_f,0.),
			'euler angles' : (0.,np.pi/2,0.)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (4,4,1),
						'base filename' : 'out/test'})
