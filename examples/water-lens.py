from scipy import constants as C
import numpy as np
import dispersion
import surface

# Example input file focusing light through a water lens.
# Lens can be analytical or constructed from a mesh.
# Can be used to study effects of mesh resolution.

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

nrefr = np.sqrt(1+dispersion.LiquidWater(mks_length).chi(1.0)[0])
mess = mess + '  H2O Refractive index at {:.0f} nm = {:.3f}\n'.format(2*np.pi*mks_length*1e9,nrefr)
lens_D = 0.3/mks_length
lens_R = 1.5/mks_length
lens_f = lens_R*(nrefr/(nrefr-1))
mess = mess + '  H2O surface lens focal length = {:.2f} meters\n'.format(lens_f*mks_length)
r00 = .02/mks_length # spot size of radiation
t00 = 1e-6*C.c/mks_length # pulse width (not important)
theta = np.pi # direction of propagation, 0 is toward vacuum, pi is toward water
f_num = lens_f/(2*r00)
paraxial_e_size = 4.0*f_num/1.0
paraxial_zR = 0.5*1.0*paraxial_e_size**2
mess = mess + '  f/# = {:.2f}\n'.format(f_num)
mess = mess + '  Theoretical paraxial spot size (mm) = {:.3f}\n'.format(1e3*mks_length*paraxial_e_size)
mess = mess + '  Theoretical paraxial Rayleigh length (mm) = {:.2f}\n'.format(1e3*mks_length*paraxial_zR)

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
					'focus' : (0.0,0.0,0.0,1/mks_length),
					'pulse shape' : 'sech',
					'supergaussian exponent' : 8})

	ray.append({	'number' : (64,64,1),
					'loading coordinates' : 'cartesian',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : (-2*r00,2*r00,-2*r00,2*r00,-2*t00,2*t00)})

	optics.append([

		# Here is the dictionary for the analytical lens model.
		# Use this for exact model of spherical lens focusing.
		# N.b. it is not a perfect focus due to realistic spherical aberration.
		# {	'object' : surface.SphericalCap('analytical-lens'),
		# 	'radius of sphere' : lens_R,
		# 	'radius of edge' : lens_D/2,
		# 	'origin' : (0.,0.,0.),
		# 	'euler angles' : (0.,np.pi,0.),
		# 	'dispersion above' : dispersion.LiquidWater(mks_length),
		# 	'dispersion beneath' : dispersion.Vacuum()},

		# Here is the dictionary for the numerical lens model.
		# The spherical surface is approximated by a triangular meshes.
		# On top of real aberrations there will be numerical ones.
		# This will also increase the run time compared to analytical lens.
		# Explore the effect of changing the number of mesh points.
		{	'object' : surface.TestMap('lens'),
			'mesh points' : (256,256),
			'size' : (lens_D,lens_D),
			'curvature radius' : lens_R,
			'origin' : (0.,0.,0.),
			'euler angles' : (0.,0.,0.),
			'dispersion beneath' : dispersion.LiquidWater(mks_length),
			'dispersion above' : dispersion.Vacuum()},

		{	'object' : surface.FullWaveProfiler('det'),
			'size' : (.003/mks_length,.003/mks_length,.001/mks_length),
			'grid points' : (1024,1024,1),
			'distance to caustic' : 0.18/mks_length,
			'dispersion beneath' : dispersion.LiquidWater(mks_length),
			'dispersion above' : dispersion.LiquidWater(mks_length),
			'origin' : (0.,0.,-0.97*lens_f),
			'euler angles' : (0.,np.pi,0.)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (5,5,1),
						'base filename' : 'out/test'})
