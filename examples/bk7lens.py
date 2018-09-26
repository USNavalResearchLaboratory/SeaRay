from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume

# Example of focusing with an extra-thick bi-convex spherical lens
# Illustrates difficulty of full wave reconstruction for highly aberrated beams
# Requires at least 16384^2 grid points for reasonable results; more tends to stress system memory.
# A trick that can be used is to put the eikonal plane after the focus and reverse the Helmholtz propagator,
# but this input file does not do that.

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

nrefr = np.sqrt(1+dispersion.BK7(mks_length).chi(1.0)[0])
mess = mess + '  BK7 Refractive index at {:.0f} nm = {:.3f}\n'.format(2*np.pi*mks_length*1e9,nrefr)
lens_D = 0.1/mks_length
lens_R = 0.1/mks_length
lens_t = 0.03/mks_length
lens_f = 1/(nrefr-1)/(1/lens_R + 1/lens_R - (nrefr-1)*lens_t/(nrefr*lens_R*lens_R))
mess = mess + '  thick lens focal length = {:.2f} meters\n'.format(lens_f*mks_length)
r00 = .01/mks_length # spot size of radiation
t00 = 1e-6*C.c/mks_length # pulse width (not important)
theta = 0 # direction of propagation, 0 is +z
f_num = lens_f/(2*r00)
a00 = 1e-3*f_num
paraxial_e_size = 4.0*f_num/1.0
paraxial_zR = 0.5*1.0*paraxial_e_size**2
mess = mess + '  f/# = {:.2f}\n'.format(f_num)
mess = mess + '  Theoretical paraxial spot size (um) = {:.3f}\n'.format(1e6*mks_length*paraxial_e_size)
mess = mess + '  Theoretical paraxial Rayleigh length (um) = {:.2f}\n'.format(1e6*mks_length*paraxial_zR)

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
		{	'object' : volume.SphericalLens('lens'),
			'dispersion inside' : dispersion.BK7(mks_length),
			'dispersion outside' : dispersion.Vacuum(),
			'thickness' : lens_t,
			'rcurv beneath' : lens_R,
			'rcurv above' : -lens_R,
			'aperture radius' : lens_D/2,
			'origin' : (0.,0.,0.),
			'euler angles' : (0.,0.,0.)},

		{	'object' : surface.FullWaveProfiler('det'),
			'size' : (.02/mks_length,.02/mks_length,.001/mks_length),
			'grid points' : (1024,1024,1),
			'distance to caustic' : .057/mks_length,
			'origin' : (0.,0.,0.05/mks_length)},

		{	'object' : surface.EikonalProfiler('terminal'),
			'size' : (.1/mks_length,.1/mks_length),
			'origin' : (0.,0.,.15/mks_length)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (4,4,1),
						'base filename' : 'out/test'})
