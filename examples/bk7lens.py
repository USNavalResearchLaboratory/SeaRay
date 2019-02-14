from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Example of focusing with an extra-thick bi-convex spherical lens
# Illustrates difficulty of full wave reconstruction for highly aberrated beams
# Requires at least 16384^2 grid points for reasonable results; more tends to stress system memory.
# A trick that can be used is to put the eikonal plane after the focus and reverse the Helmholtz propagator,
# but this input file does not do that.

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
theta = 0 # direction of propagation, 0 is +z
nrefr = np.sqrt(1+dispersion.BK7(mks_length).chi(w00)[0])
mess = mess + '  BK7 Refractive index at {:.0f} nm = {:.3f}\n'.format(2*np.pi*mks_length*1e9,nrefr)
lens_D = 0.1/mks_length
lens_R = 0.1/mks_length
lens_t = 0.03/mks_length
f = 1/(nrefr-1)/(1/lens_R + 1/lens_R - (nrefr-1)*lens_t/(nrefr*lens_R*lens_R))
mess = mess + '  thick lens focal length = {:.2f} meters\n'.format(f*mks_length)
r00 = .01/mks_length # spot size of radiation
f_num = f/(2*r00)
t00,band = helper.TransformLimitedBandwidth(w00,'100 fs',4)
a00 = helper.InitialVectorPotential(w00,1.0,f,f_num)
mess = mess + helper.ParaxialFocusMessage(w00,1.0,f,f_num)

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
			'wave grid' : (1024,1024,1),
			'distance to caustic' : .057/mks_length,
			'origin' : (0.,0.,0.05/mks_length)},

		{	'object' : surface.EikonalProfiler('terminus'),
			'size' : (.1/mks_length,.1/mks_length),
			'origin' : (0.,0.,.15/mks_length)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (4,4,1),
						'base filename' : 'out/test'})
