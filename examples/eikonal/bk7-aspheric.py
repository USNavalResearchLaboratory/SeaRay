from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Example of aspheric lens focusing and dispersive effects
# The aspheric surface is modeled using a surface mesh
# IMPORTANT: frequency and azimuthal nodes in source and detector must match

# Suggested plotter command
# python ray_plotter.py out/test det=0,4/0,0,0
# Illustrates induced chirp via Wigner transform

mks_length = 0.4e-6 / (2*np.pi)
bundle_scale = 1e-4
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
lens_D = 0.0024/mks_length
lens_R = 0.00142/mks_length
lens_t = 0.001/mks_length
f = 0.0025/mks_length
f_num = 8.0
r00 = 0.5*f/f_num # spot size of radiation
rb = r00*bundle_scale
t00,band = helper.TransformLimitedBandwidth(w00,'10 fs',4)
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
					'focus' : (0.0,0.0,0.0,-.003/mks_length),
					'supergaussian exponent' : 2})

	ray.append({	'number' : (64,128,4,1),
					'bundle radius' : (rb,rb,rb,rb),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)})

	optics.append([
		{	'object' : surface.EikonalProfiler('init'),
			'size' : (.002/mks_length,.002/mks_length),
			'origin' : (0.,0.,-0.0029/mks_length)},

		{	'object' : volume.AsphericLens('lens'),
			'dispersion inside' : dispersion.BK7(mks_length),
			'dispersion outside' : dispersion.Vacuum(),
			'thickness' : lens_t,
			'rcurv beneath' : lens_R,
			#'rcurv above' : lens_R*10000,
			'aperture radius' : lens_D/2,
			# The mesh is spherical (polar,azimuth).
			# Poor polar resolution will lead to inaccurate intensity
			# Poor azimuthal resolution will lead to numerical astigmatism.
			'mesh points' : (256,128),
			'conic constant' : 0.0,
			'aspheric coefficients' : (-.2861/f**3,1.034/f**5,-10/f**7,25.98/f**9,-31.99/f**11),
			#'aspheric coefficients' : (0.0,0.0,0.0,0.0,0.0),
			'origin' : (0.,0.,0.),
			'euler angles' : (0.,0.,0.)},

		{	'object' : surface.CylindricalProfiler('det'),
			'integrator' : 'transform',
			'frequency band' : band,
			'size' : (150e-6/mks_length,150e-6/mks_length,900e-6/mks_length),
			'wave grid' : (64,256,4,1),
			'distance to caustic' : 0.0005/mks_length,
			'origin' : (0.,0.,0.002/mks_length)},

		{	'object' : surface.EikonalProfiler('terminus'),
			'size' : (.002/mks_length,.002/mks_length),
			'origin' : (0.,0.,0.003/mks_length)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (1,4,4,1),
						'base filename' : 'out/test'})
