from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume

# Example of aspheric lens focusing and dispersive effects
# The aspheric surface is modeled using a surface mesh

mks_length = 0.4e-6 / (2*np.pi)
bundle_scale = 1e-4
sim = []
wave = []
ray = []
optics = []
diagnostics = []
mess = 'Processing input file...\n'

# Preprocessing calculations

w_las = 1.0
nrefr = np.sqrt(1+dispersion.BK7(mks_length).chi(w_las)[0])
mess = mess + '  BK7 Refractive index at {:.0f} nm = {:.3f}\n'.format(2*np.pi*mks_length*1e9,nrefr)
lens_D = 0.0024/mks_length
lens_R = 0.00142/mks_length
lens_t = 0.001/mks_length
lens_f = 0.0025/mks_length
f_num = 8.0
r00 = 0.5*lens_f/f_num # spot size of radiation
rb = r00*bundle_scale
t00 = 10e-15*C.c/mks_length # pulse width
sigma_w = 2/t00
band = (w_las - 4*sigma_w , w_las + 4*sigma_w)
a00 = 1e-3*f_num
theta = 0 # direction of propagation, 0 is +z
paraxial_e_size = 4.0*f_num/w_las
paraxial_zR = 0.5*w_las*paraxial_e_size**2
focused_a0 = w_las*lens_f*a00/(8*f_num**2)
mess = mess + '  f/# = {:.2f}\n'.format(f_num)
mess = mess + '  Theoretical paraxial spot size (um) = {:.2f}\n'.format(1e6*mks_length*paraxial_e_size)
mess = mess + '  Theoretical paraxial Rayleigh length (um) = {:.2f}\n'.format(1e6*mks_length*paraxial_zR)
mess = mess + '  Focused paraxial intensity (a^2) = {:.2f}\n'.format(focused_a0**2)

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
					'k0' : (w_las,w_las*np.sin(theta),0.0,w_las*np.cos(theta)) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,0.0,-.003/mks_length),
					'supergaussian exponent' : 2})

	ray.append({	'number' : (32,128,4,1),
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
			'mesh points' : (128,128),
			'conic constant' : 0.0,
			'aspheric coefficients' : (-.2861/lens_f**3,1.034/lens_f**5,-9.299/lens_f**7,25.98/lens_f**9,-31.99/lens_f**11),
			#'aspheric coefficients' : (0.0,0.0,0.0,0.0,0.0),
			'origin' : (0.,0.,0.),
			'euler angles' : (0.,0.,0.)},

		{	'object' : surface.CylindricalProfiler('det'),
			'integrator' : 'transform',
			'frequency band' : band,
			'size' : (150e-6/mks_length,150e-6/mks_length,900e-6/mks_length),
			'grid points' : (32,1024,4,32),
			'distance to caustic' : 0.0005/mks_length,
			'origin' : (0.,0.,0.002/mks_length)},

		{	'object' : surface.EikonalProfiler('terminal'),
			'size' : (.0006/mks_length,.0006/mks_length),
			'origin' : (0.,0.,0.003/mks_length)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (1,4,4,1),
						'base filename' : 'out/test'})
