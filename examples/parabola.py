from scipy import constants as C
import numpy as np
import dispersion
import surface
import input_tools

# Example showing 90 degree off-axis parabolic mirror
# Suggested plotter line:
# python ray_plotter out/test det=1,2/0,0/0.1
# Verify spot size and intensity against preprocessing calculation

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
theta = 0.0 # direction of propagation, 0 is +z
par_f = 0.5/mks_length
y0 = 2*par_f
z0 = -2*par_f
r00 = .05/mks_length # spot size of radiation

# N.b. the effective focal length is y0, not par_f
f_num = y0/(2*r00)
t00,band = helper.TransformLimitedBandwidth(w00,'100 fs',4)
a00 = helper.InitialVectorPotential(w00,1.0,y0,f_num)
mess = mess + helper.ParaxialFocusMessage(w00,1.0,y0,f_num)

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
					'focus' : (0.0,0.0,y0,z0),
					'supergaussian exponent' : 2})

	ray.append({	'number' : (128,32,1),
					'bundle radius' : (.001*r00,.001*r00,.001*r00,.001*r00),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : (0,3*r00,0,2*np.pi,-2*t00,2*t00)})

	optics.append([
		{	'object' : surface.Paraboloid('mirror'),
			'reflective' : True,
			'focal length' : par_f,
			'acceptance angle' : np.pi/1.8,
			'off axis angle' : 0.,
			'euler angles' : (0.,np.pi,0.)},

		{	'object' : surface.FullWaveProfiler('det'),
			'size' : (.0004/mks_length,.0004/mks_length,200e-6/mks_length),
			'grid points' : (2048,2048,1),
			'distance to caustic' : .00125/mks_length,
			'origin' : (0.,0.00125/mks_length,0.),
			'euler angles' : (0.,np.pi/2,0.)},

		{	'object' : surface.EikonalProfiler('terminus'),
			'size' : (0.3/mks_length,0.3/mks_length),
			'origin' : (0.,-par_f,0.),
			'euler angles' : (0.,np.pi/2,0.)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (2,32,1),
						'base filename' : 'out/test'})
