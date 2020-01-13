from scipy import constants as C
import numpy as np
from scipy.optimize import brentq
import dispersion
import surface
import volume
import input_tools

# Example input file for 3D paraxial wave equation.
# Rays are converted to paraxial wave which propagates through focus.
# Rays are relaunched at a symmetric point downstream.

# Suggested plotter commands:
# python ray_plotter out/test exit=1,8
#   Upturned parabolas correspond to rays defocusing out of wave zone.
# python ray_plotter out/test exit=1,8/4,.999,1.001/6,-.001,.001
#   Isolates an individual parabola (fixes w=1 and ky=0)
# The interactive viewer can be used to examine the paraxial wave itself.

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
f = .1/mks_length
f_num = 50.0
r00 = f/(2*f_num) # spot size of radiation
t00,band = helper.TransformLimitedBandwidth(w00,'100 fs',8)
a00,waist,zR = helper.ParaxialParameters(w00,1.0,f,f_num)
mess = mess + helper.ParaxialFocusMessage(w00,1.0,f,f_num)

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
					'focus' : (1.001*f,0.0,0.0,f),
					'supergaussian exponent' : 2})

	ray.append({	'number' : (64,128,8,1),
					'bundle radius' : (.001*r00,.001*r00,.001*r00,.001*r00),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)})

	optics.append([
		{	'object' : surface.EikonalProfiler('start'),
			'size' : (f/8,f/8),
			'origin' : (0.,0.,0.),
			'euler angles' : (0.,0.,0.)},

		{	'object' : volume.AnalyticBox('vacuum'),
			'density function' : '1.0',
			'density lambda' : lambda x,y,z,r2 : np.ones(x.shape),
			'density multiplier' : 1.0,
			'frequency band' : band,
			'wave grid' : (64,256,256,9),
			'wave coordinates' : 'cartesian',
			'dispersion inside' : dispersion.Vacuum(),
			'dispersion outside' : dispersion.Vacuum(),
			'size' : (36*waist,36*waist,8*zR),
			'origin' : (0.,0.,f),
			'euler angles' : (0.,0.,0.),
			'propagator' : 'paraxial',
			'subcycles' : 1},

		{	'object' : surface.EikonalProfiler('exit'),
			'size' : (f/8,f/8),
			'origin' : (0.,0.,f+4*zR+10.0),
			'euler angles' : (0.,0.,0.)},

		{	'object' : surface.EikonalProfiler('stop'),
			'size' : (f/8,f/8),
			'origin' : (0.,0.,2*f),
			'euler angles' : (0.,0.,0.)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (1,16,4,1),
						'base filename' : 'out/test'})
