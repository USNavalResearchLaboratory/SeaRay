from scipy import constants as C
import numpy as np
import surface
import input_tools

# Example of diffraction grating

# Suggested plotter command
# python ray_plotter.py out/test o3d det=4,5

mks_length = 0.8e-6 / (2*np.pi)
inch = mks_length*100/2.54
bundle_scale = 1e-4
sim = []
wave = []
ray = []
optics = []
diagnostics = []
mess = 'Processing input file...\n'

# Preprocessing calculations

helper = input_tools.InputHelper(mks_length)

# Setup pulse parameters

w00 = 1.0
r00 = 3e-3/mks_length # spot size of radiation
rb = r00*bundle_scale
t00,band = helper.TransformLimitedBandwidth(w00,'10 fs',1)
a00 = 1.0

# Work out angles for grating and sensor plane

m = 1.0 # diffracted order
g = 1e6*mks_length # groove density
incidence_angle = np.pi/4
central_diff_angle = np.arcsin(np.sin(incidence_angle)-2*np.pi*m*g/w00) # grating equation
total_angle = incidence_angle + central_diff_angle
central_direction = np.array([-np.sin(total_angle),0.0,-np.cos(total_angle)])
mess += 'diffraction angle = ' + str(central_diff_angle)

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
					'focus' : (0.0,0.0,0.0,-.05/mks_length),
					'supergaussian exponent' : 2})

	ray.append({	'number' : (64,64,16,1),
					'bundle radius' : (rb,rb,rb,rb),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)})

	optics.append([
		{	'object' : surface.EikonalProfiler('start'),
			'size' : (1/inch,1/inch),
			'origin' : (0.,0.,-0.049/mks_length)},

		{	'object' : surface.Grating('G1'),
			'size' : (2/inch,1/inch),
			'diffracted order' : m,
			'groove density' : g,
			'origin' : (0.,0.,0.),
			'euler angles' : helper.rot_zx(-incidence_angle)},

		{	'object' : surface.EikonalProfiler('det'),
			'size' : (3/inch,3/inch),
			'origin' : tuple(central_direction*.05/mks_length),
			'euler angles' : helper.rot_zx(-total_angle)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (16,4,4,1),
						'base filename' : 'out/test'})
