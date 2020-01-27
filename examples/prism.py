from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Example of dispersion through prism

# Suggested plotter command
# python ray_plotter.py out/test o3d det=4,5

mks_length = 0.2e-6 / (2*np.pi)
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
r00 = 2e-3/mks_length # spot size of radiation
rb = r00*bundle_scale
t00,band = helper.TransformLimitedBandwidth(w00,'1 fs',1)
a00 = 1.0

# Setup prism and predict central trajectory angle

material = dispersion.BK7(mks_length)
nrefr = np.sqrt(1+material.chi(w00)[0])
prism_box = (2/inch,1/inch,1/inch)
q0 = np.arctan(0.5*prism_box[2]/prism_box[0]) # half angle of the prism
q1i = -q0 # angle of incidence first surface
q1r = np.arcsin(np.sin(q1i)/nrefr) # angle of refraction first surface
q2i = q1r + 2*q0
q2r = np.arcsin(np.sin(q2i)*nrefr)
central_angle = -(q2r - q0)
central_direction = np.array([np.sin(central_angle),0.0,np.cos(central_angle)])

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
		{	'object' : volume.Prism('P1'),
			'dispersion outside' : dispersion.Vacuum(),
			'dispersion inside' : material,
			'size' : prism_box,
			'origin' : (0.,0.,0.),
			'euler angles' : (0.0,0.0,0.0)},

		{	'object' : surface.EikonalProfiler('det'),
			'size' : (2/inch,1/inch),
			'origin' : tuple(central_direction*.05/mks_length),
			'euler angles' : helper.rot_zx(-central_angle)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (16,2,2,1),
						'base filename' : 'out/test'})
