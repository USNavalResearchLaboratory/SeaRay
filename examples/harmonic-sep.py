from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Example of separating Nd:glass wavelength and its second harmonic in a Pellin-Broca prism.

# Suggested plotter command
# python ray_plotter.py out/test o3d

mks_length = 1.054e-6 / (2*np.pi)
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

a00 = 1.0
w00 = 1.0
r00 = 1e-4/mks_length # spot size of radiation
rb = r00*bundle_scale
t00 = 500e-15*C.c/mks_length
band = (0.9,2.1)

# Setup prism

material = dispersion.BK7(mks_length)
nrefr = np.sqrt(1+material.chi(w00)[0])
prism_box = (1/inch,1/inch,0.75/inch)
refraction_angle = np.pi/6
incidence_angle = np.arcsin(nrefr*np.sin(refraction_angle))
mess += 'TIR angle = ' + str(np.arcsin(1/nrefr)*180/np.pi) + ' deg\n'

# Set up dictionaries

for i in range(1):

	sim.append({'mks_length' : mks_length ,
				'mks_time' : mks_length/C.c ,
				'message' : mess})

	wave.append([{	# EM 4-potential (eA/mc^2) , component 0 not used
					'a0' : (0.0,a00,0.0,0.0) ,
					# 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
					'r0' : (t00,r00,r00,t00) ,
					# 4-wavenumber: omega,kx,ky,kz
					'k0' : (w00,0.0,0.0,w00) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,0.0,-.031/mks_length),
					'supergaussian exponent' : 2},
				{	# EM 4-potential (eA/mc^2) , component 0 not used
					'a0' : (0.0,a00,0.0,0.0) ,
					# 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
					'r0' : (t00,r00,r00,t00) ,
					# 4-wavenumber: omega,kx,ky,kz
					'k0' : (2*w00,0.0,0.0,2*w00) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,0.0,0.0,-.031/mks_length),
					'supergaussian exponent' : 2}])

	ray.append({	'number' : (64,1,4,1),
					'bundle radius' : (rb,rb,rb,rb),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)})

	optics.append([
		{	'object' : volume.PellinBroca('P1'),
			'dispersion outside' : dispersion.Vacuum(),
			'dispersion inside' : material,
			'size' : prism_box,
			'angle' : refraction_angle,
			'origin' : (-0.01/mks_length,0.,0.),
			'euler angles' : helper.rot_zx(incidence_angle)},

		{	'object' : surface.EikonalProfiler('det'),
			'size' : (1/inch,1/inch),
			'origin' : (.1/mks_length,0.0,.01/mks_length),
			'euler angles' : helper.rot_zx(-np.pi/2)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (16,1,4,1),
						'base filename' : 'out/test'})
