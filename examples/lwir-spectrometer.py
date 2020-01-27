from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume
import input_tools

material_selection = ['NaCl','ZnSe'][1]

mks_length = 10e-6 / (2*np.pi)
inch = mks_length*100/2.54
cm = mks_length*100
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
r00 = .001/mks_length # spot size of radiation
rb = r00*bundle_scale
t00 = 20e-15*C.c/mks_length
band = (0.5,5.0)

# Setup prism

if material_selection=='ZnSe':
	material = dispersion.ZnSe(mks_length)
	prism_box = (2/inch,1/inch,1.5/inch)
	refraction_angle = np.pi/8
else:
	material = dispersion.NaCl(mks_length)
	prism_box = (2/inch,1/inch,1.5/inch)
	refraction_angle = np.pi/6

nrefr = np.sqrt(1+material.chi(w00)[0])
incidence_angle = np.arcsin(nrefr*np.sin(refraction_angle))
mess += 'TIR angle = ' + str(np.arcsin(1/nrefr)*180/np.pi) + ' deg\n'

# General layout

f = 0.1/mks_length
Mdeg1 = 25.0
Mdeg2 = 10.0
RM = 0.5/inch
focus = (.1*f,0.0,0.0,0.0)

# Set up dictionaries

for i in range(1):

	sim.append({'mks_length' : mks_length ,
				'mks_time' : mks_length/C.c ,
				'message' : mess})

	wave.append([{	# EM 4-potential (eA/mc^2) , component 0 not used
					'a0' : (0.0,0.0,0.0,a00) ,
					# 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
					'r0' : (t00,r00,r00,t00) ,
					# 4-wavenumber: omega,kx,ky,kz
					'k0' : (w00,w00,0.0,0.0) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : focus,
					'supergaussian exponent' : 2},
				{	# EM 4-potential (eA/mc^2) , component 0 not used
					'a0' : (0.0,0.0,0.0,a00) ,
					# 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
					'r0' : (t00/5,r00,r00,t00/5) ,
					# 4-wavenumber: omega,kx,ky,kz
					'k0' : (5*w00,5*w00,0.0,0.0) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : focus,
					'supergaussian exponent' : 2}])

	ray.append({	'number' : (64,32,8,1),
					'bundle radius' : (rb,rb,rb,rb),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)})

	optics.append([
		{	'object' : surface.disc('M1'),
			'reflective' : True,
			'radius' : RM,
			'origin' : helper.set_pos([2/cm,0.0,0.0]),
			'euler angles' : helper.rot_zx_deg(90-Mdeg1)},

		{	'object' : surface.SphericalCap('M2'),
			'reflective' : True,
			'radius of sphere' : 2*f,
			'radius of edge' : RM,
			'origin' : helper.polar_move_zx(f-2/cm,180-2*Mdeg1),
			'euler angles' : helper.rot_zx_deg(-90-2*Mdeg1+Mdeg2)},

		{	'object' : surface.disc('M3'),
			'reflective' : True,
			'radius' : RM,
			'origin' : helper.polar_move_zx(7/cm,-2*(Mdeg1-Mdeg2)),
			'euler angles' : helper.rot_zx_deg(0.5*(90-2*(Mdeg1-Mdeg2)))},

		{	'object' : volume.PellinBroca('P1'),
			'dispersion outside' : dispersion.Vacuum(),
			'dispersion inside' : material,
			'size' : prism_box,
			'angle' : refraction_angle,
			'origin' : helper.move(-0.7*np.sin(incidence_angle)*prism_box[2],0.0,6/cm),
			'euler angles' : helper.rot_zx(incidence_angle)},

		{	'object' : surface.SphericalCap('M4'),
			'reflective' : True,
			'radius of sphere' : 2*f,
			'radius of edge' : RM,
			'origin' : helper.move(7/cm,0.0,prism_box[1]),
			'euler angles' : helper.rot_zx_deg(90-Mdeg2)},

		{	'object' : surface.EikonalProfiler('det'),
			'size' : (2/inch,2/inch),
			'origin' : helper.polar_move_zx(f,180-2*Mdeg2),
			'euler angles' : helper.rot_zx_deg(90-2*Mdeg2)}
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (32,2,8,1),
						'base filename' : 'out/test'})
