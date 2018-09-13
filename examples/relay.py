from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume

# INPUT FILE FOR FOR CO2 LASER.
# 532 nm beamline in front end
# Starting from SHG crystal
# x is up, z is the long way on the optical table
# TABLE 1: nominally 8x4 ft
# Regen breadboard: 7x1 ft
# 45 bolt holes across, 1.75" from last bolt hole to table edge
# 93 bolt holes along, 1.75" from last bolt hole to table edge
# Origin is corner bolt hole on table 1.

mks_length = 0.532e-6 / (2*np.pi)
inch = 0.0254/mks_length
feet = 12*inch
cm = 0.01/mks_length
sim = []
wave = []
ray = []
optics = []
diagnostics = []
mess = 'Processing input file...\n'

# Preprocessing calculations

table1_dims = (1*feet,4*feet,8*feet)
table2_dims = (1*feet,3*feet,8*feet)
regen_dims = (2*inch,1*feet,7*feet)
beam_height_regen = 2*inch
table1_center = (-beam_height_regen-regen_dims[0]-table1_dims[0]/2,-1.75*inch+2*feet+0.25*inch,-1.75*inch+4*feet+0.25*inch)
table2_center = (table1_center[0],table1_center[1]-0.5*feet,table1_center[2]+8*feet+66*cm)
regen_center = (-beam_height_regen-regen_dims[0]/2,4.75*inch+0.5*feet,-1.75*inch+3.5*feet)
SHG = (35*cm,10*inch,42*inch-1.75*inch-25*cm)

r0_mks = 0.005 # spot size at SHG crystal
t0_mks = 10e-9 # pulse width
U0_mks = 0.5
I0_mks = U0_mks / t0_mks / r0_mks**2 / (0.5*np.pi)**1.5
w0_mks = C.c/mks_length
eta0 = 1/C.epsilon_0/C.c
E0_mks = np.sqrt(2*eta0*I0_mks)

t00 = t0_mks*C.c/mks_length
r00 = r0_mks/mks_length
a00 = C.e*E0_mks/(C.m_e * w0_mks * C.c)

nBK7 = np.sqrt(1+dispersion.BK7(mks_length).chi(1.0)[0])
Rf100cm = (nBK7-1)/mks_length # radius of BK7 that gives f = 1 m

mess += 'Laser intensity = {:.2g}'.format(I0_mks*1e-4) + ' (W/cm^2)\n'
mess += 'Normalized amplitude = {:.2g}'.format(a00) + '\n'

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
					'k0' : (1.0,0.0,0.0,1.0) ,
					# 0-component of focus is time at which pulse reaches focal point.
					# If time=0 use paraxial wave, otherwise use spherical wave.
					# Thus in the paraxial case the pulse always starts at the waist.
					'focus' : (0.0,SHG[0],SHG[1],SHG[2]),
					'pulse shape' : 'sech',
					'supergaussian exponent' : 8})

	ray.append({	'number' : (16,16,1),
					'bundle radius' : (.001*r00,.001*r00,.001*r00,.001*r00),
					'loading coordinates' : 'cylindrical',
					# Ray box is always put at the origin
					# It will be transformed appropriately by SeaRay to start in the wave
					'box' : (0,1.5*r00,0.0,2*np.pi,-2*t00,2*t00)})

	optics.append([
		{	'object' : surface.EikonalProfiler('init'),
			'size' : (inch,inch),
			'grid points' : (128,128,1),
			'euler angles' : (0,0,0),
			'origin' : (SHG[0],SHG[1],SHG[2]+1*cm)},

		{	'object' : surface.disc('M1'),
			'radius' : 0.5*inch,
			'origin' : (SHG[0],SHG[1],SHG[2]+10*cm),
			'euler angles' : (0,-np.pi/4,0),
			'reflective' : True },

		{	'object' : surface.disc('M2'),
			'radius' : 0.5*inch,
			'origin' : (SHG[0],SHG[1]-10.5*cm,SHG[2]+10*cm),
			'euler angles' : (0,-np.pi/4,0),
			'reflective' : True },

		{	'object' : surface.disc('M3'),
			'radius' : 0.5*inch,
			'origin' : (SHG[0],SHG[1]-10.5*cm,SHG[2]+25*cm),
			'euler angles' : (0,np.pi/4,0),
			'reflective' : True },

		{	'object' : surface.disc('M4'),
			'radius' : 0.5*inch,
			'origin' : (SHG[0],SHG[1],SHG[2]+25*cm),
			'euler angles' : (-np.pi/4,np.pi/2,0),
			'reflective' : True },

		{	'object' : surface.disc('M5'),
			'radius' : 0.5*inch,
			'origin' : (0,SHG[1],SHG[2]+25*cm),
			'euler angles' : (-np.pi/2,np.pi/4,0),
			'reflective' : True },

		{	'object' : surface.disc('BS1'),
			'radius' : 0.5*inch,
			'origin' : (0,SHG[1],SHG[2]+17.5*cm),
			'euler angles' : (0,-np.pi/4,0),
			'reflective' : True },

		{	'object' : surface.disc('M6'),
			'radius' : 0.5*inch,
			'origin' : (0,SHG[1]+25*cm,SHG[2]+17.5*cm),
			'euler angles' : (0,np.pi/4,0),
			'reflective' : True },

		{	'object' : surface.disc('BS2'),
			'radius' : 0.5*inch,
			'origin' : (0,SHG[1]+25*cm,SHG[2]+25*cm),
			'euler angles' : (0,np.pi/4,0),
			'reflective' : False },

		{	'object' : volume.SphericalLens('L1'),
			'dispersion inside' : dispersion.BK7(mks_length),
			'dispersion outside' : dispersion.Vacuum(),
			'thickness' : 0.3*cm,
			'rcurv beneath' : Rf100cm*1.69,
			'rcurv above' : 1e10,
			'aperture radius' : 0.5*inch,
			'origin' : (0,SHG[1]+25*cm,SHG[2]+69*cm),
			'euler angles' : (0,0,0)},

		{	'object' : surface.EikonalProfiler('W1'),
			'size' : (inch,inch),
			'grid points' : (128,128,1),
			'euler angles' : (0,0,0),
			'origin' : (0,SHG[1]+25*cm,SHG[2]+70*cm)},

		{	'object' : surface.EikonalProfiler('waist'),
			'size' : (inch,inch),
			'grid points' : (128,128,1),
			'euler angles' : (0,0,0),
			'origin' : (0,SHG[1]+25*cm,SHG[2]+69*cm+169*cm)},

		{	'object' : surface.EikonalProfiler('W2'),
			'size' : (inch,inch),
			'grid points' : (128,128,1),
			'euler angles' : (0,0,0),
			'origin' : (0,SHG[1]+25*cm,SHG[2]+271*cm)},

		{	'object' : volume.SphericalLens('L2'),
			'dispersion inside' : dispersion.BK7(mks_length),
			'dispersion outside' : dispersion.Vacuum(),
			'thickness' : 0.3*cm,
			'rcurv beneath' : Rf100cm*1.01,
			'rcurv above' : 1e10,
			'aperture radius' : 0.5*inch,
			'origin' : (0,SHG[1]+25*cm,SHG[2]+272*cm),
			'euler angles' : (0,0,0)},

		{	'object' : surface.disc('M7'),
			'radius' : 0.5*inch,
			'origin' : (0,SHG[1]+25*cm,SHG[2]+283*cm),
			'euler angles' : (0,np.pi/4,0),
			'reflective' : True },

		{	'object' : surface.disc('M8'),
			'radius' : 0.5*inch,
			'origin' : (0,SHG[1]+41.7*cm,SHG[2]+283*cm),
			'euler angles' : (0,np.pi/4,0),
			'reflective' : True },

		{	'object' : surface.EikonalProfiler('sapphire'),
			'size' : (inch,inch),
			'grid points' : (128,128,1),
			'euler angles' : (0,0,0),
			'origin' : (0,SHG[1]+41.7*cm,SHG[2]+356*cm)},

		# The following only for visualizations
		{	'object' : volume.Box('table1'),
			'dispersion inside' : dispersion.BK7(mks_length),
			'dispersion outside' : dispersion.Vacuum(),
			'size' : (1*feet,4*feet,8*feet),
			'euler angles' : (0,0,0),
			'origin' : table1_center },

		{	'object' : volume.Box('table2'),
			'dispersion inside' : dispersion.BK7(mks_length),
			'dispersion outside' : dispersion.Vacuum(),
			'size' : (1*feet,3*feet,8*feet),
			'euler angles' : (0,0,0),
			'origin' : table2_center },

		{	'object' : volume.Box('regen'),
			'dispersion inside' : dispersion.BK7(mks_length),
			'dispersion outside' : dispersion.Vacuum(),
			'size' : (2*inch,1*feet,7*feet),
			'euler angles' : (0,0,0),
			'origin' : regen_center },
		])

	diagnostics.append({'suppress details' : False,
						'clean old files' : True,
						'orbit rays' : (4,8,1),
						'base filename' : 'out/test'})
