from scipy import constants as Cmks
import numpy as np
import modules.dispersion as dispersion
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools 

material_selection = ['KCl','NaCl','ZnSe'][0]

# Units and scales

mks_length = 10.6e-6 / (2*np.pi)
bundle_scale = 1e-4
cm = 100*mks_length
inch = 100*mks_length/2.54
fs = 1e15*mks_length/Cmks.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Setup pulse parameters

theta = -90/deg # initial propagation direction
a00 = 1.0
w00 = 1.0
r00 = 0.03/cm # spot size of radiation
rb = r00*bundle_scale
t00 = 20/fs
band = (0.5,2.0)

# Setup prism

if material_selection=='ZnSe':
	material = dispersion.ZnSe(mks_length)
	prism_box = (2/inch,1/inch,1.5/inch)
	refraction_angle = 22.5/deg
elif material_selection=='NaCl':
	material = dispersion.NaCl(mks_length)
	prism_box = (2/inch,1/inch,1.5/inch)
	refraction_angle = 30/deg
else:
	material = dispersion.KCl(mks_length)
	prism_box = (2/inch,1/inch,1.5/inch)
	refraction_angle = 30/deg

nrefr = np.sqrt(1+material.chi(w00)[0])
incidence_angle = np.arcsin(nrefr*np.sin(refraction_angle))
# Figure displacement to put input beam at edge of side A
temp_prism = volume.PellinBroca('temp')
A,B,C,D = temp_prism.SideLengths(prism_box[0],prism_box[2],refraction_angle)
DA,AB,BC,CD = temp_prism.MidplaneVertices(A,B,C,D,refraction_angle)
prism_disp_x = -(DA[0]*np.cos(incidence_angle)-DA[2]*np.sin(incidence_angle))
prism_disp_x -= 0.45/inch

mess += 'TIR angle = ' + str(np.arcsin(1/nrefr)*deg) + ' deg\n'

# General layout

f = 10/cm
Mdeg1 = 25.0
Mdeg2 = 10.0
RM = 0.5/inch
focus = (.1*f,0.0,0.0,0.0)

# Set up dictionaries

sim = {}
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/Cmks.c
sim['message'] = mess

sources = [
    {
        'rays': {
            'origin': (None,-f,0,0),
            'euler angles': helper.rot_zx(theta),
            'number': (64,32,8,None),
            'bundle radius': (None,) + (rb,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': band + (0,3*r00) + (0,2*np.pi) + (None,None)
        },
        'waves': [
            {
                'a0': (None,a00,0,None),
                'r0': (t00,r00,r00,t00),
                'k0': (w00,None,None,w00),
                'mode': (None,0,0,None),
                'basis': 'hermite'
            }
        ]
    }
]

optics.append({})
optics[-1]['object'] = surface.IdealLens('L1')
optics[-1]['focal length'] = 10/cm
optics[-1]['radius'] = RM
optics[-1]['origin'] = helper.set_pos([None,-0.99*f,0.0,0.0])
optics[-1]['euler angles'] = helper.rot_zx_deg(-90)

optics.append({})
optics[-1]['object'] = surface.disc('M1')
optics[-1]['reflective'] = True
optics[-1]['radius'] = RM
optics[-1]['origin'] = helper.set_pos([None,2/cm,0.0,0.0])
optics[-1]['euler angles'] = helper.rot_zx_deg(90-Mdeg1)

optics.append({})
optics[-1]['object'] = surface.SphericalCap('M2')
optics[-1]['reflective'] = True
optics[-1]['radius of sphere'] = 2*f
optics[-1]['radius of edge'] = RM
optics[-1]['origin'] = helper.polar_move_zx(f-2/cm,180-2*Mdeg1)
optics[-1]['euler angles'] = helper.rot_zx_deg(-90-2*Mdeg1+Mdeg2)

optics.append({})
optics[-1]['object'] = surface.disc('M3')
optics[-1]['reflective'] = True
optics[-1]['radius'] = RM
optics[-1]['origin'] = helper.polar_move_zx(7/cm,-2*(Mdeg1-Mdeg2))
optics[-1]['euler angles'] = helper.rot_zx_deg(0.5*(90-2*(Mdeg1-Mdeg2)))

optics.append({})
optics[-1]['object'] = volume.PellinBroca('P1')
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['dispersion inside'] = material
optics[-1]['size'] = prism_box
optics[-1]['angle'] = refraction_angle
optics[-1]['origin'] = helper.move(prism_disp_x,0.0,6/cm)
optics[-1]['euler angles'] = helper.rot_zx(incidence_angle)

optics.append({})
optics[-1]['object'] = surface.SphericalCap('M4')
optics[-1]['reflective'] = True
optics[-1]['radius of sphere'] = 2*f
optics[-1]['radius of edge'] = RM
optics[-1]['origin'] = helper.move(7/cm,0.0,0.86*prism_box[1])
optics[-1]['euler angles'] = helper.rot_zx_deg(90-Mdeg2)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('det')
optics[-1]['size'] = (2/inch,2/inch)
optics[-1]['origin'] = helper.polar_move_zx(f,180-2*Mdeg2)
optics[-1]['euler angles'] = helper.rot_zx_deg(90-2*Mdeg2)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (8,2,8,None)
diagnostics['base filename'] = 'out/test'
