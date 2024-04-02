from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.ionization as ionization
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools

# Simple USPL air plasma using UPPE module.
# Generally requires thousands of steps, a powerful GPU helps.

mks_length = 0.8e-6 / (2*np.pi)
cm = 100*mks_length
mm = 1000*mks_length
um = 1e6*mks_length
helper = input_tools.InputHelper(mks_length)
dnum = helper.dnum

# Control Parameters

wavelength = 0.8/um
waist = 150/um
w00 = 2*np.pi/wavelength
U00 = dnum('1 mJ')
t00 = dnum('50 fs')
propagation_range = (-20/cm,20/cm)
rbox = 3/mm
chi3 = helper.chi3(1.0,'1e-23 m2/W')
# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,8.0)
Uion = dnum('12.1 eV')
ngas = dnum('5.4e18 cm-3')
Zeff = 0.53
ionizer = ionization.StitchedPPT(mks_length,False,Uion,Zeff,lstar=0,l=0,m=0,w0=w00,terms=80)
air = dispersion.SimpleAir(mks_length)
air.add_opacity_region(0.1/cm,0.01/um,0.3/um)

# Derived Parameters

L = propagation_range[1] - propagation_range[0]
time_to_focus = abs(2*propagation_range[0])
diffraction_angle = wavelength / (np.pi*waist)
r00 = time_to_focus * diffraction_angle
a00 = helper.a0(U00,t00,r00,w00)
rgn_center = (None,0.0,0.0,0.5*(propagation_range[0]+propagation_range[1]))

# Set up dictionaries

sim = {}
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = 'Processing input file...'

helper.set_pos((None,0,0,-time_to_focus))

sources = [
    {
        'rays': {
            'origin': helper.curr_pos,
            'euler angles': (0.0,0.0,0.0),
            'number': (2049,128,2,None),
            'bundle radius': (None,) + (.001*r00,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': band + (0.0,3*r00) + (0.0,2*np.pi) + (None,None),
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
optics[-1]['origin'] = helper.move(0,0,0.1/mm)
optics[-1]['euler angles'] = (0,0,0)
optics[-1]['radius'] = 1/cm
optics[-1]['focal length'] = time_to_focus

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('air')
optics[-1]['propagator'] = 'uppe'
optics[-1]['ionizer'] = ionizer
optics[-1]['wave coordinates'] = 'cylindrical'
optics[-1]['wave grid'] = (2049,256,1,20)
optics[-1]['radial modes'] = 128
optics[-1]['density reference'] = ngas
optics[-1]['density function'] = '1.0'
optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(r2.shape)
optics[-1]['frequency band'] = band
optics[-1]['nonlinear band'] = (0.0,4.0)
optics[-1]['subcycles'] = 1
optics[-1]['minimum step'] = 1.0
optics[-1]['dispersion inside'] = air
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['chi3'] = chi3
optics[-1]['size'] = (2*rbox,2*rbox,L)
optics[-1]['origin'] = rgn_center
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['window speed'] = air.GroupVelocityMagnitude(1.0)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (4,8,2,None)
diagnostics['base filename'] = 'out/test'
