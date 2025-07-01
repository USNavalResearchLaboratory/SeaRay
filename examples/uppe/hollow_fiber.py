from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.ionization as ionization
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools

# Example input file for argon filled fiber.

mks_length = 0.8e-6 / (2*np.pi)
cm = 100*mks_length
mm = 1000*mks_length
um = 1e6*mks_length
fs = 1e15*mks_length/C.c
inch = cm/2.54
helper = input_tools.InputHelper(mks_length)
dnum = helper.dnum
mess = 'Processing input file...\n'

# Control parameters

prop_range = (0/cm,10/cm)
w00 = 1.0
r00 = dnum('0.1 mm')
U00 = dnum('0.1 mJ')
t00 = dnum('55 fs')
a00 = helper.a0(U00,t00,r00,w00)
chi3 = helper.chi3(1.0,'1e-24 m2/W')
# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,5.5)
Uion = dnum('15.1 eV')
ngas = dnum('1e19 cm-3')
Zeff = 1
ionizer = ionization.PPT_Tunneling(mks_length,False,Uion,Zeff)
argon = dispersion.Ar(mks_length)
#argon.add_opacity_region(1/cm,0.01/um,0.3/um)

# Set up dictionaries

sim = {}
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = mess

sources = [
    {
        'rays': {
            'origin': (None,0.0,0.0,-2/cm),
            'euler angles': (0.0,0.0,0.0),
            'number': (2049,128,2,None),
            'bundle radius': (None,) + (.001*r00,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': band + (0.0,2*r00) + (0.0,2*np.pi) + (None,None),
        },
        'waves': [
            # fundamental
            {
                'a0': (None,a00,0,None),
                'r0': (t00,r00,r00,t00),
                'k0': (w00,None,None,w00),
                'mode': (None,0,0,None),
                'basis': 'hermite'
            },
        ]
    }
]

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('start')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (10*r00,10*r00)
optics[-1]['origin'] = (None,0,0,prop_range[0]-1/mm)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('air')
optics[-1]['propagator'] = 'uppe'
optics[-1]['ionizer'] = ionizer
optics[-1]['rotator'] = None
optics[-1]['wave coordinates'] = 'cylindrical'
optics[-1]['wave grid'] = (2049,1024,1,11)
optics[-1]['radial modes'] = 1024
optics[-1]['density reference'] = ngas
optics[-1]['density function'] = '1.0'
optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(r2.shape)
optics[-1]['frequency band'] = band
optics[-1]['nonlinear band'] = (0.0,4.0)
optics[-1]['subcycles'] = 1
optics[-1]['minimum step'] = 1.0
optics[-1]['dispersion inside'] = argon
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['chi3'] = 0
optics[-1]['size'] = (0.2/mm,0.2/mm,prop_range[1]-prop_range[0])
optics[-1]['origin'] = (None,0,0,(prop_range[0]+prop_range[1])/2)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['window speed'] = argon.GroupVelocityMagnitude(1.0)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('stop')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (15/mm,15/mm)
optics[-1]['origin'] = (None,0,0,prop_range[1]+1/mm)
optics[-1]['euler angles'] = (0.,0.,0.)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (128,8,2,None)
diagnostics['base filename'] = 'out/test'
