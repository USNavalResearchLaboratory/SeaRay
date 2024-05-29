from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.ionization as ionization
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools
import modules.rotations as rotations

# Example input file for 2 color THz generation via UPPE module.

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

prop_range = (-15/cm,15/cm)
w00 = 1.0
r00 = dnum('3 mm')
U00 = dnum('0.1 mJ')
t00 = dnum('55 fs')
a00 = helper.a0(U00,t00,r00,w00)
chi3 = helper.chi3(1.0,'7.4e-24 m2/W')
# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,5.5)
Uion = dnum('12.1 eV')
ngas = dnum('5.4e18 cm-3')
Zeff = 0.53
ionizer = ionization.PPT_Tunneling(mks_length,False,Uion,Zeff)
air = dispersion.HumidAir(mks_length,0.4,1e-4)
air.add_opacity_region(1/cm,0.01/um,0.3/um)

jstates = np.arange(0,40)
damping = np.ones(40)/dnum('100 ps')
dnuc = 6 - 3*(jstates%2)
hbar_norm = C.hbar * (w00*C.c/mks_length) / (C.m_e*C.c**2)
N2rot = rotations.Rotator(2/dnum('1 cm'),4*np.pi*6.7e-25/dnum('1 cm-3'),dnum('.025 eV'),dnuc,damping,hbar_norm)
dnuc = jstates%2
O2rot = rotations.Rotator(1.44/dnum('1 cm'),4*np.pi*10.2e-25/dnum('1 cm-3'),dnum('.025 eV'),dnuc,damping,hbar_norm)

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
            'origin': (None,0.0,0.0,-110/cm),
            'euler angles': (0.0,0.0,0.0),
            'number': (2049,64,2,None),
            'bundle radius': (None,) + (.001*r00,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': band + (0.0,3*r00) + (0.0,2*np.pi) + (None,None),
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
optics[-1]['object'] = surface.IdealCompressor('compressor')
optics[-1]['group delay dispersion'] = 1000/fs**2
optics[-1]['center frequency'] = 1.0
optics[-1]['frequency band'] = (0.9,1.1)
optics[-1]['size'] = (1/inch,1/inch)
optics[-1]['origin'] = (None,0,0,-95/cm)
optics[-1]['euler angles'] = (0.,0.,0.)

# optics.append({})
# optics[-1]['object'] = surface.IdealLens('lens')
# optics[-1]['radius'] = 0.5/inch
# optics[-1]['focal length'] = 90/cm
# optics[-1]['origin'] = (None,0,0,-90/cm)
# optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = surface.FlyingFocus('fflens')
optics[-1]['radius'] = 0.5/inch
optics[-1]['max radius'] = 2*r00
optics[-1]['start pos'] = 90/cm
optics[-1]['end pos'] = 96/cm
optics[-1]['front velocity'] = 1.001
optics[-1]['origin'] = (None,0,0,-90/cm)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = surface.IdealHarmonicGenerator('SHG')
optics[-1]['harmonic delay'] = 20/fs
optics[-1]['harmonic number'] = 2.0
optics[-1]['frequency band'] = (0.9,1.1)
optics[-1]['efficiency'] = 0.1
optics[-1]['radius'] = 0.5/inch
optics[-1]['origin'] = (None,0,0,-80/cm)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('start')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (10*r00,10*r00)
optics[-1]['origin'] = (None,0,0,prop_range[0]-.05/mm)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('air')
optics[-1]['propagator'] = 'uppe'
optics[-1]['ionizer'] = None
optics[-1]['rotator'] = None
optics[-1]['wave coordinates'] = 'cylindrical'
optics[-1]['wave grid'] = (2049,512,1,11)
optics[-1]['radial modes'] = 512
optics[-1]['density reference'] = ngas
optics[-1]['density function'] = '1.0'
optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(r2.shape)
optics[-1]['frequency band'] = band
optics[-1]['nonlinear band'] = (0.0,4.0)
optics[-1]['subcycles'] = 1
optics[-1]['minimum step'] = 1.0
optics[-1]['dispersion inside'] = air
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['chi3'] = 0
optics[-1]['size'] = (6/mm,6/mm,prop_range[1]-prop_range[0])
optics[-1]['origin'] = (None,0,0,(prop_range[0]+prop_range[1])/2)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['window speed'] = air.GroupVelocityMagnitude(1.0)

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
