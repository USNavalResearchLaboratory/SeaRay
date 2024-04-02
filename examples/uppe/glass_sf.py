from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools

# Example input file for axisymmetric UPPE wave equation.
# Illustrates self focusing, self phase modulation, and group velocity dispersion in glass
# Effects are most easily observed using the interactive viewer

mks_length = 0.8e-6 / (2*np.pi)
um = 1e6*mks_length
mess = 'Processing input file...\n'
helper = input_tools.InputHelper(mks_length)
dnum = helper.dnum

# Preprocessing calculations

glass = dispersion.BK7(mks_length)
# Suppress out of band frequencies
glass.add_opacity_region(500/um,0.1/um,0.6/um)
glass.add_opacity_region(500/um,1.2/um,4/um)
w00 = 1.0
r00 = 100/um
t00 = dnum('15 fs')
U00 = dnum('0.0025 mJ')
a00 = helper.a0(U00,t00,r00,w00)
chi3 = helper.chi3(1.5,'2e-20 m2/W')

# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,4.0)
t00,pulse_band = helper.TransformLimitedBandwidth(w00,t00,1.0)

# Work out the dispersion length
vg1 = glass.GroupVelocityMagnitude(pulse_band[0])
vg2 = glass.GroupVelocityMagnitude(pulse_band[1])
Ldisp = t00 / np.abs(1/vg1 - 1/vg2)
Lprop = 4*Ldisp
mess = mess + '  Red speed = ' + str(vg1) + '\n'
mess = mess + '  Blue speed = ' + str(vg2) + '\n'
mess = mess + '  Dispersion length = ' + str(1e3*Ldisp*mks_length) + ' mm\n'
mess = mess + '  Propagation length = ' + str(1e3*Lprop*mks_length) + ' mm\n'

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
            'origin': (None,0.0,0.0,-1),
            'euler angles': (0.0,0.0,0.0),
            'number': (1025,64,4,None),
            'bundle radius': (None,) + (.001*r00,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': band + (0.0,4*r00) + (0.0,2*np.pi) + (None,None),
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
optics[-1]['object'] = surface.EikonalProfiler('start')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (6*r00,6*r00)
optics[-1]['origin'] = (None,0,0,-0.5)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('glass')
optics[-1]['propagator'] = 'uppe'
optics[-1]['wave coordinates'] = 'cylindrical'
optics[-1]['wave grid'] = (1025,64,1,7)
optics[-1]['radial modes'] = 64
optics[-1]['density reference'] = 1.0
optics[-1]['density function'] = '1.0'
optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(r2.shape)
optics[-1]['frequency band'] = band
optics[-1]['nonlinear band'] = (0.1,4.0)
optics[-1]['subcycles'] = 1
optics[-1]['minimum step'] = .3/um
optics[-1]['dispersion inside'] = glass
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['chi3'] = chi3
optics[-1]['size'] = (6*r00,6*r00,Lprop)
optics[-1]['origin'] = (None,0,0,Lprop/2)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['window speed'] = glass.GroupVelocityMagnitude(1.0)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('stop')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (8*r00,8*r00)
optics[-1]['origin'] = (None,0,0,2*Lprop)
optics[-1]['euler angles'] = (0.,0.,0.)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (8,4,4,None)
diagnostics['base filename'] = 'out/test'
