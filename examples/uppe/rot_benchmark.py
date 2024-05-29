from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.ionization as ionization
import modules.rotations as rotations
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools 

# Compare with Wahlstrand et al., PRA 85, 043820 (2012), Fig. 2(b).
# Output will be chi_eff.  To compare with figure work out the phase shift:
# k0*dn*Leff (Eq. 2), where dn ~ chi_eff/2, and Leff = 4.41 mm.  Probe wavelength is 580 nm.
# Overall this gives 23900*chi_eff.  N.b. the Kerr part will be underestimated by two-fold,
# due to XPM, so we double the Kerr term below to account for that.
# Propagation range in simulation only needs to match if we want to check actual probe phase shift.

# This input file can either be `uppe` or `paraxial`, controlled by parameter below
propagator = 'paraxial'

mks_length = 0.8e-6 / (2*np.pi)
cm = 100*mks_length
mm = 1000*mks_length
um = 1e6*mks_length
helper = input_tools.InputHelper(mks_length)
dnum = helper.dnum

# Control Parameters - Laser

wavelength = 0.8/um
r00 = 150/um
w00 = 2*np.pi/wavelength
t00 = dnum('0.04 ps')/np.sqrt(2*np.log(2))
a00 = C.e*np.sqrt(2*377*4.5e13*1e4)/(w00*C.c/mks_length)/C.m_e/C.c
propagation_range = (0/cm,.441/cm)
rbox = 1/mm
if propagator=='uppe':
    band = (0.0,8.0)
    freq_nodes = 2049
else:
    band = (0.5,1.5)
    freq_nodes = 256

# Control Parameters - Gas

ngas_ref = dnum('2.7e19 cm-3') # density at which parameters are given
ngas = ngas_ref*90/760 # actual density of the volume
chi3 = helper.chi3(1.0,'14.8e-24 m2/W') # doubled to account for XPM
Uion = dnum('12.1 eV')
Zeff = 0.53
ionizer = ionization.StitchedPPT(mks_length,True,Uion,Zeff,lstar=0,l=0,m=0,w0=w00,terms=80)
jstates = np.arange(0,32)
damping = np.ones(32)/dnum('100 ps')
dnuc = 6 - 3*(jstates%2)
hbar_norm = C.hbar * (w00*C.c/mks_length) / (C.m_e*C.c**2)
N2rot = rotations.Rotator(2/dnum('1 cm'),4*np.pi*6.7e-25/dnum('1 cm-3'),dnum('.025 eV'),dnuc,damping,hbar_norm)
dnuc = jstates%2
O2rot = rotations.Rotator(1.44/dnum('1 cm'),4*np.pi*10.2e-25/dnum('1 cm-3'),dnum('.025 eV'),dnuc,damping,hbar_norm)
air = dispersion.HumidAir(mks_length,0.4,1e-3)

# Derived Parameters

L = propagation_range[1] - propagation_range[0]
rgn_center = (None,0.0,0.0,0.5*(propagation_range[0]+propagation_range[1]))

# Helpers

def rect(w,w1,w2):
    return np.heaviside(w-w1,0.5) * np.heaviside(w2-w,0.5)

# Set up dictionaries

sim = {}
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = 'Processing input file...'

sources = [
    {
        'rays': {
            'origin': (None,0.0,0.0,-0.1/mm),
            'euler angles': (0.0,0.0,0.0),
            'number': (freq_nodes,128,2,None),
            'bundle radius': (None,) + (.001*r00,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': band + (0.0,3*r00) + (0.0,2*np.pi) + (None,None),
        },
        'waves': [
            # pump pulse
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
optics[-1]['object'] = volume.AnalyticBox('air')
optics[-1]['propagator'] = propagator
optics[-1]['ionizer'] = None
optics[-1]['rotator'] = N2rot
optics[-1]['wave coordinates'] = 'cylindrical'
optics[-1]['wave grid'] = (freq_nodes,256,1,5)
optics[-1]['radial modes'] = 256
optics[-1]['density reference'] = ngas_ref
optics[-1]['density function'] = str(ngas/ngas_ref)
optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(r2.shape) * ngas/ngas_ref
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

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('stop')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (10*rbox,10*rbox)
optics[-1]['origin'] = (None,0,0,L*2)
optics[-1]['euler angles'] = (0.,0.,0.)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (32,8,2,None)
diagnostics['base filename'] = 'out/test'
