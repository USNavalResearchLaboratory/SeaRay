from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.ionization as ionization
import modules.rotations as rotations
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools 

# Compare with Wahlstrand et al., PRA 85, 043820 (2012), Fig. 2(b).
# Output will be chi_eff.  To compare with figure work out the XPM phase shift (SPM*2)
# k0*2*dn*Leff (Eq. 2), where dn ~ chi_eff/2, and Leff = 4.41 mm.  Probe wavelength is 550 nm.
# Overall this gives 50350*chi_eff.
# Propagation range in simulation only needs to match if we want to check actual probe phase shift.

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
wprobe = 2*np.pi/(0.55/um)
t00 = dnum('0.04 ps')/1.18
a00 = C.e*np.sqrt(2*377*4.5e13*1e4)/(w00*C.c/mks_length)/C.m_e/C.c
propagation_range = (0/cm,.441/cm)
rbox = 1/mm
# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,8.0)

# Control Parameters - Gas

ngas_ref = dnum('2.7e19 cm-3') # density at which parameters are given
ngas = ngas_ref*90/760 # actual density of the volume
chi3 = helper.chi3(1.0,'7.4e-24 m2/W')
Uion = dnum('12.1 eV')
Zeff = 0.53
ionizer = ionization.StitchedPPT(mks_length,True,Uion,Zeff,lstar=0,l=0,m=0,w0=w00,terms=80)
jstates = np.arange(0,40)
damping = np.ones(40)/dnum('100 ps')
dnuc = 6 - 3*(jstates%2)
hbar_norm = C.hbar * (w00*C.c/mks_length) / (C.m_e*C.c**2)
N2rot = rotations.Rotator(2/dnum('1 cm'),4*np.pi*6.7e-25/dnum('1 cm-3'),dnum('.025 eV'),dnuc,damping,hbar_norm)
dnuc = jstates%2
O2rot = rotations.Rotator(1.44/dnum('1 cm'),4*np.pi*10.2e-25/dnum('1 cm-3'),dnum('.025 eV'),dnuc,damping,hbar_norm)
air = dispersion.HumidAir(mks_length,0.4,1e-3)

# Derived Parameters

L = propagation_range[1] - propagation_range[0]
rgn_center = (0.0,0.0,0.5*(propagation_range[0]+propagation_range[1]))

# Set up dictionaries

sim = {}
ray = []
wave = []
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = 'Processing input file...'

ray.append({})
ray[-1]['number'] = (2049,128,2,1)
ray[-1]['bundle radius'] = (.001*r00,.001*r00,.001*r00,.001*r00)
ray[-1]['loading coordinates'] = 'cylindrical'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = band + (0.0,3*r00) + (0.0,2*np.pi) + (0.0,0.0)

# Pump pulse
wave.append({})
wave[-1]['a0'] = (0.0,a00,0.0,0.0) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,0.0,0.0,w00) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = (0.0,0.0,0.0,-0.1/mm)
wave[-1]['supergaussian exponent'] = 2

# Probe pulse
wave.append({})
wave[-1]['a0'] = (0.0,.1*a00,0.0,0.0) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00/4,r00,r00,t00/4) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (wprobe,0.0,0.0,wprobe) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = (0.0,0.0,0.0,-0.1/mm)
wave[-1]['supergaussian exponent'] = 2

# Delay filter (delay probe with respect to pump)
optics.append({})
optics[-1]['object'] = surface.Filter('delay')
optics[-1]['origin'] = (0.0,0.0,-0.05/mm)
optics[-1]['radius'] = 5/cm
optics[-1]['transfer function'] = lambda w: np.exp(1j*w*np.heaviside(w-0.5*(wprobe+w00),0.5)*dnum('.1 ps'))

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('air')
optics[-1]['propagator'] = 'uppe'
optics[-1]['ionizer'] = None
optics[-1]['rotator'] = N2rot
optics[-1]['wave coordinates'] = 'cylindrical'
optics[-1]['wave grid'] = (2049,256,1,5)
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
optics[-1]['origin'] = (0.,0.,L*2)
optics[-1]['euler angles'] = (0.,0.,0.)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (32,8,2,1)
diagnostics['base filename'] = 'out/test'
