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

prop_range = (-20/cm,20/cm)
w00 = 1.0
r00 = dnum('3 mm')
U00 = dnum('6.3 mJ')
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
ray = []
wave = []
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = mess

ray.append({})
ray[-1]['number'] = (2049,64,2,None)
ray[-1]['bundle radius'] = (None,.001*r00,.001*r00,.001*r00)
ray[-1]['loading coordinates'] = 'cylindrical'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = band + (0.0,3*r00) + (0.0,2*np.pi) + (None,None)

wave.append({}) # fundamental
wave[-1]['a0'] = (0.0,a00,0.0,0.0) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,0.0,0.0,w00) # 4-wavenumber: omega,kx,ky,kz
# wave[-1]['focus'] = (100/cm,0.0,0.0,0.0)
wave[-1]['focus'] = (0.0,0.0,0.0,-110/cm)
wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = surface.IdealCompressor('compressor')
optics[-1]['group delay dispersion'] = 4000/fs**2
optics[-1]['center frequency'] = 1.0
optics[-1]['frequency band'] = (0.9,1.1)
optics[-1]['size'] = (1/inch,1/inch)
optics[-1]['origin'] = (0.0,0.0,-95/cm)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = surface.IdealLens('lens')
optics[-1]['radius'] = 0.5/inch
optics[-1]['focal length'] = 90/cm
optics[-1]['origin'] = (0.,0.,-90/cm)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = surface.IdealHarmonicGenerator('SHG')
optics[-1]['harmonic delay'] = 20/fs
optics[-1]['harmonic number'] = 2.0
optics[-1]['frequency band'] = (0.9,1.1)
optics[-1]['efficiency'] = 0.05
optics[-1]['radius'] = 0.5/inch
optics[-1]['origin'] = (0.0,0.0,-80/cm)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('start')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (10*r00,10*r00)
optics[-1]['origin'] = (0.,0.,prop_range[0]-1/mm)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('air')
optics[-1]['propagator'] = 'uppe'
optics[-1]['ionizer'] = ionizer
optics[-1]['rotator'] = N2rot
optics[-1]['wave coordinates'] = 'cylindrical'
optics[-1]['wave grid'] = (2049,256,1,11)
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
optics[-1]['size'] = (12/mm,12/mm,prop_range[1]-prop_range[0])
optics[-1]['origin'] = (0.,0.,(prop_range[0]+prop_range[1])/2)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['window speed'] = air.GroupVelocityMagnitude(1.0)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('stop')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (15/mm,15/mm)
optics[-1]['origin'] = (0.,0.,prop_range[1]+1/mm)
optics[-1]['euler angles'] = (0.,0.,0.)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (128,8,2,None)
diagnostics['base filename'] = 'out/test'
