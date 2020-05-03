from scipy import constants as C
import numpy as np
from scipy.optimize import brentq
import dispersion
import ionization
import surface
import volume
import input_tools

# Example input file for 2 color THz generation via UPPE module.

mks_length = 0.8e-6 / (2*np.pi)
cm = 100*mks_length
mm = 1000*mks_length
um = 1e6*mks_length
inch = cm/2.54
mess = 'Processing input file...\n'

# Preprocessing calculations

helper = input_tools.InputHelper(mks_length)

prop_range = (-10/cm,0/cm)
L = prop_range[1]-prop_range[0]
air = dispersion.HumidAir(mks_length,0.4,1e-3)
air.add_opacity_region(0.5/cm,0.05/um,0.3/um)
air.add_opacity_region(10/cm,13/um,17/um)
w00 = 1.0
r00 = 0.5/cm
P0_mks = 7e-3 / 80e-15
I0_mks = 2*P0_mks/(np.pi*r00**2*mks_length**2)
a800 = helper.Wcm2_to_a0(I0_mks*1e-4,0.8e-6)
a400 = helper.Wcm2_to_a0(0.1*I0_mks*1e-4,0.4e-6)
chi3 = .01*helper.mks_n2_to_chi3(1.0,5e-23)
mess = mess + '  a800 = ' + str(a800) + '\n'
mess = mess + '  a400 = ' + str(a400) + '\n'
mess = mess + '  chi3 = ' + str(chi3) + '\n'

# Ionization model
Uion_au = 12.1 / (C.alpha**2*C.m_e*C.c**2/C.e)
ngas_mks = 5.4e18 * 1e6
Zeff = 0.53
ionizer = ionization.PPT(0.8e-6,Uion_au,Zeff,ngas_mks,mks_length,80)

# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,5.5)
t00,pulse_band = helper.TransformLimitedBandwidth(w00,'80 fs',1.0)

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
ray[-1]['number'] = (2049,64,2,1)
ray[-1]['bundle radius'] = (.001*r00,.001*r00,.001*r00,.001*r00)
ray[-1]['loading coordinates'] = 'cylindrical'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)

wave.append({}) # fundamental
wave[-1]['a0'] = (0.0,a800,0.0,0.0) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,0.0,0.0,-w00) # 4-wavenumber: omega,kx,ky,kz
wave[-1]['focus'] = (0.0,0.0,0.0,-95/cm)
wave[-1]['supergaussian exponent'] = 2

wave.append({}) # second harmonic
wave[-1]['a0'] = (0.0,a400,0.0,0.0) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (2*w00,0.0,0.0,-2*w00) # 4-wavenumber: omega,kx,ky,kz
wave[-1]['focus'] = (0.0,0.0,0.0,-95/cm)
wave[-1]['supergaussian exponent'] = 2
wave[-1]['phase'] = np.pi/2

optics.append({})
optics[-1]['object'] = surface.SphericalCap('M1')
optics[-1]['reflective'] = True
optics[-1]['radius of sphere'] = 200/cm
optics[-1]['radius of edge'] = 0.5/inch
optics[-1]['origin'] = (0.,0.,-100/cm)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('air')
optics[-1]['propagator'] = 'uppe'
optics[-1]['ionizer'] = ionizer
optics[-1]['wave coordinates'] = 'cylindrical'
optics[-1]['wave grid'] = (2049,128,1,7)
optics[-1]['density function'] = 'exp(-4*x.s3*x.s3/'+str(L**2)+')'
optics[-1]['density lambda'] = lambda x,y,z,r2 : np.exp(-4*z**2/L**2)
optics[-1]['frequency band'] = band
optics[-1]['nonlinear band'] = (0.0,0.5)
optics[-1]['subcycles'] = 10
optics[-1]['minimum step'] = 1.0
optics[-1]['dispersion inside'] = air
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['chi3'] = chi3
optics[-1]['size'] = (6/mm,6/mm,prop_range[1]-prop_range[0])
optics[-1]['origin'] = (0.,0.,(prop_range[0]+prop_range[1])/2)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['window speed'] = air.GroupVelocityMagnitude(1.0)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('stop')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (10*r00,10*r00)
optics[-1]['origin'] = (0.,0.,40/cm)
optics[-1]['euler angles'] = (0.,0.,0.)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (4,8,2,1)
diagnostics['base filename'] = 'out/test'
