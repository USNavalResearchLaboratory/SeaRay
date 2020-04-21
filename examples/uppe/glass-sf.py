from scipy import constants as C
import numpy as np
import dispersion
import surface
import volume
import input_tools

# Example input file for axisymmetric UPPE wave equation.
# Illustrates self focusing, self phase modulation, and group velocity dispersion in glass
# Effects are most easily observed using the interactive viewer

mks_length = 0.8e-6 / (2*np.pi)
mess = 'Processing input file...\n'

# Preprocessing calculations

helper = input_tools.InputHelper(mks_length)

glass = dispersion.BK7(mks_length)
# Suppress out of band frequencies
glass.add_opacity_region(2000.0,0.1e-6,0.6e-6)
glass.add_opacity_region(2000.0,1.2e-6,4e-6)
w00 = 1.0
r00 = 100e-6 / mks_length
a00 = helper.Wcm2_to_a0(6e12,0.8e-6)
chi3 = helper.mks_n2_to_chi3(1.5,1e-20)
mess = mess + '  a0 = ' + str(a00) + '\n'
mess = mess + '  chi3 = ' + str(chi3) + '\n'

# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,2.5)
t00,pulse_band = helper.TransformLimitedBandwidth(w00,'15 fs',1.0)

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
ray = {}
wave = []
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = mess

ray['number'] = (1025,64,4,1)
ray['bundle radius'] = (.001*r00,.001*r00,.001*r00,.001*r00)
ray['loading coordinates'] = 'cylindrical'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray['box'] = band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)

wave.append({})
wave[-1]['a0'] = (0.0,a00,0.0,0.0) # EM 4-potential (eA/mc^2) , component 0 not used
wave[-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
wave[-1]['k0'] = (w00,0.0,0.0,w00) # 4-wavenumber: omega,kx,ky,kz
# 0-component of focus is time at which pulse reaches focal point.
# If time=0 use paraxial wave, otherwise use spherical wave.
# Thus in the paraxial case the pulse always starts at the waist.
wave[-1]['focus'] = (0.0,0.0,0.0,-1.0)
wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('start')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (6*r00,6*r00)
optics[-1]['origin'] = (0.,0.,-0.5)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('glass')
optics[-1]['propagator'] = 'uppe'
optics[-1]['wave coordinates'] = 'cylindrical'
optics[-1]['wave grid'] = (1025,64,1,9)
optics[-1]['density function'] = '1.0'
optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(r2.shape)
optics[-1]['frequency band'] = band
optics[-1]['subcycles'] = 4
optics[-1]['minimum step'] = .3e-6/mks_length
optics[-1]['dispersion inside'] = glass
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['chi3'] = chi3
optics[-1]['size'] = (6*r00,6*r00,Lprop)
optics[-1]['origin'] = (0.,0.,Lprop/2)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['window speed'] = glass.GroupVelocityMagnitude(1.0)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('stop')
optics[-1]['frequency band'] = (0,3)
optics[-1]['size'] = (8*r00,8*r00)
optics[-1]['origin'] = (0.,0.,2*Lprop)
optics[-1]['euler angles'] = (0.,0.,0.)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (8,4,4,1)
diagnostics['base filename'] = 'out/test'
