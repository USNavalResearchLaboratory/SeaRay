from scipy import constants as C
import numpy as np
import dispersion
import ionization
import surface
import volume
import input_tools

# Simple USPL air plasma using Paraxial module.
# Propagate from -L/2 to L/2 where L = propagation length

# Control Parameters

lambda_mks = 0.8e-6
vac_waist_radius_mks = 100e-6
pulse_energy_mks = 0.5e-3
pulse_duration_mks = 50e-15
propagation_range_mks = (-0.1,0.1)
sim_box_radius_mks = 1.5e-3
n2_air_mks = 5e-23
tstr = str(pulse_duration_mks*1e15) + ' fs'

# Control Objects

mks_length = lambda_mks  / (2*np.pi)

# Select one dispersion model for air:
# air = dispersion.Vacuum()
# air = dispersion.SimpleAir(mks_length)
# air = dispersion.DryAir(mks_length)
air = dispersion.HumidAir(mks_length,0.4,1e-3)

# Ionization model (at present only ADK or PPT tunneling works on GPU):
Uion_au = 12.1 / (C.alpha**2*C.m_e*C.c**2/C.e)
ngas_mks = 5.4e18 * 1e6
Zeff = 0.53
ionizer = ionization.ADK(Uion_au,Zeff,ngas_mks,mks_length,terms=6)

# Derived Parameters

helper = input_tools.InputHelper(mks_length)

propagation_length_mks = propagation_range_mks[1] - propagation_range_mks[0]
dist_to_focus_mks = abs(2*propagation_range_mks[0])
diffraction_angle = lambda_mks / (np.pi*vac_waist_radius_mks)
start_radius_mks = dist_to_focus_mks * diffraction_angle
P0_mks = pulse_energy_mks / pulse_duration_mks
I0_mks = 2*P0_mks/(np.pi*start_radius_mks**2)

rgn_center = (0.0,0.0,0.5*(propagation_range_mks[0]+propagation_range_mks[1]) / mks_length)
L = propagation_length_mks / mks_length
time_to_focus = dist_to_focus_mks / mks_length
rbox = sim_box_radius_mks / mks_length
w00 = 1.0
r00 = start_radius_mks / mks_length
a00 = helper.Wcm2_to_a0(I0_mks*1e-4,lambda_mks)
chi3 = helper.mks_n2_to_chi3(1.0,n2_air_mks)
t00,band = helper.TransformLimitedBandwidth(w00,tstr,50)

# Set up dictionaries

sim = {}
ray = {}
wave = []
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = 'Processing input file...'

ray['number'] = (512,64,2,1)
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
wave[-1]['focus'] = (time_to_focus,0.0,0.0,0.0)
wave[-1]['supergaussian exponent'] = 2

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('air')
optics[-1]['propagator'] = 'paraxial'
optics[-1]['ionizer'] = ionizer
optics[-1]['wave coordinates'] = 'cylindrical'
optics[-1]['wave grid'] = (512,128,1,15)
optics[-1]['density function'] = '1.0'
optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(r2.shape)
optics[-1]['frequency band'] = band
optics[-1]['subcycles'] = 4
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
diagnostics['orbit rays'] = (4,8,2,1)
diagnostics['base filename'] = 'out/test'
