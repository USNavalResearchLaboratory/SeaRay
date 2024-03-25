from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.ionization as ionization
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools 

# 3D USPL filamentation, using Paraxial module.
# Initial beam is put through a random amplitude screen.
# Once the filament forms resolution is lost.

# Suggested visualization: use viewer.ipynb to interactively view A(t,x,y)

mks_length = 0.8e-6 / (2*np.pi)
helper = input_tools.InputHelper(mks_length)
dnum = helper.dnum
cm = 100*mks_length
mm = 1000*mks_length
um = 1e6*mks_length

# Control Parameters

wavelength = 0.8/um
waist = 100/um
w00 = 2*np.pi/wavelength
U00 = dnum('5 mJ')
t00 = dnum('50 fs')
chi3 = helper.chi3(1.0,'1e-23 m2/W')
propagation_range = (-20/cm,-15/cm)
rbox = 2/mm

Uion = dnum('12.1 eV')
ngas = dnum('5.4e18 cm-3')
Zeff = 0.53
ionizer = ionization.StitchedPPT(mks_length,True,Uion,Zeff,lstar=0,l=0,m=0,w0=w00,terms=80)
air = dispersion.HumidAir(mks_length,0.4,1e-3)

# Derived Parameters

L = propagation_range[1] - propagation_range[0]
time_to_focus = abs(2*propagation_range[0])
diffraction_angle = wavelength / (np.pi*waist)
r00 = time_to_focus * diffraction_angle
a00 = helper.a0(U00,t00,r00,w00)
rgn_center = (0.0,0.0,0.5*(propagation_range[0]+propagation_range[1]))
t00,band = helper.TransformLimitedBandwidth(w00,t00,16)

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
ray[-1]['number'] = (128,32,32,None)
ray[-1]['bundle radius'] = (None,.001*r00,.001*r00,.001*r00)
ray[-1]['loading coordinates'] = 'cartesian'
# Ray box is always put at the origin
# It will be transformed appropriately by SeaRay to start in the wave
ray[-1]['box'] = band + (-3*r00,3*r00) + (-3*r00,3*r00) + (None,None)

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
optics[-1]['object'] = surface.NoiseMask('screen')
optics[-1]['grid'] = (128,128)
optics[-1]['amplitude'] = 0.7
optics[-1]['inner scale'] = 1/um
optics[-1]['outer scale'] = 300/um
optics[-1]['frequency band'] = band
optics[-1]['size'] = (2*rbox,2*rbox)
optics[-1]['origin'] = (0.0,0.0,rgn_center[2]-L/2-1.0)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('air')
optics[-1]['propagator'] = 'paraxial'
optics[-1]['ionizer'] = ionizer
optics[-1]['wave coordinates'] = 'cartesian'
optics[-1]['wave grid'] = (128,128,128,5)
optics[-1]['density reference'] = ngas
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
diagnostics['orbit rays'] = (4,4,4,None)
diagnostics['base filename'] = 'out/test'
