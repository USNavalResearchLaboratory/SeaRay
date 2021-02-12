from scipy import constants as C
import numpy as np
import dispersion
import ionization
import surface
import volume
import input_tools

# Example of an UPPE batch job
# Examines effect of varying parameters on two-color terahertz.

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
chi3 = helper.chi3(1.0,'10e-24 m2/W')
# Setting the lower frequency bound to zero triggers carrier resolved treatment
band = (0.0,5.5)
Uion = dnum('12.1 eV')
ngas = dnum('5.4e18 cm-3')
Zeff = 0.53
ionizer = ionization.StitchedPPT(mks_length,False,Uion,Zeff,lstar=0,l=0,m=0,w0=w00,terms=80)
air = dispersion.HumidAir(mks_length,0.4,1e-4)
air.add_opacity_region(1/cm,0.01/um,0.3/um)

# Set up dictionaries
# Since this is a batch job, create outer lists.
# The outer list is a list of simulations.

sim = []
ray = []
wave = []
optics = []
diagnostics = []

for irun in range(5):

    # Add dictionaries or lists of dictionaries to this simulation
    # Then access current simulation using index -1
    # To see what is varied over runs, search for irun in the dictionary values
    sim.append({})
    ray.append([])
    wave.append([])
    optics.append([])
    diagnostics.append({})

    sim[-1]['mks_length'] = mks_length
    sim[-1]['mks_time'] = mks_length/C.c
    sim[-1]['message'] = mess

    ray[-1].append({})
    ray[-1][-1]['number'] = (2049,64,2,1)
    ray[-1][-1]['bundle radius'] = (.001*r00,.001*r00,.001*r00,.001*r00)
    ray[-1][-1]['loading coordinates'] = 'cylindrical'
    # Ray box is always put at the origin
    # It will be transformed appropriately by SeaRay to start in the wave
    ray[-1][-1]['box'] = band + (0.0,3*r00) + (0.0,2*np.pi) + (0.0,0.0)

    wave[-1].append({}) # fundamental
    wave[-1][-1]['a0'] = (0.0,a00,0.0,0.0) # EM 4-potential (eA/mc^2) , component 0 not used
    wave[-1][-1]['r0'] = (t00,r00,r00,t00) # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
    wave[-1][-1]['k0'] = (w00,0.0,0.0,w00) # 4-wavenumber: omega,kx,ky,kz
    # wave[-1][-1]['focus'] = (100/cm,0.0,0.0,0.0)
    wave[-1][-1]['focus'] = (0.0,0.0,0.0,-110/cm)
    wave[-1][-1]['supergaussian exponent'] = 2

    optics[-1].append({})
    optics[-1][-1]['object'] = surface.IdealCompressor('compressor')
    optics[-1][-1]['group delay dispersion'] = 4000/fs**2
    optics[-1][-1]['center frequency'] = 1.0
    optics[-1][-1]['frequency band'] = (0.9,1.1)
    optics[-1][-1]['size'] = (1/inch,1/inch)
    optics[-1][-1]['origin'] = (0.0,0.0,-95/cm)
    optics[-1][-1]['euler angles'] = (0.,0.,0.)

    optics[-1].append({})
    optics[-1][-1]['object'] = surface.IdealLens('lens')
    optics[-1][-1]['radius'] = 0.5/inch
    optics[-1][-1]['focal length'] = 90/cm
    optics[-1][-1]['origin'] = (0.,0.,-90/cm)
    optics[-1][-1]['euler angles'] = (0.,0.,0.)

    optics[-1].append({})
    optics[-1][-1]['object'] = surface.IdealHarmonicGenerator('SHG')
    optics[-1][-1]['harmonic delay'] = 20/fs
    optics[-1][-1]['harmonic number'] = 2.0
    optics[-1][-1]['frequency band'] = (0.9,1.1)
    optics[-1][-1]['efficiency'] = 0.05
    optics[-1][-1]['radius'] = 0.5/inch
    optics[-1][-1]['origin'] = (0.0,0.0,-80/cm)
    optics[-1][-1]['euler angles'] = (0.,0.,0.)

    optics[-1].append({})
    optics[-1][-1]['object'] = surface.EikonalProfiler('start')
    optics[-1][-1]['frequency band'] = (0,3)
    optics[-1][-1]['size'] = (10*r00,10*r00)
    optics[-1][-1]['origin'] = (0.,0.,prop_range[0]-1/mm)
    optics[-1][-1]['euler angles'] = (0.,0.,0.)

    optics[-1].append({})
    optics[-1][-1]['object'] = volume.AnalyticBox('air')
    optics[-1][-1]['propagator'] = 'uppe'
    optics[-1][-1]['ionizer'] = ionizer
    optics[-1][-1]['wave coordinates'] = 'cylindrical'
    optics[-1][-1]['wave grid'] = (2049,256,1,21)
    optics[-1][-1]['radial modes'] = 128
    optics[-1][-1]['density reference'] = ngas
    optics[-1][-1]['density function'] = '1.0'
    optics[-1][-1]['density lambda'] = lambda x,y,z,r2 : np.ones(r2.shape)
    optics[-1][-1]['frequency band'] = band
    optics[-1][-1]['nonlinear band'] = (0.0,0.5)
    optics[-1][-1]['subcycles'] = 1
    optics[-1][-1]['minimum step'] = 1.0
    optics[-1][-1]['dispersion inside'] = air
    optics[-1][-1]['dispersion outside'] = dispersion.Vacuum()
    optics[-1][-1]['chi3'] = chi3*float(irun)
    optics[-1][-1]['size'] = (12/mm,12/mm,prop_range[1]-prop_range[0])
    optics[-1][-1]['origin'] = (0.,0.,(prop_range[0]+prop_range[1])/2)
    optics[-1][-1]['euler angles'] = (0.,0.,0.)
    optics[-1][-1]['window speed'] = air.GroupVelocityMagnitude(1.0)

    optics[-1].append({})
    optics[-1][-1]['object'] = surface.EikonalProfiler('stop')
    optics[-1][-1]['frequency band'] = (0,3)
    optics[-1][-1]['size'] = (15/mm,15/mm)
    optics[-1][-1]['origin'] = (0.,0.,prop_range[1]+1/mm)
    optics[-1][-1]['euler angles'] = (0.,0.,0.)

    diagnostics[-1]['suppress details'] = False
    diagnostics[-1]['clean old files'] = True
    diagnostics[-1]['orbit rays'] = (128,8,2,1)
    diagnostics[-1]['base filename'] = 'out/test'
