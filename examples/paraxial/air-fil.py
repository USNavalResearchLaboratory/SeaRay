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
pulse_energy_mks = 5e-3
pulse_duration_mks = 50e-15
propagation_length_mks = 0.2
sim_box_radius_mks = 3e-3
n2_air_mks = 5e-23
tstr = str(pulse_duration_mks*1e15) + ' fs'

# Required SeaRay input deck variables

sim = []
wave = []
ray = []
optics = []
diagnostics = []

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
ionizer = ionization.PPT(Uion_au,Zeff,ngas_mks,mks_length,terms=6)

# Derived Parameters

helper = input_tools.InputHelper(mks_length)

diffraction_angle = lambda_mks / (np.pi*vac_waist_radius_mks)
start_radius_mks = propagation_length_mks * diffraction_angle
P0_mks = pulse_energy_mks / pulse_duration_mks
I0_mks = 2*P0_mks/(np.pi*start_radius_mks**2)
L = propagation_length_mks / mks_length
rbox = sim_box_radius_mks / mks_length
w00 = 1.0
r00 = start_radius_mks / mks_length
a00 = helper.Wcm2_to_a0(I0_mks*1e-4,lambda_mks)
chi3 = helper.mks_n2_to_chi3(1.0,n2_air_mks)
t00,band = helper.TransformLimitedBandwidth(w00,tstr,32)

# Set up dictionaries

for i in range(1):

    sim.append({'mks_length' : mks_length ,
                'mks_time' : mks_length/C.c ,
                'message' : 'Processing input file...'})

    wave.append([
        {    # EM 4-potential (eA/mc^2) , component 0 not used
            'a0' : (0.0,a00,0.0,0.0) ,
            # 4-vector of pulse metrics: duration,x,y,z 1/e spot sizes
            'r0' : (t00,r00,r00,t00) ,
            # 4-wavenumber: omega,kx,ky,kz
            'k0' : (w00,0.0,0.0,w00) ,
            # 0-component of focus is time at which pulse reaches focal point.
            # If time=0 use paraxial wave, otherwise use spherical wave.
            # Thus in the paraxial case the pulse always starts at the waist.
            'focus' : (L,0.0,0.0,0.0),
            'supergaussian exponent' : 2},
        ])

    ray.append({'number' : (512,64,2,1),
                'bundle radius' : (.001*r00,.001*r00,.001*r00,.001*r00),
                'loading coordinates' : 'cylindrical',
                # Ray box is always put at the origin
                # It will be transformed appropriately by SeaRay to start in the wave
                'box' : band + (0.0,3*r00,0.0,2*np.pi,-2*t00,2*t00)})

    optics.append([

        {   'object' : volume.AnalyticBox('air'),
            'propagator' : 'paraxial',
            'ionizer' : ionizer,
            'wave coordinates' : 'cylindrical',
            'wave grid' : (512,128,1,15),
            'density function' : '1.0',
            'density lambda' : lambda x,y,z,r2 : np.ones(r2.shape),
            'frequency band' : band,
            'subcycles' : 20,
            'dispersion inside' : air,
            'dispersion outside' : dispersion.Vacuum(),
            'chi3' : chi3,
            'size' : (rbox,rbox,L),
            'origin' : (0.,0.,0.),
            'euler angles' : (0.,0.,0.),
            'window speed' : air.GroupVelocityMagnitude(1.0)},

        {   'object' : surface.EikonalProfiler('stop'),
            'frequency band' : (0,3),
            'size' : (10*rbox,10*rbox),
            'origin' : (0.,0.,L*2),
            'euler angles' : (0.,0.,0.)}
        ])

    diagnostics.append({'suppress details' : False,
                        'clean old files' : True,
                        'orbit rays' : (4,8,2,1),
                        'base filename' : 'out/test'})
