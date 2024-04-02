from scipy import constants as C
import numpy as np
from scipy.optimize import brentq
import modules.dispersion as dispersion
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools 

# Example input file for ray-in-cell propagation through ideal form plasma lens.
# The alternate test case creates a quartic lens on a grid (will have caustics).
# The ideal form lens data must be in ./extras.  Generate with synth-lens.py.

# Suggested plotter command:
# python plotter.py out/test o31
# note near perfect focus (zoom in to see caustic)

mks_length = 0.8e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
fs = 1e15*mks_length/C.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Preprocessing calculations
# Use thick lens theory to set up channel parameters for given focal length

ideal_form = True
theta = 0.0
w00 = 1.0
f = 10/mm
f_num = 1.0
Rlens = 0.75*f
if ideal_form:
	Lch = 1.5*f
	lens_object = volume.AxisymmetricGrid('plasma')
else:
	Lch = 0.2*f
	lens_object = volume.AxisymmetricTestGrid('plasma')
r00 = 0.5*f/f_num # spot size of radiation
c0 = 0.01
h0 = np.sqrt(1-c0)
t = Lch/np.sqrt(1-c0)
Omega = brentq(lambda q : q*np.tan(q) - t/(f-Lch/2), 0.0, 0.999*np.pi/2) / t
c2 = Omega**2
c4 = -Omega**4/4
x0 = 100*r00
c4 *= 1 + Lch**2*(0.33/x0**2 + 0.5*Omega**2/h0**2 + Omega**2)
c6 = 0.0
eik_to_caustic = 1/mm

rb = bundle_scale*r00
t00,band = helper.TransformLimitedBandwidth(w00,'100 fs',4)
a00 = helper.InitialVectorPotential(w00,1.0,f,f_num)
mess = mess + helper.ParaxialFocusMessage(w00,1.0,f,f_num)

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
            'origin': (None,0,0,-f),
            'euler angles': helper.rot_zx(theta),
            'number': (1,128,16,None),
            'bundle radius': (None,) + (rb,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': (0.9,1.1) + (0,1.4*r00) + (0,2*np.pi) + (None,None)
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
optics[-1]['object'] = lens_object
optics[-1]['radial coefficients'] = (c0,c2,c4,c6)
optics[-1]['mesh points'] = (400,400)
optics[-1]['file'] = 'extras/ideal-form.npy'
optics[-1]['density multiplier'] = 1.0
optics[-1]['dispersion inside'] = dispersion.ColdPlasma()
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['radius'] = Rlens
optics[-1]['length'] = Lch
optics[-1]['origin'] = (None,0,0,0)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['dt'] = Lch/1000
optics[-1]['steps'] = 1500
optics[-1]['subcycles'] = 10

# optics.append({})
# optics[-1]['object'] = surface.CylindricalProfiler('det')
# optics[-1]['size'] = (1/mm,1/mm,0.1/mm)
# optics[-1]['wave grid'] = (4096,2,32)
# optics[-1]['distance to caustic'] = eik_to_caustic
# optics[-1]['origin'] = (None,0,0,f - eik_to_caustic)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('terminus')
optics[-1]['size'] = (10/mm,10/mm)
optics[-1]['euler angles'] = (0.0,0.0,0.0)
optics[-1]['origin'] = (None,0,0,15/mm)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (1,8,4,None)
diagnostics['base filename'] = 'out/test'
