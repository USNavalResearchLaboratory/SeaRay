from scipy import constants as C
import numpy as np
from scipy.optimize import brentq
import modules.dispersion as dispersion
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools 

# Example input file for dispersive quartic plasma lens.
# For the quartic lens, most of the caustic is after the paraxial focus.
# Therefore the wave zone calculation is done in the forward direction.
# If running on a powerful computer, increase z resolution of detector.
# IMPORTANT: frequency and azimuthal nodes in source and detector must match

# Suggested plotter arguments for still image: det=3,4/0,0
# Suggested plotter arguments for movie: det=0,1/0,:/0.5 drange=3

# Units and scales

mks_length = 0.8e-6 / (2*np.pi)
bundle_scale = 1e-4
mm = 1000*mks_length
fs = 1e15*mks_length/C.c
deg = 180/np.pi
helper = input_tools.InputHelper(mks_length)
mess = 'Processing input file...\n'

# Preprocessing calculations
# Use thick lens theory to set up channel parameters for given focal length

theta = 0.0
w00 = 1.0
f = 10/mm
f_num = 5.0
Rlens = 10/mm
Lch = 2/mm

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
eik_to_caustic = 2/mm

rb = bundle_scale*r00
t00,band = helper.TransformLimitedBandwidth(w00,'50 fs',4)
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
            'origin': (None,0,0,-6/mm),
            'euler angles': helper.rot_zx(theta),
            'number': (64,128,4,None),
            'bundle radius': (None,) + (rb,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': band + (0,1.5*r00) + (0,2*np.pi) + (None,None)
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
optics[-1]['object'] = volume.AnalyticCylinder('plasma')
optics[-1]['dispersion inside'] = dispersion.ColdPlasma()
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['radius'] = Rlens
optics[-1]['length'] = Lch
optics[-1]['origin'] = (None,0,0,0)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['density function'] = str(c0)+'+'+str(c2)+'*r2+'+str(c4)+'*r2*r2'
optics[-1]['density lambda'] = lambda x,y,z,r2 : c0 + c2*r2 + c4*r2*r2
optics[-1]['dt'] = Lch/1000
# Use enough steps to make sure rays reach end of box.
# Too many steps is OK, SeaRay can adjust down automatically.
# Too few steps is not OK.
optics[-1]['steps'] = 1200
optics[-1]['subcycles'] = 10
optics[-1]['work group 1d'] = (64,) # POCL won't accept None in some cases

optics.append({})
optics[-1]['object'] = surface.CylindricalProfiler('det')
optics[-1]['frequency band'] = band
optics[-1]['size'] = (.6/mm,.6/mm,2/mm)
optics[-1]['wave grid'] = (64,1024,4,8)
optics[-1]['distance to caustic'] = eik_to_caustic
optics[-1]['origin'] = (None,0,0,f - eik_to_caustic)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('terminus')
optics[-1]['size'] = (3/mm,3/mm)
optics[-1]['euler angles'] = (0.0,0.0,0.0)
optics[-1]['origin'] = (None,0,0,20/mm)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (4,4,4,None)
diagnostics['base filename'] = 'out/test'
