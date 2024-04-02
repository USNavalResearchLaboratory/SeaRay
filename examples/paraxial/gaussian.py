from scipy import constants as C
import numpy as np
import modules.dispersion as dispersion
import modules.surface as surface
import modules.volume as volume
import modules.input_tools as input_tools 

# Example input file for 3D paraxial wave equation.
# Rays are converted to paraxial wave which propagates through focus.
# Rays are relaunched at a symmetric point downstream.

# Suggested plotter commands:
# python plotter out/test exit=1,8
#   Upturned parabolas correspond to rays defocusing out of wave zone.
# python plotter out/test exit=1,8/4,.999,1.001/6,-.001,.001
#   Isolates an individual parabola (fixes w=1 and ky=0)
# The interactive viewer can be used to examine the paraxial wave itself.

# Preprocessing calculations

mks_length = 0.8e-6 / (2*np.pi)
cm = 100*mks_length
mess = 'Processing input file...\n'
helper = input_tools.InputHelper(mks_length)

w00 = 1.0
f = 10/cm
f_num = 50.0
r00 = f/(2*f_num) # spot size of radiation
t00,band = helper.TransformLimitedBandwidth(w00,'100 fs',8)
a00,waist,zR = helper.ParaxialParameters(w00,1.0,f,f_num)
mess = mess + helper.ParaxialFocusMessage(w00,1.0,f,f_num)

# Set up dictionaries

sim = {}
optics = []
diagnostics = {}

sim['mks_length'] = mks_length
sim['mks_time'] = mks_length/C.c
sim['message'] = mess

helper.set_pos((None,0,0,-2))
sources = [
    {
        'rays': {
            'origin': helper.curr_pos,
            'euler angles': (0,0,0),
            'number': (64,64,8,None),
            'bundle radius': (None,) + (.001*r00,)*3,
            'loading coordinates': 'cylindrical',
            'bounds': band + (0,3*r00) + (0,2*np.pi) + (None,None)
        },
        'waves': [
            {
                'a0': (None,a00,0,None),
                'r0': (t00,r00,r00,t00),
                'k0': (w00,None,None,w00),
                'mode': (None,0,0,None),
                'basis': 'hermite'
            },
        ]
    }
]

optics.append({})
optics[-1]['object'] = surface.IdealLens('L1')
optics[-1]['origin'] = helper.move(0,0,1)
optics[-1]['euler angles'] = (0,0,0)
optics[-1]['radius'] = 1/cm
optics[-1]['focal length'] = f

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('start')
optics[-1]['size'] = (f/8,f/8)
optics[-1]['origin'] = (None,0,0,0)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = volume.AnalyticBox('vacuum')
optics[-1]['density reference'] = 1.0
optics[-1]['density function'] = '1.0'
optics[-1]['density lambda'] = lambda x,y,z,r2 : np.ones(x.shape)
optics[-1]['frequency band'] = band
optics[-1]['wave grid'] = (64,128,128,9)
optics[-1]['wave coordinates'] = 'cartesian'
optics[-1]['dispersion inside'] = dispersion.Vacuum()
optics[-1]['dispersion outside'] = dispersion.Vacuum()
optics[-1]['size'] = (36*waist,36*waist,8*zR)
optics[-1]['origin'] = (None,0,0,f)
optics[-1]['euler angles'] = (0.,0.,0.)
optics[-1]['propagator'] = 'paraxial'
optics[-1]['subcycles'] = 1

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('exit')
optics[-1]['size'] = (f/8,f/8)
optics[-1]['origin'] = (None,0,0,f+4*zR+10.0)
optics[-1]['euler angles'] = (0.,0.,0.)

optics.append({})
optics[-1]['object'] = surface.EikonalProfiler('stop')
optics[-1]['size'] = (f/8,f/8)
optics[-1]['origin'] = (None,0,0,2*f)
optics[-1]['euler angles'] = (0.,0.,0.)

diagnostics['suppress details'] = False
diagnostics['clean old files'] = True
diagnostics['orbit rays'] = (1,16,4,None)
diagnostics['base filename'] = 'out/test'
