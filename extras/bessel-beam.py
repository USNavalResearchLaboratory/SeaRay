import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import constants as C
from scipy.special import j0

my_color_map = 'jet'

npts = 128
rmax = 6.0*np.pi
intensity = np.zeros((npts,npts))
z = np.outer(np.ones(npts),np.linspace(-rmax,rmax,npts))
rho = np.outer(np.linspace(0.0,rmax,npts),np.ones(npts))
w0 = 1.0
kperp = w0/2.0
amplitude = np.cos(z*np.sqrt(w0**2 - kperp**2))*j0(kperp*rho)

plt.figure(1,figsize=(6,6))

plt.imshow(amplitude,origin='lower',cmap=my_color_map,aspect='equal',extent=[-rmax,rmax,0,rmax])
#b=plt.colorbar()
#b.set_label(r'Amplitude',size=18)
plt.xlabel(r'$\omega z/c$',size=18)
plt.ylabel(r'$\omega \rho/c$',size=18)

plt.tight_layout()
	
plt.show()

