import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import inputs
from scipy import constants as C

# Plot the mode incident on a parabola that leads to focused Ax component with m=0 only
# (The incident mode must have higher order azimuthal content)

def FractionalPowerScale(array,order):
	idxneg = np.where(array<0)
	idxpos = np.where(array>0)
	array[idxpos] **= 1.0/order
	array[idxneg] *= -1
	array[idxneg] **= 1.0/order
	array[idxneg] *= -1
	return array

def Smooth1D(dist,n):
	for i in range(n):
		dist_h1 = np.roll(dist,-1,axis=0)[1:-1]
		dist_h2 = np.roll(dist,1,axis=0)[1:-1]
		dist[1:-1] = 0.25*dist_h1 + 0.25*dist_h2 + 0.5*dist[1:-1]
	return dist

def Smooth(dist,n):
	for i in range(n):
		dist_h1 = np.roll(dist,-1,axis=0)[1:-1,1:-1]
		dist_h2 = np.roll(dist,1,axis=0)[1:-1,1:-1]
		dist_v1 = np.roll(dist,-1,axis=1)[1:-1,1:-1]
		dist_v2 = np.roll(dist,1,axis=1)[1:-1,1:-1]
		dist[1:-1,1:-1] = 0.125*dist_h1 + 0.125*dist_h2 + 0.125*dist_v1 + 0.125*dist_v2 + 0.5*dist[1:-1,1:-1]
	return dist

if len(sys.argv)==1:
	print('Usage: perturbed-gaussian.py f#')
	exit(1)

dynamic_range = 5
my_color_map = 'Greys_r'

npts = 128
fnum = np.double(sys.argv[1])
intensity = np.zeros((npts,npts))
x = np.outer(np.linspace(-2.0,2.0,npts),np.ones(npts))
y = np.outer(np.ones(npts),np.linspace(-2.0,2.0,npts))
phi = np.arctan(y/x)
rho = np.sqrt(x**2 + y**2)
rho0 = 1.0
f = fnum*2*rho0
q = f/rho
amplitude = np.exp(-rho**2/rho0**2)*np.sqrt(1.0 + (q/(q**2-0.25))**2*np.cos(phi)**2)
amplitude = amplitude.swapaxes(1,0)

plt.figure(1,figsize=(11,6))

plt.subplot(121)
plt.imshow(amplitude,origin='lower',cmap=my_color_map,aspect='equal',extent=[-2.0,2.0,-2.0,2.0])
#b=plt.colorbar()
#b.set_label(r'Amplitude',size=18)
plt.xlabel(r'$x/\rho_0$',size=18)
plt.ylabel(r'$y/\rho_0$',size=18)

plt.subplot(122)
plt.plot(x[:,0],amplitude[np.int(npts/2),:])
plt.plot(y[0,:],amplitude[:,np.int(npts/2)])
plt.xlabel(r'$x/\rho_0$',size=18)
plt.ylabel(r'Amplitude',size=18)	

plt.tight_layout()
	
plt.show()

