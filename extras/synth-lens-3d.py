import numpy as np
import scipy.optimize
import scipy.interpolate
from mayavi import mlab

# Plasma lens is scalable
# Length units are arbitrary
f = 1
d = 0.2*f
# Type of lens is determined here
zetam = -10000
zetap = f

# Set up the 2D phase array
Nr = 100
Nz = 100
psi = np.zeros((Nr+2,Nz+2))

# Limits of the computational box
Rbox = 0.75*f
Lbox = 1.5*f

# Set up the mesh points with spacing (dr,dz)
dr = Rbox/Nr
dz = Lbox/Nz
r_pts = np.linspace(-dr/2,Rbox+dr/2,Nr+2)
z_pts = np.linspace(-Lbox/2-dz/2,Lbox/2+dz/2,Nz+2)

# Compute the phase in length units
# psi = z' = psi[radians] * c / omega
for i,r in enumerate(r_pts):
	for k,z in enumerate(z_pts):
		g = lambda x : 0.5 + 0.5*np.tanh(2*x)
		Rm = np.sign(zetam)*np.sqrt(r**2+(z-zetam)**2)
		Rp = np.sign(zetap)*np.sqrt(r**2+(z-zetap)**2)
		R = lambda x : 1/((1-g(x/d))/Rm + g(x/d)/Rp)
		merit = lambda x : r**2 + (z-x-R(x))**2 - R(x)**2
		try:
			psi[i,k] = scipy.optimize.brentq(merit,-f,f,tol=1e-12)
		except:
			psi[i,k] = scipy.optimize.newton(merit,z,tol=1e-9)

grad = np.gradient(psi,dr,dz)
nbar1 = 1.0 - (grad[0]**2 + grad[1]**2)

# Since wave is spherical everywhere, we can clip the plasma
# inside any sphere centered on the focal point.
# r2d = np.outer(r_pts,numpy.ones(Nz+2))
# z2d = np.outer(numpy.ones(Nr+2),z_pts-zetap)
# r2 = r2d**2 + z2d**2
# nbar[np.where(r2<f/4)]=1e-10
# Throw away guard cells
nbar = nbar1[1:-1,1:-1]

# Interpolating function for 2D data

nfunc = scipy.interpolate.interp2d(r_pts,z_pts,nbar1)

# Extend to 3D for viz
x_pts = np.copy(r_pts)[1:-1]
y_pts = np.copy(r_pts)[1:-1]
z_pts = np.copy(z_pts)[1:-1]
x_pts = np.concatenate((-x_pts[::-1],x_pts))
y_pts = np.concatenate((-y_pts[::-1],y_pts))
Nx = len(x_pts)
Ny = len(y_pts)
x1 = np.einsum('i,j,k',x_pts,np.ones(Ny),np.ones(Nz))
x2 = np.einsum('i,j,k',np.ones(Nx),y_pts,np.ones(Nz))
x3 = np.einsum('i,j,k',np.ones(Nx),np.ones(Ny),z_pts)
rho = np.sqrt(np.outer(x_pts,np.ones(Ny))**2 + np.outer(np.ones(Nx),y_pts)**2)
nbar3d = np.zeros((Nx,Ny,Nz))
for k,z0 in enumerate(z_pts):
	f = scipy.interpolate.interp1d(r_pts,nbar1[:,k+1],bounds_error=False)
	nbar3d[:,:,k] = f(rho)
mlab.figure(bgcolor=(0,0,0))
# src = mlab.pipeline.array2d_source(nbar3d[:,100,:])
# obj = mlab.pipeline.image_actor(src,colormap='gist_ncar')
src = mlab.pipeline.scalar_field(x1,x2,x3,nbar3d)
obj = mlab.pipeline.iso_surface(src,contours=[0.1,0.25,0.4],opacity=1.0)

mlab.savefig('test-3d.png')
np.save('ideal-form-3d',nbar3d)

mlab.show()
