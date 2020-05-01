import numpy
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
my_color_map = 'nipy_spectral'

# Plasma lens is scalable
# Length units are arbitrary
f = 1
d = 0.2*f
# Type of lens is determined here
zetam = -10000*f
zetap = f

# Set up the 2D phase array
Nr = 400
Nz = 400
psi = numpy.zeros((Nr+2,Nz+2))

# Limits of the computational box
Rbox = 0.75*f
Lbox = 1.5*f

# Set up the mesh points with spacing (dr,dz)
dr = Rbox/Nr
dz = Lbox/Nz
r_pts = numpy.linspace(-dr/2,Rbox+dr/2,Nr+2)
z_pts = numpy.linspace(-Lbox/2-dz/2,Lbox/2+dz/2,Nz+2)

# Compute the phase in length units
# psi = z' = psi[radians] * c / omega
for i,r in enumerate(r_pts):
	for k,z in enumerate(z_pts):
		g = lambda x : 0.5 + 0.5*numpy.tanh(2*x)
		Rm = numpy.sign(zetam)*numpy.sqrt(r**2+(z-zetam)**2)
		Rp = numpy.sign(zetap)*numpy.sqrt(r**2+(z-zetap)**2)
		R = lambda x : 1/((1-g(x/d))/Rm + g(x/d)/Rp)
		merit = lambda x : r**2 + (z-x-R(x))**2 - R(x)**2
		try:
			psi[i,k] = scipy.optimize.brentq(merit,-f,f,tol=1e-11)
		except:
			psi[i,k] = scipy.optimize.newton(merit,z,tol=1e-8)

grad = numpy.gradient(psi,dr,dz)
nbar = 1.0 - (grad[0]**2 + grad[1]**2)

# Since wave is spherical everywhere, we can clip the plasma
# inside any sphere centered on the focal point.
# r2d = numpy.outer(r_pts,numpy.ones(Nz+2))
# z2d = numpy.outer(numpy.ones(Nr+2),z_pts-zetap)
# r2 = r2d**2 + z2d**2
# nbar[numpy.where(r2<f/4)]=1e-10
# Throw away guard cells
nbar = nbar[1:-1,1:-1]
if numpy.min(nbar)<0.0:
	nbar -= numpy.min(nbar)*1.000001

plt.figure(1,dpi=300)
plt.imshow(numpy.log10(nbar),origin='lower',cmap=my_color_map,extent=[-Lbox/2,Lbox/2,0,Rbox])
b=plt.colorbar(orientation='horizontal')
b.set_label(r'$n_e/n_c$',size=18)
plt.xlabel(r'$z/f$',size=18)
plt.ylabel(r'$\rho/f$',size=18)
plt.tight_layout()
plt.savefig('test.png')
numpy.save('ideal-form',nbar)

# plt.figure(2,dpi=300)
# plt.imshow(psi,origin='lower',cmap=my_color_map,extent=[-zmax,zmax,0,rmax])
# b=plt.colorbar()
# b.set_label(r'$c\psi/\omega f$',size=18)
# plt.xlabel(r'$z/f$',size=18)
# plt.ylabel(r'$\rho/f$',size=18)
# plt.tight_layout()

plt.show()
