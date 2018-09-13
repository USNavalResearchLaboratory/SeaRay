import numpy as np
from scipy.special import jn
from scipy.special import spherical_jn
from scipy.special import struve
from scipy.special import sph_harm
import scipy.interpolate
import pyopencl
import pyopencl.array
import grid_tools

def ParaxialFocus(xps,vg):
	# Return distance to paraxial focus
	# Does not deal with astigmatism
	theta = np.sqrt(vg[:,1]**2+vg[:,2]**2)/vg[:,3]
	sel = np.where(theta**2>1e-8)
	if sel[0].shape[0]>0:
		vg1 = vg[np.where(theta**2>1e-8)]
		xps1 = xps[np.where(theta**2>1e-8)]
		vrho = np.sqrt(vg1[:,1]**2+vg1[:,2]**2)
		rho = np.sqrt(xps1[:,1]**2 + xps1[:,2]**2)
		idx = np.argmin(rho)
		zf = -np.sign(vg1[idx,1]*xps1[idx,1])*rho[idx]*vg1[idx,3]/vrho[idx]
	else:
		zf = 0.0
	return zf

class FourierTool:

	def __init__(self,pts,size):
		'''pts = tuple with (Nx,Ny,Nz)
		size = tuple with (Lx,Ly,Lz)'''
		self.pts = pts
		self.size = size
		self.dx = (size[0]/pts[0],size[1]/pts[1],size[2]/pts[2])

	def FFT_w(self,dti,num):
		# Square of this is the proper eigenvalue of the FD laplacian
		# For small frequencies it corresponds to the wave frequency
		i_list = np.arange(0,num)
		sgn = np.ones(num)
		sgn[np.int(num/2)+1:] = -1.0
		return sgn*np.sqrt(2.0*dti*dti*(1.0 - np.cos(2.0*np.pi*i_list/num)));

	def GetGridInfo(self):
		x_nodes = grid_tools.cell_centers(-self.size[0]/2,self.size[0]/2,self.pts[0])
		y_nodes = grid_tools.cell_centers(-self.size[1]/2,self.size[1]/2,self.pts[1])
		x_walls = grid_tools.cell_walls(x_nodes[0],x_nodes[-1],self.pts[0])
		y_walls = grid_tools.cell_walls(y_nodes[0],y_nodes[-1],self.pts[1])
		plot_ext = np.array([x_walls[0],x_walls[-1],y_walls[0],y_walls[-1]])
		return x_nodes,y_nodes,plot_ext

	def GetBoundaryFields(self,center,xps,eiks,c):
		'''Get eikonal field component c on a flat surface.
		Surface is defined by the rays (xps,eiks).  Rays must be propagated to the surface externally.'''
		xp0 = xps[:,1:4] - center
		kr = eiks[:,0] + xps[:,4]*xps[:,0]
		x_nodes,y_nodes,plot_ext = self.GetGridInfo()
		A,ignore = grid_tools.GridFromInterpolation(xp0[:,0],xp0[:,1],eiks[:,c],x_nodes,y_nodes)
		psi,ignore = grid_tools.GridFromInterpolation(xp0[:,0],xp0[:,1],kr,x_nodes,y_nodes)
		return A*np.exp(1j*psi) , plot_ext

	def GetFields(self,k00,dz,A):
		'''dz = distance from eikonal plane to center of interrogation region
		A = complex amplitude (any Cartesian component) in eikonal plane'''
		z_list = grid_tools.cell_centers(dz-self.size[2]/2,dz+self.size[2]/2,self.pts[2])
		Ak = np.fft.fft(np.fft.fft(A,axis=0),axis=1)
		kx_list = self.FFT_w(1/self.dx[0],self.pts[0])
		ky_list = self.FFT_w(1/self.dx[1],self.pts[1])
		k_perp_2 = np.outer(kx_list**2,np.ones(self.pts[1])) + np.outer(np.ones(self.pts[0]),ky_list**2)
		kz = np.sqrt(0j + k00**2 - k_perp_2)
		phase_adv = np.einsum('ij,k',kz,z_list)
		#for i in range(self.pts[2]):
		#	phase_adv[...,i] -= np.min(phase_adv[...,i])
		Ak = np.einsum('ij,ijk->ijk',Ak,np.exp(1j*phase_adv))
		x_nodes,y_nodes,plot_ext = self.GetGridInfo()
		z_ext = np.array([-self.size[2]/2,self.size[2]/2])
		dom3d = np.concatenate((plot_ext,z_ext))
		return np.fft.ifft(np.fft.ifft(Ak,axis=0),axis=1) , dom3d

class BesselBeamTool:

	def __init__(self,pts,size,queue,kernel):
		self.pts = pts
		self.size = size
		self.mmax = np.int(pts[1]/2)
		self.H = grid_tools.HankelTransformTool(self.pts[0],self.size[0]/self.pts[0],self.mmax,queue,kernel)

	def GetGridInfo(self):
		rho_nodes = grid_tools.cell_centers(0.0,self.size[0],self.pts[0])
		phi_nodes = np.linspace(-np.pi,np.pi-2*np.pi/self.pts[1],self.pts[1])
		rho_walls = grid_tools.cell_walls(rho_nodes[0],rho_nodes[-1],self.pts[0])
		phi_walls = grid_tools.cell_walls(phi_nodes[0],phi_nodes[-1],self.pts[1])
		plot_ext = np.array([rho_walls[0],rho_walls[-1],phi_walls[0],phi_walls[-1]])
		return rho_nodes,phi_nodes,plot_ext

	def GetBoundaryFields(self,center,xps,eiks,c):
		'''Eikonal field component c on (rho,phi) grid.
		The radial points do not include the origin.
		The azimuthal points are [-pi,...,pi-2*pi/N].'''
		xp0 = xps[:,1:4] - center
		kr = eiks[:,0] + xps[:,4]*xps[:,0]
		rho = np.sqrt(xp0[:,0]**2 + xp0[:,1]**2)
		phi = np.arctan2(xp0[:,1],xp0[:,0])
		# Keep angles in range pi to -pi, and favor -pi over +pi
		phi[np.where(phi>0.9999*np.pi)] -= 2*np.pi
		rho_nodes,phi_nodes,plot_ext = self.GetGridInfo()
		A,ignore = grid_tools.CylGridFromInterpolation(rho,phi,eiks[:,c],rho_nodes,phi_nodes)
		psi,ignore = grid_tools.CylGridFromInterpolation(rho,phi,kr,rho_nodes,phi_nodes)
		return A*np.exp(1j*psi),plot_ext

	def GetFields(self,k00,dz,A):
		'''dz = distance from eikonal plane to center of interrogation region
		A[rho,phi] = complex amplitude (any Cartesian component) in eikonal plane
		mmax = highest azimuthal mode to keep [0,1,2,...]'''
		z_list = grid_tools.cell_centers(dz-self.size[2]/2,dz+self.size[2]/2,self.pts[2])
		ans = np.einsum('ij,k->ijk',A,np.ones(self.pts[2]))
		ans = np.fft.fft(ans,axis=1)
		ans = self.H.kspace(ans)
		kz = np.sqrt(0j + k00**2 - self.H.kr2())
		phase_adv = np.einsum('ij,k',kz,z_list)
		phase_adv[np.where(np.imag(phase_adv)<0)] *= -1
		ans *= np.exp(1j*phase_adv)
		ans = self.H.rspace(ans)
		ans = np.fft.ifft(ans,axis=1)
		rho_nodes,phi_nodes,plot_ext = self.GetGridInfo()
		z_ext = np.array([-self.size[2]/2,self.size[2]/2])
		dom3d = np.concatenate((plot_ext,z_ext))
		return ans,dom3d


def get_waist(rho_list,intensity,which_axis):
	rms_size = np.sqrt(np.sum(intensity*rho_list**2,axis=which_axis)/np.sum(intensity,axis=which_axis))
	idx = np.argmin(rms_size)
	return rms_size[idx]

def spherical_to_cylindrical(A,q_list,r_list,rho_pts,z_pts):
	rmax = r_list[-1] + 0.5*(r_list[1]-r_list[0])
	B = np.zeros((rho_pts,z_pts)).astype(np.complex)
	rho_list = grid_tools.cell_centers(0,rmax,rho_pts)
	z_list = grid_tools.cell_centers(-rmax,rmax,z_pts)
	rho = np.outer(rho_list,np.ones(z_pts))
	z = np.outer(np.ones(rho_pts),z_list)
	fr = scipy.interpolate.RectBivariateSpline(q_list,r_list,np.real(A),kx=3,ky=3)
	fi = scipy.interpolate.RectBivariateSpline(q_list,r_list,np.imag(A),kx=3,ky=3)
	r = np.sqrt(rho**2 + z**2)
	theta = np.arccos(z/r)
	B = fr.ev(theta,r) + 1j*fi.ev(theta,r)
	clip_rgn = np.where(rho**2+z**2 > rmax**2)
	B[clip_rgn] = 0
	return rho_list,z_list,B

def SphericalClipping(A,dr,dz,cells):
	rho = np.outer(grid_tools.cell_centers(0,dr*A.shape[0],A.shape[0]),np.ones(A.shape[1]))
	z = np.outer(np.ones(A.shape[0]),grid_tools.cell_centers(-dz*A.shape[1]/2,dz*A.shape[1]/2,A.shape[1]))
	clip_rgn = np.where(rho**2+z**2 > (dr*(A.shape[0]-cells))**2)
	A[clip_rgn] = 0.0

def AxisClipping(A):
	# 3rd order splines are destructive to data in first 3 cells along rho axis
	A[:3,:] = 0

def GetDivergence(Ax,Az,dr,dz,queue,kernel):
	ans = np.zeros(Ax.shape).astype(np.complex)
	Ax_dev = pyopencl.array.to_device(queue,Ax)
	Az_dev = pyopencl.array.to_device(queue,Az)
	ans_dev = pyopencl.array.to_device(queue,ans)
	queue.finish()
	kernel(queue,(Ax.shape[0]-2,Ax.shape[1]-2),None,Ax_dev.data,Az_dev.data,ans_dev.data,dr,dz,global_offset=(1,1))
	queue.finish()
	ans_dev.get(ary=ans)
	return ans

def GetLaplacian(A,dr,dz,m,queue,kernel):
	ans = np.zeros(A.shape).astype(np.complex)
	A_dev = pyopencl.array.to_device(queue,A)
	ans_dev = pyopencl.array.to_device(queue,ans)
	queue.finish()
	kernel(queue,(A.shape[0]-2,A.shape[1]-2),None,A_dev.data,ans_dev.data,dr,dz,m,global_offset=(1,1))
	queue.finish()
	ans_dev.get(ary=ans)
	return ans
