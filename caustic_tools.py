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

	def GetBoundaryFields(self,center,xps,eiks,c):
		'''Get eikonal field component c on a flat surface.
		Surface is defined by the rays (xps,eiks).  Rays must be propagated to the surface externally.'''
		xp0 = xps[:,1:4] - center
		kr = eiks[:,0] + xps[:,4]*xps[:,0]
		clip_rect = [-self.size[0]/2,self.size[0]/2,-self.size[1]/2,self.size[1]/2]
		A,plot_ext = grid_tools.GridFromInterpolation(xp0[:,0],xp0[:,1],eiks[:,c],(self.pts[0],self.pts[1]),clip_rgn=clip_rect)
		psi,plot_ext = grid_tools.GridFromInterpolation(xp0[:,0],xp0[:,1],kr,(self.pts[0],self.pts[1]),clip_rgn=clip_rect)
		return plot_ext , A*np.exp(1j*psi)

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
		dom3d = np.array([-self.size[0]/2,self.size[0]/2,-self.size[1]/2,self.size[1]/2,-self.size[2]/2,self.size[2]/2])
		return dom3d , np.fft.ifft(np.fft.ifft(Ak,axis=0),axis=1)

class SphericalHarmonicTool:

	def __init__(self,m,q_pts_in,q_pts_out,modes,r_pts,queue,kernel):
		'''m = azimuthal mode number
		q_pts_in = number of samples along polar angle in the input data
		q_pts_out = number of samples along polar angle to produce in the output data
		modes = number polar modes (Legendre polynomials)
		r_pts = number of radial points to produce in the output data
		queue = OpenCL queue
		kernel = OpenCL kernel function to perform matrix multiplication'''
		q_list_in = grid_tools.cell_centers(0,np.pi,q_pts_in)
		q_list_out = grid_tools.cell_centers(0,np.pi,q_pts_out)
		mode_list = np.array(range(modes))
		dq = q_list_in[1] - q_list_in[0]
		T_pts = (q_pts_in,modes,q_pts_out)
		self.m = m
		self.T_pts = T_pts
		self.q_pts = (q_pts_in,q_pts_out)
		self.r_pts = r_pts
		self.modes = modes
		self.T = np.zeros((T_pts[1],T_pts[0])).astype(np.complex)
		self.Ti = np.zeros((T_pts[2],T_pts[1])).astype(np.complex)
		self.v = (np.zeros((T_pts[0],r_pts)).astype(np.complex) , np.zeros((T_pts[1],r_pts)).astype(np.complex) , np.zeros((T_pts[2],r_pts)).astype(np.complex))

		# Set up forward transform
		for i in range(q_pts_in):
			self.T[:,i] = 2*np.pi*dq*np.sin(q_list_in[i])*sph_harm(m,mode_list,0.0,q_list_in[i])

		# Set up reverse transform
		for i in range(modes):
			self.Ti[:,i] = sph_harm(m,i,0.0,q_list_out)

		# Set up OpenCL for fast transforms
		self.queue = queue
		self.kernel = kernel
		self.T_dev = pyopencl.array.to_device(queue,self.T)
		self.Ti_dev = pyopencl.array.to_device(queue,self.Ti)
		self.v_dev = (pyopencl.array.to_device(queue,self.v[0]),pyopencl.array.to_device(queue,self.v[1]),pyopencl.array.to_device(queue,self.v[2]))
		queue.finish()

	def Transform(self,T,v,inv):
		ans = np.zeros((self.T_pts[inv+1],self.r_pts)).astype(np.complex)
		self.v_dev[inv].set(v)
		self.kernel(	self.queue,
						(self.T_pts[inv+1],self.r_pts),
						None,
						T,
						self.v_dev[inv].data,
						self.v_dev[inv+1].data,
						np.int32(self.T_pts[inv]))
		self.queue.finish()
		self.v_dev[inv+1].get(ary=ans)
		return ans

	def GetGaussianFields(self,E0_mks,r00,f,k00,rs,rd):
		q_list_in = grid_tools.cell_centers(0,np.pi,self.q_pts[0])
		# sgn goes from -1 to 1 as z goes from + to -
		sgn = np.tanh((q_list_in-np.pi/2)/(np.pi/128))
		qh = q_list_in/2
		# Amplitude of x-component of incoming wave with m=0
		# Neglecting m>0 modes means the incident wave is elliptical
		ax = 0j+(E0_mks*f/rs)*np.exp(-4*f**2*np.tan(qh)**2/r00**2)/np.cos(qh)**2
		ax += ax[::-1]
		ax *= np.exp(1j*sgn*np.pi/4)
		# k00*rs is eliminated from the phase by rounding to the nearest multiple of 2*pi + pi/4
		# Phases that give pure real or imaginary values give underdetermined results, hence the pi/4
		az = -ax*np.tan(q_list_in)
		return ax,az

	def GetBoundaryFields(self,center,xps,eiks,max_modes):
		# Get the polar function defined by Integrate(dphi*exp(i*m*phi)*A(theta,phi))
		# i.e., extract the particular azimuthal mode considered
		xp0 = xps[:,1:4] - center
		phase = eiks[:,0] + xps[:,4]*xps[:,0]
		theta = np.arccos(xp0[:,2]/np.sqrt(np.einsum('ij,ij->i',xp0,xp0)))
		phi = np.arctan2(xp0[:,1],xp0[:,0])
		ax,plot_ext = grid_tools.GridFromBinning(theta,phi,eiks[:,1],(self.q_pts[0],max_modes),clip_rgn=[0,np.pi,-np.pi-np.pi/max_modes,np.pi-np.pi/max_modes])
		psi,plot_ext = grid_tools.GridFromBinning(theta,phi,phase,(self.q_pts[0],max_modes),clip_rgn=[0,np.pi,-np.pi-np.pi/max_modes,np.pi-np.pi/max_modes])
		q = np.arange(0,max_modes)
		f = 2*np.pi*np.exp(-2j*np.pi*self.m*q/max_modes)/max_modes
		return grid_tools.cell_centers(0,np.pi,self.q_pts[0]),np.sum(ax*np.exp(1j*psi)*f,axis=1)

	def GetFields(self,k00,rs,r1,r2,A):
		# A is expected to be a 1D array representing a function of polar angle
		# which is known at r=rs
		if rs<r1 or r2<r1 or rs<r2:
			print('WARNING: r positions out of order in GetFields')
		q_list_out = grid_tools.cell_centers(0,np.pi,self.q_pts[1])
		r_list = grid_tools.cell_centers(r1,r2,self.r_pts)

		# Get the polar angle mode coefficients
		# This is done for each r, for programming convenience
		# It is r-pts fold redundant
		A_l = self.Transform(self.T_dev.data,np.outer(A,np.ones(self.r_pts)),0)
		# Get the modes at any r (they are all the same) for diagnostic return values
		A_modes = np.array(A_l[:,0])
		# Recover fields in r-theta plane
		# Asymptotic jn is used on boundary sphere to minimize numerical errors
		for l in range(self.modes):
			A_l[l,:] *= spherical_jn(l,k00*r_list)*k00*rs/np.cos(np.pi/4-(l+1)*np.pi/2)
		A = self.Transform(self.Ti_dev.data,A_l,1)

		return q_list_out,r_list,A,A_modes

class BesselBeamTool:

	def __init__(self,pts,size,queue,kernel):
		self.pts = pts
		self.size = size
		self.mmax = np.int(pts[1]/2)
		self.H = grid_tools.HankelTransformTool(self.pts[0],self.size[0]/self.pts[0],self.mmax,queue,kernel)

	def GetBoundaryFields(self,center,xps,eiks,c):
		'''Eikonal field component c on (rho,phi) grid.
		The radial points do not include the origin.
		The azimuthal points are [-pi,...,pi-2*pi/N].'''
		xp0 = xps[:,1:4] - center
		kr = eiks[:,0] + xps[:,4]*xps[:,0]
		rho = np.sqrt(xp0[:,0]**2 + xp0[:,1]**2)
		phi = np.arctan2(xp0[:,1],xp0[:,0])
		# Remove ambiguity in angle by changing any occurence of pi to -pi
		phi[np.where(phi>=0.99999*np.pi)] -= 2*np.pi
		rho_nodes = grid_tools.cell_centers(0.0,self.size[0],self.pts[0])
		clip_rect = [rho_nodes[0],rho_nodes[-1],-np.pi,np.pi-2*np.pi/self.pts[1]]
		grid_pts = (self.pts[0],self.pts[1])
		A,plot_ext = grid_tools.GridFromInterpolation(rho,1.0001*phi,eiks[:,c],grid_pts,clip_rgn=clip_rect)
		psi,plot_ext = grid_tools.GridFromInterpolation(rho,1.0001*phi,kr,grid_pts,clip_rgn=clip_rect)
		return plot_ext , A*np.exp(1j*psi)

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
		dom3d = np.array([0.0,self.size[0],-np.pi,np.pi-2*np.pi/self.pts[1],-self.size[2]/2,self.size[2]/2])
		return dom3d , ans


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
