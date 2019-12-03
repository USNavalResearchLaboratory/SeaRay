import numpy as np
import scipy.optimize
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

class CausticTool:

	def __init__(self,pts,band,center,size,cl):
		'''pts = tuple with (Nw,Nx,Ny,Nz)
		band = tuple with (low_frequency,high_frequency)
		center = tuple with (x,y,z)
		size = tuple with (Lx,Ly,Lz)
		cl = OpenCL reference object'''
		self.pts = pts
		w0,dw = grid_tools.cyclic_center_and_width(band[0],band[1])
		self.center = np.array(list((w0,) + center))
		self.size = np.array(list((dw,) + size))
		self.dx = self.size/np.array(list(pts))
		self.default_band = size[0]
		self.cl = cl
		# Derived classes must create a transverse transform tool T
	def GetTransverseTool(self):
		return self.T
	def GetFields(self,dz,A):
		'''Compute Maxwell fields in a sequence of planes in any geometry.
		The geometry is encapsulated in the TransverseModeTool self.T.
		dz = distance from eikonal plane to center of interrogation region
		A = complex amplitude (any Cartesian component) in eikonal plane'''
		w_nodes,x1_nodes,x2_nodes,plot_ext = self.GetGridInfo()
		z_nodes = grid_tools.cell_centers(dz-self.size[3]/2,dz+self.size[3]/2,self.pts[3])
		ans = np.einsum('ijk,l->ijkl',A,np.ones(self.pts[3]))
		ans = self.T.kspace(ans)
		phase_adv = np.ones(ans.shape).astype(np.complex) * z_nodes[np.newaxis,np.newaxis,np.newaxis,:]
		kz = np.sqrt(0j + np.zeros(ans.shape[:3]) + w_nodes[...,np.newaxis,np.newaxis]**2 - self.T.kr2()[np.newaxis,...])
		phase_adv *= kz[...,np.newaxis]
		# Following applies galilean transformation from z,t to z,t-z/c
		phase_adv -= np.outer(w_nodes,z_nodes)[:,np.newaxis,np.newaxis,:]
		phase_adv[np.where(np.imag(phase_adv)<0)] *= -1
		ans *= np.exp(1j*phase_adv)
		ans = self.T.rspace(ans)
		z_ext = np.array([-self.size[3]/2,self.size[3]/2])
		dom4d = np.concatenate((plot_ext,z_ext))
		return ans,dom4d
	def kspace(self,A):
		return self.T.kspace(A)
	def rspace(self,A):
		return self.T.rspace(A)
	def kr2(self):
		return self.T.kr2()

class FourierTool(CausticTool):

	def __init__(self,pts,band,center,size,cl):
		super().__init__(pts,band,center,size,cl)
		self.T = grid_tools.FourierTransformTool(self.pts,self.dx,cl)

	def GetGridInfo(self):
		w_nodes = self.center[0] + grid_tools.cyclic_nodes(-self.size[0]/2,self.size[0]/2,self.pts[0])
		x_nodes = grid_tools.cell_centers(-self.size[1]/2,self.size[1]/2,self.pts[1])
		y_nodes = grid_tools.cell_centers(-self.size[2]/2,self.size[2]/2,self.pts[2])
		w_walls = grid_tools.cell_walls(w_nodes[0],w_nodes[-1],self.pts[0],self.default_band)
		x_walls = grid_tools.cell_walls(x_nodes[0],x_nodes[-1],self.pts[1])
		y_walls = grid_tools.cell_walls(y_nodes[0],y_nodes[-1],self.pts[2])
		plot_ext = np.array([w_walls[0],w_walls[-1],x_walls[0],x_walls[-1],y_walls[0],y_walls[-1]])
		return w_nodes,x_nodes,y_nodes,plot_ext

	def GetBoundaryFields(self,xps,eiks,c):
		'''Get eikonal field component c on a flat surface.
		Surface is defined by the rays (xps,eiks).  Rays must be propagated to the surface externally.
		Direction is not discriminated.

		:param numpy.ndarray xps: phase space coordinates of primary rays, shape (bundles,8)
		:param numpy.ndarray eiks: eikonal data for primary rays, shape (bundles,4)
		:param int c: field component to interrogate'''
		w = xps[:,4]
		dx = xps[:,1:4] - self.center[1:4]
		w_nodes,x_nodes,y_nodes,plot_ext = self.GetGridInfo()
		time_window = 2*np.pi*self.pts[0]/self.size[0]
		# Reference time to the largest amplitude ray and put origin at box center
		largest = np.argmax(np.einsum('...i,...i',eiks[:,1:3],eiks[:,1:3]))
		kr = eiks[:,0] - xps[:,4]*(xps[largest,0] + 0.5*time_window)
		# Perform interpolation
		A,ignore = grid_tools.GridFromInterpolation(w,dx[:,0],dx[:,1],eiks[:,c],w_nodes,x_nodes,y_nodes)
		psi,ignore = grid_tools.GridFromInterpolation(w,dx[:,0],dx[:,1],kr,w_nodes,x_nodes,y_nodes)
		return A*np.exp(1j*psi) , plot_ext

class BesselBeamTool(CausticTool):

	def __init__(self,pts,band,center,size,cl):
		'''center remains Cartesian.
		size also remains Cartesian, but only Lx and Lz are used.'''
		self.pts = pts
		w0,dw = grid_tools.cyclic_center_and_width(band[0],band[1])
		self.center = np.array(list((w0,) + center))
		self.size = np.array(list((dw,) + size))
		self.mmax = np.int(self.pts[2]/2)
		self.default_band = size[0]
		self.T = grid_tools.HankelTransformTool(self.pts[1],0.5*self.size[1]/self.pts[1],self.mmax,cl)

	def GetGridInfo(self):
		w_nodes = self.center[0] + grid_tools.cyclic_nodes(-self.size[0]/2,self.size[0]/2,self.pts[0])
		rho_nodes = grid_tools.cell_centers(0.0,self.size[1]/2,self.pts[1])
		phi_nodes = grid_tools.cyclic_nodes(-np.pi,np.pi,self.pts[2])
		w_walls = grid_tools.cell_walls(w_nodes[0],w_nodes[-1],self.pts[0],self.default_band)
		rho_walls = grid_tools.cell_walls(rho_nodes[0],rho_nodes[-1],self.pts[1])
		phi_walls = grid_tools.cell_walls(phi_nodes[0],phi_nodes[-1],self.pts[2])
		plot_ext = np.array([w_walls[0],w_walls[-1],rho_walls[0],rho_walls[-1],phi_walls[0],phi_walls[-1]])
		return w_nodes,rho_nodes,phi_nodes,plot_ext

	def GetBoundaryFields(self,xps,eiks,c):
		'''Eikonal field component c on (w,rho,phi) grid.
		The radial points do not include the origin.
		The azimuthal points are [-pi,...,pi-2*pi/N].'''
		w = xps[:,4]
		dx = xps[:,1:4] - self.center[1:4]
		w_nodes,rho_nodes,phi_nodes,plot_ext = self.GetGridInfo()
		time_window = 2*np.pi*self.pts[0]/self.size[0]
		# Reference time to the largest amplitude ray
		largest = np.argmax(np.einsum('...i,...i',eiks[:,1:3],eiks[:,1:3]))
		kr = eiks[:,0] - xps[:,4]*(xps[largest,0] + 0.5*time_window)
		# Perform interpolation
		rho = np.sqrt(dx[:,0]**2 + dx[:,1]**2)
		phi = np.arctan2(dx[:,1],dx[:,0])
		# Keep angles in range pi to -pi, and favor -pi over +pi
		phi[np.where(phi>0.9999*np.pi)] -= 2*np.pi
		A,ignore = grid_tools.CylGridFromInterpolation(w,rho,phi,eiks[:,c],w_nodes,rho_nodes,phi_nodes)
		psi,ignore = grid_tools.CylGridFromInterpolation(w,rho,phi,kr,w_nodes,rho_nodes,phi_nodes)
		return A*np.exp(1j*psi),plot_ext

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

def GetDivergence(Ax,Az,dr,dz,cl):
	ans = np.zeros(Ax.shape).astype(np.complex)
	Ax_dev = pyopencl.array.to_device(queue,Ax)
	Az_dev = pyopencl.array.to_device(queue,Az)
	ans_dev = pyopencl.array.to_device(queue,ans)
	queue.finish()
	cl.program('caustic').divergence(cl.q,(Ax.shape[0]-2,Ax.shape[1]-2),None,Ax_dev.data,Az_dev.data,ans_dev.data,dr,dz,global_offset=(1,1))
	queue.finish()
	ans_dev.get(ary=ans)
	return ans

def GetLaplacian(A,dr,dz,m,cl):
	ans = np.zeros(A.shape).astype(np.complex)
	A_dev = pyopencl.array.to_device(queue,A)
	ans_dev = pyopencl.array.to_device(queue,ans)
	queue.finish()
	cl.program('caustic').laplacian(cl.q,(A.shape[0]-2,A.shape[1]-2),None,A_dev.data,ans_dev.data,dr,dz,m,global_offset=(1,1))
	queue.finish()
	ans_dev.get(ary=ans)
	return ans
