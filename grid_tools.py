import numpy as np
import scipy.interpolate
from scipy.linalg import eig_banded
import pyopencl
import pyopencl.array

# Notion of a cell is that data is known at the cell center.
# The set of cell centers are the nodes.
# The nodes are separated by cell walls.
# Voxels used in plotting can be considered visualizations of a cell.

def cell_centers(wall_pos_low,wall_pos_high,cells_between):
	'''Generate nodes between two cell wall positions.
	Number of cells between the two walls must be given.'''
	dx = (wall_pos_high - wall_pos_low)/cells_between
	return np.linspace(wall_pos_low+0.5*dx,wall_pos_high-0.5*dx,cells_between)

def cell_walls(node_low,node_high,nodes_between):
	'''Generate cell walls bracketing two nodes.
	Number of nodes between the walls must be given.'''
	dx = (node_high - node_low)/(nodes_between-1)
	return np.linspace(node_low-0.5*dx,node_high+0.5*dx,nodes_between+1)

def hypersurface_idx(ary,axis,cell):
	# Form tuple that indexes a hypersurface
	idx_list = [Ellipsis,]*len(ary.shape)
	idx_list[axis] = cell
	return tuple(idx_list)

def Smooth1D(dist,passes,ax=0):
	tup_low_ghost = hypersurface_idx(dist,ax,0)
	tup_low = hypersurface_idx(dist,ax,1)
	tup_high_ghost = hypersurface_idx(dist,ax,dist.shape[ax]-1)
	tup_high = hypersurface_idx(dist,ax,dist.shape[ax]-2)
	# Carry out smoothing passes
	for i in range(passes):
		dist_above = np.roll(dist,-1,axis=ax)
		dist_below = np.roll(dist,1,axis=ax)
		dist_above[tup_high_ghost] = dist_above[tup_high]
		dist_below[tup_low_ghost] = dist_below[tup_low]
		dist = 0.25*dist_above + 0.25*dist_below + 0.5*dist
	return dist

# Both interpolation and binning grids return array in c-order
# That is, y changes faster in memory, per z[x][y]
# The plot extent that is returned has [xmin,xmax,ymin,ymax]
# When interfacing with matplotlib imshow routine, the array axes should be swapped.
# This will result in the correct labeling of the axes.
# imshow(z,extent=[x0,x1,y0,y1]) gives the wrong labelling.
# imshow(z.swapaxes(0,1),extent=[x0,x1,y0,y1]) gives the right labelling.

# To avoid confusion please swap the axes only within the imshow function call.

def PulseFromInterpolation(w,x,y,A,res,clip_rgn=0,fill=0.0):
	'''Produce a pulse A(w,x,y) using ray data in a plane.
	res is a tuple with the number of cells (Nw,Nx,Ny)
	clip_rgn bounds the cells (cf. GridFromInterpolation)'''
	if clip_rgn==0:
		# If this is invoked then the extreme marginal data is put exactly on the domain boundary
		clip_rgn=[np.min(w),np.max(w),np.min(x),np.max(x),np.min(y),np.max(y)]
	# Interpolate data on the nodes
	pts = np.zeros((len(w),3))
	pts[:,0] = w
	pts[:,1] = x
	pts[:,2] = y
	wg1 = cell_centers(clip_rgn[0],clip_rgn[1],res[0])
	xg1 = cell_centers(clip_rgn[2],clip_rgn[3],res[1])
	yg1 = cell_centers(clip_rgn[4],clip_rgn[5],res[2])
	wg = np.einsum('i,j,k',wg1,np.ones(res[1]),np.ones(res[2]))
	xg = np.einsum('i,j,k',np.ones(res[0]),xg1,np.ones(res[2]))
	yg = np.einsum('i,j,k',np.ones(res[0]),np.ones(res[1]),yg1)
	the_data = scipy.interpolate.griddata(pts,A,(wg,xg,yg),fill_value=fill)
	return the_data

def CylGridFromInterpolation(x,y,z,xn=100,yn=100,fill=0.0):
	'''Special treatment for points that tend to lie on a cylindrical nodes.
	x,y,z are the data points.
	xn,yn are arrays of nodes, or numbers of nodes.
	If numbers are given, the node boundaries are obtained from the points.'''
	if type(xn) is int:
		xn = np.linspace(np.min(x),np.max(x),xn)
	if type(yn) is int:
		yn = np.linspace(np.min(y),np.max(y),yn)
	# Return the boundaries of the plotting voxels for convenience
	wx = cell_walls(xn[0],xn[-1],xn.shape[0])
	wy = cell_walls(yn[0],yn[-1],yn.shape[0])
	plot_ext = np.array([wx[0],wx[-1],wy[0],wy[-1]])
	the_data = np.zeros((xn.shape[0],yn.shape[0]))
	for j in range(yn.shape[0]):
		sel = np.where(np.logical_and(y>=wy[j],y<wy[j+1]))
		idx = np.argmin(x[sel])
		z0 = z[sel][idx]
		f = scipy.interpolate.interp1d(x[sel],z[sel],kind='linear',bounds_error=False,fill_value=(z0,0.0))
		the_data[:,j] = f(xn)
	return the_data,plot_ext

def GridFromInterpolation(x,y,z,xn=100,yn=100,fill=0.0):
	'''x,y,z are the data points.
	xn,yn are arrays of nodes, or numbers of nodes.
	If numbers are given, the node boundaries are obtained from the points.'''
	if type(xn) is int:
		xn = np.linspace(np.min(x),np.max(x),xn)
	if type(yn) is int:
		yn = np.linspace(np.min(y),np.max(y),yn)
	# Return the boundaries of the plotting voxels for convenience
	wx = cell_walls(xn[0],xn[-1],xn.shape[0])
	wy = cell_walls(yn[0],yn[-1],yn.shape[0])
	plot_ext = np.array([wx[0],wx[-1],wy[0],wy[-1]])
	pts = np.zeros((len(x),2))
	pts[:,0] = x
	pts[:,1] = y
	xg = 0.9999*np.outer(xn,np.ones(yn.shape[0]))
	yg = 0.9999*np.outer(np.ones(xn.shape[0]),yn)
	the_data = scipy.interpolate.griddata(pts,z,(xg,yg),method='cubic',fill_value=fill)
	return the_data,plot_ext

def GridFromBinning(x,y,z,xn=100,yn=100):
	'''x,y,z are the data points.
	xn,yn are arrays of nodes, or numbers of nodes.
	If numbers are given, the node boundaries are obtained from the points.'''
	if type(xn) is int:
		xn = np.linspace(np.min(x),np.max(x),xn)
	if type(yn) is int:
		yn = np.linspace(np.min(y),np.max(y),yn)
	# Return the boundaries of the plotting voxels for convenience
	# The range for the histogram is the same, but packed differently
	wx = cell_walls(xn[0],xn[-1],xn.shape[0])
	wy = cell_walls(yn[0],yn[-1],yn.shape[0])
	plot_ext = np.array([wx[0],wx[-1],wy[0],wy[-1]])
	hist_range = [[wx[0],wx[-1]],[wy[0],wy[-1]]]
	harray,hpts,vpts = np.histogram2d(x,y,weights=z,bins=(xn.shape[0],yn.shape[0]),range=hist_range)
	harray_count,hpts,vpts = np.histogram2d(x,y,bins=(xn.shape[0],yn.shape[0]),range=hist_range)
	harray /= harray_count+.0001
	return harray,plot_ext

class HankelTransformTool:
	'''Transform Cartesian components to Bessel beam basis.'''
	def __init__(self,Nr,dr,mmax,queue,kernel):
		r_list = cell_centers(0.0,Nr*dr,Nr)
		A1 = 2*np.pi*(r_list-0.5*dr)
		A2 = 2*np.pi*(r_list+0.5*dr)
		V = np.pi*((r_list+0.5*dr)**2 - (r_list-0.5*dr)**2)
		self.Lambda = np.sqrt(V)
		# Highest negative mode is never needed due to symmetry, so we have 2*mmax modes instead of 2*mmax+1.
		self.vals = np.zeros((Nr,2*mmax))
		# Save storage by only keeping eigenvectors for positive modes (negative modes are the same)
		self.Hi = np.zeros((Nr,Nr,mmax+1))
		for m in range(0,mmax+1):
			T1 = A1/(dr*V)
			T2 = -(A1 + A2)/(dr*V) - (m/r_list)**2
			T3 = A2/(dr*V)
			# Boundary conditions
			T2[0] += T1[0]
			T2[Nr-1] -= T3[Nr-1]
			# Symmetrize the matrix
			# S = Lambda * T * Lambda^-1
			# This is the root-volume weighting
			T1[1:] *= self.Lambda[1:]/self.Lambda[:-1]
			T3[:-1] *= self.Lambda[:-1]/self.Lambda[1:]
			a_band_upper = np.zeros((2,Nr))
			a_band_upper[0,:] = T1 # T3->T1 thanks to scipy packing and symmetry
			a_band_upper[1,:] = T2
			self.vals[:,m],self.Hi[:,:,m] = eig_banded(a_band_upper)
		# Set eigenvalues of negative modes (they are the same as corresponding positive modes)
		# Modes are packed in usual FFT fashion, so negative indices work as expected.
		for m in range(1,mmax):
			self.vals[:,-m] = self.vals[:,m]
		self.queue = queue
		self.kernel = kernel
	def CLeinsum(self,T,v):
		Tc = np.ascontiguousarray(np.copy(T))
		vc = np.ascontiguousarray(np.copy(v))
		T_dev = pyopencl.array.to_device(self.queue,Tc)
		vin_dev = pyopencl.array.to_device(self.queue,vc)
		vout_dev = pyopencl.array.to_device(self.queue,vc)
		self.kernel(	self.queue,
						vout_dev.shape,
						None,
						T_dev.data,
						vin_dev.data,
						vout_dev.data)
		self.queue.finish()
		vout_dev.get(ary=vc)
		return vc
	def Transform(self,T,v):
		mmax = np.int(v.shape[1]/2)
		v = np.einsum('i,i...->i...',self.Lambda,v)
		v[:,:mmax+1,...] = self.CLeinsum(T,v[:,:mmax+1,...])
		v[:,-1:-mmax:-1,...] = self.CLeinsum(T[:,:,1:mmax],v[:,-1:-mmax:-1,...])
		# v[:,:mmax+1,...] = np.einsum('ijk...,jk...->ik...',T,v[:,:mmax+1,...])
		# v[:,-1:-mmax:-1,...] = np.einsum('ijk...,jk...->ik...',T[:,:,1:mmax],v[:,-1:-mmax:-1,...])
		v = np.einsum('i,i...->i...',1/self.Lambda,v)
		return v
	def kspace(self,a):
		return self.Transform(self.Hi.swapaxes(0,1),a)
	def rspace(self,a):
		return self.Transform(self.Hi,a)
	def kr2(self):
		return -self.vals
