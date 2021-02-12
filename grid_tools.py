'''
Module: :samp:`grid_tools`
--------------------------

Tools for transforming between structured and unstructured grids, and spectral analysis.
Ray data can be viewed as known on the nodes of an unstructured grid.
Our notion of a structured grid is that the data is known at the cell center.
The set of cell centers are the nodes.  The nodes are separated by cell walls.
Plotting voxels can be considered visualizations of a cell.
'''
import numpy as np
import scipy.interpolate
import scipy.linalg
import pyopencl
import pyopencl.array

def cyclic_center_and_width(node_low,node_high):
	'''Return the center node and wall-to-wall width of a cyclic set of nodes.
	Once cyclic nodes are generated the high node is lost.
	This function helps us remember not to use the generated nodes to find the center.
	See also cyclic_nodes(...) below.'''
	return 0.5*(node_low + node_high) , node_high - node_low

def cyclic_nodes(node_low,node_high,num_nodes):
	'''Generate nodes suitable for periodic functions.
	The requested low node is included, the high node is excluded.
	The low and high nodes should resolve to the same point, e.g. 0 and 2*pi.
	Frequency nodes are arranged in unrolled FFT fashion, e.g., [+-2,-1,0,1].'''
	if num_nodes==1:
		return np.array([0.5*node_low + 0.5*node_high])
	else:
		dx = (node_high - node_low)/num_nodes
		return np.linspace(node_low,node_high-dx,num_nodes)

def cell_centers(wall_pos_low,wall_pos_high,cells_between):
	'''Generate nodes between two cell wall positions.
	Number of nodes between the two walls must be given.'''
	dx = (wall_pos_high - wall_pos_low)/cells_between
	return np.linspace(wall_pos_low+0.5*dx,wall_pos_high-0.5*dx,cells_between)

def cell_walls(node_low,node_high,nodes_between,default_voxel_width=1000.0):
	'''Generate cell walls bracketing two nodes.
	Number of nodes between the walls must be given.'''
	if nodes_between==1:
		dx = default_voxel_width
	else:
		dx = (node_high - node_low)/(nodes_between-1)
	return np.linspace(node_low-0.5*dx,node_high+0.5*dx,nodes_between+1)

def hypersurface_idx(ary,axis,cell):
	# Form tuple that indexes a hypersurface
	idx_list = [slice(None),]*len(ary.shape)
	idx_list[axis] = (cell,)
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

def AddGhostCells(a,nodes,ax):
	new_cells = a[hypersurface_idx(a,ax,0)]
	new_nodes = [nodes[0] - (nodes[1]-nodes[0])]
	ex_nodes = np.concatenate((new_nodes,nodes))
	ex_a = np.concatenate((new_cells,a),axis=ax)
	return ex_a,ex_nodes

def DataFromGrid(w,x,y,wn,xn,yn,the_data):
	'''Form irregular data from a regular grid. OK if yn.shape[0]=1.
	w,x,y are the coordinates of the irregular data points.
	Irregular points can be slightly outside the grid (extrapolation is used).
	wn,xn,yn are arrays of nodes which must be regularly spaced.
	the_data contains values on the regular grid with shape (w,x,y).
	Any coordinate transformations must happen externally.'''
	if wn.shape[0]==1:
		raise ValueError('Cannot relaunch rays with monochromatic data.')
	if xn.shape[0]==1:
		raise ValueError('Missing spatial dimension x.')
	if yn.shape[0]==1:
		yng = np.linspace(np.min(y),np.max(y),2)
		ex_data = np.concatenate((the_data,the_data),axis=2)
	else:
		yng = yn
		ex_data = the_data
	obj = scipy.interpolate.RegularGridInterpolator((wn,xn,yng),ex_data,bounds_error=False,fill_value=None)
	pts = np.zeros(w.shape+(3,))
	pts[...,0] = w
	pts[...,1] = x
	pts[...,2] = y
	return obj(pts)

def CylGridFromInterpolation(w,x,y,vals,wn=1,xn=100,yn=100,fill=0.0):
	'''Special treatment for points that tend to lie on cylindrical nodes.
	Frequency and azimuth are binned, radial data is interpolated.
	Fill value for r<min(r) is val(min(r)).
	w,x,y,vals are the irregular data points, where x,y are interpreted as rho,phi.
	(ray coordinate system is transformed externally)
	wn,xn,yn are regular arrays of nodes, or numbers of nodes.
	If numbers are given, the node boundaries are obtained from the points.'''
	if type(wn) is int:
		wn = np.linspace(np.min(w),np.max(w),wn)
	if type(xn) is int:
		xn = np.linspace(np.min(x),np.max(x),xn)
	if type(yn) is int:
		yn = np.linspace(np.min(y),np.max(y),yn)
	# Return the boundaries of the plotting voxels for convenience
	ww = cell_walls(wn[0],wn[-1],wn.shape[0])
	wx = cell_walls(xn[0],xn[-1],xn.shape[0])
	wy = cell_walls(yn[0],yn[-1],yn.shape[0])
	plot_ext = np.array([ww[0],ww[-1],wx[0],wx[-1],wy[0],wy[-1]])
	the_data = np.zeros((wn.shape[0],xn.shape[0],yn.shape[0]))
	for i in range(wn.shape[0]):
		for j in range(yn.shape[0]):
			wcond = np.logical_and(w>=ww[i],w<ww[i+1])
			ycond = np.logical_and(y>=wy[j],y<wy[j+1])
			sel = np.where(np.logical_and(wcond,ycond))
			if len(x[sel])==0:
				the_data[i,:,j] = 0.0
			else:
				idx = np.argmin(x[sel])
				v0 = vals[sel][idx]
				f = scipy.interpolate.interp1d(x[sel],vals[sel],kind='linear',bounds_error=False,fill_value=(v0,0.0))
				the_data[i,:,j] = f(xn)
	return the_data,plot_ext

def GridFromInterpolation(w,x,y,vals,wn=1,xn=100,yn=100,fill=0.0):
	'''w,x,y,vals are the irregular data points.
	wn,xn,yn are regular arrays of nodes, or numbers of nodes.
	If numbers are given, the node boundaries are obtained from the points.'''
	if type(wn) is int:
		wn = np.linspace(np.min(w),np.max(w),wn)
	if type(xn) is int:
		xn = np.linspace(np.min(x),np.max(x),xn)
	if type(yn) is int:
		yn = np.linspace(np.min(y),np.max(y),yn)
	# Return the boundaries of the plotting voxels for convenience
	ww = cell_walls(wn[0],wn[-1],wn.shape[0])
	wx = cell_walls(xn[0],xn[-1],xn.shape[0])
	wy = cell_walls(yn[0],yn[-1],yn.shape[0])
	plot_ext = np.array([ww[0],ww[-1],wx[0],wx[-1],wy[0],wy[-1]])
	the_data = np.zeros((wn.shape[0],xn.shape[0],yn.shape[0]))
	xg = 0.9999*np.outer(xn,np.ones(yn.shape[0]))
	yg = 0.9999*np.outer(np.ones(xn.shape[0]),yn)
	for i in range(wn.shape[0]):
		wcond = np.logical_and(w>=ww[i],w<ww[i+1])
		sel = np.where(wcond)
		if len(x[sel])==0:
			the_data[i,...] = 0.0
		else:
			pts = np.zeros((len(x[sel]),2))
			pts[:,0] = x[sel]
			pts[:,1] = y[sel]
			the_data[i,...] = scipy.interpolate.griddata(pts,vals[sel],(xg,yg),method='cubic',fill_value=fill)
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

class TransverseModeTool:
	'''Base class for transformation of Cartesian components to another spatial basis

	:param 4-tuple N: nodes along each dimension (0 and 3 are not used)
	:param 4-tuple dq: node separation along each dimension (must be uniform)'''
	def __init__(self,N,dq,cl):
		self.N = N
		self.dq = dq
		self.cl = cl
	def AllocateDeviceMemory(self,data_shape):
		None
	def FreeDeviceMemory(self):
		None
	def kspace(self,a):
		return a
	def rspace(self,a):
		return a
	def kr2(self):
		return np.zeros(self.N[1:3])

class FourierTransformTool(TransverseModeTool):
	'''Transform Cartesian components to plane wave basis'''
	def kspacex(self,a):
		self.cl.program('fft').FFT_axis1(self.cl.q,(a.shape[0],a.shape[2]),None,a.data,np.int32(a.shape[1]))
		self.cl.program('fft').FFT_axis2(self.cl.q,a.shape[:2],None,a.data,np.int32(a.shape[2]))
		self.cl.q.finish()
	def rspacex(self,a):
		self.cl.program('fft').IFFT_axis2(self.cl.q,a.shape[:2],None,a.data,np.int32(a.shape[2]))
		self.cl.program('fft').IFFT_axis1(self.cl.q,(a.shape[0],a.shape[2]),None,a.data,np.int32(a.shape[1]))
		self.cl.q.finish()
	def kspace(self,a):
		return np.fft.fft(np.fft.fft(a,axis=1),axis=2)
	def rspace(self,a):
		return np.fft.ifft(np.fft.ifft(a,axis=2),axis=1)
	def k_diff(self,i):
		# Square of this is the proper eigenvalue of the finite difference laplacian
		# For small frequencies it corresponds to the wave frequency
		num = self.N[i]
		dx = self.dq[i]
		i_list = np.arange(0,num)
		sgn = np.ones(num)
		sgn[np.int(num/2)+1:] = -1.0
		return sgn*np.sqrt(2.0*(1.0 - np.cos(2.0*np.pi*i_list/num)))/dx;
	def kr2(self):
		kx = self.k_diff(1)
		ky = self.k_diff(2)
		#kx = 2.0*np.pi*np.fft.fftfreq(self.N[1],d=self.dq[1])
		#ky = 2.0*np.pi*np.fft.fftfreq(self.N[2],d=self.dq[2])
		return np.outer(kx**2,np.ones(self.N[2])) + np.outer(np.ones(self.N[1]),ky**2)

class HankelTransformTool(TransverseModeTool):
	'''Transform Cartesian components to Bessel beam basis.'''
	def __init__(self,Nr,dr,mmax,cl,Nk):
		'''Nr is the number of radial cells, Nk<=Nr is the number of radial modes to keep'''
		if Nk==None:
			Nk = Nr
		self.Nk = Nk
		r_list = cell_centers(0.0,Nr*dr,Nr)
		A1 = 2*np.pi*(r_list-0.5*dr)
		A2 = 2*np.pi*(r_list+0.5*dr)
		V = np.pi*((r_list+0.5*dr)**2 - (r_list-0.5*dr)**2)
		self.Lambda = np.sqrt(V)
		# Highest negative mode is never needed due to symmetry, so we have 2*mmax modes instead of 2*mmax+1.
		# We could save storage by only keeping eigenvectors for positive modes (negative modes are the same).
		# We trade wasting storage for simpler kernel functions.
		if mmax==0:
			self.vals = np.zeros((Nr,1)) # pad out to full resolution
			Hi = np.zeros((Nr,Nk,1))
		else:
			self.vals = np.zeros((Nr,2*mmax)) # pad out to full resolution
			Hi = np.zeros((Nr,Nk,2*mmax))
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
			self.vals[:Nk,m],Hi[:,:,m] = scipy.linalg.eig_banded(a_band_upper,select='i',select_range=(Nr-Nk,Nr-1))
		# Set eigenvalues of negative modes (they are the same as corresponding positive modes)
		# Modes are packed in usual FFT fashion, so negative indices work as expected.
		for m in range(1,mmax):
			self.vals[:Nk,-m] = self.vals[:Nk,m]
			Hi[:,:,-m] = Hi[:,:,m]
		self.H = np.ascontiguousarray(Hi.swapaxes(0,1))
		self.cl = cl
		self.H_dev = None
		self.L_dev = None
		self.scratch_dev = None
	def AllocateDeviceMemory(self,data_shape):
		self.H_dev = pyopencl.array.to_device(self.cl.q,self.H)
		self.L_dev = pyopencl.array.to_device(self.cl.q,self.Lambda)
		self.scratch_dev = pyopencl.array.empty(self.cl.q,data_shape,np.complex)
		self.cl.q.finish()
	def FreeDeviceMemory(self):
		self.H_dev = None
		self.L_dev = None
		self.scratch_dev = None
	def kspacex(self,a):
		self.cl.program('fft').FFT_axis2(self.cl.q,a.shape[:2],None,a.data,np.int32(a.shape[2]))
		self.cl.program('fft').RootVolumeMultiply(self.cl.q,a.shape,None,a.data,self.L_dev.data)
		self.scratch_dev[...] = a.copy(queue=self.cl.q)
		a.fill(0.0,queue=self.cl.q)
		shp = ( a.shape[0] , self.Nk , a.shape[2] )
		self.cl.program('fft').RadialTransform(self.cl.q,shp,None,self.H_dev.data,self.scratch_dev.data,a.data,np.int32(a.shape[1]))
		self.cl.program('fft').RootVolumeDivide(self.cl.q,a.shape,None,a.data,self.L_dev.data)
		self.cl.q.finish()
	def rspacex(self,a):
		self.cl.program('fft').RootVolumeMultiply(self.cl.q,a.shape,None,a.data,self.L_dev.data)
		self.scratch_dev[...] = a.copy(queue=self.cl.q)
		self.cl.program('fft').InverseRadialTransform(self.cl.q,a.shape,None,self.H_dev.data,self.scratch_dev.data,a.data,np.int32(self.Nk))
		self.cl.program('fft').RootVolumeDivide(self.cl.q,a.shape,None,a.data,self.L_dev.data)
		self.cl.program('fft').IFFT_axis2(self.cl.q,a.shape[:2],None,a.data,np.int32(a.shape[2]))
		self.cl.q.finish()
	def trans(self,f,v):
		vc = np.ascontiguousarray(np.copy(v))
		if type(self.H_dev)==type(None):
			self.AllocateDeviceMemory(vc.shape)
			v_dev = pyopencl.array.to_device(self.cl.q,vc)
			f(v_dev)
			self.FreeDeviceMemory()
		else:
			v_dev = pyopencl.array.to_device(self.cl.q,vc)
			f(v_dev)
		return v_dev.get()
	def kspace(self,a):
		if len(a.shape)==3:
			return self.trans(self.kspacex,a)
		else:
			for k in range(a.shape[3]):
				a[...,k] = self.trans(self.kspacex,a[...,k])
			return a
	def rspace(self,a):
		if len(a.shape)==3:
			return self.trans(self.rspacex,a)
		else:
			for k in range(a.shape[3]):
				a[...,k] = self.trans(self.rspacex,a[...,k])
			return a
	def kr2(self):
		return -self.vals

def WignerTransform(A,ds):
	N = A.shape[0]
	M = np.int(N/2) + 1
	corr = np.zeros((N,M)).astype(np.complex)
	Ai = np.zeros(N*2-1).astype(np.complex)
	Ai[::2] = A
	Ai[1::2] = 0.5*(np.roll(Ai,1)+np.roll(Ai,-1))[1::2]
	for j in range(M):
		corr[:,j] = (np.conj(np.roll(Ai,j))*np.roll(Ai,-j))[::2]
	wig = np.fft.hfft(corr,axis=1)*ds/(2*np.pi)
	return np.fft.fftshift(wig,axes=1)
