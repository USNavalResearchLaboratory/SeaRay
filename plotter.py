import sys
sys.path.append('modules')
import os
import glob
import numpy as np
from scipy import constants as C
import scipy.interpolate
import modules.grid_tools as grid_tools
import PIL.Image
import logging
import importlib.util
import pathlib
import json

plotter_defaults = {	'image colors' : 'viridis' ,
						'level colors' : 'ocean' ,
						'length' : 'mm' ,
						'time' : 'ps' }

try:
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	mpl.rcParams['text.usetex'] = False
	mpl.rcParams['font.size'] = 13
	mpl_loaded = True
except:
	mpl_loaded = False

def chk_mlab():
	'''load mlab if available and not already done.
	returns the module or None.'''
	nm = 'mayavi.mlab'
	if nm in sys.modules:
		return sys.modules[nm]
	try:
		# In some cases, find_spec returns None, in others it raises ModuleNotFoundError
		spec = importlib.util.find_spec(nm)
		if spec==None:
			return None
		mlab = importlib.util.module_from_spec(spec)
		sys.modules[nm] = mlab
		spec.loader.exec_module(mlab)
		return mlab
	except ModuleNotFoundError:
		return None

if len(sys.argv)==1:
	print('==============BEGIN HELP FOR SEARAY PLOTTER==============')
	print('Usage: plotter.py base_filename [optional arguments...]')
	print('Example: plotter.py out/test units=ps,mm,ev o31 o3d')
	print('NOTE: plotter expects input file for run to be analyzed to be in working directory.')
	print('-----------------General Argument Format-----------------')
	print('Arguments with values take form key=val1,val2,val3... (NO spaces)')
	print('In this document,optional values are in brackets, key[=opt1,opt2,...].')
	print('(Do not type out the brackets)')
	print('-----------------Required Argument-----------------')
	print('base_filename: path and prefix for data files, e.g., out/test')
	print('-----------------Formatting Options-----------------')
	print('units=time,length,frequency: time may be fs,ps,ns,us,s; length may be um,mm,cm,m,in;')
	print('  frequency may be ev,icm or else omitted to use normalized units.')
	print('labels=system: system may be indexed,cart,cyl,sph')
	print('drange=dr: if dr=0, use linear color scale, otherwise log10 scale with dynamic range = dr')
	print('color=cmap: cmap is a string naming any colormap understood by Matplotlib or Mayavi')
	print('origin=x1,x2,x3: reference all spatial coordinates to this point')
	print('res: resolution points for eikonal fields, no spaces, e.g., res=200,200')
	print('-----------------Indexing-----------------')
	print('i,j,k should be replaced by a 1-digit hexidecimal number from 0-b.')
	print('0-3 maps to (ct,x,y,z).')
	print('4-7 maps to (w/c,kx,ky,kz).')
	print('8-b maps to (phase,ax,ay,az).')
	print('-----------------Orbits-----------------')
	print('oij[=h0,h1,v0,v1]: plot orbits in ij plane, e.g., o31')
	print('fijk: field reconstruction for component k in the ij plane (requires detailed orbits)')
	print('o3d: 3D plot of ray orbits')
	print('-------------Eikonal Detectors---------------')
	print('detector_name=i,j[/k,ak,bk/l,al,bl/...]: intensity in detection plane of the named detector')
	print('  First 2 indices are detection plane.')
	print('  Triples inside slashes are filters.')
	print('  First element is index of quantity to filter, next two are filter bounds.')
	print('  OK to filter on any index any number of times.')
	print('  N.b. lack of any filter means you are getting a PROJECTION not a SLICE.')
	print('------------Full Wave Detectors---------------')
	print('detector_name[=[modifiers]i,j/k,l[/f]]: field reconstruction in detection plane of the named detector')
	print('  modifiers: t causes frequency slices to be transformed to the time domain.')
	print('             e causes the eikonal plane data to be used.')
	print('             a causes a line plot to be unfolded into an image, assuming axisymmetry.')
	print('  The slashes separate plot axes from slice indices.  The number of plot axes determines the type of plot.')
	print('  The first group of indices are the plotting axes, e.g. 1,2 or 1,2,3.')
	print('  The next group of indices select slices of the remaining axes.')
	print('  The last slash, if present, precedes the fraction of the domain to display.')
	print('  You can use 5 dimensions if time and frequency are the plot axes (induces Wigner transform).')
	print('---------------Other Plots-----------------')
	print('mesh: display all surface meshes (may be very slow)')
	print('bundle=b: display the configuration of the designated ray bundle, e.g. bundle=0')
	print('-----------------Animations-----------------')
	print('Generate animations by replacing a slice index with a python range.')
	print('E.g., det=t1,2/:5,0 generates a movie in the xy plane with the first 5 time levels at z slice 0.')
	print('Please only make a single movie at a time, and rename mov.gif before creating the next one.')
	print('==============END HELP FOR SEARAY PLOTTER==============')
	exit(1)

basename = pathlib.Path(sys.argv[1])
simname = basename.name
os.chdir(basename.parent)
with open(simname + "_sim.json") as f:
	sim_dict = json.load(f)
with open(simname + "_sources.json") as f:
	sources_list = json.load(f)

# Put the arguments into a dictionary where whatever is left of '=' is the key
# and whatever is right of '=' is the value.
arg_dict = {}
for arg in sys.argv:
	try:
		arg_dict[arg.split('=')[0]] = arg.split('=')[1]
	except:
		arg_dict[arg.split('=')[0]] = 'default'

try:
	my_color_map = arg_dict['color']
except KeyError:
	my_color_map = plotter_defaults['image colors']

try:
	dynamic_range = float(arg_dict['drange'])
except KeyError:
	dynamic_range = 0.0

def cleanup(wildcarded_path):
	cleanstr = glob.glob(wildcarded_path)
	for f in cleanstr:
		os.remove(f)

def SliceAxes(plot_ax):
	'''Deduce the slice axes from the plotting axes'''
	ans = ()
	for ax in range(4):
		if ax not in plot_ax:
			ans += (ax,)
	return ans

def ParseSlices(dims,plot_ax,cmd_str):
	'''Function to generate a list of slice tuples for the movie.
	plot_ax = tuple with the plotting axes.
	cmd_str = string with comma delimited ranges or indices.
	Returns slice_tuples,data_ax,movie.'''
	sax = SliceAxes(plot_ax)
	slice_tuples = []
	range_tuples = []
	movie = False
	# Define data axes such that time and frequency have the same index
	data_ax = ()
	for ax in plot_ax:
		if ax==4:
			data_ax += (0,)
		else:
			data_ax += (ax,)
	# Construct list of range tuples
	for saxi,slice_str in enumerate(cmd_str.split(',')):
		rng = slice_str.split(':')
		tup = ()
		for i,el in enumerate(rng):
			if el=='' and i==0:
				el = '0'
			if el=='' and i==1:
				el = str(dims[sax[saxi]])
			if el=='' and i==2:
				el = '1'
			tup += (int(el),)
		range_tuples.append(tup)
	# Determine the range of the movie frames
	frame_rng = range(1)
	for rng in range_tuples:
		movie = movie or len(rng)>1
		if len(rng)==2:
			frame_rng = range(rng[0],rng[1])
		if len(rng)==3:
			frame_rng = range(rng[0],rng[1],rng[2])
	# Construct list of slice tuples
	for r in frame_rng:
		tup = ()
		for rng in range_tuples:
			if len(rng)>1:
				tup += (r,)
			else:
				tup += rng
		slice_tuples.append(tup)
	return slice_tuples,data_ax,movie

def ExtractSlice(A,plot_ax,slice_idx):
	'''Slice the data using tuples of plotting axes and slice indices
	The number of axes and indices should add to 4'''
	sax = SliceAxes(plot_ax)
	# Following assumes sax is sorted
	for i in range(len(sax)-1,-1,-1):
		if slice_idx[i]>=A.shape[sax[i]]:
			A = np.take(A,-1,sax[i])
		else:
			A = np.take(A,slice_idx[i],sax[i])
	return A

def TransformColorScale(array,dyn_rng):
	if dyn_rng!=0.0:
		low_bound = np.max(array)/10**dyn_rng
		array[np.where(array<low_bound)] = low_bound
		array[...] = np.log10(array)
		return r'${\rm log}_{10}$'
	else:
		return r''

def FractionalPowerScale(array,order):
	idxneg = np.where(array<0)
	idxpos = np.where(array>0)
	array[idxpos] **= 1.0/order
	array[idxneg] *= -1
	array[idxneg] **= 1.0/order
	array[idxneg] *= -1
	return array

def CartesianReduce(A,dom,frac):
	hc = int(A.shape[1]/2)
	vc = int(A.shape[2]/2)
	dh = int(A.shape[1]*frac/2)
	dv = int(A.shape[2]*frac/2)
	dom = np.copy(dom)
	dom[2:6] *= frac
	return A[:,hc-dh:hc+dh,vc-dv:vc+dv,:],dom

def RadialReduce(A,dom,frac):
	dh = int(A.shape[1]*frac)
	dom = np.copy(dom)
	dom[2:4] *= frac
	return A[:,:dh,...],dom

def GetPrefixList(postfix):
	ans = []
	for f in glob.glob('*_*_'+postfix+'.npy'):
		ans.append(f[:-len(postfix)-4])
	return ans

def IntegrateImage(A,rng):
	# Assumes A is arranged as [row,col]
	dh = (rng[1] - rng[0]) / A.shape[1]
	dv = (rng[3] - rng[2]) / A.shape[0]
	return np.sum(A)*dh*dv

class Units:
	def __init__(self,label_type):
		mks_length = sim_dict['mks_length']
		mks_time = sim_dict['mks_time']
		try:
			t_str = arg_dict['units'].split(',')[0]
			l_str = arg_dict['units'].split(',')[1]
		except:
			t_str = plotter_defaults['time']
			l_str = plotter_defaults['length']
		try:
			w_str = arg_dict['units'].split(',')[2]
		except:
			w_str = 'normalized'

		carrier = sources_list[0]['waves'][0]['k0'][0]

		if l_str=='um':
			l1 = mks_length*1e6
		if l_str=='mm':
			l1 = mks_length*1e3
		if l_str=='cm':
			l1 = mks_length*1e2
		if l_str=='m':
			l1 = mks_length
		if l_str=='in':
			l1 = mks_length*1e2/2.54
		l_lab = l_str.replace('um',r'$\mu$m')

		if t_str=='fs':
			t1 = mks_time*1e15
		if t_str=='ps':
			t1 = mks_time*1e12
		if t_str=='ns':
			t1 = mks_time*1e9
		if t_str=='us':
			t1 = mks_time*1e6
		if t_str=='s':
			t1 = mks_time
		t_lab = t_str.replace('us',r'$\mu$s')

		if w_str=='normalized':
			w1 = carrier
			w_lab = r'$k_{00}$'
		if w_str=='ev' or w_str=='eV':
			w1 = C.hbar*(C.c/mks_length)/C.e
			w_lab = 'eV'
		if w_str=='icm':
			w1 = .01/(2*np.pi*mks_length)
			w_lab = r'$cm^{-1}$'

		self.length_label = l_str
		self.time_label = t_str
		self.mks_length = mks_length
		self.mks_time = mks_time

		self.normalization = np.concatenate(([t1],np.ones(3)*l1,np.ones(4)*w1,np.ones(4)))
		self.lab_str = [r'$x_0$ ',r'$x_1$ ',r'$x_2$ ',r'$x_3$ ']
		if w_str=='eV':
			self.lab_str += [r'$\hbar k_0$ ',r'$\hbar k_1$ ',r'$\hbar k_2$ ',r'$\hbar k_3$ ']
		elif w_str=='icm':
			self.lab_str += [r'$\lambda^{-1}$ ',r'$k_1$ ',r'$k_2$ ',r'$k_3$ ']
		else:
			self.lab_str += [r'$k_0$ ',r'$k_1$ ',r'$k_2$ ',r'$k_3$ ']
		self.lab_str += [r'$\psi$',r'$a_1$',r'$a_2$',r'$a_3$']
		self.lab_str[0] += '('+t_lab+')'
		self.lab_str[1:4] = [s+'('+l_lab+')' for s in self.lab_str[1:4]]
		self.lab_str[4:8] = [s+'('+w_lab+')' for s in self.lab_str[4:8]]
		if label_type=='cart':
			self.lab_str[0] = self.lab_str[0].replace('x_0','t')
			self.lab_str[1] = self.lab_str[1].replace('x_1','x')
			self.lab_str[2] = self.lab_str[2].replace('x_2','y')
			self.lab_str[3] = self.lab_str[3].replace('x_3','z')
		if label_type=='cyl':
			self.lab_str[0] = self.lab_str[0].replace('x_0','t')
			self.lab_str[1] = self.lab_str[1].replace('x_1',r'\rho')
			self.lab_str[2] = r'$\varphi/\pi$'
			self.lab_str[3] = self.lab_str[3].replace('x_3','z')
			self.normalization[2] = 1/np.pi
		if label_type=='sph':
			self.lab_str[0] = self.lab_str[0].replace('x_0','t')
			self.lab_str[1] = self.lab_str[1].replace('x_1','r')
			self.lab_str[2] = r'$\varphi/\pi$'
			self.lab_str[3] = r'$\theta/\pi$'
			self.normalization[2] = 1/np.pi
			self.normalization[3] = 1/np.pi
		if label_type=='cart' or label_type=='cyl' or label_type=='sph':
			self.lab_str[4] = self.lab_str[4].replace('k_0',r'\omega').replace('k_{00}',r'\omega_0')
			self.lab_str[5] = self.lab_str[5].replace('k_1','k_x').replace('k_{00}',r'\omega_0')
			self.lab_str[6] = self.lab_str[6].replace('k_2','k_y').replace('k_{00}',r'\omega_0')
			self.lab_str[7] = self.lab_str[3].replace('k_3','k_z').replace('k_{00}',r'\omega_0')
	def GetNormalization(self):
		return self.normalization
	def GetLabels(self):
		return self.lab_str
	def PlotExt(self,dom,plot_ax):
		ans = []
		for ax in plot_ax:
			ans += [dom[2*ax]*self.normalization[ax],dom[2*ax+1]*self.normalization[ax]]
		return ans
	def LengthLabel(self):
		return self.length_label
	def TimeLabel(self):
		return self.time_label
	def GetWcm2(self,a2,w0):
		w0_mks = w0/self.mks_time
		E0_mks = np.sqrt(a2) * C.m_e * w0_mks * C.c / C.e
		eta0 = 1/C.c/C.epsilon_0
		return 1e-4 * 0.5 * E0_mks**2 / eta0


class MeshViewer:
	def __init__(self,lab_sys):
		self.label_system = lab_sys
		self.structured_mesh = []
		self.mesh = []
		self.simplex = []
		l = GetPrefixList('simplices')
		for prefix in l:
			self.mesh.append(prefix+'mesh.npy')
			self.simplex.append(prefix+'simplices.npy')
		l = GetPrefixList('mesh')
		for prefix in l:
			self.structured_mesh.append(prefix+'mesh.npy')
	def GetMeshList(self):
		return self.structured_mesh
	def Plot(self,mpl_plot_count,maya_plot_count):
		lab_str = self.label_system.GetLabels()
		normalization = self.label_system.GetNormalization()
		if 'mesh' in sys.argv:
			# For rotated meshes we have a problem here
			mpl_plot_count += 1
			fig = plt.figure(mpl_plot_count,figsize=(7,6))
			ax = fig.add_subplot(111,projection='3d')
			surf = []
			for j,path in enumerate(self.mesh):
				mesh = np.load(path)
				simplices = np.load(self.simplex[j])
				x = mesh[:,:,1].flatten()
				y = mesh[:,:,2].flatten()
				z = mesh[:,:,3].flatten()
				tri = mpl.tri.Triangulation(normalization[1]*x,normalization[2]*y,triangles=simplices)
				surf.append(ax.plot_trisurf(tri,normalization[3]*z,cmap=plotter_defaults['level colors']))
				#ax.set_zlim(-5,5)
			ax.set_xlabel(lab_str[1])
			ax.set_ylabel(lab_str[2])
			ax.set_zlabel(lab_str[3])
			plt.tight_layout()
		return mpl_plot_count,maya_plot_count

class Bundles:
	def __init__(self,lab_sys):
		self.label_system = lab_sys
		self.xp = simname+'_xp0.npy'
	def Plot(self,mpl_plot_count,maya_plot_count):
		lab_str = self.label_system.GetLabels()
		normalization = self.label_system.GetNormalization()
		if 'bundle' in arg_dict:
			mlab = chk_mlab()
			if mlab!=None:
				maya_plot_count += 1
				b = int(arg_dict['bundle'])
				xpb = np.load(self.xp)[b,...]
				print('Primary ray location =',xpb[0,1:4])
				xpb[:,1:4] -= xpb[0,1:4]
				xpb[:,1:4] *= normalization[1:4]
				extent = [np.min(xpb[:,1]),np.max(xpb[:,1]),
					np.min(xpb[:,2]),np.max(xpb[:,2]),
					np.min(xpb[:,3]),np.max(xpb[:,3])]
				mlab.quiver3d(xpb[...,1],xpb[...,2],xpb[...,3],
					xpb[...,5],xpb[...,6],xpb[...,7],mode='arrow')
				mlab.axes(xlabel=lab_str[1],ylabel=lab_str[2],zlabel=lab_str[3],extent=extent)
				mlab.outline(extent=extent)
				mlab.view(azimuth=80,elevation=30,distance=6*np.max(extent),focalpoint=(0,0,0))
		return mpl_plot_count,maya_plot_count

class Orbits:
	def __init__(self,mesh_list,lab_sys):
		self.label_system = lab_sys
		self.orbits = np.load(simname+'_orbits.npy')
		# flatten orbit data for field reconstruction (time*ray,component)
		self.xpo = self.orbits.reshape(self.orbits.shape[0]*self.orbits.shape[1],self.orbits.shape[2])
		# extract spatial part of phase (assumes monochromatic mode)
		self.xpo[:,8] += self.xpo[:,4]*self.xpo[:,0]
		self.mesh_list = mesh_list
		self.res = (200,200)
	def Plot(self,mpl_plot_count,maya_plot_count):
		lab_str = self.label_system.GetLabels()
		normalization = self.label_system.GetNormalization()
		for i in range(12):
			for j in range(12):
				for k in range(12):
					plot_key = 'f'+format(i,'01X').lower()+format(j,'01X').lower()+format(k,'01X').lower()
					if plot_key in sys.argv:
						mpl_plot_count += 1
						plt.figure(mpl_plot_count,figsize=(7,6))
						harray,plot_ext = grid_tools.GridFromInterpolation(self.xpo[:,i],self.xpo[:,j],self.xpo[:,k],self.res,self.res)
						cbar_str = TransformColorScale(harray,dynamic_range)
						plt.imshow(harray.swapaxes(0,1),origin='lower',cmap=my_color_map,aspect='auto',extent=self.label_system.PlotExt(plot_ext,i,j))
						b=plt.colorbar()
						b.set_label(cbar_str+lab_str[k],size=18)
						plt.xlabel(lab_str[i],size=18)
						plt.ylabel(lab_str[j],size=18)
						plt.tight_layout()
		wmin = np.min(self.orbits[...,4])
		wmax = np.max(self.orbits[...,4])
		a2 = self.orbits[...,9]**2 + self.orbits[...,10]**2 + self.orbits[...,11]**2
		a2max = np.max(a2[0,:])
		render_alpha = np.tanh(100/self.orbits.shape[1])
		if a2max<1e-40:
			get_color = lambda i : [0.0,0.0,0.0,1.0]
		else:
			if wmin!=wmax:
				get_color = lambda i : np.concatenate((mpl.colors.hsv_to_rgb([0.8*(self.orbits[0,i,4]-wmin)/(wmax-wmin),1.0,1.0]),[render_alpha*(a2[0,i]/a2max)**0.0001]))
			else:
				get_color = lambda i : [0.0,0.0,0.0,render_alpha*(a2[0,i]/a2max)**0.0001]
		if 'o3d' in sys.argv:
			characteristic_size = normalization[1]*(np.max(self.orbits[...,1:4]) - np.min(self.orbits[...,1:4]))
			mlab = chk_mlab()
			if mlab:
				maya_plot_count += 1
				for j in range(self.orbits.shape[1]):
					x = normalization[1]*self.orbits[:,j,1]
					y = normalization[2]*self.orbits[:,j,2]
					z = normalization[3]*self.orbits[:,j,3]
					w = self.orbits[:,j,4]
					o = a2[:,j]
					xtest = np.roll(x,1)!=x
					ytest = np.roll(y,1)!=y
					ztest = np.roll(z,1)!=z
					test = np.logical_or(np.logical_or(xtest,ytest),ztest)
					x = x[np.where(test)]
					y = y[np.where(test)]
					z = z[np.where(test)]
					w = w[np.where(test)]
					o = o[np.where(test)]
					if x.shape[0]>0:
						mlab.plot3d(x,y,z,w,tube_radius=.001*characteristic_size,vmin=wmin,vmax=wmax,colormap='gist_rainbow',opacity=0.4)
				for path in self.mesh_list:
					mesh = np.load(path)
					c = mesh[:,:,0]
					x = mesh[:,:,1]
					y = mesh[:,:,2]
					z = mesh[:,:,3]
					mlab.mesh(x*normalization[1],y*normalization[2],z*normalization[3],color=(0.5,1,0.5),opacity=0.5)
			if mpl_loaded and not mlab:
				mpl_plot_count += 1
				fig = plt.figure(mpl_plot_count,figsize=(7,6))
				ax = fig.add_subplot(111,projection='3d')
				for j in range(self.orbits.shape[1]):
					x = normalization[1]*self.orbits[:,j,1]
					y = normalization[2]*self.orbits[:,j,2]
					z = normalization[3]*self.orbits[:,j,3]
					ax.plot(x,y,z,color=get_color(j))
				surf = []
				cmap = mpl.cm.ScalarMappable(cmap=plotter_defaults['level colors'])
				needsBar = False
				for path in self.mesh_list:
					mesh = np.load(path)
					needsBar = needsBar or np.max(mesh[...,0])!=np.min(mesh[...,0])
					c = cmap.to_rgba(mesh[:,:,0]*normalization[3],alpha=0.5)
					x = mesh[:,:,1]
					y = mesh[:,:,2]
					z = mesh[:,:,3]
					#surf.append(ax.plot_surface(x*normalization[1],y*normalization[2],z*normalization[3],rcount=16,ccount=16,facecolors=c))
					surf.append(ax.plot_surface(x*normalization[1],y*normalization[2],z*normalization[3],facecolors=c))
					cmap.set_array(mesh[:,:,0]*normalization[3])
				ax.set(xlabel=lab_str[1],ylabel=lab_str[2],zlabel=lab_str[3])
				# mplot3d fixes the projection as a cube, so pad the axis limits to preserve physical aspect ratios
				xl,xh = ax.get_xlim()
				yl,yh = ax.get_ylim()
				zl,zh = ax.get_zlim()
				padding = 0.5*(np.max([xh-xl,yh-yl,zh-zl]) - np.array([xh-xl,yh-yl,zh-zl]))
				ax.set_xlim(xl-padding[0],xh+padding[0])
				ax.set_ylim(yl-padding[1],yh+padding[1])
				ax.set_zlim(zl-padding[2],zh+padding[2])
				if len(self.mesh_list)>0 and needsBar:
					fig.colorbar(cmap, shrink=0.5, aspect=10, label='height ('+self.label_system.LengthLabel()+')')
				plt.tight_layout()
		for i in range(12):
			for j in range(12):
				plot_key = 'o'+format(i,'01X').lower()+format(j,'01X').lower()
				try:
					arg = arg_dict[plot_key]
				except KeyError:
					arg = 'stop'
				if arg!='stop':
					mpl_plot_count += 1
					plt.figure(mpl_plot_count,figsize=(4,4))
					center = (origin[i],origin[j])
					if arg!='default':
						lims = (float(arg.split(',')[0]),float(arg.split(',')[1]),float(arg.split(',')[2]),float(arg.split(',')[3]))
						plt.xlim(lims[0]-center[0],lims[1]-center[0])
						plt.ylim(lims[2]-center[1],lims[3]-center[1])
					for k in range(self.orbits.shape[1]):
						plt.plot(normalization[i]*self.orbits[:,k,i]-center[0],normalization[j]*self.orbits[:,k,j]-center[1],color=get_color(k))
					plt.xlabel(lab_str[i],size=18)
					plt.ylabel(lab_str[j],size=18)
					plt.tight_layout()
		return mpl_plot_count,maya_plot_count

class EikonalWaveProfiler:
	def __init__(self,lab_sys):
		self.label_system = lab_sys
		self.name = []
		self.xp = []
		self.eik = []
		l = GetPrefixList('xps')
		for prefix in l:
			self.name.append(prefix.split('_')[-2])
			self.xp.append(prefix+'xps.npy')
			self.eik.append(prefix+'eiks.npy')
		try:
			self.res = (int(arg_dict['res'].split(',')[0]) , int(arg_dict['res'].split(',')[1]))
		except:
			self.res = (200,200)
		print('eikonal detectors =',self.name)
	def Plot(self,mpl_plot_count,maya_plot_count):
		for det_idx,det_name in enumerate(self.name):
			try:
				arg = arg_dict[det_name]
				xps = np.load(self.xp[det_idx])
				eiks = np.load(self.eik[det_idx])
				lab_str = self.label_system.GetLabels()
				normalization = self.label_system.GetNormalization()
				det_plane = arg.split('/')[0]
				filters = arg.split('/')[1:]
				i = int(det_plane.split(',')[0],16)
				j = int(det_plane.split(',')[1],16)
				sel = np.array([True]*xps.shape[0])
				for filt in filters:
					f = int(filt.split(',')[0],16)
					band = (np.double(filt.split(',')[1]),np.double(filt.split(',')[2]))
					if f<8:
						w = np.logical_and(xps[:,f]>=band[0],xps[:,f]<band[1])
					else:
						w = np.logical_and(eiks[:,f-8]>=band[0],eiks[:,f-8]<band[1])
					sel = np.logical_and(sel,w)
				mpl_plot_count += 1
				plt.figure(mpl_plot_count,figsize=(7,6))
				if i<8:
					x = xps[:,i]
				else:
					x = eiks[:,i-8]
				if j<8:
					y = xps[:,j]
				else:
					y = eiks[:,j-8]
				weights = eiks[:,1]**2 + eiks[:,2]**2 + eiks[:,3]**2
				sel = np.logical_and(sel,np.logical_and(np.logical_not(np.isnan(x)),np.logical_not(np.isnan(y))))
				harray,plot_ext = grid_tools.GridFromBinning(normalization[i]*x[sel],normalization[j]*y[sel],weights[sel],self.res[0],self.res[1])
				#harray = grid_tools.Smooth1D(harray,4,0)
				#harray = grid_tools.Smooth1D(harray,4,1)
				pre_str = TransformColorScale(harray,dynamic_range)
				plt.imshow(harray.swapaxes(0,1),origin='lower',cmap=my_color_map,aspect='auto',extent=plot_ext)
				b=plt.colorbar()
				b.set_label(pre_str+r'$|a|^2$',size=18)
				plt.xlabel(lab_str[i],size=18)
				plt.ylabel(lab_str[j],size=18)
				plt.tight_layout()
			except KeyError:
				logging.info('%s not used.',det_name)
		return mpl_plot_count,maya_plot_count

class FullWaveProfiler:
	def __init__(self,type_key,lab_sys):
		self.label_system = lab_sys
		self.name = []
		self.eik = []
		self.wave = []
		self.ext = []
		l = GetPrefixList(type_key+'_wave')
		for prefix in l:
			self.name.append(prefix.split('_')[-2])
			self.wave.append(prefix+type_key+'_wave.npy')
			self.eik.append(prefix+type_key+'_eik.npy')
			self.ext.append(prefix+type_key+'_plot_ext.npy')
	def Plot(self,mpl_plot_count,maya_plot_count):
		for det_idx,det_name in enumerate(self.name):
			try:
				arg = arg_dict[det_name]
				dom = np.load(self.ext[det_idx])
				E = np.load(self.eik[det_idx])
				A = np.load(self.wave[det_idx])
				# Make dom[0..1] the time window and dom[8..9] the frequency window
				dom = np.concatenate((dom,dom[:2]))
				w00 = 0.5*(dom[8]+dom[9])
				dw = (dom[9] - dom[8]) / A.shape[0]
				dt = 2*np.pi/(dom[9] - dom[8])
				dom[0] = 0.0
				dom[1] = 2*np.pi/dw
				time_domain = False
				eikonal_plane = False
				unfold_axisymmetry = False
				# In the following we put the requested plot into machine-friendly terms.
				# plot_ax = tuple with the requested plot axes, labelled from 0 to 4 (0 is time, 4 is frequency)
				# data_ax = tuple similar to plot_ax, except indices are actual offsets into the data array (basically 4 gets mapped back to 0)
				# slice_tuples = list of tuples, each tuple represents one plot, the tuple contains slice indices into the axes that are not plotted.
				if arg=='default':
					plot_ax = (1,2)
					slice_tuples,data_ax,movie = ParseSlices(A.shape,plot_ax,'0,0')
					wave_zone_fraction = 1.0
				else:
					while arg[0]=='t' or arg[0]=='e' or arg[0]=='a':
						if arg[0]=='t':
							time_domain = True
						if arg[0]=='e':
							eikonal_plane = True
						if arg[0]=='a':
							unfold_axisymmetry = True
						arg = arg[1:]
					plot_ax = tuple(map(int,arg.split('/')[0].split(',')))
					if 0 in plot_ax:
						time_domain = True
					slice_tuples,data_ax,movie = ParseSlices(A.shape,plot_ax,arg.split('/')[1])
					if len(arg.split('/'))>2:
						wave_zone_fraction = np.double(arg.split('/')[2])
					else:
						wave_zone_fraction = 1.0
				if eikonal_plane:
					A = E[...,np.newaxis,:]
				if A.shape[0]==1:
					#bar_label = r'$|a|^2$'
					bar_label = r'Intensity (W/cm$^2$)'
					A *= 1e-2*np.sqrt(0.5/377)*w00*(C.c/self.label_system.mks_length)*C.m_e*C.c/C.e
				else:
					if time_domain:
						bar_label = r'$|a(t)|^2$'
						A = np.fft.ifft(np.fft.ifftshift(np.conj(A),axes=0),axis=0)
						A = np.roll(A,8,axis=0)
					else:
						# Normalize so that integral(a^2*dt) = integral(a^2*dw)/(2*pi*w00)
						test_spectrum = np.ones(A.shape[0])
						test_trace = np.fft.ifft(test_spectrum)
						time_integral = np.sum(np.abs(test_trace)**2)*dt
						freq_integral = A.shape[0]*dw/(2*np.pi*w00)
						A *= np.sqrt(time_integral/freq_integral)
						bar_label = r'$|a(\omega)|^2$' + ' ('+self.label_system.TimeLabel()+')'
						A *= np.sqrt(self.label_system.GetNormalization()[0])
				wigner = False
				if len(data_ax)>1:
					if data_ax[0]==0 and data_ax[1]==0:
						bar_label = r'${\cal N}(\omega,t)$'
						wigner = True
						A2 = A[...,0]
						rdom = dom
				if not wigner:
					A2 = np.abs(A[...,0])**2 + np.abs(A[...,1])**2 + np.abs(A[...,2])**2
					A2,rdom = self.reducer(A2,dom,wave_zone_fraction)
				lab_str = self.label_system.GetLabels()
				lab_str[0] = lab_str[0].replace('t','t-z/c')
				lab_range_full = self.label_system.PlotExt(dom,plot_ax)
				lab_range_red = self.label_system.PlotExt(rdom,plot_ax)
				if movie and not wigner:
					cbar_str = TransformColorScale(A2,dynamic_range)
					val_rng = (np.min(A2),np.max(A2))
				for file_idx,slice_now in enumerate(slice_tuples):
					data_slice = ExtractSlice(A2,data_ax,slice_now)
					if wigner:
						if plot_ax[0]==4:
							data_ax = (1,0)
						else:
							data_ax = (0,1)
						ds = (dom[1] - dom[0])/A2.shape[0]
						data_slice = grid_tools.WignerTransform(data_slice,ds)
					if not movie or wigner:
						cbar_str = TransformColorScale(data_slice,dynamic_range)
						val_rng = (np.min(data_slice),np.max(data_slice))
					if len(plot_ax)==1:
						#print(det_name,'integration =',IntegrateImage(data_slice,lab_range_red))
						mpl_plot_count += 1
						plt.figure(mpl_plot_count,figsize=(5,4))
						if unfold_axisymmetry:
							N = len(data_slice)
							x = np.linspace(0.5,N-0.5,N)
							f = scipy.interpolate.interp1d(x,data_slice,kind='linear',bounds_error=False,fill_value=0.0)
							xg = np.outer(np.linspace(-N+0.5,N-0.5,2*N),np.ones(2*N))
							yg = np.outer(np.ones(2*N),np.linspace(-N+0.5,N-0.5,2*N))
							r = np.sqrt(xg**2 + yg**2)
							unfolded = f(r)
							unfolded_ext = [-lab_range_red[1]/2,lab_range_red[1]/2]
							unfolded_ext = unfolded_ext + unfolded_ext
							xl = lab_str[plot_ax[0]].replace(r'\rho',r'x')
							yl = lab_str[plot_ax[0]].replace(r'\rho',r'y')
							plt.imshow(unfolded,origin='lower',vmin=val_rng[0],vmax=val_rng[1],cmap=my_color_map,aspect='auto',extent=unfolded_ext)
							plt.xlabel(xl,size=18)
							plt.ylabel(yl,size=18)
						else:
							plt.plot(np.linspace(lab_range_red[0],lab_range_red[1],data_slice.shape[0]),data_slice)
							plt.xlabel(lab_str[plot_ax[0]],size=18)
							plt.ylabel(cbar_str+bar_label,size=18)
						plt.tight_layout()
						if movie:
							img_file = 'frame{:03d}.png'.format(file_idx)
							print('saving',img_file,'...')
							plt.savefig(img_file)
							plt.close()
					if len(plot_ax)==2:
						if data_ax[0]<data_ax[1]:
							data_slice = data_slice.swapaxes(0,1)
						print(det_name,'integration =',IntegrateImage(data_slice,lab_range_red))
						mpl_plot_count += 1
						plt.figure(mpl_plot_count,figsize=(5,4))
						plt.imshow(data_slice,origin='lower',vmin=val_rng[0],vmax=val_rng[1],cmap=my_color_map,aspect='auto',extent=lab_range_red)
						plt.xlabel(lab_str[plot_ax[0]],size=18)
						plt.ylabel(lab_str[plot_ax[1]],size=18)
						b=plt.colorbar()
						b.set_label(cbar_str+bar_label,size=18)
						plt.tight_layout()
						if movie:
							img_file = 'frame{:03d}.png'.format(file_idx)
							print('saving',img_file,'...')
							plt.savefig(img_file)
							plt.close()
					if len(plot_ax)==3:
						mlab = chk_mlab()
						if not mlab:
							print('cannot make this plot without maya')
							return
						maya_plot_count += 1
						dv = val_rng[1]-val_rng[0]
						contour_list = []
						for x in [0.25,0.75,0.9]:
							contour_list.append(val_rng[0]+x*dv)
						#mlab.clf()
						# CAUTION: the extent key has to appear in just the right places or we get confusing results
						#src = mlab.pipeline.scalar_field(x1,x2,x3,data_slice,extent=ext)
						src = mlab.pipeline.scalar_field(data_slice)
						obj = mlab.pipeline.iso_surface(src,contours=contour_list,opacity=0.3)
						#mlab.outline(extent=ext)
						#mlab.view(azimuth=-80,elevation=30,distance=3*np.max(sizes),focalpoint=origin)
						if movie:
							mlab.savefig('frame{:03d}.png'.format(file_idx))
				if movie:
					print('Consolidating into movie file...')
					images = []
					frameRateHz = 5
					for f in sorted(glob.glob('frame*.png')):
						images.append(PIL.Image.open(f))
					images[0].save('mov.gif',save_all=True,append_images=images[1:],duration=int(1000/frameRateHz),loop=0)
					cleanup('frame*.png')
					print('Done.')
			except KeyError:
				logging.info('%s not used.',det_name)
		return mpl_plot_count,maya_plot_count

class PlaneWaveProfiler(FullWaveProfiler):
	def __init__(self,lab_sys):
		super().__init__('plane',lab_sys)
		self.reducer = CartesianReduce
		print('plane wave detectors =',self.name)

class BesselBeamProfiler(FullWaveProfiler):
	def __init__(self,lab_sys):
		super().__init__('bess',lab_sys)
		self.reducer = RadialReduce
		print('Bessel beam detectors =',self.name)

mpl_plot_count = 0
maya_plot_count = 0

try:
	label_type = arg_dict['labels']
except KeyError:
	label_type = 'indexed'

try:
	origin = arg_dict['origin'].split(',')
	origin = np.array([0.0,np.double(origin[0]),np.double(origin[1]),np.double(origin[2]),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
except KeyError:
	origin = np.zeros(12)

meshPlots = MeshViewer(Units(label_type))
mpl_plot_count,maya_plot_count = meshPlots.Plot(mpl_plot_count,maya_plot_count)

bundlePlots = Bundles(Units(label_type))
mpl_plot_count,maya_plot_count = bundlePlots.Plot(mpl_plot_count,maya_plot_count)

orbitPlots = Orbits(meshPlots.GetMeshList(),Units(label_type))
mpl_plot_count,maya_plot_count = orbitPlots.Plot(mpl_plot_count,maya_plot_count)

eikonalPlots = EikonalWaveProfiler(Units(label_type))
mpl_plot_count,maya_plot_count = eikonalPlots.Plot(mpl_plot_count,maya_plot_count)

planePlots = PlaneWaveProfiler(Units(label_type))
mpl_plot_count,maya_plot_count = planePlots.Plot(mpl_plot_count,maya_plot_count)

besselPlots = BesselBeamProfiler(Units('cyl'))
mpl_plot_count,maya_plot_count = besselPlots.Plot(mpl_plot_count,maya_plot_count)

if maya_plot_count>0:
	mlab = chk_mlab()
	if mlab:
		mlab.show()

if mpl_loaded and mpl_plot_count>0:
	plt.show()
