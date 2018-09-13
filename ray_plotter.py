import os
import glob
import sys
import numpy as np
from scipy import constants as C
import grid_tools
import inputs

plotter_defaults = {	'image colors' : 'viridis' ,
						'level colors' : 'ocean' ,
						'length' : 'mm' ,
						'time' : 'ps' }

try:
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	#mpl.rcParams['text.usetex'] = True
	mpl.rcParams['font.size'] = 13
	mpl_loaded = True
except:
	mpl_loaded = False

try:
	from mayavi import mlab
	maya_loaded = True
except:
	maya_loaded = False

if len(sys.argv)==1:
	print('==============BEGIN HELP FOR SEARAY PLOTTER==============')
	print('Usage: ray_plotter.py base_filename [optional arguments...]')
	print('Example: ray_plotter.py out/test units=ps,mm o31 o3d')
	print('NOTE: plotter expects input file for run to be analyzed to be in working directory.')
	print('-----------------General Argument Format-----------------')
	print('Arguments with values take form key=val1,val2,val3... (NO spaces)')
	print('In this document,optional values are in brackets, key[=opt1,opt2,...].')
	print('(Do not type out the brackets)')
	print('-----------------Required Argument-----------------')
	print('base_filename: path and prefix for data files, e.g., out/test')
	print('-----------------Formatting Options-----------------')
	print('units=time,length: time may be fs,ps,ns,us,s; length may be um,mm,cm,m,in.')
	print('labels=system: system may be indexed,cart,cyl,sph')
	print('drange=dr: if dr=0, use linear color scale, otherwise log10 scale with dynamic range = dr')
	print('color=cmap: cmap is a string naming any colormap understood by Matplotlib or Mayavi')
	print('origin=x1,x2,x3: reference all spatial coordinates to this point')
	print('res: resolution points for eikonal fields, no spaces, e.g., res=200')
	print('-----------------Indexing-----------------')
	print('i,j,k should be replaced by a 1-digit hexidecimal number from 0-b.')
	print('0-3 maps to (ct,x,y,z).')
	print('4-7 maps to (w/c,kx,ky,kz).')
	print('8-b maps to (phase,ax,ay,az).')
	print('-----------------Plot Types-----------------')
	print('oij[=h0,h1,v0,v1]: plot orbits in ij plane, e.g., o31')
	print('hij: scatter plot in Hamiltonian phase space of the final state, e.g., h15')
	print('fijk: field reconstruction for component k in the ij plane (requires detailed orbits)')
	print('o3d: 3D plot of ray orbits')
	print('detector_name[=i,j,k,f]: field reconstruction in detection plane of the named detector')
	print('  i,j optionally select a plane in case more than one is available, with k the slice')
	print('  f is the fraction of the domain to display')
	print('  (fields may be eikonal or full wave depending on detector)')
	print('mesh: display all surface meshes (may be very slow)')
	print('==============END HELP FOR SEARAY PLOTTER==============')
	exit(1)

basename = sys.argv[1]
simname = basename.split('/')[1]
os.chdir(basename.split('/')[0])

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
except:
	my_color_map = plotter_defaults['image colors']

try:
	dynamic_range = np.float(arg_dict['drange'])
except:
	dynamic_range = 0.0

def SliceAxis(h,v):
	if h!=1 and v!=1:
		return 1
	if h!=2 and v!=2:
		return 2
	if h!=3 and v!=3:
		return 3

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

def CartesianReduce(ax,dom,hfrac,vfrac):
	hc = np.int(ax.shape[0]/2)
	vc = np.int(ax.shape[1]/2)
	dh = np.int(ax.shape[0]*hfrac/2)
	dv = np.int(ax.shape[1]*vfrac/2)
	dom = np.copy(dom)
	dom[:4] *= np.array([hfrac,hfrac,vfrac,vfrac])
	return ax[hc-dh:hc+dh,vc-dv:vc+dv,:],dom

def RadialReduce(ax,dom,frac):
	dh = np.int(ax.shape[0]*frac)
	dom = np.copy(dom)
	dom[:2] *= frac
	return ax[:dh,...],dom

def GetPrefixList(postfix):
	ans = []
	for f in glob.glob('*_*_'+postfix+'.npy'):
		ans.append(f[:-len(postfix)-4])
	return ans

class Units:
	def __init__(self,label_type):
		mks_length = inputs.sim[0]['mks_length']
		mks_time = inputs.sim[0]['mks_time']
		try:
			t_str = arg_dict['units'].split(',')[0]
			l_str = arg_dict['units'].split(',')[1]
		except:
			t_str = plotter_defaults['time']
			l_str = plotter_defaults['length']

		carrier = inputs.wave[0]['k0'][0]

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
		l_str = l_str.replace('um',r'$\mu$m')

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

		self.length_label = l_str
		self.time_label = t_str
		self.mks_length = mks_length
		self.mks_time = mks_time

		self.normalization = np.concatenate((np.ones(4)*l1,np.ones(4)/carrier,np.ones(4)))
		self.lab_str = [r'$x_0$ ',r'$x_1$ ',r'$x_2$ ',r'$x_3$ ',
			r'$ck_0/\omega$',r'$ck_1/\omega$',r'$ck_2/\omega$',r'$ck_3/\omega$',
			r'$\psi+\omega t$',r'$a_1$',r'$a_2$',r'$a_3$']
		self.lab_str[0] += '('+l_str+')'
		self.lab_str[1] += '('+l_str+')'
		self.lab_str[2] += '('+l_str+')'
		self.lab_str[3] += '('+l_str+')'

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
	def GetNormalization(self):
		return self.normalization
	def GetLabels(self):
		return self.lab_str
	def PlotExt(self,dom,h,v):
		h1 = 2*(h-1)
		h2 = 2*(h-1)+1
		v1 = 2*(v-1)
		v2 = 2*(v-1)+1
		return [dom[h1]*self.normalization[h],dom[h2]*self.normalization[h],dom[v1]*self.normalization[v],dom[v2]*self.normalization[v]]
	def LengthLabel(self):
		return self.length_label
	def GetWcm2(self,a2,w0):
		w0_mks = w0/self.mks_time
		E0_mks = np.sqrt(a2) * C.m_e * w0_mks * C.c / C.e
		eta0 = 1/C.c/C.epsilon_0
		return 1e-4 * 0.5 * E0_mks**2 / eta0


class MeshViewer:
	def __init__(self):
		self.structured_mesh = []
		self.mesh = []
		self.simplex = []
		l = GetPrefixList('simplices')
		for prefix in l:
			self.mesh.append(np.load(prefix+'mesh.npy'))
			self.simplex.append(np.load(prefix+'simplices.npy'))
		l = GetPrefixList('mesh')
		for prefix in l:
			self.structured_mesh.append(np.load(prefix+'mesh.npy'))
	def GetMeshList(self):
		return self.structured_mesh
	def Plot(self,mpl_plot_count,maya_plot_count):
		if 'mesh' in sys.argv:
			# For rotated meshes we have a problem here
			mpl_plot_count += 1
			fig = plt.figure(mpl_plot_count,figsize=(7,6))
			ax = fig.add_subplot(111,projection='3d')
			surf = []
			for j,mesh in enumerate(self.mesh):
				x = mesh[:,:,1].flatten()
				y = mesh[:,:,2].flatten()
				z = mesh[:,:,3].flatten()
				tri = mpl.tri.Triangulation(normalization[1]*x,normalization[2]*y,triangles=self.simplex[j])
				surf.append(ax.plot_trisurf(tri,normalization[3]*z,cmap=plotter_defaults['level colors']))
				#ax.set_zlim(-5,5)
			ax.set_xlabel(lab_str[1])
			ax.set_ylabel(lab_str[2])
			ax.set_zlabel(lab_str[3])
			plt.tight_layout()
		return mpl_plot_count,maya_plot_count

class PhaseSpace:
	def __init__(self):
		self.xp0 = np.load(simname+'_xp0.npy')
		self.xp = np.load(simname+'_xp.npy')
		self.eikonal = np.load(simname+'_eikonal.npy')
		self.eikonal[:,0] += self.xp[:,0,4]*self.xp[:,0,0]
		self.eikonal[:,0] -= np.min(self.eikonal[:,0])
		try:
			self.res = np.int(arg_dict['res'])
		except:
			self.res = 200
	def Plot(self,mpl_plot_count,maya_plot_count):
		for i in range(12):
			for j in range(12):
				plot_key = 'h'+format(i,'01X').lower()+format(j,'01X').lower()
				if plot_key in sys.argv:
					mpl_plot_count += 1
					plt.figure(mpl_plot_count,figsize=(7,6))
					if i<8:
						x = self.xp[:,0,i]
					else:
						x = self.eikonal[:,i-8]
					if j<8:
						y = self.xp[:,0,j]
					else:
						y = self.eikonal[:,j-8]
					weights = self.eikonal[:,1]**2 + self.eikonal[:,2]**2 + self.eikonal[:,3]**2
					sel = np.logical_and(np.logical_not(np.isnan(x)),np.logical_not(np.isnan(y)))
					harray,plot_ext = grid_tools.GridFromBinning(normalization[i]*x[sel],normalization[j]*y[sel],weights[sel],self.res,self.res)
					harray = grid_tools.Smooth1D(harray,4,0)
					harray = grid_tools.Smooth1D(harray,4,1)
					pre_str = TransformColorScale(harray,dynamic_range)
					plt.imshow(harray.swapaxes(0,1),origin='lower',cmap=my_color_map,aspect='auto',extent=plot_ext)
					b=plt.colorbar()
					b.set_label(pre_str+r'$|a|^2$',size=18)
					plt.xlabel(lab_str[i],size=18)
					plt.ylabel(lab_str[j],size=18)
					plt.tight_layout()
		return mpl_plot_count,maya_plot_count

class Orbits:
	def __init__(self,mesh_list):
		self.orbits = np.load(simname+'_orbits.npy')
		# flatten orbit data for field reconstruction (time*ray,component)
		self.xpo = self.orbits.reshape(self.orbits.shape[0]*self.orbits.shape[1],self.orbits.shape[2])
		# extract spatial part of phase (assumes monochromatic mode)
		self.xpo[:,8] += self.xpo[:,4]*self.xpo[:,0]
		self.mesh_list = mesh_list
		self.res = 200
	def Plot(self,mpl_plot_count,maya_plot_count):
		for i in range(12):
			for j in range(12):
				for k in range(12):
					plot_key = 'f'+format(i,'01X').lower()+format(j,'01X').lower()+format(k,'01X').lower()
					if plot_key in sys.argv:
						mpl_plot_count += 1
						plt.figure(mpl_plot_count,figsize=(7,6))
						harray,plot_ext = grid_tools.GridFromInterpolation(self.xpo[:,i],self.xpo[:,j],self.xpo[:,k],self.res,self.res)
						cbar_str = TransformColorScale(harray,dynamic_range)
						plt.imshow(harray.swapaxes(0,1),origin='lower',cmap=my_color_map,aspect='auto',extent=units.PlotExt(plot_ext,i,j))
						b=plt.colorbar()
						b.set_label(cbar_str+lab_str[k],size=18)
						plt.xlabel(lab_str[i],size=18)
						plt.ylabel(lab_str[j],size=18)
						plt.tight_layout()
		if 'o3d' in sys.argv:
			if maya_loaded:
				maya_plot_count += 1
				for j in range(self.orbits.shape[1]):
					x = normalization[1]*self.orbits[:,j,1]
					y = normalization[2]*self.orbits[:,j,2]
					z = normalization[3]*self.orbits[:,j,3]
					s = self.orbits[:,j,9]**2 + self.orbits[:,j,10]**2 + self.orbits[:,j,11]**2
					xtest = np.roll(x,1)!=x
					ytest = np.roll(y,1)!=y
					ztest = np.roll(z,1)!=z
					test = np.logical_or(np.logical_or(xtest,ytest),ztest)
					x = x[np.where(test)]
					y = y[np.where(test)]
					z = z[np.where(test)]
					s = s[np.where(test)]
					if x.shape[0]>0:
						characteristic_size = np.max([np.max(x)-np.min(x),np.max(y)-np.min(y),np.max(z)-np.min(z)])
						mlab.plot3d(x,y,z,s,tube_radius=.001*characteristic_size)
				for mesh in self.mesh_list:
					c = mesh[:,:,0]
					x = mesh[:,:,1]
					y = mesh[:,:,2]
					z = mesh[:,:,3]
					mlab.mesh(x*normalization[1],y*normalization[2],z*normalization[3],color=(0.5,1,0.5),opacity=0.5)
			if mpl_loaded and not maya_loaded:
				mpl_plot_count += 1
				fig = plt.figure(mpl_plot_count,figsize=(7,6))
				ax = fig.add_subplot(111,projection='3d')
				for j in range(self.orbits.shape[1]):
					ax.plot(normalization[1]*self.orbits[:,j,1],normalization[2]*self.orbits[:,j,2],normalization[3]*self.orbits[:,j,3])
				surf = []
				cmap = mpl.cm.ScalarMappable(cmap=plotter_defaults['level colors'])
				for mesh in self.mesh_list:
					c = cmap.to_rgba(mesh[:,:,0]*normalization[3])
					x = mesh[:,:,1]
					y = mesh[:,:,2]
					z = mesh[:,:,3]
					#surf.append(ax.plot_surface(x*normalization[1],y*normalization[2],z*normalization[3],rcount=16,ccount=16,facecolors=c))
					surf.append(ax.plot_surface(x*normalization[1],y*normalization[2],z*normalization[3],facecolors=c))
					cmap.set_array(mesh[:,:,0]*normalization[3])
				ax.set_xlabel(lab_str[1])
				ax.set_ylabel(lab_str[2])
				ax.set_zlabel(lab_str[3])
				if len(self.mesh_list)>0 and np.max(mesh[:,:,0])!=np.min(mesh[:,:,0]):
					fig.colorbar(cmap, shrink=0.5, aspect=10, label='height ('+units.LengthLabel()+')')
				plt.tight_layout()
		for i in range(12):
			for j in range(12):
				plot_key = 'o'+format(i,'01X').lower()+format(j,'01X').lower()
				try:
					arg = arg_dict[plot_key]
				except:
					arg = 'stop'
				if arg!='stop':
					mpl_plot_count += 1
					plt.figure(mpl_plot_count,figsize=(4,4))
					center = (origin[i],origin[j])
					if arg!='default':
						lims = (np.float(arg.split(',')[0]),np.float(arg.split(',')[1]),np.float(arg.split(',')[2]),np.float(arg.split(',')[3]))
						plt.xlim(lims[0]-center[0],lims[1]-center[0])
						plt.ylim(lims[2]-center[1],lims[3]-center[1])
					for k in range(self.orbits.shape[1]):
						plt.plot(normalization[i]*self.orbits[:,k,i]-center[0],normalization[j]*self.orbits[:,k,j]-center[1],'k')
					plt.xlabel(lab_str[i],size=18)
					plt.ylabel(lab_str[j],size=18)
					plt.tight_layout()
		return mpl_plot_count,maya_plot_count

class EikonalWaveProfiler:
	def __init__(self):
		self.name = []
		self.xp = []
		self.eik = []
		l = GetPrefixList('xps')
		for prefix in l:
			self.name.append(prefix.split('_')[-2])
			self.xp.append(np.load(prefix+'xps.npy'))
			self.eik.append(np.load(prefix+'eiks.npy'))
		try:
			self.res = np.int(arg_dict['res'])
		except:
			self.res = 200
		print('eikonal detectors =',self.name)
	def Plot(self,mpl_plot_count,maya_plot_count):
		for det_idx,eik in enumerate(self.name):
			if eik in sys.argv:
				xps = self.xp[det_idx]
				eiks = self.eik[det_idx]
				mpl_plot_count += 1
				plt.figure(eik,figsize=(6,7))
				phase = eiks[:,0] + xps[:,4]*xps[:,0]
				phase -= np.min(phase)

				plt.subplot(211)
				a1,plot_ext = grid_tools.GridFromInterpolation(xps[:,1],xps[:,2],eiks[:,1],self.res,self.res)
				a2,plot_ext = grid_tools.GridFromInterpolation(xps[:,1],xps[:,2],eiks[:,2],self.res,self.res)
				a3,plot_ext = grid_tools.GridFromInterpolation(xps[:,1],xps[:,2],eiks[:,3],self.res,self.res)
				intens = units.GetWcm2(a1**2+a2**2+a3**2,1.0)
				cbar_str = TransformColorScale(intens,dynamic_range)
				plt.imshow(intens.swapaxes(0,1),origin='lower',cmap=my_color_map,aspect='auto',extent=units.PlotExt(plot_ext,1,2))
				b=plt.colorbar()
				b.set_label(cbar_str + r'Intensity (W/cm$^2$)',size=18)
				plt.xlabel(lab_str[1],size=18)
				plt.ylabel(lab_str[2],size=18)

				plt.subplot(212)
				psi,plot_ext = grid_tools.GridFromInterpolation(xps[:,1],xps[:,2],phase,self.res,self.res,fill=np.min(phase))
				plt.imshow(psi.swapaxes(0,1),origin='lower',cmap=my_color_map,aspect='auto',extent=units.PlotExt(plot_ext,1,2))
				b=plt.colorbar()
				b.set_label(r'$\psi + \omega t$',size=18)
				plt.xlabel(lab_str[1],size=18)
				plt.ylabel(lab_str[2],size=18)

				plt.tight_layout()
		return mpl_plot_count,maya_plot_count

class PlaneWaveProfiler:
	def __init__(self):
		self.name = []
		self.eik = []
		self.wave = []
		self.ext = []
		l = GetPrefixList('plane_wave')
		for prefix in l:
			self.name.append(prefix.split('_')[-2])
			self.wave.append(np.load(prefix+'plane_wave.npy'))
			self.eik.append(np.load(prefix+'plane_eik.npy'))
			self.ext.append(np.load(prefix+'plane_plot_ext.npy'))
		print('plane wave detectors =',self.name)
	def Plot(self,mpl_plot_count,maya_plot_count):
		for det_idx,pw in enumerate(self.name):
			try:
				arg = arg_dict[pw]
				dom = self.ext[det_idx]
				if arg=='default':
					haxis=1
					vaxis=2
					slice_idx=0
					wave_zone_fraction = 1.0
				else:
					haxis = np.int(arg.split(',')[0])
					vaxis = np.int(arg.split(',')[1])
					slice_idx = np.int(arg.split(',')[2])
					wave_zone_fraction = np.double(arg.split(',')[3])
				mpl_plot_count += 1
				plt.figure(mpl_plot_count,figsize=(5,7))
				plt.subplot(211)
				A = self.eik[det_idx]
				A2 = np.abs(A[...,0])**2 + np.abs(A[...,1])**2 + np.abs(A[...,2])**2
				cbar_str = TransformColorScale(A2,dynamic_range)
				plt.imshow(A2.swapaxes(0,1),origin='lower',cmap=my_color_map,aspect='auto',extent=units.PlotExt(dom,1,2))
				b=plt.colorbar()
				b.set_label(cbar_str+r'$|a|^2(z_{\rm eik})$',size=18)
				plt.xlabel(lab_str[1],size=18)
				plt.ylabel(lab_str[2],size=18)
				plt.subplot(212)
				A = self.wave[det_idx]
				A2 = np.abs(A[...,0])**2 + np.abs(A[...,1])**2 + np.abs(A[...,2])**2
				A2,dom = CartesianReduce(A2,dom,wave_zone_fraction,wave_zone_fraction)
				A2 = np.squeeze(np.take(A2,[slice_idx],axis=SliceAxis(haxis,vaxis)-1),axis=SliceAxis(haxis,vaxis)-1)
				if haxis<vaxis:
					A2 = A2.swapaxes(0,1)
				cbar_str = TransformColorScale(A2,dynamic_range)
				plt.imshow(A2,origin='lower',cmap=my_color_map,aspect='auto',extent=units.PlotExt(dom,haxis,vaxis))
				plt.xlabel(lab_str[haxis],size=18)
				plt.ylabel(lab_str[vaxis],size=18)
				b=plt.colorbar()
				b.set_label(cbar_str+r'$|a|^2$',size=18)
				plt.tight_layout()
			except KeyError:
				print('INFO:',pw,'not used.')
		return mpl_plot_count,maya_plot_count

class BesselBeamProfiler:
	def __init__(self):
		self.name = []
		self.eik = []
		self.wave = []
		self.ext = []
		l = GetPrefixList('bess_wave')
		for prefix in l:
			self.name.append(prefix.split('_')[-2])
			self.wave.append(np.load(prefix+'bess_wave.npy'))
			self.eik.append(np.load(prefix+'bess_eik.npy'))
			self.ext.append(np.load(prefix+'bess_plot_ext.npy'))
		print('Bessel beam detectors =',self.name)
	def Plot(self,mpl_plot_count,maya_plot_count):
		for det_idx,bess in enumerate(self.name):
			try:
				arg = arg_dict[bess]
				dom = self.ext[det_idx]
				if arg.split(',')[0]=='3d':
					lab_str = cunits.GetLabels()
					wave_zone_fraction = np.double(arg.split(',')[1])
					mpl_plot_count += 1
					A = self.wave[det_idx]
					A2 = np.abs(A[...,0])**2 + np.abs(A[...,1])**2 + np.abs(A[...,2])**2
					A2,plot_ext = RadialReduce(A2,dom[:4],wave_zone_fraction)
					cbar_str = TransformColorScale(A2,dynamic_range)
					fig = plt.figure(mpl_plot_count,figsize=(8,8))
					ax = fig.add_subplot(111,projection='3d')
					z_list = np.linspace(dom[4],dom[5],A2.shape[2])
					rho_list = np.linspace(dom[0],dom[1],A2.shape[0])
					phi_list = np.linspace(0,2*np.pi*(1-1/A2.shape[1]),A2.shape[1])
					surf = []
					cmap = mpl.cm.ScalarMappable(cmap=my_color_map)
					for i,phi in enumerate(phi_list):
						x = np.outer(rho_list*np.cos(phi),np.ones(A2.shape[2]))
						y = np.outer(rho_list*np.sin(phi),np.ones(A2.shape[2]))
						z = np.outer(np.ones(A2.shape[0]),z_list)
						c = cmap.to_rgba(A2[:,i,:])
						surf.append(ax.plot_surface(x*normalization[1],y*normalization[2],z*normalization[3],facecolors=c))
					cmap.set_array(A2[:,i,:])
					plt.colorbar(cmap,shrink=0.5, aspect=10, label=cbar_str+r'$|a|^2$')
					ax.set_xlabel(lab_str[1])
					ax.set_ylabel(lab_str[2])
					ax.set_zlabel(lab_str[3])
					plt.tight_layout()
				else:
					if arg=='default':
						haxis=1
						vaxis=2
						slice_idx=0
						wave_zone_fraction = 1.0
					else:
						haxis = np.int(arg.split(',')[0])
						vaxis = np.int(arg.split(',')[1])
						slice_idx = np.int(arg.split(',')[2])
						wave_zone_fraction = np.double(arg.split(',')[3])
					lab_str = cunits.GetLabels()
					mpl_plot_count += 1
					plt.figure(mpl_plot_count,figsize=(5,7))
					plt.subplot(211)
					A = self.eik[det_idx]
					A2 = np.abs(A[...,0])**2 + np.abs(A[...,1])**2 + np.abs(A[...,2])**2
					cbar_str = TransformColorScale(A2,dynamic_range)
					plt.imshow(A2.swapaxes(0,1),origin='lower',cmap=my_color_map,aspect='auto',extent=cunits.PlotExt(dom,1,2))
					b=plt.colorbar()
					b.set_label(cbar_str+r'$|a|^2(z_E)$',size=18)
					plt.xlabel(lab_str[1],size=18)
					plt.ylabel(lab_str[2],size=18)
					plt.subplot(212)
					A = self.wave[det_idx]
					A2 = np.abs(A[...,0])**2 + np.abs(A[...,1])**2 + np.abs(A[...,2])**2
					A2,dom = RadialReduce(A2,dom,wave_zone_fraction)
					A2 = np.squeeze(np.take(A2,[slice_idx],axis=SliceAxis(haxis,vaxis)-1),axis=SliceAxis(haxis,vaxis)-1)
					if haxis<vaxis:
						A2 = A2.swapaxes(0,1)
					cbar_str = TransformColorScale(A2,dynamic_range)
					plt.imshow(A2,origin='lower',cmap=my_color_map,aspect='auto',extent=cunits.PlotExt(dom,haxis,vaxis))
					plt.xlabel(lab_str[haxis],size=18)
					plt.ylabel(lab_str[vaxis],size=18)
					b=plt.colorbar()
					b.set_label(cbar_str+r'$|a|^2$',size=18)
					plt.tight_layout()
			except KeyError:
				print('INFO:',bess,'not used.')
		return mpl_plot_count,maya_plot_count

mpl_plot_count = 0
maya_plot_count = 0

try:
	label_type = arg_dict['labels']
except:
	label_type = 'indexed'

try:
	origin = arg_dict['origin'].split(',')
	origin = np.array([0.0,np.double(origin[0]),np.double(origin[1]),np.double(origin[2]),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
except:
	origin = np.zeros(12)

units = Units(label_type)
normalization = units.GetNormalization()
lab_str = units.GetLabels()
cunits = Units('cyl')

meshPlots = MeshViewer()
mpl_plot_count,maya_plot_count = meshPlots.Plot(mpl_plot_count,maya_plot_count)

orbitPlots = Orbits(meshPlots.GetMeshList())
mpl_plot_count,maya_plot_count = orbitPlots.Plot(mpl_plot_count,maya_plot_count)

phaseSpacePlots = PhaseSpace()
mpl_plot_count,maya_plot_count = phaseSpacePlots.Plot(mpl_plot_count,maya_plot_count)

eikonalPlots = EikonalWaveProfiler()
mpl_plot_count,maya_plot_count = eikonalPlots.Plot(mpl_plot_count,maya_plot_count)

planePlots = PlaneWaveProfiler()
mpl_plot_count,maya_plot_count = planePlots.Plot(mpl_plot_count,maya_plot_count)

besselPlots = BesselBeamProfiler()
mpl_plot_count,maya_plot_count = besselPlots.Plot(mpl_plot_count,maya_plot_count)

if maya_loaded and maya_plot_count>0:
	mlab.show()

if mpl_loaded and mpl_plot_count>0:
	plt.show()
