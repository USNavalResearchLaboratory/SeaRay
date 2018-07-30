import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import inputs
import caustic_tools
import init
from scipy.special import erf
from scipy.special import erfi
from scipy import constants as C
eta0 = np.sqrt(C.mu_0/C.epsilon_0)

def paraxial_ax(rho,z,zR,w00):
	Rci = z/(z**2+zR**2)
	rho0 = np.sqrt(2.0*zR/w00)
	rhoz = rho0*np.sqrt(1+z**2/zR**2)
	Phi = w00*z + 0.5*w00*rho**2*Rci-np.arctan(z/zR)
	return np.exp(-rho**2/rhoz**2)*(rho0/rhoz)*np.exp(1j*Phi)

def paraxial_az(rho,z,zR,w00):
	Rci = z/(z**2+zR**2)
	rho0 = np.sqrt(2.0*zR/w00)
	rhoz = rho0*np.sqrt(1+z**2/zR**2)
	Phi = w00*z + 0.5*w00*rho**2*Rci-np.arctan(z/zR)
	return -(Rci+2j/(w00*rhoz**2))*rho*np.exp(-rho**2/rhoz**2)*(rho0/rhoz)*np.exp(1j*Phi)

err_plot=True
mode_plot=False
amp_plot=True
phase_plot=True

if len(sys.argv)==1:
	print('Usage: gaussian.py run device=[n] platform=[n] rho_inc[m] f_parabola[m] r_diag[m]')
	exit(1)

# Set up OpenCL
queue,program,args = init.setup_opencl(sys.argv,'caustic.cl','')

dynamic_range = 5
#my_color_map = 'nipy_spectral'
my_color_map = 'Accent'
#my_color_map = 'jet'

# units
l1_mks = 0.8e-6/(2*np.pi)
l1_um = l1_mks*1e6

# command line arguments
rho_inc_mks = np.double(args[1])
f_par_mks = np.double(args[2])
r_diag_mks = np.double(args[3])

# pulse metrics
r00 = rho_inc_mks/l1_mks
f = f_par_mks/l1_mks
rd = r_diag_mks/l1_mks # radius of sphere within which fields are computed
rs = f/4 # radius of sphere with boundary data
print(rs)
w00 = 1.0
k00 = 1.0
P_watts = 1e15
intensity_inc_mks = 2*P_watts/(np.pi*r00**2*l1_mks**2)
E_peak_inc_mks = np.sqrt(2*eta0*intensity_inc_mks)
q_pts = 512
modes = 32
fig_num = 0

# Compute paraxial mode parameters for comparison
f_num = 0.5*f/r00
paraxial_e_size = 4.0*f_num/k00
paraxial_zR = 0.5*k00*paraxial_e_size**2
paraxial_intensity_mks = 2.0*P_watts/(np.pi*paraxial_e_size**2*l1_mks**2)
paraxial_peak_E_mks = np.sqrt(paraxial_intensity_mks*2*eta0)
print('f/# =',f_num)
print('Theoretical paraxial spot size (um) =',l1_um*paraxial_e_size)
print('Theoretical paraxial Rayleigh length (um) =',l1_um*paraxial_zR)
print('Power (TW) =',1e-12*P_watts)
print('Incoming intensity (W/cm2) =',intensity_inc_mks*1e-4)
print('Theoretical peak intensity (W/cm2) =',paraxial_intensity_mks*1e-4)

field_tool = caustic_tools.SphericalHarmonicTool(q_pts,q_pts,modes,q_pts,queue,program.transform)
q_list,r_list,Ex,Ez,Ex_modes,Ez_modes = field_tool.GetGaussianFields(E_peak_inc_mks,r00,f,k00,rs,rd)

intensity_mks = 0.5*(np.abs(Ex)**2 + np.abs(Ez)**2)/eta0
peak_E = np.max(np.sqrt(intensity_mks*2*eta0))
rms_waist = caustic_tools.get_waist(r_list,intensity_mks,1)
print('Peak Intensity (W/cm2) =',1e-4*np.max(intensity_mks))
print('RMS Size (um) =',l1_um*rms_waist)
print('1/e Size (2*rms) =',l1_um*2*rms_waist)

# Put results on cylindrical grid
rho_list,z_list,Ex = caustic_tools.spherical_to_cylindrical(Ex,q_list,r_list,q_pts,q_pts)
rho_list,z_list,Ez = caustic_tools.spherical_to_cylindrical(Ez,q_list,r_list,q_pts,q_pts)
plot_ext = [-rd*l1_um,rd*l1_um,0,rd*l1_um]
intensity_mks = 0.5*(np.abs(Ex)**2 + np.abs(Ez)**2)/eta0

rho = np.outer(rho_list,np.ones(q_pts))
z = np.outer(np.ones(q_pts),z_list)

phase = np.angle(1e-10+Ex)
slow_phase_z = np.angle(1e-10+1j*Ez*np.exp(-1j*k00*z))
slow_phase = np.angle(1e-10+Ex*np.exp(-1j*k00*z))
Ex_par = paraxial_peak_E_mks*paraxial_ax(rho,z,paraxial_zR,k00)
Ez_par = paraxial_peak_E_mks*paraxial_az(rho,z,paraxial_zR,k00)
slow_phase_par = np.angle(1e-10+Ex_par*np.exp(-1j*k00*z))
slow_phase_par_z = np.angle(1e-10+1j*Ez_par*np.exp(-1j*k00*z))
caustic_tools.SphericalClipping(Ex_par,rho_list[1]-rho_list[0],z_list[1]-z_list[0],0)
caustic_tools.SphericalClipping(Ez_par,rho_list[1]-rho_list[0],z_list[1]-z_list[0],0)
caustic_tools.SphericalClipping(slow_phase_par,rho_list[1]-rho_list[0],z_list[1]-z_list[0],0)
caustic_tools.SphericalClipping(slow_phase_par_z,rho_list[1]-rho_list[0],z_list[1]-z_list[0],0)

def ErrorMeasures(Ex,Ez):
	drho = (rho_list[1]-rho_list[0])
	dz = (z_list[1]-z_list[0])
	Ech = np.sqrt(np.abs(Ex)**2 + np.abs(Ez)**2)
	Ech += np.max(Ech)/100
	# Divergence Error
	div_E = caustic_tools.GetDivergence(Ex,Ez,drho,dz,queue,program.divergence)
	div_err = div_E/(k00 * Ech)
	caustic_tools.SphericalClipping(div_err,drho,dz,4)
	caustic_tools.AxisClipping(div_err)
	# Laplacian Error
	del2_Ex = caustic_tools.GetLaplacian(Ex,drho,dz,np.double(0.0),queue,program.laplacian)
	del2_Ez = caustic_tools.GetLaplacian(Ez,drho,dz,np.double(1.0),queue,program.laplacian)
	Helm_x = del2_Ex + k00**2*Ex
	Helm_z = del2_Ez + k00**2*Ez
	helm_err = np.sqrt(Helm_x**2 + Helm_z**2)/(k00**2 * Ech)
	caustic_tools.SphericalClipping(helm_err,drho,dz,4)
	caustic_tools.AxisClipping(helm_err)
	return div_err,helm_err

# Plot Error Measures
if err_plot==True:
	div_err,helm_err = ErrorMeasures(Ex,Ez)
	print('Maximum error =',np.max([np.abs(div_err),np.abs(helm_err)]))
	fig_num += 1
	plt.figure(fig_num,figsize=(6,8))
	plt.subplot(211)
	plt.imshow(np.abs(div_err),origin='lower',cmap='jet',aspect='equal',extent=plot_ext)
	b=plt.colorbar()
	b.set_label(r'$|\nabla\cdot {\bf E}/kE|$',size=18)
	plt.xlabel(r'$z$ (um)',size=18)
	plt.ylabel(r'$\rho$ (um)',size=18)
	plt.subplot(212)
	plt.imshow(np.abs(helm_err),origin='lower',cmap='jet',aspect='equal',extent=plot_ext)
	b=plt.colorbar()
	b.set_label(r'$|\nabla^2 {\bf E} + k^2{\bf E}|/k^2E$',size=18)
	plt.xlabel(r'$z$ (um)',size=18)
	plt.ylabel(r'$\rho$ (um)',size=18)


# Plot Mode Coefficients
if mode_plot==True:
	fig_num += 1
	plt.figure(fig_num,figsize=(10,5))
	plt.subplot(211)
	plt.plot(range(modes),np.abs(Ex_modes))
	plt.xlabel(r'mode',size=18)
	plt.ylabel(r'amplitude',size=18)
	plt.subplot(212)
	plt.plot(range(modes),np.abs(Ez_modes))
	plt.xlabel(r'mode',size=18)
	plt.ylabel(r'amplitude',size=18)
	plt.tight_layout()

# Plot fields

# Amplitude Plots
if amp_plot==True:
	fig_num += 1
	plt.figure(fig_num,figsize=(6,8))
	plt.subplot(211)
	#plt.imshow(intensity_mks*1e-4,origin='lower',cmap=my_color_map,aspect='equal',extent=plot_ext)
	plt.imshow(np.real(Ex)*1e-12,origin='lower',vmin=-2500,vmax=2500,cmap=my_color_map,aspect='equal',extent=plot_ext)
	b=plt.colorbar()
	b.set_label(r'$\Re\left\{E_x\right\}$ (TV/m)',size=18)
	#b.set_label(r'Intensity (W$/$cm$^2$)',size=18)
	plt.xlabel(r'$z$ (um)',size=18)
	plt.ylabel(r'$\rho$ (um)',size=18)

	plt.subplot(212)
	plt.imshow(np.real(Ez)*1e-12,origin='lower',cmap=my_color_map,aspect='equal',extent=plot_ext)
	b=plt.colorbar()
	b.set_label(r'$\Re\left\{E_z\right\}/\cos\varphi$ (TV/m)',size=18)
	plt.xlabel(r'$z$ (um)',size=18)
	plt.ylabel(r'$\rho$ (um)',size=18)

	fig_num += 1
	plt.figure(fig_num,figsize=(6,8))
	plt.subplot(211)
	plt.imshow(np.real(Ex_par)*1e-12,vmin=-2500,vmax=2500,origin='lower',cmap=my_color_map,aspect='equal',extent=plot_ext)
	b=plt.colorbar()
	b.set_label(r'$\Re\left\{E_x\right\}$ (TV/m)',size=18)
	#b.set_label(r'Intensity (W$/$cm$^2$)',size=18)
	plt.xlabel(r'$z$ (um)',size=18)
	plt.ylabel(r'$\rho$ (um)',size=18)

	plt.subplot(212)
	plt.imshow(np.real(Ez_par)*1e-12,origin='lower',cmap=my_color_map,aspect='equal',extent=plot_ext)
	b=plt.colorbar()
	b.set_label(r'$\Re\left\{E_z\right\}/\cos\varphi$ (TV/m)',size=18)
	plt.xlabel(r'$z$ (um)',size=18)
	plt.ylabel(r'$\rho$ (um)',size=18)

# Phase Plots
if phase_plot==True:
	fig_num += 1
	plt.figure(fig_num,figsize=(10,5))

	vidx = 64
	plt.subplot(121)
	plt.plot(l1_um*z_list,slow_phase[vidx,:],'k-')
	plt.plot(l1_um*z_list[1:-1],slow_phase_par[vidx,1:-1],'b--')
	plt.xlabel(r'$z$ (um)',size=18)
	plt.ylabel(r'x-phase',size=18)

	plt.subplot(122)
	plt.plot(l1_um*z_list,slow_phase_z[vidx,:])
	plt.plot(l1_um*z_list[1:-1],slow_phase_par_z[vidx,1:-1],'b--')
	plt.xlabel(r'$z$ (um)',size=18)
	plt.ylabel(r'z-phase',size=18)

	plt.tight_layout()

plt.show()
