import sys
import numpy as np
import pyopencl
import matplotlib.pyplot as plt
import inputs
import init
import grid_tools
import paraxial_kernel

if len(sys.argv)==1:
	print('Usage: paraxial.py cmd [device=string] [platform=string] [iterations=n]')
	print('cmd = list --- displays all platforms and devices')
	print('cmd = run --- executes calculation')
	print('string = something in desired device/platform name, or numerical id')
	print('large numbers (>10) are assumed to be part of a name rather than an id')
	print('defaults are the last platform/device in the list')
	print('iterations = iterations to use in optimization (default=1)')
	exit(1)

# Set up OpenCL
queue,program,args = init.setup_opencl(sys.argv,'caustic.cl','')

# Run one case for each set of dictionaries
a0_list = np.zeros(len(inputs.sim))
a2_ratio = np.zeros(len(inputs.sim))
for i in range(len(inputs.sim)):

	print('Begin run',i)
	Nr = 256
	Nt = 512
	dr = 200*np.pi
	dt = 2*np.pi
	rho00 = 50*dr
	w00 = 1.0
	zR = 0.5*rho00**2
	t_list = grid_tools.cell_centers(-dt*Nt/2,dt*Nt/2,Nt)
	r_list = grid_tools.cell_centers(0,dr*Nr,Nr)
	w_list = w00 + paraxial_kernel.FFT_w(1/dt,Nt)
	psi = np.outer(np.ones(Nr),t_list*w00)
	rho = np.outer(r_list,np.ones(Nt))
	wp2 = paraxial_kernel.plasma_channel_dens(r_list,0.0,zperiod=100.0/inputs.sim[i]['mks_length'],ne0=0.01,nc=1.0)
	chiw = -np.outer(wp2,1/w_list**2)
	n0 = np.sqrt(1 + chiw[0,0])
	ng = 1/n0
	chiw0 = np.array(chiw[0,:])
	chiw -= np.outer(np.ones(Nr),chiw0)
	a = inputs.wave[i]['a0']*paraxial_kernel.H00_envelope(psi,rho,z=0.0,tau0=100e-15/inputs.sim[i]['mks_time'],
		rho0=rho00,w0=w00,k0=w00*n0,
		pulse_shape=inputs.wave[i]['pulse shape'])
	H = paraxial_kernel.HankelTransformTool(Nr,dr)
	
	print('a2(0) =',np.max(np.abs(a)**2))
	print('action(0) =',paraxial_kernel.wave_action(a,dr,dt))
	np.save('out/chiw',chiw)
	np.save('out/chiw0',chiw0)

	z = 0.0
	dz = .696/inputs.sim[i]['mks_length']/20
	diagnostic_period = 1
	for iter in range(20):
		a = paraxial_kernel.propagator(H,a,chiw0,chiw,n0,ng,w00,dr,dt,dz)
		if iter%diagnostic_period==0:
			np.save('out/amp'+str(np.int(iter/diagnostic_period)),a)
		z += dz

	a0_list[i] = inputs.wave[i]['a0']
	a2_ratio[i] = (np.max(np.abs(a))/inputs.wave[i]['a0'])**2
	print('a2(t) =',np.max(np.abs(a)**2))
	print('action(t) =',paraxial_kernel.wave_action(a,dr,dt))
plt.plot(a0_list,a2_ratio)
plt.show()
