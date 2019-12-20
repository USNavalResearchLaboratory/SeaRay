#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef double2 tw_Complex;

tw_Complex cmul(tw_Complex a,tw_Complex b)
{
	return (tw_Complex)(a.s0*b.s0-a.s1*b.s1,a.s0*b.s1+a.s1*b.s0);
}

tw_Complex cexp(tw_Complex x)
{
	return (tw_Complex)(cos(x.s1)*exp(-x.s0),sin(x.s1)*exp(-x.s0));
}

__kernel void ReducedPotentialToField(__global double * q,__global double * kz,__global double * kg,__global double * w,const double z)
{
	// POINT PROTOCOL
	// q = [w][x][y][2]
	// Forming E = iw*A = iw*exp(i*(kz+kg)*z)*q
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = 2*(i0*Nj*Nk + j0*Nk + k0);
	const tw_Complex Q = (tw_Complex)(q[idx],q[idx+1]);
	const double Kr = kz[idx];
	const double Ki = kz[idx+1];
	const tw_Complex iKz = (tw_Complex)(-Ki*z,(Kr+kg[i0])*z);
	const tw_Complex iw = (tw_Complex)(0.0,w[i0]);
	const tw_Complex ans = cmul(iw,cmul(cexp(iKz),Q));
	q[idx] = ans.s0;
	q[idx+1] = ans.s1;
}

__kernel void CurrentToODERHS(__global double * J,__global double * kz,__global double * kg,const double z)
{
	// POINT PROTOCOL
	// J = [w][x][y][2]
	// Forming S = 0.5i*np.exp(-i*kz*z)*J/real(kz)
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = 2*(i0*Nj*Nk + j0*Nk + k0);
	const tw_Complex j = (tw_Complex)(J[idx],J[idx+1]);
	const double Kr = kz[idx];
	const double Ki = kz[idx+1];
	const tw_Complex iKz = (tw_Complex)(-Ki*z,(Kr+kg[i0])*z);
	const tw_Complex i2 = (tw_Complex)(0.0,0.5);
	const tw_Complex ans = cmul(i2,cmul(cexp(-iKz),j))/Kr;
	J[idx] = ans.s0;
	J[idx+1] = ans.s1;
}

__kernel void PolarizationToCurrent(__global double * A,__global double * w)
{
	// SPECTRAL POINT PROTOCOL
	// A = [w][x][y][2]
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = 2*(i0*Nj*Nk + j0*Nk + k0);
	double Ar = A[idx];
	double Ai = A[idx+1];
	double w0 = w[i0];
	A[idx] = w0*Ai;
	A[idx+1] = -w0*Ar;
}

__kernel void ComputeRate(
	__global double * rate,
	__global double * E,
	const double E_conv,
	const double rate_conv,
	const double E_cutoff,
	const double C_pre,
	const double C_pow,
	const double C_exp)
{
	// TEMPORAL POINT PROTOCOL
	// rate,E are 3d arrays with [t][x][y]
	// E in simulation units, rate output in simulation units
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const double Emag = E_conv*fabs(E[idx]) + E_cutoff;
	rate[idx] = rate_conv*C_pre*pow(Emag,C_pow)*exp(C_exp/Emag);
}

__kernel void ComputePlasmaDensity(
	__global double * ne,
	__global double * ng,
	const double refOnCrit,
	const double dt,
	const int steps)
{
	// STRIP PROTOCOL
	// ne = [t][x][y]
	// On input ne contains rate, on output electron density
	// ng = [x][y], normalized to ref density
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int Nx = get_global_size(0);
	const int Ny = get_global_size(1);
	const int offset = j0 + i0*Ny;
	const int s = Nx*Ny;

	double prev_rate = 0.0;
	double curr_rate = 0.0;
	double ne_raw = 0.0;
	ne[offset+(steps-1)*s] = 0.0;
	for (int k=steps-1;k>=0;--k)
	{
		curr_rate = ne[offset+k*s];
		ne_raw += 0.5*(curr_rate+prev_rate)*dt;
		ne[offset+k*s] = refOnCrit*ng[offset]*(1.0-exp(-ne_raw));
		prev_rate = curr_rate;
	}
}

__kernel void ComputePlasmaPolarization(
	__global double * P,
	__global double * E,
	__global double * ne,
	const double Omega,
	const double gamma,
	const double dt,
	const int steps)
{
	// STRIP PROTOCOL
	// P,F are 3d arrays with [t][x][y]
	// P is expected to be zeros
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int Nx = get_global_size(0);
	const int Ny = get_global_size(1);
	const int offset = j0 + i0*Ny;
	const int s = Nx*Ny;

	for (int k=steps-8;k>=0;--k)
	{
		P[offset+k*s] = dt*dt*ne[offset+k*s+s]*E[offset+k*s+s] + (2.0+dt*gamma)*P[offset+k*s+s] - P[offset+k*s+2*s];
		P[offset+k*s] /= 1.0 + dt*gamma + dt*dt*Omega*Omega;
	}
}

__kernel void AddKerrPolarization(__global double * P, __global double * E,const double chi3)
{
	// TEMPORAL POINT PROTOCOL
	// P = [t][x][y]
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const double Ei = E[idx];
	P[idx] += chi3*Ei*Ei*Ei;
}
