#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

__kernel void PotentialToField(__global double * A,__global double * w)
{
	// POINT PROTOCOL
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
	A[idx] = -w0*Ai;
	A[idx+1] = w0*Ar;
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
