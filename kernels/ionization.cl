#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef double tw_Float;
typedef double2 tw_Complex;
// Operations with tw_Float and tw_Complex:
// a*b : one or both operands must be tw_Float
// a+b : both operands must be the same type
// a/b : b must be tw_Float
// a-b : both operands must be the same type

tw_Complex cmul(tw_Complex a,tw_Complex b)
{
	return (tw_Complex)(a.s0*b.s0-a.s1*b.s1,a.s0*b.s1+a.s1*b.s0);
}

tw_Complex cexp(tw_Complex x)
{
	return (tw_Complex)(cos(x.s1)*exp(-x.s0),sin(x.s1)*exp(-x.s0));
}

__kernel void ExplicitRate(
	__global tw_Float * rate,
	__global tw_Float * E,
	const tw_Float E_conv,
	const tw_Float rate_conv,
	const tw_Float E_cutoff,
	const tw_Float C_pre,
	const tw_Float C_pow,
	const tw_Float C_exp)
{
	// TEMPORAL POINT PROTOCOL
	// rate,E are real 3d arrays with [t][x][y]
	// E in simulation units, rate output in simulation units
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Float Emag = E_conv*fabs(E[idx]) + E_cutoff;
	rate[idx] = rate_conv*C_pre*pow(Emag,C_pow)*exp(C_exp/Emag);
}

__kernel void EnvelopeRate(
	__global tw_Float * rate,
	__global tw_Complex * E,
	const tw_Float E_conv,
	const tw_Float rate_conv,
	const tw_Float E_cutoff,
	const tw_Float C_pre,
	const tw_Float C_pow,
	const tw_Float C_exp)
{
	// TEMPORAL POINT PROTOCOL
	// rate,E are real,complex 3d arrays with [t][x][y]
	// E in simulation units, rate output in simulation units
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Float Esim = sqrt(E[idx].s0*E[idx].s0 + E[idx].s1*E[idx].s1);
	const tw_Float Eau = E_conv*Esim + E_cutoff;
	rate[idx] = rate_conv*C_pre*pow(Eau,C_pow)*exp(C_exp/Eau);
}

__kernel void ExplicitRateSeries(
	__global tw_Float * rate,
	__global tw_Float * E,
	const tw_Float E_conv,
	const tw_Float rate_conv,
	const tw_Float E_cutoff,
	const tw_Float c0,
	const tw_Float c1,
	const tw_Float c2,
	const tw_Float c3,
	const tw_Float c4,
	const tw_Float c5)
{
	// TEMPORAL POINT PROTOCOL
	// rate,E are real 3d arrays with [t][x][y]
	// E in simulation units, rate output in simulation units
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Float Eau = E_conv*fabs(E[idx]);
	const tw_Float logE = log(Eau + E_cutoff);
	rate[idx] = rate_conv*exp(c0 + c1*logE + c2*logE*logE + c3*pow(logE,3) + c4*pow(logE,4) + c5*pow(logE,5));
}

__kernel void EnvelopeRateSeries(
	__global tw_Float * rate,
	__global tw_Complex * E,
	const tw_Float E_conv,
	const tw_Float rate_conv,
	const tw_Float E_cutoff,
	const tw_Float c0,
	const tw_Float c1,
	const tw_Float c2,
	const tw_Float c3,
	const tw_Float c4,
	const tw_Float c5)
{
	// TEMPORAL POINT PROTOCOL
	// rate,E are real,complex 3d arrays with [t][x][y]
	// E in simulation units, rate output in simulation units
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Float Eau = E_conv*sqrt(E[idx].s0*E[idx].s0 + E[idx].s1*E[idx].s1);
	const tw_Float logE = log(Eau + E_cutoff);
	rate[idx] = rate_conv*exp(c0 + c1*logE + c2*logE*logE + c3*pow(logE,3) + c4*pow(logE,4) + c5*pow(logE,5));
}

__kernel void ComputePlasmaDensity(
	__global tw_Float * ne,
	__global tw_Float * ng,
	const tw_Float dt,
	const int steps)
{
	// STRIP PROTOCOL
	// ne = [t][x][y]
	// On input ne contains rate, on output electron density
	// ng = [x][y], any units (units of ne will be the same)
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int Nx = get_global_size(0);
	const int Ny = get_global_size(1);
	const int offset = j0 + i0*Ny;
	const int s = Nx*Ny;

	tw_Float prev_rate = 0.0;
	tw_Float curr_rate = 0.0;
	tw_Float ne_raw = 0.0;
	ne[offset+(steps-1)*s] = 0.0;
	for (int k=steps-1;k>=0;--k)
	{
		curr_rate = ne[offset+k*s];
		ne_raw += 0.5*(curr_rate+prev_rate)*dt;
		ne[offset+k*s] = ng[offset]*(1.0-exp(-ne_raw));
		prev_rate = curr_rate;
	}
}
