#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef double tw_Float;
typedef double2 tw_Complex;
// Operations with tw_Float and tw_Complex:
// a*b : one or both operands must be tw_Float
// a+b : both operands must be the same type
// a/b : b must be tw_Float
// a-b : both operands must be the same type

__constant tw_Float pi = 3.141592653589793;

tw_Complex cmul(tw_Complex a,tw_Complex b)
{
	return (tw_Complex)(a.s0*b.s0-a.s1*b.s1,a.s0*b.s1+a.s1*b.s0);
}

tw_Complex abstar(tw_Complex a,tw_Complex b)
{
	return (tw_Complex)(a.s0*b.s0+a.s1*b.s1,-a.s0*b.s1+a.s1*b.s0);
}

tw_Complex cexp(tw_Complex x)
{
	return (tw_Complex)(cos(x.s1)*exp(-x.s0),sin(x.s1)*exp(-x.s0));
}

__kernel void AddChiParaxial(
	// 3D arrays [t][x][y]
	__global tw_Float * dchi,
	__global tw_Complex *E,
	// Transverse dependence [x][y]
	__global tw_Float * ngas,
	// Depend only on state [state]
	__global tw_Float * Tjj, // space resolution would allow for grad(T)
	__global tw_Complex * wdt,
	// scalars
	const tw_Float dt,
	const tw_Float dalpha,
	const tw_Float hbar,
	const int steps,
	const int states)
{
	// STRIP PROTOCOL
	const int x0 = get_global_id(0);
	const int y0 = get_global_id(1);
	const int Nx = get_global_size(0);
	const int Ny = get_global_size(1);
	const int xyoff = x0*Ny + y0;
	
	tw_Float F,Sjj,cum;
	tw_Complex exp_iwt, exp_iwdth, rhojj, term;
	for (int j=2; j<states; j++)
	{
		Sjj = 2.0*j*(j-1)/(15*(2*j-1));
		// init time n+1/2
		rhojj = (tw_Complex)(0,0);
		exp_iwt = (tw_Complex)(1,0);
		exp_iwdth = (tw_Complex)(cos(0.5*wdt[j].s0),sin(0.5*wdt[j].s0))*exp(-0.5*wdt[j].s1);
		// init time n
		cum = 0.0;
		for (int k=steps-1;k>=0;--k)
		{
			int idx = k*Nx*Ny + xyoff;
			// add this transition's contribution
			term = Tjj[j] * Sjj * cmul(abstar(rhojj,exp_iwt),wdt[j]);
			cum += 2 * term.s1 * ngas[xyoff] * dalpha;
			dchi[idx] += cum;
			// driving term with frequency units at full time steps
			F = 0.5 * (dalpha/hbar) * 0.5 * (E[idx].s0*E[idx].s0 + E[idx].s1*E[idx].s1);
			exp_iwt = cmul(exp_iwt,exp_iwdth);
			// update j <-> j-2 transition at half time steps
			rhojj -= dt * F * cmul((tw_Complex)(0,1),exp_iwt);
			exp_iwt = cmul(exp_iwt,exp_iwdth);
		}
	}
}

__kernel void AddChiUPPE(
	// 3D arrays [t][x][y]
	__global tw_Float * dchi,
	__global tw_Float *E,
	// Transverse dependence [x][y]
	__global tw_Float * ngas,
	// Depend only on state [state]
	__global tw_Float * Tjj, // space resolution would allow for grad(T)
	__global tw_Complex * wdt,
	// scalars
	const tw_Float dt,
	const tw_Float dalpha,
	const tw_Float hbar,
	const int steps,
	const int states)
{
	// STRIP PROTOCOL
	const int x0 = get_global_id(0);
	const int y0 = get_global_id(1);
	const int Nx = get_global_size(0);
	const int Ny = get_global_size(1);
	const int xyoff = x0*Ny + y0;
	
	tw_Float F,Sjj,cum;
	tw_Complex exp_iwt, exp_iwdth, rhojj, term;
	for (int j=2; j<states; j++)
	{
		Sjj = 2.0*j*(j-1)/(15*(2*j-1));
		// init time n+1/2
		rhojj = (tw_Complex)(0,0);
		exp_iwt = (tw_Complex)(1,0);
		exp_iwdth = (tw_Complex)(cos(0.5*wdt[j].s0),sin(0.5*wdt[j].s0))*exp(-0.5*wdt[j].s1);
		// init time n
		cum = 0.0;
		for (int k=steps-1;k>=0;--k)
		{
			int idx = k*Nx*Ny + xyoff;
			// add this transition's contribution
			term = Tjj[j] * Sjj * cmul(abstar(rhojj,exp_iwt),wdt[j]);
			cum += 2 * term.s1 * ngas[xyoff] * dalpha;
			dchi[idx] += cum;
			// driving term with frequency units at full time steps
			F = 0.5 * (dalpha/hbar) * E[idx] * E[idx];
			exp_iwt = cmul(exp_iwt,exp_iwdth);
			// update j <-> j-2 transition at half time steps
			rhojj -= dt * F * cmul((tw_Complex)(0,1),exp_iwt);
			exp_iwt = cmul(exp_iwt,exp_iwdth);
		}
	}
}
