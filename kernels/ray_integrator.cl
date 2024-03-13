inline double4 raise(const double4 x)
{
	return (double4)(x.s0,-x.s1,-x.s2,-x.s3);
}

inline double dot4(const double4 x,const double4 y)
{
	return x.s0*y.s0 - x.s1*y.s1 - x.s2*y.s2 - x.s3*y.s3;
}

inline double4 cross4(const double4 x,const double4 y)
{
	return (double4)(0.0,x.s2*y.s3-x.s3*y.s2,x.s3*y.s1-x.s1*y.s3,x.s1*y.s2-x.s2*y.s1);
}

inline double ComputeRayVolume(int vidx,__global double *xp)
{
	double4 v1,v2,v3,v1xv2;
	v1 = vload4(vidx+2,xp) - vload4(vidx+4,xp);
	v2 = vload4(vidx+6,xp) - vload4(vidx+8,xp);
	v3 = vload4(vidx+10,xp) - vload4(vidx+12,xp);
	v1xv2 = cross4(v1,v2);
	return fabs(dot4(v1xv2,v3));
}

inline double4 D_alpha_x(const double4 x,const double4 k)
{
	// function D_alpha is installed dynamically
	double ans[4],h[7] = {0.0,0.0,0.0,1e-5,0.0,0.0,0.0};
	double4 hv;
	int i;
	for (i=0;i<4;i++)
	{
		hv = vload4(0,&h[3-i]);
		ans[i] = 0.5*(D_alpha(x+hv,k) - D_alpha(x-hv,k))/h[3];
	}
    return vload4(0,ans);
}

inline double4 D_alpha_k(const double4 x,const double4 k)
{
	// function D_alpha is installed dynamically
	double ans[4],h[7] = {0.0,0.0,0.0,1e-5,0.0,0.0,0.0};
	double4 hv;
	int i;
	for (i=0;i<4;i++)
	{
		hv = vload4(0,&h[3-i]);
		ans[i] = 0.5*(D_alpha(x,k+hv) - D_alpha(x,k-hv))/h[3];
	}
    return vload4(0,ans);
}

__kernel void Symplectic(	__global double * xp,
							__global double * eikonal,
							const double ds0,
							const int steps)
{
    // xp is a 3d array with [bundle][ray][component]
    // components are x,k which are 4-vectors with signature +---
    // eikonal is a 2d array with [bundle][component]
    // components are phase,ax,ay,az
	const int bundle = get_global_id(0);
	const int vidx = bundle*7*2;

	double ray_volume,ds=ds0;
	double4 x0,k0,x1,k1,x2,k2,x3,k3,x4,k4,x5,k5,x6,k6,v00,A,Dk;

	// Propagate the Primary Ray
	// Prevent it from leaving volume by collapsing time step
	// The function "outside" is installed dynamically

	ray_volume = ComputeRayVolume(vidx,xp);
	x0 = vload4(vidx+0,xp);
	k0 = vload4(vidx+1,xp);
	x1 = vload4(vidx+2,xp);
	k1 = vload4(vidx+3,xp);
	x2 = vload4(vidx+4,xp);
	k2 = vload4(vidx+5,xp);
	x3 = vload4(vidx+6,xp);
	k3 = vload4(vidx+7,xp);
	x4 = vload4(vidx+8,xp);
	k4 = vload4(vidx+9,xp);
	x5 = vload4(vidx+10,xp);
	k5 = vload4(vidx+11,xp);
	x6 = vload4(vidx+12,xp);
	k6 = vload4(vidx+13,xp);
	A = vload4(bundle,eikonal);

	for (int i=0;i<steps;i++)
	{
		v00 = x0;
		Dk = D_alpha_k(v00,k0);
		v00 += ds*raise(Dk)/Dk.s0;
		ds *= 1.0-outside(v00);
		v00 = k0;

		Dk = D_alpha_k(x0,k0);
		x0 += ds*raise(Dk)/Dk.s0;
		k0 -= ds*raise(D_alpha_x(x0,k0))/Dk.s0;
		v00 = cross4(cross4(v00,k0),A);
		v00.s0 = -ds*(k0.s1*Dk.s1+k0.s2*Dk.s2+k0.s3*Dk.s3)/Dk.s0;
		A += v00;

		x1 += ds*raise(D_alpha_k(x1,k1))/Dk.s0;
		k1 -= ds*raise(D_alpha_x(x1,k1))/Dk.s0;

		x2 += ds*raise(D_alpha_k(x2,k2))/Dk.s0;
		k2 -= ds*raise(D_alpha_x(x2,k2))/Dk.s0;

		x3 += ds*raise(D_alpha_k(x3,k3))/Dk.s0;
		k3 -= ds*raise(D_alpha_x(x3,k3))/Dk.s0;

		x4 += ds*raise(D_alpha_k(x4,k4))/Dk.s0;
		k4 -= ds*raise(D_alpha_x(x4,k4))/Dk.s0;

		x5 += ds*raise(D_alpha_k(x5,k5))/Dk.s0;
		k5 -= ds*raise(D_alpha_x(x5,k5))/Dk.s0;

		x6 += ds*raise(D_alpha_k(x6,k6))/Dk.s0;
		k6 -= ds*raise(D_alpha_x(x6,k6))/Dk.s0;
	}

	vstore4(x0,vidx+0,xp);
	vstore4(k0,vidx+1,xp);
	vstore4(x1,vidx+2,xp);
	vstore4(k1,vidx+3,xp);
	vstore4(x2,vidx+4,xp);
	vstore4(k2,vidx+5,xp);
	vstore4(x3,vidx+6,xp);
	vstore4(k3,vidx+7,xp);
	vstore4(x4,vidx+8,xp);
	vstore4(k4,vidx+9,xp);
	vstore4(x5,vidx+10,xp);
	vstore4(k5,vidx+11,xp);
	vstore4(x6,vidx+12,xp);
	vstore4(k6,vidx+13,xp);

	// Update Amplitude
	ray_volume /= ComputeRayVolume(vidx,xp);
	ray_volume = sqrt(ray_volume);
	A.s1 *= ray_volume;
	A.s2 *= ray_volume;
	A.s3 *= ray_volume;
	vstore4(A,bundle,eikonal);
}
