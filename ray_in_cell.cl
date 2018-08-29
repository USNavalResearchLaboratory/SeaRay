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

inline double Gather(__global double *dens,const double4 x)
{
	// cart,cyl are constant doubles installed dynamically
	// spacing,size are constant double4's installed dynamically
	// num is a constant int[4] installed dynamically
	// Assumes coordinate lies in the interior grid
	double ans = 0.0;
	const double x1 = (cart*x.s1 + cyl*sqrt(x.s1*x.s1 + x.s2*x.s2))/spacing.s1;
	const double x2 = cart*x.s2/spacing.s2;
	const double x3 = x.s3/spacing.s3;
	const int c1 = 2 + ((int)cart)*(num[1]-4)/2 + (int)x1 - (x1<0);
	const int c2 = 2 + (num[2]-4)/2 + (int)x2 - (x2<0);
	const int c3 = 2 + (num[3]-4)/2 + (int)x3 - (x3<0);
	const int s1 = (c1>0)*(c1<num[1]-1)*num[2]*num[3];
	const int s2 = (c2>0)*(c2<num[2]-1)*num[3];
	const int s3 = (c3>0)*(c3<num[3]-1);
	const double q1 = x1 - (int)x1 + (x1<0) - 0.5;
	const double q2 = cart*(x2 - (int)x2 + (x2<0) - 0.5);
	const double q3 = x3 - (int)x3 + (x3<0) - 0.5;
	const double w1[3] = {0.125-0.5*q1+0.5*q1*q1,0.75-q1*q1,0.125+0.5*q1+0.5*q1*q1};
	const double w2[3] = {0.125-0.5*q2+0.5*q2*q2,0.75-q2*q2,0.125+0.5*q2+0.5*q2*q2};
	const double w3[3] = {0.125-0.5*q3+0.5*q3*q3,0.75-q3*q3,0.125+0.5*q3+0.5*q3*q3};
	for (int i1=0;i1<3;i1++)
		for (int i2=0;i2<3;i2++)
			for (int i3=0;i3<3;i3++)
				ans += w1[i1]*w2[i2]*w3[i3]*dens[(c1+i1-1)*s1 + (c2+i2-1)*s2 + (c3+i3-1)*s3];
	return ans;
}

inline double D_alpha(__global double *dens,const double4 x,const double4 k)
{
	return Gather(dens,x) - dot4(k,k);
}

inline double4 D_alpha_x(__global double *dens,const double4 x,const double4 k)
{
	// function D_alpha is installed dynamically
	double ans[4],h[7] = {0.0,0.0,0.0,1e-5,0.0,0.0,0.0};
	double4 hv;
	for (int i=0;i<4;i++)
	{
		hv = vload4(0,&h[3-i]);
		ans[i] = 0.5*(D_alpha(dens,x+hv,k) - D_alpha(dens,x-hv,k))/h[3];
	}
	return vload4(0,ans);
}

inline double4 D_alpha_k(__global double *dens,const double4 x,const double4 k)
{
	// function D_alpha is installed dynamically
	double ans[4],h[7] = {0.0,0.0,0.0,1e-5,0.0,0.0,0.0};
	double4 hv;
	for (int i=0;i<4;i++)
	{
		hv = vload4(0,&h[3-i]);
		ans[i] = 0.5*(D_alpha(dens,x,k+hv) - D_alpha(dens,x,k-hv))/h[3];
	}
	return vload4(0,ans);
}

__kernel void Symplectic(	__global double * xp,
							__global double * eikonal,
							__global double * dens,
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
		Dk = D_alpha_k(dens,v00,k0);
		v00 += ds*raise(Dk)/Dk.s0;
		ds *= 1.0-outside(v00);
		v00 = k0;

		Dk = D_alpha_k(dens,x0,k0);
		x0 += ds*raise(Dk)/Dk.s0;
		k0 -= ds*raise(D_alpha_x(dens,x0,k0))/Dk.s0;
		v00 = cross4(cross4(v00,k0),A);
		v00.s0 = -ds*dot(k0,Dk)/Dk.s0;
		A += v00;

		x1 += ds*raise(D_alpha_k(dens,x1,k1))/Dk.s0;
		k1 -= ds*raise(D_alpha_x(dens,x1,k1))/Dk.s0;

		x2 += ds*raise(D_alpha_k(dens,x2,k2))/Dk.s0;
		k2 -= ds*raise(D_alpha_x(dens,x2,k2))/Dk.s0;

		x3 += ds*raise(D_alpha_k(dens,x3,k3))/Dk.s0;
		k3 -= ds*raise(D_alpha_x(dens,x3,k3))/Dk.s0;

		x4 += ds*raise(D_alpha_k(dens,x4,k4))/Dk.s0;
		k4 -= ds*raise(D_alpha_x(dens,x4,k4))/Dk.s0;

		x5 += ds*raise(D_alpha_k(dens,x5,k5))/Dk.s0;
		k5 -= ds*raise(D_alpha_x(dens,x5,k5))/Dk.s0;

		x6 += ds*raise(D_alpha_k(dens,x6,k6))/Dk.s0;
		k6 -= ds*raise(D_alpha_x(dens,x6,k6))/Dk.s0;
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

__kernel void GetDensity(	__global double * xp,
							__global double * grid,
							__global double * ans)
{
	// xp is a 3d array with [bundle][ray][component]
	// components are x,k which are 4-vectors with signature +---
	// The function "outside" is installed dynamically
	const int bundle = get_global_id(0);
	const int ray = get_global_id(1);
	const int vidx = bundle*7*2;

	double4 x0 = vload4(vidx,xp);
	ans[ray+bundle*7] = Gather(grid,x0);
}
