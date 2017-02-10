//
//  MomentumAdaDelta.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

#include <metal_stdlib>
using namespace metal;

constant float rho [[ function_constant(0) ]];
constant float gamma [[ function_constant(1) ]];
constant float epsilon [[ function_constant(2) ]];

struct parameter_t {
	float4x4 uu;
	float4x4 gv;
	float4x4 vv;
};
kernel void MomentumAdaDeltaOptimize(device float4x4 * const theta [[ buffer(0) ]],
									 device parameter_t * const parameters [[ buffer(1) ]],
									 device const float4x4 * const delta [[ buffer(2) ]],
									 uint const n [[ thread_position_in_grid ]]) {
	parameter_t p = parameters[n];
	float4x4 const g = delta[n];
	float4x4 const m = float4x4(fabs(g[0]),
								fabs(g[1]),
								fabs(g[2]),
								fabs(g[3]));
	
	p.gv[0] = max(m[0], p.gv[0]) * rho;
	p.gv[1] = max(m[1], p.gv[1]) * rho;
	p.gv[2] = max(m[2], p.gv[2]) * rho;
	p.gv[3] = max(m[3], p.gv[3]) * rho;
	
	float4x4 const s = float4x4(p.vv[0] / p.gv[0],
								p.vv[1] / p.gv[1],
								p.vv[2] / p.gv[2],
								p.vv[3] / p.gv[3]);
	
	float4x4 const t = float4x4(select(1, max(s[0], epsilon), isnormal(s[0])),
								select(1, max(s[1], epsilon), isnormal(s[1])),
								select(1, max(s[2], epsilon), isnormal(s[2])),
								select(1, max(s[3], epsilon), isnormal(s[3])));
	
	float4x4 const v = float4x4(t[0] * m[0],
								t[1] * m[1],
								t[2] * m[2],
								t[3] * m[3]);
	/*
	p.uu[0] = mix(t[0]*g[0], p.uu[0], gamma);
	p.uu[1] = mix(t[1]*g[1], p.uu[1], gamma);
	p.uu[2] = mix(t[2]*g[2], p.uu[2], gamma);
	p.uu[3] = mix(t[3]*g[3], p.uu[3], gamma);
	theta[n] += p.uu;
	*/
	theta[n] += float4x4(t[0] * (p.uu[0] = mix(g[0], p.uu[0], gamma)),
						 t[1] * (p.uu[1] = mix(g[1], p.uu[1], gamma)),
						 t[2] * (p.uu[2] = mix(g[2], p.uu[2], gamma)),
						 t[3] * (p.uu[3] = mix(g[3], p.uu[3], gamma)));
	
	p.vv[0] = max(v[0], p.vv[0]) * rho;
	p.vv[1] = max(v[1], p.vv[1]) * rho;
	p.vv[2] = max(v[2], p.vv[2]) * rho;
	p.vv[3] = max(v[3], p.vv[3]) * rho;
	
	parameters[n] = p;
}
/*
kernel void MomentumAdaDeltaOptimize(device float4x4 * const theta [[ buffer(0) ]],
									 device parameter_t * const parameters [[ buffer(1) ]],
									 device const float4x4 * const delta [[ buffer(2) ]],
									 uint const n [[ thread_position_in_grid ]]) {
	float const e = 1e-6;
	float4x4 const g = delta[n];
	parameter_t p = parameters[n];
	
	p.gu[0] = rho * p.gu[0] + ( 1 - rho ) * g[0];
	p.gu[1] = rho * p.gu[1] + ( 1 - rho ) * g[1];
	p.gu[2] = rho * p.gu[2] + ( 1 - rho ) * g[2];
	p.gu[3] = rho * p.gu[3] + ( 1 - rho ) * g[3];
	
	//L1
	p.gv[0] = rho * p.gv[0] + ( 1 - rho ) * fabs(g[0]);
	p.gv[1] = rho * p.gv[1] + ( 1 - rho ) * fabs(g[1]);
	p.gv[2] = rho * p.gv[2] + ( 1 - rho ) * fabs(g[2]);
	p.gv[3] = rho * p.gv[3] + ( 1 - rho ) * fabs(g[3]);
	//L2
	p.gv[0] = sqrt(p.gv[0]*p.gv[0]*rho*rho+(1-rho*rho)*g[0]*g[0]);
	p.gv[1] = sqrt(p.gv[1]*p.gv[1]*rho*rho+(1-rho*rho)*g[1]*g[1]);
	p.gv[2] = sqrt(p.gv[2]*p.gv[2]*rho*rho+(1-rho*rho)*g[2]*g[2]);
	p.gv[3] = sqrt(p.gv[3]*p.gv[3]*rho*rho+(1-rho*rho)*g[3]*g[3]);
	//Linf
	
	p.gv[0] = max(rho*p.gv[0], fabs(g[0]));
	p.gv[1] = max(rho*p.gv[1], fabs(g[1]));
	p.gv[2] = max(rho*p.gv[2], fabs(g[2]));
	p.gv[3] = max(rho*p.gv[3], fabs(g[3]));
	
	float4x4 const v = float4x4(p.gu[0]*max(p.vv[0]/(p.gv[0]+e),1e-6),
								p.gu[1]*max(p.vv[1]/(p.gv[1]+e),1e-6),
								p.gu[2]*max(p.vv[2]/(p.gv[2]+e),1e-6),
								p.gu[3]*max(p.vv[3]/(p.gv[3]+e),1e-6));
	
	//float4x4 const v = (s-rho*p.vu)*(1/(1-rho));//L1orLinf
	
	//float4x4 const v = mul(g, sqrt(p.vv+e), rsqrt(p.gv+e));//L2
	//float4x4 const v = mul(p.gu, sqrt(p.vv+e), rsqrt(p.gv+e));//L2
	//float4x4 const v = (mul(p.gu, sqrt(p.vv+e), rsqrt(p.gv+e))-rho*p.vu)*(1/(1-rho));//L2
	
	//p.vu = v;//mix(v, p.vu, rho);
	//p.vv = mix(fabs(v), p.vv, rho);//L1
	//p.vv = mix(sq(v), p.vv, rho*rho);//L2
	
	//L1
	p.vv[0] = rho * p.vv[0] + ( 1 - rho ) * fabs(v[0]);
	p.vv[1] = rho * p.vv[1] + ( 1 - rho ) * fabs(v[1]);
	p.vv[2] = rho * p.vv[2] + ( 1 - rho ) * fabs(v[2]);
	p.vv[3] = rho * p.vv[3] + ( 1 - rho ) * fabs(v[3]);
	//L2
	p.vv[0] = sqrt(p.vv[0]*p.vv[0]*rho*rho+(1-rho*rho)*v[0]*v[0]);
	p.vv[1] = sqrt(p.vv[1]*p.vv[1]*rho*rho+(1-rho*rho)*v[1]*v[1]);
	p.vv[2] = sqrt(p.vv[2]*p.vv[2]*rho*rho+(1-rho*rho)*v[2]*v[2]);
	p.vv[3] = sqrt(p.vv[3]*p.vv[3]*rho*rho+(1-rho*rho)*v[3]*v[3]);
	//Linf
	
	p.vv[0] = max(rho*p.vv[0], fabs(v[0]));
	p.vv[1] = max(rho*p.vv[1], fabs(v[1]));
	p.vv[2] = max(rho*p.vv[2], fabs(v[2]));
	p.vv[3] = max(rho*p.vv[3], fabs(v[3]));
	
	theta[n] += v;
	parameters[n] = p;
}
*/
/*
kernel void MomentumAdaDeltaOptimize(device float4x4 * const theta [[ buffer(0) ]],
									 device parameter_t * const parameters [[ buffer(1) ]],
									 device const float4x4 * const delta [[ buffer(2) ]],
									 uint const n [[ thread_position_in_grid ]]) {
	//float4 const ev = float4(epsilon);
	//float4x4 const e = float4x4(ev, ev, ev, ev);
	float4x4 const g = delta[n];
	parameter_t p = parameters[n];
	
	p.gu = mix(g, p.gu, rho);
	//p.gv = mix(fabs(g), p.gv, rho);//L1
	//p.gv = mix(sq(g), p.gv, rho*rho);//L2
	
	p.gv[0] = max(rho*p.gv[0], fabs(g[0]));
	p.gv[1] = max(rho*p.gv[1], fabs(g[1]));
	p.gv[2] = max(rho*p.gv[2], fabs(g[2]));
	p.gv[3] = max(rho*p.gv[3], fabs(g[3]));
	
	//max(rho*p.gv, fabs(g));//Linf
	
	//float4x4 const v = muldiv(g, p.vv+e, p.gv+e);//L1orLinf
	//float4x4 const v = muldiv(p.gu, p.vv+e, p.gv+e);//L1orLinf
	
	float4x4 const s = float4x4(p.gu[0]*(p.vv[0]+epsilon)/(p.gv[0]+epsilon),
								p.gu[1]*(p.vv[1]+epsilon)/(p.gv[1]+epsilon),
								p.gu[2]*(p.vv[2]+epsilon)/(p.gv[2]+epsilon),
								p.gu[3]*(p.vv[3]+epsilon)/(p.gv[3]+epsilon));
	
	float4x4 const v = (s-rho*p.vu)*(1/(1-rho));//L1orLinf
	
	//float4x4 const v = mul(g, sqrt(p.vv+e), rsqrt(p.gv+e));//L2
	//float4x4 const v = mul(p.gu, sqrt(p.vv+e), rsqrt(p.gv+e));//L2
	//float4x4 const v = (mul(p.gu, sqrt(p.vv+e), rsqrt(p.gv+e))-rho*p.vu)*(1/(1-rho));//L2
	
	p.vu = s;//mix(v, p.vu, rho);
	//p.vv = mix(fabs(v), p.vv, rho);//L1
	//p.vv = mix(sq(v), p.vv, rho*rho);//L2
	
	p.vv[0] = max(rho*p.vv[0], fabs(v[0]));
	p.vv[1] = max(rho*p.vv[1], fabs(v[1]));
	p.vv[2] = max(rho*p.vv[2], fabs(v[2]));
	p.vv[3] = max(rho*p.vv[3], fabs(v[3]));
	
	theta[n] += v;
	parameters[n] = p;
}
*/
