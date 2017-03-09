//
//  Adam.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/02/01.
//
//

#include <metal_stdlib>
using namespace metal;

constant float alpha [[ function_constant(0) ]];
constant float beta [[ function_constant(1) ]];
constant float gamma [[ function_constant(2) ]];
constant float epsilon [[ function_constant(3) ]];

kernel void AdamOptimize(device float * const theta [[ buffer(0) ]],
						 device float2 * const parameters [[ buffer(1) ]],
						 device const float * const delta [[ buffer(2) ]],
						 uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float const g = delta[idx];
	float2 p = parameters[idx];
	
	p.x = mix(g, p.x, beta);
	//p.v = mix(fabs(g), p.v, beta);//L1
	p.y = mix(g*g, p.y, gamma);//L2
	//p.v = max(fabs(g), beta*p.v);//L-Inf
	
	//theta[n] += alpha * div(p.u, sup(p.v, epsilon));//L1orL-Inf
	theta[idx] -= alpha * p.x * rsqrt(p.y + epsilon);
	
	parameters[idx] = p;
}
