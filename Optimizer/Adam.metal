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
						 constant uint const & N [[ buffer(3) ]],
						 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const g = delta[idx];
		float2 p = parameters[idx];
		
		p = mix(float2(g, g*g), p, float2(beta, gamma));
	
		float const r = rsqrt(p.y+epsilon);
		theta[idx] -= alpha * p.x * select(0.0, r, isnormal(r));
		
		parameters[idx] = p;
	}
}
