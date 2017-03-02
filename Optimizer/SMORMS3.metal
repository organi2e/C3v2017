//
//  SMORMS3.metal
//  macOS
//
//  Created by Kota Nakano on 2/28/17.
//
//

#include <metal_stdlib>
using namespace metal;

constant float alpha [[ function_constant(0) ]];
constant float epsilon [[ function_constant(1) ]];

kernel void SMORMS3Optimize(device float * const theta [[ buffer(0) ]],
							device float4 * const parameters [[ buffer(1) ]],
							device const float * const delta [[ buffer(2) ]],
							uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float4 p = parameters[idx];
	float const g = delta[idx];
	
	float const r = 1 / ( 1 + p.x );
	float const s = rsqrt(p.z = mix(g*g, p.z, r));
	float const t = select(0.0, s, isnormal(s));//Avoid epsilon
	float const u = abs(p.y = mix(g, p.y, r)) * t;
	float const v = u * u;
	
	p.x = fma(p.x, 1.0 - v, 1.0);
	
	theta[idx] += g * v * t;
	parameters[idx] = p;
	
}
