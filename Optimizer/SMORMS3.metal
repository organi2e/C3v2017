//
//  SMORMS3.metal
//  macOS
//
//  Created by Kota Nakano on 2/28/17.
//
//

#include <metal_stdlib>
using namespace metal;

constant float alpha [[ function_constant(0) ]];//not used on this customized version
constant float epsilon [[ function_constant(1) ]];//not used on this customized version

kernel void SMORMS3Optimize(device float * const theta [[ buffer(0) ]],
							device float3 * const parameters [[ buffer(1) ]],
							device const float * const delta [[ buffer(2) ]],
							constant uint & N [[ buffer(3) ]],
							uint const n [[ thread_position_in_grid ]]) {
	
	if ( n < N ) {
		
		int const idx = n;
		
		//fetch
		float3 p = parameters[idx];
		float const g = delta[idx];
		
		//compute
		float const r = 1 / ( 1 + p.x );
		float const s = rsqrt(p.z = mix(g*g, p.z, r));
		float const t = select(0.0, s, isnormal(s));//Avoid epsilon
		float const u = (p.y = mix(g, p.y, r)) * t;
		float const v = abs(u);//or u * u;
		p.x = fma(p.x, 1 - v, 1);
		
		//update
		theta[idx] -= alpha * v * t * g;//or min(alpha, v)
		parameters[idx] = p;
		
	}
}
