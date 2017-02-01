//
//  Momentum.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

#include <metal_stdlib>
using namespace metal;

constant float eta [[ function_constant(0) ]];
constant float gamma [[ function_constant(1) ]];

inline float4x4 mix(float4x4 const a, float4x4 const b, float r) {
	return float4x4(mix(a[0], b[0], r), mix(a[1], b[1], r), mix(a[2], b[2], r), mix(a[3], b[3], r));
}
kernel void MomentumOptimize(device float4x4 * theta [[ buffer(0) ]],
							 device float4x4 * parameters [[ buffer(1) ]],
							 device float4x4 const * const delta [[ buffer(2) ]],
							 uint const n [[ thread_position_in_grid ]]) {
	int const k = n;
	theta[k] += ( parameters[k] = mix(eta * delta[k], parameters[k], gamma) );
}
