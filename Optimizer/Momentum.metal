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

kernel void MomentumOptimize(device float4 * theta [[ buffer(0) ]],
							 device float4 * parameters [[ buffer(1) ]],
							 device float4 const * const delta [[ buffer(2) ]],
							 uint const i [[ thread_position_in_grid ]]) {
	theta[i] += ( parameters[i] = gamma * parameters[i] + eta * delta[i] );
}
