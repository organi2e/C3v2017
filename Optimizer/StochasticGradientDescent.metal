//
//  StochasticGradientDescent.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/01/25.
//
//

#include <metal_stdlib>
using namespace metal;

constant float eta [[ function_constant(0) ]];
constant float epsilon [[ function_constant(1) ]];

kernel void StochasticGradientDescentOptimize(device float4 * const value [[ buffer(0) ]],
											  device const float4 * const delta [[ buffer(1) ]],
											  uint const i [[ thread_position_in_grid ]]) {
	value[i] += eta * delta[i];
}
