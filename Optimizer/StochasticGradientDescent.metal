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

kernel void StochasticGradientDescentOptimize(device float * const value [[ buffer(0) ]],
											  device const float * const delta [[ buffer(1) ]],
											  constant uint & N [[ buffer(2) ]],
											  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		value[idx] += eta * delta[idx];
	}
}
