//
//  Ridge.metal
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

#include <metal_stdlib>
using namespace metal;
constant float lambda [[ function_constant(0) ]];
kernel void RidgeGradient(device float * const delta [[ buffer(0) ]],
						  device float const * const theta [[ buffer(2) ]],
						  device float const * const phi [[ buffer(3) ]],
						  constant float const & lambda [[ buffer(4) ]],
						  constant uint const & N [[ buffer(5) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		delta[idx] -= lambda * theta[idx];
	}
}
