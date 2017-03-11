//
//  Lasso.metal
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

#include <metal_stdlib>
using namespace metal;
constant float lambda [[ function_constant(0) ]];
kernel void LassoGradient(device float * const delta [[ buffer(0) ]],
						  device float const * const theta [[ buffer(1) ]],
						  device float const * const phi [[ buffer(2) ]],
						  constant uint const & N [[ buffer(3) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		delta[idx] += lambda * sign(theta[idx]);
	}
}
kernel void LassoAdapt(device float * const phi [[ buffer(0) ]],
					   device float const * const theta [[ buffer(1) ]],
					   device float const * const delta [[ buffer(2) ]],
					   constant uint const & N [[ buffer(3) ]],
					   uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		phi[idx] -= delta[idx] + lambda * sign(theta[idx]);
	}
}
