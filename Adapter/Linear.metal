//
//  Linear.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/03/10.
//
//

#include <metal_stdlib>
using namespace metal;
kernel void LinearGradient(device float * const delta [[ buffer(0) ]],
						  device float const * const theta [[ buffer(1) ]],
						  device float const * const phi [[ buffer(2) ]],
						  constant float const & L1 [[ buffer(3) ]],
						  constant float const & L2 [[ buffer(4) ]],
						  constant uint const & N [[ buffer(5) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		delta[idx] += L1 * sign(theta[idx]) + L2 * theta[idx];
	}
}
