//
//  OptimizerTests.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/02/01.
//
//

#include <metal_stdlib>
using namespace metal;
kernel void dydx(device float * const dydx [[ buffer(0) ]],
				 device float const * const x [[ buffer(1) ]],
				 uint i [[ thread_position_in_grid ]],
				 uint I [[ threads_per_grid ]]) {
	const float w[8] = {
		100,
		10,
		1,
		10,
		0.1,
		1,
		0.1,
		0.01
	};
	dydx[i] = - w[i%8] * ( x[i] - i - 1 + 64 );
}
