//
//  Regular.metal
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

#include <metal_stdlib>
using namespace metal;
kernel void RegularGenerate(device float * const theta [[ buffer(0) ]],
								device float const * const phi [[ buffer(1) ]],
								constant uint const & N [[ buffer(2) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		theta[idx] = copysign( log( 1 + fabs( phi[idx] ) ), phi[idx]);
	}
}
kernel void RegularGradient(device float * const delta [[ buffer(0) ]],
								device float const * const theta [[ buffer(1) ]],
								device float const * const phi [[ buffer(2) ]],
								constant uint const & N [[ buffer(3) ]],
								uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		delta[idx] /= ( 1 + fabs( phi[idx] ) );
	}
}
kernel void RegularAdapt(device float * const phi [[ buffer(0) ]],
						 device float const * const theta [[ buffer(1) ]],
						 device float const * const delta [[ buffer(2) ]],
						 constant uint const & N [[ buffer(3) ]],
						 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		phi[idx] -= delta[idx] / ( 1 + fabs( phi[idx] ) );
	}
}
