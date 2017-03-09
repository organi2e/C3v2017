//
//  vector.metal
//  macOS
//
//  Created by Kota Nakano on 2017/03/07.
//
//

#include <metal_stdlib>
using namespace metal;

kernel void add(device float * y [[ buffer(0) ]],
				device float * a [[ buffer(1) ]],
				device float * b [[ buffer(2) ]],
				constant uint & N [[ buffer(3) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = a [ idx ] + b [ idx ];
	}
}
kernel void sub(device float * z [[ buffer(0) ]],
				device float * y [[ buffer(1) ]],
				device float * x [[ buffer(2) ]],
				constant uint & N [[ buffer(3) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		z [ idx ] = y [ idx ] - x [ idx ];
	}
}
kernel void mul(device float * z [[ buffer(0) ]],
				device float * y [[ buffer(1) ]],
				device float * x [[ buffer(2) ]],
				constant uint & N [[ buffer(3) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		z [ idx ] = y [ idx ] * x [ idx ];
	}
}
kernel void div(device float * z [[ buffer(0) ]],
				device float * y [[ buffer(1) ]],
				device float * x [[ buffer(2) ]],
				constant uint & N [[ buffer(3) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		z [ idx ] = y [ idx ] / x [ idx ];
	}
}
kernel void fma(device float * d [[ buffer(0) ]],
				device float * a [[ buffer(1) ]],
				device float * b [[ buffer(2) ]],
				device float * c [[ buffer(3) ]],
				constant uint & N [[ buffer(4) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		d[idx] = fma(a[idx], b[idx], c[idx]);
	}
}
kernel void exp(device float * y [[ buffer(0) ]],
				device float * x [[ buffer(1) ]],
				constant uint & N [[ buffer(2) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = exp ( x [ idx ] );
	}
}
kernel void log(device float * y [[ buffer(0) ]],
				device float * x [[ buffer(1) ]],
				constant uint & N [[ buffer(2) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = log ( x [ idx ] );
	}
}
kernel void cos(device float * y [[ buffer(0) ]],
				device float * x [[ buffer(1) ]],
				constant uint & N [[ buffer(2) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = cos ( x [ idx ] );
	}
}
kernel void sin(device float * y [[ buffer(0) ]],
				device float * x [[ buffer(1) ]],
				constant uint & N [[ buffer(2) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = sin ( x [ idx ] );
	}
}
kernel void tan(device float * y [[ buffer(0) ]],
				device float * x [[ buffer(1) ]],
				constant uint & N [[ buffer(2) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = tan ( x [ idx ] );
	}
}
kernel void abs(device float * y [[ buffer(0) ]],
				device float * x [[ buffer(1) ]],
				constant uint & N [[ buffer(2) ]],
				uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = abs ( x [ idx ] );
	}
}
kernel void sign(device float * y [[ buffer(0) ]],
				 device float * x [[ buffer(1) ]],
				 constant uint & N [[ buffer(2) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = sign ( x [ idx ] );
	}
}
kernel void tanh(device float * y [[ buffer(0) ]],
				 device float * x [[ buffer(1) ]],
				 constant uint & N [[ buffer(2) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = tanh ( x [ idx ] );
	}
}
kernel void sigm(device float * y [[ buffer(0) ]],
				 device float * x [[ buffer(1) ]],
				 constant uint & N [[ buffer(2) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = fma(tanh(0.5*x[idx]), 0.5, 0.5);
	}
}
kernel void relu(device float * y [[ buffer(0) ]],
				 device float * x [[ buffer(1) ]],
				 constant uint & N [[ buffer(2) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		y [ idx ] = max(0.0, x[idx]);
	}
}
kernel void soft(device float * y [[ buffer(0) ]],
				 device float * x [[ buffer(1) ]],
				 constant uint & N [[ buffer(2) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const v = x[idx];
		float const e = exp(v);
		y[idx] = select(log(1+e), v, isinf(e));
	}
}
kernel void regu(device float * const y [[ buffer(0) ]],
				 device float const * const x [[ buffer(1) ]],
				 constant uint & N [[ buffer(2) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	if ( n < N ) {
		int const idx = n;
		float const v = x[idx];
		y[idx] = copysign(log(1+abs(v)), v);
	}
}
