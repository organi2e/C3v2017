//
//  nonlinear.metal
//  macOS
//
//  Created by Kota Nakano on 2017/01/28.
//
//

#include <metal_stdlib>
using namespace metal;
inline float4 sigm(float4);
//Compute nonlinear math functions of 16-packed single precision floatings numbers, 4 times
kernel void exp(device float * const y [[ buffer(0) ]],
				device float const * const x [[ buffer(1) ]],
				uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	y[idx] = exp(x[idx]);
}
kernel void log(device float4x4 * const y [[ buffer(0) ]],
				device float4x4 const * const x [[ buffer(1) ]],
				constant uint const & length [[ buffer(2) ]],
				uint const n [[ thread_position_in_grid ]]) {
	uint4 const ofs = uint4(0, 1, 2, 3);
	uint4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	if ( can.x ) { float4x4 const v = x[idx.x]; y[idx.x] = float4x4(log(v[0]), log(v[1]), log(v[2]), log(v[3]));}
	if ( can.y ) { float4x4 const v = x[idx.y]; y[idx.y] = float4x4(log(v[0]), log(v[1]), log(v[2]), log(v[3]));}
	if ( can.z ) { float4x4 const v = x[idx.w]; y[idx.z] = float4x4(log(v[0]), log(v[1]), log(v[2]), log(v[3]));}
	if ( can.w ) { float4x4 const v = x[idx.z]; y[idx.w] = float4x4(log(v[0]), log(v[1]), log(v[2]), log(v[3]));}
}
kernel void step(device float4x4 * const y [[ buffer(0) ]],
				 device float4x4 const * const x [[ buffer(1) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float4x4 const v = x[idx];
	y[idx] = float4x4(step(0.0, v[0]),
					  step(0.0, v[1]),
					  step(0.0, v[2]),
					  step(0.0, v[3]));
}
kernel void sign(device float4x4 * const y [[ buffer(0) ]],
				 device float4x4 const * const x [[ buffer(1) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float4x4 const v = x[idx];
	y[idx] = float4x4(sign(v[0]),
					  sign(v[1]),
					  sign(v[2]),
					  sign(v[3]));}
/*
 Compute sigmoid function, by tvOS sec for 1024 * 1024 * 1024 = 5.2(CPU) vs 1.4(GPU)
 */
kernel void sigm2(device float4x4 * const y [[ buffer(0) ]],
				 device float4x4 const * const x [[ buffer(1) ]],
				 constant uint const & length [[ buffer(2) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	int4 const ofs = int4(0, 1, 2, 3);
	int4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	if ( can.x )
	{ float4x4 const v = x[idx.x]; y[idx.x] = float4x4(sigm(v[0]), sigm(v[1]), sigm(v[2]), sigm(v[3])); }
	if ( can.y )
	{ float4x4 const v = x[idx.y]; y[idx.y] = float4x4(sigm(v[0]), sigm(v[1]), sigm(v[2]), sigm(v[3])); }
	if ( can.z )
	{ float4x4 const v = x[idx.z]; y[idx.z] = float4x4(sigm(v[0]), sigm(v[1]), sigm(v[2]), sigm(v[3])); }
	if ( can.w )
	{ float4x4 const v = x[idx.w]; y[idx.w] = float4x4(sigm(v[0]), sigm(v[1]), sigm(v[2]), sigm(v[3])); }
}
/*
 Compute sigmoid function, by tvOS sec for 1024 * 1024 * 1024 = 5.2(CPU) vs 1.4(GPU)
 */
kernel void sigm(device float4x4 * const y [[ buffer(0) ]],
				 device float4x4 const * const x [[ buffer(1) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	float4x4 const v = x[n];
	y[n] = float4x4(sigm(v[0]), sigm(v[1]), sigm(v[2]), sigm(v[3]));
}

kernel void tanh(device float4x4 * const y [[ buffer(0) ]],
				 device float4x4 const * const x [[ buffer(1) ]],
				 constant uint const & length [[ buffer(2) ]],
				 uint const n [[ thread_position_in_grid ]]) {
	uint4 const ofs = uint4(0, 1, 2, 3);
	uint4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	float4x4 const a = x[idx.x];
	float4x4 const b = x[idx.y];
	float4x4 const c = x[idx.z];
	float4x4 const d = x[idx.w];
	if ( can.x ) y[idx.x] = float4x4(tanh(a[ofs.x]), tanh(a[ofs.y]), tanh(a[ofs.z]), tanh(a[ofs.w]));
	if ( can.y ) y[idx.y] = float4x4(tanh(b[ofs.x]), tanh(b[ofs.y]), tanh(b[ofs.z]), tanh(b[ofs.w]));
	if ( can.z ) y[idx.z] = float4x4(tanh(c[ofs.x]), tanh(c[ofs.y]), tanh(c[ofs.z]), tanh(c[ofs.w]));
	if ( can.w ) y[idx.w] = float4x4(tanh(d[ofs.x]), tanh(d[ofs.y]), tanh(d[ofs.z]), tanh(d[ofs.w]));
}
inline float4 sigm(float4 x) {
	return 0.5 + 0.5 * tanh ( 0.5 * x );
}
