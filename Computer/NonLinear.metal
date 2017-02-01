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
kernel void exp(device float4x4 * const y [[ buffer(0) ]],
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
	if ( can.x ) y[idx.x] = float4x4(exp(a[ofs.x]), exp(a[ofs.y]), exp(a[ofs.z]), exp(a[ofs.w]));
	if ( can.y ) y[idx.y] = float4x4(exp(b[ofs.x]), exp(b[ofs.y]), exp(b[ofs.z]), exp(b[ofs.w]));
	if ( can.z ) y[idx.z] = float4x4(exp(c[ofs.x]), exp(c[ofs.y]), exp(c[ofs.z]), exp(c[ofs.w]));
	if ( can.w ) y[idx.w] = float4x4(exp(d[ofs.x]), exp(d[ofs.y]), exp(d[ofs.z]), exp(d[ofs.w]));
}
kernel void log(device float4x4 * const y [[ buffer(0) ]],
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
	if ( can.x ) y[idx.x] = float4x4(log(a[ofs.x]), log(a[ofs.y]), log(a[ofs.z]), log(a[ofs.w]));
	if ( can.y ) y[idx.y] = float4x4(log(b[ofs.x]), log(b[ofs.y]), log(b[ofs.z]), log(b[ofs.w]));
	if ( can.z ) y[idx.z] = float4x4(log(c[ofs.x]), log(c[ofs.y]), log(c[ofs.z]), log(c[ofs.w]));
	if ( can.w ) y[idx.w] = float4x4(log(d[ofs.x]), log(d[ofs.y]), log(d[ofs.z]), log(d[ofs.w]));
}
kernel void sign(device float4x4 * const y [[ buffer(0) ]],
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
	if ( can.x ) y[idx.x] = float4x4(sign(a[ofs.x]), sign(a[ofs.y]), sign(a[ofs.z]), sign(a[ofs.w]));
	if ( can.y ) y[idx.y] = float4x4(sign(b[ofs.x]), sign(b[ofs.y]), sign(b[ofs.z]), sign(b[ofs.w]));
	if ( can.z ) y[idx.z] = float4x4(sign(c[ofs.x]), sign(c[ofs.y]), sign(c[ofs.z]), sign(c[ofs.w]));
	if ( can.w ) y[idx.w] = float4x4(sign(d[ofs.x]), sign(d[ofs.y]), sign(d[ofs.z]), sign(d[ofs.w]));
}
kernel void sigm(device float4x4 * const y [[ buffer(0) ]],
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
	if ( can.x ) y[idx.x] = float4x4(sigm(a[ofs.x]), sigm(a[ofs.y]), sigm(a[ofs.z]), sigm(a[ofs.w]));
	if ( can.y ) y[idx.y] = float4x4(sigm(b[ofs.x]), sigm(b[ofs.y]), sigm(b[ofs.z]), sigm(b[ofs.w]));
	if ( can.z ) y[idx.z] = float4x4(sigm(c[ofs.x]), sigm(c[ofs.y]), sigm(c[ofs.z]), sigm(c[ofs.w]));
	if ( can.w ) y[idx.w] = float4x4(sigm(d[ofs.x]), sigm(d[ofs.y]), sigm(d[ofs.z]), sigm(d[ofs.w]));
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
