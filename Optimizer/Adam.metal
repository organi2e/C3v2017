//
//  Adam.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/02/01.
//
//

#include <metal_stdlib>
using namespace metal;

constant float alpha [[ function_constant(0) ]];
constant float beta [[ function_constant(1) ]];
constant float gamma [[ function_constant(2) ]];
constant float epsilon [[ function_constant(3) ]];

inline float4x4 sq(float4x4 const x) {
	return float4x4(x[0]*x[0], x[1]*x[1], x[2]*x[2], x[3]*x[3]);
}
inline float4x4 sup(float4x4 w, float e) {
	return float4x4(w[0]+e, w[1]+e, w[2]+e, w[3]+e);
}
inline float4x4 rsqrt(float4x4 const x) {
	return float4x4(rsqrt(x[0]), rsqrt(x[1]), rsqrt(x[2]), rsqrt(x[3]));
}
inline float4x4 mix(float4x4 const a, float4x4 const b, float r) {
	return float4x4(mix(a[0], b[0], r), mix(a[1], b[1], r), mix(a[2], b[2], r), mix(a[3], b[3], r));
}
inline float4x4 mul(float4x4 const a, float4x4 const b) {
	return float4x4(a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3]);
}
struct parameter_t {
	float4x4 v[2];
};
kernel void AdamOptimize(device float4x4 * const theta [[ buffer(0) ]],
						 device parameter_t * const parameters [[ buffer(1) ]],
						 device const float4x4 * const delta [[ buffer(2) ]],
						 uint const n [[ thread_position_in_grid ]]) {
	float4x4 const g = delta[n];
	parameter_t p = parameters[n];
	
	p.v[0] = mix(g, p.v[0], beta);
	p.v[1] = mix(sq(g), p.v[1], gamma);
	
	theta[n] += alpha * mul(p.v[0], rsqrt(sup(p.v[1], epsilon)));
	parameters[n] = p;
}
