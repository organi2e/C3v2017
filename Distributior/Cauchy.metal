//
//  Cauchy.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

#include <metal_stdlib>
using namespace metal;

constant uint3 xorshift [[ function_constant(0) ]];

kernel void CauchyRng(device float4 * const value [[ buffer(0) ]],
					  device const float4 * const mu [[ buffer(1) ]],
					  device const float4 * const sigma [[ buffer(2) ]],
					  device const uint4 * const seeds [[ buffer(3) ]],
					  constant const uint & length [[ buffer(4) ]],
					  uint const n [[ thread_position_in_grid ]],
					  uint const N [[ threads_per_grid ]]) {
	
	uint4 seq = seeds[n];
	seq = select ( seq, -1, seq == 0 );
	
	for ( uint k = n, K = length ; k < K ; k += N ) {
		
		float4 const u = ( float4 ( seq ) + 0.5 ) / 4294967296.0 - 0.5;
		
		value [ k ] = tanpi(u);
		
		seq ^= seq << xorshift.x;
		seq ^= seq >> xorshift.y;
		seq ^= seq << xorshift.z;
		
	}
}
kernel void CauchyCDF(device float4 * const f [[ buffer(0) ]],
					  device const float4 * const mu [[ buffer(1) ]],
					  device const float4 * const sigma [[ buffer(2) ]],
					  uint const n [[ thread_position_in_grid ]]) {
	float4 const u = mu[n];
	float4 const s = sigma[n];
	f[n] = 0.5 + M_1_PI_F * atan(u/s);
}
kernel void CauchyPDF(device float4 * const p [[ buffer(0) ]],
					  device const float4 * const mu [[ buffer(1) ]],
					  device const float4 * const sigma [[ buffer(2) ]],
					  uint const n [[ thread_position_in_grid ]]) {
	float4 const u = mu[n];
	float4 const s = sigma[n];
	p[n] = M_1_PI_F * s / ( s * s + u * u );
}
kernel void CauchyStatistics(device float2x4 * const f [[ buffer(0) ]],
							 device const float4 * const mu [[ buffer(1) ]],
							 device const float4 * const sigma [[ buffer(2) ]],
							 uint const n [[ thread_position_in_grid ]]) {
	float4 const u = mu[n];
	float4 const s = sigma[n];
	f[n] = float2x4(0.5+M_1_PI_F*atan(u/s), M_1_PI_F*s*(u*u+s*s));
}
kernel void CauchyGradient(device float2x4 * const d [[ buffer(0) ]],
						   device const float4 * const mu [[ buffer(1) ]],
						   device const float4 * const sigma [[ buffer(2) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	float4 const u = mu[n];
	float4 const s = sigma[n];
	float4 const v = M_1_PI_F * ( u * u + s * s );
	d[n] = float2x4(s*v, -u*v);
}
