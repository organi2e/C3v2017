//
//  matrix.metal
//  macOS
//
//  Created by Kota Nakano on 2017/03/07.
//
//

#include <metal_stdlib>
using namespace metal;

kernel void gemvc(device float * const y [[ buffer(0) ]],
				  device float const * const w [[ buffer(1) ]],
				  device float const * const x [[ buffer(2) ]],
				  constant float2 & r [[ buffer(3) ]],
				  constant uint2 & s [[ buffer(4) ]],
				  threadgroup float4 * shared [[ threadgroup(0) ]],
				  uint const t [[ thread_position_in_threadgroup ]],
				  uint const T [[ threads_per_threadgroup ]],
				  uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(s);
	int3 const d = 4 * int3(T, t, n);
	
	int4 const row = d.z + int4(0, 1, 2, 3);
	bool4 const rows_mask = row < size.x;
	
	float4 value(0);
	for ( int4 col = d.y + int4(0, 1, 2, 3) ; col.x < size.y ; col += d.x ) {
		bool4 const cols_mask = col < size.y;
		int4 const idx = row * size.y + col.x;
		value += select(0, *(device float4*)(x + col.x), cols_mask) * float4x4(select(0, *(device float4*)(w + idx.x), rows_mask.x && cols_mask),
																			   select(0, *(device float4*)(w + idx.y), rows_mask.y && cols_mask),
																			   select(0, *(device float4*)(w + idx.z), rows_mask.z && cols_mask),
																			   select(0, *(device float4*)(w + idx.w), rows_mask.w && cols_mask));
	}
	
	int const a = t;
	int b = T;
	
	threadgroup float4 * accum = shared + a;
	
	*accum = value;
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) {
			*accum += accum[b];
		}
	}
	if ( a ) {
		
	} else if ( rows_mask.w ) {
		*(device float4*)(y+row.x) = accum->xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(y+row.x) = accum->xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(y+row.x) = accum->xy;
	} else if ( rows_mask.x ) {
		*(device float *)(y+row.x) = accum->x;
	}
}
kernel void gemvr(device float * const y [[ buffer(0) ]],
				  device float const * const w [[ buffer(1) ]],
				  device float const * const x [[ buffer(2) ]],
				  constant float2 & r [[ buffer(3) ]],
				  constant uint2 & s [[ buffer(4) ]],
				  threadgroup float4 * shared [[ threadgroup(0) ]],
				  uint const t [[ thread_position_in_threadgroup ]],
				  uint const T [[ threads_per_threadgroup ]],
				  uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(s);
	int3 const d = 4 * int3(T, t, n);
	
	int4 const row = d.z + int4(0, 1, 2, 3);
	bool4 const rows_mask = row < size.x;
	
	float4 value(0);
	for ( int4 col = d.y + int4(0, 1, 2, 3) ; col.x < size.y ; col += d.x ) {
		bool4 const cols_mask = col < size.y;
		int4 const idx = col * size.x + row.x;
		value += float4x4(select(0, *(device float4*)(w + idx.x), rows_mask && cols_mask.x),
						  select(0, *(device float4*)(w + idx.y), rows_mask && cols_mask.y),
						  select(0, *(device float4*)(w + idx.z), rows_mask && cols_mask.z),
						  select(0, *(device float4*)(w + idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(x + col.x), cols_mask);
	}
	
	int const a = t;
	int b = T;
	
	threadgroup float4 * accum = shared + a;
	
	*accum = value;
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) {
			*accum += accum[b];
		}
	}
	if ( a ) {
		
	} else if ( rows_mask.w ) {
		*(device float4*)(y+row.x) = accum->xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(y+row.x) = accum->xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(y+row.x) = accum->xy;
	} else if ( rows_mask.x ) {
		*(device float *)(y+row.x) = accum->x;
	}
}
