//
//  Linear.metal
//  macOS
//
//  Created by Kota Nakano on 2017/01/28.
//
//

#include <metal_stdlib>
using namespace metal;
inline float4x4 sq(float4x4 x);
kernel void add(device float4x4 * const z [[ buffer(0) ]],
				device float4x4 const * const y [[ buffer(1) ]],
				device float4x4 const * const x [[ buffer(2) ]],
				constant uint const & length [[ buffer(3) ]],
				uint n [[ thread_position_in_grid ]]) {
	int4 const ofs = int4(0, 1, 2, 3);
	int4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	float4x4 const yx = y[idx.x];
	float4x4 const yy = y[idx.y];
	float4x4 const yz = y[idx.z];
	float4x4 const yw = y[idx.w];
	float4x4 const xx = x[idx.x];
	float4x4 const xy = x[idx.y];
	float4x4 const xz = x[idx.z];
	float4x4 const xw = x[idx.w];
	if ( can.x ) z[idx.x] = yx + xx;
	if ( can.y ) z[idx.y] = yy + xy;
	if ( can.z ) z[idx.z] = yz + xz;
	if ( can.w ) z[idx.w] = yw + xw;
}
kernel void sub(device float4x4 * const z [[ buffer(0) ]],
				device float4x4 const * const y [[ buffer(1) ]],
				device float4x4 const * const x [[ buffer(2) ]],
				constant uint const & length [[ buffer(3) ]],
				uint n [[ thread_position_in_grid ]]) {
	int4 const ofs = int4(0, 1, 2, 3);
	int4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	float4x4 const yx = y[idx.x];
	float4x4 const yy = y[idx.y];
	float4x4 const yz = y[idx.z];
	float4x4 const yw = y[idx.w];
	float4x4 const xx = x[idx.x];
	float4x4 const xy = x[idx.y];
	float4x4 const xz = x[idx.z];
	float4x4 const xw = x[idx.w];
	if ( can.x ) z[idx.x] = yx - xx;
	if ( can.y ) z[idx.y] = yy - xy;
	if ( can.z ) z[idx.z] = yz - xz;
	if ( can.w ) z[idx.w] = yw - xw;
}
kernel void mul(device float4x4 * const z [[ buffer(0) ]],
				device float4x4 const * const y [[ buffer(1) ]],
				device float4x4 const * const x [[ buffer(2) ]],
				constant uint const & length [[ buffer(3) ]],
				uint const n [[ thread_position_in_grid ]]) {
	int4 const ofs = int4(0, 1, 2, 3);
	int4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	float4x4 const yx = y[idx.x];
	float4x4 const yy = y[idx.y];
	float4x4 const yz = y[idx.z];
	float4x4 const yw = y[idx.w];
	float4x4 const xx = x[idx.x];
	float4x4 const xy = x[idx.y];
	float4x4 const xz = x[idx.z];
	float4x4 const xw = x[idx.w];
	if ( can.x ) z[idx.x] = float4x4(yx[0]*xx[0], yy[1]*xy[1], yz[2]*xz[2], yw[3]*xw[3]);
	if ( can.y ) z[idx.y] = float4x4(yx[0]*xx[0], yy[1]*xy[1], yz[2]*xz[2], yw[3]*xw[3]);
	if ( can.z ) z[idx.z] = float4x4(yx[0]*xx[0], yy[1]*xy[1], yz[2]*xz[2], yw[3]*xw[3]);
	if ( can.w ) z[idx.w] = float4x4(yx[0]*xx[0], yy[1]*xy[1], yz[2]*xz[2], yw[3]*xw[3]);
}
kernel void div(device float4x4 * const z [[ buffer(0) ]],
				device float4x4 const * const y [[ buffer(1) ]],
				device float4x4 const * const x [[ buffer(2) ]],
				constant uint const & length [[ buffer(3) ]],
				uint n [[ thread_position_in_grid ]]) {
	int4 const ofs = int4(0, 1, 2, 3);
	int4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	float4x4 const yx = y[idx.x];
	float4x4 const yy = y[idx.y];
	float4x4 const yz = y[idx.z];
	float4x4 const yw = y[idx.w];
	float4x4 const xx = x[idx.x];
	float4x4 const xy = x[idx.y];
	float4x4 const xz = x[idx.z];
	float4x4 const xw = x[idx.w];
	if ( can.x ) z[idx.x] = float4x4(yx[0]/xx[0], yy[1]/xy[1], yz[2]/xz[2], yw[3]/xw[3]);
	if ( can.y ) z[idx.y] = float4x4(yx[0]/xx[0], yy[1]/xy[1], yz[2]/xz[2], yw[3]/xw[3]);
	if ( can.z ) z[idx.z] = float4x4(yx[0]/xx[0], yy[1]/xy[1], yz[2]/xz[2], yw[3]/xw[3]);
	if ( can.w ) z[idx.w] = float4x4(yx[0]/xx[0], yy[1]/xy[1], yz[2]/xz[2], yw[3]/xw[3]);
}
kernel void gemm16(device float4x4 * const C [[ buffer(0) ]],
				   device float4x4 const * const A [[ buffer(1) ]],
				   device float4x4 const * const B [[ buffer(2) ]],
				   constant uint4 const & mnkl [[ buffer(3) ]],
				   threadgroup float4x4 * const cacheC [[ threadgroup(0) ]],
				   threadgroup float4x4 * const cacheA [[ threadgroup(1) ]],
				   threadgroup float4x4 * const cacheB [[ threadgroup(2) ]],
				   uint const p [[ thread_index_in_threadgroup ]],
				   uint2 const g [[ threadgroup_position_in_grid ]]) {
	//	int const M = mnkl.x;
	int const N = mnkl.y;
	int const K = mnkl.z;
	int const L = mnkl.w;
	
	int const gx = g.x;
	int const gy = g.y;
	int const tx = p / 4;
	int const ty = p % 4;
	
	int4 const lx = 4 * tx + int4(0, 1, 2, 3);
	int4 const ly = ty + 4 * int4(0, 1, 2, 3);
	
	float4x4 c = float4x4(0);
	
	int const arow = gx * L + p;
	int const bcol = gy;
	
	for ( int k = 0 ; k < K ; ++ k ) {
		
		int const acol = k;
		int const brow = k * L + p;
		
		float4x4 const a = A [ arow * K + acol ];
		
		cacheA[lx.x][ty] = a[0];
		cacheA[lx.y][ty] = a[1];
		cacheA[lx.z][ty] = a[2];
		cacheA[lx.w][ty] = a[3];
		
		float4x4 const b = B [ brow * N + bcol ];
		
		cacheB[lx.x][ty] = b[0];
		cacheB[lx.y][ty] = b[1];
		cacheB[lx.z][ty] = b[2];
		cacheB[lx.w][ty] = b[3];
		
		threadgroup_barrier(mem_flags::mem_threadgroup);
		
		cacheC[p]
		= cacheB[ly.x] * cacheA[lx.x]
		+ cacheB[ly.y] * cacheA[lx.y]
		+ cacheB[ly.z] * cacheA[lx.z]
		+ cacheB[ly.w] * cacheA[lx.w];
		
		threadgroup_barrier(mem_flags::mem_threadgroup);
		
		c[0] += cacheC[lx.x][ty];
		c[1] += cacheC[lx.y][ty];
		c[2] += cacheC[lx.z][ty];
		c[3] += cacheC[lx.w][ty];
		
	}
	
	//threadgroup_barrier(mem_flags::mem_device);
	C [ arow * N + bcol ] = c;
	
}
kernel void gemv16(device float4x4 * const Y [[ buffer(0) ]],
				   device float4x4 const * const W [[ buffer(1) ]],
				   device float4x4 const * const X [[ buffer(2) ]],
				   constant uint const & length [[ buffer(3) ]],
				   threadgroup float4x4 * const accumulator [[ threadgroup(0) ]],
				   uint const t [[ thread_position_in_threadgroup ]],
				   uint const T [[ threads_per_threadgroup ]],
				   uint const m [[ threadgroup_position_in_grid ]]) {
	int4 const ref[4] = {
		( 16 * m + 0x0 + int4(0, 1, 2, 3)) * length,
		( 16 * m + 0x4 + int4(0, 1, 2, 3)) * length,
		( 16 * m + 0x8 + int4(0, 1, 2, 3)) * length,
		( 16 * m + 0xc + int4(0, 1, 2, 3)) * length
	};
	float4x4 v = float4x4(0);
	for ( int k = t, K = length ; k < K ; k += T ) {
		float4x4 const x = X[k];
		{
			int4 const idx = k + ref[0];
			float4x4 const w[4] = {W[idx.x], W[idx.y], W[idx.z], W[idx.w]};
			v[0] +=	x[0] * float4x4(w[0][0], w[1][0], w[2][0], w[3][0]) +
					x[1] * float4x4(w[0][1], w[1][1], w[2][1], w[3][1]) +
					x[2] * float4x4(w[0][2], w[1][2], w[2][2], w[3][2]) +
					x[3] * float4x4(w[0][3], w[1][3], w[2][3], w[3][3]);
		}
		{
			int4 const idx = k + ref[1];
			float4x4 const w[4] = {W[idx.x], W[idx.y], W[idx.z], W[idx.w]};
			v[1] +=	x[0] * float4x4(w[0][0], w[1][0], w[2][0], w[3][0]) +
					x[1] * float4x4(w[0][1], w[1][1], w[2][1], w[3][1]) +
					x[2] * float4x4(w[0][2], w[1][2], w[2][2], w[3][2]) +
					x[3] * float4x4(w[0][3], w[1][3], w[2][3], w[3][3]);
		}
		{
			int4 const idx = k + ref[2];
			float4x4 const w[4] = {W[idx.x], W[idx.y], W[idx.z], W[idx.w]};
			v[2] +=	x[0] * float4x4(w[0][0], w[1][0], w[2][0], w[3][0]) +
					x[1] * float4x4(w[0][1], w[1][1], w[2][1], w[3][1]) +
					x[2] * float4x4(w[0][2], w[1][2], w[2][2], w[3][2]) +
					x[3] * float4x4(w[0][3], w[1][3], w[2][3], w[3][3]);
		}
		{
			int4 const idx = k + ref[3];
			float4x4 const w[4] = {W[idx.x], W[idx.y], W[idx.z], W[idx.w]};
			v[3] +=	x[0] * float4x4(w[0][0], w[1][0], w[2][0], w[3][0]) +
					x[1] * float4x4(w[0][1], w[1][1], w[2][1], w[3][1]) +
					x[2] * float4x4(w[0][2], w[1][2], w[2][2], w[3][2]) +
					x[3] * float4x4(w[0][3], w[1][3], w[2][3], w[3][3]);
		}
	}
	int const a = t;
	int b = T;
	accumulator [ a ] = v;
	while ( b >>= 1 ) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( a < b ) accumulator[a] += accumulator[a+b];
	}
	if ( !a ) Y[m] = accumulator[a];
}
kernel void gemvt16(device float4x4 * const Y [[ buffer(0) ]],
					device float4x4 const * const W [[ buffer(1) ]],
					device float4x4 const * const X [[ buffer(2) ]],
					constant uint const & length [[ buffer(3) ]],
					threadgroup float4x4 * const accumulator [[ threadgroup(0) ]],
					uint const t [[ thread_position_in_threadgroup ]],
					uint const T [[ threads_per_threadgroup ]],
					uint const m [[ threadgroup_position_in_grid ]]) {
	int4 const ref[4] = {
		( 16 * m + 0x0 + int4(0, 1, 2, 3)) * length,
		( 16 * m + 0x4 + int4(0, 1, 2, 3)) * length,
		( 16 * m + 0x8 + int4(0, 1, 2, 3)) * length,
		( 16 * m + 0xc + int4(0, 1, 2, 3)) * length
	};
	float4x4 v = float4x4(0);
	for ( int k = t, K = length ; k < K ; k += T ) {
		float4x4 const x = X[k];
		{
			int4 const idx = k + ref[0];
			float4x4 const w[4] = {W[idx.x], W[idx.y], W[idx.z], W[idx.w]};
			v[0] +=	x[0] * float4x4(w[0][0], w[1][0], w[2][0], w[3][0]) +
			x[1] * float4x4(w[0][1], w[1][1], w[2][1], w[3][1]) +
			x[2] * float4x4(w[0][2], w[1][2], w[2][2], w[3][2]) +
			x[3] * float4x4(w[0][3], w[1][3], w[2][3], w[3][3]);
		}
		{
			int4 const idx = k + ref[1];
			float4x4 const w[4] = {W[idx.x], W[idx.y], W[idx.z], W[idx.w]};
			v[1] +=	x[0] * float4x4(w[0][0], w[1][0], w[2][0], w[3][0]) +
			x[1] * float4x4(w[0][1], w[1][1], w[2][1], w[3][1]) +
			x[2] * float4x4(w[0][2], w[1][2], w[2][2], w[3][2]) +
			x[3] * float4x4(w[0][3], w[1][3], w[2][3], w[3][3]);
		}
		{
			int4 const idx = k + ref[2];
			float4x4 const w[4] = {W[idx.x], W[idx.y], W[idx.z], W[idx.w]};
			v[2] +=	x[0] * float4x4(w[0][0], w[1][0], w[2][0], w[3][0]) +
			x[1] * float4x4(w[0][1], w[1][1], w[2][1], w[3][1]) +
			x[2] * float4x4(w[0][2], w[1][2], w[2][2], w[3][2]) +
			x[3] * float4x4(w[0][3], w[1][3], w[2][3], w[3][3]);
		}
		{
			int4 const idx = k + ref[3];
			float4x4 const w[4] = {W[idx.x], W[idx.y], W[idx.z], W[idx.w]};
			v[3] +=	x[0] * float4x4(w[0][0], w[1][0], w[2][0], w[3][0]) +
			x[1] * float4x4(w[0][1], w[1][1], w[2][1], w[3][1]) +
			x[2] * float4x4(w[0][2], w[1][2], w[2][2], w[3][2]) +
			x[3] * float4x4(w[0][3], w[1][3], w[2][3], w[3][3]);
		}
	}
	int const a = t;
	int b = T;
	accumulator [ a ] = v;
	while ( b >>= 1 ) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( a < b ) accumulator[a] += accumulator[a+b];
	}
	if ( !a ) Y[m] = accumulator[a];
}
inline float4x4 sq(float4x4 x) {
	return float4x4(x[0]*x[0], x[1]*x[1], x[2]*x[2], x[3]*x[3]);
}
