//
//  Linear.metal
//  macOS
//
//  Created by Kota Nakano on 2017/01/28.
//
//

#include<metal_stdlib>
using namespace metal;
inline float4x4 sq(float4x4 x);
kernel void add(device float * const z [[ buffer(0) ]],
				device float const * const y [[ buffer(1) ]],
				device float const * const x [[ buffer(2) ]],
				constant uint const & length [[ buffer(3) ]],
				uint n [[ thread_position_in_grid ]]) {
	int const idx = n;
	z[idx] = y[idx] + x[idx];
}
kernel void sub(device float * const z [[ buffer(0) ]],
				device float const * const y [[ buffer(1) ]],
				device float const * const x [[ buffer(2) ]],
				constant uint const & length [[ buffer(3) ]],
				uint n [[ thread_position_in_grid ]]) {
	int const idx = n;
	z[idx] = y[idx] - x[idx];
}

kernel void mul(device float * const z [[ buffer(0) ]],
				device float const * const y [[ buffer(1) ]],
				device float const * const x [[ buffer(2) ]],
				constant uint const & length [[ buffer(3) ]],
				uint n [[ thread_position_in_grid ]]) {
	int const idx = n;
	z[idx] = y[idx] * x[idx];
}

kernel void div(device float * const z [[ buffer(0) ]],
				device float const * const y [[ buffer(1) ]],
				device float const * const x [[ buffer(2) ]],
				constant uint const & length [[ buffer(3) ]],
				uint n [[ thread_position_in_grid ]]) {
	int const idx = n;
	z[idx] = y[idx] / x[idx];
}
kernel void gemm(device float * const C [[ buffer(0) ]],
				 device float const * const A [[ buffer(1) ]],
				 device float const * const B [[ buffer(2) ]],
				 constant uint4 const & mnkl [[ buffer(3) ]],
				 threadgroup float4x4 * const sharedA [[ threadgroup(0) ]],
				 threadgroup float4x4 * const sharedB [[ threadgroup(1) ]],
				 uint2 const t [[ thread_position_in_threadgroup ]],
				 uint2 const g [[ threadgroup_position_in_grid ]]) {
	int const M = mnkl.x;
	int const N = mnkl.y;
	int const K = mnkl.z;
	int const L = mnkl.w;
	threadgroup float4x4 * const cacheA = sharedA + t.x * L;
	threadgroup float4x4 * const cacheB = sharedB + t.y * L;
	float4x4 ra, rb, rc(0);
	int2 const b = 4 * int2(g*L+t);
	bool4 const arm = b.x + int4(0, 1, 2, 3) < M;
	bool4 const bcm = b.y + int4(0, 1, 2, 3) < N;
	for ( int4 p = 4 * int4(t.x, t.y, 0, L) ; p.z < K ; p.xyz += p.w ) {
		bool4 const brm = p.x + int4(0, 1, 2, 3) < K;
		bool4 const acm = p.y + int4(0, 1, 2, 3) < K;
		for ( int3 row = int3(b.x, p.x, 0) ; row.z < 4 ; ++ row ) {
			ra [ row.z ] = select(0, *(device float4*)(A + row.x * K + p.y), arm [ row.z ] && acm );
			rb [ row.z ] = select(0, *(device float4*)(B + row.y * N + b.y), brm [ row.z ] && bcm );
			//for ( int3 col = int3(p.y, b.y, 0); col.z < 4 ; ++ col ) {
			//	ra [ row.z ] [ col.z ] = arm [ row.z ] && acm [ col.z ] ? A [ row.x * K + col.x ] : 0;
			//	rb [ row.z ] [ col.z ] = brm [ row.z ] && bcm [ col.z ] ? B [ row.y * N + col.y ] : 0;
			//}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		cacheA[t.y] = ra;
		cacheB[t.x] = rb;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( int l = 0 ; l < L ; ++ l ) {
			//	rc += cacheB[l] * cacheA[l];//slow
			ra = cacheA[l]; rb = cacheB[l];
			rc[0] += rb * ra[0];
			rc[1] += rb * ra[1];
			rc[2] += rb * ra[2];
			rc[3] += rb * ra[3];
		}
	}
	for ( int2 row = int2(0, b.x) ; row.x < 4 ; ++ row ) {
		for ( int2 col = int2(0, b.y) ; col.x < 4 ; ++ col ) {
			if ( arm [ row.x ] && bcm [ col.x ] ) C [ row.y * N + col.y ] = rc [ row.x ] [ col.x ];
		}
	}
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
