//
//  Gauss.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

inline float4x4 BoxMuller(float4x4,float4x4,float4x4);
inline float4x4 BoxMuller(float,float,float);
inline float4x4 exp(float4x4);
inline float4x4 step(float, float4x4);
inline float4x4 sqrt(float4x4);
inline float4x4 sq(float4x4 const);
inline float4x4 mul(float4x4, float4x4);
inline float4x4 div(float4x4, float4x4);
inline float4x4 rsqrt(float4x4);

constant uint3 const xorshift [[ function_constant(0) ]];
template<typename T> T sq(T x) {
	return x * x;
}
template<typename T> T erf(T z) {
	T const v = 1.0 / fma(fabs(z), 0.5, 1.0);
	//	T const e = 1.0-v*exp(-z*z-1.26551223+v*(1.00002368+v*(0.37409196+v*(0.09678418+v*(-0.18628806+v*(0.27886807+v*(-1.13520398+v*(1.48851587+v*(-0.82215223+v*(0.17087277))))))))));
	return copysign(fma(-v,
						exp(fma(v,
								fma(v,
									fma(v,
										fma(v,
											fma(v,
												fma(v,
													fma(v,
														fma(v,
															fma(v,
																0.17087277,
																-0.82215223),
															1.48851587),
														-1.13520398),
													0.27886807),
												-0.18628806),
											0.09678418),
										0.37409196),
									1.00002368),
								-z*z-1.26551223)),
						1),
					z);
}
kernel void GaussActivateP(device float * const YF [[ buffer(0) ]],
						   device float * const YP [[ buffer(1) ]],
						   device float const * const VU [[ buffer(2) ]],
						   device float const * const VS [[ buffer(3) ]],
						   constant uchar * const seed [[ buffer(4) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	YF[idx] = step(float(seed[idx])/256.0, YP[idx] = fma(erf(M_SQRT1_2_F*VU[idx]/VS[idx]), 0.5, 0.5));
}
kernel void GaussCollect(device float * const vu [[ buffer(0) ]],
						 device float * const vs [[ buffer(1) ]],
						 device float const * const su [[ buffer(2) ]],
						 device float const * const ss [[ buffer(3) ]],
						 uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	vu[idx] =      su[idx];
	vs[idx] = sqrt(ss[idx]);
}
kernel void GaussCollectW(device float * const su [[ buffer(0) ]],
						  device float * const ss [[ buffer(1) ]],
						  device float const * const wu [[ buffer(2) ]],
						  device float const * const ws [[ buffer(3) ]],
						  device float const * const x [[ buffer(4) ]],
						  threadgroup float4 * shared_u [[ threadgroup(0) ]],
						  threadgroup float4 * shared_s [[ threadgroup(1) ]],
						  constant uint2 & S [[ buffer(5) ]],
						  uint const t [[ thread_position_in_threadgroup ]],
						  uint const T [[ threads_per_threadgroup ]],
						  uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(S);
	
	float4 u = 0;
	float4 s = 0;
	
	int4 const row = 4 * n + int4(0, 1, 2, 3);
	bool4 const rows_mask = row < size.x;
	
	for ( int k = 4 * t, K = size.y ; k < K ; k += 4 * T ) {
		
		int4 const col = k + int4(0, 1, 2, 3);
		bool4 const cols_mask = col < size.y;
		/*
		 for ( int r = 0 ; r < 4 ; ++ r ) {
			for ( int c = 0 ; c < 4 ; ++ c ) {
		 um[r][c] = rows_mask[r] && cols_mask[c] ? w_mu[row[r]*size.y+col[c]] : 0;
		 sm[r][c] = rows_mask[r] && cols_mask[c] ? w_sigma[row[r]*size.y+col[c]] : 0;
			}
			f[r] = cols_mask[r] ? x[col[r]] : 0;
			sm[r] *= sm[r];
		}
		*/
		int4 const idx = row * size.y + k;
		
		float4 const f = select(0, *(device float4*)(x + k), cols_mask);
		
		u += f * float4x4(select(0, *(device float4*)(wu + idx.x), rows_mask.x && cols_mask),
						  select(0, *(device float4*)(wu + idx.y), rows_mask.y && cols_mask),
						  select(0, *(device float4*)(wu + idx.z), rows_mask.z && cols_mask),
						  select(0, *(device float4*)(wu + idx.w), rows_mask.w && cols_mask));
		
		s += sq(f) * float4x4(sq(select(0, *(device float4*)(ws + idx.x), rows_mask.x && cols_mask)),
							  sq(select(0, *(device float4*)(ws + idx.y), rows_mask.y && cols_mask)),
							  sq(select(0, *(device float4*)(ws + idx.z), rows_mask.z && cols_mask)),
							  sq(select(0, *(device float4*)(ws + idx.w), rows_mask.w && cols_mask)));
		
	}
	
	int const a = t;
	int b = T;
	
	threadgroup float4 * accum_u = shared_u + a;
	threadgroup float4 * accum_s = shared_s + a;
	
	*accum_u = u;
	*accum_s = s;
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) {
			*accum_u += accum_u[b];
			*accum_s += accum_s[b];
		}
	}
	if ( a ) {
	
	} else if ( rows_mask.w ) {
		*(device float4*)(su+row.x) += accum_u->xyzw;
		*(device float4*)(ss+row.x) += accum_s->xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(su+row.x) += accum_u->xyz;
		*(device float3*)(ss+row.x) += accum_s->xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(su+row.x) += accum_u->xy;
		*(device float2*)(ss+row.x) += accum_s->xy;
	} else if ( rows_mask.x ) {
		*(device float *)(su+row.x) += accum_u->x;
		*(device float *)(ss+row.x) += accum_s->x;
	}
}
kernel void GaussCollectC(device float * const su [[ buffer(0) ]],
						  device float * const ss [[ buffer(1) ]],
						  device float const * const cu [[ buffer(2) ]],
						  device float const * const cs [[ buffer(3) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	su[idx] +=    cu[idx];
	ss[idx] += sq(cs[idx]);
}
kernel void GaussCollectD(device float * const su [[ buffer(0) ]],
						  device float * const ss [[ buffer(1) ]],
						  device float const * const d [[ buffer(2) ]],
						  device float const * const vu [[ buffer(3) ]],
						  device float const * const vs [[ buffer(4) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float const r = d[idx];
	su[idx] +=    r*vu[idx];
	ss[idx] += sq(r*vs[idx]);
}
constant float const M_SQRT1_2PI_F = 0.5 * M_2_SQRTPI_F * M_SQRT1_2_F;
kernel void GaussDerivateP(device float * const dU [[ buffer(0) ]],
						   device float * const dS [[ buffer(1) ]],
						   device float * const gU [[ buffer(2) ]],
						   device float * const gS [[ buffer(3) ]],
						   device float const * const D [[ buffer(4) ]],
						   device float const * const P [[ buffer(5) ]],
						   device float const * const U [[ buffer(6) ]],
						   device float const * const S [[ buffer(7) ]],
						   constant uint const & length [[ buffer(8) ]],
						   uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float const u = U[idx];
	float const s = S[idx];
	float const x = u / s;
	//float const p = P[idx];
	float const d = D[idx];// * select(1.0, p * ( 1.0 - p ), 0.0 < p && p < 1.0);
	float const g = M_SQRT1_2PI_F * exp( -0.5 * x * x );
	float const gu = g / s;
	float const gs = gu * -x;
	dU[idx] = d * (gU[idx] = gu);
	dS[idx] = d * (gS[idx] = gs);
}
kernel void GaussJacobian(device float * const ju [[ buffer(0) ]],
						  device float * const js [[ buffer(1) ]],
						  device float const * const u [[ buffer(2) ]],
						  device float const * const s [[ buffer(3) ]],
						  device float const * const su [[ buffer(4) ]],
						  device float const * const ss [[ buffer(5) ]],
						  constant uint & ld [[ buffer(6) ]],
						  uint2 const ij [[ thread_position_in_grid ]]) {
	int const rows = ij.x;
	int const cols = ij.y;
	int const idx = rows * ld + cols;
	ju[idx] = su[idx];
	js[idx] = ss[idx] / s[rows];
}
kernel void GaussJacobianA(device float * const ju [[ buffer(0) ]],
						   device float * const js [[ buffer(1) ]],
						   device float const * const au [[ buffer(2) ]],
						   device float const * const as [[ buffer(3) ]],
						   device float const * const x [[ buffer(4) ]],
						   constant uint2 & ld [[ buffer(5) ]],
						   uint2 const ij [[ thread_position_in_grid ]]) {
	int const rows = ij.x;
	int const cols = ij.y;
	int const idx = cols + ld.y * rows * ld.x;
	float const xv = x [ cols ];
	float const sv = as [ cols + ld.y * rows ];
	ju [ idx ] += xv;
	js [ idx ] += xv * xv * sv;
}
kernel void GaussJacobianB(device float * const ju [[ buffer(0) ]],
						   device float * const js [[ buffer(1) ]],
						   device float const * const Bu [[ buffer(2) ]],
						   device float const * const Bs [[ buffer(3) ]],
						   device float const * const Y [[ buffer(4) ]],
						   device float const * const Ju [[ buffer(5) ]],
						   device float const * const Js [[ buffer(6) ]],
						   device float const * const Pu [[ buffer(7) ]],
						   device float const * const Ps [[ buffer(8) ]],
						   constant uint4 const & mnkl [[ buffer(9) ]],
						   threadgroup float4x4 * const sharedB [[ threadgroup(0) ]],
						   threadgroup float4x4 * const sharedP [[ threadgroup(1) ]],
						   uint2 const t [[ thread_position_in_threadgroup ]],
						   uint2 const g [[ threadgroup_position_in_grid ]]) {
	int const M = mnkl.x;
	int const N = mnkl.y;
	int const K = mnkl.z;
	int const L = mnkl.w;
	int const tx = t.x;
	int const ty = t.y;
	threadgroup float4x4 * const cacheB = sharedB + tx * L;
	threadgroup float4x4 * const cacheP = sharedP + ty * L;
	float4x4 ru(0), rs(0);
	int2 const b = 4 * int2( g * L + t );
	bool4 const brm = b.x + int4(0, 1, 2, 3) < M;
	bool4 const pcm = b.y + int4(0, 1, 2, 3) < N;
	for ( int4 p = 4 * int4(t.x, t.y, 0, L) ; p.z < K ; p.xyz += p.w ) {
		float4x4 rbu, rpu, rbs, rps;
		bool4 const prm = p.x + int4(0, 1, 2, 3) < K;
		bool4 const bcm = p.y + int4(0, 1, 2, 3) < K;
		for ( int3 row = int3(b.x, p.x, 0) ; row.z < 4 ; ++ row ) {
			for ( int3 col = int3(p.y, b.y, 0); col.z < 4 ; ++ col ) {
				rbu [ row.z ] [ col.z ] = brm [ row.z ] && bcm [ col.z ] ?    Bu [ row.x * K + col.x ]  * Ju[col.x] : 0;
				rbs [ row.z ] [ col.z ] = brm [ row.z ] && bcm [ col.z ] ? sq(Bs [ row.x * K + col.x ]) * Js[col.x] * Y[col.x] : 0;
				rpu [ row.z ] [ col.z ] = prm [ row.z ] && pcm [ col.z ] ? Pu [ row.y * N + col.y ] : 0;
				rps [ row.z ] [ col.z ] = prm [ row.z ] && pcm [ col.z ] ? Ps [ row.y * N + col.y ] : 0;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		cacheB[ty] = rbu;
		cacheP[tx] = rpu;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( int l = 0 ; l < L ; ++ l ) {
			//	ru += cacheB[l] * cacheA[l];//slow
			rbu = cacheB[l]; rpu = cacheP[l];
			ru[0] += rpu * rbu[0];
			ru[1] += rpu * rbu[1];
			ru[2] += rpu * rbu[2];
			ru[3] += rpu * rbu[3];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		cacheB[ty] = rbs;
		cacheP[tx] = rps;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( int l = 0 ; l < L ; ++ l ) {
			//	rs += cacheB[l] * cacheA[l];//slow
			rbs = cacheB[l]; rps = cacheP[l];
			rs[0] += rps * rbs[0];
			rs[1] += rps * rbs[1];
			rs[2] += rps * rbs[2];
			rs[3] += rps * rbs[3];
		}
	}
	for ( int2 row = int2(0, b.x) ; row.x < 4 ; ++ row ) {
		for ( int2 col = int2(0, b.y) ; col.x < 4 ; ++ col ) {
			if ( brm [ row.x ] && pcm [ col.x ] ) {
				ju [ row.y * N + col.y ] += ru [ row.x ] [ col.x ];
				js [ row.y * N + col.y ] += rs [ row.x ] [ col.x ];
			}
		}
	}
}
kernel void GaussJacobianC(device float * const ju [[ buffer(0) ]],
						   device float * const js [[ buffer(1) ]],
						   device float const * const cu [[ buffer(2) ]],
						   device float const * const cs [[ buffer(3) ]],
						   constant uint & ld [[buffer(4) ]],
						   uint const k [[ thread_position_in_grid ]]) {
	int const idx = k * ld;
	ju[idx] += 1.0;
	js[idx] += cs[k];
}
kernel void GaussJacobianD(device float * const ju [[ buffer(0) ]],
						   device float * const js [[ buffer(1) ]],
						   device float * const d [[ buffer(2) ]],
						   device float const * const pu [[ buffer(3) ]],
						   device float const * const ps [[ buffer(4) ]],
						   constant uint & ld [[buffer(5) ]],
						   uint2 const ij [[ thread_position_in_grid ]]) {
	int const rows = ij.x;
	int const cols = ij.y;
	int const idx = rows * ld + cols;
	float const r = d[rows];
	ju[idx] +=    r*pu[idx];
	js[idx] += sq(r*ps[idx]);
}
kernel void GaussJacobianX(device float * const ju [[ buffer(0) ]],
						   device float * const js [[ buffer(1) ]],
						   device float const * const x [[ buffer(2) ]],
						   device float const * const wu [[ buffer(3) ]],
						   device float const * const ws [[ buffer(4) ]],
						   constant uint & ld [[ buffer(5) ]],
						   uint2 const ij [[ thread_position_in_grid ]]) {
	int const rows = ij.x;
	int const cols = ij.y;
	int const idx = rows * ld + cols;
	float const u = wu[idx];
	float const s = ws[idx];
	ju[idx] += u;
	js[idx] += s * s * x[cols];
}
kernel void GaussDeltaW(device float * const du [[ buffer(0) ]],
						device float * const ds [[ buffer(1) ]],
						device float const * const wu [[ buffer(2) ]],
						device float const * const ws [[ buffer(3) ]],
						device float const * const x [[ buffer(4) ]],
						device float const * const vu [[ buffer(5) ]],
						device float const * const vs [[ buffer(6) ]],
						device float const * const dypdvu [[ buffer(7) ]],
						device float const * const dypdvs [[ buffer(8) ]],
						constant uint2 & IJ [[ buffer(9) ]],
						uint2 const ij [[ thread_position_in_grid ]]) {
	int const rows = ij.x;
	int const cols = ij.y;
	int const idx = rows * IJ.y + cols;
	float const f = x[cols];
	du[idx] = dypdvu[rows] * f;
	ds[idx] = dypdvs[rows] * f * f * ws[idx] / vs[rows];
}
kernel void GaussDeltaC(device float * const du [[ buffer(0) ]],
						device float * const ds [[ buffer(1) ]],
						device float const * const cu [[ buffer(2) ]],
						device float const * const cs [[ buffer(3) ]],
						device float const * const vu [[ buffer(4) ]],
						device float const * const vs [[ buffer(5) ]],
						device float const * const dypdvu [[ buffer(6) ]],
						device float const * const dypdvs [[ buffer(7) ]],
						uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	du[idx] = dypdvu[idx];
	ds[idx] = dypdvs[idx] * cs[idx] / vs[idx];
}
/*
 sugar function to compute Δx <- Δμ(y), Δσ(y), μ(v), σ(v), μ(w), σ(w), x  
 */
kernel void GaussDeltaS(device float * const dx [[ buffer(0) ]],
						device float const * const x [[ buffer(1) ]],
						device float const * const wu [[ buffer(2) ]],
						device float const * const ws [[ buffer(3) ]],
						device float const * const vu [[ buffer(4) ]],
						device float const * const vs [[ buffer(5) ]],
						device float const * const dypdvu [[ buffer(6) ]],
						device float const * const dypdvs [[ buffer(7) ]],
						constant uint2 & S [[ buffer(8) ]],
						threadgroup float4 * shared [[ threadgroup(0) ]],
						uint const t [[ thread_position_in_threadgroup ]],
						uint const T [[ threads_per_threadgroup ]],
						uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(S);
	
	float4 u = 0;
	float4 s = 0;
	
	int4 const row = 4 * n + int4(0, 1, 2, 3);
	bool4 const rows_mask = row < size.x;
	
	for ( int k = 4 * t, K = size.y ; k < K ; k += 4 * T ) {
		
		int4 const col = k + int4(0, 1, 2, 3);
		bool4 const cols_mask = col < size.y;
		
		int4 const idx = col * size.x + row.x;
		
		u +=
		float4x4(select(0, *(device float4*)(wu + idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(wu + idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(wu + idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(wu + idx.w), rows_mask && cols_mask.w)) *
		(select(0, *(device float4*)(dypdvu + k), cols_mask));
		
		s +=
		float4x4(sq(select(0, *(device float4*)(ws + idx.x), rows_mask && cols_mask.x)),
				 sq(select(0, *(device float4*)(ws + idx.y), rows_mask && cols_mask.y)),
				 sq(select(0, *(device float4*)(ws + idx.z), rows_mask && cols_mask.z)),
				 sq(select(0, *(device float4*)(ws + idx.w), rows_mask && cols_mask.w))) *
		(select(0, *(device float4*)(dypdvs + k), cols_mask) / select(0, *(device float4*)(vs + k), cols_mask));
		
	}

	int const a = t;
	int b = T;
	
	threadgroup float4 * accum = shared + a;
	*accum = fma(select(0, *(device float4*)(x+row.x), rows_mask), s, u);
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) *accum += accum[b];
	}
	
	if ( a );
	else if ( rows_mask.w ) *(device float4*)(dx+row.x) += accum->xyzw;
	else if ( rows_mask.z ) *(device float3*)(dx+row.x) += accum->xyz;
	else if ( rows_mask.y ) *(device float2*)(dx+row.x) += accum->xy;
	else if ( rows_mask.x ) *(device float *)(dx+row.x) += accum->x;
}
kernel void GaussDeltaG(device float * const du [[ buffer(0) ]],
						device float * const ds [[ buffer(1) ]],
						device float const * const ju [[ buffer(2) ]],
						device float const * const js [[ buffer(3) ]],
						device float const * const dydu [[ buffer(4) ]],
						device float const * const dyds [[ buffer(5) ]],
						constant uint & ld [[ buffer(6) ]],
						uint2 const ij [[ thread_position_in_grid ]]) {
	int const rows = ij.x;
	int const cols = ij.y;
	int const idx = rows * ld + cols;
	du[idx] = ju[idx] * dydu[rows];
	ds[idx] = js[idx] * dyds[rows];
}
kernel void GaussDeltaJ(device float * const du [[ buffer(0) ]],
						device float * const ds [[ buffer(1) ]],
						device float const * const ju [[ buffer(2) ]],
						device float const * const js [[ buffer(3) ]],
						device float const * const dydu [[ buffer(4) ]],
						device float const * const dyds [[ buffer(5) ]],
						constant uint2 & S [[ buffer(6) ]],
						threadgroup float4 * shared_u [[ threadgroup(0) ]],
						threadgroup float4 * shared_s [[ threadgroup(1) ]],
						uint const t [[ thread_position_in_threadgroup ]],
						uint const T [[ threads_per_threadgroup ]],
						uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(S);
	
	float4 u = 0;
	float4 s = 0;
	
	int4 const row = 4 * n + int4(0, 1, 2, 3);
	bool4 const rows_mask = row < size.x;
	
	for ( int k = 4 * t, K = size.y ; k < K ; k += 4 * T ) {
		
		int4 const col = k + int4(0, 1, 2, 3);
		bool4 const cols_mask = col < size.y;

		int4 const idx = col * size.x + row.x;
		
		u +=
		float4x4(select(0, *(device float4*)(ju + idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(ju + idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(ju + idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(ju + idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(dydu + k), cols_mask);
		
		s +=
		float4x4(select(0, *(device float4*)(js + idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(js + idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(js + idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(js + idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(dyds + k), cols_mask);
	}
	
	int const a = t;
	int b = T;
	
	threadgroup float4 * accum_u = shared_u + a;
	threadgroup float4 * accum_s = shared_s + a;
	
	*accum_u = u;
	*accum_s = s;
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) {
			*accum_u += accum_u[b];
			*accum_s += accum_s[b];
		}
	}
	if ( a ) {
		
	} else if ( rows_mask.w ) {
		*(device float4*)(du+row.x) += accum_u->xyzw;
		*(device float4*)(ds+row.x) += accum_s->xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(du+row.x) += accum_u->xyz;
		*(device float3*)(ds+row.x) += accum_s->xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(du+row.x) += accum_u->xy;
		*(device float2*)(ds+row.x) += accum_s->xy;
	} else if ( rows_mask.x ) {
		*(device float *)(du+row.x) += accum_u->x;
		*(device float *)(ds+row.x) += accum_s->x;
	}
}
kernel void GaussDeltaX(device float * const dx [[ buffer(0) ]],
						device float const * const ju [[ buffer(1) ]],
						device float const * const js [[ buffer(2) ]],
						device float const * const dydu [[ buffer(3) ]],
						device float const * const dyds [[ buffer(4) ]],
						constant uint2 & S [[ buffer(5) ]],
						threadgroup float4 * shared [[ threadgroup(0) ]],
						uint const t [[ thread_position_in_threadgroup ]],
						uint const T [[ threads_per_threadgroup ]],
						uint const n [[ threadgroup_position_in_grid ]]) {
	
	int2 const size = int2(S);
	
	float4 x = 0;
	
	int4 const row = 4 * n + int4(0, 1, 2, 3);
	bool4 const rows_mask = row < size.x;
	
	for ( int k = 4 * t, K = size.y ; k < K ; k += 4 * T ) {
		
		int4 const col = k + int4(0, 1, 2, 3);
		bool4 const cols_mask = col < size.y;
		
		int4 const idx = col * size.x + row.x;
		
		x +=
		float4x4(select(0, *(device float4*)(ju + idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(ju + idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(ju + idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(ju + idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(dydu + k), cols_mask);
		
		x +=
		float4x4(select(0, *(device float4*)(js + idx.x), rows_mask && cols_mask.x),
				 select(0, *(device float4*)(js + idx.y), rows_mask && cols_mask.y),
				 select(0, *(device float4*)(js + idx.z), rows_mask && cols_mask.z),
				 select(0, *(device float4*)(js + idx.w), rows_mask && cols_mask.w)) * select(0, *(device float4*)(dyds + k), cols_mask);
	}
	
	int const a = t;
	int b = T;
	
	threadgroup float4 * accum = shared + a;
	
	*accum = x;
	
	while ( b >>= 1 ) {
		threadgroup_barrier( mem_flags :: mem_threadgroup );
		if ( a < b ) {
			*accum += accum[b];
		}
	}
	if ( a ) {
		
	} else if ( rows_mask.w ) {
		*(device float4*)(dx+row.x) += accum->xyzw;
	} else if ( rows_mask.z ) {
		*(device float3*)(dx+row.x) += accum->xyz;
	} else if ( rows_mask.y ) {
		*(device float2*)(dx+row.x) += accum->xy;
	} else if ( rows_mask.x ) {
		*(device float *)(dx+row.x) += accum->x;
	}
}
kernel void GaussJacobianMuA(device float * const G [[ buffer(0) ]],
							 device float const * const X [[ buffer(1) ]],
							 constant uint2 & IJ [[ buffer(2) ]],
							 uint2 const ij [[ thread_position_in_grid ]]) {
	int const i = ij.x;
	int const j = ij.y;
	int const ldx = IJ.y * ( IJ.x + 1 );
	G [ i * ldx + j ] += X [ j ];
}
kernel void GaussJacobianMuB(device float * const G [[ buffer(0) ]],
							 device float const * const B [[ buffer(1) ]],
							 device float const * const J [[ buffer(2) ]],
							 device float const * const P [[ buffer(3) ]],
							 constant uint4 const & mnkl [[ buffer(4) ]],
							 threadgroup float4x4 * const sharedB [[ threadgroup(0) ]],
							 threadgroup float4x4 * const sharedP [[ threadgroup(1) ]],
							 uint2 const t [[ thread_position_in_threadgroup ]],
							 uint2 const g [[ threadgroup_position_in_grid ]]) {
	int const M = mnkl.x;
	int const N = mnkl.y;
	int const K = mnkl.z;
	int const L = mnkl.w;
	int const tx = t.x;
	int const ty = t.y;
	threadgroup float4x4 * const cacheB = sharedB + tx * L;
	threadgroup float4x4 * const cacheP = sharedP + ty * L;
	float4x4 rb, rp, rg(0);
	int2 const b = 4 * int2( g * L + t );
	bool4 const brm = b.x + int4(0, 1, 2, 3) < M;
	bool4 const pcm = b.y + int4(0, 1, 2, 3) < N;
	for ( int4 p = 4 * int4(t.x, t.y, 0, L) ; p.z < K ; p.xyz += p.w ) {
		bool4 const prm = p.x + int4(0, 1, 2, 3) < K;
		bool4 const bcm = p.y + int4(0, 1, 2, 3) < K;
		for ( int3 row = int3(b.x, p.x, 0) ; row.z < 4 ; ++ row ) {
			//ra [ row.z ] = select(0, *(device float4*)(A + row.x * K + p.y), arm [ row.z ] && acm );
			//rb [ row.z ] = select(0, *(device float4*)(B + row.y * N + b.y), brm [ row.z ] && bcm );
			for ( int3 col = int3(p.y, b.y, 0); col.z < 4 ; ++ col ) {
				rb [ row.z ] [ col.z ] = brm [ row.z ] && bcm [ col.z ] ? B [ row.x * K + col.x ] * J [ col.x ] : 0;
				rp [ row.z ] [ col.z ] = prm [ row.z ] && pcm [ col.z ] ? P [ row.y * N + col.y ] : 0;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		cacheB[ty] = rb;
		cacheP[tx] = rp;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( int l = 0 ; l < L ; ++ l ) {
			//	rc += cacheB[l] * cacheA[l];//slow
			rb = cacheB[l]; rp = cacheP[l];
			rg[0] += rp * rb[0];
			rg[1] += rp * rb[1];
			rg[2] += rp * rb[2];
			rg[3] += rp * rb[3];
		}
	}
	for ( int2 row = int2(0, b.x) ; row.x < 4 ; ++ row ) {
		for ( int2 col = int2(0, b.y) ; col.x < 4 ; ++ col ) {
			if ( brm [ row.x ] && pcm [ col.x ] ) G [ row.y * N + col.y ] += rg [ row.x ] [ col.x ];
		}
	}
}
kernel void GaussJacobianMuC(device float * const G [[ buffer(0) ]],
							 device float const * const C [[ buffer(1) ]],
							 constant uint & K [[ buffer(2) ]],
							 uint const k [[ thread_position_in_grid ]]) {
	G [ k * ( K + 1 ) ] += 1.0;
}
kernel void GaussJacobianMuD(device float * const G [[ buffer(0) ]],
							 device float const * const D [[ buffer(1) ]],
							 device float const * const P [[ buffer(2) ]],
							 constant uint2 & IJ [[ buffer(3) ]],
							 uint2 const ij [[ thread_position_in_grid ]]) {
	int const i = ij.x;//, I = IJ.x;
	int const j = ij.y, J = IJ.y;
	G [ i * J + j ] += D [ i ] * P [ i * J + j ];
}
kernel void GaussJacobianMuX(device float * const G [[ buffer(0) ]],
							 device float const * const W [[ buffer(1) ]],
							 constant uint2 & IJ [[ buffer(2) ]],
							 uint2 const ij [[ thread_position_in_grid ]]) {
	int const i = ij.x;//, I = IJ.x;
	int const j = ij.y, J = IJ.y;
	G [ i * J + j ] += W [ i * J + j ];
}
kernel void GaussJacobianSigmaA(device float * const g [[ buffer(0) ]],
								device float const * const v [[ buffer(1) ]],
								device float const * const a [[ buffer(2) ]],
								device float const * const x [[ buffer(3) ]],
								uint2 const ij [[ thread_position_in_grid ]],
								uint2 const IJ [[ threads_per_grid ]]) {
	int const i = ij.x, I = IJ.x;
	int const j = ij.y, J = IJ.y;
	int const ldx = J * ( I + 1 );
	float const xv = x [ j ];
	g [ i * ldx + j ] = rsqrt ( v [ i ] ) * a [ i * J + j ] * xv * xv;
}
kernel void GaussJacobianSigmaB(device float * const G [[ buffer(0) ]],
							 device float const * const V [[ buffer(1) ]],
							 device float const * const B [[ buffer(2) ]],
							 device float const * const Y [[ buffer(3) ]],
							 device float const * const J [[ buffer(4) ]],
							 device float const * const P [[ buffer(5) ]],
							 constant uint4 const & mnkl [[ buffer(6) ]],
							 threadgroup float4x4 * const sharedB [[ threadgroup(0) ]],
							 threadgroup float4x4 * const sharedP [[ threadgroup(1) ]],
							 uint2 const t [[ thread_position_in_threadgroup ]],
							 uint2 const g [[ threadgroup_position_in_grid ]]) {
	int const M = mnkl.x;
	int const N = mnkl.y;
	int const K = mnkl.z;
	int const L = mnkl.w;
	int const tx = t.x;
	int const ty = t.y;
	threadgroup float4x4 * const cacheB = sharedB + tx * L;
	threadgroup float4x4 * const cacheP = sharedP + ty * L;
	float4x4 rb, rp, rg(0);
	int2 const b = 4 * int2( g * L + t );
	bool4 const brm = b.x + int4(0, 1, 2, 3) < M;
	bool4 const pcm = b.y + int4(0, 1, 2, 3) < N;
	for ( int4 p = 4 * int4(t.x, t.y, 0, L) ; p.z < K ; p.xyz += p.w ) {
		bool4 const prm = p.x + int4(0, 1, 2, 3) < K;
		bool4 const bcm = p.y + int4(0, 1, 2, 3) < K;
		for ( int3 row = int3(b.x, p.x, 0) ; row.z < 4 ; ++ row ) {
			//ra [ row.z ] = select(0, *(device float4*)(A + row.x * K + p.y), arm [ row.z ] && acm );
			//rb [ row.z ] = select(0, *(device float4*)(B + row.y * N + b.y), brm [ row.z ] && bcm );
			for ( int3 col = int3(p.y, b.y, 0); col.z < 4 ; ++ col ) {
				rb [ row.z ] [ col.z ] = brm [ row.z ] && bcm [ col.z ] ? B [ row.x * K + col.x ] * rsqrt( V [ row.x ] ) * Y [ col.x ] * J [ col.x ] : 0;
				rp [ row.z ] [ col.z ] = prm [ row.z ] && pcm [ col.z ] ? P [ row.y * N + col.y ] : 0;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		cacheB[ty] = rb;
		cacheP[tx] = rp;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( int l = 0 ; l < L ; ++ l ) {
			//	rc += cacheB[l] * cacheA[l];//slow
			rb = cacheB[l]; rp = cacheP[l];
			rg[0] += rp * rb[0];
			rg[1] += rp * rb[1];
			rg[2] += rp * rb[2];
			rg[3] += rp * rb[3];
		}
	}
	for ( int2 row = int2(0, b.x) ; row.x < 4 ; ++ row ) {
		for ( int2 col = int2(0, b.y) ; col.x < 4 ; ++ col ) {
			if ( brm [ row.x ] && pcm [ col.x ] ) G [ row.y * N + col.y ] += rg [ row.x ] [ col.x ];
		}
	}
}

kernel void GaussJacobianSigmaC(device float * const G [[ buffer(0) ]],
								device float const * const V [[ buffer(1) ]],
								device float const * const C [[ buffer(2) ]],
								constant uint & K [[ buffer(3) ]],
								uint const k [[ thread_position_in_grid ]]) {
	float const l = rsqrt( V [ k ] );
	float const c = C [ k ];
	G [ k * ( K + 1 ) ] += l * c;
}
kernel void GaussJacobianSigmaD(device float * const G [[ buffer(0) ]],
								device float const * const V [[ buffer(1) ]],
								device float const * const D [[ buffer(2) ]],
								device float const * const S [[ buffer(3) ]],
								device float const * const P [[ buffer(4) ]],
								constant uint2 & IJ [[ buffer(5) ]],
								uint2 const ij [[ thread_position_in_grid ]]) {
	int const i = ij.x;//, I = IJ.x;
	int const j = ij.y, J = IJ.y;
	float const l = rsqrt( V [ i ] );
	float const d = D [ i ];
	float const s = S [ i ];
	float const p = P [ i * J + j ];
	G [ i * J + j ] += l * d * d * s * p;
}
kernel void GaussJacobianSigmaX(device float * const G [[ buffer(0) ]],
								device float const * const V [[ buffer(1) ]],
								device float const * const W [[ buffer(2) ]],
								device float const * const X [[ buffer(3) ]],
								constant uint2 & IJ [[ buffer(4) ]],
								uint2 const ij [[ thread_position_in_grid ]]) {
	int const i = ij.x;//, I = IJ.x;
	int const j = ij.y, J = IJ.y;
	float const l = rsqrt( V [ i ] );
	float const w = W [ i * J + j ];
	float const x = X [ j ];
	G [ i * J + j ] += l * w * w * x;
}
kernel void GaussJacobianFinalize(device float * const jmu [[ buffer(0) ]],
								  device float * const jsigma [[ buffer(1) ]],
								  device float const * const mu [[ buffer(2) ]],
								  device float const * const sigma [[ buffer(3) ]],
								  device float const * const sum_jmu [[ buffer(4) ]],
								  device float const * const sum_jsigma [[ buffer(5) ]],
								  constant uint3 & KIJ [[ buffer(6) ]],
								  uint2 const kij [[ thread_position_in_grid ]]) {
	
}
kernel void GaussGradient16(device float4x4 * const delta_mu [[ buffer(0) ]],
							device float4x4 * const delta_sigma [[ buffer(1) ]],
							device float4x4 const * const mu [[ buffer(2) ]],
							device float4x4 const * const sigma [[ buffer(3) ]],
							uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float4x4 const u = mu[idx];
	float4x4 const s = sigma[idx];
	float4x4 const x = float4x4(u[0]/s[0],
								u[1]/s[1],
								u[2]/s[2],
								u[3]/s[3]);
	float4x4 const g = M_SQRT1_2PI_F * float4x4(exp(-0.5*x[0]*x[0])/s[0],
												exp(-0.5*x[1]*x[1])/s[1],
												exp(-0.5*x[2]*x[2])/s[2],
												exp(-0.5*x[3]*x[3])/s[3]);
	delta_mu[idx] = g;
	delta_sigma[idx] = float4x4(-g[0]*x[0],
								-g[1]*x[1],
								-g[2]*x[2],
								-g[3]*x[3]);
}

kernel void GaussErrorState16(device float4x4 * const error [[ buffer(0) ]],
							  device float4x4 const * const target [[ buffer(1) ]],
							  device float4x4 const * const mu [[ buffer(2) ]],
							  device float4x4 const * const variance [[ buffer(3) ]],
							  uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float4x4 const t = target[idx];
	float4x4 const u = mu[idx];
	float4x4 const v = variance[idx];
	float4x4 const x = M_SQRT1_2_F * float4x4(u[0]*rsqrt(v[0]),
											  u[1]*rsqrt(v[1]),
											  u[2]*rsqrt(v[2]),
											  u[3]*rsqrt(v[3]));
	error[idx] = float4x4(t[0]-0.5-0.5*erf(x[0]),
						  t[1]-0.5-0.5*erf(x[1]),
						  t[2]-0.5-0.5*erf(x[2]),
						  t[3]-0.5-0.5*erf(x[3]));
}

kernel void GaussErrorValue16(device float4x4 * const error [[ buffer(0) ]],
							  device float4x4 const * const target [[ buffer(1) ]],
							  device float4x4 const * const mu [[ buffer(2) ]],
							  device float4x4 const * const variance [[ buffer(3) ]],
							  uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	error[idx] = target[idx] - mu[idx];
}

kernel void GaussDeltaState16(device float4x4 * const delta_mu [[ buffer(0) ]],
							  device float4x4 * const delta_sigma [[ buffer(1) ]],
							  device float4x4 const * const mu [[ buffer(2)  ]],
							  device float4x4 const * const sigma [[ buffer(3) ]],
							  device float4x4 const * const delta [[ buffer(4) ]],
							  uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float4x4 const u = mu[idx];
	float4x4 const s = sigma[idx];
	float4x4 const d = delta[idx];
	float4x4 const x = float4x4(u[0]/s[0],
								u[1]/s[1],
								u[2]/s[2],
								u[3]/s[3]);
	float4x4 const g = M_SQRT1_2PI_F * float4x4(d[0]*exp(-0.5*x[0]*x[0])/s[0],
												d[1]*exp(-0.5*x[1]*x[1])/s[1],
												d[2]*exp(-0.5*x[2]*x[2])/s[2],
												d[3]*exp(-0.5*x[3]*x[3])/s[3]);
	delta_mu[idx] = g;
	delta_sigma[idx] = float4x4(-g[0]*x[0],
								-g[1]*x[1],
								-g[2]*x[2],
								-g[3]*x[3]);
}

kernel void GaussDeltaValue16(device float4x4 * const delta_mu [[ buffer(0) ]],
							  device float4x4 * const delta_sigma [[ buffer(1) ]],
							  device float4x4 const * const mu [[ buffer(2)  ]],
							  device float4x4 const * const sigma [[ buffer(3) ]],
							  device float4x4 const * const delta [[ buffer(4) ]],
							  uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float4x4 const s = sigma[idx];
	float4x4 const d = delta[idx];
	delta_mu[idx] = d;
	delta_sigma[idx] = 2.0 * float4x4((d[0]*d[0]-s[0]*s[0])*s[0],
									  (d[1]*d[1]-s[1]*s[1])*s[1],
									  (d[2]*d[2]-s[2]*s[2])*s[2],
									  (d[3]*d[3]-s[3]*s[3])*s[3]);
}

kernel void GaussSynthesize16(device float4x4 * const sigma [[ buffer(0) ]],
							  device float4x4 const * const variance [[ buffer(1) ]],
							  uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float4x4 const v = variance[idx];
	sigma[idx] = float4x4(sqrt(v[0]),
						  sqrt(v[1]),
						  sqrt(v[2]),
						  sqrt(v[3]));
}
kernel void GaussGradient(device float4x4 * const mu [[ buffer(0) ]],
						  device float4x4 * const sigma [[ buffer(1) ]],
						  device float4x4 const * const sum_mu [[ buffer(2) ]],
						  device float4x4 const * const sum_sigma [[ buffer(3) ]],
						  uint const n [[ thread_position_in_grid ]],
						  uint const N [[ threads_per_grid ]]) {
	
	float const scale = 0.5 * M_2_SQRTPI_F * M_SQRT1_2_F;
	
	int const k = n;
	
	float4x4 const u = mu[k];
	float4x4 const s = sum_sigma[n];
	
	float4x4 const l = float4x4(rsqrt(s[0]),
								rsqrt(s[1]),
								rsqrt(s[2]),
								rsqrt(s[3]));
	
	float4x4 const x = float4x4(l[0]*u[0],
								l[1]*u[1],
								l[2]*u[2],
								l[3]*u[3]);
	
	float4x4 const g = scale * float4x4(exp(-0.5*x[0]*x[0])*l[0],
										exp(-0.5*x[1]*x[1])*l[1],
										exp(-0.5*x[2]*x[2])*l[2],
										exp(-0.5*x[3]*x[3])*l[3]);
	
	mu[k] = float4x4(g[0],
					 g[1],
					 g[2],
					 g[3]);
	
	sigma[k] = float4x4(-g[0]*x[0],
						-g[1]*x[1],
						-g[2]*x[2],
						-g[3]*x[3]);
	
}
constant float M_1_UINT32MAX_F = 1 / 4294967296.0;
kernel void GaussRNG16(device float4x4 * const value [[ buffer(0) ]],
					   device const float4x4 * const mu [[ buffer(1) ]],
					   device const float4x4 * const sigma [[ buffer(2) ]],
					   constant const uint4 * const seeds [[ buffer(3) ]],
					   constant const uint & length [[ buffer(4) ]],
					   uint const n [[ thread_position_in_grid ]],
					   uint const N [[ threads_per_grid ]]) {
	uint4 seq = seeds[n];
	seq = select ( seq, ~0, seq == 0 );
	
	for ( int k = n, K = length ; k < K ; k += N ) {
		
		float4x4 u;
		
		seq ^= seq << xorshift.x, seq ^= seq >> xorshift.y, seq ^= seq << xorshift.z; u[0] = float4(seq) - 0.5;
		seq ^= seq << xorshift.x, seq ^= seq >> xorshift.y, seq ^= seq << xorshift.z; u[1] = float4(seq) - 0.5;
		seq ^= seq << xorshift.x, seq ^= seq >> xorshift.y, seq ^= seq << xorshift.z; u[2] = float4(seq) - 0.5;
		seq ^= seq << xorshift.x, seq ^= seq >> xorshift.y, seq ^= seq << xorshift.z; u[3] = float4(seq) - 0.5;
		
		value[k] = BoxMuller(mu[k], sigma[k], M_1_UINT32MAX_F*u);
		
	}
}
/*
kernel void GaussCDF(device float4x4 * const f [[ buffer(0) ]],
					 device const float4x4 * const mu [[ buffer(1) ]],
					 device const float4x4 * const sigma [[ buffer(2) ]],
					 constant const uint & length [[ buffer(3) ]],
					 uint const n [[ thread_position_in_grid ]]) {
	int4 const ofs = int4(0, 1, 2, 3);
	int4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	if ( can.x ) {
		float4x4 const u = M_SQRT1_2_F * mu[idx.x];
		float4x4 const s = sigma[idx.x];
		f[idx.x] = 0.5 * float4x4(1+erf(u[ofs.x]/s[ofs.x]), 1+erf(u[ofs.y]/s[ofs.y]), 1+erf(u[ofs.z]/s[ofs.z]), 1+erf(u[ofs.w]/s[ofs.w]));
	}
	if ( can.y ) {
		float4x4 const u = M_SQRT1_2_F * mu[idx.y];
		float4x4 const s = sigma[idx.y];
		f[idx.y] = 0.5 * float4x4(1+erf(u[ofs.x]/s[ofs.x]), 1+erf(u[ofs.y]/s[ofs.y]), 1+erf(u[ofs.z]/s[ofs.z]), 1+erf(u[ofs.w]/s[ofs.w]));
	}
	if ( can.z ) {
		float4x4 const u = M_SQRT1_2_F * mu[idx.z];
		float4x4 const s = sigma[idx.z];
		f[idx.z] = 0.5 * float4x4(1+erf(u[ofs.x]/s[ofs.x]), 1+erf(u[ofs.y]/s[ofs.y]), 1+erf(u[ofs.z]/s[ofs.z]), 1+erf(u[ofs.w]/s[ofs.w]));
	}
	if ( can.w ) {
		float4x4 const u = M_SQRT1_2_F * mu[idx.w];
		float4x4 const s = sigma[idx.w];
		f[idx.w] = 0.5 * float4x4(1+erf(u[ofs.x]/s[ofs.x]), 1+erf(u[ofs.y]/s[ofs.y]), 1+erf(u[ofs.z]/s[ofs.z]), 1+erf(u[ofs.w]/s[ofs.w]));
	}
}
kernel void GaussPDF(device float4x4 * const p [[ buffer(0) ]],
					 device const float4x4 * const mu [[ buffer(1) ]],
					 device const float4x4 * const sigma [[ buffer(2) ]],
					 constant const uint & length [[ buffer(3) ]],
					 uint const n [[ thread_position_in_grid ]]) {
	uint4 const ofs = uint4(0, 1, 2, 3);
	uint4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	if ( can.x ) {
		uint const k = idx.x;
		float4x4 const s = sigma[k];
		p[idx.x] = standardization * div(exp(-0.5*sq(div(mu[k], s))), s);
	}
	if ( can.y ) {
		uint const k = idx.y;
		float4x4 const s = sigma[k];
		p[idx.y] = standardization * div(exp(-0.5*sq(div(mu[k], s))), s);
	}
	if ( can.z ) {
		uint const k = idx.z;
		float4x4 const s = sigma[k];
		p[idx.z] = standardization * div(exp(-0.5*sq(div(mu[k], s))), s);
	}
	if ( can.w ) {
		uint const k = idx.w;
		float4x4 const s = sigma[k];
		p[idx.w] = standardization * div(exp(-0.5*sq(div(mu[k], s))), s);
	}
}
*/
inline float4 shuffle(float4x4 const w[4], float4x4 const x) {
	return	x[0]*float4x4(w[0][0], w[1][0], w[2][0], w[3][0])+
			x[1]*float4x4(w[0][1], w[1][1], w[2][1], w[3][1])+
			x[2]*float4x4(w[0][2], w[1][2], w[2][2], w[3][2])+
			x[3]*float4x4(w[0][3], w[1][3], w[2][3], w[3][3]);
}
inline float4 sqshuffle(float4x4 const w[4], float4x4 const x) {
	return	x[0]*x[0]*float4x4(w[0][0]*w[0][0], w[1][0]*w[1][0], w[2][0]*w[2][0], w[3][0]*w[3][0])+
			x[1]*x[1]*float4x4(w[0][1]*w[0][1], w[1][1]*w[1][1], w[2][1]*w[2][1], w[3][1]*w[3][1])+
			x[2]*x[2]*float4x4(w[0][2]*w[0][2], w[1][2]*w[1][2], w[2][2]*w[2][2], w[3][2]*w[3][2])+
			x[3]*x[3]*float4x4(w[0][3]*w[0][3], w[1][3]*w[1][3], w[2][3]*w[2][3], w[3][3]*w[3][3]);
}
kernel void GaussCollectW16(device float4x4 * const y_value [[ buffer(0) ]],
							device float4x4 * const y_mean [[ buffer(1) ]],
							device float4x4 * const y_variance [[ buffer(2) ]],
							device float4x4 const * const w_value [[ buffer(3) ]],
							device float4x4 const * const w_mu [[ buffer(4) ]],
							device float4x4 const * const w_sigma [[ buffer(5) ]],
							device float4x4 const * const x [[ buffer(6) ]],
							constant uint const & length [[ buffer(7) ]],
							threadgroup float4x4 * accumulator_value [[ threadgroup(0) ]],
							threadgroup float4x4 * accumulator_mean [[ threadgroup(1) ]],
							threadgroup float4x4 * accumulator_variance [[ threadgroup(2) ]],
							uint const t [[ thread_position_in_threadgroup ]],
							uint const T [[ threads_per_threadgroup ]],
							uint const m [[ threadgroup_position_in_grid ]]) {
	int4 const ref[4] = {
		( 16 * m + 0x0 + int4(0, 1, 2, 3)) * length,
		( 16 * m + 0x4 + int4(0, 1, 2, 3)) * length,
		( 16 * m + 0x8 + int4(0, 1, 2, 3)) * length,
		( 16 * m + 0xc + int4(0, 1, 2, 3)) * length
	};
	
	float4x4 value = float4x4(0);
	float4x4 mean = float4x4(0);
	float4x4 variance = float4x4(0);
	
	//split each element because several gpu architecture has few register
	for ( int k = t, K = length ; k < K ; k += T ) {
		float4x4 const f = x[k];
		{
			int4 const idx = k + ref[0];
			{
				float4x4 const weight_value[4] = {w_value[idx.x], w_value[idx.y], w_value[idx.z], w_value[idx.w]};
				value[0] += shuffle(weight_value, f);
			}
			{
				float4x4 const weight_mean[4] = {w_mu[idx.x], w_mu[idx.y], w_mu[idx.z], w_mu[idx.w]};
				mean[0] += shuffle(weight_mean, f);
			}
		}
		{
			int4 const idx = k + ref[1];
			{
				float4x4 const weight_value[4] = {w_value[idx.x], w_value[idx.y], w_value[idx.z], w_value[idx.w]};
				value[1] += shuffle(weight_value, f);
			}
			{
				float4x4 const weight_mean[4] = {w_mu[idx.x], w_mu[idx.y], w_mu[idx.z], w_mu[idx.w]};
				mean[1] += shuffle(weight_mean, f);
			}
		}
		{
			int4 const idx = k + ref[2];
			{
				float4x4 const weight_value[4] = {w_value[idx.x], w_value[idx.y], w_value[idx.z], w_value[idx.w]};
				value[2] += shuffle(weight_value, f);
			}
			{
				float4x4 const weight_mean[4] = {w_mu[idx.x], w_mu[idx.y], w_mu[idx.z], w_mu[idx.w]};
				mean[2] += shuffle(weight_mean, f);
			}
		}
		{
			int4 const idx = k + ref[3];
			{
				float4x4 const weight_value[4] = {w_value[idx.x], w_value[idx.y], w_value[idx.z], w_value[idx.w]};
				value[3] += shuffle(weight_value, f);
			}
			{
				float4x4 const weight_mean[4] = {w_mu[idx.x], w_mu[idx.y], w_mu[idx.z], w_mu[idx.w]};
				mean[3] += shuffle(weight_mean, f);
			}
		}
	}
	for ( int k = t, K = length ; k < K ; k += T ) {
		float4x4 const f2 = sq(x[k]);
		{
			int4 const idx = k + ref[0];
			float4x4 const weight_variance[4] = {sq(w_sigma[idx.x]), sq(w_sigma[idx.y]), sq(w_sigma[idx.z]), sq(w_sigma[idx.w])};
			variance[0] += shuffle(weight_variance, f2);
		}
		{
			int4 const idx = k + ref[1];
			float4x4 const weight_variance[4] = {sq(w_sigma[idx.x]), sq(w_sigma[idx.y]), sq(w_sigma[idx.z]), sq(w_sigma[idx.w])};
			variance[1] += shuffle(weight_variance, f2);
		}
		{
			int4 const idx = k + ref[2];
			float4x4 const weight_variance[4] = {sq(w_sigma[idx.x]), sq(w_sigma[idx.y]), sq(w_sigma[idx.z]), sq(w_sigma[idx.w])};
			variance[2] += shuffle(weight_variance, f2);
		}
		{
			int4 const idx = k + ref[3];
			float4x4 const weight_variance[4] = {sq(w_sigma[idx.x]), sq(w_sigma[idx.y]), sq(w_sigma[idx.z]), sq(w_sigma[idx.w])};
			variance[3] += shuffle(weight_variance, f2);
		}
	}
	
	int const a = t;
	int b = T;
	
	accumulator_value[a] = value;
	accumulator_mean[a] = mean;
	accumulator_variance[a] = variance;
	
	while ( b >>= 1 ) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( a < b ) {
			accumulator_value[a] += accumulator_value[a+b];
			accumulator_mean[a] += accumulator_mean[a+b];
			accumulator_variance[a] += accumulator_variance[a+b];
		}
	}
	if ( !a ) {
		int const idx = m;
		y_value[idx] += accumulator_value[a];
		y_mean[idx] += accumulator_mean[a];
		y_variance[idx] += accumulator_variance[a];
	}
	
}
kernel void GaussCollectC16(device float4x4 * const y_value [[ buffer(0) ]],
							device float4x4 * const y_mean [[ buffer(1) ]],
							device float4x4 * const y_variance [[ buffer(2) ]],
							device float4x4 const * const x_value [[ buffer(3) ]],
							device float4x4 const * const x_mu [[ buffer(4) ]],
							device float4x4 const * const x_sigma [[ buffer(5) ]],
							uint const n [[ thread_position_in_grid ]]) {
	int const idx = n;
	float4x4 const s = x_sigma[idx];
	y_variance[idx] += float4x4(s[0]*s[0],
								s[1]*s[1],
								s[2]*s[2],
								s[3]*s[3]);
	y_mean[idx] += x_mu[idx];
	y_value[idx] += x_value[idx];
}
kernel void GaussCollectD16(device float4x4 * const y_value [[ buffer(0) ]],
							device float4x4 * const y_mean [[ buffer(1) ]],
							device float4x4 * const y_variance [[ buffer(2) ]],
							device float4x4 const * const decay [[ buffer(3) ]],
							device float4x4 const * const x_value [[ buffer(4) ]],
							device float4x4 const * const x_mu [[ buffer(5) ]],
							device float4x4 const * const x_sigma [[ buffer(6) ]],
							uint const n [[ thread_position_in_grid ]]) {
	int const k = n;
	float4x4 const d = decay[k];
	float4x4 const v = x_value[k];
	float4x4 const m = x_mu[k];
	float4x4 const s = x_sigma[k];
	y_value[k] += float4x4(d[0]*v[0],
						   d[1]*v[1],
						   d[2]*v[2],
						   d[3]*v[3]);
	y_mean[k] += float4x4(d[0]*m[0],
						  d[1]*m[1],
						  d[2]*m[2],
						  d[3]*m[3]);
	y_variance[k] += float4x4(d[0]*d[0]*s[0]*s[0],
							  d[1]*d[1]*s[1]*s[1],
							  d[2]*d[2]*s[2]*s[2],
							  d[3]*d[3]*s[3]*s[3]);
}
/*
kernel void GaussGradient(device float2x4 * const g [[ buffer(0) ]],
						  device const float4 * const mu [[ buffer(1) ]],
						  device const float4 * const sigma [[ buffer(2) ]],
						  uint const n [[ thread_position_in_grid ]]) {
	float4 const u = mu[n];
	float4 const s = sigma[n];
	float4 const v = M_1_PI_F * ( u * u + s * s );
	g[n] = float2x4(
		v, v
	);
}
kernel void GaussDecay(device float4 * const y [[ buffer(0) ]],
					   device float4 * const u [[ buffer(1) ]],
					   device float4 * const s [[ buffer(2) ]],
					   constant const float4 & r [[ buffer(3) ]],
					   uint t [[ thread_position_in_grid ]],
					   uint T [[ threads_per_grid ]]) {
	y[t] *= r;
	u[t] *= r;
	s[t] *= r * r;
}
kernel void gaussianSynth(device float4 * const value [[ buffer(0) ]],
						  device float4 * const mu [[ buffer(1) ]],
						  device float4 * const lambda [[ buffer(2) ]],
						  device const float4 * const x [[ buffer(3) ]],
						  device const float4 * const u [[ buffer(4) ]],
						  device const float4 * const s [[ buffer(5) ]],
						  uint k [[ thread_position_in_grid ]],
						  uint K [[ threads_per_grid ]]) {
	value[k] = step(0.0, x[k]);
	mu[k] = u[k];
	lambda[k] = rsqrt(s[k]);
}
kernel void gaussianAdd(device float4 * const accumulator_x [[ buffer(0) ]],
						device float4 * const accumulator_u [[ buffer(1) ]],
						device float4 * const accumulator_s [[ buffer(2) ]],
						device const float4 * const x [[ buffer(3) ]],
						device const float4 * const u [[ buffer(4) ]],
						device const float4 * const s [[ buffer(5) ]],
						uint k [[ thread_position_in_grid ]],
						uint K [[ threads_per_grid ]]) {
	accumulator_x[k] += x[k];
	accumulator_u[k] += u[k];
	accumulator_s[k] += s[k] * s[k];
}
kernel void gaussianCollect(device float4 * const Yx [[ buffer(0) ]],
							device float4 * const Yu [[ buffer(1) ]],
							device float4 * const Ys [[ buffer(2) ]],
							device const float4 * const Wx [[ buffer(3) ]],
							device const float4 * const Wu [[ buffer(4) ]],
							device const float4 * const Ws [[ buffer(5) ]],
							device const float4 * const X [[ buffer(6) ]],
							threadgroup float4 * const accumulator_x [[ threadgroup(0) ]],
							threadgroup float4 * const accumulator_u [[ threadgroup(1) ]],
							threadgroup float4 * const accumulator_s [[ threadgroup(2) ]],
							uint t [[ thread_position_in_threadgroup ]],
							uint T [[ threads_per_threadgroup ]],
							uint g [[ threadgroup_position_in_grid ]],
							uint G [[ threadgroups_per_grid ]]) {
	float4 const x = X [ t ];
	uint4 const idx = ( 4 * g + uint4(0, 1, 2, 3) ) * T + t;
	
	threadgroup float4 * const Ax = accumulator_x + t;
	threadgroup float4 * const Au = accumulator_u + t;
	threadgroup float4 * const As = accumulator_s + t;
	
	*Ax = ( x ) * float4x4(Wx[idx.x],
						   Wx[idx.y],
						   Wx[idx.z],
						   Wx[idx.w]);
	*Au = ( x ) * float4x4(Wu[idx.x],
						   Wu[idx.y],
						   Wu[idx.z],
						   Wu[idx.w]);
	*As = ( x * x ) * float4x4(Ws[idx.x]*Ws[idx.x],
							   Ws[idx.y]*Ws[idx.y],
							   Ws[idx.z]*Ws[idx.z],
							   Ws[idx.w]*Ws[idx.w]);
	uint offset = T;
	while ( offset >>= 1 ) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( t < offset ) {
			*Ax += Ax [ offset ];
			*Au += Au [ offset ];
			*As += As [ offset ];
		}
	}
	if ( !t ) {
		Yx[g] += *Ax;
		Yu[g] += *Au;
		Ys[g] += *As;
	}
}
kernel void gaussCollect(device float4 * const y [[ buffer(0) ]],
						 device float4 * const u [[ buffer(1) ]],
						 device float4 * const s [[ buffer(2) ]],
						 device const float4 * const Y [[ buffer(3) ]],
						 device const float4 * const U [[ buffer(4) ]],
						 device const float4 * const S [[ buffer(5) ]],
						 device const float4 * const X [[ buffer(6) ]],
						 threadgroup float4 * const accum_y [[ threadgroup(0) ]],
						 threadgroup float4 * const accum_u [[ threadgroup(1) ]],
						 threadgroup float4 * const accum_s [[ threadgroup(2) ]],
						 uint t [[ thread_position_in_threadgroup ]],
						 uint T [[ threads_per_threadgroup ]],
						 uint g [[ threadgroup_position_in_grid ]],
						 uint G [[ threadgroups_per_grid ]]) {
	float4 const x = X [ t ];
	uint4 const idx = ( 4 * g + uint4(0, 1, 2, 3) ) * T + t;
	accum_y [ t ] = x * float4x4(Y[idx.x],
								 Y[idx.y],
								 Y[idx.z],
								 Y[idx.w]);
	accum_u [ t ] = x * float4x4(U[idx.x],
								 U[idx.y],
								 U[idx.z],
								 U[idx.w]);
	accum_s [ t ] = ( x * x ) * float4x4(S[idx.x]*S[idx.x],
										 S[idx.y]*S[idx.y],
										 S[idx.z]*S[idx.z],
										 S[idx.w]*S[idx.w]);
	uint offset = T;
	while ( offset >>= 1 ) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( t < offset ) {
			accum_y [ t ] += accum_y [ t + offset ];
			accum_u [ t ] += accum_u [ t + offset ];
			accum_s [ t ] += accum_s [ t + offset ];
		}
	}
	if ( !t ) {
		s[g] = sqrt(s[g]*s[g]+accum_s[0]);
		u[g] += accum_u[0];
		y[g] += accum_y[0];
	}
}
 */
//an approximation
inline float4x4 BoxMuller(float4x4 const m, float4x4 const s, float4x4 const u) {
	float4x4 const n = transpose(float4x4(cospi(2.0*u[0]), sinpi(2.0*u[1]), sqrt(-2.0*log(u[2])), sqrt(-2.0*log(u[3]))));
	return m + float4x4(s[0]*n[0].xyxy*n[0].zzww,
						s[1]*n[1].xyxy*n[1].zzww,
						s[2]*n[2].xyxy*n[2].zzww,
						s[3]*n[3].xyxy*n[3].zzww);
}
inline float4x4 exp(float4x4 x) {
	return float4x4(exp(x[0]), exp(x[1]), exp(x[2]), exp(x[3]));
}
inline float4x4 step(float edge, float4x4 x) {
	return float4x4(step(edge, x[0]), step(edge, x[1]), step(edge, x[2]), step(edge, x[3]));
}
inline float4x4 sqrt(float4x4 x) {
	return float4x4(sqrt(x[0]), sqrt(x[1]), sqrt(x[2]), sqrt(x[3]));
}
inline float4x4 sq(float4x4 const x) {
	return float4x4(x[0]*x[0], x[1]*x[1], x[2]*x[2], x[3]*x[3]);
}
inline float4x4 mul(float4x4 x, float4x4 y) {
	return float4x4(x[0]*y[0], x[1]*y[1], x[2]*y[2], x[3]*y[3]);
}
inline float4x4 div(float4x4 x, float4x4 y) {
	return float4x4(x[0]/y[0], x[1]/y[1], x[2]/y[2], x[3]/y[3]);
}
inline float4x4 rsqrt(float4x4 const x) {
	return float4x4(rsqrt(x[0]), rsqrt(x[1]), rsqrt(x[2]), rsqrt(x[3]));
}
