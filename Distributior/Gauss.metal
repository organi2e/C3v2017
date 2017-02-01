//
//  Gauss.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

#include <metal_stdlib>
using namespace metal;

inline float4 erf(float4);

inline float4x4 BoxMuller(float4x4,float4x4,float4x4);
inline float4x4 exp(float4x4);
inline float4x4 step(float, float4x4);
inline float4x4 sqrt(float4x4);
inline float4x4 sq(float4x4 const);
inline float4x4 mul(float4x4, float4x4);
inline float4x4 div(float4x4, float4x4);

constant uint3 const xorshift [[ function_constant(0) ]];
constant float const scale = 1.0 / 4294967296.0;
constant float const standardization = 0.5 * M_2_SQRTPI_F * M_SQRT1_2_F;

kernel void GaussRNG(device float4x4 * const value [[ buffer(0) ]],
					 device const float4x4 * const mu [[ buffer(1) ]],
					 device const float4x4 * const sigma [[ buffer(2) ]],
					 constant const uint4 * const seeds [[ buffer(3) ]],
					 constant const uint & length [[ buffer(4) ]],
					 uint const n [[ thread_position_in_grid ]],
					 uint const N [[ threads_per_grid ]]) {
	
	uint4 seq = seeds[n];
	seq = select ( seq, ~0, seq == 0 );
	
	for ( uint k = n ; k < length ; k += N ) {
		
		float4x4 u;
		
		seq ^= seq << xorshift.x, seq ^= seq >> xorshift.y, seq ^= seq << xorshift.z; u[0] = float4(seq) - 0.5;
		seq ^= seq << xorshift.x, seq ^= seq >> xorshift.y, seq ^= seq << xorshift.z; u[1] = float4(seq) - 0.5;
		seq ^= seq << xorshift.x, seq ^= seq >> xorshift.y, seq ^= seq << xorshift.z; u[2] = float4(seq) - 0.5;
		seq ^= seq << xorshift.x, seq ^= seq >> xorshift.y, seq ^= seq << xorshift.z; u[3] = float4(seq) - 0.5;
		
		value[k] = BoxMuller(mu[k], sigma[k], scale*u);
		
	}
}
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
kernel void GaussSynth(device float4x4 * const fire [[ buffer(0) ]],
					   device float4x4 * const mu [[ buffer(1) ]],
					   device float4x4 * const sigma [[ buffer(2) ]],
					   device float4x4 const * const value [[ buffer(3) ]],
					   constant uint const & length [[ buffer(4) ]],
					   uint const n [[ thread_position_in_grid ]],
					   uint const N [[ threads_per_grid]]) {
	int4 const ofs = int4(0, 1, 2, 3);
	int4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	if ( can.x ) {
		int const k = idx.x;
		fire[k] = step(0, value[k]);
		sigma[k] = sqrt(sigma[k]);
	}
	if ( can.y ) {
		int const k = idx.y;
		fire[k] = step(0, value[k]);
		sigma[k] = sqrt(sigma[k]);
	}
	if ( can.z ) {
		int const k = idx.z;
		fire[k] = step(0, value[k]);
		sigma[k] = sqrt(sigma[k]);
	}
	if ( can.w ) {
		int const k = idx.w;
		fire[k] = step(0, value[k]);
		sigma[k] = sqrt(sigma[k]);
	}
}
float4 shuffle(float4x4 const w[4], float4x4 const x);
float4 shuffle(float4x4 const w[4], float4x4 const x) {
	return float4(x[0]*float4x4(w[0][0], w[1][0], w[2][0], w[3][0]) +
				  x[1]*float4x4(w[0][1], w[1][1], w[2][1], w[3][1]) +
				  x[2]*float4x4(w[0][2], w[1][2], w[2][2], w[3][2]) +
				  x[3]*float4x4(w[0][3], w[1][3], w[2][3], w[3][3]));
}
kernel void GaussCollectChain(device float4x4 * const y_value [[ buffer(0) ]],
							  device float4x4 * const y_mu [[ buffer(1) ]],
							  device float4x4 * const y_sigma [[ buffer(2) ]],
							  device float4x4 const * const w_value [[ buffer(3) ]],
							  device float4x4 const * const w_mu [[ buffer(4) ]],
							  device float4x4 const * const w_sigma [[ buffer(5) ]],
							  device float4x4 const * const x [[ buffer(6) ]],
							  constant uint const & length [[ buffer(7) ]],
							  threadgroup float4x4 * accumulator_value [[ threadgroup(0) ]],
							  threadgroup float4x4 * accumulator_mu [[ threadgroup(1) ]],
							  threadgroup float4x4 * accumulator_sigma [[ threadgroup(2) ]],
							  uint const t [[ thread_position_in_threadgroup ]],
							  uint const T [[ threads_per_threadgroup ]],
							  uint const m [[ threadgroup_position_in_grid ]]) {
	int4 const ref[4] = {
		(16 * m + int4(0x0, 0x1, 0x2, 0x3))*length,
		(16 * m + int4(0x4, 0x5, 0x6, 0x7))*length,
		(16 * m + int4(0x8, 0x9, 0xa, 0xb))*length,
		(16 * m + int4(0xc, 0xd, 0xe, 0xf))*length
	};
	
	float4x4 value = float4x4(0);
	float4x4 mu = float4x4(0);
	float4x4 sigma = float4x4(0);
	
	//split each element because several gpu architecture has few register memory
	for ( int k = t, K = length ; k < K ; k += T ) {
		float4x4 const f = x[k];
		{
			int4 const idx = k + ref[0];
			float4x4 const weight_value[4] = {w_value[idx.x], w_value[idx.y], w_value[idx.z], w_value[idx.w]};
			value[0] += shuffle(weight_value, f);
		}
		{
			int4 const idx = k + ref[1];
			float4x4 const weight_value[4] = {w_value[idx.x], w_value[idx.y], w_value[idx.z], w_value[idx.w]};
			value[1] += shuffle(weight_value, f);
		}
		{
			int4 const idx = k + ref[2];
			float4x4 const weight_value[4] = {w_value[idx.x], w_value[idx.y], w_value[idx.z], w_value[idx.w]};
			value[2] += shuffle(weight_value, f);
		}
		{
			int4 const idx = k + ref[3];
			float4x4 const weight_value[4] = {w_value[idx.x], w_value[idx.y], w_value[idx.z], w_value[idx.w]};
			value[3] += shuffle(weight_value, f);
		}
	}
	for ( int k = t, K = length ; k < K ; k += T ) {
		float4x4 const f = x[k];
		{
			int4 const idx = k + ref[0];
			float4x4 const weight_mu[4] = {w_mu[idx.x], w_mu[idx.y], w_mu[idx.z], w_mu[idx.w]};
			mu[0] += shuffle(weight_mu, f);
		}
		{
			int4 const idx = k + ref[1];
			float4x4 const weight_mu[4] = {w_mu[idx.x], w_mu[idx.y], w_mu[idx.z], w_mu[idx.w]};
			mu[1] += shuffle(weight_mu, f);
		}
		{
			int4 const idx = k + ref[2];
			float4x4 const weight_mu[4] = {w_mu[idx.x], w_mu[idx.y], w_mu[idx.z], w_mu[idx.w]};
			mu[2] += shuffle(weight_mu, f);
		}
		{
			int4 const idx = k + ref[3];
			float4x4 const weight_mu[4] = {w_mu[idx.x], w_mu[idx.y], w_mu[idx.z], w_mu[idx.w]};
			mu[3] += shuffle(weight_mu, f);
		}
	}
	for ( int k = t, K = length ; k < K ; k += T ) {
		float4x4 const f2 = sq(x[k]);
		{
			int4 const idx = k + ref[0];
			float4x4 const weight_sigma[4] = {sq(w_sigma[idx.x]), sq(w_sigma[idx.y]), sq(w_sigma[idx.z]), sq(w_sigma[idx.w])};
			sigma[0] += shuffle(weight_sigma, f2);
		}
		{
			int4 const idx = k + ref[1];
			float4x4 const weight_sigma[4] = {sq(w_sigma[idx.x]), sq(w_sigma[idx.y]), sq(w_sigma[idx.z]), sq(w_sigma[idx.w])};
			sigma[1] += shuffle(weight_sigma, f2);
		}
		{
			int4 const idx = k + ref[2];
			float4x4 const weight_sigma[4] = {sq(w_sigma[idx.x]), sq(w_sigma[idx.y]), sq(w_sigma[idx.z]), sq(w_sigma[idx.w])};
			sigma[2] += shuffle(weight_sigma, f2);
		}
		{
			int4 const idx = k + ref[3];
			float4x4 const weight_sigma[4] = {sq(w_sigma[idx.x]), sq(w_sigma[idx.y]), sq(w_sigma[idx.z]), sq(w_sigma[idx.w])};
			sigma[3] += shuffle(weight_sigma, f2);
		}
	}
	
	int const a = t;
	int b = T;
	
	accumulator_value[a] = value;
	accumulator_mu[a] = mu;
	accumulator_sigma[a] = sigma;
	
	while ( b >>= 1 ) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( a < b ) {
			accumulator_value[a] += accumulator_value[a+b];
			accumulator_mu[a] += accumulator_mu[a+b];
			accumulator_sigma[a] += accumulator_sigma[a+b];
		}
	}
	if ( !a ) {
		int const idx = m;
		y_value[idx] += accumulator_value[a];
		y_mu[idx] += accumulator_mu[a];
		y_sigma[idx] += accumulator_sigma[a];
	}
}
kernel void GaussCollectGain(device float4x4 * const y_value [[ buffer(0) ]],
							 device float4x4 * const y_mu [[ buffer(1) ]],
							 device float4x4 * const y_sigma [[ buffer(2) ]],
							 device float4x4 const * const weight [[ buffer(3) ]],
							 device float4x4 const * const x_value [[ buffer(4) ]],
							 device float4x4 const * const x_mu [[ buffer(5) ]],
							 device float4x4 const * const x_sigma [[ buffer(6) ]],
							 constant uint const & length [[ buffer(7) ]],
							 uint const n [[ thread_position_in_grid ]]) {
	uint4 const ofs = uint4(0, 1, 2, 3);
	uint4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	if ( can.x ) {
		int const k = idx.x;
		float4x4 w = weight[k];
		y_value[k] += mul(w, x_value[k]);
		y_mu[k] += mul(w, x_mu[k]);
		y_sigma[k] += sq(mul(w, x_sigma[k]));
	}
	if ( can.y ) {
		int const k = idx.y;
		float4x4 const w = weight[k];
		y_value[k] += mul(w, x_value[k]);
		y_mu[k] += mul(w, x_mu[k]);
		y_sigma[k] += sq(mul(w, x_sigma[k]));
	}
	if ( can.z ) {
		int const k = idx.z;
		float4x4 const w = weight[k];
		y_value[k] += mul(w, x_value[k]);
		y_mu[k] += mul(w, x_mu[k]);
		y_sigma[k] += sq(mul(w, x_sigma[k]));
	}
	if ( can.w ) {
		int const k = idx.w;
		float4x4 const w = weight[k];
		y_value[k] += mul(w, x_value[k]);
		y_mu[k] += mul(w, x_mu[k]);
		y_sigma[k] += sq(mul(w, x_sigma[k]));
	}
}
kernel void GaussCollectBias(device float4x4 * const y_value [[ buffer(0) ]],
							 device float4x4 * const y_mu [[ buffer(1) ]],
							 device float4x4 * const y_sigma [[ buffer(2) ]],
							 device float4x4 const * const x_value [[ buffer(3) ]],
							 device float4x4 const * const x_mu [[ buffer(4) ]],
							 device float4x4 const * const x_sigma [[ buffer(5) ]],
							 constant uint const & length [[ buffer(6) ]],
							 uint const n [[ thread_position_in_grid ]]) {
	uint4 const ofs = uint4(0, 1, 2, 3);
	uint4 const idx = 4 * n + ofs;
	bool4 const can = idx < length;
	if ( can.x ) {
		int const k = idx.x;
		y_value[k] += x_value[k];
		y_mu[k] += x_mu[k];
		y_sigma[k] += sq(x_sigma[k]);
	}
	if ( can.y ) {
		int const k = idx.y;
		float4x4 const xs = x_sigma[k];
		y_value[k] += x_value[k];
		y_mu[k] += x_mu[k];
		y_sigma[k] += sq(x_sigma[k]);
	}
	if ( can.z ) {
		int const k = idx.z;
		y_value[k] += x_value[k];
		y_mu[k] += x_mu[k];
		y_sigma[k] += sq(x_sigma[k]);
	}
	if ( can.w ) {
		int const k = idx.w;
		y_value[k] += x_value[k];
		y_mu[k] += x_mu[k];
		y_sigma[k] += sq(x_sigma[k]);
	}
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
inline float4 erf(float4 z) {
	float4 const v = 1.0 / (1.0 + 0.5 * abs(z));
	float4 const e = 1.0 - v * exp ( - z * z -  1.26551223 +
									v * ( 1.00002368 +
										 v * ( 0.37409196 +
											  v * ( 0.09678418 +
												   v * ( -0.18628806 +
														v * ( 0.27886807 +
															 v * ( -1.13520398 +
																  v * ( 1.48851587 +
																	   v * ( -0.82215223 +
																			v * ( 0.17087277 )
																			)
																	   )
																  )
															 )
														)
												   )
											  )
										 )
									);
	return select( -e, e, 0 < z);
}
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