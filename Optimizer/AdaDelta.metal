//
//  MomentumAdaDelta.metal
//  tvOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

#include <metal_stdlib>
using namespace metal;

constant float rho [[ function_constant(0) ]];
constant float epsilon [[ function_constant(1) ]];

kernel void AdaDeltaOptimize(device float4 * const theta [[ buffer(0) ]],
							 device float2x4 * const parameters [[ buffer(1) ]],
							 device const float4 * const delta [[ buffer(2) ]],
							 uint const i [[ thread_position_in_grid ]]) {
	//load
	float2x4 parameter = parameters[i];
	float4 const g = delta[i];
	
	//update
	parameter[0] = mix(g*g, parameter[0], rho);
	float4 const v = sqrt((parameter[1]+epsilon)/(parameter[1]+epsilon))*g;
	parameter[1] = mix(v*v, parameter[1], rho);
	
	//store
	theta[i] += v;
	parameters[i] = parameter;
}
