//
//  Distribution.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal

public protocol Stochastic {
	
	func shuffle(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int)
	
	func errorState(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, ψ: MTLBuffer)
	func errorValue(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, ψ: MTLBuffer)
	func deltaState(commandBuffer: MTLCommandBuffer, Δμ: MTLBuffer, Δσ: MTLBuffer, Δ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer)
	func deltaValue(commandBuffer: MTLCommandBuffer, Δμ: MTLBuffer, Δσ: MTLBuffer, Δ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer					)
	
}
public protocol Synthesis {
	func synthesize(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer)
	func collect(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer)
	func collect(commandBuffer: MTLCommandBuffer, r: MTLBuffer, x: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer))
	func collect(commandBuffer: MTLCommandBuffer, w: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, refer: Int)
	func reset(commandBuffer: MTLCommandBuffer)
}
public protocol Distributor: Stochastic, Synthesis {
	
}
extension MTLLibrary {
	internal func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}

