//
//  Distribution.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal
import LaObjet

public protocol Stochastic {
	func shuffle(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, from: (μ: MTLBuffer, σ: MTLBuffer), count: Int)
}
public protocol Synthesis {
	func μ(_: LaObjet) -> LaObjet
	func σ(_: LaObjet) -> LaObjet
	func synthesize(commandBuffer: MTLCommandBuffer, ϝ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), count: Int)
	/*
	func collect(r: MTLBuffer, x: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer))
	func collect(w: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer)
	func collect(x: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer))
	func clear()
	*/
}
public protocol Distributor: Stochastic, Synthesis {
	
}
extension MTLLibrary {
	internal func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}

