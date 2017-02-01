//
//  Distribution.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal

public protocol Stochastic {
	func encode(commandBuffer: MTLCommandBuffer, CDF: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int)
	func encode(commandBuffer: MTLCommandBuffer, PDF: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int)
	func encode(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int)
}
public protocol Synthesis {
	//Probably
	//Fire exp
	func collect(commandBuffer: MTLCommandBuffer,
	             Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             w: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             x: MTLBuffer,
	             count: (rows: Int, cols: Int)
	)
	func collect(commendBuffer: MTLCommandBuffer,
	             Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             w: MTLBuffer,
	             x: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             count: (rows: Int, cols: Int)
	)
	func collect(commandBuffer: MTLCommandBuffer,
	             Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             w: MTLBuffer,
	             x: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             count: Int
	)
	func collect(commandBuffer: MTLCommandBuffer,
	             Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             b: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             count: Int
	)
}
public protocol Distributor: Stochastic, Synthesis {

}
extension MTLLibrary {
	internal func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}
