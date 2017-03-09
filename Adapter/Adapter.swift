//
//  Adapter.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

import Metal
public protocol Adapter {
	func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer)
	func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer)
}
public class Linear {
	
}
/*
extension Linear: Adapter {
	public func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer, count: Int) {
		
		assert( commandBuffer.device === θ.device && count * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && count * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: φ, sourceOffset: 0, to: θ, destinationOffset: 0, size: count*MemoryLayout<Float>.size)
		encoder.endEncoding()
		
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer, count: Int) {
		
	}
}*/
internal extension MTLLibrary {
	internal func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}
