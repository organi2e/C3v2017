//
//  Linear.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/03/10.
//
//

import Metal
public class Linear {
	let gradient: MTLComputePipelineState
	let limit: Int
	let groups: MTLSize
	let threads: MTLSize
	public var L1: Float = 0
	public var L2: Float = 1e-4
	private init(pipeline: MTLComputePipelineState, count: Int) {
		let width: Int = pipeline.threadExecutionWidth
		gradient = pipeline
		limit = count
		groups = MTLSize(width: (count-1)/width+1, height: 1, depth: 1)
		threads = MTLSize(width: width, height: 1, depth: 1)
	}
	public static func factory() -> (MTLDevice) throws -> (Int) -> Adapter {
		let bundle: Bundle = Bundle(for: self)
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let pipeline: MTLComputePipelineState = try library.make(name: "LinearGradient")
			return {
				Linear(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension Linear: Adapter {
	public func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
		assert( commandBuffer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: φ, sourceOffset: 0, to: θ, destinationOffset: 0, size: min(θ.length, φ.length))
		encoder.endEncoding()
		
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
		assert( commandBuffer.device === gradient.device )
		assert( commandBuffer.device === Δ.device && limit * MemoryLayout<Float>.size <= Δ.length )
		assert( commandBuffer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = gradient.threadExecutionWidth
		encoder.setComputePipelineState(gradient)
		encoder.setBuffer(Δ, offset: 0, at: 0)
		encoder.setBuffer(θ, offset: 0, at: 1)
		encoder.setBuffer(φ, offset: 0, at: 2)
		encoder.setBytes(&L1, length: MemoryLayout<Float>.size, at: 3)
		encoder.setBytes(&L2, length: MemoryLayout<Float>.size, at: 4)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(.init(width: (limit-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
}
