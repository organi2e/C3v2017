//
//  Ridge.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

import Metal
public class Ridge {
	let gradient: MTLComputePipelineState
	let groups: MTLSize
	let threads: MTLSize
	let limit: Int
	private init(pipeline: MTLComputePipelineState, count: Int) {
		let width: Int = pipeline.threadExecutionWidth
		gradient = pipeline
		limit = count
		threads = MTLSize(width: width, height: 1, depth: 1)
		groups = MTLSize(width: (count-1)/width+1, height: 1, depth: 1)
	}
	public static func factory(λ: Float) -> (MTLDevice) throws -> (Int) -> Adapter {
		let bundle: Bundle = Bundle(for: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([λ], type: .float, withName: "lambda")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let pipeline: MTLComputePipelineState = try library.make(name: "RidgeGradient", constantValues: constantValues)
			return {
				Ridge(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension Ridge: Adapter {
	public func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
		assert( commandBuffer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: φ, sourceOffset: 0, to: θ, destinationOffset: 0, size: limit * MemoryLayout<Float>.size)
		encoder.endEncoding()
		
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
		assert( commandBuffer.device === gradient.device )
		assert( commandBuffer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(gradient)
		encoder.setBuffer(Δ, offset: 0, at: 0)
		encoder.setBuffer(θ, offset: 0, at: 1)
		encoder.setBuffer(φ, offset: 0, at: 2)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
		encoder.endEncoding()
		
	}
}
