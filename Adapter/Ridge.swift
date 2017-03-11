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
	let adapt: MTLComputePipelineState
	let limit: Int
	private init(pipeline: (MTLComputePipelineState, MTLComputePipelineState), count: Int) {
		limit = count
		gradient = pipeline.0
		adapt = pipeline.1
	}
	public static func factory(λ: Float) -> (MTLDevice) throws -> (Int) -> Adapter {
		let bundle: Bundle = Bundle(for: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([λ], type: .float, withName: "lambda")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let pipeline: MTLComputePipelineState = try library.make(name: "RidgeGradient", constantValues: constantValues)
			let adapt: MTLComputePipelineState = try library.make(name: "RidgeAdapt", constantValues: constantValues)
			return {
				Ridge(pipeline: (pipeline, adapt), count: $0)
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
		assert( commandBuffer.device === Δ.device && limit * MemoryLayout<Float>.size <= Δ.length )
		assert( commandBuffer.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = gradient.threadExecutionWidth
		encoder.setComputePipelineState(gradient)
		encoder.setBuffer(Δ, offset: 0, at: 0)
		encoder.setBuffer(θ, offset: 0, at: 1)
		encoder.setBuffer(φ, offset: 0, at: 2)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(.init(width: (limit-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	
	public func adapt(commandBuffer: MTLCommandBuffer, φ: MTLBuffer, θ: MTLBuffer, Δ: MTLBuffer) {
		
		assert( adapt.device === commandBuffer.device )
		assert( adapt.device === φ.device && limit * MemoryLayout<Float>.size <= φ.length )
		assert( adapt.device === θ.device && limit * MemoryLayout<Float>.size <= θ.length )
		assert( adapt.device === Δ.device && limit * MemoryLayout<Float>.size <= Δ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = adapt.threadExecutionWidth
		encoder.setComputePipelineState(adapt)
		encoder.setBuffer(φ, offset: 0, at: 0)
		encoder.setBuffer(θ, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.setBytes([uint(limit)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(.init(width: (limit-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
}
