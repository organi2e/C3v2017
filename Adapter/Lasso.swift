//
//  Lasso.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//

import Metal
public class Lasso {
	let gradient: MTLComputePipelineState
	let limit: Int
	let groups: MTLSize
	let threads: MTLSize
	private init(pipeline: MTLComputePipelineState, count: Int) {
		let width: Int = pipeline.threadExecutionWidth
		gradient = pipeline
		limit = N
		groups = .init(width: (count-1)/width+1, height: 1, depth: 1)
		threads = .init(width: width, height: 1, depth: 1)
	}
	public static func factory(λ: Float) -> (MTLDevice) throws -> (Int) -> Adapter {
		let bundle: Bundle = Bundle(for: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([λ], type: .float, withName: "lambda")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let pipeline: MTLComputePipelineState = try library.make(name: "LassoGradient", constantValues: constantValues)
			return {
				Lasso(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension Lasso: Adapter {
	public func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
		assert( commandBuffer.device === θ.device && count * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && count * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: φ, sourceOffset: 0, to: θ, destinationOffset: 0, size: count*MemoryLayout<Float>.size)
		encoder.endEncoding()
		
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
		assert( commandBuffer.device === gradient.device )
		assert( commandBuffer.device === θ.device && count * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && count * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(gradient)
		encoder.setBuffer(Δ, offset: 0, at: 0)
		encoder.setBuffer(θ, offset: 0, at: 1)
		encoder.setBuffer(φ, offset: 0, at: 2)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
		encoder.endEncoding()
		
	}
}
