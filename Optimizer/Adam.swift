//
//  Adam.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/02/01.
//
//

import Metal

public class Adam {
	let optimizer: MTLComputePipelineState
	let parameters: MTLBuffer
	let groups: MTLSize
	let threads: MTLSize
	private init(pipeline: MTLComputePipelineState, count: Int) {
		groups = MTLSize(width: count, height: 1, depth: 1)
		threads = MTLSize(width: 1, height: 1, depth: 1)
		optimizer = pipeline
		parameters = pipeline.device.makeBuffer(length: 2*groups.width*MemoryLayout<Float>.size, options: .storageModePrivate)
	}
	public static func factory(α: Float = 1e-3, β: Float = 0.9, γ: Float = 0.999, ε: Float = 1e-8) -> (MTLDevice) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([α], type: .float, withName: "alpha")
		constantValues.setConstantValue([β], type: .float, withName: "beta")
		constantValues.setConstantValue([γ], type: .float, withName: "gamma")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let function: MTLFunction = try library.makeFunction(name: "AdamOptimize", constantValues: constantValues)
			let pipeline: MTLComputePipelineState = try $0.makeComputePipelineState(function: function)
			return {
				Adam(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension Adam: Optimizer {
	public func optimize(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer) {
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		
		assert( optimizer.device === encoder.device )
		
		assert( optimizer.device === θ.device && groups.width * MemoryLayout<Float>.size <= θ.length )
		assert( optimizer.device === Δ.device && groups.width * MemoryLayout<Float>.size <= Δ.length )
		
		encoder.setComputePipelineState(optimizer)
		encoder.setBuffer(θ, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
		encoder.endEncoding()
		
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: parameters, range: NSRange(location: 0, length: parameters.length), value: 0)
		encoder.endEncoding()
	}
}

