//
//  Momentum.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

import Metal

public class Momentum {
	let pipeline: MTLComputePipelineState
	let parameters: MTLBuffer
	let threads: MTLSize
	let groups: MTLSize
	private init(pipeline state: MTLComputePipelineState, count: Int) {
		groups = MTLSize(width: (count+15)/16, height: 1, depth: 1)
		threads = MTLSize(width: 1, height: 1, depth: 1)
		pipeline = state
		parameters = state.device.makeBuffer(length: 16*groups.width*MemoryLayout<Float>.size, options: .storageModePrivate)
	}
	public static func factory(η: Float = 1e-2, γ: Float = 0.9) -> (MTLDevice) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([η], type: .float, withName: "eta")
		constantValues.setConstantValue([γ], type: .float, withName: "gamma")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
			let function: MTLFunction = try library.makeFunction(name: "MomentumOptimize", constantValues: constantValues)
			let pipeline: MTLComputePipelineState = try $0.makeComputePipelineState(function: function)
			return {
				Momentum(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension Momentum: Optimizer {
	public func encode(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δθ: MTLBuffer) {
		
		let length: Int = parameters.length
		
		assert(length<=θ.length)
		assert(length<=Δθ.length)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(pipeline)
		encoder.setBuffer(θ, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(Δθ, offset: 0, at: 2)
		encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
		encoder.endEncoding()
		
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: parameters, range: NSRange(location: 0, length: parameters.length), value: 0)
		encoder.endEncoding()
	}
}
