//
//  MomentumAdaDelta.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

import Metal

public class MomentumAdaDelta {
	let parameters: MTLBuffer
	let pipeline: MTLComputePipelineState
	let groups: MTLSize
	let threads: MTLSize
	private init(pipeline state: MTLComputePipelineState, count: Int) {
		groups = MTLSize(width: (count+15)/16, height: 1, depth: 1)
		threads = MTLSize(width: 1, height: 1, depth: 1)
		pipeline = state
		parameters = state.device.makeBuffer(length: 64*groups.width, options: .storageModePrivate)
	}
	public static func factory(ρ: Float = 1023/1024.0, ε: Float = 1) -> (MTLDevice) throws -> (Int) -> Optimizer {
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([ρ], type: .float, withName: "rho")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: Bundle(for: self))
			let function: MTLFunction = try library.makeFunction(name: "MomentumAdaDeltaOptimize", constantValues: constantValues)
			let pipeline: MTLComputePipelineState = try $0.makeComputePipelineState(function: function)
			return {
				MomentumAdaDelta(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension MomentumAdaDelta: Optimizer {
	public func encode(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δθ: MTLBuffer) {

		let length: Int = 16 * groups.width * MemoryLayout<Float>.size
		
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
typealias MAD = MomentumAdaDelta
