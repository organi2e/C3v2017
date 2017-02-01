//
//  MomentumAdaDelta.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/26.
//
//

import Metal

public class MomentumAdaDelta {
	let pipeline: MTLComputePipelineState
	let parameters: MTLBuffer
	public init(device: MTLDevice, count: Int, ρ: Float = 0.95, ε: Float = 1e-8) throws {
		assert(0<count)
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: Bundle(for: type(of: self)))
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([ρ], type: .float, withName: "rho")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		let function: MTLFunction = try library.makeFunction(name: "MomentumAdaDeltaOptimize", constantValues: constantValues)
		pipeline = try device.makeComputePipelineState(function: function)
		parameters = device.makeBuffer(length: 16*((count+3)/4)*MemoryLayout<Float>.size, options: .storageModePrivate)
	}
	public static func factory(ρ: Float = 0.95, ε: Float = 1e-8) -> (MTLDevice, Int) throws -> MomentumAdaDelta {
		return {
			try MomentumAdaDelta(device: $0.0, count: $0.1, ρ: ρ, ε: ε)
		}
	}
}
extension MomentumAdaDelta: Optimizer {
	public func encode(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δθ: MTLBuffer) {
		
		let length: Int = parameters.length / 4
		let count: Int = length / MemoryLayout<Float>.size
		
		assert(length<=θ.length)
		assert(length<=Δθ.length)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(pipeline)
		encoder.setBuffer(θ, offset: 0, at: 0)
		encoder.setBuffer(parameters, offset: 0, at: 1)
		encoder.setBuffer(Δθ, offset: 0, at: 2)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/4+1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: parameters, range: NSRange(location: 0, length: parameters.length), value: 0)
		encoder.endEncoding()
	}
}
typealias MAD = MomentumAdaDelta
