//
//  SMORMS3.swift
//  macOS
//
//  Created by Kota Nakano on 2/28/17.
//
//

import Metal

public class SMORMS3 {
	let parameters: MTLBuffer
	let optimizer: MTLComputePipelineState
	let groups: MTLSize
	let threads: MTLSize
	private init(pipeline: MTLComputePipelineState, count: Int) {
		groups = MTLSize(width: count, height: 1, depth: 1)
		threads = MTLSize(width: 1, height: 1, depth: 1)
		optimizer = pipeline
		parameters = pipeline.device.makeBuffer(length: 4*count*MemoryLayout<Float>.size, options: .storageModePrivate)
	}
	public static func factory(α: Float = 1e-3, ε: Float = 1e-12) -> (MTLDevice) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([α], type: .float, withName: "alpha")
		constantValues.setConstantValue([ε], type: .float, withName: "epsilon")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let function: MTLFunction = try library.makeFunction(name: "SMORMS3Optimize", constantValues: constantValues)
			let pipeline: MTLComputePipelineState = try $0.makeComputePipelineState(function: function)
			return {
				SMORMS3(pipeline: pipeline, count: $0)
			}
		}
	}
}
extension SMORMS3: Optimizer {
	public func optimize(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δ: MTLBuffer) {
		
		assert(groups.width * MemoryLayout<Float>.size<=θ.length)
		assert(groups.width * MemoryLayout<Float>.size<=Δ.length)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
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
