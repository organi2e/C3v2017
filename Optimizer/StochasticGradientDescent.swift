//
//  StochasticGradientDescent.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/25.
//
//

import Accelerate
import Metal

public class StochasticGradientDescent {
	let optimizer: (MTLCommandBuffer, MTLBuffer, MTLBuffer) -> Void
	public init(η: Float) {
		optimizer = { (commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δθ: MTLBuffer) in
			commandBuffer.addCompletedHandler { (_: MTLCommandBuffer) in
				cblas_saxpy(Int32(min(θ.length, Δθ.length)/MemoryLayout<Float>.size), η, UnsafePointer<Float>(OpaquePointer(Δθ.contents())), 1, UnsafeMutablePointer<Float>(OpaquePointer(θ.contents())), 1)
			}
		}
	}
	public init(device: MTLDevice, η: Float) throws {
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: Bundle(for: type(of: self)))
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([η], type: .float, withName: "eta")
		let function: MTLFunction = try library.makeFunction(name: "StochasticGradientDescentOptimize", constantValues: constantValues)
		let pipeline: MTLComputePipelineState = try device.makeComputePipelineState(function: function)
		optimizer = { (commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δθ: MTLBuffer) in
			let length: Int = min(θ.length, Δθ.length) / MemoryLayout<Float>.size
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(pipeline)
			encoder.setBuffer(θ, offset: 0, at: 0)
			encoder.setBuffer(Δθ, offset: 0, at: 1)
			encoder.dispatchThreadgroups(MTLSize(width: (length-1)/4+1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public static func factory(η: Float) -> (MTLDevice, Int) throws -> Optimizer {
		return {
			try StochasticGradientDescent(device: $0.0, η: η)
		}
	}
}
extension StochasticGradientDescent: Optimizer {
	public func encode(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δθ: MTLBuffer) {
		optimizer(commandBuffer, θ, Δθ)
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		
	}
}
public typealias SGD = StochasticGradientDescent
