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
	private init(pipeline: MTLComputePipelineState, η: Float, count: Int) {
		let groups: MTLSize = MTLSize(width: (count+15)/16, height: 1, depth: 1)
		let threads: MTLSize = MTLSize(width: 1, height: 1, depth: 1)
		optimizer = { (commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δθ: MTLBuffer) in
			assert(groups.width*16*MemoryLayout<Float>.size<=θ.length)
			assert(groups.width*16*MemoryLayout<Float>.size<=Δθ.length)
			assert(groups.width*16*MemoryLayout<Float>.size<=θ.length)
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(pipeline)
			encoder.setBuffer(θ, offset: 0, at: 0)
			encoder.setBuffer(Δθ, offset: 0, at: 1)
			encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
			encoder.endEncoding()
		}
	}
	public static func factory(η: Float = 1e-3) -> (MTLDevice) throws -> (Int) -> Optimizer {
		let bundle: Bundle = Bundle(for: self)
		let constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()
		constantValues.setConstantValue([η], type: .float, withName: "eta")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let function: MTLFunction = try library.makeFunction(name: "StochasticGradientDescentOptimize", constantValues: constantValues)
			let pipeline: MTLComputePipelineState = try $0.makeComputePipelineState(function: function)
			return {
				StochasticGradientDescent(pipeline: pipeline, η: η, count: $0)
			}
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
