//
//  Exponential.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/09.
//
//
import Metal
public class Exponential {
	let generate: MTLComputePipelineState
	let gradient: MTLComputePipelineState
	let limit: Array<uint>
	let groups: MTLSize
	let threads: MTLSize
	private init(pipeline: (MTLComputePipelineState, MTLComputePipelineState), count: Int) {
		let width: Int = min(pipeline.0.threadExecutionWidth, pipeline.1.threadExecutionWidth)
		generate = pipeline.0
		gradient = pipeline.1
		limit = Array<uint>(arrayLiteral: uint(count))
		groups = MTLSize(width: (count-1)/width+1, height: 1, depth: 1)
		threads = MTLSize(width: width, height: 1, depth: 1)
	}
	public static func factory() -> (MTLDevice) throws -> (Int) -> Adapter {
		let bundle: Bundle = Bundle(for: self)
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let generate: MTLComputePipelineState = try library.make(name: "ExponentialGenerate")
			let gradient: MTLComputePipelineState = try library.make(name: "ExponentialGradient")
			return {
				Exponential(pipeline: (generate, gradient), count: $0)
			}
		}
	}
}
extension Exponential: Adapter {
	public func generate(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
		assert( commandBuffer.device === generate.device )
		assert( commandBuffer.device === θ.device && Int(limit[0]) * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && Int(limit[0]) * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(generate)
		encoder.setBuffer(θ, offset: 0, at: 0)
		encoder.setBuffer(φ, offset: 0, at: 1)
		encoder.setBytes(limit, length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
		encoder.endEncoding()
		
	}
	public func gradient(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, θ: MTLBuffer, φ: MTLBuffer) {
		
		assert( commandBuffer.device === gradient.device )
		assert( commandBuffer.device === θ.device && Int(limit[0]) * MemoryLayout<Float>.size <= θ.length )
		assert( commandBuffer.device === φ.device && Int(limit[0]) * MemoryLayout<Float>.size <= φ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(gradient)
		encoder.setBuffer(Δ, offset: 0, at: 0)
		encoder.setBuffer(θ, offset: 0, at: 1)
		encoder.setBuffer(φ, offset: 0, at: 2)
		encoder.setBytes(limit, length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
		encoder.endEncoding()
		
	}
}
