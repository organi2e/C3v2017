//
//  Cauchy.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal
/*
public class Cauchy {
	let rng: MTLComputePipelineState
	let pdf: MTLComputePipelineState
	let cdf: MTLComputePipelineState
	let seed: Array<uint>
	init(device: MTLDevice, parallel: Int = 64, xorshift: (Int, Int, Int) = (13, 17, 5)) throws {
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: Bundle(for: type(of: self)))
		let const: MTLFunctionConstantValues = MTLFunctionConstantValues()
		const.setConstantValue(Array<uint>(arrayLiteral: uint(xorshift.0), uint(xorshift.1), uint(xorshift.2)), type: .uint3, withName: "xorshift")
		rng = try device.makeComputePipelineState(function: try library.makeFunction(name: "CauchyRNG", constantValues: const))
		pdf = try device.makeComputePipelineState(function: try library.makeFunction(name: "CauchyPDF", constantValues: MTLFunctionConstantValues()))
		cdf = try device.makeComputePipelineState(function: try library.makeFunction(name: "CauchyCDF", constantValues: MTLFunctionConstantValues()))
		seed = Array<uint>(repeating: 0, count: 4*parallel)
	}
}
extension Cauchy: Distributior {
	public func encode(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) {
		
		let count: Int = min(χ.length, μ.length, σ.length) / MemoryLayout<Float>.size
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		
		arc4random_buf(UnsafeMutablePointer<uint>(mutating: seed), MemoryLayout<uint>.size*seed.count)
		
		encoder.setComputePipelineState(rng)
		encoder.setBuffer(χ, offset: 0, at: 0)
		encoder.setBuffer(μ, offset: 0, at: 1)
		encoder.setBuffer(σ, offset: 0, at: 2)
		encoder.setBytes(seed, length: MemoryLayout<uint>.size*seed.count, at: 3)
		encoder.setBytes([uint(count-1)/4+1], length: MemoryLayout<uint>.size, at: 4)
		encoder.dispatchThreadgroups(MTLSize(width: seed.count/4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	public func encode(commandBuffer: MTLCommandBuffer, CDF: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) {
		let count: Int = min(CDF.length, μ.length, σ.length) / MemoryLayout<Float>.size
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(cdf)
		encoder.setBuffer(CDF, offset: 0, at: 0)
		encoder.setBuffer(μ, offset: 0, at: 1)
		encoder.setBuffer(σ, offset: 0, at: 2)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/4+1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func encode(commandBuffer: MTLCommandBuffer, PDF: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) {
		let count: Int = min(PDF.length, μ.length, σ.length) / MemoryLayout<Float>.size
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(pdf)
		encoder.setBuffer(PDF, offset: 0, at: 0)
		encoder.setBuffer(μ, offset: 0, at: 1)
		encoder.setBuffer(σ, offset: 0, at: 2)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/4+1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
}
*/
