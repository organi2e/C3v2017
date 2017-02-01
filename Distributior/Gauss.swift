//
//  Gauss.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal

public class Gauss {
	let rng: MTLComputePipelineState
	let pdf: MTLComputePipelineState
	let cdf: MTLComputePipelineState
	let chain: MTLComputePipelineState
	let threads: MTLSize
	let seed: Array<UInt8>
	public init(device: MTLDevice, xorshift: (Int, Int, Int) = (13, 17, 5)) throws {
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: Bundle(for: type(of: self)))
		let const: MTLFunctionConstantValues = MTLFunctionConstantValues()
		const.setConstantValue(Array<uint>(arrayLiteral: uint(xorshift.0), uint(xorshift.1), uint(xorshift.2)), type: .uint3, withName: "xorshift")
		rng = try library.make(name: "GaussRNG", constantValues: const)
		pdf = try library.make(name: "GaussPDF")
		cdf = try library.make(name: "GaussCDF")
		chain = try library.make(name: "GaussCollectChain")
		seed = Array<UInt8>(repeating: 0, count: 4096)//upper limit
		threads = MTLSize(width: 64, height: 1, depth: 1)
	}
}
extension Gauss: Stochastic {
	public func encode(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int) {
		
		assert(commandBuffer.device===rng.device)
		assert(MemoryLayout<Float>.size*count<=χ.length&&rng.device===χ.device)
		assert(MemoryLayout<Float>.size*count<=μ.length&&rng.device===μ.device)
		assert(MemoryLayout<Float>.size*count<=σ.length&&rng.device===σ.device)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		
		arc4random_buf(UnsafeMutablePointer<UInt8>(mutating: seed), seed.count)
		
		encoder.setComputePipelineState(rng)
		encoder.setBuffer(χ, offset: 0, at: 0)
		encoder.setBuffer(μ, offset: 0, at: 1)
		encoder.setBuffer(σ, offset: 0, at: 2)
		encoder.setBytes(seed, length: seed.count, at: 3)
		encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 4)
		encoder.dispatchThreadgroups(MTLSize(width: 1+(seed.count-1)/MemoryLayout<uint>.size/4/threads.width, height: 1, depth: 1),
		                             threadsPerThreadgroup: threads)
		encoder.endEncoding()
		
	}
	public func encode(commandBuffer: MTLCommandBuffer, CDF: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int) {
		
		assert(commandBuffer.device===cdf.device)
		assert(MemoryLayout<Float>.size*count<=CDF.length&&rng.device===CDF.device)
		assert(MemoryLayout<Float>.size*count<=μ.length&&rng.device===μ.device)
		assert(MemoryLayout<Float>.size*count<=σ.length&&rng.device===σ.device)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		
		encoder.setComputePipelineState(cdf)
		encoder.setBuffer(CDF, offset: 0, at: 0)
		encoder.setBuffer(μ, offset: 0, at: 1)
		encoder.setBuffer(σ, offset: 0, at: 2)
		encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/MemoryLayout<Float>.size/16/threads.width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: threads)
		encoder.endEncoding()
	}
	public func encode(commandBuffer: MTLCommandBuffer, PDF: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int) {
		
		assert(commandBuffer.device===pdf.device)
		assert(MemoryLayout<Float>.size*count<=PDF.length&&rng.device===PDF.device)
		assert(MemoryLayout<Float>.size*count<=μ.length&&rng.device===μ.device)
		assert(MemoryLayout<Float>.size*count<=σ.length&&rng.device===σ.device)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		
		encoder.setComputePipelineState(pdf)
		encoder.setBuffer(PDF, offset: 0, at: 0)
		encoder.setBuffer(μ, offset: 0, at: 1)
		encoder.setBuffer(σ, offset: 0, at: 2)
		encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/MemoryLayout<Float>.size/16/threads.width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: threads)
		encoder.endEncoding()
	}
}
extension Gauss: Synthesis {
	public func synthesize(commandBuffer: MTLCommandBuffer,
	                ϝ: (χ: MTLBuffer, P: MTLBuffer),
	                Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		
	}
	public func collect(commandBuffer: MTLCommandBuffer,
	             Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             w: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             x: MTLBuffer,
	             count: (rows: Int, cols: Int)) {
		
		let rows: Int = count.rows
		let cols: Int = count.cols
		
		assert(MemoryLayout<Float>.size*rows<=Σ.χ.length && chain.device === Σ.χ.device)
		assert(MemoryLayout<Float>.size*rows<=Σ.μ.length && chain.device === Σ.μ.device)
		assert(MemoryLayout<Float>.size*rows<=Σ.σ.length && chain.device === Σ.σ.device)
		
		assert(MemoryLayout<Float>.size*rows*cols<=w.χ.length && chain.device === w.χ.device)
		assert(MemoryLayout<Float>.size*rows*cols<=w.μ.length && chain.device === w.μ.device)
		assert(MemoryLayout<Float>.size*rows*cols<=w.σ.length && chain.device === w.σ.device)
		
		assert(MemoryLayout<Float>.size*cols<=x.length && chain.device === x.device)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(chain)
		encoder.setBuffer(Σ.χ, offset: 0, at: 0)
		encoder.setBuffer(Σ.μ, offset: 0, at: 1)
		encoder.setBuffer(Σ.σ, offset: 0, at: 2)
		encoder.setBuffer(w.χ, offset: 0, at: 3)
		encoder.setBuffer(w.μ, offset: 0, at: 4)
		encoder.setBuffer(w.σ, offset: 0, at: 5)
		encoder.setBuffer(x, offset: 0, at: 6)
		encoder.setBytes([uint(cols-1)/16+1], length: MemoryLayout<uint>.size, at: 7)
		encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.size*16*64, at: 0)
		encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.size*16*64, at: 1)
		encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.size*16*64, at: 2)
		encoder.dispatchThreadgroups(MTLSize(width: (rows-1)/16+1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	public func collect(commendBuffer: MTLCommandBuffer,
	             Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             w: MTLBuffer,
	             x: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             count: (rows: Int, cols: Int)) {
	}
	public func collect(commandBuffer: MTLCommandBuffer,
	             Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             w: MTLBuffer,
	             x: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             count: Int) {
	}
	public func collect(commandBuffer: MTLCommandBuffer,
	             Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             b: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer),
	             count: Int) {
	}
}
extension Gauss: Distributor {
	
}
