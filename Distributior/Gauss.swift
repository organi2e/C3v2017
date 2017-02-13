//
//  Gauss.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Accelerate
import simd

import Metal
import LaObjet

public class Gauss {
	let value: MTLBuffer
	let mean: MTLBuffer
	let variance: MTLBuffer
	let width: Int
	let rng16: MTLComputePipelineState
	let synthesize16: MTLComputePipelineState
	let deltaState16: MTLComputePipelineState
	let deltaValue16: MTLComputePipelineState
	let errorState16: MTLComputePipelineState
	let errorValue16: MTLComputePipelineState
	let collectW16: MTLComputePipelineState
	let collectC16: MTLComputePipelineState
	let collectD16: MTLComputePipelineState
	private init(device: MTLDevice,
	             width: Int,
	             rng: MTLComputePipelineState,
	             synthesize: MTLComputePipelineState,
	             deltaValue: MTLComputePipelineState,
	             deltaState: MTLComputePipelineState,
	             errorValue: MTLComputePipelineState,
	             errorState: MTLComputePipelineState,
	             collectW: MTLComputePipelineState,
	             collectC: MTLComputePipelineState,
	             collectD: MTLComputePipelineState) {
		let length: Int = width * MemoryLayout<Float>.size
		self.value = device.makeBuffer(length: length, options: .storageModePrivate)
		self.mean = device.makeBuffer(length: length, options: .storageModePrivate)
		self.variance = device.makeBuffer(length: length, options: .storageModePrivate)
		self.width = width
		self.rng16 = rng
		self.synthesize16 = synthesize
		self.deltaState16 = deltaState
		self.deltaValue16 = deltaValue
		self.errorState16 = errorState
		self.errorValue16 = errorValue
		self.collectW16 = collectW
		self.collectC16 = collectC
		self.collectD16 = collectD
	}
	public static func factory(xorshift: (Int, Int, Int) = (13, 17, 5)) -> (MTLDevice) throws -> (Int) -> Distributor {
		let bundle: Bundle = Bundle(for: self)
		let const: MTLFunctionConstantValues = MTLFunctionConstantValues()
		const.setConstantValue(Array<uint>(arrayLiteral: uint(xorshift.0), uint(xorshift.1), uint(xorshift.2)), type: .uint3, withName: "xorshift")
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let rng16: MTLComputePipelineState = try library.make(name: "GaussRNG16", constantValues: const)
			let synthesize16: MTLComputePipelineState = try library.make(name: "GaussSynthesize16")
			let deltaState16: MTLComputePipelineState = try library.make(name: "GaussDeltaState16")
			let deltaValue16: MTLComputePipelineState = try library.make(name: "GaussDeltaValue16")
			let errorState16: MTLComputePipelineState = try library.make(name: "GaussErrorState16")
			let errorValue16: MTLComputePipelineState = try library.make(name: "GaussErrorValue16")
			let collectW16: MTLComputePipelineState = try library.make(name: "GaussCollectW16")
			let collectC16: MTLComputePipelineState = try library.make(name: "GaussCollectC16")
			let collectD16: MTLComputePipelineState = try library.make(name: "GaussCollectD16")
			return {
				Gauss(device: library.device,
				      width: $0,
				      rng: rng16,
				      synthesize: synthesize16,
				      deltaValue: deltaValue16,
				      deltaState: deltaState16,
				      errorValue: errorValue16,
				      errorState: errorState16,
				      collectW: collectW16,
				      collectC: collectC16,
				      collectD: collectD16);
			}
		}
	}
}
extension Gauss: Stochastic {
	public func shuffle(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int) {
		
		assert( count * MemoryLayout<Float>.size <= χ.length && χ.device === rng16.device )
		assert( count * MemoryLayout<Float>.size <= μ.length && μ.device === rng16.device )
		assert( count * MemoryLayout<Float>.size <= σ.length && σ.device === rng16.device )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let seeds: Array<UInt8> = Array<UInt8>(repeating: 0, count: 4096)
		arc4random_buf(UnsafeMutablePointer<UInt8>(mutating: seeds), seeds.count)
		encoder.setComputePipelineState(rng16)
		encoder.setBuffer(χ, offset: 0, at: 0)
		encoder.setBuffer(μ, offset: 0, at: 1)
		encoder.setBuffer(σ, offset: 0, at: 2)
		encoder.setBytes(seeds, length: seeds.count, at: 3)
		encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 4)
		encoder.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func errorValue(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, ψ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int) {
		
		assert( count * MemoryLayout<Float>.size <= Δ.length && Δ.device === errorValue16.device )
		assert( count * MemoryLayout<Float>.size <= ψ.length && ψ.device === errorValue16.device )
		assert( count * MemoryLayout<Float>.size <= μ.length && μ.device === errorValue16.device )
		assert( count * MemoryLayout<Float>.size <= σ.length && σ.device === errorValue16.device )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(errorValue16)
		encoder.setBuffer(Δ, offset: 0, at: 0)
		encoder.setBuffer(ψ, offset: 0, at: 1)
		encoder.setBuffer(μ, offset: 0, at: 2)
		encoder.setBuffer(σ, offset: 0, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/16+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func errorState(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, ψ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int) {
		
		assert( count * MemoryLayout<Float>.size <= Δ.length && Δ.device === errorState16.device )
		assert( count * MemoryLayout<Float>.size <= ψ.length && ψ.device === errorState16.device )
		assert( count * MemoryLayout<Float>.size <= μ.length && μ.device === errorState16.device )
		assert( count * MemoryLayout<Float>.size <= σ.length && σ.device === errorState16.device )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(errorState16)
		encoder.setBuffer(Δ, offset: 0, at: 0)
		encoder.setBuffer(ψ, offset: 0, at: 1)
		encoder.setBuffer(μ, offset: 0, at: 2)
		encoder.setBuffer(σ, offset: 0, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/16+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func deltaValue(commandBuffer: MTLCommandBuffer, Δμ: MTLBuffer, Δσ: MTLBuffer, Δ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int) {
		
		assert( count * MemoryLayout<Float>.size <= Δμ.length && Δμ.device === deltaValue16.device )
		assert( count * MemoryLayout<Float>.size <= Δσ.length && Δσ.device === deltaValue16.device )
		assert( count * MemoryLayout<Float>.size <= Δ.length && Δ.device === deltaValue16.device )
		assert( count * MemoryLayout<Float>.size <= μ.length && μ.device === deltaValue16.device )
		assert( count * MemoryLayout<Float>.size <= σ.length && σ.device === deltaValue16.device )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(deltaValue16)
		encoder.setBuffer(Δμ, offset: 0, at: 0)
		encoder.setBuffer(Δσ, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.setBuffer(μ, offset: 0, at: 3)
		encoder.setBuffer(σ, offset: 0, at: 4)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/16+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func deltaState(commandBuffer: MTLCommandBuffer, Δμ: MTLBuffer, Δσ: MTLBuffer, Δ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int) {
		
		assert( count * MemoryLayout<Float>.size <= Δμ.length && Δμ.device === deltaValue16.device )
		assert( count * MemoryLayout<Float>.size <= Δσ.length && Δσ.device === deltaValue16.device )
		assert( count * MemoryLayout<Float>.size <= Δ.length && Δ.device === deltaValue16.device )
		assert( count * MemoryLayout<Float>.size <= μ.length && μ.device === deltaValue16.device )
		assert( count * MemoryLayout<Float>.size <= σ.length && σ.device === deltaValue16.device )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(deltaState16)
		encoder.setBuffer(Δμ, offset: 0, at: 0)
		encoder.setBuffer(Δσ, offset: 0, at: 1)
		encoder.setBuffer(Δ, offset: 0, at: 2)
		encoder.setBuffer(μ, offset: 0, at: 3)
		encoder.setBuffer(σ, offset: 0, at: 4)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/16+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
}
extension Gauss: Synthesis {
	public func synthesize(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) {
		
		assert( width * MemoryLayout<Float>.size <= χ.length && synthesize16.device === χ.device )
		assert( width * MemoryLayout<Float>.size <= μ.length && synthesize16.device === μ.device )
		assert( width * MemoryLayout<Float>.size <= σ.length && synthesize16.device === σ.device )
		
		do {
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(synthesize16)
			encoder.setBuffers([σ, variance], offsets: [0, 0], with: NSRange(0..<2))
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
			encoder.endEncoding()
		}
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.copy(from: value, sourceOffset: 0, to: χ, destinationOffset: 0, size: value.length)
			encoder.copy(from: mean, sourceOffset: 0, to: μ, destinationOffset: 0, size: mean.length)
			encoder.endEncoding()
		}
	}
	public func collect(commandBuffer: MTLCommandBuffer, w: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: Int) {
		
		assert( width * count * MemoryLayout<Float>.size <= w.χ.length && collectW16.device === w.χ.device )
		assert( width * count * MemoryLayout<Float>.size <= w.μ.length && collectW16.device === w.μ.device )
		assert( width * count * MemoryLayout<Float>.size <= w.σ.length && collectW16.device === w.σ.device )
		
		assert( count * MemoryLayout<Float>.size <= x.length && collectW16.device === x.device )
		
		do {
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(collectW16)
			encoder.setBuffers([value, mean, variance, w.χ, w.μ, w.σ, x], offsets: [0, 0, 0, 0, 0, 0, 0], with: NSRange(0..<7))
			encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 7)
			encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.size*16*64, at: 0)
			encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.size*16*64, at: 1)
			encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.size*16*64, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func collect(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) {
		
		assert( width * MemoryLayout<Float>.size <= χ.length && collectC16.device === χ.device )
		assert( width * MemoryLayout<Float>.size <= μ.length && collectC16.device === μ.device )
		assert( width * MemoryLayout<Float>.size <= σ.length && collectC16.device === σ.device )
		
		do {
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(collectC16)
			encoder.setBuffers([value, mean, variance, χ, μ, σ], offsets: [0, 0, 0, 0, 0, 0], with: NSRange(0..<6))
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func collect(commandBuffer: MTLCommandBuffer, r: MTLBuffer, x: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer)) {
		
		assert( width * MemoryLayout<Float>.size <= r.length && collectD16.device === r.device )
		
		assert( width * MemoryLayout<Float>.size <= x.χ.length && collectC16.device === x.χ.device )
		assert( width * MemoryLayout<Float>.size <= x.μ.length && collectC16.device === x.μ.device )
		assert( width * MemoryLayout<Float>.size <= x.σ.length && collectC16.device === x.σ.device )
		
		do {
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(collectD16)
			encoder.setBuffers([value, mean, variance, r, x.χ, x.μ, x.σ], offsets: [0, 0, 0, 0, 0, 0, 0], with: NSRange(0..<7))
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func reset(commandBuffer: MTLCommandBuffer) {
		do {
			let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.fill(buffer: value, range: NSRange(0..<value.length), value: 0)
			encoder.fill(buffer: mean, range: NSRange(0..<mean.length), value: 0)
			encoder.fill(buffer: variance, range: NSRange(0..<variance.length), value: 0)
			encoder.endEncoding()
		}
		
	}
}
extension Gauss: Distributor {
	
}
/*
public class Gauss {
	let Σ: (χ: Buffer, μ: Buffer, σ: Buffer)
	let rng: ComputePipelineState
	let collectA: ComputePipelineState
	let collectC: ComputePipelineState
	let collectD: ComputePipelineState
	let synthesize: ComputePipelineState
	let threads: MTLSize
	let computer: Computer
	let count: Int
	private init(computer: Computer,
	             pipelines: (
					rng: ComputePipelineState,
					collectA: ComputePipelineState,
					collectC: ComputePipelineState,
					collectD: ComputePipelineState,
					synthesize: ComputePipelineState),
	             count: Int) {
		Σ = (χ: computer.make(length: count * MemoryLayout<Float>.size, options: .storageModePrivate),
			 μ: computer.make(length: count * MemoryLayout<Float>.size, options: .storageModePrivate),
			 σ: computer.make(length: count * MemoryLayout<Float>.size, options: .storageModePrivate)
		)
		rng = pipelines.rng
		collectA = pipelines.collectA
		collectC = pipelines.collectC
		collectD = pipelines.collectD
		synthesize = pipelines.synthesize
		self.computer = computer
		self.count = count
		threads = MTLSize(width: 64, height: 1, depth: 1)
	}
	public static func make(xorshift: (Int, Int, Int) = (13, 17, 5)) -> (Computer) throws -> (Int) -> Distributor {
		let bundle: Bundle = Bundle(for: self)
		let const: MTLFunctionConstantValues = MTLFunctionConstantValues()
		const.setConstantValue(Array<uint>(arrayLiteral: uint(xorshift.0), uint(xorshift.1), uint(xorshift.2)), type: .uint3, withName: "xorshift")
		return {
			let computer: Computer = $0
			let library: Library = try $0.device.makeDefaultLibrary(bundle: bundle)
			let rng: ComputePipelineState = try library.make(name: "GaussRNG", constantValues: const)
			let collectA: ComputePipelineState = try library.make(name: "GaussCollectA")
			let collectC: ComputePipelineState = try library.make(name: "GaussCollectC")
			let collectD: ComputePipelineState = try library.make(name: "GaussCollectD")
			let synthesize: ComputePipelineState = try library.make(name: "GaussSynthesize")
			return {
				return Gauss(computer: computer,
				         pipelines: (
							rng: rng,
							collectA: collectA,
							collectC: collectC,
							collectD: collectD,
							synthesize: synthesize
						), count: $0)
			}
		}
	}
}
extension Gauss: Synthesis {
	public func synthesize(commandBuffer: MTLCommandBuffer, ϝ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), count: Int) {
	
	}
	public func μ(_ μ: LaObjet) -> LaObjet {
		return μ
	}
	public func σ(_ σ: LaObjet) -> LaObjet {
		return σ * σ
	}
	public func synthesize(ϝ: Buffer, μ: Buffer, σ: Buffer) {
		
		assert(count*MemoryLayout<Float>.size <= ϝ.length && ϝ.device === computer.device)
		assert(count*MemoryLayout<Float>.size <= μ.length && μ.device === computer.device)
		assert(count*MemoryLayout<Float>.size <= σ.length && σ.device === computer.device)
		
		computer.compute {
			let encoder: ComputeCommandEncoder = $0.makeComputeCommandEncoder()
			encoder.setComputePipelineState(synthesize)
			encoder.setBuffer(ϝ, offset: 0, at: 0)
			encoder.setBuffer(μ, offset: 0, at: 1)
			encoder.setBuffer(σ, offset: 0, at: 2)
			encoder.setBuffer(Σ.χ, offset: 0, at: 3)
			encoder.setBuffer(Σ.μ, offset: 0, at: 4)
			encoder.setBuffer(Σ.σ, offset: 0, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func collect(r: Buffer, x: (χ: Buffer, μ: Buffer, σ: Buffer)) {
	
	}
	public func collect(w: (χ: Buffer, μ: Buffer, σ: Buffer), x: Buffer) {
		
		let rows: Int = count
		let cols: Int = x.length / MemoryLayout<Float>.size
		
		assert(MemoryLayout<Float>.size*rows*cols <= w.χ.length && computer.device === w.χ.device)
		assert(MemoryLayout<Float>.size*rows*cols <= w.μ.length && computer.device === w.μ.device)
		assert(MemoryLayout<Float>.size*rows*cols <= w.σ.length && computer.device === w.σ.device)
		
		assert(computer.device === x.device)
		
		computer.compute {
			let encoder: MTLComputeCommandEncoder = $0.makeComputeCommandEncoder()
			encoder.setComputePipelineState(collectA)
			encoder.setBuffer(Σ.χ, offset: 0, at: 0)
			encoder.setBuffer(Σ.μ, offset: 0, at: 1)
			encoder.setBuffer(Σ.σ, offset: 0, at: 2)
			encoder.setBuffer(w.χ, offset: 0, at: 3)
			encoder.setBuffer(w.μ, offset: 0, at: 4)
			encoder.setBuffer(w.σ, offset: 0, at: 5)
			encoder.setBuffer(x, offset: 0, at: 6)
			encoder.setBytes([uint(cols-1)/16+1], length: MemoryLayout<uint>.size, at: 7)
			encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.size*16*threads.width, at: 0)
			encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.size*16*threads.width, at: 1)
			encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.size*16*threads.width, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (rows-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: threads)
			encoder.endEncoding()
		}
	}
	public func collect(x: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer)) {
		
		assert(count*MemoryLayout<Float>.size <= x.χ.length && computer.device === x.χ.device)
		assert(count*MemoryLayout<Float>.size <= x.μ.length && computer.device === x.μ.device)
		assert(count*MemoryLayout<Float>.size <= x.σ.length && computer.device === x.σ.device)
		
		computer.compute {
			let encoder: ComputeCommandEncoder = $0.makeComputeCommandEncoder()
			encoder.setComputePipelineState(collectC)
			encoder.setBuffer(Σ.χ, offset: 0, at: 0)
			encoder.setBuffer(Σ.μ, offset: 0, at: 1)
			encoder.setBuffer(Σ.σ, offset: 0, at: 2)
			encoder.setBuffer(x.χ, offset: 0, at: 3)
			encoder.setBuffer(x.μ, offset: 0, at: 4)
			encoder.setBuffer(x.σ, offset: 0, at: 5)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func clear() {
		computer.compute {
			let encoder: BlitCommandEncoder = $0.makeBlitCommandEncoder()
			[Σ.χ, Σ.μ, Σ.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
	}
}
extension Gauss: Stochastic {
	public func shuffle(χ: Buffer, from: (μ: Buffer, σ: Buffer)) {
		
		let count: Int = χ.length / MemoryLayout<Float>.size
		
		assert(count * MemoryLayout<Float>.size <= χ.length && computer.device === χ.device)
		assert(count * MemoryLayout<Float>.size <= from.μ.length && computer.device === from.μ.device)
		assert(count * MemoryLayout<Float>.size <= from.σ.length && computer.device === from.σ.device)
		
		computer.compute {
			let encoder: MTLComputeCommandEncoder = $0.makeComputeCommandEncoder()
			let seed: Array<UInt8> = Array<UInt8>(repeating: 0, count: 4096)//upper limit
			arc4random_buf(UnsafeMutablePointer<UInt8>(mutating: seed), seed.count)
			encoder.setComputePipelineState(rng)
			encoder.setBuffer(χ, offset: 0, at: 0)
			encoder.setBuffer(from.μ, offset: 0, at: 1)
			encoder.setBuffer(from.σ, offset: 0, at: 2)
			encoder.setBytes(seed, length: seed.count, at: 3)
			encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(MTLSize(width: 1+(seed.count-1)/MemoryLayout<uint>.size/4/threads.width, height: 1, depth: 1), threadsPerThreadgroup: threads)
			encoder.endEncoding()
		}
	}
}
extension Gauss: Distributor {
	
}
/*
public class Gauss {
	let rng: MTLComputePipelineState
	let pdf: MTLComputePipelineState
	let cdf: MTLComputePipelineState
	let chain: MTLComputePipelineState
	let collectBias: MTLComputePipelineState
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
		collectBias = try library.make(name: "GaussCollectBias")
		seed = Array<UInt8>(repeating: 0, count: 4096)//upper limit
		threads = MTLSize(width: 64, height: 1, depth: 1)
	}
	/*
	private init(queue: MTLCommandQueue, pipelines: (rng: MTLComputePipelineState, pdf: MTLComputePipelineState, cdf: MTLComputePipelineState, chain: MTLComputePipelineState)) {
		rng = pipelines.rng
		pdf = pipelines.pdf
		cdf = pipelines.cdf
		chain = pipelines.chain
		seed = Array<UInt8>(repeating: 0, count: 4096)//upper limit
		threads = MTLSize(width: 64, height: 1, depth: 1)
		enqueue = { queue.makeCommandBuffer() }
	}
	public static func make(xorshift: (Int, Int, Int) = (13, 17, 5)) -> (MTLCommandQueue) throws -> () -> Distributor {
		let bundle: Bundle = Bundle(for: self)
		let const: MTLFunctionConstantValues = MTLFunctionConstantValues()
		const.setConstantValue(Array<uint>(arrayLiteral: uint(xorshift.0), uint(xorshift.1), uint(xorshift.2)), type: .uint3, withName: "xorshift")
		return {
			let x = $0
			let library: MTLLibrary = try $0.device.makeDefaultLibrary(bundle: bundle)
			let rng: MTLComputePipelineState = try library.make(name: "GaussRNG", constantValues: const)
			let pdf: MTLComputePipelineState = try library.make(name: "GaussPDF")
			let cdf: MTLComputePipelineState = try library.make(name: "GaussCDF")
			let chain: MTLComputePipelineState = try library.make(name: "GaussCollectChain")
			return {
				Gauss(queue: x, pipelines: (rng: rng, pdf: pdf, cdf: cdf, chain: chain))
			}
		}
	}
	*/
}
extension Gauss: Stochastic {
	public func compute(commandBuffer: MTLCommandBuffer, cdf to: MTLBuffer, of: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		
		assert(count*MemoryLayout<Float>.size<=to.length&&rng.device===to.device)
		assert(count*MemoryLayout<Float>.size<=of.μ.length&&rng.device===of.μ.device)
		assert(count*MemoryLayout<Float>.size<=of.σ.length&&rng.device===of.σ.device)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		
		encoder.setComputePipelineState(cdf)
		encoder.setBuffer(to, offset: 0, at: 0)
		encoder.setBuffer(of.μ, offset: 0, at: 1)
		encoder.setBuffer(of.σ, offset: 0, at: 2)
		encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/MemoryLayout<Float>.size/16/threads.width+1, height: 1, depth: 1), threadsPerThreadgroup: threads)
		encoder.endEncoding()
	}
	public func compute(commandBuffer: MTLCommandBuffer, pdf to: MTLBuffer, of: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		
		assert(count*MemoryLayout<Float>.size<=to.length&&rng.device===to.device)
		assert(count*MemoryLayout<Float>.size<=of.μ.length&&rng.device===of.μ.device)
		assert(count*MemoryLayout<Float>.size<=of.σ.length&&rng.device===of.σ.device)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		
		encoder.setComputePipelineState(pdf)
		encoder.setBuffer(to, offset: 0, at: 0)
		encoder.setBuffer(of.μ, offset: 0, at: 1)
		encoder.setBuffer(of.σ, offset: 0, at: 2)
		encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/MemoryLayout<Float>.size/16/threads.width+1, height: 1, depth: 1), threadsPerThreadgroup: threads)
		encoder.endEncoding()
	}
	public func shuffle(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, from: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		
		assert(count*MemoryLayout<Float>.size<=χ.length&&rng.device===χ.device)
		assert(count*MemoryLayout<Float>.size<=from.μ.length&&rng.device===from.μ.device)
		assert(count*MemoryLayout<Float>.size<=from.σ.length&&rng.device===from.σ.device)
	
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		
		arc4random_buf(UnsafeMutablePointer<UInt8>(mutating: seed), seed.count)
		
		encoder.setComputePipelineState(rng)
		encoder.setBuffer(χ, offset: 0, at: 0)
		encoder.setBuffer(from.μ, offset: 0, at: 1)
		encoder.setBuffer(from.σ, offset: 0, at: 2)
		encoder.setBytes(seed, length: seed.count, at: 3)
		encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 4)
		encoder.dispatchThreadgroups(MTLSize(width: 1+(seed.count-1)/MemoryLayout<uint>.size/4/threads.width, height: 1, depth: 1), threadsPerThreadgroup: threads)
		encoder.endEncoding()
	}
}
extension Gauss: Synthesis {
	public func synthesize(commandBuffer: MTLCommandBuffer,
	                       ϝ: (χ: MTLBuffer, P: MTLBuffer),
	                       Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.endEncoding()
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
	public func collect(commandBuffer: MTLCommandBuffer,
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
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(collectBias)
		encoder.setBuffer(Σ.χ, offset: 0, at: 0)
		encoder.setBuffer(Σ.μ, offset: 0, at: 1)
		encoder.setBuffer(Σ.σ, offset: 0, at: 2)
		encoder.setBuffer(b.χ, offset: 0, at: 3)
		encoder.setBuffer(b.μ, offset: 0, at: 4)
		encoder.setBuffer(b.σ, offset: 0, at: 5)
		encoder.dispatchThreadgroups(MTLSize(width: (count-1)/16+1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
}
extension Gauss: Distributor {
	
}
*/
*/
