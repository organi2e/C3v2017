//
//  Gauss.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Accelerate
import Metal
import simd

public class GaussDistributor {
	let activateP: MTLComputePipelineState
	let derivateP: MTLComputePipelineState
	let collect: MTLComputePipelineState
	let collectW: MTLComputePipelineState
	let collectC: MTLComputePipelineState
	let collectD: MTLComputePipelineState
	let jacobian: MTLComputePipelineState
	let jacobianA: MTLComputePipelineState
	let jacobianB: MTLComputePipelineState
	let jacobianC: MTLComputePipelineState
	let jacobianD: MTLComputePipelineState
	let jacobianX: MTLComputePipelineState
	let deltaJ: MTLComputePipelineState
	let deltaG: MTLComputePipelineState
	let deltaX: MTLComputePipelineState
	public init(activateP: MTLComputePipelineState,
	            derivateP: MTLComputePipelineState,
	            collect: MTLComputePipelineState,
	            collectW: MTLComputePipelineState,
	            collectC: MTLComputePipelineState,
	            collectD: MTLComputePipelineState,
	            jacobian: MTLComputePipelineState,
	            jacobianA: MTLComputePipelineState,
	            jacobianB: MTLComputePipelineState,
	            jacobianC: MTLComputePipelineState,
	            jacobianD: MTLComputePipelineState,
	            jacobianX: MTLComputePipelineState,
	            deltaJ: MTLComputePipelineState,
	            deltaG: MTLComputePipelineState,
	            deltaX: MTLComputePipelineState) {
		self.activateP = activateP
		self.derivateP = derivateP
		self.collect = collect
		self.collectW = collectW
		self.collectC = collectC
		self.collectD = collectD
		self.jacobian = jacobian
		self.jacobianA = jacobianA
		self.jacobianB = jacobianB
		self.jacobianC = jacobianC
		self.jacobianD = jacobianD
		self.jacobianX = jacobianX
		self.deltaJ = deltaJ
		self.deltaG = deltaG
		self.deltaX = deltaX
	}
	public static func factory() -> (MTLDevice) throws -> Distributor {
		let bundle: Bundle = Bundle(for: self)
		//let const: MTLFunctionConstantValues = MTLFunctionConstantValues()
		return {
			let library: MTLLibrary = try $0.makeDefaultLibrary(bundle: bundle)
			let activateP: MTLComputePipelineState = try library.make(name: "GaussActivateP")
			let derivateP: MTLComputePipelineState = try library.make(name: "GaussDerivateP")
			let collect: MTLComputePipelineState = try library.make(name: "GaussCollect")
			let collectW: MTLComputePipelineState = try library.make(name: "GaussCollectW")
			let collectC: MTLComputePipelineState = try library.make(name: "GaussCollectC")
			let collectD: MTLComputePipelineState = try library.make(name: "GaussCollectD")
			let jacobian: MTLComputePipelineState = try library.make(name: "GaussJacobian")
			let jacobianA: MTLComputePipelineState = try library.make(name: "GaussJacobianA")
			let jacobianB: MTLComputePipelineState = try library.make(name: "GaussJacobianB")
			let jacobianC: MTLComputePipelineState = try library.make(name: "GaussJacobianC")
			let jacobianD: MTLComputePipelineState = try library.make(name: "GaussJacobianD")
			let jacobianX: MTLComputePipelineState = try library.make(name: "GaussJacobianX")
			let deltaJ: MTLComputePipelineState = try library.make(name: "GaussDeltaJ")
			let deltaG: MTLComputePipelineState = try library.make(name: "GaussDeltaG")
			let deltaX: MTLComputePipelineState = try library.make(name: "GaussDeltaX")
			return GaussDistributor(activateP: activateP,
			                        derivateP: derivateP,
			                        collect: collect,
			                        collectW: collectW,
			                        collectC: collectC,
			                        collectD: collectD,
			                        jacobian: jacobian,
			                        jacobianA: jacobianA,
			                        jacobianB: jacobianB,
			                        jacobianC: jacobianC,
			                        jacobianD: jacobianD,
			                        jacobianX: jacobianX,
			                        deltaJ: deltaJ,
			                        deltaG: deltaG,
			                        deltaX: deltaX)
		}
	}
}
extension GaussDistributor: Activative {
	public func activate(commandBuffer: MTLCommandBuffer,
	                     y: (χ: MTLBuffer, p: MTLBuffer),
	                     v: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = activateP.threadExecutionWidth
		let seed: Array<UInt8> = Array<UInt8>(repeating: 0, count: count)
		
		assert( encoder.device === activateP.device )
		
		assert( count * MemoryLayout<Float>.size <= y.χ.length && activateP.device === y.χ.device )
		assert( count * MemoryLayout<Float>.size <= y.p.length && activateP.device === y.p.device )
		assert( count * MemoryLayout<Float>.size <= v.μ.length && activateP.device === v.μ.device )
		assert( count * MemoryLayout<Float>.size <= v.σ.length && activateP.device === v.σ.device )
		
		arc4random_buf(UnsafeMutablePointer<UInt8>(mutating: seed), seed.count)
		encoder.setComputePipelineState(activateP)
		encoder.setBuffer(y.χ, offset: 0, at: 0)
		encoder.setBuffer(y.p, offset: 0, at: 1)
		encoder.setBuffer(v.μ, offset: 0, at: 2)
		encoder.setBuffer(v.σ, offset: 0, at: 3)
		encoder.setBytes(seed, length: seed.count, at: 4)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(.init(width: (count-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func collect(commandBuffer: MTLCommandBuffer,
	                    v: (μ: MTLBuffer, σ: MTLBuffer),
	                    Σ: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = collect.threadExecutionWidth
		
		assert( collect.device === encoder.device )
		
		assert( collect.device === v.μ.device && count * MemoryLayout<Float>.size <= v.μ.length )
		assert( collect.device === v.σ.device && count * MemoryLayout<Float>.size <= v.σ.length )
		
		assert( collect.device === Σ.μ.device && count * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( collect.device === Σ.σ.device && count * MemoryLayout<Float>.size <= Σ.σ.length)
		
		encoder.setComputePipelineState(collect)
		encoder.setBuffer(v.μ, offset: 0, at: 0)
		encoder.setBuffer(v.σ, offset: 0, at: 1)
		encoder.setBuffer(Σ.μ, offset: 0, at: 2)
		encoder.setBuffer(Σ.σ, offset: 0, at: 3)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 4)
		encoder.dispatchThreadgroups(.init(width: (count-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func collect(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), w: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: (rows: Int, cols: Int)) {
		
		assert( collectW.device === commandBuffer.device )
		
		assert( collectW.device === Σ.μ.device && count.rows * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( collectW.device === Σ.σ.device && count.rows * MemoryLayout<Float>.size <= Σ.σ.length)
		
		assert( collectW.device === w.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= w.μ.length )
		assert( collectW.device === w.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= w.σ.length )
		
		assert( collectW.device === x.device && count.cols * MemoryLayout<Float>.size <= x.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = collectW.threadExecutionWidth
		encoder.setComputePipelineState(collectW)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(w.μ, offset: 0, at: 2)
		encoder.setBuffer(w.σ, offset: 0, at: 3)
		encoder.setBuffer(x, offset: 0, at: 4)
		encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.setThreadgroupMemoryLength(4*width*MemoryLayout<Float>.size, at: 0)
		encoder.setThreadgroupMemoryLength(4*width*MemoryLayout<Float>.size, at: 1)
		encoder.dispatchThreadgroups(.init(width: (count.rows+3)/4, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func collect(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), c: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		
		assert( collectC.device === commandBuffer.device )
		
		assert( collectC.device === Σ.μ.device && count * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( collectC.device === Σ.σ.device && count * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( collectC.device === c.μ.device && count * MemoryLayout<Float>.size <= c.μ.length )
		assert( collectC.device === c.σ.device && count * MemoryLayout<Float>.size <= c.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = collectC.threadExecutionWidth
		encoder.setComputePipelineState(collectC)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(c.μ, offset: 0, at: 2)
		encoder.setBuffer(c.σ, offset: 0, at: 3)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 4)
		encoder.dispatchThreadgroups(.init(width: (count-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
	}
	
	public func collect(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, v: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		
		assert( collectD.device === commandBuffer.device )
		
		assert( collectD.device === Σ.μ.device && count * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( collectD.device === Σ.σ.device && count * MemoryLayout<Float>.size <= Σ.σ.length )
		
		assert( collectD.device === d.device && count * MemoryLayout<Float>.size <= d.length )
		
		assert( collectD.device === v.μ.device && count * MemoryLayout<Float>.size <= v.μ.length )
		assert( collectD.device === v.σ.device && count * MemoryLayout<Float>.size <= v.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = collectD.threadExecutionWidth
		
		encoder.setComputePipelineState(collectD)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(d, offset: 0, at: 2)
		encoder.setBuffer(v.μ, offset: 0, at: 3)
		encoder.setBuffer(v.σ, offset: 0, at: 4)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(.init(width: (count-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
	}

}
extension GaussDistributor: Derivative {
	public func derivate(commandBuffer: MTLCommandBuffer,
	                     Δ: (μ: MTLBuffer, σ: MTLBuffer),
	                     g: (μ: MTLBuffer, σ: MTLBuffer),
	                     y: (Δ: MTLBuffer, p: MTLBuffer),
	                     v: (μ: MTLBuffer, σ: MTLBuffer), count: Int) {
		
		assert( derivateP.device === commandBuffer.device )
		
		assert( activateP.device === Δ.μ.device && count * MemoryLayout<Float>.size <= Δ.μ.length )
		assert( activateP.device === Δ.σ.device && count * MemoryLayout<Float>.size <= Δ.σ.length )
		assert( activateP.device === g.μ.device && count * MemoryLayout<Float>.size <= g.μ.length )
		assert( activateP.device === g.σ.device && count * MemoryLayout<Float>.size <= g.σ.length )
		assert( activateP.device === y.Δ.device && count * MemoryLayout<Float>.size <= y.Δ.length )
		assert( activateP.device === y.p.device && count * MemoryLayout<Float>.size <= y.p.length )
		assert( activateP.device === v.μ.device && count * MemoryLayout<Float>.size <= v.μ.length )
		assert( activateP.device === v.σ.device && count * MemoryLayout<Float>.size <= v.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = derivateP.threadExecutionWidth
		encoder.setComputePipelineState(derivateP)
		encoder.setBuffer(Δ.μ, offset: 0, at: 0)
		encoder.setBuffer(Δ.σ, offset: 0, at: 1)
		encoder.setBuffer(g.μ, offset: 0, at: 2)
		encoder.setBuffer(g.σ, offset: 0, at: 3)
		encoder.setBuffer(y.Δ, offset: 0, at: 4)
		encoder.setBuffer(y.p, offset: 0, at: 5)
		encoder.setBuffer(v.μ, offset: 0, at: 6)
		encoder.setBuffer(v.σ, offset: 0, at: 7)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 8)
		encoder.dispatchThreadgroups(.init(width: (count-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	public func delta(commandBuffer: MTLCommandBuffer,
	                  Δ: MTLBuffer,
	                  j: (μ: MTLBuffer, σ: MTLBuffer),
	                  g: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int)) {
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = deltaJ.threadExecutionWidth
		encoder.setComputePipelineState(deltaX)
		encoder.setBuffer(Δ, offset: 0, at: 0)
		encoder.setBuffer(j.μ, offset: 0, at: 1)
		encoder.setBuffer(j.σ, offset: 0, at: 2)
		encoder.setBuffer(g.μ, offset: 0, at: 3)
		encoder.setBuffer(g.σ, offset: 0, at: 4)
		encoder.setBytes([uint(count.cols), uint(count.rows)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.setThreadgroupMemoryLength(4*width*MemoryLayout<Float>.size, at: 0)
		encoder.dispatchThreadgroups(.init(width: (count.cols+3)/4, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func delta(commandBuffer: MTLCommandBuffer,
	                  Δ: (μ: MTLBuffer, σ: MTLBuffer),
	                  j: (μ: MTLBuffer, σ: MTLBuffer),
	                  g: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int), rtrl: Bool = false) {
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		
		encoder.setBuffer(Δ.μ, offset: 0, at: 0)
		encoder.setBuffer(Δ.σ, offset: 0, at: 1)
		encoder.setBuffer(j.μ, offset: 0, at: 2)
		encoder.setBuffer(j.σ, offset: 0, at: 3)
		encoder.setBuffer(g.μ, offset: 0, at: 4)
		encoder.setBuffer(g.σ, offset: 0, at: 5)
		
		if rtrl {
			
			assert( deltaJ.device === encoder.device )
			assert( deltaJ.device === Δ.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Δ.μ.length )
			assert( deltaJ.device === Δ.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Δ.σ.length )
			assert( deltaJ.device === j.μ.device && count.rows * count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
			assert( deltaJ.device === j.σ.device && count.rows * count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
			assert( deltaJ.device === g.μ.device && count.rows * MemoryLayout<Float>.size <= g.μ.length )
			assert( deltaJ.device === g.σ.device && count.rows * MemoryLayout<Float>.size <= g.σ.length )
			
			let width: Int = deltaJ.threadExecutionWidth
			encoder.setBytes([uint(count.rows*count.cols), uint(count.rows)], length: 2*MemoryLayout<uint>.size, at: 6)
			encoder.setThreadgroupMemoryLength(4*width*MemoryLayout<Float>.size, at: 0)
			encoder.setThreadgroupMemoryLength(4*width*MemoryLayout<Float>.size, at: 1)
			encoder.setComputePipelineState(deltaJ)
			encoder.dispatchThreadgroups(.init(width: (count.rows*count.cols+3)/4, height: 1, depth: 1),
			                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		} else {
			
			assert( deltaG.device === encoder.device )
			assert( deltaG.device === Δ.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Δ.μ.length )
			assert( deltaG.device === Δ.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Δ.σ.length )
			assert( deltaG.device === j.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.μ.length )
			assert( deltaG.device === j.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= j.σ.length )
			assert( deltaG.device === g.μ.device && count.rows * MemoryLayout<Float>.size <= g.μ.length )
			assert( deltaG.device === g.σ.device && count.rows * MemoryLayout<Float>.size <= g.σ.length )
			
			encoder.setBytes([uint(count.cols)], length: MemoryLayout<uint>.size, at: 6)
			encoder.setComputePipelineState(deltaG)
			encoder.dispatchThreadgroups(.init(width: count.rows, height: count.cols, depth: 1),
			                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		}
		encoder.endEncoding()
	}
	//synthesize jacob with v
	public func jacobian(commandBuffer: MTLCommandBuffer,
	                     j: (μ: MTLBuffer, σ: MTLBuffer),
	                     v: (μ: MTLBuffer, σ: MTLBuffer),
	                     Σ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int), rtrl: Bool = false) {
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let ref: Int = count.cols * ( rtrl ? count.rows : 1 )
		
		assert( jacobian.device === encoder.device )
		
		assert( jacobian.device === j.μ.device && count.rows * ref * MemoryLayout<Float>.size <= j.μ.length )
		assert( jacobian.device === j.σ.device && count.rows * ref * MemoryLayout<Float>.size <= j.σ.length )
		assert( jacobian.device === v.μ.device && count.rows * MemoryLayout<Float>.size <= v.μ.length )
		assert( jacobian.device === v.σ.device && count.rows * MemoryLayout<Float>.size <= v.σ.length )
		assert( jacobian.device === Σ.μ.device && count.rows * ref * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( jacobian.device === Σ.σ.device && count.rows * ref * MemoryLayout<Float>.size <= Σ.σ.length )
		
		encoder.setComputePipelineState(jacobian)
		encoder.setBuffer(j.μ, offset: 0, at: 0)
		encoder.setBuffer(j.σ, offset: 0, at: 1)
		encoder.setBuffer(v.μ, offset: 0, at: 2)
		encoder.setBuffer(v.σ, offset: 0, at: 3)
		encoder.setBuffer(Σ.μ, offset: 0, at: 4)
		encoder.setBuffer(Σ.σ, offset: 0, at: 5)
		encoder.setBytes([uint(ref)], length: MemoryLayout<uint>.size, at: 6)
		encoder.dispatchThreadgroups(.init(width: count.rows, height: ref, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	//ja
	public func jacobian(commandBuffer: MTLCommandBuffer,
	                     Σ: (μ: MTLBuffer, σ: MTLBuffer),
	                     a: (μ: MTLBuffer, σ: MTLBuffer),
	                     x: MTLBuffer, count: (rows: Int, cols: Int), rtrl: Bool = false) {
		
		let ld: Int = rtrl ? count.rows + 1 : 1
		let ref: Int = count.cols * ( rtrl ? count.rows : 1 )
		
		assert( jacobianA.device === commandBuffer.device )
		
		assert( jacobianA.device === Σ.μ.device && count.rows * ref * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( jacobianA.device === Σ.σ.device && count.rows * ref * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( jacobianA.device === a.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= a.μ.length )
		assert( jacobianA.device === a.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= a.σ.length )
		assert( jacobianA.device === x.device && count.cols * MemoryLayout<Float>.size <= x.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianA)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(a.μ, offset: 0, at: 2)
		encoder.setBuffer(a.σ, offset: 0, at: 3)
		encoder.setBuffer(x, offset: 0, at: 4)
		encoder.setBytes([uint(ld), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(.init(width: count.rows, height: count.cols, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	//jb
	public func jacobian(commandBuffer: MTLCommandBuffer,
	                     Σ: (μ: MTLBuffer, σ: MTLBuffer),
	                     b: (μ: MTLBuffer, σ: MTLBuffer),
	                     g: (μ: MTLBuffer, σ: MTLBuffer),
	                     j: (μ: MTLBuffer, σ: MTLBuffer),
	                     y: MTLBuffer, count: (rows: Int, cols: Int), rtrl: Bool = false) {
		
		let ref: Int = count.cols * ( rtrl ? count.rows : 1 )
		
		assert( jacobianB.device === commandBuffer.device )
		
		assert( jacobianB.device === Σ.μ.device && count.rows * ref * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( jacobianB.device === Σ.σ.device && count.rows * ref * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( jacobianB.device === b.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= b.μ.length )
		assert( jacobianB.device === b.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= b.σ.length )
		assert( jacobianB.device === y.device && count.rows * MemoryLayout<Float>.size <= y.length )
		assert( jacobianB.device === g.μ.device && count.rows * MemoryLayout<Float>.size <= g.μ.length )
		assert( jacobianB.device === g.σ.device && count.rows * MemoryLayout<Float>.size <= g.σ.length )
		assert( jacobianB.device === j.μ.device && count.rows * ref * MemoryLayout<Float>.size <= j.μ.length )
		assert( jacobianB.device === j.σ.device && count.rows * ref * MemoryLayout<Float>.size <= j.σ.length )
		
		let L: Int = 8
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianB)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(b.μ, offset: 0, at: 2)
		encoder.setBuffer(b.σ, offset: 0, at: 3)
		encoder.setBuffer(y, offset: 0, at: 4)
		encoder.setBuffer(g.μ, offset: 0, at: 5)
		encoder.setBuffer(g.σ, offset: 0, at: 6)
		encoder.setBuffer(j.μ, offset: 0, at: 7)
		encoder.setBuffer(j.σ, offset: 0, at: 8)
		encoder.setBytes([uint(count.rows), uint(ref), uint(count.rows), uint(L)], length: 4*MemoryLayout<uint>.size, at: 9)
		encoder.setThreadgroupMemoryLength(16*L*L*MemoryLayout<Float>.size, at: 0)
		encoder.setThreadgroupMemoryLength(16*L*L*MemoryLayout<Float>.size, at: 1)
		encoder.dispatchThreadgroups(.init(width: (count.rows-1)/L+1, height: (ref-1)/L+1, depth: 1),
		                             threadsPerThreadgroup: .init(width: L, height: L, depth: 1))
		encoder.endEncoding()
		
	}
	//jc
	public func jacobian(commandBuffer: MTLCommandBuffer,
	                     Σ: (μ: MTLBuffer, σ: MTLBuffer),
	                     c: (μ: MTLBuffer, σ: MTLBuffer), count: Int, rtrl: Bool = false) {
		
		assert( jacobianC.device === commandBuffer.device )
		
		let rows: Int = count
		let cols: Int = rtrl ? count : 1
		let ld: Int = rtrl ? count + 1 : 1
		
		assert( jacobianC.device === Σ.μ.device && rows * cols * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( jacobianC.device === Σ.σ.device && rows * cols * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( jacobianC.device === c.μ.device && rows * MemoryLayout<Float>.size <= c.μ.length )
		assert( jacobianC.device === c.σ.device && rows * MemoryLayout<Float>.size <= c.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = jacobianC.threadExecutionWidth
		encoder.setComputePipelineState(jacobianC)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(c.μ, offset: 0, at: 2)
		encoder.setBuffer(c.σ, offset: 0, at: 3)
		encoder.setBytes([uint(ld)], length: MemoryLayout<uint>.size, at: 4)
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(.init(width: (rows-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
	}
	//jd
	public func jacobian(commandBuffer: MTLCommandBuffer,
	                     Σ: (μ: MTLBuffer, σ: MTLBuffer),
	                     d: MTLBuffer,
	                     j: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int), rtrl: Bool = false) {
		
		let rows: Int = count.rows
		let cols: Int = count.cols * ( rtrl ? count.rows : 1 )
		
		assert( jacobianD.device === commandBuffer.device )
		
		assert( jacobianD.device === Σ.μ.device && rows * cols * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( jacobianD.device === Σ.σ.device && rows * cols * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( jacobianD.device === d.device && rows * MemoryLayout<Float>.size <= d.length )
		assert( jacobianD.device === j.μ.device && rows * cols * MemoryLayout<Float>.size <= j.μ.length )
		assert( jacobianD.device === j.σ.device && rows * cols * MemoryLayout<Float>.size <= j.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianD)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(d, offset: 0, at: 2)
		encoder.setBuffer(j.μ, offset: 0, at: 3)
		encoder.setBuffer(j.σ, offset: 0, at: 4)
		encoder.setBytes([uint(cols)], length: MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(.init(width: rows, height: cols, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	//jx
	public func jacobian(commandBuffer: MTLCommandBuffer,
	                     Σ: (μ: MTLBuffer, σ: MTLBuffer),
	                     x: MTLBuffer,
	                     a: (μ: MTLBuffer, σ: MTLBuffer),
	                     count: (rows: Int, cols: Int), rtrl: Bool = false) {
		
		let rows: Int = count.rows
		let cols: Int = count.cols * ( rtrl ? count.rows : 1 )
		
		assert( jacobianX.device === commandBuffer.device )
		
		assert( jacobianX.device === Σ.μ.device && rows * cols * MemoryLayout<Float>.size <= Σ.μ.length )
		assert( jacobianX.device === Σ.σ.device && rows * cols * MemoryLayout<Float>.size <= Σ.σ.length )
		assert( jacobianX.device === x.device && cols * MemoryLayout<Float>.size <= x.length )
		assert( jacobianX.device === a.μ.device && rows * cols * MemoryLayout<Float>.size <= a.μ.length )
		assert( jacobianX.device === a.σ.device && rows * cols * MemoryLayout<Float>.size <= a.σ.length )
		
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianX)
		encoder.setBuffer(Σ.μ, offset: 0, at: 0)
		encoder.setBuffer(Σ.σ, offset: 0, at: 1)
		encoder.setBuffer(x, offset: 0, at: 2)
		encoder.setBuffer(a.μ, offset: 0, at: 3)
		encoder.setBuffer(a.σ, offset: 0, at: 4)
		encoder.setBytes([uint(cols)], length: MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(.init(width: rows, height: cols, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
}
extension GaussDistributor: Distributor {
	public func clear(commandBuffer: MTLCommandBuffer, μ: MTLBuffer, σ: MTLBuffer) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: μ, range: NSRange(location: 0, length: μ.length), value: 0)
		encoder.fill(buffer: σ, range: NSRange(location: 0, length: σ.length), value: 0)
		encoder.endEncoding()
	}
}
extension GaussDistributor {
	/*
	public func delta(commandBuffer: MTLCommandBuffer,
	                  Δw: (μ: MTLBuffer, σ: MTLBuffer),
	                  w: (μ: MTLBuffer, σ: MTLBuffer),
	                  x: MTLBuffer,
	                  v: (μ: MTLBuffer, σ: MTLBuffer),
	                  Δ: (μ: MTLBuffer, σ: MTLBuffer),
	                  count: (rows: Int, cols: Int)) {
		
		assert( deltaW.device === commandBuffer.device )
		
		assert( deltaW.device === Δw.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Δw.μ.length )
		assert( deltaW.device === Δw.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= Δw.σ.length )
		
		assert( deltaW.device === w.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= w.μ.length )
		assert( deltaW.device === w.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= w.σ.length )
		
		assert( deltaW.device === x.device && count.cols * MemoryLayout<Float>.size <= x.length )
		
		assert( deltaW.device === v.μ.device && count.rows * MemoryLayout<Float>.size <= v.μ.length )
		assert( deltaW.device === v.σ.device && count.rows * MemoryLayout<Float>.size <= v.σ.length )
		
		assert( deltaW.device === Δ.μ.device && count.rows * MemoryLayout<Float>.size <= Δ.μ.length )
		assert( deltaW.device === Δ.σ.device && count.rows * MemoryLayout<Float>.size <= Δ.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(deltaW)
		encoder.setBuffer(Δw.μ, offset: 0, at: 0)
		encoder.setBuffer(Δw.σ, offset: 0, at: 1)
		encoder.setBuffer(w.μ, offset: 0, at: 2)
		encoder.setBuffer(w.σ, offset: 0, at: 3)
		encoder.setBuffer(x, offset: 0, at: 4)
		encoder.setBuffer(v.μ, offset: 0, at: 5)
		encoder.setBuffer(v.σ, offset: 0, at: 6)
		encoder.setBuffer(Δ.μ, offset: 0, at: 7)
		encoder.setBuffer(Δ.σ, offset: 0, at: 8)
		encoder.setBytes([uint(count.rows), uint(count.cols)], length: 2*MemoryLayout<uint>.size, at: 9)
		encoder.dispatchThreadgroups(.init(width: count.rows, height: count.cols, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func delta(commandBuffer: MTLCommandBuffer,
	                  Δc: (μ: MTLBuffer, σ: MTLBuffer),
	                  c: (μ: MTLBuffer, σ: MTLBuffer),
	                  v: (μ: MTLBuffer, σ: MTLBuffer),
	                  Δ: (μ: MTLBuffer, σ: MTLBuffer),
	                  count: Int) {
		
		assert( deltaC.device === commandBuffer.device )
		
		assert( deltaC.device === Δc.μ.device && count * MemoryLayout<Float>.size <= Δc.μ.length )
		assert( deltaC.device === Δc.σ.device && count * MemoryLayout<Float>.size <= Δc.σ.length )
		
		assert( deltaC.device === c.μ.device && count * MemoryLayout<Float>.size <= c.μ.length )
		assert( deltaC.device === c.σ.device && count * MemoryLayout<Float>.size <= c.σ.length )
		
		assert( deltaC.device === v.μ.device && count * MemoryLayout<Float>.size <= v.μ.length )
		assert( deltaC.device === v.σ.device && count * MemoryLayout<Float>.size <= v.σ.length )
		
		assert( deltaC.device === Δ.μ.device && count * MemoryLayout<Float>.size <= Δ.μ.length )
		assert( deltaC.device === Δ.σ.device && count * MemoryLayout<Float>.size <= Δ.σ.length )
		
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(deltaC)
		encoder.setBuffer(Δc.μ, offset: 0, at: 0)
		encoder.setBuffer(Δc.σ, offset: 0, at: 1)
		encoder.setBuffer(c.μ, offset: 0, at: 2)
		encoder.setBuffer(c.σ, offset: 0, at: 3)
		encoder.setBuffer(v.μ, offset: 0, at: 4)
		encoder.setBuffer(v.σ, offset: 0, at: 5)
		encoder.setBuffer(Δ.μ, offset: 0, at: 6)
		encoder.setBuffer(Δ.σ, offset: 0, at: 7)
		encoder.dispatchThreadgroups(.init(width: count, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func delta(commandBuffer: MTLCommandBuffer,
	                  Δx: MTLBuffer,
	                  x: MTLBuffer,
	                  w: (μ: MTLBuffer, σ: MTLBuffer),
	                  v: (μ: MTLBuffer, σ: MTLBuffer),
	                  Δ: (μ: MTLBuffer, σ: MTLBuffer),
	                  count: (rows: Int, cols: Int)) {
		
		assert( deltaW.device === commandBuffer.device )
		
		assert( deltaW.device === Δx.device && count.cols * MemoryLayout<Float>.size <= Δx.length )
		assert( deltaW.device ===  x.device && count.cols * MemoryLayout<Float>.size <=  x.length )
		
		assert( deltaW.device === w.μ.device && count.rows * count.cols * MemoryLayout<Float>.size <= w.μ.length )
		assert( deltaW.device === w.σ.device && count.rows * count.cols * MemoryLayout<Float>.size <= w.σ.length )
		
		assert( deltaW.device === v.μ.device && count.rows * MemoryLayout<Float>.size <= v.μ.length )
		assert( deltaW.device === v.σ.device && count.rows * MemoryLayout<Float>.size <= v.σ.length )
		
		assert( deltaW.device === Δ.μ.device && count.rows * MemoryLayout<Float>.size <= Δ.μ.length )
		assert( deltaW.device === Δ.σ.device && count.rows * MemoryLayout<Float>.size <= Δ.σ.length )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = collectW.threadExecutionWidth
		encoder.setComputePipelineState(deltaW)
		encoder.setBuffer(Δx, offset: 0, at: 0)
		encoder.setBuffer(x, offset: 0, at: 1)
		encoder.setBuffer(w.μ, offset: 0, at: 2)
		encoder.setBuffer(w.σ, offset: 0, at: 3)
		encoder.setBuffer(v.μ, offset: 0, at: 4)
		encoder.setBuffer(v.σ, offset: 0, at: 5)
		encoder.setBuffer(Δ.μ, offset: 0, at: 6)
		encoder.setBuffer(Δ.σ, offset: 0, at: 7)
		encoder.setBytes([uint(count.cols), uint(count.rows)], length: 2*MemoryLayout<uint>.size, at: 8)
		encoder.setThreadgroupMemoryLength(width*MemoryLayout<Float>.size, at: 0)
		encoder.dispatchThreadgroups(.init(width: (count.cols-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	*/
}
extension GaussDistributor {
}
extension GaussDistributor {
	
	
}

/*
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
	let jacobianMuA: MTLComputePipelineState
	let jacobianMuB: MTLComputePipelineState
	let jacobianMuC: MTLComputePipelineState
	let jacobianMuD: MTLComputePipelineState
	let jacobianMuX: MTLComputePipelineState
	let jacobianA: MTLComputePipelineState
	let jacobianB: MTLComputePipelineState
	let jacobianC: MTLComputePipelineState
	let jσA: MTLComputePipelineState
	let jσB: MTLComputePipelineState
	let jσC: MTLComputePipelineState
	let jσD: MTLComputePipelineState
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
	             collectD: MTLComputePipelineState,
	             jacobianMuA: MTLComputePipelineState,
	             jacobianMuB: MTLComputePipelineState,
	             jacobianMuC: MTLComputePipelineState,
	             jacobianMuD: MTLComputePipelineState,
	             jacobianMuX: MTLComputePipelineState,
	             jσA: MTLComputePipelineState,
	             jσB: MTLComputePipelineState,
	             jσC: MTLComputePipelineState,
	             jσD: MTLComputePipelineState,
	             jacobianA: MTLComputePipelineState,
	             jacobianB: MTLComputePipelineState,
	             jacobianC: MTLComputePipelineState) {
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
		self.jacobianMuA = jacobianMuA
		self.jacobianMuB = jacobianMuB
		self.jacobianMuC = jacobianMuC
		self.jacobianMuD = jacobianMuD
		self.jacobianMuX = jacobianMuX
		self.jσA = jσA
		self.jσB = jσB
		self.jσC = jσC
		self.jσD = jσD
		self.jacobianA = jacobianA
		self.jacobianB = jacobianB
		self.jacobianC = jacobianC
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
			let jacobianMuA: MTLComputePipelineState = try library.make(name: "GaussJacobianMuA")
			let jacobianMuB: MTLComputePipelineState = try library.make(name: "GaussJacobianMuB")
			let jacobianMuC: MTLComputePipelineState = try library.make(name: "GaussJacobianMuC")
			let jacobianMuD: MTLComputePipelineState = try library.make(name: "GaussJacobianMuD")
			let jacobianMuX: MTLComputePipelineState = try library.make(name: "GaussJacobianMuX")
			let jσA: MTLComputePipelineState = try library.make(name: "GaussJacobianSigmaA")
			let jσB: MTLComputePipelineState = try library.make(name: "GaussJacobianSigmaB")
			let jσC: MTLComputePipelineState = try library.make(name: "GaussJacobianSigmaC")
			let jσD: MTLComputePipelineState = try library.make(name: "GaussJacobianSigmaD")
			let jacobianA: MTLComputePipelineState = try library.make(name: "GaussJacobianA")
			let jacobianB: MTLComputePipelineState = try library.make(name: "GaussJacobianB")
			let jacobianC: MTLComputePipelineState = try library.make(name: "GaussJacobianC")
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
				      collectD: collectD16,
				      jacobianMuA: jacobianMuA,
				      jacobianMuB: jacobianMuB,
				      jacobianMuC: jacobianMuC,
				      jacobianMuD: jacobianMuD,
				      jacobianMuX: jacobianMuX,
				      jσA: jσA,
				      jσB: jσB,
				      jσC: jσC,
				      jσD: jσD,
				      jacobianA: jacobianA,
				      jacobianB: jacobianB,
				      jacobianC: jacobianC);
			}
		}
	}
}
extension Gauss: Stochastic {
	public func shuffle(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int) {
		
		assert( count * MemoryLayout<Float>.size <= χ.length && χ.device === rng16.device )
		assert( count * MemoryLayout<Float>.size <= μ.length && μ.device === rng16.device )
		assert( count * MemoryLayout<Float>.size <= σ.length && σ.device === rng16.device )
		
		do {
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let seeds: Array<UInt8> = Array<UInt8>(repeating: 0, count: 4096)
			arc4random_buf(UnsafeMutablePointer<UInt8>(mutating: seeds), seeds.count)
			encoder.setComputePipelineState(rng16)
			encoder.setBuffers([χ, μ, σ], offsets: [0, 0, 0], with: NSRange(0..<3))
			encoder.setBytes(seeds, length: seeds.count, at: 3)
			encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 4)
			encoder.dispatchThreadgroups(.init(width: 4, height: 1, depth: 1),
			                             threadsPerThreadgroup: .init(width: 64, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func errorValue(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, ψ: MTLBuffer) {
		
		assert( width * MemoryLayout<Float>.size <= Δ.length && Δ.device === errorValue16.device )
		assert( width * MemoryLayout<Float>.size <= ψ.length && ψ.device === errorValue16.device )
		
		do {
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(errorValue16)
			encoder.setBuffers([Δ, ψ, mean, variance], offsets: [0, 0, 0, 0], with: NSRange(0..<4))
			encoder.dispatchThreadgroups(.init(width: (width-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func errorState(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, ψ: MTLBuffer) {
		
		assert( width * MemoryLayout<Float>.size <= Δ.length && Δ.device === errorState16.device )
		assert( width * MemoryLayout<Float>.size <= ψ.length && ψ.device === errorState16.device )
		
		do {
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(errorState16)
			encoder.setBuffers([Δ, ψ, mean, variance], offsets: [0, 0, 0, 0], with: NSRange(0..<4))
			encoder.dispatchThreadgroups(.init(width: (width-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func deltaValue(commandBuffer: MTLCommandBuffer, Δμ: MTLBuffer, Δσ: MTLBuffer, Δ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) {
		
		assert( width * MemoryLayout<Float>.size <= Δμ.length && Δμ.device === deltaValue16.device )
		assert( width * MemoryLayout<Float>.size <= Δσ.length && Δσ.device === deltaValue16.device )
		assert( width * MemoryLayout<Float>.size <= Δ.length && Δ.device === deltaValue16.device )
		assert( width * MemoryLayout<Float>.size <= μ.length && μ.device === deltaValue16.device )
		assert( width * MemoryLayout<Float>.size <= σ.length && σ.device === deltaValue16.device )
		
		do {
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(deltaValue16)
			encoder.setBuffers([Δμ, Δσ, Δ, μ, σ], offsets: [0, 0, 0, 0, 0], with: NSRange(0..<5))
			encoder.dispatchThreadgroups(.init(width: (width-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func deltaState(commandBuffer: MTLCommandBuffer, Δμ: MTLBuffer, Δσ: MTLBuffer, Δ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) {
		
		assert( width * MemoryLayout<Float>.size <= Δμ.length && Δμ.device === deltaValue16.device )
		assert( width * MemoryLayout<Float>.size <= Δσ.length && Δσ.device === deltaValue16.device )
		assert( width * MemoryLayout<Float>.size <= Δ.length && Δ.device === deltaValue16.device )
		assert( width * MemoryLayout<Float>.size <= μ.length && μ.device === deltaValue16.device )
		assert( width * MemoryLayout<Float>.size <= σ.length && σ.device === deltaValue16.device )
		
		do {
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(deltaState16)
			encoder.setBuffers([Δμ, Δσ, Δ, μ, σ], offsets: [0, 0, 0, 0, 0], with: NSRange(0..<5))
			encoder.dispatchThreadgroups(.init(width: (width-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
}
extension Gauss: Derivative {
	public func jacobian(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), w: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, width: Int, refer: Int) {//jμA, jσA
		
		assert( commandBuffer.device === jacobianA.device )
		
		assert( width * width * refer * MemoryLayout<Float>.size <= j.μ.length && j.μ.device === jacobianA.device )
		assert( width * width * refer * MemoryLayout<Float>.size <= j.σ.length && j.σ.device === jacobianA.device )
		assert( width * refer * MemoryLayout<Float>.size <= w.μ.length && w.μ.device === jacobianA.device )
		assert( width * refer * MemoryLayout<Float>.size <= w.σ.length && w.σ.device === jacobianA.device )
		assert( refer * MemoryLayout<Float>.size <= x.length && x.device === jacobianA.device)

		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianA)
		encoder.setBuffers([j.μ, j.σ, w.μ, w.σ, x], offsets: [0, 0, 0, 0, 0], with: NSRange(0..<5))
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 5)
		encoder.dispatchThreadgroups(.init(width: width, height: refer, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()

	}
	public func jacobian(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), b: (μ: MTLBuffer, σ: MTLBuffer), d: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, width: Int, refer: Int) {//jμB, jσB
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), c: (μ: MTLBuffer, σ: MTLBuffer), width: Int) {//jμC, jσC
		
		assert( commandBuffer.device === jacobianC.device )
		
		assert( width * width * MemoryLayout<Float>.size <= j.μ.length && j.μ.device === jacobianC.device )
		assert( width * width * MemoryLayout<Float>.size <= j.σ.length && j.σ.device === jacobianC.device )
		assert( width * MemoryLayout<Float>.size <= c.μ.length && j.μ.device === jacobianC.device )
		assert( width * MemoryLayout<Float>.size <= c.σ.length && j.σ.device === jacobianC.device )
		
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianC)
		encoder.setBuffers([j.μ, j.σ, c.μ, c.σ], offsets: [0, 0, 0, 0], with: NSRange(0..<4))
		encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 4)
		encoder.dispatchThreadgroups(.init(width: width, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, x: MTLBuffer, refer: Int) {//ua
		
		assert( commandBuffer.device === jacobianMuA.device )
		assert( width * width * refer * MemoryLayout<Float>.size <= jμ.length && jμ.device === jacobianMuA.device)
		assert( refer * MemoryLayout<Float>.size <= x.length && x.device === jacobianMuA.device)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianMuA)
		encoder.setBuffers([jμ, x], offsets: [0, 0], with: NSRange(0..<2))
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 2)
		encoder.dispatchThreadgroups(.init(width: width, height: refer, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, b: MTLBuffer, j: MTLBuffer, p: MTLBuffer, refer: Int) {//ub
		
		assert( commandBuffer.device === jacobianMuB.device )
		
		assert( width * width * refer * MemoryLayout<Float>.size <= jμ.length && jμ.device === jacobianMuB.device)
		assert( width * width * MemoryLayout<Float>.size <= b.length && b.device === jacobianMuB.device)
		assert( width * MemoryLayout<Float>.size <= j.length && j.device === jacobianMuB.device)
		assert( width * width * refer * MemoryLayout<Float>.size <= p.length && p.device === jacobianMuB.device)
		
		let L: Int = 8
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianMuB)
		encoder.setBuffers([jμ, b, j, p], offsets: [0, 0, 0, 0], with: NSRange(0..<4))
		encoder.setBytes([uint(width), uint(width*refer), uint(width), uint(L)],
		                 length: 4 * MemoryLayout<uint>.size, at: 4)
		encoder.setThreadgroupMemoryLength(16 * L * L * MemoryLayout<Float>.size, at: 0)
		encoder.setThreadgroupMemoryLength(16 * L * L * MemoryLayout<Float>.size, at: 1)
		encoder.dispatchThreadgroups(.init(width: (width-1)/L+1, height: (width*refer-1)/L+1, depth: 1),
		                             threadsPerThreadgroup: .init(width: L, height: L, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, c: MTLBuffer) {//uc
		
		assert( commandBuffer.device === jacobianMuC.device )
		assert( width * MemoryLayout<Float>.size <= jμ.length && jμ.device === jacobianMuC.device )
		assert( width * MemoryLayout<Float>.size <= c.length && c.device === jacobianMuC.device )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianMuC)
		encoder.setBuffers([jμ, c], offsets: [0, 0], with: NSRange(0..<2))
		encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 2)
		encoder.dispatchThreadgroups(.init(width: width, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, w: MTLBuffer, refer: Int) {//ux
		
		assert( commandBuffer.device === jacobianMuX.device )
		assert( width * MemoryLayout<Float>.size <= jμ.length && jμ.device === jacobianMuX.device)
		assert( width * MemoryLayout<Float>.size <= w.length && w.device === jacobianMuX.device)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianMuX)
		encoder.setBuffers([jμ, w], offsets: [0, 0], with: NSRange(0..<2))
		encoder.setBytes([uint(width), uint(refer)], length: 2*MemoryLayout<uint>.size, at: 2)
		encoder.dispatchThreadgroups(.init(width: width, height: refer, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, d: MTLBuffer, p: MTLBuffer, refer: Int) {//ud
		
		assert( commandBuffer.device === jacobianMuD.device )
		assert( width * width * refer * MemoryLayout<Float>.size <= jμ.length && jμ.device === jacobianMuD.device )
		assert( width * MemoryLayout<Float>.size <= d.length && d.device === jacobianMuD.device )
		assert( width * width * refer * MemoryLayout<Float>.size <= p.length && p.device === jacobianMuD.device )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jacobianMuD)
		encoder.setBuffers([jμ, d, p], offsets: [0, 0, 0], with: NSRange(0..<3))
		encoder.setBytes([uint(width), uint(width*refer)], length: 2*MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(.init(width: width, height: width*refer, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, jσ: MTLBuffer, w: MTLBuffer, x: MTLBuffer, refer: Int) {//σa
		
		assert( commandBuffer.device === jσA.device )
		assert( width * width * refer * MemoryLayout<Float>.size <= jσ.length && jσ.device === jσA.device )
		assert( width * MemoryLayout<Float>.size <= variance.length && variance.device === jσA.device )
		assert( refer * MemoryLayout<Float>.size <= x.length && x.device === jσA.device )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jσA)
		encoder.setBuffers([jσ, variance, w, x], offsets: [0, 0, 0, 0], with: NSRange(0..<4))
		encoder.dispatchThreadgroups(.init(width: width, height: refer, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, jσ: MTLBuffer, b: MTLBuffer, j: MTLBuffer, p: MTLBuffer, refer: Int) {//σb
		
		assert( commandBuffer.device === jσB.device )
		assert( width * width * refer * MemoryLayout<Float>.size <= jσ.length && jσ.device === jσB.device)
		assert( width * width * MemoryLayout<Float>.size <= b.length && b.device === jσB.device)
		assert( width * MemoryLayout<Float>.size <= j.length && j.device === jσB.device)
		assert( width * width * refer * MemoryLayout<Float>.size <= p.length && p.device === jσB.device)
		
		let L: Int = 8
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jσB)
		encoder.setBuffers([jσ, b, j, p], offsets: [0, 0, 0, 0], with: NSRange(0..<4))
		encoder.setBytes([uint(width), uint(width*refer), uint(width), uint(L)],
		                 length: 4 * MemoryLayout<uint>.size, at: 4)
		encoder.setThreadgroupMemoryLength(16 * L * L * MemoryLayout<Float>.size, at: 0)
		encoder.setThreadgroupMemoryLength(16 * L * L * MemoryLayout<Float>.size, at: 1)
		encoder.dispatchThreadgroups(.init(width: (width-1)/L+1, height: (width*refer-1)/L+1, depth: 1),
		                             threadsPerThreadgroup: .init(width: L, height: L, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, jσ: MTLBuffer, c: MTLBuffer) {//σc
		
		assert( commandBuffer.device === jacobianMuC.device )
		assert( width * MemoryLayout<Float>.size <= jσ.length && jσ.device === jacobianMuC.device)
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jσC)
		encoder.setBuffers([jσ, variance, c], offsets: [0, 0, 0], with: NSRange(0..<3))
		encoder.setBytes([uint(width)], length: MemoryLayout<uint>.size, at: 3)
		encoder.dispatchThreadgroups(.init(width: width, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func jacobian(commandBuffer: MTLCommandBuffer, jσ: MTLBuffer, d: MTLBuffer, p: MTLBuffer, refer: Int) {//σd
		
		assert( commandBuffer.device === jσD.device )
		assert( width * width * refer * MemoryLayout<Float>.size <= jσ.length && jσ.device === jσD.device )
		assert( width * MemoryLayout<Float>.size <= d.length && d.device === jσD.device )
		assert( width * width * refer * MemoryLayout<Float>.size <= p.length && p.device === jσD.device )
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(jσD)
		encoder.setBuffers([jσ, variance, d, p], offsets: [0, 0, 0, 0], with: NSRange(0..<4))
		encoder.setBytes([uint(width), uint(width*refer)], length: 2*MemoryLayout<uint>.size, at: 4)
		encoder.dispatchThreadgroups(.init(width: width, height: width*refer, depth: 1),
		                             threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
		
	}
	public func clear(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer)) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: j.μ, range: NSRange(0..<j.μ.length), value: 0)
		encoder.fill(buffer: j.σ, range: NSRange(0..<j.σ.length), value: 0)
		encoder.endEncoding()
	}
}
extension Gauss: Synthesis {
	public func synthesize(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) {
		
		assert( commandBuffer.device === synthesize16.device )
		
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
	public func collect(commandBuffer: MTLCommandBuffer, w: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, refer: Int) {
		
		assert( commandBuffer.device === collectW16.device )
		
		assert( width * refer * MemoryLayout<Float>.size <= w.χ.length && collectW16.device === w.χ.device )
		assert( width * refer * MemoryLayout<Float>.size <= w.μ.length && collectW16.device === w.μ.device )
		assert( width * refer * MemoryLayout<Float>.size <= w.σ.length && collectW16.device === w.σ.device )
		
		assert( refer * MemoryLayout<Float>.size <= x.length && collectW16.device === x.device )
		
		do {
			let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			let length: Int = MemoryLayout<Float>.size * 16 * 64
			encoder.setComputePipelineState(collectW16)
			encoder.setBuffers([value, mean, variance, w.χ, w.μ, w.σ, x], offsets: [0, 0, 0, 0, 0, 0, 0], with: NSRange(0..<7))
			encoder.setBytes([uint(refer-1)/16+1], length: MemoryLayout<uint>.size, at: 7)
			encoder.setThreadgroupMemoryLength(length, at: 0)
			encoder.setThreadgroupMemoryLength(length, at: 1)
			encoder.setThreadgroupMemoryLength(length, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (width-1)/16+1, height: 1, depth: 1),
			                             threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
			encoder.endEncoding()
		}
	}
	public func collect(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) {
		
		assert( commandBuffer.device === collectC16.device )
		
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
		
		assert( commandBuffer.device === collectD16.device )
		
		assert( width * MemoryLayout<Float>.size <= r.length && collectD16.device === r.device )
		
		assert( width * MemoryLayout<Float>.size <= x.χ.length && collectD16.device === x.χ.device )
		assert( width * MemoryLayout<Float>.size <= x.μ.length && collectD16.device === x.μ.device )
		assert( width * MemoryLayout<Float>.size <= x.σ.length && collectD16.device === x.σ.device )
		
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
		
		assert( commandBuffer.device === value.device )
		assert( commandBuffer.device === mean.device )
		assert( commandBuffer.device === variance.device )
		
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
*/
