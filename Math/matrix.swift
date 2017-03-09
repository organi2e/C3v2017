//
//  matrix.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/07.
//
//

import Metal

extension Math {
	public func gemv(commandBuffer: MTLCommandBuffer, y: MTLBuffer, w: MTLBuffer, x: MTLBuffer, transpose: Bool, α: Float, β: Float, rows: Int, cols: Int) {
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let pipeline: MTLComputePipelineState = transpose ? gemvr : gemvc
		let width: Int = pipeline.threadExecutionWidth
		
		assert( pipeline.device === encoder.device )
		assert( pipeline.device === y.device && rows * MemoryLayout<Float>.size <= y.length )
		assert( pipeline.device === w.device && rows * cols * MemoryLayout<Float>.size <= w.length )
		assert( pipeline.device === x.device && cols * MemoryLayout<Float>.size <= x.length )
		
		encoder.setComputePipelineState(pipeline)
		encoder.setBuffer(y, offset: 0, at: 0)
		encoder.setBuffer(w, offset: 0, at: 1)
		encoder.setBuffer(x, offset: 0, at: 2)
		encoder.setBytes([α, β], length: 2*MemoryLayout<Float>.size, at: 3)
		encoder.setBytes([uint(rows), uint(cols)], length: 2*MemoryLayout<uint>.size, at: 4)
		encoder.setThreadgroupMemoryLength(width*4*MemoryLayout<Float>.size, at: 0)
		encoder.dispatchThreadgroups(.init(width: (rows-1)/4+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func gemm(commandBuffer: MTLCommandBuffer, y: MTLBuffer, w: MTLBuffer, x: MTLBuffer, transpose: (Bool, Bool), α: Float, β: Float, rows: Int, cols: Int) {
		
	}
	public func inner(commandBuffer: MTLCommandBuffer, y: MTLBuffer, w: MTLBuffer, x: MTLBuffer, transpose: (Bool, Bool), α: Float, β: Float, rows: Int, cols: Int) {
		
	}
	public func outer(commandBuffer: MTLCommandBuffer, y: MTLBuffer, w: MTLBuffer, x: MTLBuffer, transpose: (Bool, Bool), α: Float, β: Float, rows: Int, cols: Int) {
		
	}
}
