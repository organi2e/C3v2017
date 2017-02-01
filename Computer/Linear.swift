//
//  Linear.swift
//  macOS
//
//  Created by Kota Nakano on 2017/01/28.
//
//

import Metal

extension Computer {
	private func invoke(z: Buffer, y: Buffer, x: Buffer, count: Int, pipelineState: ComputePipelineState) {
		assert(MemoryLayout<Float>.size*count<=z.length && z.device === queue.device)
		assert(MemoryLayout<Float>.size*count<=y.length && y.device === queue.device)
		assert(MemoryLayout<Float>.size*count<=x.length && x.device === queue.device)
		let commandBuffer: CommandBuffer = queue.makeCommandBuffer()
		if true {
			let encoder: ComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(pipelineState)
			encoder.setBuffer(z, offset: 0, at: 0)
			encoder.setBuffer(y, offset: 0, at: 1)
			encoder.setBuffer(x, offset: 0, at: 2)
			encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 3)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/(threads*64)+1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		commandBuffer.commit()
	}
	public func add(z: Buffer, y: Buffer, x: Buffer, count: Int) {
		invoke(z: z, y: y, x: x, count: count, pipelineState: add)
	}
	public func sub(z: Buffer, y: Buffer, x: Buffer, count: Int) {
		invoke(z: z, y: y, x: x, count: count, pipelineState: sub)
	}
	public func mul(z: Buffer, y: Buffer, x: Buffer, count: Int) {
		invoke(z: z, y: y, x: x, count: count, pipelineState: mul)
	}
	public func div(z: Buffer, y: Buffer, x: Buffer, count: Int) {
		invoke(z: z, y: y, x: x, count: count, pipelineState: div)
	}
	public func gemv(y: Buffer, w: Buffer, x: Buffer, rows: Int, cols: Int) {
		assert(MemoryLayout<Float>.size*rows<=y.length)
		assert(MemoryLayout<Float>.size*rows*cols<=w.length)
		assert(MemoryLayout<Float>.size*cols<=x.length)
		
		let commandBuffer: CommandBuffer = queue.makeCommandBuffer()
		let encoder: ComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(gemv)
		encoder.setBuffer(y, offset: 0, at: 0)
		encoder.setBuffer(w, offset: 0, at: 1)
		encoder.setBuffer(x, offset: 0, at: 2)
		encoder.setBytes([uint(cols-1)/16+1], length: MemoryLayout<uint>.size, at: 3)
		encoder.setThreadgroupMemoryLength(MemoryLayout<Float>.size*16*64, at: 0)
		encoder.dispatchThreadgroups(MTLSize(width: (rows-1)/16+1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
		encoder.endEncoding()
		commandBuffer.commit()
		
	}
}
