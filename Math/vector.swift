//
//  vector.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/07.
//
//

import Metal
extension Math {
	private func invoke(commandBuffer: MTLCommandBuffer, pipeline: MTLComputePipelineState, buffers: Array<MTLBuffer?>, count: Int) {
		
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		let width: Int = pipeline.threadExecutionWidth
		
		assert( pipeline.device === encoder.device )
		buffers.forEach {
			if let buffer: MTLBuffer = $0 {
				assert( pipeline.device === buffer.device && count * MemoryLayout<Float>.size <= buffer.length )
			} else {
				assertionFailure()
			}
		}
		encoder.setComputePipelineState(pipeline)
		encoder.setBuffers(buffers, offsets: Array<Int>(repeating: 0, count: buffers.count), with: NSRange(location: 0, length: buffers.count))
		encoder.setBytes([uint(count)], length: MemoryLayout<uint>.size, at: buffers.count)
		encoder.dispatchThreadgroups(.init(width: (count-1)/width+1, height: 1, depth: 1),
		                             threadsPerThreadgroup: .init(width: width, height: 1, depth: 1))
		encoder.endEncoding()
	}
	public func add(commandBuffer: MTLCommandBuffer, y: MTLBuffer, a: MTLBuffer, b: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: add, buffers: [y, a, b], count: count)
	}
	public func sub(commandBuffer: MTLCommandBuffer, y: MTLBuffer, a: MTLBuffer, b: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: sub, buffers: [y, a, b], count: count)
	}
	public func mul(commandBuffer: MTLCommandBuffer, y: MTLBuffer, a: MTLBuffer, b: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: mul, buffers: [y, a, b], count: count)
	}
	public func div(commandBuffer: MTLCommandBuffer, y: MTLBuffer, a: MTLBuffer, b: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: div, buffers: [y, a, b], count: count)
	}
	public func fma(commandBuffer: MTLCommandBuffer, y: MTLBuffer, a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: fma, buffers: [y, a, b, c], count: count)
	}
	public func exp(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: exp, buffers: [y, x], count: count)
	}
	public func log(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: log, buffers: [y, x], count: count)
	}
	public func cos(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: cos, buffers: [y, x], count: count)
	}
	public func sin(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: sin, buffers: [y, x], count: count)
	}
	public func tan(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: tan, buffers: [y, x], count: count)
	}
	public func abs(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: abs, buffers: [y, x], count: count)
	}
	public func tanh(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: tanh, buffers: [y, x], count: count)
	}
	public func sign(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: sign, buffers: [y, x], count: count)
	}
	public func sigm(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: sigm, buffers: [y, x], count: count)
	}
	public func relu(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: relu, buffers: [y, x], count: count)
	}
	public func soft(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: soft, buffers: [y, x], count: count)
	}
	public func regu(commandBuffer: MTLCommandBuffer, y: MTLBuffer, x: MTLBuffer, count: Int) {
		invoke(commandBuffer: commandBuffer, pipeline: regu, buffers: [y, x], count: count)
	}
}
