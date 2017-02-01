//
//  NonLinear.swift
//  macOS
//
//  Created by Kota Nakano on 2017/01/28.
//
//

import Accelerate
import Metal

extension Computer {
	private func invoke(y: Buffer, x: Buffer, count: Int, pipelineState: ComputePipelineState, vForce: @escaping(UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void) {
		assert(MemoryLayout<Float>.size*count<=y.length && y.device === queue.device)
		assert(MemoryLayout<Float>.size*count<=x.length && x.device === queue.device)
		let commandBuffer: CommandBuffer = queue.makeCommandBuffer()
		if true {
			let encoder: ComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
			encoder.setComputePipelineState(sigm)
			encoder.setBuffer(y, offset: 0, at: 0)
			encoder.setBuffer(x, offset: 0, at: 1)
			encoder.setBytes([uint(count-1)/16+1], length: MemoryLayout<uint>.size, at: 2)
			encoder.dispatchThreadgroups(MTLSize(width: (count-1)/(threads*64)+1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
			encoder.endEncoding()
		}
		/*
		if 0 < rem {
			let yr: UnsafeMutablePointer<Float> = y.pointer.advanced(by: quo*threads)
			let xr: UnsafeMutablePointer<Float> = x.pointer.advanced(by: quo*threads)
			commandBuffer.addCompletedHandler { (_: CommandBuffer) in
				var count: Int32 = Int32(rem)
				vForce(yr, xr, &count)
			}
		}
		*/
		commandBuffer.commit()
	}
	public func log(y: Buffer, x: Buffer, count: Int) {
		invoke(y: y, x: x, count: count, pipelineState: log, vForce: vvlogf)
	}
	public func exp(y: Buffer, x: Buffer, count: Int) {
		invoke(y: y, x: x, count: count, pipelineState: exp, vForce: vvexpf)
	}
	public func tanh(y: Buffer, x: Buffer, count: Int) {
	
	}
	public func sigm(y: Buffer, x: Buffer, count: Int) {
		invoke(y: y, x: x, count: count, pipelineState: sigm) {
			vDSP_vneg($0.1, 1, $0.0, 1, vDSP_Length($0.2.pointee))
			vvexpf($0.0, $0.0, $0.2)
			vDSP_vsadd($0.0, 1, [Float(1)], $0.0, 1, vDSP_Length($0.2.pointee))
			vvrecf($0.0, $0.0, $0.2)
		}
	}
}
