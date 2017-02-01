//
//  OptimizerTests.swift
//  OptimizerTests
//
//  Created by Kota Nakano on 2017/01/25.
//
//

import Accelerate
import Metal
import MetalKit
import XCTest
@testable import Optimizer

class OptimizerTests: XCTestCase {
	
	let count: Int = 8
	
	func uniform(count: Int, range: (α: Float, β: Float) = (α: 0, β: 1)) -> Array<Float> {
		let seed: Array<UInt16> = Array<UInt16>(repeating: 0, count: count)
		let buff: Array<Float> = Array<Float>(repeating: 0, count: count)
		arc4random_buf(UnsafeMutablePointer<UInt16>(mutating: seed), seed.count*MemoryLayout<UInt16>.size)
		vDSP_vfltu16(seed, 1, UnsafeMutablePointer<Float>(mutating: buff), 1, vDSP_Length(count))
		vDSP_vsmul(buff, 1, [(range.β-range.α)/65536.0], UnsafeMutablePointer<Float>(mutating: buff), 1, vDSP_Length(count))
		vDSP_vsadd(buff, 1, [range.α], UnsafeMutablePointer<Float>(mutating: buff), 1, vDSP_Length(count))
		return buff
	}
	func rmse(x: Array<Float>, y: Array<Float>) -> Float {
		var rms: Float = 0
		let val: Array<Float> = Array<Float>(repeating: 0, count: min(x.count, y.count))
		vDSP_vsub(x, 1, y, 1, UnsafeMutablePointer<Float>(mutating: val), 1, vDSP_Length(val.count))
		vDSP_rmsqv(val, 1, &rms, vDSP_Length(val.count))
		return rms
	}
	func uniform(x: MTLBuffer, range: (α: Float, β: Float) = (α: 0, β: 1)) {
		let count: Int = x.length / MemoryLayout<Float>.size
		let seed: Array<UInt16> = Array<UInt16>(repeating: 0, count: count)
		arc4random_buf(UnsafeMutablePointer<UInt16>(mutating: seed), seed.count*MemoryLayout<UInt16>.size)
		vDSP_vfltu16(seed, 1, x.floatPointer, 1, vDSP_Length(count))
		vDSP_vsmul(x.floatPointer, 1, [(range.β-range.α)/65536.0], x.floatPointer, 1, vDSP_Length(count))
		vDSP_vsadd(x.floatPointer, 1, [range.α], x.floatPointer, 1, vDSP_Length(count))
	}
	func prepare(device: MTLDevice, name: String) throws -> MTLComputePipelineState {
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: Bundle(for: type(of: self)))
		let function: MTLFunction = try library.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
		return try device.makeComputePipelineState(function: function)
	}
	func apply(commandBuffer: MTLCommandBuffer, pipeline: MTLComputePipelineState, dydx: MTLBuffer, x: MTLBuffer) {
		let encoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
		encoder.setComputePipelineState(pipeline)
		encoder.setBuffer(dydx, offset: 0, at: 0)
		encoder.setBuffer(x, offset: 0, at: 1)
		encoder.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encoder.endEncoding()
	}
	func optimizerTests(device: MTLDevice, optimizer: Optimizer) {
		do {
			let gradient: MTLComputePipelineState = try prepare(device: device, name: "dydx")
			let θ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let Δθ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			//uniform(x: θ)
			(0..<16384).forEach { (_) in
				do {
					let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
					apply(commandBuffer: commandBuffer, pipeline: gradient, dydx: Δθ, x: θ)
					commandBuffer.commit()
				}
				do {
					let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
					optimizer.encode(commandBuffer: commandBuffer, θ: θ, Δθ: Δθ)
					commandBuffer.commit()
				}
			}
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			print(θ.floatArray)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testMomentumAdaDelta() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let optimizer: Optimizer = try MomentumAdaDelta(device: device, count: count)
			optimizerTests(device: device, optimizer: optimizer)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testAdaDelta() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let optimizer: Optimizer = try AdaDelta(device: device, count: count)
			optimizerTests(device: device, optimizer: optimizer)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testMomentum() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let optimizer: Optimizer = try Momentum(device: device, count: count)
			optimizerTests(device: device, optimizer: optimizer)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testStochasticGradientDescent() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let optimizer: Optimizer = StochasticGradientDescent(device: device, η: 1e-4)
		optimizerTests(device: device, optimizer: optimizer)
	}
}
extension MTLBuffer {
	var floatPointer: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
	var floatBuffer: UnsafeMutableBufferPointer<Float> {
		return UnsafeMutableBufferPointer<Float>(start: UnsafeMutablePointer<Float>(OpaquePointer(contents())), count: length/MemoryLayout<Float>.size)
	}
	var floatArray: Array<Float> {
		return Array<Float>(UnsafeMutableBufferPointer<Float>(start: UnsafeMutablePointer<Float>(OpaquePointer(contents())), count: length/MemoryLayout<Float>.size))
	}
	var mse: Float {
		var mse: Float = 0
		vDSP_rmsqv(UnsafePointer<Float>(OpaquePointer(contents())), 1, &mse, vDSP_Length(length/MemoryLayout<Float>.size))
		return mse
	}
}
