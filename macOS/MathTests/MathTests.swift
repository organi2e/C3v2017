//
//  MathTests.swift
//  MathTests
//
//  Created by Kota Nakano on 2017/03/07.
//
//
import Accelerate
import Metal

import XCTest
@testable import Math

class MathTests: XCTestCase {
	func testFourArithmeticOperations() {
		let length: Int = uniform(100, 600)
		let count: Int = uniform(100, length)
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let a: MTLBuffer = device.makeBuffer(bytes: uniform(length, 0.0, 1.0), length: length * MemoryLayout<Float>.size, options: .storageModeShared)
		let b: MTLBuffer = device.makeBuffer(bytes: uniform(length, 0.0, 1.0), length: length * MemoryLayout<Float>.size, options: .storageModeShared)
		let c: MTLBuffer = device.makeBuffer(bytes: uniform(length, 0.0, 1.0), length: length * MemoryLayout<Float>.size, options: .storageModeShared)
		let y: MTLBuffer = device.makeBuffer(length: length * MemoryLayout<Float>.size, options: .storageModeShared)
		do {
			let math: Math = try Math(device: device)
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				math.add(commandBuffer: commandBuffer, y: y, a: a, b: b, count: count)
				commandBuffer.commit()
				
				let d: la_object_t = la_sum(a.matrix(rows: count, cols: 1), b.matrix(rows: count, cols: 1))
				XCTAssert(!d.hasErr)
				let e: la_object_t = la_difference(y.matrix(rows: count, cols: 1), d)
				XCTAssert(!e.hasErr)
				
				commandBuffer.waitUntilCompleted()
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(rmse<1e-24)
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				math.mul(commandBuffer: commandBuffer, y: y, a: a, b: b, count: count)
				commandBuffer.commit()
				
				let d: la_object_t = la_elementwise_product(a.matrix(rows: count, cols: 1), b.matrix(rows: count, cols: 1))
				XCTAssert(!d.hasErr)
				let e: la_object_t = la_difference(y.matrix(rows: count, cols: 1), d)
				XCTAssert(!e.hasErr)
				
				commandBuffer.waitUntilCompleted()
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(rmse<1e-24)
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				math.fma(commandBuffer: commandBuffer, y: y, a: a, b: b, c: c, count: count)
				commandBuffer.commit()
				
				let d: la_object_t = la_sum(la_elementwise_product(a.matrix(rows: count, cols: 1), b.matrix(rows: count, cols: 1)), c.matrix(rows: count, cols: 1))
				XCTAssert(!d.hasErr)
				let e: la_object_t = la_difference(y.matrix(rows: count, cols: 1), d)
				XCTAssert(!e.hasErr)
				
				commandBuffer.waitUntilCompleted()
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(rmse<1e-5)
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				math.fma(commandBuffer: commandBuffer, y: y, a: a, b: b, c: c, count: count)
				math.sub(commandBuffer: commandBuffer, y: y, a: y, b: c, count: count)
				math.div(commandBuffer: commandBuffer, y: y, a: y, b: b, count: count)
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
				
				let d: la_object_t = a.matrix(rows: count, cols: 1)
				XCTAssert(!d.hasErr)
				let e: la_object_t = la_difference(y.matrix(rows: count, cols: 1), d)
				XCTAssert(!e.hasErr)
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(rmse<1e-4)
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testNonlinear() {
		let length: Int = 1024
		let count: Int = uniform(128, length)
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let x: MTLBuffer = device.makeBuffer(bytes: uniform(length, 0.0, 1.0), length: length * MemoryLayout<Float>.size, options: .storageModeShared)
		let y: MTLBuffer = device.makeBuffer(length: length * MemoryLayout<Float>.size, options: .storageModeShared)
		let d: MTLBuffer = device.makeBuffer(length: length * MemoryLayout<Float>.size, options: .storageModeShared)
		do {
			let math: Math = try Math(device: device)
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				math.exp(commandBuffer: commandBuffer, y: y, x: x, count: count)
				commandBuffer.commit()
				vvexpf(d.ref, x.ref, [Int32(count)])
				commandBuffer.waitUntilCompleted()
				
				let e: la_object_t = la_difference(y.matrix(rows: count, cols: 1), d.matrix(rows: count, cols: 1))
				XCTAssert(!e.hasErr)
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(rmse<1e-5)
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				math.log(commandBuffer: commandBuffer, y: y, x: x, count: count)
				commandBuffer.commit()
				vvlogf(d.ref, x.ref, [Int32(count)])
				commandBuffer.waitUntilCompleted()
				
				let e: la_object_t = la_difference(y.matrix(rows: count, cols: 1), d.matrix(rows: count, cols: 1))
				XCTAssert(!e.hasErr)
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(rmse<1e-3)
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				math.sigm(commandBuffer: commandBuffer, y: y, x: x, count: count)
				commandBuffer.commit()
				
				vDSP_vneg(x.ref, 1, d.ref, 1, vDSP_Length(count))
				vvexpf(d.ref, d.ref, [Int32(count)])
				vDSP_vsadd(d.ref, 1, [1], d.ref, 1, vDSP_Length(count))
				vvrecf(d.ref, d.ref, [Int32(count)])
				commandBuffer.waitUntilCompleted()
				
				let e: la_object_t = la_difference(y.matrix(rows: count, cols: 1), d.matrix(rows: count, cols: 1))
				XCTAssert(!e.hasErr)
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(!rmse.isNaN && rmse<1e-3)
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				math.regu(commandBuffer: commandBuffer, y: y, x: x, count: count)
				commandBuffer.commit()
				
				vvfabsf(d.ref, x.ref, [Int32(count)])
				vDSP_vsadd(d.ref, 1, [1], d.ref, 1, vDSP_Length(count))
				vvlogf(d.ref, d.ref, [Int32(count)])
				commandBuffer.waitUntilCompleted()
				
				let e: la_object_t = la_difference(y.matrix(rows: count, cols: 1), d.matrix(rows: count, cols: 1))
				XCTAssert(!e.hasErr)
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(!rmse.isNaN && rmse<1e-3)
			
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testGEMV() {
		let cols: Int = 1024//uniform(128, 512)
		let rows: Int = 1024//uniform(128, 512)
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let x: MTLBuffer = device.makeBuffer(bytes: uniform(cols, 0.0, 1.0), length: cols*MemoryLayout<Float>.size, options: .storageModeShared)
		let w: MTLBuffer = device.makeBuffer(bytes: uniform(rows*cols, 0.0, 1.0), length: rows*cols*MemoryLayout<Float>.size, options: .storageModeShared)
		let y: MTLBuffer = device.makeBuffer(length: rows*MemoryLayout<Float>.size, options: .storageModeShared)
		do {
			let math: Math = try Math(device: device)
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				let beg: Date = Date()
				for _ in 0..<1024 {
					math.gemv(commandBuffer: commandBuffer, y: y, w: w, x: x, transpose: false, α: 1.0, β: 0.0, rows: rows, cols: cols)
				}
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
				let end: Date = Date()
				print("elapsed: ", end.timeIntervalSince(beg))
				
				let d: la_object_t = la_matrix_product(w.matrix(rows: rows, cols: cols), x.matrix(rows: cols, cols: 1))
				let e: la_object_t = la_difference(d, y.matrix(rows: rows, cols: 1))
				XCTAssert(!e.hasErr)
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(rmse<1e-2)
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				let beg: Date = Date()
				for _ in 0..<1024 {
					math.gemv(commandBuffer: commandBuffer, y: y, w: w, x: x, transpose: true, α: 1.0, β: 0.0, rows: rows, cols: cols)
				}
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
				let end: Date = Date()
				print("elapsed: ", end.timeIntervalSince(beg))
				
				let d: la_object_t = la_matrix_product(la_transpose(w.matrix(rows: cols, cols: rows)), x.matrix(rows: cols, cols: 1))
				let e: la_object_t = la_difference(d, y.matrix(rows: rows, cols: 1))
				XCTAssert(!e.hasErr)
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(rmse<1e-2)
			}
			do {
				let beg: Date = Date()
				for _ in 0..<1024 {
					cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(rows), Int32(cols), 1.0, w.ref, Int32(cols), x.ref, 1, 0.0, y.ref, 1)
				}
				let end: Date = Date()
				print("elapsed: ", end.timeIntervalSince(beg))
				
				let d: la_object_t = la_matrix_product(w.matrix(rows: cols, cols: rows), x.matrix(rows: cols, cols: 1))
				let e: la_object_t = la_difference(d, y.matrix(rows: rows, cols: 1))
				XCTAssert(!e.hasErr)
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(rmse<1e-3)
				print(rmse)
			}
			
			do {
				let beg: Date = Date()
				for _ in 0..<1024 {
					cblas_sgemv(CblasRowMajor, CblasTrans, Int32(rows), Int32(cols), 1.0, w.ref, Int32(cols), x.ref, 1, 0.0, y.ref, 1)
				}
				let end: Date = Date()
				print("elapsed: ", end.timeIntervalSince(beg))
				
				let d: la_object_t = la_matrix_product(la_transpose(w.matrix(rows: cols, cols: rows)), x.matrix(rows: cols, cols: 1))
				let e: la_object_t = la_difference(d, y.matrix(rows: rows, cols: 1))
				XCTAssert(!e.hasErr)
				
				let rmse: Float = la_norm_as_float(e, norm)
				XCTAssert(rmse<1e-2)
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
extension MathTests {
	func uniform(_ count: Int, _ α: Float, _ β: Float) -> Array<Float> {
		var result: Array<Float> = Array<Float>(repeating: 0, count: count)
		assert(α<β)
		arc4random_buf(&result, result.count*MemoryLayout<Float>.size)
		vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(result)), 1, &result, 1, vDSP_Length(count))
		vDSP_vsdiv(result, 1, [Float(UInt32.max)], &result, 1, vDSP_Length(count))
		vDSP_vsmsa(result, 1, [β-α], [α], &result, 1, vDSP_Length(count))
		return result
	}
	func uniform(_ α: Int, _ β: Int) -> Int {
		assert(α<β)
		return α+Int(arc4random_uniform(UInt32(β-α)))
	}
}
extension MTLBuffer {
	func write(to: URL) throws {
		try Data(bytesNoCopy: contents(), count: length, deallocator: .none).write(to: to)
	}
	var count: Int {
		return length / MemoryLayout<Float>.size
	}
	var ref: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
	var buffer: UnsafeMutableBufferPointer<Float> {
		return UnsafeMutableBufferPointer<Float>(start: ref, count: count)
	}
	func matrix(count: (rows: Int, cols: Int)) -> la_object_t {
		return matrix(rows: count.rows, cols: count.cols)
	}
	func matrix(rows: Int, cols: Int) -> la_object_t {
		return la_matrix_from_float_buffer_nocopy(ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr)
	}
}
extension la_object_t {
	var eye: la_object_t {
		let rows: Int = Int(la_matrix_rows(self))
		let cols: Int = Int(la_matrix_cols(self))
		let cache: Array<Float> = Array<Float>(repeating: 0, count: rows*rows*cols)
		assert(la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: cache), la_count_t((rows+1)*cols), self)==0)
		return la_matrix_from_float_buffer(cache, la_count_t(rows), la_count_t(rows*cols), la_count_t(rows*cols), hint, attr)
	}
	var array: Array<Float> {
		let rows: Int = Int(la_matrix_rows(self))
		let cols: Int = Int(la_matrix_cols(self))
		let cache: Array<Float> = Array<Float>(repeating: 0, count: rows*cols)
		assert(la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: cache), la_matrix_cols(self), self)==0)
		return cache
	}
	var hasErr: Bool {
		return la_status(self) != 0
	}
}
private let hint: la_hint_t = la_hint_t(LA_NO_HINT)
private let attr: la_attribute_t = la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING)
private let norm: la_norm_t = la_norm_t(LA_L2_NORM)
