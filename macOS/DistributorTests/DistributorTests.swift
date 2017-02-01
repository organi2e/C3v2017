//
//  DistributorTests.swift
//  DistributorTests
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Accelerate
import XCTest
import Distributor

class DistributorTests: XCTestCase {
	
	func testGaussCDF() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 65536
			let distributor: Distributor = try Gauss(device: device)
			let χ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let μ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let σ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			
			vDSP_vfill([Float(0.5)], UnsafeMutablePointer<Float>(OpaquePointer(σ.contents())), 1, vDSP_Length(count))
			vDSP_vgen([Float(-10)], [Float(10)], UnsafeMutablePointer<Float>(OpaquePointer(μ.contents())), 1, vDSP_Length(count))
			
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			
			distributor.encode(commandBuffer: commandBuffer, CDF: χ, μ: μ, σ: σ, count: count)
			
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			let χref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(χ.contents()))
			let μref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(μ.contents()))
			let σref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(σ.contents()))
			
			let ybuf: Array<Float> = (0..<count).map {
				0.5+0.5*erf(μref[$0]/σref[$0]*Float(M_SQRT1_2))
			}
			let E: Array<Float> = (0..<count).map {
				ybuf[$0] - χref[$0]
			}
			let MSE: Float = E.reduce(Float(0)) {
				$0.0 + ( $0.1 * $0.1 )
				} / Float(E.count)
			let RMSE: Float = sqrt(MSE)
			
			//try Data(bytes: χ.contents(), count: χ.length).write(to: URL(fileURLWithPath: "/tmp/x.raw"))
			//try Data(bytes: ybuf, count: ybuf.count*MemoryLayout<Float>.size).write(to: URL(fileURLWithPath: "/tmp/y.raw"))
			
			XCTAssert(RMSE<1e-7)
			
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	func testGaussPDF() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 65536
			let distributor: Distributor = try Gauss(device: device)
			let χ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let μ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let σ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			
			vDSP_vfill([Float(0.5)], UnsafeMutablePointer<Float>(OpaquePointer(σ.contents())), 1, vDSP_Length(count))
			vDSP_vgen([Float(-10.5)], [Float(10)], UnsafeMutablePointer<Float>(OpaquePointer(μ.contents())), 1, vDSP_Length(count))
			
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			
			distributor.encode(commandBuffer: commandBuffer, PDF: χ, μ: μ, σ: σ, count: count)
			
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			let χref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(χ.contents()))
			let μref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(μ.contents()))
			let σref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(σ.contents()))
			
			let ybuf: Array<Float> = (0..<count).map {
				exp(-0.5*μref[$0]*μref[$0]/σref[$0]/σref[$0])/σref[$0]*Float(0.5*M_SQRT1_2*M_2_SQRTPI)
			}
			let E: Array<Float> = (0..<count).map {
				ybuf[$0] - χref[$0]
			}
			let MSE: Float = E.reduce(Float(0)) {
				$0.0 + ( $0.1 * $0.1 )
				} / Float(E.count)
			let RMSE: Float = sqrt(MSE)
			
			//try Data(bytes: χ.contents(), count: χ.length).write(to: URL(fileURLWithPath: "/tmp/x.raw"))
			//try Data(bytes: ybuf, count: ybuf.count*MemoryLayout<Float>.size).write(to: URL(fileURLWithPath: "/tmp/y.raw"))
			
			XCTAssert(RMSE<1e-7)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	func testGaussRNG() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 1024 * 1024 * 16
			
			let distributor: Distributor = try Gauss(device: device)
			
			var dstμ: Float = 100
			var estμ: Float = 0
			var dstσ: Float = 10.0
			var estσ: Float = 0
			let χ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let μ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let σ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			
			vDSP_vfill(&dstμ, UnsafeMutablePointer<Float>(OpaquePointer(μ.contents())), 1, vDSP_Length(count))
			vDSP_vfill(&dstσ, UnsafeMutablePointer<Float>(OpaquePointer(σ.contents())), 1, vDSP_Length(count))
			
			let queue: MTLCommandQueue = device.makeCommandQueue()
			measure {
				for _ in 0..<64 {
					let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
					distributor.encode(commandBuffer: commandBuffer, χ: χ, μ: μ, σ: σ, count: count)
					commandBuffer.commit()
				}
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
			
			vDSP_normalize(UnsafePointer<Float>(OpaquePointer(χ.contents())), 1, nil, 1, &estμ, &estσ, vDSP_Length(count))
			
			XCTAssert(fabs(dstμ-estμ)/dstμ<1e-3)
			XCTAssert(fabs(dstσ-estσ)/dstσ<1e-1)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	func testGaussChain() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let distributor: Distributor = try Gauss(device: device)
			
			let hint: la_hint_t = la_hint_t(LA_ATTRIBUTE_ENABLE_LOGGING)
			let attr: la_attribute_t = la_attribute_t(LA_DEFAULT_ATTRIBUTES)
			let norm: la_norm_t = la_norm_t(LA_L2_NORM)
			
			let count: (rows: Int, cols: Int) = (rows: 1024, cols: 1024)
			let rows: Int = count.rows
			let cols: Int = count.cols
			
			let Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) = (
				χ: device.makeBuffer(length: MemoryLayout<Float>.size*rows, options: .storageModeShared),
				μ: device.makeBuffer(length: MemoryLayout<Float>.size*rows, options: .storageModeShared),
				σ: device.makeBuffer(length: MemoryLayout<Float>.size*rows, options: .storageModeShared)
			)
			let w: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) = (
				χ: device.makeBuffer(length: MemoryLayout<Float>.size*rows*cols, options: .storageModeShared),
				μ: device.makeBuffer(length: MemoryLayout<Float>.size*rows*cols, options: .storageModeShared),
				σ: device.makeBuffer(length: MemoryLayout<Float>.size*rows*cols, options: .storageModeShared)
			)
			
			let χ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*cols, options: .storageModeShared)
			
			vDSP_vclr(Σ.χ.ref, 1, vDSP_Length(rows))
			vDSP_vclr(Σ.μ.ref, 1, vDSP_Length(rows))
			vDSP_vclr(Σ.σ.ref, 1, vDSP_Length(rows))
			
			[w.χ, w.μ, w.σ, χ].forEach {
				let count: Int = $0.length / MemoryLayout<Float>.size
				arc4random_buf($0.ref, $0.length)
				vDSP_vflt32(UnsafePointer<Int32>(OpaquePointer($0.ref)), 1, $0.ref, 1, vDSP_Length(count))
				vDSP_vsmul($0.ref, 1, [1.0/Float(65536)], $0.ref, 1, vDSP_Length(count))
				vDSP_vsmul($0.ref, 1, [1.0/Float(65536)], $0.ref, 1, vDSP_Length(count))
			}
			
			let Σmat: (χ: la_object_t, μ: la_object_t, σ: la_object_t) = (
				χ: la_matrix_from_float_buffer_nocopy(Σ.χ.ref, la_count_t(rows), la_count_t(1), la_count_t(1), hint, nil, attr),
				μ: la_matrix_from_float_buffer_nocopy(Σ.μ.ref, la_count_t(rows), la_count_t(1), la_count_t(1), hint, nil, attr),
				σ: la_matrix_from_float_buffer_nocopy(Σ.σ.ref, la_count_t(rows), la_count_t(1), la_count_t(1), hint, nil, attr)
			)
			
			let wmat: (χ: la_object_t, μ: la_object_t, σ: la_object_t) = (
				χ: la_matrix_from_float_buffer_nocopy(w.χ.ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr),
				μ: la_matrix_from_float_buffer_nocopy(w.μ.ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr),
				σ: la_matrix_from_float_buffer_nocopy(w.σ.ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr)
			)
			let χmat: la_object_t = la_matrix_from_float_buffer_nocopy(χ.ref, la_count_t(cols), la_count_t(1), la_count_t(1), hint, nil, attr)
			
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.collect(commandBuffer: commandBuffer, Σ: Σ, w: w, x: χ, count: count)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			let rmse: (χ: Float, μ: Float, σ: Float) = (
				χ: sqrt(la_norm_as_float(la_difference(Σmat.χ, la_matrix_product(wmat.χ, χmat)), norm)),
				μ: sqrt(la_norm_as_float(la_difference(Σmat.μ, la_matrix_product(wmat.μ, χmat)), norm)),
				σ: sqrt(la_norm_as_float(la_difference(Σmat.σ, la_matrix_product(la_elementwise_product(wmat.σ, wmat.σ), la_elementwise_product(χmat, χmat))), norm))
			)
			
			XCTAssert(rmse.χ<1e-2)
			XCTAssert(rmse.μ<1e-2)
			XCTAssert(rmse.σ<1e-2)
			
			//try Σ.σ.write(to: URL(fileURLWithPath: "/tmp/gpu.raw"))
			//try la_matrix_product(la_elementwise_product(wmat.σ, wmat.σ), la_elementwise_product(χmat, χmat)).write(to: URL(fileURLWithPath: "/tmp/cpu.raw"))
			
			/*
			measure {
			for _ in 0..<256 {
			if true {
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.collect(commandBuffer: commandBuffer, Σ: Σ, w: w, x: χ, count: count)
			commandBuffer.commit()
			} else {
			la_matrix_to_float_buffer(Σχref, la_count_t(1), la_matrix_product(wmat.χ, χmat))
			la_matrix_to_float_buffer(Σμref, la_count_t(1), la_matrix_product(wmat.μ, χmat))
			la_matrix_to_float_buffer(Σσref, la_count_t(1), la_matrix_product(la_elementwise_product(wmat.σ, wmat.σ), la_elementwise_product(χmat, χmat)))
			}
			}
			if true {
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			}
			}
			*/
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
extension MTLBuffer {
	var ref: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
	func write(to: URL) throws {
		try Data(bytesNoCopy: contents(), count: length, deallocator: .none).write(to: to)
	}
}
extension la_object_t {
	func write(to: URL) throws {
		let rows: la_count_t = la_matrix_rows(self)
		let cols: la_count_t = la_matrix_cols(self)
		var data: Data = Data(count: MemoryLayout<Float>.size*Int(rows*cols))
		assert(la_status_t(LA_SUCCESS) == data.withUnsafeMutableBytes {
			la_matrix_to_float_buffer($0, cols, self)
			})
		try data.write(to: to)
	}
}
