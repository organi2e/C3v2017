//
//  ComputerTests.swift
//  ComputerTests
//
//  Created by Kota Nakano on 1/28/17.
//
//

import XCTest
import Accelerate
import Metal
import Computer

class ComputerTests: XCTestCase {
	let N: Int = 1024
	let K: Int = 64
	func testGPU() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = N
			let computer: Computer = try Computer(device: device)
			let x: Buffer = computer.make(length: N*MemoryLayout<Float>.size, options: .storageModeShared)
			let y: Buffer = computer.make(length: N*MemoryLayout<Float>.size, options: .storageModeShared)
			let z: Buffer = computer.make(length: N*MemoryLayout<Float>.size, options: .storageModeShared)
			let xref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(x.contents()))
			let yref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(y.contents()))
			let zref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(z.contents()))
			let range: Range<Int> = 0..<K
			/*
			arc4random_buf(y.contents(), MemoryLayout<Float>.size*N)
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(y.contents())), 1, xref, 1, vDSP_Length(N))
			vDSP_vsmul(xref, 1, [1.0/Float(65536)], xref, 1, vDSP_Length(N))
			vDSP_vsmul(xref, 1, [1.0/Float(65536)], xref, 1, vDSP_Length(N))
			vDSP_vclr(yref, 1, vDSP_Length(N))
			*/
			vDSP_vfill([Float(10.0)], xref, 1, vDSP_Length(N))
			vDSP_vfill([Float(-1.5)], yref, 1, vDSP_Length(N))
			print("before", zref[N-1])
			measure {
				for _ in range.lowerBound..<range.upperBound {
					computer.sigm(y: y, x: x, count: count)
					//computer.mul(z: z, y: y, x: x)
				}
			}
			computer.wait()
			print("after", zref[N-1])
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testCPU() {
		let xref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>.allocate(capacity: N)
		let yref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>.allocate(capacity: N)
		let zref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>.allocate(capacity: N)
		var count: Int32 = Int32(N)
		var one: Float = 1.0
		let range: Range<Int> = 0..<K
		/*
		arc4random_buf(yref, MemoryLayout<Float>.size*N)
		vDSP_vflt32(UnsafePointer<Int32>(OpaquePointer(yref)), 1, xref, 1, vDSP_Length(N))
		vDSP_vsmul(xref, 1, [1.0/Float(65536)], xref, 1, vDSP_Length(N))
		vDSP_vsmul(xref, 1, [1.0/Float(65536)], xref, 1, vDSP_Length(N))
		vDSP_vclr(yref, 1, vDSP_Length(N))
		*/
		vDSP_vfill([Float(10.0)], xref, 1, vDSP_Length(N))
		vDSP_vfill([Float(-1.5)], yref, 1, vDSP_Length(N))
		print("before", zref[N-1])
		measure {
			for _ in range.lowerBound..<range.upperBound {
				//vDSP_vmul(xref, 1, yref, 1, zref, 1, vDSP_Length(count))
				///*
				vDSP_vneg(xref, 1, yref, 1, vDSP_Length(count))
				vvexpf(yref, yref, &count)
				vDSP_vsadd(yref, 1, &one, yref, 1, vDSP_Length(count))
				vvrecf(yref, yref, &count)
				//*/
			}
		}
		print("after", zref[N-1])
		zref.deallocate(capacity: N)
		yref.deallocate(capacity: N)
		xref.deallocate(capacity: N)
	}
	func testGEMVGPU() {
		let rows: Int = 16 * 233
		let cols: Int = 16 * 233
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let computer: Computer = try Computer(device: device)
			let y: Buffer = computer.make(length: MemoryLayout<Float>.size*rows, options: .storageModeShared)
			let w: Buffer = computer.make(length: MemoryLayout<Float>.size*rows*cols, options: .storageModeShared)
			let x: Buffer = computer.make(length: MemoryLayout<Float>.size*cols, options: .storageModeShared)
			let yref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(y.contents()))
			let wref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(w.contents()))
			let xref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(x.contents()))
			let range: Range<Int> = 0..<K
			/*
			arc4random_buf(y.contents(), MemoryLayout<Float>.size*N)
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(y.contents())), 1, xref, 1, vDSP_Length(N))
			vDSP_vsmul(xref, 1, [1.0/Float(65536)], xref, 1, vDSP_Length(N))
			vDSP_vsmul(xref, 1, [1.0/Float(65536)], xref, 1, vDSP_Length(N))
			vDSP_vclr(yref, 1, vDSP_Length(N))
			*/
			vDSP_vgen([Float(-10)], [Float(10)], wref, 1, vDSP_Length(rows*cols))
			vDSP_vgen([Float(-1)], [Float(1)], xref, 1, vDSP_Length(cols))
			print("before", yref[rows-1])
			measure {
				for _ in range.lowerBound..<range.upperBound {
					computer.gemv(y: y, w: w, x: x, rows: rows, cols: cols)
					//computer.mul(z: z, y: y, x: x)
				}
			}
			computer.wait()
			print("after", yref[rows-1])
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testGEMVCPU() {
		let rows: Int = 16 * 233
		let cols: Int = 16 * 233
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let computer: Computer = try Computer(device: device)
			let y: Buffer = computer.make(length: MemoryLayout<Float>.size*rows, options: .storageModeShared)
			let w: Buffer = computer.make(length: MemoryLayout<Float>.size*rows*cols, options: .storageModeShared)
			let x: Buffer = computer.make(length: MemoryLayout<Float>.size*cols, options: .storageModeShared)
			let yref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(y.contents()))
			let wref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(w.contents()))
			let xref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(x.contents()))
			let wmat: la_object_t = la_matrix_from_float_buffer_nocopy(wref, la_count_t(rows), la_count_t(cols), la_count_t(cols), la_hint_t(LA_NO_HINT), nil, la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING))
			let xmat: la_object_t = la_matrix_from_float_buffer_nocopy(xref, la_count_t(rows), la_count_t(1), la_count_t(1), la_hint_t(LA_NO_HINT), nil, la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING))
			let range: Range<Int> = 0..<K
			/*
			arc4random_buf(y.contents(), MemoryLayout<Float>.size*N)
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(y.contents())), 1, xref, 1, vDSP_Length(N))
			vDSP_vsmul(xref, 1, [1.0/Float(65536)], xref, 1, vDSP_Length(N))
			vDSP_vsmul(xref, 1, [1.0/Float(65536)], xref, 1, vDSP_Length(N))
			vDSP_vclr(yref, 1, vDSP_Length(N))
			*/
			vDSP_vgen([Float(-10)], [Float(10)], wref, 1, vDSP_Length(rows*cols))
			vDSP_vgen([Float(-1)], [Float(1)], xref, 1, vDSP_Length(cols))
			print("before", yref[rows-1])
			measure {
				for _ in range.lowerBound..<range.upperBound {
					la_matrix_to_float_buffer(yref, la_count_t(1), la_matrix_product(wmat, la_elementwise_product(xmat, xmat)))
					la_matrix_to_float_buffer(yref, la_count_t(1), la_matrix_product(wmat, la_elementwise_product(xmat, xmat)))
				}
			}
			print("after", yref[rows-1])
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
