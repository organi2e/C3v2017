//
//  DistributorTests.swift
//  DistributorTests
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Accelerate
import Distributor
import XCTest

let hint: la_hint_t = la_hint_t(LA_NO_HINT)
let attr: la_attribute_t = la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING)
let norm: la_norm_t = la_norm_t(LA_L2_NORM)

func sqrt(_ x: la_object_t) -> la_object_t {
	let rows: Int = Int(la_matrix_rows(x))
	let cols: Int = Int(la_matrix_cols(x))
	let count: Int = rows * cols
	let cache: Array<Float> = Array<Float>(repeating: 0, count: count)
	la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: cache), la_count_t(cols), x)
	vvsqrtf(UnsafeMutablePointer<Float>(mutating: cache), cache, [Int32(count)])
	return la_matrix_from_float_buffer(cache, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, attr)
}

class DistributorTests: XCTestCase {
	
	func testGaussRNG() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 65536
			let distributor: Distributor = try Gauss.factory()(device)(count)
			var dstμ: Float = 100
			var estμ: Float = 0
			var dstσ: Float = 10.0
			var estσ: Float = 0
			let χ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let μ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let σ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			
			vDSP_vfill(&dstμ, UnsafeMutablePointer<Float>(OpaquePointer(μ.contents())), 1, vDSP_Length(count))
			vDSP_vfill(&dstσ, UnsafeMutablePointer<Float>(OpaquePointer(σ.contents())), 1, vDSP_Length(count))
			
			let commandQueue: MTLCommandQueue = device.makeCommandQueue()
			measure {
				for _ in 0..<256 {
					let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer()
					distributor.shuffle(commandBuffer: commandBuffer, χ: χ, μ: μ, σ: σ, count: count)
					commandBuffer.commit()
				}
			}
			let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer()
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			vDSP_normalize(UnsafePointer<Float>(OpaquePointer(χ.contents())), 1, nil, 1, &estμ, &estσ, vDSP_Length(count))
			
			print(dstμ, estμ)
			print(dstσ, estσ)
			
			XCTAssert(fabs(dstμ-estμ)/dstμ<1e-3)
			XCTAssert(fabs(dstσ-estσ)/dstσ<1e-1)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	func testGaussError() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 1024
			let distributor: Distributor = try Gauss.factory()(device)(count)
			
			let queue: MTLCommandQueue = device.makeCommandQueue()
			
			let ΔS: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let ΔX: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			
			let ψs: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let ψx: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let Δs: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let Δx: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let χ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let μ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let σ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			
			[ψs, ψx, Δs, Δx, μ, σ].forEach {
				let count: Int = $0.length / MemoryLayout<Float>.size
				arc4random_buf($0.ref, $0.length)
				vDSP_vfltu32(UnsafePointer< UInt32>(OpaquePointer($0.ref)), 1, $0.ref, 1, vDSP_Length(count))
				vDSP_vsmul($0.ref, 1, [1.0/Float(65536)], $0.ref, 1, vDSP_Length(count))
				vDSP_vsmul($0.ref, 1, [1.0/Float(65536)], $0.ref, 1, vDSP_Length(count))
			}
			
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.reset(commandBuffer: commandBuffer)
			distributor.collect(commandBuffer: commandBuffer, χ: χ, μ: μ, σ: σ)
			distributor.errorValue(commandBuffer: commandBuffer, Δ: Δx, ψ: ψx)
			distributor.errorState(commandBuffer: commandBuffer, Δ: Δs, ψ: ψs)
			commandBuffer.commit()
			
			for k in 0..<count {
				ΔX.ref[k] = ψx.ref[k] - μ.ref[k]
				ΔS.ref[k] = ψs.ref[k] - 0.5 - 0.5 * erf(μ.ref[k]/σ.ref[k]*Float(M_SQRT1_2))
			}
			
			commandBuffer.waitUntilCompleted()
			
			XCTAssert(la_norm_as_float(la_difference(
				la_matrix_from_float_buffer_nocopy(ΔX.ref, la_count_t(count), 1, 1, hint, nil, attr),
				la_matrix_from_float_buffer_nocopy(Δx.ref, la_count_t(count), 1, 1, hint, nil, attr)), norm
				) < 1e-3
			)
			
			XCTAssert(la_norm_as_float(la_difference(
				la_matrix_from_float_buffer_nocopy(ΔS.ref, la_count_t(count), 1, 1, hint, nil, attr),
				la_matrix_from_float_buffer_nocopy(Δs.ref, la_count_t(count), 1, 1, hint, nil, attr)), norm
				) < 1e-3
			)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	func testGaussCollectW() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: (rows: Int, cols: Int) = (rows: 1024, cols: 1024)
			let rows: Int = count.rows
			let cols: Int = count.cols
			let distributor: Distributor = try Gauss.factory()(device)(rows)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			
			let y: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) = (
				χ: device.makeBuffer(length: MemoryLayout<Float>.size*rows, options: .storageModeShared),
				μ: device.makeBuffer(length: MemoryLayout<Float>.size*rows, options: .storageModeShared),
				σ: device.makeBuffer(length: MemoryLayout<Float>.size*rows, options: .storageModeShared)
			)
			let w: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) = (
				χ: device.makeBuffer(length: MemoryLayout<Float>.size*rows*cols, options: .storageModeShared),
				μ: device.makeBuffer(length: MemoryLayout<Float>.size*rows*cols, options: .storageModeShared),
				σ: device.makeBuffer(length: MemoryLayout<Float>.size*rows*cols, options: .storageModeShared)
			)
			let x: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*cols, options: .storageModeShared)
			[x, w.χ, w.μ, w.σ].forEach {
				let count: Int = $0.length / MemoryLayout<Float>.size
				arc4random_buf($0.ref, $0.length)
				vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer($0.ref)), 1, $0.ref, 1, vDSP_Length(count))
				vDSP_vsmul($0.ref, 1, [1.0/Float(65536)], $0.ref, 1, vDSP_Length(count))
				vDSP_vsmul($0.ref, 1, [1.0/Float(65536)], $0.ref, 1, vDSP_Length(count))
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.reset(commandBuffer: commandBuffer)
				distributor.collect(commandBuffer: commandBuffer, w: w, x: x, refer: cols)
				distributor.synthesize(commandBuffer: commandBuffer, χ: y.χ, μ: y.μ, σ: y.σ)
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
			XCTAssert(la_norm_as_float(la_difference(
				la_matrix_from_float_buffer_nocopy(y.χ.ref, la_count_t(rows), 1, 1, hint, nil, attr),
				la_matrix_product(la_matrix_from_float_buffer_nocopy(w.χ.ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr), la_matrix_from_float_buffer_nocopy(x.ref, la_count_t(cols), la_count_t(1), la_count_t(1), hint, nil, attr))), norm
				) < 1e-2
			)
			XCTAssert(la_norm_as_float(la_difference(
				la_matrix_from_float_buffer_nocopy(y.μ.ref, la_count_t(rows), 1, 1, hint, nil, attr),
				la_matrix_product(la_matrix_from_float_buffer_nocopy(w.μ.ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr), la_matrix_from_float_buffer_nocopy(x.ref, la_count_t(cols), la_count_t(1), la_count_t(1), hint, nil, attr))), norm
				) < 1e-2
			)
			XCTAssert(la_norm_as_float(la_difference(
				la_matrix_from_float_buffer_nocopy(y.σ.ref, la_count_t(rows), 1, 1, hint, nil, attr),
				sqrt(la_matrix_product(la_elementwise_product(la_matrix_from_float_buffer_nocopy(w.σ.ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr), la_matrix_from_float_buffer_nocopy(w.σ.ref, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr)), la_elementwise_product(la_matrix_from_float_buffer_nocopy(x.ref, la_count_t(cols), la_count_t(1), la_count_t(1), hint, nil, attr), la_matrix_from_float_buffer_nocopy(x.ref, la_count_t(cols), la_count_t(1), la_count_t(1), hint, nil, attr))))), norm
				) < 1e-4
			)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testGaussCollectC() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 1024
			let distributor: Distributor = try Gauss.factory()(device)(count)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let y: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) = (
				χ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
				μ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
				σ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			)
			let c: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) = (
				χ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
				μ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
				σ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			)
			vDSP_vgen([-1.0], [1.0], c.χ.reference(), 1, vDSP_Length(count))
			vDSP_vgen([-2.0], [2.0], c.μ.reference(), 1, vDSP_Length(count))
			vDSP_vgen([ 4.0], [9.0], c.σ.reference(), 1, vDSP_Length(count))
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.reset(commandBuffer: commandBuffer)
				distributor.collect(commandBuffer: commandBuffer, χ: c.χ, μ: c.μ, σ: c.σ)
				distributor.synthesize(commandBuffer: commandBuffer, χ: y.χ, μ: y.μ, σ: y.σ)
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
			XCTAssert(la_norm_as_float(la_difference(
				la_matrix_from_float_buffer_nocopy(c.χ.ref, la_count_t(count), 1, 1, hint, nil, attr),
				la_matrix_from_float_buffer_nocopy(y.χ.ref, la_count_t(count), 1, 1, hint, nil, attr)), norm
				) < 1e-3
			)
			XCTAssert(la_norm_as_float(la_difference(
				la_matrix_from_float_buffer_nocopy(c.μ.ref, la_count_t(count), 1, 1, hint, nil, attr),
				la_matrix_from_float_buffer_nocopy(y.μ.ref, la_count_t(count), 1, 1, hint, nil, attr)), norm
				) < 1e-3
			)
			XCTAssert(
				la_norm_as_float(la_difference(
					la_matrix_from_float_buffer_nocopy(c.σ.ref, la_count_t(count), 1, 1, hint, nil, attr),
					la_matrix_from_float_buffer_nocopy(y.σ.ref, la_count_t(count), 1, 1, hint, nil, attr)), norm
					) < 1e-3
			)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testGaussCollectD() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 1024
			let distributor: Distributor = try Gauss.factory()(device)(count)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let y: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) = (
				χ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
				μ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
				σ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			)
			let d: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let c: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) = (
				χ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
				μ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
				σ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			)
			vDSP_vgen([-1.0], [1.0], c.χ.reference(), 1, vDSP_Length(count))
			vDSP_vgen([-2.0], [2.0], c.μ.reference(), 1, vDSP_Length(count))
			vDSP_vgen([ 4.0], [9.0], c.σ.reference(), 1, vDSP_Length(count))
			vDSP_vgen([10.0], [1.0], d.reference(), 1, vDSP_Length(count))
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.reset(commandBuffer: commandBuffer)
				distributor.collect(commandBuffer: commandBuffer, r: d, x: (χ: c.χ, μ: c.μ, σ: c.σ))
				distributor.synthesize(commandBuffer: commandBuffer, χ: y.χ, μ: y.μ, σ: y.σ)
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
			XCTAssert(la_norm_as_float(la_difference(
				la_elementwise_product(
					la_matrix_from_float_buffer_nocopy(d.ref, la_count_t(count), 1, 1, hint, nil, attr),
					la_matrix_from_float_buffer_nocopy(c.χ.ref, la_count_t(count), 1, 1, hint, nil, attr)
				), la_matrix_from_float_buffer_nocopy(y.χ.ref, la_count_t(count), 1, 1, hint, nil, attr)
				), norm
				) < 1e-3
			)
			XCTAssert(la_norm_as_float(la_difference(
				la_elementwise_product(
					la_matrix_from_float_buffer_nocopy(d.ref, la_count_t(count), 1, 1, hint, nil, attr),
					la_matrix_from_float_buffer_nocopy(c.μ.ref, la_count_t(count), 1, 1, hint, nil, attr)
				), la_matrix_from_float_buffer_nocopy(y.μ.ref, la_count_t(count), 1, 1, hint, nil, attr)
				), norm
				) < 1e-3
			)
			XCTAssert(la_norm_as_float(la_difference(
				la_elementwise_product(
					la_matrix_from_float_buffer_nocopy(d.ref, la_count_t(count), 1, 1, hint, nil, attr),
					la_matrix_from_float_buffer_nocopy(c.σ.ref, la_count_t(count), 1, 1, hint, nil, attr)
				),	la_matrix_from_float_buffer_nocopy(y.σ.ref, la_count_t(count), 1, 1, hint, nil, attr)
				), norm
				) < 1e-3
			)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	/*
	func testGaussSynthesize() {
	guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
	do {
	let count: Int = 1024
	let distributor: Distributor = try Gauss.factory()(device)(count)
	
	let Σ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) = (
	χ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
	μ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
	σ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
	)
	
	let ϝ: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer) = (
	χ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
	μ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared),
	σ: device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
	)
	
	vDSP_vgen([-1.0], [1.0], Σ.χ.reference(), 1, vDSP_Length(count))
	vDSP_vgen([-2.0], [2.0], Σ.μ.reference(), 1, vDSP_Length(count))
	vDSP_vgen([ 4.0], [9.0], Σ.σ.reference(), 1, vDSP_Length(count))
	
	let commandQueue: MTLCommandQueue = device.makeCommandQueue()
	measure {
	for _ in 0..<256 {
	let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer()
	distributor.synthesize(commandBuffer: commandBuffer, χ: ϝ.χ, μ: Σ.μ, σ: Σ.σ)
	commandBuffer.commit()
	}
	let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer()
	commandBuffer.commit()
	commandBuffer.waitUntilCompleted()
	
	}
	do {
	let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer()
	distributor.reset(commandBuffer: commandBuffer)
	commandBuffer.commit()
	}
	
	
	
	do {
	var E: Float = 0
	for k in 0..<count {
	let e: Float = ϝ.χ.ref[k] - step(Σ.χ.ref[k], edge: 0)
	E += e * e
	}
	XCTAssert(E/Float(count)<1e-3)
	}
	do {
	var E: Float = 0
	for k in 0..<count {
	let e: Float = ϝ.μ.ref[k] - Σ.μ.ref[k]
	E += e * e
	}
	XCTAssert(E/Float(count)<1e-3)
	}
	do {
	var E: Float = 0
	for k in 0..<count {
	let e: Float = ϝ.σ.ref[k] - sqrt(Σ.σ.ref[k])
	E += e * e
	}
	XCTAssert(E/Float(count)<1e-3)
	}
	} catch {
	XCTFail(String(describing: error))
	}
	}
	*/
	/*
	func testSynthesize() {
	guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
	do {
	let computer: Computer = try Computer(device: device)
	let count: Int = 65536
	guard let distributor: Gauss = try Gauss.make()(computer)(count) as? Gauss else { XCTFail()(); return }
	} catch {
	XCTFail(String(describing: error))
	}
	
	}
	
	func testGradient() {
	
	}
	
	func testCollectC() {
	guard let device: Device = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
	do {
	let computer: Computer = try Computer(device: device)
	let count: Int = 1024
	guard let distributor: Gauss = try Gauss.make()(computer)(count) as? Gauss else { XCTFail(); return }
	
	distributor.clear()
	
	
	} catch {
	XCTFail(String(describing: error))
	}
	}
	
	func testGaussCollectA() {
	guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
	do {
	let computer: Computer = try Computer(device: device)
	
	let hint: la_hint_t = la_hint_t(LA_ATTRIBUTE_ENABLE_LOGGING)
	let attr: la_attribute_t = la_attribute_t(LA_DEFAULT_ATTRIBUTES)
	let norm: la_norm_t = la_norm_t(LA_L2_NORM)
	
	let count: (rows: Int, cols: Int) = (rows: 1024, cols: 1024)
	let rows: Int = count.rows
	let cols: Int = count.cols
	
	guard let distributor: Gauss = try Gauss.make()(computer)(rows) as? Gauss else { XCTFail(); return }
	
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
	
	distributor.collect(w: w, x: χ)
	computer.compute {
	let encoder: BlitCommandEncoder = $0.makeBlitCommandEncoder()
	encoder.copy(from: distributor.Σ.χ, sourceOffset: 0, to: Σ.χ, destinationOffset: 0, size: rows*MemoryLayout<Float>.size)
	encoder.copy(from: distributor.Σ.μ, sourceOffset: 0, to: Σ.μ, destinationOffset: 0, size: rows*MemoryLayout<Float>.size)
	encoder.copy(from: distributor.Σ.σ, sourceOffset: 0, to: Σ.σ, destinationOffset: 0, size: rows*MemoryLayout<Float>.size)
	encoder.endEncoding()
	}
	computer.wait()
	
	let χrmse: Float = la_norm_as_float(la_difference(Σmat.χ, la_matrix_product(wmat.χ, χmat)), norm)
	let μrmse: Float = la_norm_as_float(la_difference(Σmat.μ, la_matrix_product(wmat.μ, χmat)), norm)
	let σrmse: Float = la_norm_as_float(la_difference(Σmat.σ, la_matrix_product(la_elementwise_product(wmat.σ, wmat.σ), la_elementwise_product(χmat, χmat))), norm)
	
	print("error", χrmse, μrmse, σrmse)
	
	XCTAssert(χrmse < 1e-3)
	XCTAssert(μrmse < 1e-3)
	XCTAssert(σrmse < 1e-3)
	
	//try Σ.σ.write(to: URL(fileURLWithPath: "/tmp/gpu.raw"))
	//try la_matrix_product(la_elementwise_product(wmat.σ, wmat.σ), la_elementwise_product(χmat, χmat)).write(to: URL(fileURLWithPath: "/tmp/cpu.raw"))
	
	
	} catch {
	XCTFail(String(describing: error))
	}
	}
	*/
}
extension DistributorTests {
	func testJacobianUA() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 256
			let distributor: Distributor = try Gauss.factory()(device)(count)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let jμ: MTLBuffer = device.makeBuffer(length: count*count*count*MemoryLayout<Float>.size, options: .storageModeShared)
			let jy: MTLBuffer = device.makeBuffer(length: count*count*count*MemoryLayout<Float>.size, options: .storageModeShared)
			let x: MTLBuffer = device.makeBuffer(length: count*MemoryLayout<Float>.size, options: .storageModeShared)
			shuffle(array: [x])
			measure {
				do {
					let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
					distributor.clear(commandBuffer: commandBuffer, jμ: jμ, jσ: jy)
					distributor.jacobian(commandBuffer: commandBuffer, jμ: jμ, x: x, refer: count)
					commandBuffer.commit()
					commandBuffer.waitUntilCompleted()
				}
				let la_x: la_object_t = la_matrix_from_float_buffer_nocopy(x.ref, la_count_t(count), la_count_t(1), la_count_t(1), hint, nil, attr)
				let la_o: la_object_t = la_vector_from_splat(la_splat_from_float(1.0, attr), la_count_t(count))
				la_matrix_to_float_buffer(jy.ref, la_count_t(count*(count+1)), la_outer_product(la_o, la_x))
				let la_A: la_object_t = la_matrix_from_float_buffer_nocopy(jμ.ref, la_count_t(count), la_count_t(count*count), la_count_t(count*count), hint, nil, attr)
				let la_B: la_object_t = la_matrix_from_float_buffer_nocopy(jy.ref, la_count_t(count), la_count_t(count*count), la_count_t(count*count), hint, nil, attr)
				XCTAssert(la_norm_as_float(la_difference(la_A, la_B), norm)<1e-3)
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianUC() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 1024
			let distributor: Distributor = try Gauss.factory()(device)(count)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let jμ: MTLBuffer = device.makeBuffer(length: count*count*MemoryLayout<Float>.size, options: .storageModeShared)
			let jσ: MTLBuffer = device.makeBuffer(length: count*count*MemoryLayout<Float>.size, options: .storageModeShared)
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.clear(commandBuffer: commandBuffer, jμ: jμ, jσ: jσ)
				distributor.jacobian(commandBuffer: commandBuffer, jμ: jμ)
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
			let la_A: la_object_t = la_matrix_from_float_buffer_nocopy(jμ.ref, la_count_t(count), la_count_t(count), la_count_t(count), hint, nil, attr)
			let la_B: la_object_t = la_identity_matrix(la_count_t(count), la_scalar_type_t(LA_SCALAR_TYPE_FLOAT), attr)
			XCTAssert(la_norm_as_float(la_difference(la_A, la_B), norm)<1e-3)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianUX() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 1024
			let distributor: Distributor = try Gauss.factory()(device)(count)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let jμ: MTLBuffer = device.makeBuffer(length: count*count*MemoryLayout<Float>.size, options: .storageModeShared)
			let jσ: MTLBuffer = device.makeBuffer(length: count*count*MemoryLayout<Float>.size, options: .storageModeShared)
			let w: MTLBuffer = device.makeBuffer(length: count*count*MemoryLayout<Float>.size, options: .storageModeShared)
			shuffle(array: [w])
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.clear(commandBuffer: commandBuffer, jμ: jμ, jσ: jσ)
				distributor.jacobian(commandBuffer: commandBuffer, jμ: jμ, w: w, refer: count)
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
			let la_w: la_object_t = la_matrix_from_float_buffer_nocopy(w.ref, la_count_t(count), la_count_t(count), la_count_t(count), hint, nil, attr)
			let la_A: la_object_t = la_matrix_from_float_buffer_nocopy(jμ.ref, la_count_t(count), la_count_t(count), la_count_t(count), hint, nil, attr)
			let la_B: la_object_t = la_w
			XCTAssert(la_norm_as_float(la_difference(la_A, la_B), norm)<1e-3)
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
extension DistributorTests {
	func shuffle(array: Array<MTLBuffer>) {
		array.forEach {
			let count: vDSP_Length = vDSP_Length($0.length / MemoryLayout<Float>.size)
			arc4random_buf($0.ref, $0.length)
			vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer($0.contents())), 1, $0.ref, 1, count)
			vDSP_vsmul($0.ref, 1, [1.0/Float(65536)], $0.ref, 1, count)
			vDSP_vsmul($0.ref, 1, [1.0/Float(65536)], $0.ref, 1, count)
		}
	}
}
extension MTLBuffer {
	var ref: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
	var array: Array<Float> {
		let buffer: UnsafeBufferPointer<Float> = UnsafeBufferPointer<Float>(start: UnsafePointer<Float>(OpaquePointer(contents())), count: length/MemoryLayout<Float>.size)
		return Array<Float>(buffer)
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
	var array: Array<Float> {
		let rows: UInt = la_matrix_rows(self)
		let cols: UInt = la_matrix_cols(self)
		let array: Array<Float> = Array<Float>(repeating: 0, count: Int(rows*cols))
		la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: array), cols, self)
		return array
	}
}
extension MTLBuffer {
	public func reference<T>() -> UnsafeMutablePointer<T> {
		return UnsafeMutablePointer<T>(OpaquePointer(contents()))
	}
}
