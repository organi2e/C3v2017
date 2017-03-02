//
//  DistributorTests.swift
//  DistributorTests
//
//  Created by Kota Nakano on 2017/01/27.
//
//

/*
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
				$0.didModifyRange(NSRange(0..<$0.length))
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
}
extension DistributorTests {
	func testJacobianμA() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let width: Int = 64
			let refer: Int = 128
			let distributor: Distributor = try Gauss.factory()(device)(width)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let jμ: MTLBuffer = device.makeBuffer(length: width*width*refer*MemoryLayout<Float>.size, options: .storageModeShared)
			let jσ: MTLBuffer = device.makeBuffer(length: width*width*refer*MemoryLayout<Float>.size, options: .storageModeShared)
			let x: MTLBuffer = device.makeBuffer(length: refer*MemoryLayout<Float>.size, options: .storageModeShared)
			
			shuffle(array: [x])
			
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.clear(commandBuffer: commandBuffer, jμ: jμ, jσ: jσ)
			distributor.jacobian(commandBuffer: commandBuffer, jμ: jμ, x: x, refer: refer)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			let la_x: la_object_t = la_matrix_from_float_buffer_nocopy(x.ref, la_count_t(refer), la_count_t(1), la_count_t(1), hint, nil, attr)
			let la_o: la_object_t = la_vector_from_splat(la_splat_from_float(1.0, attr), la_count_t(width))
			la_matrix_to_float_buffer(jσ.ref, la_count_t((width+1)*refer), la_outer_product(la_o, la_x))
			let la_A: la_object_t = la_matrix_from_float_buffer_nocopy(jμ.ref, la_count_t(width), la_count_t(width*refer), la_count_t(width*refer), hint, nil, attr)
			let la_B: la_object_t = la_matrix_from_float_buffer_nocopy(jσ.ref, la_count_t(width), la_count_t(width*refer), la_count_t(width*refer), hint, nil, attr)
			let la_C: la_object_t = la_difference(la_A, la_B)
			
			XCTAssert(la_status(la_C)==0)
			XCTAssert(la_norm_as_float(la_C, norm)<1e-3)
			
			//try la_A.write(to: URL(fileURLWithPath: "/tmp/a.raw"))
			//try la_B.write(to: URL(fileURLWithPath: "/tmp/b.raw"))

		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianμB() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let width: Int = 72
			let refer: Int = 159
			let distributor: Distributor = try Gauss.factory()(device)(width)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let jμ: MTLBuffer = device.makeBuffer(length: width * width * refer * MemoryLayout<Float>.size, options: .storageModeShared)
			let jσ: MTLBuffer = device.makeBuffer(length: width * width * refer * MemoryLayout<Float>.size, options: .storageModeShared)
			let b: MTLBuffer = device.makeBuffer(length: width * width * MemoryLayout<Float>.size, options: .storageModeShared)
			let j: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let p: MTLBuffer = device.makeBuffer(length: width * width * refer * MemoryLayout<Float>.size, options: .storageModeShared)
			shuffle(array: [b, j, p])
			do {
				//measure {
				do {
					let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
					distributor.clear(commandBuffer: commandBuffer, jμ: jμ, jσ: jσ)
					distributor.jacobian(commandBuffer: commandBuffer, jμ: jμ, b: b, j: j, p: p, refer: refer) //ub
					commandBuffer.commit()
					commandBuffer.waitUntilCompleted()
				}
				let rows: la_count_t = la_count_t(width)
				let cols: la_count_t = la_count_t(width*refer)
				let la_jμ: la_object_t = la_matrix_from_float_buffer_nocopy(jμ.ref, rows, cols, cols, hint, nil, attr)
				let la_b: la_object_t = la_matrix_from_float_buffer_nocopy(b.ref, rows, rows, rows, hint, nil, attr)
				let la_df: la_object_t = la_diagonal_matrix_from_vector(la_matrix_from_float_buffer_nocopy(j.ref, rows, 1, 1, hint, nil, attr), 0)
				let la_p: la_object_t = la_matrix_from_float_buffer_nocopy(p.ref, rows, cols, cols, hint, nil, attr)
				let la_A: la_object_t = la_jμ
				let la_B: la_object_t = la_matrix_product(la_b, la_matrix_product(la_df, la_p))
				let la_C: la_object_t = la_difference(la_A, la_B)
				XCTAssert(la_status(la_C) == 0)
				XCTAssert(la_norm_as_float(la_C, norm)<1e-3)
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianμC() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 1024
			let distributor: Distributor = try Gauss.factory()(device)(count)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let jμ: MTLBuffer = device.makeBuffer(length: count*count*MemoryLayout<Float>.size, options: .storageModeShared)
			let jσ: MTLBuffer = device.makeBuffer(length: count*count*MemoryLayout<Float>.size, options: .storageModeShared)
			let c: MTLBuffer = device.makeBuffer(length: count*MemoryLayout<Float>.size, options: .storageModeShared)
			shuffle(array: [c])
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.clear(commandBuffer: commandBuffer, jμ: jμ, jσ: jσ)
				distributor.jacobian(commandBuffer: commandBuffer, jμ: jμ, c: c)
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
			let la_A: la_object_t = la_matrix_from_float_buffer_nocopy(jμ.ref, la_count_t(count), la_count_t(count), la_count_t(count), hint, nil, attr)
			let la_B: la_object_t = la_identity_matrix(la_count_t(count), la_scalar_type_t(LA_SCALAR_TYPE_FLOAT), attr)
			let la_C: la_object_t = la_difference(la_A, la_B)
			XCTAssert(la_status(la_C)==0)
			XCTAssert(la_norm_as_float(la_C, norm)<1e-3)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianμD() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let width: Int = 64
			let refer: Int = 128
			let distributor: Distributor = try Gauss.factory()(device)(width)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let jμ: MTLBuffer = device.makeBuffer(length: width*width*refer*MemoryLayout<Float>.size, options: .storageModeShared)
			let jσ: MTLBuffer = device.makeBuffer(length: width*width*refer*MemoryLayout<Float>.size, options: .storageModeShared)
			let d: MTLBuffer = device.makeBuffer(length: width*MemoryLayout<Float>.size, options: .storageModeShared)
			let p: MTLBuffer = device.makeBuffer(length: width*width*refer*MemoryLayout<Float>.size, options: .storageModeShared)
			shuffle(array: [d, p])
			do {
				//measure {
				do {
					let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
					distributor.clear(commandBuffer: commandBuffer, jμ: jμ, jσ: jσ)
					distributor.jacobian(commandBuffer: commandBuffer, jμ: jμ, d: d, p: p, refer: refer)
					commandBuffer.commit()
					commandBuffer.waitUntilCompleted()
				}
				let la_d: la_object_t = la_diagonal_matrix_from_vector(la_matrix_from_float_buffer_nocopy(d.ref, la_count_t(width), la_count_t(1), la_count_t(1), hint, nil, attr), la_index_t(0))
				let la_p: la_object_t = la_matrix_from_float_buffer_nocopy(p.ref, la_count_t(width), la_count_t(width*refer), la_count_t(width*refer), hint, nil, attr)
				let la_A: la_object_t = la_matrix_from_float_buffer_nocopy(jμ.ref, la_count_t(width), la_count_t(width*refer), la_count_t(width*refer), hint, nil, attr)
				let la_B: la_object_t = la_matrix_product(la_d, la_p)
				let la_C: la_object_t = la_difference(la_A, la_B)
				XCTAssert(la_status(la_C) == 0)
				XCTAssert(la_norm_as_float(la_C, norm)<1e-9)
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianμX() {
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
			let la_C: la_object_t = la_difference(la_A, la_B)
			XCTAssert(la_status(la_C)==0)
			XCTAssert(la_norm_as_float(la_C, norm)<1e-3)
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
extension DistributorTests {
	func testJacobianσA() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let width: Int = 64
			let refer: Int = 128
			let distributor: Distributor = try Gauss.factory()(device)(width)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			
			let jσ: MTLBuffer = device.makeBuffer(length: width * width * refer * MemoryLayout<Float>.size, options: .storageModeShared)
			let jμ: MTLBuffer = device.makeBuffer(length: width * width * refer * MemoryLayout<Float>.size, options: .storageModeShared)
			
			let w: MTLBuffer = device.makeBuffer(length: width * refer * MemoryLayout<Float>.size, options: .storageModeShared)
			
			let χ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let μ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let σ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let λ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			
			let x: MTLBuffer = device.makeBuffer(length: refer * MemoryLayout<Float>.size, options: .storageModeShared)
			shuffle(array: [w, χ, μ, σ, x])
			
			let rows: la_count_t = la_count_t(width)
			let cols: la_count_t = la_count_t(width*refer)
			
//			vDSP_vfill([1.0], σ.ref, 1, vDSP_Length(width))
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.reset(commandBuffer: commandBuffer)
				distributor.clear(commandBuffer: commandBuffer, jμ: jμ, jσ: jσ)
				distributor.collect(commandBuffer: commandBuffer, χ: χ, μ: μ, σ: σ)
				commandBuffer.commit()
			}
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.jacobian(commandBuffer: commandBuffer, jσ: jσ, w: w, x: x, refer: refer)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			vvrecf(λ.ref, σ.ref, [Int32(width)])
			
			let la_λ: la_object_t = la_matrix_from_float_buffer_nocopy(λ.ref, la_count_t(width), 1, 1, hint, nil, attr)
			let la_w: la_object_t = la_matrix_from_float_buffer_nocopy(w.ref, la_count_t(width), la_count_t(refer), la_count_t(refer), hint, nil, attr)
			let la_x: la_object_t = la_matrix_from_float_buffer_nocopy(x.ref, la_count_t(refer), 1, 1, hint, nil, attr)

			let la_y: la_object_t = la_elementwise_product(la_w, la_outer_product(la_λ, la_elementwise_product(la_x, la_x)))
			
			la_matrix_to_float_buffer(jμ.ref, la_count_t((width+1)*refer), la_y)
			
			let la_A: la_object_t = la_matrix_from_float_buffer_nocopy(jσ.ref, rows, cols, cols, hint, nil, attr)
			let la_B: la_object_t = la_matrix_from_float_buffer_nocopy(jμ.ref, rows, cols, cols, hint, nil, attr)
			let la_C: la_object_t = la_difference(la_A, la_B)
			
			XCTAssert(la_status(la_C) == 0)
			
			let rmse: Float = la_norm_as_float(la_C, norm)
			
			XCTAssert( rmse < 1e-3 && !rmse.isNaN )
			
			//try la_A.write(to: URL(fileURLWithPath: "/tmp/a.raw"))
			//try la_B.write(to: URL(fileURLWithPath: "/tmp/b.raw"))
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testJacobianσC() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let width: Int = 1001
			
			let distributor: Distributor = try Gauss.factory()(device)(width)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			
			let jμ: MTLBuffer = device.makeBuffer(length: width*width*MemoryLayout<Float>.size, options: .storageModeShared)
			let jσ: MTLBuffer = device.makeBuffer(length: width*width*MemoryLayout<Float>.size, options: .storageModeShared)
			
			let χ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let μ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let σ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let λ: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			
			let c: MTLBuffer = device.makeBuffer(length: width*MemoryLayout<Float>.size, options: .storageModeShared)
			shuffle(array: [χ, μ, σ, c])
			
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.reset(commandBuffer: commandBuffer)
				distributor.clear(commandBuffer: commandBuffer, jμ: jμ, jσ: jσ)
				distributor.collect(commandBuffer: commandBuffer, χ: χ, μ: μ, σ: σ)
				commandBuffer.commit()
			}
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.jacobian(commandBuffer: commandBuffer, jσ: jσ, c: c)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			vvrecf(λ.ref, σ.ref, [Int32(width)])
			
			let la_λ: la_object_t = la_matrix_from_float_buffer_nocopy(λ.ref, la_count_t(width), 1, 1, hint, nil, attr)
			let la_c: la_object_t = la_matrix_from_float_buffer_nocopy(c.ref, la_count_t(width), 1, 1, hint, nil, attr)
			
			let la_A: la_object_t = la_matrix_from_float_buffer_nocopy(jσ.ref, la_count_t(width), la_count_t(width), la_count_t(width), hint, nil, attr)
			let la_B: la_object_t = la_diagonal_matrix_from_vector(la_elementwise_product(la_λ, la_c), 0)
			let la_C: la_object_t = la_difference(la_A, la_B)
			
			XCTAssert(la_status(la_C) == 0)
			
			let rmse: Float = la_norm_as_float(la_C, norm)
			
			XCTAssert( rmse < 1e-3 && !rmse.isNaN )
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	/*
	func testJacobianUD() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let width: Int = 64
			let refer: Int = 128
			let distributor: Distributor = try Gauss.factory()(device)(width)
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let jμ: MTLBuffer = device.makeBuffer(length: width * width * refer * MemoryLayout<Float>.size, options: .storageModeShared)
			let jσ: MTLBuffer = device.makeBuffer(length: width * width * refer * MemoryLayout<Float>.size, options: .storageModeShared)
			let b: MTLBuffer = device.makeBuffer(length: width * width * MemoryLayout<Float>.size, options: .storageModeShared)
			let df: MTLBuffer = device.makeBuffer(length: width * MemoryLayout<Float>.size, options: .storageModeShared)
			let p: MTLBuffer = device.makeBuffer(length: width * width * refer * MemoryLayout<Float>.size, options: .storageModeShared)
			shuffle(array: [b, df, p])
			do {
				//measure {
				do {
					let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
					distributor.clear(commandBuffer: commandBuffer, jμ: jμ, jσ: jσ)
					distributor.jacobian(commandBuffer: commandBuffer, jμ: jμ, b: b, df: df, p: p, refer: refer) //ub
					commandBuffer.commit()
					commandBuffer.waitUntilCompleted()
				}
				let rows: la_count_t = la_count_t(width)
				let cols: la_count_t = la_count_t(width*refer)
				let la_jμ: la_object_t = la_matrix_from_float_buffer_nocopy(jμ.ref, rows, cols, cols, hint, nil, attr)
				let la_b: la_object_t = la_matrix_from_float_buffer_nocopy(b.ref, rows, rows, rows, hint, nil, attr)
				let la_df: la_object_t = la_diagonal_matrix_from_vector(la_matrix_from_float_buffer_nocopy(df.ref, rows, 1, 1, hint, nil, attr), 0)
				let la_p: la_object_t = la_matrix_from_float_buffer_nocopy(p.ref, rows, cols, cols, hint, nil, attr)
				let la_A: la_object_t = la_jμ
				let la_B: la_object_t = la_matrix_product(la_b, la_matrix_product(la_df, la_p))
				let la_C: la_object_t = la_difference(la_A, la_B)
				XCTAssert(la_status(la_C) == 0)
				XCTAssert(la_norm_as_float(la_C, norm)<1e-3)
				try la_A.write(to: URL(fileURLWithPath: "/tmp/a.raw"))
				try la_B.write(to: URL(fileURLWithPath: "/tmp/b.raw"))
			}
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
			let la_C: la_object_t = la_difference(la_A, la_B)
			XCTAssert(la_status(la_C)==0)
			XCTAssert(la_norm_as_float(la_C, norm)<1e-3)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	*/
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
*/
