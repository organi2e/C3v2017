//
//  GaussDistributorTests.swift
//  macOS
//
//  Created by Kota Nakano on 2/18/17.
//
//

import XCTest
import simd

import Accelerate
import Metal

import Distributor

private let hint: la_hint_t = la_hint_t(LA_NO_HINT)
private let attr: la_attribute_t = la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING)
private let norm: la_norm_t = la_norm_t(LA_L2_NORM)

class GaussDistributorTests: XCTestCase {
	func uniform(count: Int, α: Float = 0, β: Float = 1) -> Array<Float> {
		var result: Array<Float> = Array<Float>(repeating: 0, count: count)
		arc4random_buf(&result, result.count*MemoryLayout<Float>.size)
		vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(result)), 1, &result, 1, vDSP_Length(count))
		vDSP_vsdiv(result, 1, [Float(UInt32.max)], &result, 1, vDSP_Length(count))
		vDSP_vsmsa(result, 1, [β-α], [α], &result, 1, vDSP_Length(count))
		return result
	}
	func shuffle(count: Int) -> Array<Float> {
		var result: Array<Float> = Array<Float>(repeating: 0, count: count)
		arc4random_buf(&result, result.count*MemoryLayout<Float>.size)
		vDSP_vflt32(UnsafePointer<Int32>(OpaquePointer(result)), 1, &result, 1, vDSP_Length(count))
		vDSP_vsdiv(result, 1, [Float(Int32.max)], &result, 1, vDSP_Length(count))
		return result
	}
}
extension GaussDistributorTests {
	func testActivate() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: Int = 16 + Int(arc4random_uniform(240))
		let buf_VU: Array<Float> = shuffle(count: count)
		let buf_VS: Array<Float> = shuffle(count: count)
		let buf_YP: Array<Float> = zip(buf_VU, buf_VS).map { 0.5+0.5*erf($0.0/$0.1/sqrt(Float(2.0))) }
		let buf_Yχ: Array<Float> = shuffle(count: count)
		let la_YP: la_object_t = la_matrix_from_float_buffer(buf_YP, la_count_t(count), la_count_t(1), la_count_t(1), hint, attr)
		let la_Yχ: la_object_t = la_matrix_from_float_buffer(buf_Yχ, la_count_t(count), la_count_t(1), la_count_t(1), hint, attr)
		let y: (χ: MTLBuffer, p: MTLBuffer) = (
			χ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: []),
			p: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: [])
		)
		let v: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_VU, length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_VS, length: count*MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.activate(commandBuffer: commandBuffer, y: y, v: v, count: count)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		let rmse: (χ: Float, p: Float) = (
			χ: la_norm_as_float(la_difference(la_Yχ, y.χ.matrix(rows: count, cols: 1)), norm),
			p: la_norm_as_float(la_difference(la_YP, y.p.matrix(rows: count, cols: 1)), norm)
		)
		XCTAssert( ( rmse.p == 0 || rmse.p.isNormal ) && rmse.p < 1e-3 )
	}
	func testCollect() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: Int = 16 + Int(arc4random_uniform(UInt32(48)))
		let v: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: [])
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count), length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: uniform(count: count), length: count*MemoryLayout<Float>.size, options: [])
		)
		let buf_σ: Array<Float> = Array<Float>(repeating: 0, count: count)
		vvsqrtf(UnsafeMutablePointer<Float>(mutating: buf_σ), UnsafePointer<Float>(OpaquePointer(Σ.σ.contents())), [Int32(count)])
		let la_vμ: la_object_t = Σ.μ.matrix(rows: count, cols: 1)
		let la_vσ: la_object_t = la_matrix_from_float_buffer(buf_σ, la_count_t(count), 1, 1, hint, attr)
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.collect(commandBuffer: commandBuffer, v: v, Σ: Σ, count: count)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		
		let Δvμ: la_object_t = la_difference(la_vμ, v.μ.matrix(rows: count, cols: 1))
		let Δvσ: la_object_t = la_difference(la_vσ, v.σ.matrix(rows: count, cols: 1))
		
		XCTAssert(!Δvμ.hasErr)
		XCTAssert(!Δvσ.hasErr)
		
		let rmseμ: Float = la_norm_as_float(Δvμ, norm)
		let rmseσ: Float = la_norm_as_float(Δvσ, norm)
		
		XCTAssert(!rmseμ.isNaN && rmseμ < 1e-3)
		XCTAssert(!rmseσ.isNaN && rmseσ < 1e-3)
		

	}
	func testCollectW() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let M: Int = 16 + Int(arc4random_uniform(240))
		let N: Int = 16 + Int(arc4random_uniform(240))
		let buf_X: Array<Float> = shuffle(count: N)
		let buf_WU: Array<Float> = shuffle(count: M*N)
		let buf_WS: Array<Float> = shuffle(count: M*N)
		let buf_YU: Array<Float> = shuffle(count: M)
		let buf_YS: Array<Float> = shuffle(count: M)
		let la_X: la_object_t = la_matrix_from_float_buffer(buf_X, la_count_t(N), 1, 1, hint, attr)
		let la_WU: la_object_t = la_matrix_from_float_buffer(buf_WU, la_count_t(M), la_count_t(N), la_count_t(N), hint, attr)
		let la_WS: la_object_t = la_matrix_from_float_buffer(buf_WS, la_count_t(M), la_count_t(N), la_count_t(N), hint, attr)
		let la_YU: la_object_t = la_sum(la_matrix_from_float_buffer(buf_YU, la_count_t(M), 1, 1, hint, attr), la_matrix_product(la_WU, la_X))
		let la_YS: la_object_t = la_sum(la_matrix_from_float_buffer(buf_YS, la_count_t(M), 1, 1, hint, attr), la_matrix_product(la_elementwise_product(la_WS, la_WS), la_elementwise_product(la_X, la_X)))
		let x: MTLBuffer = device.makeBuffer(bytes: buf_X, length: N*MemoryLayout<Float>.size, options: [])
		let w: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_WU, length: M*N*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_WS, length: M*N*MemoryLayout<Float>.size, options: [])
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_YU, length: M*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_YS, length: M*MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.collect(commandBuffer: commandBuffer, Σ: Σ, w: w, x: x, count: (rows: M, cols: N))
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		let rmse: (μ: Float, σ: Float) = (
			μ: la_norm_as_float(la_difference(la_YU, Σ.μ.matrix(rows: M, cols: 1)), norm),
			σ: la_norm_as_float(la_difference(la_YS, Σ.σ.matrix(rows: M, cols: 1)), norm)
		)
		XCTAssert( ( rmse.μ == 0 || rmse.μ.isNormal ) && rmse.μ < 1e-3 )
		XCTAssert( ( rmse.σ == 0 || rmse.σ.isNormal ) && rmse.σ < 1e-3 )
	}
	func testCollectC() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: Int = 16 + Int(arc4random_uniform(240))
		let buf_CU: Array<Float> = shuffle(count: count)
		let buf_CS: Array<Float> = shuffle(count: count)
		let buf_YU: Array<Float> = shuffle(count: count)
		let buf_YS: Array<Float> = shuffle(count: count)
		let la_CU: la_object_t = la_matrix_from_float_buffer(buf_CU, la_count_t(count), la_count_t(1), la_count_t(1), hint, attr)
		let la_CS: la_object_t = la_matrix_from_float_buffer(buf_CS, la_count_t(count), la_count_t(1), la_count_t(1), hint, attr)
		let la_YU: la_object_t = la_sum(la_matrix_from_float_buffer(buf_YU, la_count_t(count), 1, 1, hint, attr), la_CU)
		let la_YS: la_object_t = la_sum(la_matrix_from_float_buffer(buf_YS, la_count_t(count), 1, 1, hint, attr), la_elementwise_product(la_CS, la_CS));
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_YU, length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_YS, length: count*MemoryLayout<Float>.size, options: [])
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_CU, length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_CS, length: count*MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.collect(commandBuffer: commandBuffer, Σ: Σ, c: c, count: count)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		let rmse: (μ: Float, σ: Float) = (
			μ: la_norm_as_float(la_difference(la_YU, Σ.μ.matrix(rows: count, cols: 1)), norm),
			σ: la_norm_as_float(la_difference(la_YS, Σ.σ.matrix(rows: count, cols: 1)), norm)
		)
		XCTAssert( ( rmse.μ == 0 || rmse.μ.isNormal ) && rmse.μ < 1e-3 )
		XCTAssert( ( rmse.σ == 0 || rmse.σ.isNormal ) && rmse.σ < 1e-3 )
	}
	func testCollectD() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let M: Int = 16 + Int(arc4random_uniform(240))
		let buf_VU: Array<Float> = shuffle(count: M)
		let buf_VS: Array<Float> = shuffle(count: M)
		let buf_D: Array<Float> = shuffle(count: M)
		let buf_YU: Array<Float> = shuffle(count: M)
		let buf_YS: Array<Float> = shuffle(count: M)
		let la_VU: la_object_t = la_matrix_from_float_buffer(buf_VU, la_count_t(M), 1, 1, hint, attr)
		let la_VS: la_object_t = la_matrix_from_float_buffer(buf_VS, la_count_t(M), 1, 1, hint, attr)
		let la_D: la_object_t = la_matrix_from_float_buffer(buf_D, la_count_t(M), 1, 1, hint, attr)
		let la_YU: la_object_t = la_sum(la_matrix_from_float_buffer(buf_YU, la_count_t(M), 1, 1, hint, attr), la_elementwise_product(la_D, la_VU))
		let la_YS: la_object_t = la_sum(la_matrix_from_float_buffer(buf_YS, la_count_t(M), 1, 1, hint, attr), la_elementwise_product(la_elementwise_product(la_D, la_D), la_elementwise_product(la_VS, la_VS)))
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_YU, length: M*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_YS, length: M*MemoryLayout<Float>.size, options: [])
		)
		let d: MTLBuffer = device.makeBuffer(bytes: buf_D, length: M*MemoryLayout<Float>.size, options: [])
		let v: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_VU, length: M*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_VS, length: M*MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.collect(commandBuffer: commandBuffer, Σ: Σ, d: d, v: v, count: M)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		let rmse: (μ: Float, σ: Float) = (
			μ: la_norm_as_float(la_difference(la_YU, Σ.μ.matrix(rows: M, cols: 1)), norm),
			σ: la_norm_as_float(la_difference(la_YS, Σ.σ.matrix(rows: M, cols: 1)), norm)
		)
		XCTAssert( ( rmse.μ == 0 || rmse.μ.isNormal ) && rmse.μ < 1e-3 )
		XCTAssert( ( rmse.σ == 0 || rmse.σ.isNormal ) && rmse.σ < 1e-3 )
	}
}
extension GaussDistributorTests {
	func testDerivate() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: Int = 16 + Int(arc4random_uniform(240))
		let buf_VU: Array<Float> = shuffle(count: count)
		let buf_VS: Array<Float> = shuffle(count: count)
		let buf_JU: Array<Float> = zip(buf_VU, buf_VS).map {  exp(-0.5*$0.0*$0.0/$0.1/$0.1)/$0.1*rsqrt(2.0*Float(M_PI)) }
		let buf_JS: Array<Float> = zip(buf_VU, buf_VS).map { -exp(-0.5*$0.0*$0.0/$0.1/$0.1)/$0.1*rsqrt(2.0*Float(M_PI))*$0.0/$0.1 }
		let buf_YP: Array<Float> = zip(buf_VU, buf_VS).map { 0.5+0.5*erf($0.0/$0.1/sqrt(Float(2.0))) }
		let buf_YΔ: Array<Float> = shuffle(count: count)
		let _: la_object_t = la_matrix_from_float_buffer(buf_YP, la_count_t(count), la_count_t(1), la_count_t(1), hint, attr)
		let la_YΔ: la_object_t = la_matrix_from_float_buffer(buf_YΔ, la_count_t(count), la_count_t(1), la_count_t(1), hint, attr)
		let la_GU: la_object_t = la_matrix_from_float_buffer(buf_JU, la_count_t(count), la_count_t(1), la_count_t(1), hint, attr)
		let la_GS: la_object_t = la_matrix_from_float_buffer(buf_JS, la_count_t(count), la_count_t(1), la_count_t(1), hint, attr)
		let la_Δμ: la_object_t = la_elementwise_product(la_GU, la_YΔ)
		let la_Δσ: la_object_t = la_elementwise_product(la_GS, la_YΔ)
		let Δ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: [])
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: [])
		)
		let y: (Δ: MTLBuffer, p: MTLBuffer) = (
			Δ: device.makeBuffer(bytes: buf_YΔ, length: count*MemoryLayout<Float>.size, options: []),
			p: device.makeBuffer(bytes: buf_YP, length: count*MemoryLayout<Float>.size, options: [])
		)
		let v: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_VU, length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_VS, length: count*MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.derivate(commandBuffer: commandBuffer, Δ: Δ, g: g, y: y, v: v, count: count)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		let rmse: (Δμ: Float, Δσ: Float, gμ: Float, gσ: Float) = (
			Δμ: la_norm_as_float(la_difference(la_Δμ, Δ.μ.matrix(rows: count, cols: 1)), norm),
			Δσ: la_norm_as_float(la_difference(la_Δσ, Δ.σ.matrix(rows: count, cols: 1)), norm),
			gμ: la_norm_as_float(la_difference(la_GU, g.μ.matrix(rows: count, cols: 1)), norm),
			gσ: la_norm_as_float(la_difference(la_GS, g.σ.matrix(rows: count, cols: 1)), norm)
		)
		XCTAssert( ( rmse.Δμ == 0 || rmse.Δμ.isNormal ) && rmse.Δμ < 1e-3 )
		XCTAssert( ( rmse.Δσ == 0 || rmse.Δσ.isNormal ) && rmse.Δσ < 1e-3 )
		XCTAssert( ( rmse.gμ == 0 || rmse.gμ.isNormal ) && rmse.gμ < 1e-3 )
		XCTAssert( ( rmse.gσ == 0 || rmse.gσ.isNormal ) && rmse.gσ < 1e-3 )
	}
	func testDelta() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: (rows: Int, cols: Int) = (rows: 16 + Int(arc4random_uniform(UInt32(64))),
		                                     cols: 16 + Int(arc4random_uniform(UInt32(64))))
		let Δ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows), length: count.rows*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows), length: count.rows*MemoryLayout<Float>.size, options: [])
		)
		let la_Δμ: la_object_t = la_matrix_product(la_diagonal_matrix_from_vector(g.μ.matrix(rows: count.rows, cols: 1), 0), j.μ.matrix(rows: count.rows, cols: count.cols))
		let la_Δσ: la_object_t = la_matrix_product(la_diagonal_matrix_from_vector(g.σ.matrix(rows: count.rows, cols: 1), 0), j.σ.matrix(rows: count.rows, cols: count.cols))
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.delta(commandBuffer: commandBuffer, Δ: Δ, j: j, g: g, count: count, rtrl: false)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		
		let δΔμ: la_object_t = la_difference(la_Δμ, Δ.μ.matrix(rows: count.rows, cols: count.cols))
		let δΔσ: la_object_t = la_difference(la_Δσ, Δ.σ.matrix(rows: count.rows, cols: count.cols))
		
		XCTAssert(!δΔμ.hasErr)
		XCTAssert(!δΔσ.hasErr)
		
		let rmseμ: Float = la_norm_as_float(δΔμ, norm)
		let rmseσ: Float = la_norm_as_float(δΔσ, norm)
		
		XCTAssert(!rmseμ.isNaN)
		XCTAssert(rmseμ < 1e-7)
		XCTAssert(!rmseσ.isNaN)
		XCTAssert(rmseσ < 1e-7)
	}
	func testDelta_RTRL() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: (rows: Int, cols: Int) = (rows: 16 + Int(arc4random_uniform(UInt32(48))), cols: 16 + Int(arc4random_uniform(UInt32(48))))
		let Δ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows*count.rows*count.cols), length: count.rows*count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: uniform(count: count.rows*count.rows*count.cols), length: count.rows*count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows), length: count.rows*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows), length: count.rows*MemoryLayout<Float>.size, options: [])
		)
		let la_Δμ: la_object_t = la_matrix_product(la_transpose(j.μ.matrix(rows: count.rows, cols: count.rows*count.cols)), g.μ.matrix(rows: count.rows, cols: 1))
		let la_Δσ: la_object_t = la_matrix_product(la_transpose(j.σ.matrix(rows: count.rows, cols: count.rows*count.cols)), g.σ.matrix(rows: count.rows, cols: 1))
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.delta(commandBuffer: commandBuffer, Δ: Δ, j: j, g: g, count: count, rtrl: true)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		
		let δΔμ: la_object_t = la_difference(la_Δμ, Δ.μ.matrix(rows: count.rows*count.cols, cols: 1))
		let δΔσ: la_object_t = la_difference(la_Δσ, Δ.σ.matrix(rows: count.rows*count.cols, cols: 1))
		
		XCTAssert(!δΔμ.hasErr)
		XCTAssert(!δΔσ.hasErr)
		
		let rmseμ: Float = la_norm_as_float(δΔμ, norm)
		let rmseσ: Float = la_norm_as_float(δΔσ, norm)
		
		XCTAssert(!rmseμ.isNaN && rmseμ < 1e-4)
		XCTAssert(!rmseσ.isNaN && rmseσ < 1e-4)
	}
	func testDelta_X() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: (rows: Int, cols: Int) = (rows: 64 + Int(arc4random_uniform(UInt32(128))),
		                                     cols: 64 + Int(arc4random_uniform(UInt32(128))))
		let Δ: MTLBuffer = device.makeBuffer(length: count.cols*MemoryLayout<Float>.size, options: [])
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows), length: count.rows*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows), length: count.rows*MemoryLayout<Float>.size, options: [])
		)
		let la_Δμ: la_object_t = la_matrix_product(la_transpose(j.μ.matrix(rows: count.rows, cols: count.cols)), g.μ.matrix(rows: count.rows, cols: 1))
		let la_Δσ: la_object_t = la_matrix_product(la_transpose(j.σ.matrix(rows: count.rows, cols: count.cols)), g.σ.matrix(rows: count.rows, cols: 1))
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.delta(commandBuffer: commandBuffer, Δ: Δ, j: j, g: g, count: count)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		
		let δ: la_object_t = la_difference(la_sum(la_Δμ, la_Δσ), Δ.matrix(rows: count.cols, cols: 1))
		
		XCTAssert(!δ.hasErr)
		
		let rmse: Float = la_norm_as_float(δ, norm)
		
		XCTAssert(!rmse.isNaN && rmse < 1e-5)
	
		print(la_Δμ.array)
		print(Array(Δ.buffer))
		
	}
	func testJacobian() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: (rows: Int, cols: Int) = (rows: 16 + Int(arc4random_uniform(UInt32(48))), cols: 16 + Int(arc4random_uniform(UInt32(48))))
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let v: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows), length: count.rows*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows), length: count.rows*MemoryLayout<Float>.size, options: [])
		)
		let buf_λ: Array<Float> = Array<Float>(repeating: 0, count: count.rows)
		vvrecf(UnsafeMutablePointer<Float>(mutating: buf_λ), UnsafePointer<Float>(OpaquePointer(v.σ.contents())), [Int32(count.rows)])
		let la_λ: la_object_t = la_diagonal_matrix_from_vector(la_matrix_from_float_buffer(buf_λ, la_count_t(count.rows), 1, 1, hint, attr), 0)
		let la_jμ: la_object_t = Σ.μ.matrix(rows: count.rows, cols: count.cols)
		let la_jσ: la_object_t = la_matrix_product(la_λ, Σ.σ.matrix(rows: count.rows, cols: count.cols))
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.jacobian(commandBuffer: commandBuffer, j: j, v: v, Σ: Σ, count: count, rtrl: false)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		
		let Δjμ: la_object_t = la_difference(la_jμ, j.μ.matrix(rows: count.rows, cols: count.cols))
		let Δjσ: la_object_t = la_difference(la_jσ, j.σ.matrix(rows: count.rows, cols: count.cols))
		
		XCTAssert(!Δjμ.hasErr)
		XCTAssert(!Δjσ.hasErr)
		
		let rmseμ: Float = la_norm_as_float(Δjμ, norm)
		let rmseσ: Float = la_norm_as_float(Δjσ, norm)
		
		XCTAssert(!rmseμ.isNaN && rmseμ < 1e-3)
		XCTAssert(!rmseσ.isNaN && rmseσ < 1e-3)
		
	}
	func testJacobianA() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: (rows: Int, cols: Int) = (rows: 16 + Int(arc4random_uniform(UInt32(48))), cols: 16 + Int(arc4random_uniform(UInt32(48))))
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let A: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let x: MTLBuffer = device.makeBuffer(bytes: shuffle(count: count.cols), length: count.cols*MemoryLayout<Float>.size, options: [])
		let la_Aσ: la_object_t = A.σ.matrix(rows: count.rows, cols: count.cols)
		let la_x: la_object_t = x.matrix(rows: count.cols, cols: 1)
		let la_one: la_object_t = la_vector_from_splat(la_splat_from_float(1, attr), la_count_t(count.rows))
		let la_jμ: la_object_t = la_outer_product(la_one, la_x)
		let la_jσ: la_object_t = la_elementwise_product(la_outer_product(la_one, la_elementwise_product(la_x, la_x)), la_Aσ)
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, a: A, x: x, count: count, rtrl: false)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		
		let Δjμ: la_object_t = la_difference(la_jμ, Σ.μ.matrix(rows: count.rows, cols: count.cols))
		let Δjσ: la_object_t = la_difference(la_jσ, Σ.σ.matrix(rows: count.rows, cols: count.cols))
		
		XCTAssert(!Δjμ.hasErr)
		XCTAssert(!Δjσ.hasErr)
		
		let rmseμ: Float = la_norm_as_float(Δjμ, norm)
		let rmseσ: Float = la_norm_as_float(Δjσ, norm)
		
		XCTAssert(!rmseμ.isNaN && rmseμ < 1e-16)
		XCTAssert(!rmseσ.isNaN && rmseσ < 1e-16)
	}
	func testJacobianA_RTRL() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: (rows: Int, cols: Int) = (rows: 16 + Int(arc4random_uniform(UInt32(48))), cols: 16 + Int(arc4random_uniform(UInt32(48))))
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count.rows*count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count.rows*count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let a: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let x: MTLBuffer = device.makeBuffer(bytes: shuffle(count: count.cols), length: count.cols*MemoryLayout<Float>.size, options: [])
		let la_x: la_object_t = x.matrix(rows: 1, cols: count.cols)
		
		let dst_jμ: la_object_t = la_outer_product(la_vector_from_splat(la_splat_from_float(1.0, attr), la_count_t(count.rows)), la_x).eye
		let dst_jσ: la_object_t = la_matrix_product(a.σ.matrix(rows: count.rows, cols: count.cols), la_diagonal_matrix_from_vector(la_elementwise_product(la_x, la_x), 0)).eye
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.clear(commandBuffer: commandBuffer, μ: Σ.μ, σ: Σ.σ)
			distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, a: a, x: x, count: count, rtrl: true)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		let Σjμ: la_object_t = la_difference(dst_jμ, Σ.μ.matrix(rows: count.rows, cols: count.rows*count.cols))
		let Σjσ: la_object_t = la_difference(dst_jσ, Σ.σ.matrix(rows: count.rows, cols: count.rows*count.cols))
		
		XCTAssert(!Σjμ.hasErr)
		XCTAssert(!Σjσ.hasErr)
		
		let rmseμ: Float = la_norm_as_float(Σjμ, norm)
		let rmseσ: Float = la_norm_as_float(Σjσ, norm)
		
		XCTAssert(!rmseμ.isNaN && rmseμ < 1e-8)
		XCTAssert(!rmseσ.isNaN && rmseσ < 1e-8)
	}
	func testJacobianB_RTRL() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let refs: Int = 64 + Int(arc4random_uniform(UInt32(64)))
		let count: (rows: Int, cols: Int) = (rows: refs, cols: refs)
		let buf_bμ: Array<Float> = shuffle(count: count.rows*count.rows)
		let buf_bσ: Array<Float> = shuffle(count: count.rows*count.rows)
		let buf_Y: Array<Float> = shuffle(count: count.rows)
		let buf_gμ: Array<Float> = shuffle(count: count.rows)
		let buf_gσ: Array<Float> = shuffle(count: count.rows)
		let buf_jμ: Array<Float> = shuffle(count: count.rows*count.rows*count.cols)
		let buf_jσ: Array<Float> = shuffle(count: count.rows*count.rows*count.cols)
		let la_bμ: la_object_t = la_matrix_from_float_buffer(buf_bμ, la_count_t(count.rows), la_count_t(count.cols), la_count_t(count.cols), hint, attr)
		let la_bσ: la_object_t = la_matrix_from_float_buffer(buf_bσ, la_count_t(count.rows), la_count_t(count.cols), la_count_t(count.cols), hint, attr)
		let la_Y: la_object_t = la_diagonal_matrix_from_vector(la_matrix_from_float_buffer(buf_Y, la_count_t(count.rows), la_count_t(1), la_count_t(1), hint, attr), 0)
		let la_gμ: la_object_t = la_diagonal_matrix_from_vector(la_matrix_from_float_buffer(buf_gμ, la_count_t(count.rows), la_count_t(1), la_count_t(1), hint, attr), 0)
		let la_gσ: la_object_t = la_diagonal_matrix_from_vector(la_matrix_from_float_buffer(buf_gσ, la_count_t(count.rows), la_count_t(1), la_count_t(1), hint, attr), 0)
		let la_jμ: la_object_t = la_matrix_from_float_buffer(buf_jμ, la_count_t(count.rows), la_count_t(count.rows*count.cols), la_count_t(count.rows*count.cols), hint, attr)
		let la_jσ: la_object_t = la_matrix_from_float_buffer(buf_jσ, la_count_t(count.rows), la_count_t(count.rows*count.cols), la_count_t(count.rows*count.cols), hint, attr)
		let la_Σμ: la_object_t = la_matrix_product(la_bμ, la_matrix_product(la_gμ, la_jμ))
		let la_Σσ: la_object_t = la_matrix_product(la_elementwise_product(la_bσ, la_bσ), la_matrix_product(la_elementwise_product(la_Y, la_gσ), la_jσ))
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: buf_jμ.count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: buf_jσ.count*MemoryLayout<Float>.size, options: [])
		)
		let b: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_bμ, length: buf_bμ.count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_bσ, length: buf_bσ.count*MemoryLayout<Float>.size, options: [])
		)
		let y: MTLBuffer = device.makeBuffer(bytes: buf_Y, length: buf_Y.count*MemoryLayout<Float>.size, options: [])
		let g: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_gμ, length: buf_gμ.count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_gσ, length: buf_gσ.count*MemoryLayout<Float>.size, options: [])
		)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: buf_jμ, length: buf_jμ.count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: buf_jσ, length: buf_jσ.count*MemoryLayout<Float>.size, options: [])
		)
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, b: b, g: g, j: j, y: y, count: count, rtrl: true)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		let rmse: (μ: Float, σ: Float) = (
			μ: la_norm_as_float(la_difference(la_Σμ, Σ.μ.matrix(rows: count.rows, cols: count.rows*count.cols)), norm),
			σ: la_norm_as_float(la_difference(la_Σσ, Σ.σ.matrix(rows: count.rows, cols: count.rows*count.cols)), norm)
		)
		XCTAssert( !rmse.μ.isNaN && rmse.μ < 1e-3 )
		XCTAssert( !rmse.σ.isNaN && rmse.σ < 1e-3 )
	}
	func testJacobianC() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: Int = 16 + Int(arc4random_uniform(UInt32(48)))
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: [])
		)
		let C: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count), length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count), length: count*MemoryLayout<Float>.size, options: [])
		)
		let la_jμ: la_object_t = la_vector_from_splat(la_splat_from_float(1, attr), la_count_t(count))
		let la_jσ: la_object_t = C.σ.matrix(rows: count, cols: 1)
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, c: C, count: count, rtrl: false)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		
		let Δjμ: la_object_t = la_difference(la_jμ, Σ.μ.matrix(rows: count, cols: 1))
		let Δjσ: la_object_t = la_difference(la_jσ, Σ.σ.matrix(rows: count, cols: 1))
		
		XCTAssert(!Δjμ.hasErr)
		XCTAssert(!Δjσ.hasErr)
		
		let rmseμ: Float = la_norm_as_float(Δjμ, norm)
		let rmseσ: Float = la_norm_as_float(Δjσ, norm)
		
		XCTAssert(!rmseμ.isNaN && rmseμ < 1e-24)
		XCTAssert(!rmseσ.isNaN && rmseσ < 1e-24)

	}
	func testJacobianC_RTRL() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: Int = 64
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count*count), length: count*count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count*count), length: count*count*MemoryLayout<Float>.size, options: [])
		)
		let c: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count*MemoryLayout<Float>.size, options: [])
		)
		let dst_jμ: la_object_t = la_identity_matrix(la_count_t(count), la_scalar_type_t(LA_SCALAR_TYPE_FLOAT), attr)
		let dst_jσ: la_object_t = la_diagonal_matrix_from_vector(c.σ.matrix(rows: count, cols: 1), 0)
		
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			distributor.clear(commandBuffer: commandBuffer, μ: Σ.μ, σ: Σ.σ)
			distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, c: c, count: count, rtrl: true)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			XCTFail(String(describing: error))
		}
		
		let Σjμ: la_object_t = la_difference(dst_jμ, Σ.μ.matrix(rows: count, cols: count))
		let Σjσ: la_object_t = la_difference(dst_jσ, Σ.σ.matrix(rows: count, cols: count))
		
		XCTAssert(!Σjμ.hasErr)
		XCTAssert(!Σjσ.hasErr)
		
		let rmseμ: Float = la_norm_as_float(Σjμ, norm)
		let rmseσ: Float = la_norm_as_float(Σjσ, norm)
		
		XCTAssert(!rmseμ.isNaN && rmseμ < 1e-0)
		XCTAssert(!rmseσ.isNaN && rmseσ < 1e-8)
		
	}
	func testJacobianD() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: (rows: Int, cols: Int) = (rows: 64 + Int(arc4random_uniform(UInt32(64))),
		                                     cols: 64 + Int(arc4random_uniform(UInt32(64)))
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let d: MTLBuffer = device.makeBuffer(bytes: shuffle(count: count.rows), length: count.rows*MemoryLayout<Float>.size, options: .storageModeShared)
		let j: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let la_d: la_object_t = la_diagonal_matrix_from_vector(la_matrix_from_float_buffer_nocopy(d.ref, la_count_t(count.rows), la_count_t(1), la_count_t(1), hint, nil, attr), la_index_t(0))
		let la_jμ: la_object_t = la_matrix_from_float_buffer_nocopy(j.μ.ref, la_count_t(count.rows), la_count_t(count.cols), la_count_t(count.cols), hint, nil, attr)
		let la_jσ: la_object_t = la_matrix_from_float_buffer_nocopy(j.σ.ref, la_count_t(count.rows), la_count_t(count.cols), la_count_t(count.cols), hint, nil, attr)
		let la_Σμ: la_object_t = la_matrix_product(la_d, la_jμ)
		let la_Σσ: la_object_t = la_matrix_product(la_elementwise_product(la_d, la_d), la_elementwise_product(la_jσ, la_jσ))
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.clear(commandBuffer: commandBuffer, μ: Σ.μ, σ: Σ.σ)
				commandBuffer.commit()
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, d: d, j: j, count: count, rtrl: false)
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
		} catch {
			XCTFail(String(describing: error))
		}
		let rmse: (μ: Float, σ: Float) = (
			μ: la_norm_as_float(la_difference(la_Σμ, Σ.μ.matrix(rows: count.rows, cols: count.cols)), norm),
			σ: la_norm_as_float(la_difference(la_Σσ, Σ.σ.matrix(rows: count.rows, cols: count.cols)), norm)
		)
		XCTAssert( ( rmse.μ == 0 || rmse.μ.isNormal ) && rmse.μ < 1e-3 )
		XCTAssert( ( rmse.σ == 0 || rmse.σ.isNormal ) && rmse.σ < 1e-3 )
	}
	func testJacobianX() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		let queue: MTLCommandQueue = device.makeCommandQueue()
		let count: (rows: Int, cols: Int) = (rows: 64 + Int(arc4random_uniform(UInt32(64))),
		                                     cols: 64 + Int(arc4random_uniform(UInt32(64)))
		)
		let Σ: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let a: (μ: MTLBuffer, σ: MTLBuffer) = (
			μ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: []),
			σ: device.makeBuffer(bytes: shuffle(count: count.rows*count.cols), length: count.rows*count.cols*MemoryLayout<Float>.size, options: [])
		)
		let x: MTLBuffer = device.makeBuffer(bytes: shuffle(count: count.rows), length: count.cols*MemoryLayout<Float>.size, options: [])
		let la_x: la_object_t = la_diagonal_matrix_from_vector(la_matrix_from_float_buffer_nocopy(x.ref, la_count_t(count.cols), 1, 1, hint, nil, attr), 0)
		let la_aμ: la_object_t = la_matrix_from_float_buffer_nocopy(a.μ.ref, la_count_t(count.rows), la_count_t(count.cols), la_count_t(count.cols), hint, nil, attr)
		let la_aσ: la_object_t = la_matrix_from_float_buffer_nocopy(a.σ.ref, la_count_t(count.rows), la_count_t(count.cols), la_count_t(count.cols), hint, nil, attr)
		let la_Σμ: la_object_t = la_aμ
		let la_Σσ: la_object_t = la_matrix_product(la_elementwise_product(la_aσ, la_aσ), la_x)
		do {
			let distributor: Distributor = try GaussDistributor.factory()(device)
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.clear(commandBuffer: commandBuffer, μ: Σ.μ, σ: Σ.σ)
				commandBuffer.commit()
			}
			do {
				let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
				distributor.jacobian(commandBuffer: commandBuffer, Σ: Σ, x: x, a: a, count: count, rtrl: false)
				commandBuffer.commit()
				commandBuffer.waitUntilCompleted()
			}
		} catch {
			XCTFail(String(describing: error))
		}
		
		let rmse: (μ: Float, σ: Float) = (
			μ: la_norm_as_float(la_difference(la_Σμ, Σ.μ.matrix(rows: count.rows, cols: count.cols)), norm),
			σ: la_norm_as_float(la_difference(la_Σσ, Σ.σ.matrix(rows: count.rows, cols: count.cols)), norm)
		)
		XCTAssert( ( rmse.μ == 0 || rmse.μ.isNormal ) && rmse.μ < 1e-3 )
		XCTAssert( ( rmse.σ == 0 || rmse.σ.isNormal ) && rmse.σ < 1e-3 )
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
