//
//  C³Tests.swift
//  C³Tests
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Accelerate
import XCTest
import Optimizer
@testable import C3
class C3Tests: XCTestCase {
	func testProp() {
		do {
			let context: Context = try Context(optimizer: SMORMS3.factory())
			context.layout = .storageModeShared
			
			let I: Cell = try context.make(name: "I", width: uniform(8, 64), input: [])
			let H: Cell = try context.make(name: "H", width: uniform(8, 64), input: [I])
			let O: Cell = try context.make(name: "O", width: uniform(8, 64), input: [H])
			
			for _ in 0..<3 {
				
				O.collect_clear()
				I.correct_clear()
				
				O.target = uniform(O.width, 0, 1)
				I.source = uniform(I.width, 0, 1)
				
				O.collect()
				I.correct()
				
			}
			
			context.sync()
			
			do {
				let Δμ: la_object_t = O.delta.current.μ.matrix(rows: O.width, cols: 1)
				XCTAssert(0<la_norm_as_float(Δμ, norm))
				
				let x: la_object_t = H.state.current.matrix(rows: H.width, cols: 1)
				XCTAssert(0<la_norm_as_float(x, norm))
				
				let Δwμ: la_object_t = O.input.first!.μ.Δ.matrix(rows: O.width, cols: H.width)
				XCTAssert(0<la_norm_as_float(Δwμ, norm))
				
				let δ: la_object_t = la_difference(Δwμ, la_outer_product(Δμ, x))
				XCTAssert(!δ.hasErr)
				
				let rmse: Float = la_norm_as_float(δ, norm)
				XCTAssert(rmse<1e-6)
			}
			
			do {
				let Δμ: la_object_t = O.delta.current.μ.matrix(rows: O.width, cols: 1)
				XCTAssert(0<la_norm_as_float(Δμ, norm))
				
				let Δσ: la_object_t = O.delta.current.σ.matrix(rows: O.width, cols: 1)
				XCTAssert(0<la_norm_as_float(Δσ, norm))
				
				let wμ: la_object_t = O.input.first!.μ.θ.matrix(rows: O.width, cols: H.width)
				XCTAssert(0<la_norm_as_float(wμ, norm))
				
				let wσ: la_object_t = O.input.first!.σ.θ.matrix(rows: O.width, cols: H.width)
				XCTAssert(0<la_norm_as_float(wσ, norm))
				
				let jμ: la_object_t = O.input.first!.j.current.x.μ.matrix(rows: O.width, cols: H.width)
				XCTAssert(0<la_norm_as_float(jμ, norm))
				
				let jσ: la_object_t = O.input.first!.j.current.x.σ.matrix(rows: O.width, cols: H.width)
				XCTAssert(0<la_norm_as_float(jσ, norm))
				
				let x: la_object_t = H.state.current.matrix(rows: H.width, cols: 1)
				XCTAssert(0<la_norm_as_float(x, norm))
				
				let vσ: la_object_t = O.value.current.σ.matrix(rows: O.width, cols: 1)
				
				XCTAssert(la_norm_as_float(la_difference(jμ, wμ), norm)<1e-3)
				XCTAssert(la_norm_as_float(la_difference(la_matrix_product(la_diagonal_matrix_from_vector(vσ, 0), jσ),
				                                         la_matrix_product(la_elementwise_product(wσ, wσ), la_diagonal_matrix_from_vector(x, 0))), norm)<1e-3)
				
				let Δx: la_object_t = H.study.current.matrix(rows: H.width, cols: 1)
				XCTAssert(0<la_norm_as_float(Δx, norm))
				
				let δ: la_object_t = la_difference(Δx, la_sum(la_matrix_product(la_transpose(jμ), Δμ), la_matrix_product(la_transpose(jσ), Δσ)))
				XCTAssert(!δ.hasErr)
				
				let rmse: Float = la_norm_as_float(δ, norm)
				XCTAssert(rmse<1e-3)
			}
			
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	func testContext() {
		do {
//			let context: Context = try Context(optimizer: Adam.factory(α: 1e-3))
			let context: Context = try Context(optimizer: SMORMS3.factory(α: 1e-3))
			//let context: Context = try Context(optimizer: SGD.factory(η: 1e-1))
			//			context.layout = .storageModeShared
			
			let IS: [[Float]] = [[0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0]]
			let OS: [[Float]] = [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]]
			let I: Cell = try context.make(name: "I", width: 4, input: [])
			let H: Cell = try context.make(name: "H", width:256, input: [I])
			let G: Cell = try context.make(name: "G", width:256, input: [H])
			//let F: Cell = try context.make(name: "F", width:512, input: [G])
			//let E: Cell = try context.make(name: "E", width:512, input: [F])
			//let D: Cell = try context.make(name: "D", width:512, input: [E])
			//let C: Cell = try context.make(name: "C", width:256, input: [D])
			//let B: Cell = try context.make(name: "B", width:128, input: [C])
			//let A: Cell = try context.make(name: "A", width:64, input: [B])
			let O: Cell = try context.make(name: "O", width: 4, input: [G])
			
			for k in 0..<2000 {
				print(k, terminator: "\r")
				
				O.collect_clear()
				I.correct_clear()
				
				I.source = IS[k%4]
				O.target = OS[k%4]
				
				O.collect()
				I.correct()
				
			}
			
			context.sync()
			try context.save()
			
			//			print("μ", Array(O.delta.current.μ.buffer))
			//			print("Δ", Array(O.input.first!.μ.Δ.buffer))
			//			print("θ", Array(O.input.first!.μ.θ.buffer))
			//			print("φ", Array(O.input.first!.μ.φ.buffer))
			
			for k in 0..<4 {
				O.collect_clear()
				I.correct_clear()
				I.source = IS[k]
				O.target = OS[k]
				O.collect()
				//				I.correct()
				//				context.sync()
				print(O.target, O.source)
				//				print("μ", Array(O.delta.current.μ.buffer))
				//				print("Δ", Array(O.input.first!.μ.Δ.buffer))
				//				print("θ", Array(O.input.first!.μ.θ.buffer))
				//				print("φ", Array(O.input.first!.μ.φ.buffer))
			}
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
extension C3Tests {
	func uniform(_ count: Int, _ α: Float, _ β: Float) -> Array<Float> {
		var result: Array<Float> = Array<Float>(repeating: 0, count: count)
		arc4random_buf(&result, result.count*MemoryLayout<Float>.size)
		vDSP_vfltu32(UnsafePointer<UInt32>(OpaquePointer(result)), 1, &result, 1, vDSP_Length(count))
		vDSP_vsdiv(result, 1, [Float(UInt32.max)], &result, 1, vDSP_Length(count))
		vDSP_vsmsa(result, 1, [β-α], [α], &result, 1, vDSP_Length(count))
		return result
	}
	func uniform(_ α: Int, _ β: Int) -> Int {
		//XCTAssert(α<β)
		return α+Int(arc4random_uniform(UInt32(β-α)))
	}
}
extension MTLBuffer {
	func write(_ to: URL) throws {
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
	var array: Array<Float> {
		let rows: Int = Int(la_matrix_rows(self))
		let cols: Int = Int(la_matrix_cols(self))
		let cache: Array<Float> = Array<Float>(repeating: 0, count: rows*cols)
		XCTAssert(la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(mutating: cache), la_matrix_cols(self), self)==0)
		return cache
	}
	var hasErr: Bool {
		return la_status(self) != 0
	}
}
private let hint: la_hint_t = la_hint_t(LA_NO_HINT)
private let attr: la_attribute_t = la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING)
private let norm: la_norm_t = la_norm_t(LA_L2_NORM)
