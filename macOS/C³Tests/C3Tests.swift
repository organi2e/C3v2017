//
//  C_Tests.swift
//  C³Tests
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import XCTest
@testable import C3
class C3Tests: XCTestCase {
	func testContext() {
		do {
			let context: Context = try Context()
			let I: Cell = try context.make(name: "I", width: 16, input: [])
			let O: Cell = try context.make(name: "O", width: 16, input: [I])
			try context.save()
			O.collect_clear()
			I.correct_clear()
			context.computer.wait()
			for k in 0..<16 {
				I.φ[k] = 1
			}
			let _ = O.collect()
			context.computer.wait()
			print(Array(O.ϝ),
			      Array(O.φ),
				  Array(O.input.first!.value.χ.buffer),
			      Array(O.input.first!.value.μ.buffer),
			      Array(O.input.first!.value.σ.buffer)
			)
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
