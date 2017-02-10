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
			I.source = [1.0, 2.0, 3.0, 4.0, 5.0]
			O.collect()
			print(O.source)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
