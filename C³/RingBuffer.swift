//
//  RingBuffer.swift
//  C³
//
//  Created by Kota on 10/3/16.
//
//

import Foundation

internal struct RingBuffer<T> {
	private var cursor: Int
	private let buffer: Array<T>
	mutating func progress() {
		cursor = ( cursor + 1 ) % length
	}
	init(array: Array<T> = Array<T>()) {
		cursor = 0
		buffer = array
	}
	var current: T {
		return buffer[(cursor+1)%buffer.count]
	}
	var previous: T {
		return buffer[(cursor+0)%buffer.count]
	}
	subscript(index: Int) -> T {
		return buffer[index%buffer.count]
	}
	var length: Int {
		return buffer.count
	}
}
