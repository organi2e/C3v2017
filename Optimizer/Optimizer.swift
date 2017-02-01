//
//  Optimizer.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/25.
//
//

import Metal

public protocol Optimizer {
	func encode(commandBuffer: MTLCommandBuffer, θ: MTLBuffer, Δθ: MTLBuffer)
	func reset(commandBuffer: MTLCommandBuffer)
}
