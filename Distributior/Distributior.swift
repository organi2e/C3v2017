//
//  Distribution.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal


//
//  Gauss.swift
//  tvOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Accelerate
import Metal
public protocol Derivative {
	func derivate(commandBuffer: MTLCommandBuffer, Δ: (μ: MTLBuffer, σ: MTLBuffer), g: (μ: MTLBuffer, σ: MTLBuffer), y: (Δ: MTLBuffer, p: MTLBuffer), v: (μ: MTLBuffer, σ: MTLBuffer), count: Int)
	
	func delta(commandBuffer: MTLCommandBuffer, Δ: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer), g: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int), rtrl: Bool)
	
	func jacobian(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), v: (μ: MTLBuffer, σ: MTLBuffer), Σ: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int), rtrl: Bool)
	func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), a: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: (rows: Int, cols: Int), rtrl: Bool)
	func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), b: (μ: MTLBuffer, σ: MTLBuffer), g: (μ: MTLBuffer, σ: MTLBuffer), j: (μ: MTLBuffer, σ: MTLBuffer), y: MTLBuffer, count: (rows: Int, cols: Int), rtrl: Bool)
	func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), c: (μ: MTLBuffer, σ: MTLBuffer), count: Int, rtrl: Bool)
	func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int), rtrl: Bool)
	func jacobian(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, a: (μ: MTLBuffer, σ: MTLBuffer), count: (rows: Int, cols: Int), rtrl: Bool)
}
public protocol Activative {
	
	func activate(commandBuffer: MTLCommandBuffer, y: (χ: MTLBuffer, p: MTLBuffer), v: (μ: MTLBuffer, σ: MTLBuffer), count: Int)
	func collect(commandBuffer: MTLCommandBuffer, v: (μ: MTLBuffer, σ: MTLBuffer), Σ: (μ: MTLBuffer, σ: MTLBuffer), count: Int)
	func collect(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), w: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, count: (rows: Int, cols: Int))
	func collect(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), c: (μ: MTLBuffer, σ: MTLBuffer), count: Int)
	func collect(commandBuffer: MTLCommandBuffer, Σ: (μ: MTLBuffer, σ: MTLBuffer), d: MTLBuffer, v: (μ: MTLBuffer, σ: MTLBuffer), count: Int)
}
public protocol Distributor: Activative, Derivative {
	func clear(commandBuffer: MTLCommandBuffer, μ: MTLBuffer, σ: MTLBuffer)
}
/*

public protocol Stochastic {
	
	func shuffle(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer, count: Int)
	
	func errorState(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, ψ: MTLBuffer)
	func errorValue(commandBuffer: MTLCommandBuffer, Δ: MTLBuffer, ψ: MTLBuffer)
	func deltaState(commandBuffer: MTLCommandBuffer, Δμ: MTLBuffer, Δσ: MTLBuffer, Δ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer)
	func deltaValue(commandBuffer: MTLCommandBuffer, Δμ: MTLBuffer, Δσ: MTLBuffer, Δ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer					)
	
}
public protocol Derivative {
	
	func jacobian(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), w: (μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, width: Int, refer: Int)//jμA, jσA
	func jacobian(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer), c: (μ: MTLBuffer, σ: MTLBuffer), width: Int)//jμC, jσC
	
	func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, x: MTLBuffer, refer: Int)//μa
	func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, b: MTLBuffer, j: MTLBuffer, p: MTLBuffer, refer: Int)//μb
	func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, c: MTLBuffer)//μc
	func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, d: MTLBuffer, p: MTLBuffer, refer: Int)//μd

	func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, w: MTLBuffer, refer: Int)//μx
	
	func jacobian(commandBuffer: MTLCommandBuffer, jσ: MTLBuffer, w: MTLBuffer, x: MTLBuffer, refer: Int)//σa
	func jacobian(commandBuffer: MTLCommandBuffer, jσ: MTLBuffer, b: MTLBuffer, j: MTLBuffer, p: MTLBuffer, refer: Int)//σb
	func jacobian(commandBuffer: MTLCommandBuffer, jσ: MTLBuffer, c: MTLBuffer)//σc
	func jacobian(commandBuffer: MTLCommandBuffer, jσ: MTLBuffer, d: MTLBuffer, p: MTLBuffer, refer: Int)//σd
	
	//func jacobian(commandBuffer: MTLCommandBuffer, jμ: MTLBuffer, w: MTLBuffer, refer: Int)//σx
	
	func clear(commandBuffer: MTLCommandBuffer, j: (μ: MTLBuffer, σ: MTLBuffer))
}
public protocol Synthesis {
	func synthesize(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer)
	func collect(commandBuffer: MTLCommandBuffer, χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer)
	func collect(commandBuffer: MTLCommandBuffer, r: MTLBuffer, x: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer))
	func collect(commandBuffer: MTLCommandBuffer, w: (χ: MTLBuffer, μ: MTLBuffer, σ: MTLBuffer), x: MTLBuffer, refer: Int)
	func reset(commandBuffer: MTLCommandBuffer)
}
public protocol Distributor: Stochastic, Synthesis, Derivative {
	
}
*/
extension MTLLibrary {
	internal func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}

