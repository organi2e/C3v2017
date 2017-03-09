//
//  Math.swift
//  macOS
//
//  Created by Kota Nakano on 2017/03/07.
//
//

import Metal
public class Math {
	let add: MTLComputePipelineState
	let sub: MTLComputePipelineState
	let mul: MTLComputePipelineState
	let div: MTLComputePipelineState
	let fma: MTLComputePipelineState
	let exp: MTLComputePipelineState
	let log: MTLComputePipelineState
	let sin: MTLComputePipelineState
	let cos: MTLComputePipelineState
	let tan: MTLComputePipelineState
	let abs: MTLComputePipelineState
	let tanh: MTLComputePipelineState
	let sign: MTLComputePipelineState
	let sigm: MTLComputePipelineState
	let relu: MTLComputePipelineState
	let soft: MTLComputePipelineState
	let regu: MTLComputePipelineState
	let gemvc: MTLComputePipelineState
	let gemvr: MTLComputePipelineState
	public init(device: MTLDevice) throws {
		let bundle: Bundle = Bundle(for: type(of: self))
		let library: MTLLibrary = try device.makeDefaultLibrary(bundle: bundle)
		add = try library.make(name: "add")
		sub = try library.make(name: "sub")
		mul = try library.make(name: "mul")
		div = try library.make(name: "div")
		fma = try library.make(name: "fma")
		exp = try library.make(name: "exp")
		log = try library.make(name: "log")
		sin = try library.make(name: "sin")
		cos = try library.make(name: "cos")
		tan = try library.make(name: "tan")
		abs = try library.make(name: "abs")
		tanh = try library.make(name: "tanh")
		sign = try library.make(name: "sign")
		sigm = try library.make(name: "sigm")
		relu = try library.make(name: "relu")
		soft = try library.make(name: "soft")
		regu = try library.make(name: "regu")
		gemvc = try library.make(name: "gemvc")
		gemvr = try library.make(name: "gemvr")
	}
}
extension Math {
	public func copy(commandBuffer: MTLCommandBuffer, target: (MTLBuffer, Int), source: (MTLBuffer, Int), count: Int) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: source.0, sourceOffset: source.1, to: target.0, destinationOffset: target.1, size: count)
		encoder.endEncoding()
	}
	public func fill(commandBuffer: MTLCommandBuffer, target: (MTLBuffer, Int), value: UInt8, count: Int) {
		let encoder: MTLBlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: target.0, range: NSRange(location: target.1, length: count), value: value)
		encoder.endEncoding()
	}
}
internal extension MTLLibrary {
	internal func make(name: String, constantValues: MTLFunctionConstantValues = MTLFunctionConstantValues()) throws -> MTLComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}
