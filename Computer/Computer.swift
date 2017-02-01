//
//  Computer.swift
//  macOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal

public class Computer {
	let queue: MTLCommandQueue
	
	let add: ComputePipelineState
	let sub: ComputePipelineState
	let mul: ComputePipelineState
	let div: ComputePipelineState
	
	let exp: ComputePipelineState
	let log: ComputePipelineState
	
	let sign: ComputePipelineState
	let sigm: ComputePipelineState
	
	let gemv: ComputePipelineState
	
	let threads: Int
	public init(device: MTLDevice) throws {
		
		let library: Library = try device.makeDefaultLibrary(bundle: Bundle(for: type(of: self)))
		
		add = try library.make(name: "add")
		sub = try library.make(name: "sub")
		mul = try library.make(name: "mul")
		div = try library.make(name: "div")
		
		exp = try library.make(name: "exp")
		log = try library.make(name: "log")
		
		sign = try library.make(name: "sign")
		sigm = try library.make(name: "sigm")
		
		gemv = try library.make(name: "gemv")
		
		queue = device.makeCommandQueue()
		
		threads = 64
	}
	public func wait() {
		let commandBuffer: CommandBuffer = queue.makeCommandBuffer()
		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()
	}
}
extension Computer {
	public func zero(buffers: [Buffer]) {
		let commandBuffer: CommandBuffer = queue.makeCommandBuffer()
		let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
		buffers.forEach {
			encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
		}
		encoder.endEncoding()
		commandBuffer.commit()
	}
	public func zero(buffer: Buffer) {
		let commandBuffer: CommandBuffer = queue.makeCommandBuffer()
		let encoder = commandBuffer.makeBlitCommandEncoder()
		encoder.fill(buffer: buffer, range: NSRange(location: 0, length: buffer.length), value: 0)
		commandBuffer.commit()
	}
	public func copy(to: Buffer, from: Buffer) {
		let commandBuffer: CommandBuffer = queue.makeCommandBuffer()
		let encoder = commandBuffer.makeBlitCommandEncoder()
		encoder.copy(from: from, sourceOffset: 0, to: to, destinationOffset: 0, size: min(from.length, to.length))
		commandBuffer.commit()
	}
}
extension Computer {
	public var device: Device {
		return queue.device
	}
	public func compute(block: (CommandBuffer)->Void) {
		let commandBuffer: CommandBuffer = queue.makeCommandBuffer()
		block(commandBuffer)
		commandBuffer.commit()
	}
	public func make(length: Int, options: ResourceOptions) -> Buffer {
		return queue.device.makeBuffer(length: length, options: options)
	}
	public func make(data: Data, options: MTLResourceOptions) -> Buffer {
		let cache: Array<UInt8> = Array<UInt8>(repeating: 0, count: data.count)
		data.copyBytes(to: UnsafeMutablePointer<UInt8>(mutating: cache), count: cache.count)
		return queue.device.makeBuffer(bytes: cache, length: cache.count, options: options)
	}
}
extension Buffer {
	internal var pointer: UnsafeMutablePointer<Float> {
		return UnsafeMutablePointer<Float>(OpaquePointer(contents()))
	}
}
extension Library {
	internal func make(name: String, constantValues: FunctionConstantValues = FunctionConstantValues()) throws -> ComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}

typealias ComputePipelineState = MTLComputePipelineState
typealias ComputeCommandEncoder = MTLComputeCommandEncoder
typealias BlitCommandEncoder = MTLBlitCommandEncoder
typealias Library = MTLLibrary
typealias Function = MTLFunction
typealias FunctionConstantValues = MTLFunctionConstantValues

public typealias Device = MTLDevice
public typealias Buffer = MTLBuffer
public typealias CommandBuffer = MTLCommandBuffer
public typealias ResourceOptions = MTLResourceOptions
