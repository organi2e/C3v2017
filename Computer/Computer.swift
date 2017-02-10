//
//  Computer.swift
//  macOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal

final public class Computer {
	
	let queue: CommandQueue
	
	let add: ComputePipelineState
	let sub: ComputePipelineState
	let mul: ComputePipelineState
	let div: ComputePipelineState
	
	let exp: ComputePipelineState
	let log: ComputePipelineState
	
	let sign: ComputePipelineState
	let sigm: ComputePipelineState
	
	let gemv16: ComputePipelineState
	let gemm16: ComputePipelineState
	
	let threads: MTLSize
	public init(device: Device) throws {
		
		let library: Library = try device.makeDefaultLibrary(bundle: Bundle(for: type(of: self)))
		
		add = try library.make(name: "add")
		sub = try library.make(name: "sub")
		mul = try library.make(name: "mul")
		div = try library.make(name: "div")
		
		exp = try library.make(name: "exp")
		log = try library.make(name: "log")
		
		sign = try library.make(name: "sign")
		sigm = try library.make(name: "sigm")
		
		gemv16 = try library.make(name: "gemv16")
		gemm16 = try library.make(name: "gemv16")
		
		queue = device.makeCommandQueue()
		
		threads = MTLSize(width: 64, height: 1, depth: 1)
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
	public func make() -> CommandBuffer {
		return queue.makeCommandBuffer()
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
	public func reference<T>() -> UnsafeMutablePointer<T> {
		return UnsafeMutablePointer<T>(OpaquePointer(contents()))
	}
	public func feed<T>(buffer: [T]) {
		Data(bytes: buffer, count: buffer.count*MemoryLayout<T>.size).copyBytes(to: UnsafeMutablePointer<UInt8>(OpaquePointer(contents())), count: length)
		//didModifyRange(NSRange(location: 0, length: length))
	}
}
extension Library {
	internal func make(name: String, constantValues: FunctionConstantValues = FunctionConstantValues()) throws -> ComputePipelineState {
		return try device.makeComputePipelineState(function: try makeFunction(name: name, constantValues: constantValues))
	}
}

public protocol Compute {
	func compute(block: ((CommandBuffer)->Void)?) -> CommandBuffer
}

public typealias ComputePipelineState = MTLComputePipelineState
public typealias ComputeCommandEncoder = MTLComputeCommandEncoder
public typealias BlitCommandEncoder = MTLBlitCommandEncoder
public typealias Library = MTLLibrary
public typealias Function = MTLFunction
public typealias FunctionConstantValues = MTLFunctionConstantValues

public typealias Device = MTLDevice
public typealias Buffer = MTLBuffer
public typealias CommandQueue = MTLCommandQueue
public typealias CommandBuffer = MTLCommandBuffer
public typealias ResourceOptions = MTLResourceOptions
