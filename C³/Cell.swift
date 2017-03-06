//
//  Cell.swift
//  macOS
//
//  Created by Kota Nakano on 1/27/17.
//
//

import LaObjet
import CoreData
import Computer
import Distributor
import Metal

public class Cell: ManagedObject {
	internal enum Ready {
		case Source
		case Target
		case Adjust
	}
	internal var ready: Set<Ready> = Set<Ready>()
	internal var state: RingBuffer<(χ: Buffer, p: Buffer)> = RingBuffer<(χ: Buffer, p: Buffer)>()
	internal var study: RingBuffer<(χ: Buffer, Δ: Buffer)> = RingBuffer<(χ: Buffer, Δ: Buffer)>()
	internal var value: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
	internal var nabla: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
	internal var delta: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
}
extension Cell {
	internal func setup() {
		let length: Int = width * MemoryLayout<Float>.size
		let array: Array<Void> = Array<Void>(repeating: (), count: 2)
		state = RingBuffer<(χ: Buffer, p: Buffer)>(array: array.map { (
			context.make(length: length, options: .storageModePrivate),
			context.make(length: length, options: .storageModePrivate)
			)}
		)
		study = RingBuffer<(χ: Buffer, Δ: Buffer)>(array: array.map { (
			context.make(length: length, options: .storageModePrivate),
			context.make(length: length, options: .storageModePrivate)
			)}
		)
		value = RingBuffer<(μ: Buffer, σ: Buffer)>(array: array.map { (
			context.make(length: length, options: .storageModePrivate),
			context.make(length: length, options: .storageModePrivate)
			)}
		)
		nabla = RingBuffer<(μ: Buffer, σ: Buffer)>(array: array.map { (
			context.make(length: length, options: .storageModePrivate),
			context.make(length: length, options: .storageModePrivate)
			)}
		)
		delta = RingBuffer<(μ: Buffer, σ: Buffer)>(array: array.map { (
			context.make(length: length, options: .storageModePrivate),
			context.make(length: length, options: .storageModePrivate)
			)}
		)
	}
	public override func awakeFromFetch() {
		super.awakeFromFetch()
		setup()
	}
	public override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		setup()
	}
}
extension Cell {
	public func collect_clear(ignore: Set<Cell> = Set<Cell>()) {
		let commandBuffer: MTLCommandBuffer = context.make()
		input.forEach {
			$0.collect_clear(commandBuffer: commandBuffer, ignore: ignore.union([self]))
		}
		bias.collect_clear(commandBuffer: commandBuffer)
		state.progress()
		value.progress()
		do {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			[state.current.χ, state.current.p, value.current.μ, value.current.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
		commandBuffer.commit()
		ready.remove(.Source)
	}
	public func collect(ignore: Set<Cell> = Set<Cell>()) -> (χ: Buffer, p: Buffer) {
		guard !ignore.contains(self) else { return state.previous }
		if !ready.contains(.Source) {
			let distributor: Distributor = context.gaussFactory
			input.forEach {
				$0.collect(distributor: distributor, Σ: value.current, ignore: ignore.union([self]))
			}
			bias.collect(distributor: distributor, Σ: value.current, ignore: ignore.union([self]))
			do {
				let commandBuffer: MTLCommandBuffer = context.make()
				distributor.collect(commandBuffer: commandBuffer, v: value.current, Σ: value.current, count: width)
				distributor.activate(commandBuffer: commandBuffer, y: state.current, v: value.current, count: width)
				commandBuffer.commit()
			}
			ready.insert(.Source)
		}
		return state.current
	}
	public func correct_clear(ignore: Set<Cell> = Set<Cell>()) {
		let commandBuffer: CommandBuffer = context.make()
		output.forEach {
			$0.correct_clear(commandBuffer: commandBuffer, ignore: ignore)
		}
		bias.correct_clear(commandBuffer: commandBuffer, ignore: ignore)
		study.progress()
		nabla.progress()
		do {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			[study.current.χ, study.current.Δ, nabla.current.μ, nabla.current.σ, delta.current.μ, delta.current.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
		commandBuffer.commit()
		ready.remove(.Target)
	}
	public func correct(ignore: Set<Cell> = Set<Cell>()) -> (μ: Buffer, σ: Buffer) {
		guard !ignore.contains(self) else { return delta.previous }
		if !ready.contains(.Adjust) {
			let distributor: Distributor = context.gaussFactory
			if ready.contains(.Target) {
				context.computer.sub(z: study.current.Δ, y: study.current.χ, x: state.current.χ, count: width)
			} else {
				
			}
			let commandBuffer: CommandBuffer = context.make()
			distributor.derivate(commandBuffer: commandBuffer,
			                     Δ: delta.current,
			                     g: nabla.current,
			                     y: (Δ: study.current.Δ, p: state.current.p),
			                     v: value.current, count: width)
			commandBuffer.commit()
			ready.insert(.Adjust)
		}
		return delta.current
	}
}
extension Cell {
	var isDecayed: Bool {
		return decay != nil
	}
	var isRecurrent: Bool {
		return feedback != nil
	}
}
extension Cell {
	@NSManaged var name: String
	@NSManaged var width: Int
	@NSManaged var attributes: Dictionary<String, String>
	@NSManaged var date: Date
	@NSManaged var bias: Bias
	@NSManaged var input: Set<Edge>
	@NSManaged var output: Set<Edge>
	@NSManaged var decay: Decay?
	@NSManaged var feedback: Feedback?
}
extension Cell {
	public enum Modes {
		case State//Output binary state
		case Value//Output numeric sample
	}
	public var modes: Modes {
		get {
			return .State
		}
		set {
			switch newValue {
			case .State:
				break
			case .Value:
				break
			}
		}
	}
}
extension Cell {
	public enum Distribution {
		case Cauchy
		case Gauss
	}
	public var dists: Distribution {
		get {
			return .Gauss
		}
		set {
			switch newValue {
			case .Cauchy:
				break
			case .Gauss:
				break
			}
		}
	}
}
extension Cell {
	public var source: Array<Float> {
		get {
			let source: Buffer = state.current.χ
			let target: Buffer = context.make(length: width*MemoryLayout<Float>.size, options: .storageModeShared)
			let commandBuffer: CommandBuffer = context.make()
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.copy(from: source, sourceOffset: 0, to: target, destinationOffset: 0, size: min(source.length, target.length))
			encoder.endEncoding()
			commandBuffer.addCompletedHandler { ( _: CommandBuffer) in
				target.setPurgeableState(.empty)
			}
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			return Array<Float>(target.buffer)
		}
		set {
			let commandBuffer: CommandBuffer = context.make()
			let source: Buffer = context.make(array: newValue, options: .storageModePrivate)
			let target: Buffer = state.current.χ
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.copy(from: source, sourceOffset: 0, to: target, destinationOffset: 0, size: min(source.length, target.length))
			encoder.endEncoding()
			commandBuffer.addCompletedHandler { (_) in
				source.setPurgeableState(.empty)
			}
			commandBuffer.commit()
			ready.insert(.Source)
		}
	}
	public var target: Array<Float> {
		get {
			let result: Array<Float> = Array<Float>(repeating: 0, count: width)
			let source: Buffer = study.current.χ
			let commandBuffer: CommandBuffer = context.make()
			commandBuffer.addCompletedHandler { ( _: CommandBuffer) in
				assert(MemoryLayout<Float>.size*result.count==Data(bytesNoCopy: source.contents(), count: source.length, deallocator: .none)
					.copyBytes(to: UnsafeMutableBufferPointer<Float>(start: UnsafeMutablePointer<Float>(mutating: result), count: result.count)))
			}
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			return result
		}
		set {
			let commandBuffer: CommandBuffer = context.make()
			let source: Buffer = context.make(array: newValue, options: .storageModePrivate)
			let target: Buffer = study.current.χ
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			encoder.copy(from: source, sourceOffset: 0, to: target, destinationOffset: 0, size: min(source.length, target.length))
			encoder.endEncoding()
			commandBuffer.addCompletedHandler { (_) in
				source.setPurgeableState(.empty)
			}
			commandBuffer.commit()
			ready.insert(.Target)
		}
	}
}
extension Context {
	public func make(name: String, width: Int, output: [Cell] = [], input: [Cell] = [], decay: Bool = false, recurrent: Bool = false) throws -> Cell {
		assert(0<width)
		let cell: Cell = try make()
		cell.name = name
		cell.width = width
		cell.attributes = Dictionary<String, String>()
		cell.date = Date()
		try input.forEach {
			let _: Edge = try make(output: cell, input: $0)
		}
		try output.forEach {
			let _: Edge = try make(output: $0, input: cell)
		}
		if decay {
			
		}
		if recurrent {
			
		}
		cell.bias = try make(cell: cell)
		cell.setup()
		return cell
	}
}
internal protocol Collectee {
	func collect_clear()
	func collect()
	func correct_clear()
	func correct()
}
