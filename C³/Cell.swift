//
//  Cell.swift
//  macOS
//
//  Created by Kota Nakano on 1/27/17.
//
//

import CoreData
import Distributor

public class Cell: NSManagedObject {
	internal enum Ready {
		case Source
		case Target
		case Adjust
	}
	internal var ready: Set<Ready> = Set<Ready>()
	internal var state: RingBuffer<Buffer> = RingBuffer<Buffer>()
	internal var study: RingBuffer<Buffer> = RingBuffer<Buffer>()
	internal var ratio: RingBuffer<Buffer> = RingBuffer<Buffer>()
	internal var value: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
	internal var nabla: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
	internal var delta: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
}
extension Cell {
	internal func setup() {
		let length: Int = width * MemoryLayout<Float>.size
		let array: Array<Void> = Array<Void>(repeating: (), count: 2)
		state = RingBuffer<Buffer>(array: array.map {
			context.make(length: length, options: .storageModePrivate)
		})
		ratio = RingBuffer<Buffer>(array: array.map {
			context.make(length: length, options: .storageModePrivate)
		})
		study = RingBuffer<Buffer>(array: array.map {
			context.make(length: length, options: .storageModePrivate)
		})
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
			$0.collect_clear(ignore: ignore.union([self]))
		}
		bias.collect_clear(commandBuffer: commandBuffer)
		state.progress()
		value.progress()
		ratio.progress()
		[state.current, ratio.current, value.current.μ, value.current.σ].forEach {
			context.math.fill(commandBuffer: commandBuffer, target: ($0, 0), value: 0, count: $0.length)
		}
		commandBuffer.commit()
		ready.remove(.Source)
	}
	public func collect() {
		let _: Buffer = collect(ignore: Set<Cell>())
	}
	internal func collect(ignore: Set<Cell>) -> Buffer {
		guard !ignore.contains(self) else { return state.previous }
		if !ready.contains(.Source) {
			let commandBuffer: MTLCommandBuffer = context.make()
			let distributor: Distributor = context.gaussFactory
			input.forEach {
				$0.collect(distributor: distributor, Σ: value.current, ignore: ignore.union([self]))
			}
			bias.collect(distributor: distributor, Σ: value.current, ignore: ignore.union([self]))
			distributor.collect(commandBuffer: commandBuffer, v: value.current, Σ: value.current, count: width)
			distributor.activate(commandBuffer: commandBuffer, y: (χ: state.current, p: ratio.current), v: value.current, count: width)
			commandBuffer.commit()
			ready.insert(.Source)
		}
		return state.current
	}
	public func correct_clear(ignore: Set<Cell> = Set<Cell>()) {
		let commandBuffer: CommandBuffer = context.make()
		output.forEach {
			$0.correct_clear(ignore: ignore)
		}
		bias.correct_clear(commandBuffer: commandBuffer, ignore: ignore)
		study.progress()
		nabla.progress()
		delta.progress()
		[study.current, nabla.current.μ, nabla.current.σ, delta.current.μ, delta.current.σ].forEach {
			context.math.fill(commandBuffer: commandBuffer, target: ($0, 0), value: 0, count: $0.length)
		}
		commandBuffer.commit()
		ready.subtract([.Target, .Adjust])
	}
	public func correct() {
		let _: (g: (μ: Buffer, σ: Buffer), v: (μ: Buffer, σ: Buffer)) = correct(ignore: Set<Cell>())
	}
	internal func correct(ignore: Set<Cell> = Set<Cell>()) -> (g: (μ: Buffer, σ: Buffer), v: (μ: Buffer, σ: Buffer)) {
		guard !ignore.contains(self) else { return (g: delta.previous, v: value.previous) }
		if !ready.contains(.Adjust) {
			let commandBuffer: CommandBuffer = context.make()
			let distributor: Distributor = context.gaussFactory
			if ready.contains(.Target) {
				context.math.sub(commandBuffer: commandBuffer, y: study.current, a: state.current, b: study.current, count: width)
			} else {
				output.forEach {
					$0.correct(distributor: distributor, Σ: study.current, ignore: ignore.union([self]))
				}
			}
			distributor.derivate(commandBuffer: commandBuffer,
			                     Δ: delta.current,
			                     g: nabla.current,
			                     y: (Δ: study.current, p: ratio.current),
			                     v: value.current, count: width)
			commandBuffer.commit()
			ready.insert(.Adjust)
		}
		return (g: delta.current, v: value.current)
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
			let source: Buffer = ratio.current
			let target: Buffer = context.make(length: width*MemoryLayout<Float>.size, options: .storageModeShared)
			let commandBuffer: CommandBuffer = context.make()
			context.math.copy(commandBuffer: commandBuffer, target: (target, 0), source: (source, 0), count: min(source.length, target.length))
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			defer {
				target.setPurgeableState(.empty)
			}
			return Array<Float>(target.buffer)
		}
		set {
			let commandBuffer: CommandBuffer = context.make()
			let source: Buffer = context.make(array: newValue, options: .storageModePrivate)
			let target: Buffer = state.current
			context.math.copy(commandBuffer: commandBuffer, target: (target, 0), source: (source, 0), count: min(source.length, target.length))
			commandBuffer.addCompletedHandler { ( _: CommandBuffer ) in
				source.setPurgeableState(.empty)
			}
			commandBuffer.commit()
			ready.insert(.Source)
		}
	}
	public var target: Array<Float> {
		get {
			let source: Buffer = study.current
			let target: Buffer = context.make(length: width*MemoryLayout<Float>.size, options: .storageModeShared)
			let commandBuffer: CommandBuffer = context.make()
			context.math.copy(commandBuffer: commandBuffer, target: (target, 0), source: (source, 0), count: min(source.length, target.length))
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			defer {
				target.setPurgeableState(.empty)
			}
			return Array<Float>(target.buffer)
		}
		set {
			let commandBuffer: CommandBuffer = context.make()
			let source: Buffer = context.make(array: newValue, options: .storageModePrivate)
			let target: Buffer = study.current
			context.math.copy(commandBuffer: commandBuffer, target: (target, 0), source: (source, 0), count: min(source.length, target.length))
			commandBuffer.addCompletedHandler { ( _: CommandBuffer ) in
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
