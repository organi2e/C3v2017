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
	internal var state: RingBuffer<Buffer> = RingBuffer<Buffer>()
	internal var study: RingBuffer<Buffer> = RingBuffer<Buffer>()
	internal var value: RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)> = RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)>()
	internal var delta: RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)> = RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)>()
}
extension Cell {
	internal func setup() {
		let length: Int = width * MemoryLayout<Float>.size
		let array: Array<Void> = Array<Void>(repeating: (), count: 2)
		value = RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)>(array: array.map { (
			context.make(length: length, options: .storageModeShared),
			context.make(length: length, options: .storageModeShared),
			context.make(length: length, options: .storageModeShared)
			)}
		)
		delta = RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)>(array: array.map { (
			context.make(length: length, options: .storageModeShared),
			context.make(length: length, options: .storageModeShared),
			context.make(length: length, options: .storageModeShared)
			)}
		)
		state = RingBuffer<Buffer>(array: array.map { context.make(length: length, options: .storageModeShared) })
		study = RingBuffer<Buffer>(array: array.map { context.make(length: length, options: .storageModeShared) })
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
			[state.current, value.current.χ, value.current.μ, value.current.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
		commandBuffer.commit()
		ready.remove(.Source)
	}
	public func collect(ignore: Set<Cell> = Set<Cell>()) -> LaObjet {
		guard !ignore.contains(self) else { return make(nocopy: state.previous.contents(), rows: width, cols: 1) }
		if !ready.contains(.Source) {
			
			let commandBuffer: CommandBuffer = context.make()
			
			let Σ: [(χ: LaObjet, μ: LaObjet, σ: LaObjet)] = input.map {
				$0.collect(commandBuffer: commandBuffer, ignore: ignore.union([self]))
			} + []
			let φ: (χ: LaObjet, μ: LaObjet, σ: LaObjet) = Σ.reduce(bias.collect(commandBuffer: commandBuffer, ignore: ignore)) {
				($0.0.χ + $0.1.χ, $0.0.μ + $0.1.μ, $0.0.σ + $0.1.σ)
			}
			
			let χ: Buffer = value.current.χ
			let μ: Buffer = value.current.μ
			let σ: Buffer = value.current.σ
			
			commandBuffer.addCompletedHandler { (_) in
				φ.χ.render(to: χ.contents())
				φ.μ.render(to: μ.contents())
				φ.σ.render(to: σ.contents())
			}
			commandBuffer.commit()
						
			do {
				let commandBuffer: CommandBuffer = context.make()
				context.computer.step(commandBuffer: commandBuffer, y: state.current, x: value.current.χ, count: width)
				commandBuffer.commit()
			}
			
			ready.insert(.Source)
		}
		return make(nocopy: state.current.contents(), rows: width, cols: 1)
	}
	public func correct_clear(ignore: Set<Cell> = Set<Cell>()) {
		let commandBuffer: CommandBuffer = context.make()
		output.forEach {
			$0.correct_clear(commandBuffer: commandBuffer, ignore: ignore)
		}
		bias.correct_clear(commandBuffer: commandBuffer, ignore: ignore)
		study.progress()
		delta.progress()
		do {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			[study.current, delta.current.χ, delta.current.μ, delta.current.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
		commandBuffer.commit()
		ready.remove(.Target)
	}
	public func correct(ignore: Set<Cell> = Set<Cell>()) -> (Δμ: LaObjet, Δσ: LaObjet) {
		guard !ignore.contains(self) else { return (Δμ: make(nocopy: delta.previous.μ.contents(), rows: width, cols: 1),
		                                            Δσ: make(nocopy: delta.previous.σ.contents(), rows: width, cols: 1)
			)
		}
		if !ready.contains(.Adjust) {
			let commandBuffer: CommandBuffer = context.make()
			if ready.contains(.Target) {
				switch modes {
				case .State:
					
					break
				case .Value:
					assertionFailure("not implemented")
				}
			} else {
				let zero: LaObjet = make(value: 0)
				let Σ: LaObjet = output.map {
					$0.correct(commandBuffer: commandBuffer, ignore: ignore.union([self]))
				}.reduce(zero) {
					$0.0 + $0.1
				}
			}
			commandBuffer.commit()
			ready.insert(.Adjust)
		}
		return (Δμ: make(nocopy: delta.current.μ.contents(), rows: width, cols: 1),
		        Δσ: make(nocopy: delta.current.σ.contents(), rows: width, cols: 1)
		)
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
	internal var distributor: Distributor {
		return context.gauss
	}
}
extension Cell {
	public var source: Array<Float> {
		get {
			let result: Array<Float> = Array<Float>(repeating: 0, count: width)
			let source: Buffer = state.current
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
			let target: Buffer = state.current
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
			let source: Buffer = study.current
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
			let target: Buffer = study.current
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
		cell.width = width - ( width + 15 ) % 16 + 15
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
