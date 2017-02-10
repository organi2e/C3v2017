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

public class Cell: ManagedObject {
	var state: RingBuffer<Buffer> = RingBuffer<Buffer>()
	var value: RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)> = RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)>()
	var delta: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
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
	internal func setup() {
		let length: Int = width * MemoryLayout<Float>.size
		let array: Array<Void> = Array<Void>(repeating: (), count: 2)
		state = RingBuffer<Buffer>(array: array.map {
			context.make(length: length, options: .storageModeShared)
		})
		value = RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)>(array: array.map {(
			context.make(length: length, options: .storageModeShared),
			context.make(length: length, options: .storageModeShared),
			context.make(length: length, options: .storageModeShared)
			)
		})
		delta = RingBuffer<(μ: Buffer, σ: Buffer)>(array: array.map {(
			context.make(length: MemoryLayout<Float>.size*width, options: .storageModeShared),
			context.make(length: MemoryLayout<Float>.size*width, options: .storageModeShared)
			)
		})
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
		state.progress()
		value.progress()
		input.forEach {
			$0.collect_clear(commandBuffer: commandBuffer, ignore: ignore.union([self]))
		}
		bias.collect_clear(commandBuffer: commandBuffer)
		commandBuffer.commit()
	}
	public func collect(ignore: Set<Cell> = Set<Cell>()) -> LaObjet {
		do {
			let commandBuffer: CommandBuffer = context.make()
			
			let Σ: [(χ: LaObjet, μ: LaObjet, σ: LaObjet)] = input.map {
				$0.collect(commandBuffer: commandBuffer, ignore: ignore.union([self]))
			} + []
			let φ: (χ: LaObjet, μ: LaObjet, σ: LaObjet) = Σ.reduce(bias.collect(commandBuffer: commandBuffer, ignore: ignore)) {
				($0.0.χ + $0.1.χ, $0.0.μ + $0.1.μ, $0.0.σ + $0.1.σ)
			}

			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			φ.χ.render(to: value.current.χ.contents())
			φ.μ.render(to: value.current.μ.contents())
			φ.σ.render(to: value.current.σ.contents())
		
		}
		do {
			let commandBuffer: CommandBuffer = context.make()
			distributor.synthesize(commandBuffer: commandBuffer, ϝ: (χ: state.current, μ: value.current.μ, σ: value.current.σ), Σ: value.current, count: width)
			commandBuffer.commit()
			//commandBuffer.waitUntilCompleted()
		}
		return make(nocopy: state.current.contents(), rows: width, cols: 1)
	}
	public func correct_clear() {
		delta.progress()
	}
	public func correct(ignore: Set<Cell> = Set<Cell>()) {
		
	}
}
extension Cell {
	
}
extension Cell {
	public var φ: UnsafeMutableBufferPointer<Float> {
		let buffer: Buffer = value.current.χ
		return UnsafeMutableBufferPointer<Float>(start: UnsafeMutablePointer<Float>(OpaquePointer(buffer.contents())), count: buffer.length/MemoryLayout<Float>.size)
	}
	public var ϝ: UnsafeMutableBufferPointer<Float> {
		let buffer: Buffer = state.current
		return UnsafeMutableBufferPointer<Float>(start: UnsafeMutablePointer<Float>(OpaquePointer(buffer.contents())), count: buffer.length/MemoryLayout<Float>.size)
	}
}
extension Cell {
	var distributor: Distributor {
		return context.gauss
	}
	var isRecurrent: Bool {
		return feedback != nil
	}
}
protocol Collectee {
	func collect(distributor: Distributor) -> (χ: LaObjet, μ: LaObjet, σ: LaObjet)
}
extension Context {
	public func make(name: String, width: Int, attributes: Dictionary<String, String> = Dictionary<String, String>(), output: [Cell] = [], input: [Cell] = [], decay: Bool = false, recurrent: Bool = false) throws -> Cell {
		assert(0<width)
		let cell: Cell = try make()
		cell.name = name
		cell.width = width - ( width + 15 ) % 16 + 15
		cell.attributes = attributes
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
