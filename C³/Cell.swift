//
//  Cell.swift
//  macOS
//
//  Created by Kota Nakano on 1/27/17.
//
//

import CoreData
import Computer
import Distributor

public class Cell: ManagedObject {
	internal var distributor: Distributor!
	internal var φ: RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)> = RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)>()
	//internal var ϝ: RingBuffer<(χ: Buffer, P: Buffer)> = RingBuffer<(F: Buffer, P: Buffer)>()
	internal var Δ: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
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
		guard let context: Context = managedObjectContext as? Context else { fatalError(Context.ErrorCases.InvalidContext.description) }
		let array: Array<Void> = Array<Void>(repeating: (), count: 2)
		φ = RingBuffer<(χ: Buffer, μ: Buffer, σ: Buffer)>(array: array.map {(
			context.make(length: MemoryLayout<Float>.size*width, options: .storageModePrivate),
			context.make(length: MemoryLayout<Float>.size*width, options: .storageModeShared),
			context.make(length: MemoryLayout<Float>.size*width, options: .storageModeShared)
		)})
		Δ = RingBuffer<(μ: Buffer, σ: Buffer)>(array: array.map {(
			context.make(length: MemoryLayout<Float>.size*width, options: .storageModePrivate),
			context.make(length: MemoryLayout<Float>.size*width, options: .storageModePrivate)
		)})
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
		φ.progress()
		context.clr([φ.current.χ, φ.current.μ, φ.current.σ])
		input.forEach {
			$0.collect_clear(distributor: distributor, ignore: ignore.union([self]))
		}
	}
	public func correct_clear() {
		Δ.progress()
		context.clr([Δ.current.μ, Δ.current.σ])
	}
	public func collect(ignore: Set<Cell> = Set<Cell>()) {
		input.forEach {
			$0.collect(distributor: distributor, ignore: ignore.union([self]))
		}
	}
	public func correct(ignore: Set<Cell> = Set<Cell>()) {
		
	}
}
extension Cell {
	
}
extension Cell {
	var isRecurrent: Bool {
		return feedback != nil
	}
}
extension Context {
	public func make(name: String, width: Int, attributes: Dictionary<String, String> = Dictionary<String, String>(), output: [Cell] = [], input: [Cell] = [], decay: Bool = false, recurrent: Bool = false) throws -> Cell {
		assert(0<width)
		let cell: Cell = try make()
		cell.name = name
		cell.width = width
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
		return cell
	}
}
