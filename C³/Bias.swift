//
//  Bias.swift
//  macOS
//
//  Created by Kota Nakano on 1/27/17.
//
//

import CoreData
import Distributor

internal class Bias: Arcane {
	var j: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
}
extension Bias {
	@NSManaged var cell: Cell
}
extension Bias {
	override func setup() {
		let length: Int = cell.width * cell.width * MemoryLayout<Float>.size
		super.setup()
		j = RingBuffer<(μ: Buffer, σ: Buffer)>(array: Array<Void>(repeating: (), count: 2).map {(
			context.make(length: length, options: .storageModePrivate),
			context.make(length: length, options: .storageModePrivate)
			)
		})
	}
	internal func collect_clear(commandBuffer: CommandBuffer) {
		refresh(commandBuffer: commandBuffer)
	}
	internal func collect(distributor: Distributor, Σ: (μ: Buffer, σ: Buffer), ignore: Set<Cell>) {
		let count: Int = cell.width
		let commandBuffer: CommandBuffer = context.make()
		distributor.collect(commandBuffer: commandBuffer, Σ: Σ, c: θ, count: count)
		distributor.jacobian(commandBuffer: commandBuffer, Σ: j.current, c: θ, count: count, rtrl: false)
		commandBuffer.commit()
	}
	internal func correct_clear(commandBuffer: CommandBuffer, ignore: Set<Cell>) {
		j.progress()
		[j.current.μ, j.current.σ].forEach {
			context.math.fill(commandBuffer: commandBuffer, target: ($0, 0), value: 0, count: $0.length)
		}
	}
	internal func correct(distributor: Distributor, ignore: Set<Cell>) {
		let count: (rows: Int, cols: Int) = (rows: cell.width, cols: 1)
		let (g, v): ((μ: Buffer, σ: Buffer), (μ: Buffer, σ: Buffer)) = cell.correct(ignore: ignore)
		let commandBuffer: CommandBuffer = context.make()
		distributor.jacobian(commandBuffer: commandBuffer, j: j.current, v: v, Σ: j.current, count: count, rtrl: false)
		distributor.delta(commandBuffer: commandBuffer, Δ: Δ, j: j.current, g: g, count: count, rtrl: cell.isRecurrent)
		update(commandBuffer: commandBuffer)
		commandBuffer.commit()
	}
}
extension Context {
	@nonobjc internal func make(cell: Cell) throws -> Bias {
		let bias: Bias = try make()
		bias.cell = cell
		bias.location = Data(count: MemoryLayout<Float>.size*cell.width)
		bias.logscale = Data(count: MemoryLayout<Float>.size*cell.width)
		bias.setup()
		return bias
	}
}
