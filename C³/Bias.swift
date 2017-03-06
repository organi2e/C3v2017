//
//  Bias.swift
//  macOS
//
//  Created by Kota Nakano on 1/27/17.
//
//

import CoreData
import Computer
import Distributor

internal class Bias: Arcane {
	fileprivate var j: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
}
extension Bias {
	@NSManaged var cell: Cell
}
extension Bias {
	override func setup() {
		let length: Int = cell.width * cell.width * MemoryLayout<Float>.size
		super.setup()
		j = RingBuffer<(μ: Buffer, σ: Buffer)>(array: Array<Void>(repeating: (), count: 2).map {(
			context.make(length: length),
			context.make(length: length)
			)
		})
	}
	internal func collect_clear(commandBuffer: CommandBuffer) {
		refresh(commandBuffer: commandBuffer)
	}
	internal func collect(distributor: Distributor, Σ: (μ: Buffer, σ: Buffer), ignore: Set<Cell>) {
		let count: Int = cell.width
		let c: (μ: Buffer, σ: Buffer) = (μ: μ, σ: σ)
		let commandBuffer: CommandBuffer = context.make()
		distributor.collect(commandBuffer: commandBuffer, Σ: Σ, c: c, count: count)
		distributor.jacobian(commandBuffer: commandBuffer, Σ: j.current, c: c, count: count, rtrl: false)
		commandBuffer.commit()
	}
	internal func correct_clear(commandBuffer: CommandBuffer, ignore: Set<Cell>) {
		j.progress()
		do {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			[j.current.μ, j.current.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
		commandBuffer.commit()
	}
	internal func correct(commandBuffer: CommandBuffer, ignore: Set<Cell>) {
		
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
