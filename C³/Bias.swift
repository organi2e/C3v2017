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
	var j: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>()
}
extension Bias {
	@NSManaged var cell: Cell
}
extension Bias {
	override func setup() {
		let length: Int = cell.width * cell.width * MemoryLayout<Float>.size
		let array: Array<Void> = Array<Void>(repeating: (), count: 2)
		super.setup()
		j = RingBuffer<(μ: Buffer, σ: Buffer)>(array: array.map {(
			context.make(length: length),
			context.make(length: length)
			)
		})
	}
	internal func collect_clear(commandBuffer: CommandBuffer) {
		refresh(commandBuffer: commandBuffer)
	}
	internal func collect(distributor: Distributor, ignore: Set<Cell>) {
		
		
		
		/*let width: Int = cell.width
		return (χ: make(nocopy: χ.contents(), rows: width, cols: 1),
		        μ: distributor.μ(make(nocopy: μ.contents(), rows: width, cols: 1)),
		        σ: distributor.σ(make(nocopy: σ.contents(), rows: width, cols: 1))
		)
		*/
	}
	internal func correct_clear(commandBuffer: CommandBuffer, ignore: Set<Cell>) {
		j.progress()
		
	}
	internal func correct(commandBuffer: CommandBuffer, ignore: Set<Cell>) {
		let width: Int = cell.width
		let Δ: (μ: LaObjet, σ: LaObjet) = cell.correct(ignore: ignore)
		
		do {
			let jμ: LaObjet = make(nocopy: j.current.μ.contents(), rows: width, cols: width)
			let jσ: LaObjet = make(nocopy: j.current.σ.contents(), rows: width, cols: width)
			func block(_: CommandBuffer) {
				matrix_product(Δ.μ, jμ).render(to: Δμ.contents())
				matrix_product(Δ.σ, jσ).render(to: Δσ.contents())
			}
			let commandBuffer: CommandBuffer = context.make()
			commandBuffer.addCompletedHandler(block)
			commandBuffer.commit()
		}
		
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
