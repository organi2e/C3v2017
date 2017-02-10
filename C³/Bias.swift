//
//  Bias.swift
//  macOS
//
//  Created by Kota Nakano on 1/27/17.
//
//

import LaObjet
import CoreData
import Computer
import Distributor

internal class Bias: Arcane {

}
extension Bias {
	@NSManaged var cell: Cell
}
extension Bias {
	override func setup() {
		super.setup()
	}
	internal func collect_clear(commandBuffer: MTLCommandBuffer) {
		refresh(commandBuffer: commandBuffer)
		cell.distributor.shuffle(commandBuffer: commandBuffer, χ: χ, from: (μ: μ, σ: σ), count: cell.width)
	}
	internal func collect(commandBuffer: CommandBuffer, ignore: Set<Cell>) -> (χ: LaObjet, μ: LaObjet, σ: LaObjet) {
		let distributor: Distributor = cell.distributor
		let width: Int = cell.width
		return (χ: make(nocopy: χ.contents(), rows: width, cols: 1),
		        μ: distributor.μ(make(nocopy: μ.contents(), rows: width, cols: 1)),
		        σ: distributor.σ(make(nocopy: σ.contents(), rows: width, cols: 1))
		)
	}
	internal func correct_clear(commandBuffer: CommandBuffer, ignore: Set<Cell>) {
	
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
