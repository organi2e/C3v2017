//
//  Bias.swift
//  macOS
//
//  Created by Kota Nakano on 1/27/17.
//
//

import CoreData

internal class Bias: Arcane {

}
extension Bias {
	@NSManaged var cell: Cell
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
