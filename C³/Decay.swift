//
//  Decay.swift
//  macOS
//
//  Created by Kota Nakano on 2017/01/28.
//
//

import CoreData

internal class Decay: ManagedObject {
	
}
extension Decay {
	@NSManaged var value: Data
	@NSManaged var cell: Cell
}
extension Context {
	@nonobjc internal func make(cell: Cell) throws -> Decay {
		let decay: Decay = try make()
		decay.cell = cell
		decay.value = Data(count: cell.width*MemoryLayout<Float>.size)
		return decay
	}
}
