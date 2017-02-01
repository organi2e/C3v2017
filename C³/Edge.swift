//
//  Edge.swift
//  macOS
//
//  Created by Kota Nakano on 1/27/17.
//
//

import CoreData
import Distributor

internal class Edge: Arcane {
	
}
extension Edge {
	internal func collect_clear(distributor: Distributor, ignore: Set<Cell>) {
		refresh()
		context.compute {
			distributor.encode(commandBuffer: $0, χ: χ, μ: μ, σ: σ, count: count)
		}
	}
	internal func collect(distributor: Distributor, ignore: Set<Cell>) {
		
	}
}
extension Edge {
	@NSManaged var input: Cell
	@NSManaged var output: Cell
}
extension Context {
	@nonobjc internal func make(output: Cell, input: Cell) throws -> Edge {
		let edge: Edge = try make()
		edge.output = output
		edge.input = input
		edge.location = Data(count: MemoryLayout<Float>.size*output.width*input.width)
		edge.logscale = Data(count: MemoryLayout<Float>.size*output.width*input.width)
		edge.setup()
		return edge
	}
}
