//
//  Edge.swift
//  macOS
//
//  Created by Kota Nakano on 1/27/17.
//
//

import LaObjet
import CoreData
import Computer
import Distributor

internal class Edge: Arcane {
	
}
extension Edge {
	internal func collect_clear(commandBuffer: CommandBuffer, ignore: Set<Cell>) {
		input.collect_clear(ignore: ignore)
		refresh(commandBuffer: commandBuffer)
		output.distributor.shuffle(commandBuffer: commandBuffer, χ: χ, μ: μ, σ: σ, count: output.width*input.width)
	}
	internal func collect(commandBuffer: CommandBuffer, ignore: Set<Cell>) -> (χ: LaObjet, μ: LaObjet, σ: LaObjet) {
		let distributor: Distributor = output.distributor
		let rows: Int = output.width
		let cols: Int = input.width
		let state: LaObjet = input.collect(ignore: ignore)
		return (χ: matrix_product(make(nocopy: χ.contents(), rows: rows, cols: cols), state),
		        μ: matrix_product(distributor.μ(make(nocopy: μ.contents(), rows: rows, cols: cols)), distributor.μ(state)),
		        σ: matrix_product(distributor.σ(make(nocopy: σ.contents(), rows: rows, cols: cols)), distributor.σ(state))
		)
	}
	internal func correct_clear(commandBuffer: CommandBuffer, ignore: Set<Cell>) {
		output.correct_clear(ignore: ignore)
	}
	internal func correct(commandBuffer: CommandBuffer, ignore: Set<Cell>) -> LaObjet {
		let(Δμ, Δσ) = output.correct(ignore: ignore)
		return Δμ
	}
	override func setup() {
		super.setup()
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
