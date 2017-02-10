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
		output.distributor.shuffle(commandBuffer: commandBuffer, χ: χ, from: (μ: μ, σ: σ), count: output.width*input.width)
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
	/*
	internal func collect_clear(distributor: Distributor, ignore: Set<Cell>) {
		input.collect_clear(ignore: ignore)
		refresh()
		context.compute {
			distributor.shuffle(commandBuffer: $0, χ: χ, from: (μ: μ, σ: σ), count: count)
		}
	}
	internal func collect(distributor: Distributor, Σ: (χ: Buffer, μ: Buffer, σ: Buffer), ignore: Set<Cell>) {
	
	}
	*/
	/*
	internal func collect(commandBuffer: CommandBuffer, ignore: Set<Cell>) -> (χ: LaObjet, μ: LaObjet, σ: LaObjet) {
		let rows: Int = output.width
		let cols: Int = input.width
		let state: LaObjet = input.collect(group: group, ignore: ignore)
		return (χ: matrix_product(make(pointer: χ.contents(), rows: rows, cols: cols), state),
		        μ: matrix_product(distributor.collect(μ: make(pointer: μ.contents(), rows: rows, cols: cols)), distributor.collect(μ: state)),
		        σ: matrix_product(distributor.collect(σ: make(pointer: σ.contents(), rows: rows, cols: cols)), distributor.collect(σ: state))
		)
	}
	*/
	/*
	internal func collect(group: DispatchGroup, distributor: Distributor, ignore: Set<Cell>) -> (χ: LaObjet, μ: LaObjet, σ: LaObjet) {
		let rows: Int = output.width
		let cols: Int = input.width
		let state: LaObjet = input.collect(group: group, ignore: ignore)
		return (χ: matrix_product(make(pointer: χ.contents(), rows: rows, cols: cols), state),
		        μ: matrix_product(distributor.collect(μ: make(pointer: μ.contents(), rows: rows, cols: cols)), distributor.collect(μ: state)),
		        σ: matrix_product(distributor.collect(σ: make(pointer: σ.contents(), rows: rows, cols: cols)), distributor.collect(σ: state))
		)
	}
	*/
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
