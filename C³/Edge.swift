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
	var j: RingBuffer<(a: (μ: Buffer, σ: Buffer), x: (μ: Buffer, σ: Buffer))> = RingBuffer<(a: (μ: Buffer, σ: Buffer), x: (μ: Buffer, σ: Buffer))>(array: [])
}
extension Edge {
	override func setup() {
		j = RingBuffer<(a: (μ: Buffer, σ: Buffer), x: (μ: Buffer, σ: Buffer))>(array: Array<Void>(repeating: (), count: 2).map {(
			a: (μ: context.make(length: rows*cols*MemoryLayout<Float>.size, options: .storageModePrivate),
			    σ: context.make(length: rows*cols*MemoryLayout<Float>.size, options: .storageModePrivate)),
			x: (μ: context.make(length: rows*cols*MemoryLayout<Float>.size, options: .storageModePrivate),
			    σ: context.make(length: rows*cols*MemoryLayout<Float>.size, options: .storageModePrivate))
			)}
		)
		super.setup()
	}
}
extension Edge {
	internal func collect_clear(ignore: Set<Cell>) {
		let commandBuffer: CommandBuffer = context.make()
		input.collect_clear(ignore: ignore)
		refresh(commandBuffer: commandBuffer)
		commandBuffer.commit()
	}
	internal func collect(distributor: Distributor, Σ: (μ: Buffer, σ: Buffer), ignore: Set<Cell>) {
		let x: Buffer = input.collect(ignore: ignore)
		let commandBuffer: CommandBuffer = context.make()
		distributor.collect(commandBuffer: commandBuffer, Σ: Σ, w: χ, x: x, count: shape)
		distributor.jacobian(commandBuffer: commandBuffer, Σ: j.current.a, a: χ, x: x, count: shape, rtrl: false)
		distributor.jacobian(commandBuffer: commandBuffer, Σ: j.current.x, x: x, a: χ, count: shape, rtrl: false)
		commandBuffer.commit()
	}
	internal func correct_clear(ignore: Set<Cell>) {
		let commandBuffer: CommandBuffer = context.make()
		output.correct_clear(ignore: ignore)
		j.progress()
		[j.current.a.μ, j.current.a.σ, j.current.x.μ, j.current.x.σ].forEach {
			context.math.fill(commandBuffer: commandBuffer, target: ($0, 0), value: 0, count: $0.length)
		}
		commandBuffer.commit()
	}
	internal func correct(distributor: Distributor, Σ: Buffer, ignore: Set<Cell>) {
		let (g, v): ((μ: Buffer, σ: Buffer),(μ: Buffer, σ: Buffer)) = output.correct(ignore: ignore)
		let commandBuffer: CommandBuffer = context.make()
		distributor.jacobian(commandBuffer: commandBuffer, j: j.current.x, v: v, Σ: j.current.x, count: shape, rtrl: false)
		distributor.jacobian(commandBuffer: commandBuffer, j: j.current.a, v: v, Σ: j.current.a, count: shape, rtrl: false)
		distributor.delta(commandBuffer: commandBuffer, Δ: Σ, j: j.current.x, g: g, count: shape)
		distributor.delta(commandBuffer: commandBuffer, Δ: Δ, j: j.current.a, g: g, count: shape, rtrl: false)
		update(commandBuffer: commandBuffer)
		commandBuffer.commit()
	}
}
extension Edge {
	@NSManaged var input: Cell
	@NSManaged var output: Cell
	var shape: (rows: Int, cols: Int) {
		return (rows: rows, cols: cols)
	}
	var rows: Int {
		return output.width
	}
	var cols: Int {
		return input.width
	}
}
extension Context {
	@nonobjc internal func make(output: Cell, input: Cell) throws -> Edge {
		let edge: Edge = try make()
		edge.output = output
		edge.input = input
		edge.location = Data(count: output.width*input.width*MemoryLayout<Float>.size)
		edge.logscale = Data(count: output.width*input.width*MemoryLayout<Float>.size)
		edge.setup()
		return edge
	}
}
