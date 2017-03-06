//
//  Edge.swift
//  macOS
//
//  Created by Kota Nakano on 1/27/17.
//
//

import CoreData
import Computer
import Distributor

internal class Edge: Arcane {
	fileprivate var ja: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>(array: [])
	fileprivate var jx: RingBuffer<(μ: Buffer, σ: Buffer)> = RingBuffer<(μ: Buffer, σ: Buffer)>(array: [])
}
extension Edge {
	override func setup() {
		let rows: Int = output.width
		let cols: Int = input.width
		ja = RingBuffer<(μ: Buffer, σ: Buffer)>(array: Array<Void>(repeating: (), count: 2).map {(
			μ: context.make(length: rows*cols*MemoryLayout<Float>.size, options: .storageModePrivate),
			σ: context.make(length: rows*cols*MemoryLayout<Float>.size, options: .storageModePrivate))
		})
		jx = RingBuffer<(μ: Buffer, σ: Buffer)>(array: Array<Void>(repeating: (), count: 2).map {(
			μ: context.make(length: rows*cols*MemoryLayout<Float>.size, options: .storageModePrivate),
			σ: context.make(length: rows*cols*MemoryLayout<Float>.size, options: .storageModePrivate))
		})
		super.setup()
	}
}
extension Edge {
	internal func collect_clear(commandBuffer: CommandBuffer, ignore: Set<Cell>) {
		input.collect_clear(ignore: ignore)
		refresh(commandBuffer: commandBuffer)
	}
	internal func collect(distributor: Distributor, Σ: (μ: Buffer, σ: Buffer), ignore: Set<Cell>) {
		let count: (rows: Int, cols: Int) = (rows: output.width, cols: input.width)
		let a: (μ: Buffer, σ: Buffer) = (μ: μ, σ: σ)
		let (χ, p): (χ: Buffer, p: Buffer) = input.collect(ignore: ignore)
		let commandBuffer: CommandBuffer = context.make()
		distributor.collect(commandBuffer: commandBuffer, Σ: Σ, w: a, x: χ, count: count)
		distributor.jacobian(commandBuffer: commandBuffer, Σ: ja.current, a: a, x: p, count: count, rtrl: false)
		distributor.jacobian(commandBuffer: commandBuffer, Σ: jx.current, x: p, a: a, count: count, rtrl: false)
		commandBuffer.commit()
	}
	internal func correct_clear(commandBuffer: CommandBuffer, ignore: Set<Cell>) {
		output.correct_clear(ignore: ignore)
		ja.progress()
		jx.progress()
		do {
			let encoder: BlitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
			[ja.current.μ, ja.current.σ, jx.current.μ, jx.current.σ].forEach {
				encoder.fill(buffer: $0, range: NSRange(location: 0, length: $0.length), value: 0)
			}
			encoder.endEncoding()
		}
	}
	internal func correct(distributor: Distributor, Σ: (χ: Buffer, Δ: Buffer), ignore: Set<Cell>) {
		let count: (rows: Int, cols: Int) = (rows: output.width, cols: input.width)
		let a: (μ: Buffer, σ: Buffer) = (μ: μ, σ: σ)
		let (χ, p): (χ: Buffer, p: Buffer) = input.collect(ignore: ignore)
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
