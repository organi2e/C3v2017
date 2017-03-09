//
//  Arcane.swift
//  macOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import CoreData
import Adapter
import Optimizer

public class Arcane: NSManagedObject {
	var μ: Buffer!
	var σ: Buffer!
	var Δμ: Buffer!
	var Δσ: Buffer!
	var logμ: Buffer!
	var logσ: Buffer!
	var μoptimizer: Optimizer!
	var σoptimizer: Optimizer!
	var μadapter: Adapter!
	var σadapter: Adapter!
}
extension Arcane {
	private static let locationkey: String = "location"
	private static let logscalekey: String = "logscale"
	internal func update(commandBuffer: CommandBuffer) {
		let count: Int = min(location.count, logscale.count) / MemoryLayout<Float>.size
		func willChange(_: CommandBuffer) {
			willChangeValue(forKey: Arcane.locationkey)
			willChangeValue(forKey: Arcane.logscalekey)
		}
		func didChange(_: CommandBuffer) {
			didChangeValue(forKey: Arcane.locationkey)
			didChangeValue(forKey: Arcane.logscalekey)
		}
		commandBuffer.addScheduledHandler(willChange)
		
		μadapter.gradient(commandBuffer: commandBuffer, Δ: Δμ, θ: μ, φ: logμ)
		σadapter.gradient(commandBuffer: commandBuffer, Δ: Δσ, θ: σ, φ: logσ)
		
		μoptimizer.optimize(commandBuffer: commandBuffer, θ: logμ, Δ: Δμ)
		σoptimizer.optimize(commandBuffer: commandBuffer, θ: logσ, Δ: Δσ)
		
		commandBuffer.addCompletedHandler(didChange)
	}
	internal func refresh(commandBuffer: CommandBuffer) {
		//context.math.copy(commandBuffer: commandBuffer, target: (μ, 0), source: (logμ, 0), count: min(μ.length, logμ.length))
		//context.math.exp(commandBuffer: commandBuffer, y: σ, x: logσ, count: min(σ.length, logσ.length)/MemoryLayout<Float>.size)
		μadapter.generate(commandBuffer: commandBuffer, θ: μ, φ: logμ)
		σadapter.generate(commandBuffer: commandBuffer, θ: σ, φ: logσ)
		
	}
	internal func setup() {
		
		let count: Int = min(location.count, logscale.count) / MemoryLayout<Float>.size
		
		μoptimizer = context.make(count: count)
		σoptimizer = context.make(count: count)
		
		μadapter = context.lassoFactory(count)
		σadapter = context.expFactory(count)
		
		μ = context.make(length: location.count, options: .storageModePrivate)
		σ = context.make(length: logscale.count, options: .storageModePrivate)
		
		Δμ = context.make(length: location.count, options: .storageModePrivate)
		Δσ = context.make(length: logscale.count, options: .storageModePrivate)
		
		logμ = context.make(data: location, options: .storageModeShared)
		logσ = context.make(data: logscale, options: .storageModeShared)
		
		setPrimitiveValue(Data(bytesNoCopy: logμ.contents(), count: location.count, deallocator: .none), forKey: Arcane.locationkey)
		setPrimitiveValue(Data(bytesNoCopy: logσ.contents(), count: logscale.count, deallocator: .none), forKey: Arcane.logscalekey)
		
	}
}
extension Arcane {
	public override func awakeFromFetch() {
		super.awakeFromFetch()
		setup()
	}
	public override func awake(fromSnapshotEvents flags: NSSnapshotEventType) {
		super.awake(fromSnapshotEvents: flags)
		setup()
	}
}
extension Arcane {
	internal var χ: (μ: Buffer, σ: Buffer) {
		return (μ: μ, σ: σ)
	}
	internal var Δ: (μ: Buffer, σ: Buffer) {
		return (μ: Δμ, σ: Δσ)
	}
}
extension Arcane {
	@NSManaged var location: Data
	@NSManaged var logscale: Data
}
