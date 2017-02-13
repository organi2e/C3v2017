//
//  Arcane.swift
//  macOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import CoreData
import Computer
import Optimizer
public class Arcane: ManagedObject {
	var χ: Buffer!
	var μ: Buffer!
	var σ: Buffer!
	var Δμ: Buffer!
	var Δσ: Buffer!
	var logσ: Buffer!
	var μoptimizer: Optimizer!
	var σoptimizer: Optimizer!
}
extension Arcane {
	private static let locationkey: String = "location"
	private static let logscalekey: String = "logscale"
	internal func update(commandBuffer: CommandBuffer) {
		func willChange(_: CommandBuffer) {
			willChangeValue(forKey: type(of: self).locationkey)
			willChangeValue(forKey: type(of: self).logscalekey)
		}
		func didChange(_: CommandBuffer) {
			didChangeValue(forKey: type(of: self).locationkey)
			didChangeValue(forKey: type(of: self).logscalekey)
		}
		context.computer.mul(commandBuffer: commandBuffer, z: Δσ, y: Δσ, x: σ, count: count)
		commandBuffer.addScheduledHandler(willChange)
		μoptimizer.optimize(commandBuffer: commandBuffer, θ: μ, Δ: Δμ)
		σoptimizer.optimize(commandBuffer: commandBuffer, θ: σ, Δ: Δσ)
		commandBuffer.addCompletedHandler(didChange)
	}
	internal func refresh(commandBuffer: MTLCommandBuffer) {
		context.computer.exp(commandBuffer: commandBuffer, y: σ, x: logσ, count: count)
	}
	internal func setup() {
		μoptimizer = context.make(count: count)
		σoptimizer = context.make(count: count)
		χ = context.make(length: length, options: .storageModeShared)
		σ = context.make(length: length, options: .storageModeShared)
		Δμ = context.make(length: length, options: .storageModeShared)
		Δσ = context.make(length: length, options: .storageModeShared)
		μ = context.make(data: location, options: .storageModeShared)
		logσ = context.make(data: logscale, options: .storageModeShared)
		setPrimitiveValue(Data(bytesNoCopy: μ.contents(), count: location.count, deallocator: .none), forKey: type(of: self).locationkey)
		setPrimitiveValue(Data(bytesNoCopy: logσ.contents(), count: logscale.count, deallocator: .none), forKey: type(of: self).logscalekey)
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
	internal var count: Int {
		return length / MemoryLayout<Float>.size
	}
	internal var length: Int {
		return min(location.count, logscale.count)
	}
	internal var value: (χ: Buffer, μ: Buffer, σ: Buffer) {
		return (χ: χ, μ: μ, σ: σ)
	}
}
extension Arcane {
	@NSManaged var location: Data
	@NSManaged var logscale: Data
}
