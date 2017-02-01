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
	internal func update() {
		context.mul(Δσ, Δσ, σ, σcount)
		func block(commandBuffer: CommandBuffer) {
			func willChange(_: CommandBuffer) {
				willChangeValue(forKey: type(of: self).locationkey)
				willChangeValue(forKey: type(of: self).logscalekey)
			}
			func didChange(_: CommandBuffer) {
				didChangeValue(forKey: type(of: self).locationkey)
				didChangeValue(forKey: type(of: self).logscalekey)
			}
			μoptimizer.encode(commandBuffer: commandBuffer, θ: μ, Δθ: Δμ)
			σoptimizer.encode(commandBuffer: commandBuffer, θ: σ, Δθ: Δσ)
			commandBuffer.addScheduledHandler(willChange)
			commandBuffer.addCompletedHandler(didChange)
		}
		context.compute(block)
	}
	internal func refresh() {
		context.exp(σ, logσ, σcount)
	}
	internal func setup() {
		μoptimizer = context.make(count: μcount)
		σoptimizer = context.make(count: σcount)
		μ = context.make(data: location, options: .storageModeShared)
		logσ = context.make(data: logscale, options: .storageModeShared)
		σ = context.make(length: logscale.count, options: .storageModePrivate)
		Δμ = context.make(length: location.count, options: .storageModePrivate)
		Δσ = context.make(length: logscale.count, options: .storageModePrivate)
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
	var μcount: Int {
		return location.count / MemoryLayout<Float>.size
	}
	var σcount: Int {
		return logscale.count / MemoryLayout<Float>.size
	}
}
extension Arcane {
	@NSManaged var location: Data
	@NSManaged var logscale: Data
}
