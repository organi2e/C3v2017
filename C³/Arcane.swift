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
	struct Structs {
		let φ: Buffer
		let θ: Buffer
		let Δ: Buffer
		let adapter: Adapter
		let optimizer: Optimizer
		func refresh(commandBuffer: CommandBuffer) {
			adapter.generate(commandBuffer: commandBuffer, θ: θ, φ: φ)
		}
		func reset(commandBuffer: CommandBuffer) {
			optimizer.reset(commandBuffer: commandBuffer)
		}
		func update(commandBuffer: CommandBuffer) {
			adapter.gradient(commandBuffer: commandBuffer, Δ: Δ, θ: θ, φ: φ)
			optimizer.optimize(commandBuffer: commandBuffer, θ: φ, Δ: Δ)
		}
		var data: Data {
			return Data(bytesNoCopy: φ.contents(), count: φ.length, deallocator: .none)
		}
	}
	var μ: Structs!
	var σ: Structs!
}
extension Arcane {
	private static let locationkey: String = "location"
	private static let logscalekey: String = "logscale"
	internal func update(commandBuffer: CommandBuffer) {
		func will( _: CommandBuffer ) {
			willChangeValue(forKey: Arcane.locationkey)
			willChangeValue(forKey: Arcane.logscalekey)
		}
		func done( _: CommandBuffer ) {
			didChangeValue(forKey: Arcane.locationkey)
			didChangeValue(forKey: Arcane.logscalekey)
		}
		commandBuffer.addScheduledHandler(will)
		
		μ.update(commandBuffer: commandBuffer)
		σ.update(commandBuffer: commandBuffer)
		
		commandBuffer.addCompletedHandler(done)
		
	}
	internal func refresh(commandBuffer: CommandBuffer) {
		μ.refresh(commandBuffer: commandBuffer)
		σ.refresh(commandBuffer: commandBuffer)
		
	}
	internal func setup() {
		let commandBuffer: CommandBuffer = context.make()
		μ = Structs(φ: context.make(data: location, options: .storageModeShared),
		            θ: context.make(length: location.count, options: .storageModePrivate),
		            Δ: context.make(length: location.count, options: .storageModePrivate),
		            adapter: context.linFactory(location.count/MemoryLayout<Float>.size),
		            optimizer: context.optimizerFactory(location.count/MemoryLayout<Float>.size)
		)
		σ = Structs(φ: context.make(data: location, options: .storageModeShared),
		            θ: context.make(length: location.count, options: .storageModePrivate),
		            Δ: context.make(length: location.count, options: .storageModePrivate),
		            adapter: context.expFactory(location.count/MemoryLayout<Float>.size),
		            optimizer: context.optimizerFactory(location.count/MemoryLayout<Float>.size)
		)
		μ.reset(commandBuffer: commandBuffer)
		σ.reset(commandBuffer: commandBuffer)
		setPrimitiveValue(μ.data, forKey: Arcane.locationkey)
		setPrimitiveValue(σ.data, forKey: Arcane.logscalekey)
		commandBuffer.commit()
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
	internal var θ: (μ: Buffer, σ: Buffer) {
		return (μ: μ.θ, σ: σ.θ)
	}
	internal var Δ: (μ: Buffer, σ: Buffer) {
		return (μ: μ.Δ, σ: σ.Δ)
	}
}
extension Arcane {
	@NSManaged var location: Data
	@NSManaged var logscale: Data
}
