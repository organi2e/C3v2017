//
//  Context.swift
//  macOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal
import Math
import CoreData
import Distributor
import Optimizer

internal typealias Device = MTLDevice
internal typealias ResourceOptions = MTLResourceOptions
internal typealias CommandQueue = MTLCommandQueue
internal typealias CommandBuffer = MTLCommandBuffer
internal typealias Buffer = MTLBuffer
internal typealias ManagedObject = NSManagedObject

public class Context: NSManagedObjectContext {
	var layout: ResourceOptions?
	let device: Device
	let queue: CommandQueue
	let math: Math
	let gaussFactory: Distributor
	let optimizerFactory: (Int) -> Optimizer
	enum ErrorCase: Error, CustomStringConvertible {
		case InvalidContext
		case InvalidEntity(name: String)
		case NoModelFound
		case NoDeviceFound
		var description: String {
			switch self {
			case let .InvalidEntity(name):
				return "Invalid entity\(name)"
			case .InvalidContext:
				return "This context is invalid"
			case .NoModelFound:
				return "No CoreData definition was found"
			case .NoDeviceFound:
				return "No available Metal device found"
			}
		}
	}
	public init(storage: URL? = nil,
	            optimizer: (MTLDevice) throws -> (Int) -> Optimizer = SGD.factory(η: 1e-3),
	            concurrencyType: NSManagedObjectContextConcurrencyType = .privateQueueConcurrencyType) throws {
		guard let mtl: Device = MTLCreateSystemDefaultDevice() else { throw ErrorCase.NoDeviceFound }
		device = mtl
		queue = device.makeCommandQueue()
		math = try Math(device: device)
		gaussFactory = try GaussDistributor.factory()(device)
		optimizerFactory = try optimizer(device)
		super.init(concurrencyType: concurrencyType)
		guard let model: NSManagedObjectModel = NSManagedObjectModel.mergedModel(from: [Bundle(for: type(of: self))]) else { throw ErrorCase.NoModelFound }
		let store: NSPersistentStoreCoordinator = NSPersistentStoreCoordinator(managedObjectModel: model)
		let type: String = storage?.pathExtension == "sqlite" ? NSSQLiteStoreType : storage != nil ? NSBinaryStoreType : NSInMemoryStoreType
		try store.addPersistentStore(ofType: type, configurationName: nil, at: storage, options: nil)
		persistentStoreCoordinator = store
	}
	public required init?(coder aDecoder: NSCoder) {
		guard let mtl: Device = MTLCreateSystemDefaultDevice() else { fatalError(ErrorCase.NoDeviceFound.description) }
		device = mtl
		queue = device.makeCommandQueue()
		math = try!Math(device: device)
		gaussFactory = try!GaussDistributor.factory()(device)
		optimizerFactory = try!SGD.factory()(device)
		super.init(coder: aDecoder)
		fatalError()
	}
	public override func encode(with aCoder: NSCoder) {
		super.encode(with: aCoder)
	}
}
extension Context {
}
extension Context {
	internal func make(count: Int) -> Optimizer {
		return optimizerFactory(count)
	}
	internal func make() -> CommandBuffer {
		return queue.makeCommandBuffer()
	}
	internal func make<T>(array: Array<T>, options: ResourceOptions = []) -> Buffer {
		return device.makeBuffer(bytes: array, length: array.count*MemoryLayout<T>.size, options: layout ?? options)
	}
	internal func make(data: Data, options: ResourceOptions = []) -> Buffer {
		return device.makeBuffer(bytes: (data as NSData).bytes, length: data.count, options: layout ?? options)
	}
	internal func make(length: Int, options: ResourceOptions = []) -> Buffer {
		return device.makeBuffer(length: length, options: layout ?? options)
	}
	internal func make<T: ManagedObject>() throws -> T {
		let name: String = String(describing: T.self)
		guard let entity: T = NSEntityDescription.insertNewObject(forEntityName: name, into: self)as?T else { throw ErrorCase.InvalidEntity(name: name) }
		return entity
	}
	internal func sync() {
		let commandBuffer: CommandBuffer = make()
		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()
	}
}
extension NSManagedObject {
	internal var context: Context {
		guard let context: Context = managedObjectContext as? Context else { fatalError(Context.ErrorCase.InvalidContext.description) }
		return context
	}
}
extension MTLBuffer {
	internal var buffer: UnsafeMutableBufferPointer<Float> {
		return UnsafeMutableBufferPointer<Float>(start: UnsafeMutablePointer<Float>(OpaquePointer(contents())), count: length/MemoryLayout<Float>.size)
	}
}
