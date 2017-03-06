//
//  Context.swift
//  macOS
//
//  Created by Kota Nakano on 2017/01/27.
//
//

import Metal
import CoreData
import Computer
import Distributor
import Optimizer

public class Context: NSManagedObjectContext {
	let computer: Computer
	let gaussFactory: Distributor
	let optimizerFactory: (Int) -> Optimizer
	enum ErrorCases: Error, CustomStringConvertible {
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
	public init(storage: URL? = nil, device: MTLDevice? = nil, concurrencyType: NSManagedObjectContextConcurrencyType = .privateQueueConcurrencyType) throws {
		guard let device: MTLDevice = device ?? MTLCreateSystemDefaultDevice() else { throw ErrorCases.NoDeviceFound }
		computer = try Computer(device: device)
		gaussFactory = try GaussDistributor.factory()(device)
		optimizerFactory = try SMORMS3.factory()(device)
		super.init(concurrencyType: concurrencyType)
		guard let model: NSManagedObjectModel = NSManagedObjectModel.mergedModel(from: [Bundle(for: type(of: self))]) else { throw ErrorCases.NoModelFound }
		let store: NSPersistentStoreCoordinator = NSPersistentStoreCoordinator(managedObjectModel: model)
		let type: String = storage?.pathExtension == "sqlite" ? NSSQLiteStoreType : storage != nil ? NSBinaryStoreType : NSInMemoryStoreType
		try store.addPersistentStore(ofType: type, configurationName: nil, at: storage, options: nil)
		persistentStoreCoordinator = store
	}
	public required init?(coder aDecoder: NSCoder) {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { fatalError(ErrorCases.NoDeviceFound.description) }
		computer = try!Computer(device: device)
		gaussFactory = try!GaussDistributor.factory()(device)
		optimizerFactory = try!SMORMS3.factory()(device)
		super.init(coder: aDecoder)
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
		return computer.make()
	}
	internal func make<T>(array: Array<T>, options: ResourceOptions = []) -> Buffer {
		return computer.make(array: array, options: options)
	}
	internal func make(data: Data, options: ResourceOptions = []) -> Buffer {
		return computer.make(data: data, options: options)
	}
	internal func make(length: Int, options: ResourceOptions = []) -> Buffer {
		return computer.make(length: length, options: options)
	}
	internal func make<T: NSManagedObject>() throws -> T {
		let name: String = String(describing: T.self)
		guard let entity: T = NSEntityDescription.insertNewObject(forEntityName: name, into: self)as?T else { throw ErrorCases.InvalidEntity(name: name) }
		return entity
	}
}
public typealias ManagedObject = NSManagedObject
extension ManagedObject {
	internal var context: Context {
		guard let context: Context = managedObjectContext as? Context else { fatalError(Context.ErrorCases.InvalidContext.description) }
		return context
	}
}
extension MTLBuffer {
	internal var buffer: UnsafeMutableBufferPointer<Float> {
		return UnsafeMutableBufferPointer<Float>(start: UnsafeMutablePointer<Float>(OpaquePointer(contents())), count: length/MemoryLayout<Float>.size)
	}
}
