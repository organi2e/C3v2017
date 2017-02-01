//
//  DistributorTests.swift
//  DistributorTests
//
//  Created by Kota Nakano on 1/28/17.
//
//

import XCTest
import Accelerate
import Distributor

class DistributorTests: XCTestCase {
	
	func testGaussCDF() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 65536
			let distributor: Distributior = try Gauss(device: device)
			let χ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let μ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let σ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			
			vDSP_vfill([Float(1)], UnsafeMutablePointer<Float>(OpaquePointer(σ.contents())), 1, vDSP_Length(count))
			vDSP_vgen([Float(-10)], [Float(10)], UnsafeMutablePointer<Float>(OpaquePointer(μ.contents())), 1, vDSP_Length(count))
			
			
			let queue: MTLCommandQueue = device.makeCommandQueue()
			measure {
				for _ in 0..<1024 {
					let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
					distributor.encode(commandBuffer: commandBuffer, CDF: χ, μ: μ, σ: σ)
					commandBuffer.commit()
				}
			}
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			let χref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(χ.contents()))
			let μref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(μ.contents()))
			let σref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(σ.contents()))
			
			let ybuf: Array<Float> = (0..<count).map {
				0.5+0.5*erf(μref[$0]/σref[$0]*Float(M_SQRT1_2))
			}
			let E: Array<Float> = (0..<count).map {
				ybuf[$0] - χref[$0]
			}
			let MSE: Float = E.reduce(Float(0)) {
				$0.0 + ( $0.1 * $0.1 )
				} / Float(E.count)
			let RMSE: Float = sqrt(MSE)
			
			//try Data(bytes: χ.contents(), count: χ.length).write(to: URL(fileURLWithPath: "/tmp/x.raw"))
			//try Data(bytes: ybuf, count: ybuf.count*MemoryLayout<Float>.size).write(to: URL(fileURLWithPath: "/tmp/y.raw"))
			
			XCTAssert(RMSE<1e-6)
			
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	func testGaussPDF() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 65536
			let distributor: Distributior = try Gauss(device: device)
			let χ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let μ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let σ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			
			vDSP_vfill([Float(1)], UnsafeMutablePointer<Float>(OpaquePointer(σ.contents())), 1, vDSP_Length(count))
			vDSP_vgen([Float(-10)], [Float(10)], UnsafeMutablePointer<Float>(OpaquePointer(μ.contents())), 1, vDSP_Length(count))
			
			
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			
			distributor.encode(commandBuffer: commandBuffer, PDF: χ, μ: μ, σ: σ)
			
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			let χref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(χ.contents()))
			let μref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(μ.contents()))
			let σref: UnsafeMutablePointer<Float> = UnsafeMutablePointer<Float>(OpaquePointer(σ.contents()))
			
			let ybuf: Array<Float> = (0..<count).map {
				exp(-0.5*μref[$0]*μref[$0]/σref[$0]/σref[$0])/σref[$0]*Float(0.5*M_SQRT1_2*M_2_SQRTPI)
			}
			let E: Array<Float> = (0..<count).map {
				ybuf[$0] - χref[$0]
			}
			let MSE: Float = E.reduce(Float(0)) {
				$0.0 + ( $0.1 * $0.1 )
				} / Float(E.count)
			let RMSE: Float = sqrt(MSE)
			
			//try Data(bytes: χ.contents(), count: χ.length).write(to: URL(fileURLWithPath: "/tmp/x.raw"))
			//try Data(bytes: ybuf, count: ybuf.count*MemoryLayout<Float>.size).write(to: URL(fileURLWithPath: "/tmp/y.raw"))
			
			XCTAssert(RMSE<1e-6)
			
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	func testGaussRNG() {
		guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
		do {
			let count: Int = 65536
			var dstμ: Float = 100
			var estμ: Float = 0
			var dstσ: Float = 10.0
			var estσ: Float = 0
			let χ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let μ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			let σ: MTLBuffer = device.makeBuffer(length: MemoryLayout<Float>.size*count, options: .storageModeShared)
			
			vDSP_vfill(&dstμ, UnsafeMutablePointer<Float>(OpaquePointer(μ.contents())), 1, vDSP_Length(count))
			vDSP_vfill(&dstσ, UnsafeMutablePointer<Float>(OpaquePointer(σ.contents())), 1, vDSP_Length(count))
			
			let queue: MTLCommandQueue = device.makeCommandQueue()
			let commandBuffer: MTLCommandBuffer = queue.makeCommandBuffer()
			try Gauss(device: device).encode(commandBuffer: commandBuffer, χ: χ, μ: μ, σ: σ)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
			
			vDSP_normalize(UnsafePointer<Float>(OpaquePointer(χ.contents())), 1, nil, 1, &estμ, &estσ, vDSP_Length(count))
			
			XCTAssert(fabs(dstμ-estμ)/dstμ<1e-3)
			XCTAssert(fabs(dstσ-estσ)/dstσ<1e-2)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
}
