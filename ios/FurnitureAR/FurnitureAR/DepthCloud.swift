import ARKit
import simd

/// Points from the raw LiDAR depth of each camera frame, gathered while the person walks around.
/// ARKit's scene mesh smooths away anything thinner than a few centimetres (rods vanish, boards on
/// the floor melt into it); the depth map keeps them. Points fall into 1 cm cells, and only cells
/// seen in at least two frames count, which drops the odd stray depth reading.
final class DepthCloud: @unchecked Sendable {
    /// One frame's depth, copied out of ARKit so the frame itself is released at once.
    struct Frame {
        let depth: [Float32]
        let confidence: [UInt8]
        let width: Int
        let height: Int
        let fx: Float, fy: Float, cx: Float, cy: Float   // intrinsics at the depth map's size
        let cameraToWorld: simd_float4x4
    }
    struct Area { let centre: SIMD2<Float>; let radius: Float; let floor: Float; let top: Float }

    private struct Cell { var sum: SIMD3<Float>; var hits: UInt16 }
    private var cells: [SIMD3<Int32>: Cell] = [:]
    private let queue = DispatchQueue(label: "furnisure.depthcloud", qos: .userInitiated)
    private static let cellSize: Float = 0.01
    private static let maxCells = 400_000

    /// Copies the depth out of an ARKit frame. Call on the main thread.
    static func frame(from frame: ARFrame) -> Frame? {
        guard let scene = frame.sceneDepth, let confidenceMap = scene.confidenceMap else { return nil }
        let map = scene.depthMap
        let width = CVPixelBufferGetWidth(map), height = CVPixelBufferGetHeight(map)
        guard let depth: [Float32] = copy(map, width: width, height: height),
              let confidence: [UInt8] = copy(confidenceMap, width: width, height: height) else { return nil }
        // intrinsics are for the full camera image; the depth map is the same view, smaller
        let k = frame.camera.intrinsics
        let sx = Float(width) / Float(frame.camera.imageResolution.width)
        let sy = Float(height) / Float(frame.camera.imageResolution.height)
        return Frame(depth: depth, confidence: confidence, width: width, height: height,
                     fx: k.columns.0.x * sx, fy: k.columns.1.y * sy, cx: k.columns.2.x * sx, cy: k.columns.2.y * sy,
                     cameraToWorld: frame.camera.transform)
    }

    private static func copy<T>(_ buffer: CVPixelBuffer, width: Int, height: Int) -> [T]? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        var out = [T]()
        out.reserveCapacity(width * height)
        for row in 0..<height {
            let line = (base + row * rowBytes).assumingMemoryBound(to: T.self)
            out.append(contentsOf: UnsafeBufferPointer(start: line, count: width))
        }
        return out
    }

    func reset() { queue.async { self.cells.removeAll(keepingCapacity: true) } }

    /// Adds a frame's points inside `area` (world metres). Work happens off the main thread.
    func add(_ frame: Frame, area: Area) {
        queue.async {
            guard self.cells.count < Self.maxCells else { return }
            let m = frame.cameraToWorld
            for v in 0..<frame.height {
                for u in 0..<frame.width {
                    let i = v * frame.width + u
                    // 0 low, 1 medium, 2 high; medium keeps dark boards, the two-frame rule keeps it clean
                    guard frame.confidence[i] >= 1 else { continue }
                    let d = frame.depth[i]
                    guard d > 0.15, d < 3.0 else { continue }
                    // camera space: x right, y up, looking down -z; image rows run downwards
                    let x = (Float(u) + 0.5 - frame.cx) * d / frame.fx
                    let y = -(Float(v) + 0.5 - frame.cy) * d / frame.fy
                    let w = m * SIMD4(x, y, -d, 1)
                    guard w.y > area.floor, w.y < area.top,
                          simd_length(SIMD2(w.x, w.z) - area.centre) < area.radius else { continue }
                    let key = SIMD3<Int32>(Int32((w.x / Self.cellSize).rounded(.down)),
                                           Int32((w.y / Self.cellSize).rounded(.down)),
                                           Int32((w.z / Self.cellSize).rounded(.down)))
                    let p = SIMD3(w.x, w.y, w.z)
                    if var cell = self.cells[key] {
                        if cell.hits < .max { cell.sum += p; cell.hits += 1 }
                        self.cells[key] = cell
                    } else {
                        self.cells[key] = Cell(sum: p, hits: 1)
                    }
                }
            }
        }
    }

    /// The points seen in at least two frames, one per `cell` metres (at least 1 cm), at most `limit`.
    func points(cell: Float = 0.01, limit: Int = 150_000) async -> [SIMD3<Float>] {
        await withCheckedContinuation { done in
            queue.async {
                var out = [SIMD3<Float>]()
                if cell <= Self.cellSize {
                    out.reserveCapacity(self.cells.count)
                    for c in self.cells.values where c.hits >= 2 { out.append(c.sum / Float(c.hits)) }
                } else {
                    var coarse = [SIMD3<Int32>: SIMD3<Float>]()
                    for c in self.cells.values where c.hits >= 2 {
                        let p = c.sum / Float(c.hits)
                        coarse[SIMD3<Int32>(Int32((p.x / cell).rounded(.down)), Int32((p.y / cell).rounded(.down)),
                                            Int32((p.z / cell).rounded(.down)))] = p
                    }
                    out = Array(coarse.values)
                }
                if out.count > limit {
                    let step = Double(out.count) / Double(limit)
                    out = (0..<limit).map { out[Int(Double($0) * step)] }
                }
                done.resume(returning: out)
            }
        }
    }
}
