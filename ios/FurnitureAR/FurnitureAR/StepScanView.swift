import ARKit
import RealityKit
import SwiftUI

/// "Scan my step": the person taps the floor, lines a see-through ghost of the expected furniture up
/// with the real thing, walks around it while LiDAR scans, and gets the server's verdict drawn over
/// the furniture: matching parts green, missing or wrong ones red. Needs a LiDAR iPhone (Pro models).
struct StepScanView: View {
    let request: ScanRequest
    let server: String
    let close: (_ next: Bool) -> Void
    @StateObject private var scan = StepScanModel()

    var body: some View {
        ZStack {
            ARContainer(model: scan).ignoresSafeArea()
            VStack(spacing: 12) {
                header
                Spacer()
                if let part = scan.selected { partCard(part) }
                footer
            }
            .padding()
        }
        .task { await scan.start(request: request, server: server) }
        .onDisappear { scan.stop() }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Step \(request.step) · \(request.name)").font(.headline)
                Text(scan.hint).font(.subheadline).fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            Button { close(false) } label: {
                Image(systemName: "xmark").font(.headline).padding(10)
            }
            .background(.thinMaterial, in: Circle())
            .accessibilityLabel("Close")
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    @ViewBuilder private var footer: some View {
        switch scan.phase {
        case .loading, .placing:
            EmptyView()
        case .aligning:
            HStack {
                Button { scan.tipOver() } label: { Label("Tip over", systemImage: "rotate.3d") }
                    .buttonStyle(.bordered)
                Button { scan.startScan() } label: { Text("Start scan").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent)
            }
            .controlSize(.large)
            .padding(12)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        case .scanning(let progress):
            VStack(spacing: 8) {
                ProgressView(value: progress)
                Text("\(Int(ceil((1 - progress) * StepScanModel.scanSeconds))) s left · keep moving slowly")
                    .font(.footnote).monospacedDigit()
            }
            .padding(14)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        case .checking:
            HStack(spacing: 10) {
                ProgressView()
                Text("Checking your step…")
            }
            .padding(14)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        case .result(let result):
            resultCard(result)
        case .failed(let message):
            VStack(spacing: 10) {
                Text(message).multilineTextAlignment(.center)
                Button("Try again") { scan.restart() }.buttonStyle(.borderedProminent)
            }
            .padding(14)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
    }

    private func resultCard(_ result: CheckResult) -> some View {
        let problems = result.parts.filter { $0.status == "missing" || $0.status == "misplaced" }
        let resized = result.resized.map { $0 > 1 ? "Your furniture is \(Int(($0 - 1) * 100 + 0.5))% bigger than the model; the model was resized to fit."
                                                  : "Your furniture is \(Int((1 - $0) * 100 + 0.5))% smaller than the model; the model was resized to fit." }
        let unsure = result.parts.filter { $0.status == "unsure" }
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: result.verdict == "ok" ? "checkmark.circle.fill" :
                        result.verdict == "problems" ? "exclamationmark.triangle.fill" : "questionmark.circle.fill")
                    .foregroundStyle(result.verdict == "ok" ? .green : result.verdict == "problems" ? .red : .orange)
                Text(result.verdict == "ok" ? "This step looks right" :
                        result.verdict == "problems" ? "Something needs fixing" : "Couldn't check everything")
                    .font(.headline)
            }
            if let missing = result.missing, !missing.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Missing").font(.subheadline.bold()).foregroundStyle(.red)
                    ForEach(missing, id: \.self) { Text("• " + $0).font(.subheadline) }
                }
            }
            if let wrong = result.misplaced, !wrong.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Out of place").font(.subheadline.bold()).foregroundStyle(.red)
                    ForEach(wrong, id: \.self) { Text("• " + $0).font(.subheadline) }
                }
            }
            if let summary = result.summary { Text(summary).font(.subheadline) }
            if let advice = result.advice {
                Text(advice.summary).font(.subheadline)
                ForEach(advice.tips, id: \.self) { Text("• " + $0).font(.footnote) }
            }
            ForEach(Array(result.extras.enumerated()), id: \.offset) { Text("• " + $0.element.message).font(.footnote).foregroundStyle(.red) }
            if let resized { Text(resized).font(.footnote).foregroundStyle(.secondary) }
            if !unsure.isEmpty {
                Text("Not sure about: " + Self.plainList(unsure.map(\.name)).joined(separator: ", ")).font(.footnote).foregroundStyle(.secondary)
            }
            if !problems.isEmpty || !result.extras.isEmpty {
                Text("Green parts are right. Tap a red part to see what's wrong.").font(.footnote).foregroundStyle(.secondary)
            }
            HStack {
                Button { scan.restart() } label: { Text("Scan again").frame(maxWidth: .infinity) }
                    .buttonStyle(.bordered)
                Button { close(!request.isLast) } label: {
                    Text(request.isLast ? "Done" : "Next step").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }
            .controlSize(.large)
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    /// "Rod_2_1", "Rod_3_2" -> "2 x rod", like the server's missing list.
    static func plainList(_ names: [String]) -> [String] {
        var counts: [(String, Int)] = []
        for name in names {
            var plain = name.replacingOccurrences(of: #"([_\s]+\d+)+$"#, with: "", options: .regularExpression)
            plain = plain.replacingOccurrences(of: "_", with: " ").lowercased()
            if let i = counts.firstIndex(where: { $0.0 == plain }) { counts[i].1 += 1 } else { counts.append((plain, 1)) }
        }
        return counts.map { $0.1 > 1 ? "\($0.1) x \($0.0)" : $0.0 }
    }

    private func partCard(_ part: StepScanModel.Selection) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text(part.title).font(.headline)
                Text(part.message).font(.subheadline)
            }
            Spacer()
            Button { scan.selected = nil } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

private struct ARContainer: UIViewRepresentable {
    let model: StepScanModel
    func makeUIView(context: Context) -> ARView { model.arView }
    func updateUIView(_ view: ARView, context: Context) {}
}

@MainActor
final class StepScanModel: NSObject, ObservableObject {
    enum Phase {
        case loading, placing, aligning, scanning(Double), checking, result(CheckResult), failed(String)
    }
    struct Selection { let title: String; let message: String }

    static let scanSeconds = 20.0
    @Published var phase: Phase = .loading
    @Published var hint = "Loading the model…"
    @Published var selected: Selection?

    let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    private var request: ScanRequest!
    private var server = ""
    private var ghost: GhostResponse?
    private var anchor: AnchorEntity?
    private var handle: ModelEntity?           // what the person drags, twists and pinches
    private var turn = Entity()                // lays the model on its back or side
    private var modelNode = Entity()           // the parts, in model units
    private var resultAnchor: AnchorEntity?
    private var tipIndex = 0
    private var scanTask: Task<Void, Never>?
    private var result: CheckResult?
    private var tapInstalled = false
    private var fitted: simd_float4x4?           // where the last scan placed the model (with its size)
    private let cloud = DepthCloud()
    private var fitTask: Task<Void, Never>?
    private var liveGhost: simd_float4x4?        // where the live fit last moved the model

    private static let tips: [simd_quatf] = [
        simd_quatf(angle: 0, axis: [1, 0, 0]),
        simd_quatf(angle: -.pi / 2, axis: [1, 0, 0]),   // on its back
        simd_quatf(angle: .pi / 2, axis: [1, 0, 0]),    // on its front
        simd_quatf(angle: .pi / 2, axis: [0, 0, 1]),    // on its left side
        simd_quatf(angle: -.pi / 2, axis: [0, 0, 1]),   // on its right side
        simd_quatf(angle: .pi, axis: [1, 0, 0]),        // upside down
    ]

    func start(request: ScanRequest, server: String) async {
        self.request = request
        self.server = server
        guard ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) else {
            fail("Scanning needs an iPhone with LiDAR (a Pro model). This phone can't measure the furniture's shape.")
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        config.sceneReconstruction = .mesh
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)   // raw LiDAR depth: keeps rods and thin boards the mesh smooths away
        }
        arView.session.run(config)
        if !tapInstalled {
            arView.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(tapped(_:))))
            tapInstalled = true
        }
        do {
            ghost = try await StepCheckAPI.ghost(server: server, request: request)
            phase = .placing
            hint = "Point at the floor next to your furniture and tap it."
        } catch {
            fail("Couldn't load the model: \(error.localizedDescription)")
        }
    }

    func stop() {
        scanTask?.cancel()
        fitTask?.cancel()
        arView.session.pause()
    }

    private func fail(_ message: String) {
        phase = .failed(message)
        hint = "Something went wrong."
    }

    // MARK: placing and lining up the ghost

    @objc private func tapped(_ gesture: UITapGestureRecognizer) {
        let point = gesture.location(in: arView)
        switch phase {
        case .placing:
            guard let hit = arView.raycast(from: point, allowing: .estimatedPlane, alignment: .horizontal).first else {
                hint = "Couldn't find the floor there. Move the phone around a little and tap the floor again."
                return
            }
            place(at: Self.facingCamera(at: hit.worldTransform, camera: arView.cameraTransform.translation))
        case .result:
            var entity = arView.entity(at: point)
            while let current = entity, !current.name.hasPrefix("part-"), !current.name.hasPrefix("extra-") {
                entity = current.parent
            }
            guard let name = entity?.name, let result else { selected = nil; return }
            if name.hasPrefix("extra-"), let i = Int(name.dropFirst(6)), i < result.extras.count {
                selected = Selection(title: "Out of place", message: result.extras[i].message)
            } else if let id = Int(name.dropFirst(5)), let part = result.parts.first(where: { $0.id == id }) {
                let status = ["ok": "In place", "missing": "Missing", "misplaced": "Out of place", "unsure": "Not sure"][part.status] ?? part.status
                selected = Selection(title: "\(part.name) · \(status)", message: part.message)
            }
        default:
            break
        }
    }

    /// The tapped floor point, turned so the model's front (+z) faces the camera. A floor hit's own
    /// rotation follows ARKit's plane, which points any which way.
    private static func facingCamera(at hit: simd_float4x4, camera: SIMD3<Float>) -> simd_float4x4 {
        let spot = SIMD3(hit.columns.3.x, hit.columns.3.y, hit.columns.3.z)
        let toward = camera - spot
        let yaw = atan2(toward.x, toward.z)
        var m = simd_float4x4(simd_quatf(angle: yaw, axis: [0, 1, 0]))
        m.columns.3 = SIMD4(spot, 1)
        return m
    }

    private func place(at transform: simd_float4x4) {
        guard let ghost else { return }
        let anchor = AnchorEntity(world: transform)
        let handle = ModelEntity()
        modelNode = Entity()
        turn = Entity()
        for part in ghost.parts {
            let color: UIColor = part.active ? .systemOrange : .white
            if let entity = Self.partEntity(part, color: color, opacity: part.active ? 0.55 : 0.35) { modelNode.addChild(entity) }
        }
        // Real-size kits use their own scale; others start at a typical furniture size and get pinched to fit
        let scale = ghost.metres_per_unit ?? 1.2 / max(ghost.diagonal, 0.01)
        modelNode.scale = SIMD3(repeating: scale)
        turn.addChild(modelNode)
        handle.addChild(turn)
        anchor.addChild(handle)
        arView.scene.addAnchor(anchor)
        self.anchor = anchor
        self.handle = handle
        tipIndex = 0
        settle()
        let gestures: ARView.EntityGestures = [.translation, .rotation, .scale]   // pinch to match furniture of any size; the server fits the exact size
        arView.installGestures(gestures, for: handle)
        phase = .aligning
        hint = "Drag the ghost onto your furniture. It doesn't have to be exact: the scan works out the turn and size. Orange parts are this step's."
    }

    /// Tips the ghost onto its next side, for steps built lying down.
    func tipOver() {
        tipIndex = (tipIndex + 1) % Self.tips.count
        settle()
    }

    /// Stands the ghost on the floor, centred on the handle, after it is tipped.
    private func settle() {
        guard let handle else { return }
        turn.orientation = Self.tips[tipIndex]
        turn.position = .zero
        let bounds = modelNode.visualBounds(relativeTo: handle)
        turn.position = SIMD3(-bounds.center.x, -bounds.min.y, -bounds.center.z)
        let box = modelNode.visualBounds(relativeTo: handle)
        handle.collision = CollisionComponent(shapes: [ShapeResource.generateBox(size: box.extents).offsetBy(translation: box.center)])
    }

    private static func partEntity(_ part: GhostResponse.Part, color: UIColor, opacity: Float) -> ModelEntity? {
        guard let positionData = Data(base64Encoded: part.positions), let indexData = Data(base64Encoded: part.indices) else { return nil }
        let flat = positionData.array(of: Float.self)
        let indices = indexData.array(of: UInt32.self)
        guard flat.count >= 9, indices.count >= 3 else { return nil }
        let positions = stride(from: 0, to: flat.count - 2, by: 3).map { SIMD3(flat[$0], flat[$0 + 1], flat[$0 + 2]) }
        // both windings, so the see-through ghost shows its inside faces too
        var both = indices
        for t in stride(from: 0, to: indices.count - 2, by: 3) { both += [indices[t], indices[t + 2], indices[t + 1]] }
        var descriptor = MeshDescriptor(name: "part-\(part.id)")
        descriptor.positions = MeshBuffers.Positions(positions)
        descriptor.primitives = .triangles(both)
        guard let mesh = try? MeshResource.generate(from: [descriptor]) else { return nil }
        var material = UnlitMaterial(color: color)
        material.blending = .transparent(opacity: .init(floatLiteral: opacity))
        let entity = ModelEntity(mesh: mesh, materials: [material])
        entity.name = "part-\(part.id)"
        return entity
    }

    // MARK: scanning

    func startScan() {
        guard handle != nil else { return }
        // the ghost stays put while scanning
        arView.gestureRecognizers?.filter { $0 is EntityGestureRecognizer }.forEach { arView.removeGestureRecognizer($0) }
        arView.debugOptions.insert(.showSceneUnderstanding)   // shows the mesh building up as they walk
        hint = "Walk slowly all the way around your furniture, keeping it on screen. The ghost moves onto it as it's found."
        let area = scanArea()
        cloud.reset()
        liveGhost = nil
        scanTask = Task { [weak self] in
            let start = Date()
            var tick = 0
            while let self, !Task.isCancelled {
                let progress = min(1, Date().timeIntervalSince(start) / Self.scanSeconds)
                self.phase = .scanning(progress)
                if progress >= 1 { break }
                // five depth frames a second is plenty while walking slowly
                if tick % 2 == 0, let frame = self.arView.session.currentFrame, let depth = DepthCloud.frame(from: frame) {
                    self.cloud.add(depth, area: area)
                }
                tick += 1
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            guard let self, !Task.isCancelled else { return }
            await self.finishScan()
        }
        fitTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            while let self, !Task.isCancelled {
                await self.liveFit()
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    /// A round area around the ghost, not its box: a ghost turned the wrong way must not cut off
    /// part of the furniture (the server finds the right turn and size itself)
    private func scanArea() -> ScanArea {
        let box = modelNode.visualBounds(relativeTo: nil)
        let reach = simd_length(SIMD2(box.extents.x, box.extents.z)) / 2
        return ScanArea(centre: SIMD2(box.center.x, box.center.z), radius: 1.5 * reach + 0.4,
                        floor: box.min.y - 0.1, top: box.min.y + 2 * max(box.extents.y, reach) + 0.3)
    }

    /// Sends the scan so far to the server and slides, turns and resizes the ghost onto the real
    /// furniture when it fits. The first fit searches widely; later ones only nudge.
    private func liveFit() async {
        let points = await cloud.points(cell: 0.02, limit: 20_000)
        guard points.count >= 300, case .scanning = phase else { return }
        let ghost = liveGhost ?? modelNode.transformMatrix(relativeTo: nil)
        do {
            let fit = try await StepCheckAPI.fit(server: server, request: request, ghost: ghost, points: points, near: liveGhost != nil)
            guard case .scanning = phase, !Task.isCancelled, let matrix = Self.matrix(fit.transform), let handle else { return }
            liveGhost = matrix
            let inner = modelNode.transformMatrix(relativeTo: handle)
            handle.move(to: Transform(matrix: matrix * inner.inverse), relativeTo: nil, duration: 0.6, timingFunction: .easeInOut)
            hint = "Found your furniture: the ghost is on it now. Keep walking around it."
            log("live fit: size \(fit.size ?? 0), \(Int((fit.precision ?? 0) * 100))% on the model")
        } catch {
            log("live fit failed: \(error.localizedDescription)")
        }
    }

    private func finishScan() async {
        guard let frame = arView.session.currentFrame, handle != nil else { fail("The camera stopped. Try again."); return }
        fitTask?.cancel()
        phase = .checking
        hint = "Collecting the scan…"
        let started = Date()
        let ghostMatrix = liveGhost ?? modelNode.transformMatrix(relativeTo: nil)
        // Everything the background work needs, copied out of RealityKit and ARKit on the main thread
        let area = scanArea()
        let meshes = frame.anchors.compactMap { $0 as? ARMeshAnchor }
        let pixels = frame.capturedImage
        var points = await cloud.points()
        let photo = await Task.detached(priority: .userInitiated) { Self.photo(pixels) }.value
        log("finish: \(points.count) depth points")
        if points.count < 500 {   // no depth (or almost none): fall back to the scene mesh
            points = await Task.detached(priority: .userInitiated) { Self.scannedPoints(meshes: meshes, area: area) }.value
            log("using \(points.count) mesh points instead")
        }
        log("collected \(points.count) points, photo \(photo?.count ?? 0) bytes in \(Self.seconds(since: started))")
        guard points.count >= 100 else {
            fail("The scan picked up almost nothing near the ghost (\(points.count) points). Line the ghost up with your furniture and walk all the way around it.")
            return
        }
        hint = "Comparing your furniture with the model… up to a minute."
        do {
            let result = try await StepCheckAPI.check(server: server, request: request, ghost: ghostMatrix,
                                                      points: points, photo: photo, fitted: liveGhost != nil) { [weak self] status in
                Task { @MainActor in self?.hint = status }
            }
            log("result \(result.verdict) after \(Self.seconds(since: started))")
            show(result, fallback: ghostMatrix)
        } catch {
            log("check failed after \(Self.seconds(since: started)): \(error)")
            fail("The check failed: \(error.localizedDescription)")
        }
    }

    private func log(_ message: String) { print("[StepScan] " + message) }
    private static func seconds(since date: Date) -> String { String(format: "%.1f s", Date().timeIntervalSince(date)) }

    /// The LiDAR mesh vertices near the ghost, in world metres, thinned to one per centimetre.
    typealias ScanArea = DepthCloud.Area

    nonisolated private static func scannedPoints(meshes: [ARMeshAnchor], area: ScanArea) -> [SIMD3<Float>] {
        var seen = Set<SIMD3<Int32>>()
        var points = [SIMD3<Float>]()
        for mesh in meshes {
            let vertices = mesh.geometry.vertices
            let base = UnsafeRawPointer(vertices.buffer.contents())
            let toWorld = mesh.transform
            for i in 0..<vertices.count {
                // vertices are packed 12-byte float triples: read each float, never a 16-byte SIMD3
                let at = vertices.offset + vertices.stride * i
                let v = SIMD4<Float>(base.load(fromByteOffset: at, as: Float.self),
                                     base.load(fromByteOffset: at + 4, as: Float.self),
                                     base.load(fromByteOffset: at + 8, as: Float.self), 1)
                let world = toWorld * v
                guard world.y > area.floor, world.y < area.top,
                      simd_length(SIMD2(world.x, world.z) - area.centre) < area.radius else { continue }
                let key = SIMD3<Int32>(Int32((world.x * 100).rounded()), Int32((world.y * 100).rounded()), Int32((world.z * 100).rounded()))
                if seen.insert(key).inserted { points.append(SIMD3(world.x, world.y, world.z)) }
                if points.count >= 120_000 { return points }
            }
        }
        return points
    }

    /// One camera frame as a JPEG, upright, about 1000 pixels tall, for the screw check.
    nonisolated private static func photo(_ pixels: CVPixelBuffer) -> Data? {
        let image = CIImage(cvPixelBuffer: pixels).oriented(.right)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: 0.55, y: 0.55))
        guard let cg = CIContext().createCGImage(scaled, from: scaled.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.7)
    }

    // MARK: the verdict

    private func show(_ result: CheckResult, fallback: simd_float4x4) {
        self.result = result
        guard let ghost else { return }
        anchor?.isEnabled = false
        resultAnchor?.removeFromParent()
        let world = AnchorEntity(world: matrix_identity_float4x4)
        let node = Entity()
        node.transform = Transform(matrix: Self.matrix(result.transform) ?? fallback)
        fitted = Self.matrix(result.transform)
        let status = Dictionary(result.parts.map { ($0.id, $0.status) }, uniquingKeysWith: { a, _ in a })
        for part in ghost.parts where !part.fastener || status[part.id] == "missing" || status[part.id] == "misplaced" {
            let (color, opacity): (UIColor, Float) = switch status[part.id] {
            case "ok": (.systemGreen, 0.45)
            case "missing", "misplaced": (.systemRed, 0.7)
            default: (.systemYellow, 0.4)
            }
            if let entity = Self.partEntity(part, color: color, opacity: opacity) {
                entity.generateCollisionShapes(recursive: false)
                node.addChild(entity)
            }
        }
        for (i, extra) in result.extras.enumerated() where extra.lo.count == 3 && extra.hi.count == 3 {
            let lo = SIMD3(extra.lo[0], extra.lo[1], extra.lo[2]), hi = SIMD3(extra.hi[0], extra.hi[1], extra.hi[2])
            var material = UnlitMaterial(color: .systemRed)
            material.blending = .transparent(opacity: .init(floatLiteral: 0.6))
            let box = ModelEntity(mesh: .generateBox(size: hi - lo), materials: [material])
            box.position = (lo + hi) / 2
            box.name = "extra-\(i)"
            box.generateCollisionShapes(recursive: false)
            node.addChild(box)
        }
        world.addChild(node)
        arView.scene.addAnchor(world)
        resultAnchor = world
        phase = .result(result)
        hint = result.verdict == "ok" ? "Green parts match the model." : "Green parts match. Red ones are missing or out of place."
    }

    private static func matrix(_ values: [Float]?) -> simd_float4x4? {
        guard let v = values, v.count == 16 else { return nil }
        return simd_float4x4(SIMD4(v[0], v[1], v[2], v[3]), SIMD4(v[4], v[5], v[6], v[7]),
                             SIMD4(v[8], v[9], v[10], v[11]), SIMD4(v[12], v[13], v[14], v[15]))
    }

    /// Back to lining up the ghost, keeping where it was, for a fresh scan after a fix.
    func restart() {
        scanTask?.cancel()
        fitTask?.cancel()
        selected = nil
        result = nil
        resultAnchor?.removeFromParent()
        resultAnchor = nil
        arView.debugOptions.remove(.showSceneUnderstanding)
        if let anchor, let handle {
            anchor.isEnabled = true
            // start the next scan from where the last one put the model: the furniture's real place and size
            if let fitted {
                let inner = modelNode.transformMatrix(relativeTo: handle)
                handle.setTransformMatrix(fitted * inner.inverse, relativeTo: nil)
            }
            let gestures: ARView.EntityGestures = [.translation, .rotation, .scale]   // pinch to match furniture of any size; the server fits the exact size
            arView.installGestures(gestures, for: handle)
            phase = .aligning
            hint = "Check the ghost still sits over your furniture, then scan again."
        } else if ghost != nil {
            phase = .placing
            hint = "Point at the floor next to your furniture and tap it."
        } else {
            phase = .loading
            Task { await start(request: request, server: server) }
        }
    }
}
