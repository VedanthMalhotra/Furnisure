import Foundation
import simd

/// What the kit player sends when the person taps "Scan my step".
struct ScanRequest: Decodable, Identifiable {
    struct Hardware: Codable { let name: String; let count: Int }
    let id = UUID()
    let kit: String
    let name: String
    let step: String
    let isLast: Bool
    let captions: [String]
    let installed: [Int]
    let active: [Int]
    let future: [Int]
    let hardware: [Hardware]

    enum CodingKeys: String, CodingKey {
        case kit, name, step, isLast, captions, installed, active, future, hardware
    }
}

/// The parts expected after the step, from GET /api/stepcheck/{kit}/ghost.
struct GhostResponse: Decodable {
    struct Part: Decodable {
        let id: Int
        let name: String
        let fastener: Bool
        let active: Bool
        let positions: String   // base64 float32 x,y,z, model units
        let indices: String     // base64 uint32 triangles
    }
    let parts: [Part]
    let lo: [Float]
    let hi: [Float]
    let metres_per_unit: Float?   // nil: the model is not at real size, so the person pinches to fit
    let diagonal: Float
}

/// The verdict from POST /api/stepcheck/{kit}.
struct CheckResult: Decodable {
    struct Part: Decodable, Identifiable {
        let id: Int
        let name: String
        let status: String     // ok, missing, misplaced, unsure, hardware
        let message: String
        let active: Bool?
    }
    struct Extra: Decodable {
        let lo: [Float]
        let hi: [Float]
        let message: String
    }
    struct Advice: Decodable {
        let screws: String
        let summary: String
        let tips: [String]
    }
    let verdict: String        // ok, problems, unsure
    let summary: String?
    let transform: [Float]?    // model units -> world metres, column-major
    let parts: [Part]
    let extras: [Extra]
    let advice: Advice?
    let missing: [String]?     // short plain names, e.g. "back panel", "2 x rod"
    let misplaced: [String]?   // on but wrong, e.g. "top right panel slanted"
    let resized: Float?        // real furniture size / model size, when they differ by more than 5%
}

private struct CheckBody: Encodable {
    let step: String
    let captions: [String]
    let installed: [Int]
    let active: [Int]
    let future: [Int]
    let hardware: [ScanRequest.Hardware]
    let ghost: [Float]
    let points: String
    let photo: String?
    let fitted: Bool
}

enum StepCheckAPI {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    static func ghost(server: String, request: ScanRequest) async throws -> GhostResponse {
        let nodes = (request.installed + request.active).map(String.init).joined(separator: ",")
        let active = request.active.map(String.init).joined(separator: ",")
        let path = "/api/stepcheck/\(request.kit.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? request.kit)/ghost"
        guard var parts = URLComponents(string: server + path) else { throw Failure(message: "Bad server address") }
        parts.queryItems = [URLQueryItem(name: "nodes", value: nodes), URLQueryItem(name: "active", value: active)]
        let (data, response) = try await URLSession.shared.data(from: parts.url!)
        try check(response, data)
        return try JSONDecoder().decode(GhostResponse.self, from: data)
    }

    /// One session with a hard limit on the whole exchange, not just on each quiet spell
    private static let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 90
        config.timeoutIntervalForResource = 150
        config.waitsForConnectivity = false
        return URLSession(configuration: config)
    }()

    static func check(server: String, request: ScanRequest, ghost: simd_float4x4,
                      points: [SIMD3<Float>], photo: Data?, fitted: Bool, status: @escaping (String) -> Void) async throws -> CheckResult {
        var flat = [Float]()
        flat.reserveCapacity(points.count * 3)
        for p in points { flat.append(p.x); flat.append(p.y); flat.append(p.z) }
        let pointData = flat.withUnsafeBufferPointer { Data(buffer: $0) }
        let columns = [ghost.columns.0, ghost.columns.1, ghost.columns.2, ghost.columns.3]
        let body = CheckBody(step: request.step, captions: request.captions, installed: request.installed,
                             active: request.active, future: request.future, hardware: request.hardware,
                             ghost: columns.flatMap { [$0.x, $0.y, $0.z, $0.w] },
                             points: pointData.base64EncodedString(), photo: photo?.base64EncodedString(), fitted: fitted)
        let path = "/api/stepcheck/\(request.kit.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? request.kit)"
        guard let url = URL(string: server + path) else { throw Failure(message: "Bad server address") }
        var http = URLRequest(url: url, timeoutInterval: 90)
        http.httpMethod = "POST"
        http.setValue("application/json", forHTTPHeaderField: "Content-Type")
        http.httpBody = try JSONEncoder().encode(body)
        let megabytes = String(format: "%.1f", Double(http.httpBody?.count ?? 0) / 1_000_000)
        print("[StepScan] sending \(points.count) points, \(megabytes) MB to \(url)")
        status("Sending the scan (\(megabytes) MB)…")
        let (data, response) = try await session.data(for: http)
        print("[StepScan] server answered \((response as? HTTPURLResponse)?.statusCode ?? 0), \(data.count) bytes")
        try check(response, data)
        return try JSONDecoder().decode(CheckResult.self, from: data)
    }

    struct LiveFit: Decodable {
        let transform: [Float]?   // null until the scan so far fits the model well
        let size: Float?
        let precision: Float?
        let recall: Float?
    }

    private struct FitBody: Encodable {
        let installed: [Int]
        let active: [Int]
        let near: Bool
        let ghost: [Float]
        let points: String
    }

    /// Where the model sits on the scan so far, so the ghost can move onto the real furniture.
    static func fit(server: String, request: ScanRequest, ghost: simd_float4x4, points: [SIMD3<Float>], near: Bool) async throws -> LiveFit {
        var flat = [Float]()
        flat.reserveCapacity(points.count * 3)
        for p in points { flat.append(p.x); flat.append(p.y); flat.append(p.z) }
        let columns = [ghost.columns.0, ghost.columns.1, ghost.columns.2, ghost.columns.3]
        let body = FitBody(installed: request.installed, active: request.active, near: near,
                           ghost: columns.flatMap { [$0.x, $0.y, $0.z, $0.w] },
                           points: flat.withUnsafeBufferPointer { Data(buffer: $0) }.base64EncodedString())
        let path = "/api/stepcheck/\(request.kit.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? request.kit)/fit"
        guard let url = URL(string: server + path) else { throw Failure(message: "Bad server address") }
        var http = URLRequest(url: url, timeoutInterval: 20)
        http.httpMethod = "POST"
        http.setValue("application/json", forHTTPHeaderField: "Content-Type")
        http.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await session.data(for: http)
        try check(response, data)
        return try JSONDecoder().decode(LiveFit.self, from: data)
    }

    private static func check(_ response: URLResponse, _ data: Data) throws {
        guard let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) else { return }
        let detail = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"] as? String
        throw Failure(message: detail ?? "The server answered \(http.statusCode).")
    }
}

extension Data {
    func array<T>(of: T.Type) -> [T] {
        withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
    }
}
