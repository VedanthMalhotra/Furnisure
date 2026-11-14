import SwiftUI
import WebKit

@main
struct FurnitureARApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

/// Loads the assembly app from the server running on your Mac.
/// The Gemini key stays on the Mac; the phone only talks to the server.
struct ContentView: View {
    // New key, so a saved address from home Wi-Fi does not override the new default
    @AppStorage("serverAddress") private var serverURL = "http://Roshans-Macbook-Pro.local:8765"
    @State private var showSettings = false
    @State private var loadError: String?
    @State private var reloadToken = 0
    @StateObject private var bridge = WebBridge()

    var body: some View {
        ZStack {
            WebView(url: URL(string: serverURL), reloadToken: reloadToken, loadError: $loadError, bridge: bridge)
                .ignoresSafeArea()
                // Two-finger long press anywhere opens the server settings
                .simultaneousGesture(LongPressGesture(minimumDuration: 1.2).onEnded { _ in showSettings = true }, including: .all)

            if let loadError {
                VStack(spacing: 14) {
                    Image(systemName: "wifi.exclamationmark").font(.system(size: 40)).foregroundStyle(.orange)
                    Text("Can't reach the server").font(.headline)
                    Text(loadError).font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    Text("Make sure the Mac is running the server and this phone is on the same Wi-Fi.")
                        .font(.footnote).multilineTextAlignment(.center)
                    HStack {
                        Button("Retry") { self.loadError = nil; reloadToken += 1 }.buttonStyle(.borderedProminent)
                        Button("Server settings") { showSettings = true }.buttonStyle(.bordered)
                    }
                }
                .padding(24)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
                .padding()
            }
        }
        // "Scan my step" in the kit player opens the AR check (StepScanView.swift)
        .fullScreenCover(item: $bridge.request) { request in
            StepScanView(request: request, server: serverURL) { next in
                bridge.request = nil
                bridge.finish(next: next)
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(serverURL: $serverURL) {
                loadError = nil
                reloadToken += 1
            }
        }
    }
}

struct SettingsView: View {
    @Binding var serverURL: String
    var onSave: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var draft = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("http://Roshans-Macbook-Pro.local:8765", text: $draft)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Server address")
                } footer: {
                    Text("The address of the Mac running server/app.py. Open these settings any time with a long press.")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        serverURL = draft.trimmingCharacters(in: .whitespaces)
                        onSave()
                        dismiss()
                    }
                }
            }
            .onAppear { draft = serverURL }
        }
    }
}

struct WebView: UIViewRepresentable {
    let url: URL?
    let reloadToken: Int
    @Binding var loadError: String?
    let bridge: WebBridge

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.userContentController.add(bridge, name: "furnisure")
        let web = WKWebView(frame: .zero, configuration: config)
        web.navigationDelegate = context.coordinator
        web.scrollView.bounces = false
        web.scrollView.contentInsetAdjustmentBehavior = .never
        web.isInspectable = true
        bridge.webView = web
        // A short timeout shows the error screen instead of a blank page when the server is unreachable
        if let url { web.load(URLRequest(url: url, timeoutInterval: 10)) }
        context.coordinator.lastToken = reloadToken
        return web
    }

    func updateUIView(_ web: WKWebView, context: Context) {
        context.coordinator.parent = self
        if context.coordinator.lastToken != reloadToken, let url {
            context.coordinator.lastToken = reloadToken
            web.load(URLRequest(url: url, timeoutInterval: 10))
        }
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var parent: WebView
        var lastToken = 0
        init(_ parent: WebView) { self.parent = parent }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            parent.loadError = error.localizedDescription
        }
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            parent.loadError = nil
        }
    }
}

/// Messages between the web page and the app. The kit player posts `scanStep` to
/// `window.webkit.messageHandlers.furnisure`; the app answers through `window.furnisureScanDone`.
final class WebBridge: NSObject, ObservableObject, WKScriptMessageHandler {
    @Published var request: ScanRequest?
    weak var webView: WKWebView?

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let body = message.body as? [String: Any], body["type"] as? String == "scanStep",
              let data = try? JSONSerialization.data(withJSONObject: body),
              let request = try? JSONDecoder().decode(ScanRequest.self, from: data) else { return }
        self.request = request
    }

    func finish(next: Bool) {
        webView?.evaluateJavaScript("window.furnisureScanDone && window.furnisureScanDone({next: \(next)})")
    }
}
