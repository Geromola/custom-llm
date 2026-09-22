// Screenshot a running page with WebKit, after optionally driving it with JavaScript.
//
// The interfaces in this repository are real web pages, so their evidence should be an
// actual capture of them running, not a mock-up. This loads a URL in a WKWebView, runs a
// snippet (to type a prompt, press a button, switch a tab), waits for the page to settle
// and writes a PNG.
//
//   swift tools/capture.swift <url> <out.png> [width] [height] [settleSeconds] [js]
//
// The JavaScript may return a Promise; the capture waits for it to resolve, then waits
// `settleSeconds` more so animations and fetches finish.
import Cocoa
import WebKit

let args = CommandLine.arguments
guard args.count >= 3, let url = URL(string: args[1]) else {
    FileHandle.standardError.write("usage: capture.swift <url> <out.png> [w] [h] [settle] [js]\n".data(using: .utf8)!)
    exit(2)
}
let out = URL(fileURLWithPath: args[2])
let width = args.count > 3 ? Double(args[3])! : 1280
let height = args.count > 4 ? Double(args[4])! : 900
let settle = args.count > 5 ? Double(args[5])! : 2.5
let script = args.count > 6 ? args[6] : ""

final class Capturer: NSObject, WKNavigationDelegate {
    let webView: WKWebView
    init(frame: NSRect) {
        let configuration = WKWebViewConfiguration()
        configuration.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")
        webView = WKWebView(frame: frame, configuration: configuration)
        super.init()
        webView.navigationDelegate = self
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        let run = {
            DispatchQueue.main.asyncAfter(deadline: .now() + settle) {
                let config = WKSnapshotConfiguration()
                config.rect = CGRect(x: 0, y: 0, width: width, height: height)
                webView.takeSnapshot(with: config) { image, error in
                    guard let image = image,
                          let tiff = image.tiffRepresentation,
                          let rep = NSBitmapImageRep(data: tiff),
                          let png = rep.representation(using: .png, properties: [:]) else {
                        FileHandle.standardError.write("snapshot failed: \(error?.localizedDescription ?? "unknown")\n".data(using: .utf8)!)
                        exit(1)
                    }
                    try? png.write(to: out)
                    print("wrote \(out.path)")
                    exit(0)
                }
            }
        }
        if script.isEmpty {
            run()
        } else {
            webView.callAsyncJavaScript(script, arguments: [:], in: nil, in: .page) { result in
                if case .failure(let error) = result {
                    FileHandle.standardError.write("script error: \(error)\n".data(using: .utf8)!)
                }
                run()
            }
        }
    }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        FileHandle.standardError.write("load failed: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(1)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let capturer = Capturer(frame: NSRect(x: 0, y: 0, width: width, height: height))
let window = NSWindow(contentRect: capturer.webView.frame, styleMask: [.borderless],
                      backing: .buffered, defer: false)
window.contentView = capturer.webView
capturer.webView.load(URLRequest(url: url))
DispatchQueue.main.asyncAfter(deadline: .now() + 90) {
    FileHandle.standardError.write("timed out\n".data(using: .utf8)!)
    exit(1)
}
app.run()
