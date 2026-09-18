// Render an HTML file laid out as fixed 612x792 px ".page" blocks into a
// letter-size PDF (1 CSS px = 1 pt) using WebKit, one createPDF rect per page.
// usage: swift render.swift in.html out.pdf [pages]
import Cocoa
import WebKit
import PDFKit

let args = CommandLine.arguments
let htmlURL = URL(fileURLWithPath: args[1])
let outURL = URL(fileURLWithPath: args[2])
let pageCount = args.count > 3 ? Int(args[3])! : 2
let W: CGFloat = 612, H: CGFloat = 792

final class Renderer: NSObject, WKNavigationDelegate {
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            webView.evaluateJavaScript("window.__report ? window.__report() : 'no report'") { res, _ in
                print("LAYOUT:", res ?? "nil")
                let doc = PDFDocument()
                var i = 0
                func next() {
                    if i == pageCount {
                        doc.write(to: outURL)
                        print("wrote \(doc.pageCount) page(s) to \(outURL.path)")
                        // PNG previews next to the script for visual checking
                        for n in 0..<doc.pageCount {
                            guard let p = doc.page(at: n) else { continue }
                            let img = p.thumbnail(of: NSSize(width: W * 2, height: H * 2), for: .mediaBox)
                            if let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
                               let png = rep.representation(using: .png, properties: [:]) {
                                let u = URL(fileURLWithPath: "preview-\(n + 1).png")
                                try? png.write(to: u)
                                print("preview:", u.path)
                            }
                        }
                        exit(0)
                    }
                    let cfg = WKPDFConfiguration()
                    cfg.rect = CGRect(x: 0, y: CGFloat(i) * H, width: W, height: H)
                    webView.createPDF(configuration: cfg) { result in
                        switch result {
                        case .success(let data):
                            if let d = PDFDocument(data: data), let p = d.page(at: 0) {
                                doc.insert(p, at: doc.pageCount)
                            }
                        case .failure(let e):
                            print("createPDF failed:", e); exit(1)
                        }
                        i += 1
                        next()
                    }
                }
                next()
            }
        }
    }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        print("load failed:", error); exit(1)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
let frame = CGRect(x: 0, y: 0, width: W, height: H * CGFloat(pageCount))
let window = NSWindow(contentRect: frame, styleMask: [.borderless], backing: .buffered, defer: false)
let webView = WKWebView(frame: frame)
window.contentView = webView
let renderer = Renderer()
webView.navigationDelegate = renderer
webView.loadFileURL(htmlURL, allowingReadAccessTo: htmlURL.deletingLastPathComponent())
DispatchQueue.main.asyncAfter(deadline: .now() + 60) { print("timeout"); exit(2) }
app.run()
