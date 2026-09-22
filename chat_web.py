"""A small web chat that loads model.pt directly and answers with the trained nanoGPT.

    python chat_web.py --model results/exp2-expanded/model.pt
    # then open http://127.0.0.1:8000

Standard library only apart from torch: no framework, no API key, no second model. Every
reply is produced by `run_evals.generate_reply` on the weights loaded from --model, and
every exchange is appended to --transcript. Generating a reply never updates the weights:
the model's hash is checked before and after each request and the server refuses to
continue if it ever changes.

The interactive explorer in viz/ is a second, static interface that runs the same weights
in the browser. This one exists so there is also an interface that loads model.pt itself.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from run_evals import generate_reply, load_model, model_hash

PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Chat with my tiny language model</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root{color-scheme:dark light;--bg:#11141a;--panel:#181c24;--line:#2a3140;--ink:#e6e9ef;
   --dim:#97a1b4;--accent:#6ea8fe;--warn:#ff8f8f}
 @media (prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--line:#dfe3ea;
   --ink:#161a20;--dim:#57617a;--accent:#2563c9;--warn:#c4362f}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,
   BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;display:flex;justify-content:center}
 main{width:100%;max-width:720px;padding:26px 16px 60px}
 h1{font-size:20px;margin:0 0 4px}
 .sub{color:var(--dim);font-size:13px;margin:0 0 16px}
 .id{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--dim);
   background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px 11px;margin-bottom:14px}
 #log{min-height:180px}
 .turn{border:1px solid var(--line);background:var(--panel);border-radius:10px;
   padding:10px 12px;margin-bottom:10px}
 .who{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);margin-bottom:3px}
 .reply{font-family:ui-monospace,Menlo,monospace;font-size:13.5px}
 .flag{color:var(--warn);font-size:12.5px;margin-top:6px}
 form{display:flex;gap:8px;margin-top:14px}
 input,button{font:inherit;border-radius:8px;border:1px solid var(--line);padding:9px 12px}
 input{flex:1;background:var(--panel);color:var(--ink)}
 button{background:var(--accent);color:#fff;border-color:transparent;cursor:pointer}
 .note{color:var(--dim);font-size:12.5px;margin-top:14px;border-left:2px solid var(--line);padding-left:10px}
</style>
<main>
 <h1>Chat with my tiny language model</h1>
 <p class="sub">122k parameters, trained from scratch on a small corpus. It continues a sentence;
   it does not answer questions. Every prompt starts fresh — there is no conversation memory.</p>
 <div class="id" id="identity">loading…</div>
 <div id="log"></div>
 <form id="form"><input id="prompt" autocomplete="off" placeholder="the report about the customer explains the"><button>Send</button></form>
 <p class="note">Words outside the model's vocabulary arrive as &lt;UNK&gt; and carry no meaning.
   Prompts longer than the context window keep only the most recent tokens. Replies are sampled,
   so the same prompt can continue differently; nothing you type here trains the model.</p>
</main>
<script>
const log = document.getElementById("log");
fetch("/api/identity").then(r => r.json()).then(d => {
  document.getElementById("identity").textContent =
    `${d.model} · ${d.parameters.toLocaleString()} parameters · ${d.vocabulary} words · ` +
    `${d.completed_steps} training steps · weights sha256 ${d.model_sha256.slice(0,16)}… · ` +
    `context ${d.block_size} tokens`;
});
document.getElementById("form").addEventListener("submit", async event => {
  event.preventDefault();
  const box = document.getElementById("prompt");
  const prompt = box.value.trim();
  if (!prompt) return;
  box.value = "";
  const reply = await (await fetch("/api/chat", {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify({prompt})})).json();
  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `<div class="who">you</div><div>${prompt.replace(/[<>&]/g, c =>
      ({"<":"&lt;", ">":"&gt;", "&":"&amp;"}[c]))}</div>` +
    `<div class="who" style="margin-top:8px">the model continues</div>` +
    `<div class="reply">${reply.response ? reply.response.replace(/[<>&]/g, c =>
      ({"<":"&lt;", ">":"&gt;", "&":"&amp;"}[c])) : "[empty response]"}</div>` +
    (reply.unknown_prompt_words.length
      ? `<div class="flag">Unknown words, read as &lt;UNK&gt;: ${reply.unknown_prompt_words.join(", ")}</div>` : "") +
    (reply.prompt_truncated ? `<div class="flag">Prompt longer than the context window; only the most recent tokens were used.</div>` : "");
  log.append(turn);
  turn.scrollIntoView({behavior: "smooth", block: "end"});
});
</script>
</html>
"""


def build_handler(model, vocabulary, saved, identity, transcript_path, record):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload, content_type="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/identity":
                self._send(identity)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/api/chat":
                return self.send_error(404)
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            prompt = str(request.get("prompt", ""))[:2000]
            seed = 2026 + len(record["turns"])
            reply = generate_reply(model, vocabulary, prompt, seed=seed)
            if model_hash(model) != identity["model_sha256"]:
                raise RuntimeError("the weights changed during inference")
            record["turns"].append({"prompt": prompt, "seed": seed, "interface": "web", **reply})
            transcript_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f'you: {prompt}\nmodel: {reply["response"] or "[empty]"}\n', flush=True)
            self._send(reply)

        def log_message(self, *args):
            pass
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("results/exp2-expanded/model.pt"))
    parser.add_argument("--transcript", type=Path, default=Path("results/chat/web_transcript.json"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="open a browser window")
    args = parser.parse_args()

    model, vocabulary, saved = load_model(args.model)
    identity = {"model": str(args.model), "model_sha256": model_hash(model),
                "parameters": sum(p.numel() for p in model.parameters()),
                "vocabulary": len(vocabulary), "block_size": model.config.block_size,
                "completed_steps": saved.get("completed_steps")}
    record = {**identity, "fresh_context_per_prompt": True, "temperature": 0.8,
              "max_tokens": 24, "interface": "chat_web.py", "turns": []}
    args.transcript.parent.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 build_handler(model, vocabulary, saved, identity, args.transcript, record))
    print(f"{identity['model']} · {identity['parameters']:,} parameters · "
          f"{identity['vocabulary']} words · sha256 {identity['model_sha256'][:16]}…")
    print(f"open http://127.0.0.1:{args.port}   (ctrl-c to stop; transcript -> {args.transcript})")
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        args.transcript.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("saved transcript:", args.transcript)


if __name__ == "__main__":
    main()
