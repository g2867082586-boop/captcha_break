"""Lesson 38: prelabel and manually verify the authorized 500-image dataset."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from captcha_break.annotation import AnnotationWorkspace, Prelabel
from captcha_break.ddddocr_adapter import DdddOcrRecognizer
from captcha_break.project_generator import PROJECT_ALPHABET

HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>验证码标注</title><style>
:root{font-family:system-ui,sans-serif;color:#172033;background:#eef2f7}body{margin:0}
main{max-width:760px;margin:24px auto;background:#fff;padding:24px;border-radius:14px;box-shadow:0 8px 30px #18315322}
h1{margin-top:0}.captcha{display:block;width:min(100%,603px);image-rendering:auto;border:1px solid #94a3b8;margin:18px auto;background:#fff}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.panel{padding:12px;background:#f8fafc;border-radius:8px}
label{display:grid;gap:6px;margin:16px 0}input,button{font:inherit;padding:10px;border-radius:7px;border:1px solid #94a3b8}
input{text-transform:uppercase;font-size:24px;letter-spacing:.25em;text-align:center}button{cursor:pointer;background:#1265b0;color:white}
button.secondary{background:white;color:#172033}.actions{display:flex;gap:8px;flex-wrap:wrap}.status{min-height:1.5em}.muted{color:#52606d}
@media(max-width:600px){main{margin:0;border-radius:0}.grid{grid-template-columns:1fr}.actions button{flex:1}}
</style></head><body><main><h1>真实验证码标注</h1>
<p id="progress"></p><img id="image" class="captcha" alt="待标注验证码">
<div class="grid"><div class="panel">Beta：<strong id="beta"></strong></div><div class="panel">Default：<strong id="default"></strong></div></div>
<label>人工确认标签<input id="label" maxlength="4" pattern="[A-Z0-9]{4}" autocomplete="off"></label>
<div class="actions"><button id="previous" class="secondary">← 上一张</button><button id="confirm">确认并下一张（Enter）</button>
<button id="skip" class="secondary">跳过</button><button id="next" class="secondary">下一张 →</button><button id="export" class="secondary">导出已确认数据</button></div>
<p id="status" class="status"></p><p class="muted">方向键切换；Enter 确认。所有修改只写入独立标注工作区。</p>
</main><script>
const token=__TOKEN__;let index=0;let total=0;
const api=async(url,options={})=>{options.headers={...(options.headers||{}),"x-annotation-token":token};const response=await fetch(url,options);const body=await response.json();if(!response.ok)throw new Error(body.error||"请求失败");return body};
const nodes={progress:document.querySelector("#progress"),image:document.querySelector("#image"),beta:document.querySelector("#beta"),default:document.querySelector("#default"),label:document.querySelector("#label"),status:document.querySelector("#status")};
async function load(next){const state=await api(`/api/state?index=${next}`);index=state.index;total=state.progress.total;const r=state.record;
nodes.progress.textContent=`第 ${index+1}/${total} 张｜已确认 ${state.progress.confirmed}｜待确认 ${state.progress.pending}｜跳过 ${state.progress.skipped}`;
nodes.image.src=`/image/${encodeURIComponent(r.filename)}?token=${encodeURIComponent(token)}`;nodes.beta.textContent=r.beta_prediction||"无有效结果";nodes.default.textContent=r.default_prediction||"无有效结果";
nodes.label.value=r.label||r.suggested_label||"";nodes.status.textContent=`状态：${r.status}`;nodes.label.focus();nodes.label.select()}
async function action(name,label=null){try{const body={index};if(label!==null)body.label=label;const result=await api(`/api/${name}`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});await load(result.next_index)}catch(error){nodes.status.textContent=error.message}}
document.querySelector("#confirm").onclick=()=>action("confirm",nodes.label.value);document.querySelector("#skip").onclick=()=>action("skip");
document.querySelector("#previous").onclick=()=>load(Math.max(0,index-1));document.querySelector("#next").onclick=()=>load(Math.min(total-1,index+1));
document.querySelector("#export").onclick=async()=>{try{const result=await api("/api/export",{method:"POST",headers:{"content-type":"application/json"},body:"{}"});nodes.status.textContent=`已导出：${result.output}`}catch(error){nodes.status.textContent=error.message}};
document.addEventListener("keydown",event=>{if(event.key==="Enter"){event.preventDefault();action("confirm",nodes.label.value)}else if(event.key==="ArrowLeft")load(Math.max(0,index-1));else if(event.key==="ArrowRight")load(Math.min(total-1,index+1))});load(0);
</script></body></html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=Path.home() / "Downloads" / "captcha_authorized_raw",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "Downloads" / "captcha_annotation_workspace",
    )
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def build_predictor():
    beta = DdddOcrRecognizer(beta=True)
    default = DdddOcrRecognizer(beta=False)

    def predict(image: bytes) -> Prelabel:
        return Prelabel(beta.predict(image), default.predict(image))

    return predict


def make_handler(workspace: AnnotationWorkspace, token: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _authorized(self) -> bool:
            header = self.headers.get("x-annotation-token", "")
            query = parse_qs(urlparse(self.path).query)
            return secrets.compare_digest(header or query.get("token", [""])[0], token)

        def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page = HTML.replace("__TOKEN__", json.dumps(token)).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(page)))
                self.send_header("cache-control", "no-store")
                self.send_header(
                    "content-security-policy",
                    "default-src 'self'; img-src 'self'; script-src 'unsafe-inline'; "
                    "style-src 'unsafe-inline'; object-src 'none'; base-uri 'none'",
                )
                self.end_headers()
                self.wfile.write(page)
                return
            if not self._authorized():
                self._json({"error": "unauthorized"}, HTTPStatus.FORBIDDEN)
                return
            if parsed.path == "/api/state":
                try:
                    index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                    self._json(workspace.state(index))
                except (TypeError, ValueError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if parsed.path.startswith("/image/"):
                filename = unquote(parsed.path.removeprefix("/image/"))
                if filename != Path(filename).name:
                    self._json({"error": "invalid filename"}, HTTPStatus.BAD_REQUEST)
                    return
                image_path = workspace.images_dir / filename
                if not image_path.is_file():
                    self._json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
                    return
                data = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("content-type", "image/png")
                self.send_header("content-length", str(len(data)))
                self.send_header("cache-control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if not self._authorized():
                self._json({"error": "unauthorized"}, HTTPStatus.FORBIDDEN)
                return
            if self.headers.get("content-type", "").split(";", 1)[0] != "application/json":
                self._json(
                    {"error": "application/json required"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE
                )
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                if length < 0 or length > 1024:
                    raise ValueError("request body is too large")
                body = json.loads(self.rfile.read(length))
                parsed = urlparse(self.path)
                if parsed.path == "/api/confirm":
                    next_index = workspace.confirm(int(body["index"]), str(body["label"]))
                    self._json({"next_index": next_index})
                elif parsed.path == "/api/skip":
                    next_index = workspace.skip(int(body["index"]))
                    self._json({"next_index": next_index})
                elif parsed.path == "/api/export":
                    output = workspace.export_confirmed()
                    self._json({"output": str(output)})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    configure_console()
    args = build_parser().parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("port must be in 1024..65535")
    workspace = AnnotationWorkspace(
        args.workspace,
        alphabet=PROJECT_ALPHABET,
        label_length=4,
    )
    added = workspace.prepare(args.source, build_predictor())
    progress = workspace.progress()
    print(f"工作区：{workspace.path}")
    print(f"本次新增：{added}；总数：{progress['total']}；已确认：{progress['confirmed']}")
    if args.prepare_only:
        return
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(workspace, token))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"标注页面：{url}")
    print("按 Ctrl+C 停止；进度会逐张保存，可以稍后继续。")
    webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n标注服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
