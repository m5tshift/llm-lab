"""Builds the report PDF: REPORT.md -> HTML -> paged layout (Paged.js) -> PDF from headless Chrome.

Usage: uv run --locked python -m scripts.build_pdf [--input REPORT.md] [--output build/REPORT.pdf]

The layout follows the usual Russian report style: a title page from the front matter of
REPORT.md, a table of contents with page numbers, numbered sections, "Таблица N – ..." captions
above tables and "Рисунок N – ..." captions below Mermaid figures. Needs Google Chrome or
Chromium (--chrome, $CHROME or the usual install locations) and network access: Mermaid and
Paged.js are loaded from a CDN. Chrome is driven over --remote-debugging-pipe, so the PDF is
printed only after the page layout has finished. --html-only writes just the HTML.
"""

import argparse
import base64
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from markdown_it import MarkdownIt

MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"
PAGED_JS = "https://cdn.jsdelivr.net/npm/pagedjs@0.4.3/dist/paged.polyfill.min.js"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
]
CAPTION = re.compile(r"^(Таблица|Рисунок) ([А-Я]\.)?(\d+) – ")
# Headings that start on a new page: the list of sources and each appendix.
NEW_PAGE = re.compile(r"^(Список источников|Приложение )")

CSS = """
@page {
  size: A4;
  margin: 20mm 15mm 20mm 30mm;
  @bottom-center { content: counter(page); font: 12pt "Times New Roman", "Liberation Serif", serif; }
}
@page title { @bottom-center { content: none; } }
html { font: 14pt/1.5 "Times New Roman", "Liberation Serif", serif; color: #000; }
body { margin: 0; }
p { margin: 0; text-indent: 1.25cm; text-align: justify; hyphens: auto; orphans: 2; widows: 2; }
h2, h3, h4 { font-family: Arial, "Liberation Sans", sans-serif; font-weight: bold;
  break-after: avoid; text-align: left; }
h2 { font-size: 16pt; margin: 18pt 0 10pt; }
h3 { font-size: 14pt; margin: 14pt 0 8pt; }
h2.new-page { break-before: page; }
a { color: inherit; text-decoration: none; }
code { font: 12pt "Courier New", "Liberation Mono", monospace; }
pre { margin: 8pt 0; padding: 6pt 10pt; border: 1px solid #999; font-size: 11pt;
  line-height: 1.3; white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }
pre code { font-size: 11pt; }
ul, ol { margin: 0; padding-left: 0; list-style: none; }
li { text-indent: 1.25cm; text-align: justify; }
ul > li::before { content: "– "; }
p.caption-table { text-indent: 0; text-align: left; margin-top: 8pt; break-after: avoid; }
p.caption-figure { text-indent: 0; text-align: center; margin: 4pt 0 10pt; }
p.formula { text-indent: 0; text-align: center; margin: 6pt 0; }
p.source { text-indent: 0; margin-bottom: 6pt; text-align: left; hyphens: none; overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 12pt;
  font-size: 12pt; line-height: 1.25; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
tbody tr:first-child { break-before: avoid; }
th, td { border: 1px solid #000; padding: 4pt 6pt; vertical-align: top; text-align: left; }
th { background: #eee; font-weight: normal; }
td code, th code { font-size: 10.5pt; overflow-wrap: anywhere; }
.figure { margin: 10pt 0 0; text-align: center; break-inside: avoid; break-after: avoid; }
.figure pre.mermaid { border: none; padding: 0; margin: 0; white-space: normal; }
.figure svg { max-width: 100%; max-height: 120mm; height: auto; }
.figure p { text-indent: 0; text-align: center; }
hr { display: none; }

.title-page { page: title; break-after: page; height: 255mm; display: flex; flex-direction: column;
  text-align: center; font-size: 14pt; }
.title-page .discipline { margin-top: 20mm; }
.title-page h1 { font: bold 20pt Arial, "Liberation Sans", sans-serif; margin: 55mm 0 12mm; }
.title-page .work { margin-bottom: 14mm; }
.title-page .people { margin: 22mm 0 0 75mm; text-align: left; line-height: 1.6; }
.title-page .year { margin-top: auto; }

nav.toc { break-after: page; }
nav.toc h2 { margin-top: 0; }
nav.toc ul { list-style: none; padding: 0; margin: 0; line-height: 1.3; }
nav.toc li { text-indent: 0; text-align: left; margin: 0; }
nav.toc li::before { content: none; }
nav.toc li.l2 { margin-top: 5pt; }
nav.toc li.l3 { padding-left: 1cm; }
nav.toc a { display: flex; align-items: baseline; }
nav.toc a .dots { flex: 1; margin: 0 4pt; min-width: 10pt; height: 1em; background:
  radial-gradient(circle, #000 0.9px, transparent 1.1px) left bottom / 6px 4px repeat-x; }
/* Paged.js fills the counter; it does not support leader(), hence the dotted span. */
nav.toc a::after { content: target-counter(attr(href), page); }
"""

SCRIPT = f"""
<script>window.PagedConfig = {{ auto: false }};</script>
<script src="{MERMAID_JS}"></script>
<script>
  // Configured right after loading: Mermaid's own startOnLoad would otherwise render first.
  mermaid.initialize({{
    startOnLoad: false, theme: "base",
    flowchart: {{ useMaxWidth: false, htmlLabels: true }},
    themeVariables: {{ primaryColor: "#ffffff", primaryBorderColor: "#000000",
      primaryTextColor: "#000000", lineColor: "#000000", clusterBkg: "#ffffff",
      clusterBorder: "#000000", edgeLabelBackground: "#ffffff",
      fontFamily: "Times New Roman, serif", fontSize: "15px" }}
  }});
</script>
<script src="{PAGED_JS}"></script>
<script>
  window.addEventListener("load", async () => {{
    try {{ await mermaid.run({{ querySelector: ".mermaid" }}); }} catch (e) {{ console.error(e); }}
    try {{ await window.PagedPolyfill.preview(); }} catch (e) {{ window.__error = String(e); }}
    window.__ready = true;
  }});
</script>
"""


def split_front_matter(text: str):
    """Returns ({key: value}, body) for a leading '---' block of key: value lines."""
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if value:
            meta[key.strip()] = value.strip()
    return meta, text[match.end() :]


def title_page(meta: dict) -> str:
    e = {k: html.escape(v) for k, v in meta.items()}
    roles = (("Выполнил(а)", "author"), ("Группа", "group"), ("Преподаватель", "teacher"))
    people = "".join(f"<div>{label}: {e[key]}</div>" for label, key in roles if key in e)
    return (
        '<section class="title-page">'
        f'<div class="discipline">{e.get("discipline", "")}</div>'
        f"<h1>{e.get('title', '')}</h1>"
        f'<div class="work">{e.get("work", "")}</div>'
        f'<div class="subtitle">{e.get("subtitle", "")}</div>'
        f'<div class="people">{people}</div>'
        f'<div class="year">{e.get("year", "")}</div>'
        "</section>"
    )


def check_captions(tokens) -> list:
    """Returns warnings for table and figure numbers that are not consecutive in their series."""
    last, warnings = {}, []
    for i, token in enumerate(tokens):
        if token.type != "paragraph_open":
            continue
        match = CAPTION.match(tokens[i + 1].content)
        if match:
            kind, prefix, number = match.group(1), match.group(2) or "", int(match.group(3))
            expected = last.get((kind, prefix), 0) + 1
            if number != expected:
                warnings.append(f"{kind} {prefix}{number}: expected {prefix}{expected}")
            last[(kind, prefix)] = number
    return warnings


def render_html(markdown: str) -> str:
    meta, body_md = split_front_matter(markdown)
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    default_fence = md.renderer.rules["fence"]

    def fence(tokens, idx, options, env):
        token = tokens[idx]
        if token.info.strip() == "mermaid":
            source = html.escape(token.content)
            return f'<div class="figure"><pre class="mermaid">{source}</pre></div>\n'
        return default_fence(tokens, idx, options, env)

    md.renderer.rules["fence"] = fence
    tokens = md.parse(body_md)
    for warning in check_captions(tokens):
        print(f"[pdf] Warning: caption numbering: {warning}", file=sys.stderr)

    toc, title_seen = [], False
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            text = tokens[i + 1].content
            if token.tag == "h1" and not title_seen:
                # The title page replaces the document heading.
                token.attrSet("style", "display:none")
                title_seen = True
                continue
            anchor = f"s{i}"
            token.attrSet("id", anchor)
            if token.tag == "h2" and NEW_PAGE.match(text):
                token.attrSet("class", "new-page")
            if token.tag == "h2" or (token.tag == "h3" and re.match(r"^\d", text)):
                # Appendix subsections (В.1 ...) stay out of the table of contents.
                toc.append((token.tag, text, anchor))
        elif token.type == "paragraph_open":
            text = tokens[i + 1].content
            match = CAPTION.match(text)
            if match:
                kind = "caption-table" if match.group(1) == "Таблица" else "caption-figure"
                token.attrSet("class", kind)
            elif re.match(r"^\[\d+\] ", text):
                token.attrSet("class", "source")

    body = md.renderer.render(tokens, md.options, {})
    title = meta.get("title") or "Отчёт"
    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f"{title_page(meta) if meta else ''}{toc_html(toc)}{body}{SCRIPT}</body></html>"
    )


def toc_html(entries) -> str:
    inline = MarkdownIt("commonmark")
    items = []
    for tag, text, anchor in entries:
        label = re.sub(r"<[^>]+>", "", inline.renderInline(text))
        items.append(
            f'<li class="l{tag[1]}"><a href="#{anchor}"><span>{label}</span>'
            '<span class="dots"></span></a></li>'
        )
    return f'<nav class="toc"><h2>Оглавление</h2><ul>{"".join(items)}</ul></nav>'


def find_chrome(explicit) -> str:
    for candidate in [explicit, os.environ.get("CHROME"), *CHROME_CANDIDATES]:
        if not candidate:
            continue
        path = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if path:
            return path
    raise SystemExit("Chrome/Chromium not found: pass --chrome or set CHROME, or use --html-only")


class ChromePipe:
    """Minimal DevTools client over --remote-debugging-pipe (fd 3 in, fd 4 out, NUL-separated)."""

    def __init__(self, chrome: str, profile: str):
        cmd_r, self._cmd_w = os.pipe()
        self._res_r, res_w = os.pipe()

        def child_fds():
            # dup2 onto the same number keeps close-on-exec, so clear it explicitly.
            for source, target in ((cmd_r, 3), (res_w, 4)):
                if source != target:
                    os.dup2(source, target)
                os.set_inheritable(target, True)

        args = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            f"--user-data-dir={profile}",
            "--remote-debugging-pipe",
            "about:blank",
        ]
        self.proc = subprocess.Popen(
            args,
            # Inherited descriptors stay open so that the duplicates on fds 3 and 4 survive.
            close_fds=False,
            preexec_fn=child_fds,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.close(cmd_r)
        os.close(res_w)
        self._buffer = b""
        self._next_id = 0

    def call(self, method: str, params=None, session=None, timeout=120):
        self._next_id += 1
        message = {"id": self._next_id, "method": method, "params": params or {}}
        if session:
            message["sessionId"] = session
        os.write(self._cmd_w, json.dumps(message).encode() + b"\0")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            while b"\0" not in self._buffer:
                chunk = os.read(self._res_r, 1 << 20)
                if not chunk:
                    raise RuntimeError("Chrome closed the DevTools pipe")
                self._buffer += chunk
            raw, self._buffer = self._buffer.split(b"\0", 1)
            reply = json.loads(raw)
            if reply.get("id") == self._next_id:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply.get("result", {})
        raise TimeoutError(method)

    def close(self):
        try:
            self.call("Browser.close", timeout=10)
        except Exception:
            self.proc.kill()
        self.proc.wait(timeout=20)


def print_pdf(chrome: str, html_path: Path, pdf_path: Path, timeout_s: int) -> None:
    pdf_path.unlink(missing_ok=True)
    wait_for_layout = (
        "new Promise(r => { const t = () => window.__ready ? r(window.__error || 'ok')"
        " : setTimeout(t, 200); t(); })"
    )
    with tempfile.TemporaryDirectory() as profile:
        browser = ChromePipe(chrome, profile)
        try:
            target = browser.call("Target.createTarget", {"url": "about:blank"})
            session = browser.call(
                "Target.attachToTarget", {"targetId": target["targetId"], "flatten": True}
            )["sessionId"]
            browser.call("Page.navigate", {"url": html_path.resolve().as_uri()}, session)
            state = None
            for _ in range(20):
                try:
                    state = browser.call(
                        "Runtime.evaluate",
                        {
                            "expression": wait_for_layout,
                            "awaitPromise": True,
                            "returnByValue": True,
                        },
                        session,
                        timeout_s,
                    )["result"].get("value")
                    break
                except RuntimeError as error:
                    # The blank page's context goes away while the report loads; ask again.
                    if "context was destroyed" not in str(error):
                        raise
                    time.sleep(0.2)
            if state != "ok":
                print(f"[pdf] Warning: page layout reported: {state}", file=sys.stderr)
            data = browser.call(
                "Page.printToPDF",
                {"preferCSSPageSize": True, "printBackground": True},
                session,
                timeout_s,
            )["data"]
        finally:
            browser.close()
    pdf_path.write_bytes(base64.b64decode(data))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("REPORT.md"))
    parser.add_argument("--output", type=Path, default=Path("build/REPORT.pdf"))
    parser.add_argument("--chrome", help="Path to Chrome or Chromium")
    parser.add_argument("--html-only", action="store_true", help="Write HTML and skip the PDF")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds to wait for the layout")
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html_path = args.output.with_suffix(".html")
    html_path.write_text(render_html(args.input.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"[pdf] HTML: {html_path}")
    if args.html_only:
        return
    print_pdf(find_chrome(args.chrome), html_path, args.output, args.timeout)
    print(f"[pdf] PDF: {args.output}")


if __name__ == "__main__":
    main()
