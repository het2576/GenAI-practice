"""
The Assignment Desk — a newsroom front end for the search / source / draft / desk
research pipeline defined in agents.py and pipeline.py.
"""

import html
import os
import re
from datetime import date

import markdown as md
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

STAGES = ["lead", "source", "draft", "desk"]
STAGE_META = {
    "lead": {"label": "Lead", "role": "Search correspondent", "desc": "Wires in the first leads on the story.", "code": "01", "verb": "Scout"},
    "source": {"label": "Source", "role": "Field reader", "desc": "Pulls the fullest account from the strongest lead.", "code": "02", "verb": "Verify"},
    "draft": {"label": "Draft", "role": "Staff writer", "desc": "Files the story for the desk.", "code": "03", "verb": "Write"},
    "desk": {"label": "Desk", "role": "Copy editor", "desc": "Reads it back, marks it up, stamps a grade.", "code": "04", "verb": "Review"},
}
STATUS_TEXT = {"pending": "PENDING", "running": "ON THE WIRE", "done": "FILED", "error": "KILLED"}

st.set_page_config(page_title="The Assignment Desk", page_icon="🗞️", layout="wide", initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "stage_status" not in st.session_state:
    st.session_state.stage_status = {s: "pending" for s in STAGES}
if "results" not in st.session_state:
    st.session_state.results = {}
if "errors" not in st.session_state:
    st.session_state.errors = {}
if "topic" not in st.session_state:
    st.session_state.topic = ""


def reset_run():
    st.session_state.stage_status = {s: "pending" for s in STAGES}
    st.session_state.results = {}
    st.session_state.errors = {}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def extract_text(content) -> str:
    """Normalize a LangChain message .content into plain text.

    Some providers return content as a list of content blocks
    (e.g. [{'type': 'text', 'text': '...'}]) rather than a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.M)
    return text


def linkify(escaped_text: str) -> str:
    return re.sub(
        r"(https?://[^\s<)\]]+)",
        r'<a href="\1" target="_blank" rel="noopener">\1</a>',
        escaped_text,
    )


def telex_html(raw: str) -> str:
    body = linkify(esc(strip_markdown(raw))).replace("\n", "<br>")
    return f'<div class="telex-feed">{body}</div>'


def clip_html(raw: str) -> str:
    body = esc(strip_markdown(raw))
    if len(body) > 1400:
        body = body[:1400].rsplit(" ", 1)[0] + "…"
    body = body.replace("\n", "<br>")
    return f'<div class="clip-sheet"><div class="clip-tape">SOURCE COPY</div>{body}</div>'


def _collapse_table_blanks(text: str) -> str:
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        if (
            line.strip() == ""
            and out
            and out[-1].strip().startswith("|")
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("|")
        ):
            continue
        out.append(line)
    return "\n".join(out)


def _collapse_hrs(text: str) -> str:
    lines = text.split("\n")
    out, prev_hr = [], False
    for line in lines:
        is_hr = bool(re.match(r"^-{3,}\s*$", line.strip()))
        if is_hr:
            if prev_hr or not out:
                continue
            out.append(line)
            prev_hr = True
        else:
            if line.strip() != "":
                prev_hr = False
            out.append(line)
    return "\n".join(out)


def _promote_bold_headers(text: str) -> str:
    top_level = re.compile(r"^\*\*(introduction|key findings|conclusion|sources)\**:?\s*$", re.I)
    numbered = re.compile(r"^\*\*(\d+\.\s*.+?)\*\*:?\s*$")
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        m = top_level.match(stripped)
        if m:
            out.append(f"## {m.group(1).title()}")
            continue
        m = numbered.match(stripped)
        if m:
            out.append(f"### {m.group(1)}")
            continue
        out.append(line)
    return "\n".join(out)


def _autolink(text: str) -> str:
    return re.sub(r'(?<!\]\()(https?://[^\s<>"\)]+)', r"[\1](\1)", text)


def report_to_html(raw: str) -> str:
    text = esc(raw.strip())
    text = _autolink(text)
    text = _collapse_table_blanks(text)
    text = _collapse_hrs(text)
    text = _promote_bold_headers(text)
    body = md.markdown(text, extensions=["tables", "sane_lists"])
    body = body.replace("<p>", '<p class="lede">', 1)
    return f'<div class="report-body">{body}</div>'


def parse_critic(raw: str):
    score_m = re.search(r"Score:\s*(\d{1,2})\s*/\s*10", raw)
    strengths_m = re.search(r"Strengths:(.*?)Areas to Improve:", raw, re.S)
    areas_m = re.search(r"Areas to Improve:(.*?)One line verdict:", raw, re.S)
    verdict_m = re.search(r"One line verdict:(.*)", raw, re.S)
    if not (score_m and strengths_m and areas_m and verdict_m):
        return None

    def bullets(block: str):
        items = []
        for ln in block.strip().split("\n"):
            ln = ln.strip().lstrip("-*").strip()
            if ln:
                items.append(esc(strip_markdown(ln)))
        return items

    return {
        "score": score_m.group(1),
        "strengths": bullets(strengths_m.group(1)),
        "areas": bullets(areas_m.group(1)),
        "verdict": esc(strip_markdown(verdict_m.group(1).strip())),
    }


def memo_html(raw: str) -> str:
    parsed = parse_critic(raw)
    if not parsed:
        return f'<div class="memo"><div class="telex-feed">{esc(raw).replace(chr(10), "<br>")}</div></div>'
    strengths = "".join(f"<li>{s}</li>" for s in parsed["strengths"])
    areas = "".join(f"<li>{a}</li>" for a in parsed["areas"])
    return f"""
    <div class="memo">
      <div class="stamp" aria-hidden="true">
        <span class="stamp-value">{parsed['score']}</span>
        <span class="stamp-label">/ 10</span>
      </div>
      <h4 class="memo-head">Strengths</h4>
      <ul class="redline-list redline-list--up">{strengths}</ul>
      <h4 class="memo-head">Areas to improve</h4>
      <ul class="redline-list redline-list--down">{areas}</ul>
      <p class="verdict">&ldquo;{parsed['verdict']}&rdquo;</p>
    </div>
    """


def status_dot(status: str) -> str:
    return f'<span class="dot dot--{status}" aria-hidden="true"></span>'


def stepper_html() -> str:
    tabs = []
    for s in STAGES:
        status = st.session_state.stage_status[s]
        meta = STAGE_META[s]
        tabs.append(
            f"""
            <div class="docket-tab is-{status}">
              <div class="docket-tab-top"><span class="stage-code">{meta['code']}</span>{status_dot(status)}<span class="docket-tab-status">{STATUS_TEXT[status]}</span></div>
              <div class="docket-tab-label"><span>{meta['label']}</span><span class="docket-tab-verb">{meta['verb']}</span></div>
              <div class="docket-tab-role">{meta['role']}</div>
            </div>
            """
        )
    return f'<section class="docket" aria-label="Assignment workflow"><div class="wire-rail" aria-hidden="true"></div>{"".join(tabs)}</section>'


def stage_card_html(stage: str) -> str:
    meta = STAGE_META[stage]
    status = st.session_state.stage_status[stage]
    header = f'''<div class="doc-card-head">
      <div class="doc-card-kicker"><span class="stage-code">{meta["code"]}</span><span class="doc-card-eyebrow">{meta["verb"]} / {meta["role"]}</span></div>
      <div class="doc-card-title-row"><h2 class="doc-card-title">{meta["label"]}</h2><span class="card-status card-status--{status}">{STATUS_TEXT[status]}</span></div>
      <p class="doc-card-desc">{meta["desc"]}</p>
    </div>'''

    if status == "pending":
        return f'<article class="doc-card doc-card--empty">{header}<div class="empty-state"><span aria-hidden="true">↳</span><p>Waiting for the desk to open this file.</p></div></article>'
    if status == "error":
        err = esc(st.session_state.errors.get(stage, "Unknown error."))
        return f'<article class="doc-card doc-card--error">{header}<div class="error-state"><strong>File interrupted</strong><span>{err}</span></div></article>'
    if status == "running":
        return f'<article class="doc-card doc-card--running">{header}<div class="running-state"><span class="signal-bars" aria-hidden="true"><i></i><i></i><i></i></span><span>Working this file</span><span class="cursor">▍</span></div></article>'

    content = st.session_state.results.get(stage, "")
    if stage == "lead":
        inner = telex_html(content)
    elif stage == "source":
        inner = clip_html(content)
    elif stage == "draft":
        inner = f'<div class="report-sheet">{report_to_html(content)}</div>'
    else:
        inner = memo_html(content)
    return f'<article class="doc-card doc-card--{stage}">{header}{inner}</article>'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.html(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Newsreader:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --ink-950: #0D1017;
  --ink-800: #171B28;
  --ink-700: #232838;
  --paper: #EAE3D2;
  --brass: #CC8B3C;
  --redline: #B23F2E;
  --slate: #8A93A6;
}

.stApp { background: var(--ink-950); }
[data-testid="stSidebar"] { background: var(--ink-800); border-right: 1px solid var(--ink-700); }
[data-testid="stSidebar"] * { color: var(--paper); }
.stApp, .stApp p, .stApp li, .stApp label { font-family: 'Newsreader', Georgia, serif; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif; }

/* Masthead */
.masthead { padding: 0.4rem 0 1.2rem 0; border-bottom: 3px double var(--ink-700); margin-bottom: 2rem; }
.masthead-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em; color: var(--brass); text-transform: uppercase; }
.masthead-title { font-family: 'Fraunces', serif; font-optical-sizing: auto; font-weight: 700; font-size: clamp(2.4rem, 5vw, 4rem); color: var(--paper); margin: 0.15rem 0 0.3rem 0; letter-spacing: -0.01em; }
.masthead-sub { font-family: 'Newsreader', serif; font-style: italic; color: var(--slate); font-size: 1.05rem; }

/* Docket stepper */
.docket { display: flex; gap: 0; border: 1px solid var(--ink-700); border-radius: 4px; overflow: hidden; margin-bottom: 1.6rem; }
.docket-tab { flex: 1; padding: 0.85rem 1rem; background: var(--ink-800); border-right: 1px solid var(--ink-700); transition: background 0.3s ease; }
.docket-tab:last-child { border-right: none; }
.docket-tab.is-running { background: #1D2032; }
.docket-tab.is-error { background: #241716; }
.docket-tab-top { display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.3rem; }
.docket-tab-status { font-family: 'JetBrains Mono', monospace; font-size: 0.64rem; letter-spacing: 0.1em; color: var(--slate); }
.docket-tab.is-running .docket-tab-status { color: var(--brass); }
.docket-tab.is-done .docket-tab-status { color: var(--brass); }
.docket-tab.is-error .docket-tab-status { color: var(--redline); }
.docket-tab-label { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.15rem; color: var(--paper); }
.docket-tab-role { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--slate); margin-top: 0.15rem; }

.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; border: 1.5px solid var(--slate); background: transparent; }
.dot--running { background: var(--brass); border-color: var(--brass); animation: pulse 1.1s ease-in-out infinite; }
.dot--done { background: var(--brass); border-color: var(--brass); }
.dot--error { background: var(--redline); border-color: var(--redline); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

/* Document cards */
.doc-card { background: var(--ink-800); border: 1px solid var(--ink-700); border-radius: 4px; padding: 1.3rem 1.5rem 1.5rem; margin-bottom: 1.4rem; animation: fadeUp 0.5s ease; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.doc-card-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.08em; color: var(--slate); text-transform: uppercase; }
.doc-card-head { margin-bottom: 0.9rem; padding-bottom: 0.6rem; border-bottom: 1px solid var(--ink-700); }
.empty-state { color: var(--slate); font-style: italic; }
.running-state { color: var(--brass); font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
.doc-card--error { border-color: var(--redline); }
.error-state { color: var(--redline); font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
.cursor { animation: blink 1s step-start infinite; }
@keyframes blink { 50% { opacity: 0; } }

/* Telex ticker (lead / search) */
.telex-feed { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; line-height: 1.7; color: #D8CFAE; max-height: 260px; overflow-y: auto; padding-right: 0.5rem; border-left: 2px solid var(--brass); padding-left: 0.9rem; }
.telex-feed a { color: var(--brass); }
.telex-feed::-webkit-scrollbar { width: 6px; }
.telex-feed::-webkit-scrollbar-thumb { background: var(--ink-700); border-radius: 3px; }

/* Clip sheet (source / scrape) */
.clip-sheet { background: var(--paper); color: #2A2620; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.65; padding: 1.1rem 1.2rem; max-height: 260px; overflow-y: auto;
  clip-path: polygon(0 6px, 6px 0, calc(100% - 6px) 0, 100% 6px, 100% calc(100% - 6px), calc(100% - 6px) 100%, 6px 100%, 0 calc(100% - 6px)); }
.clip-tape { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; letter-spacing: 0.12em; color: #8a7a55; margin-bottom: 0.5rem; }

/* Report sheet (draft / writer) */
.report-sheet { background: var(--paper); color: #262218; padding: 1.8rem 2.2rem; border-radius: 2px; }
.report-sheet h2 { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--redline); border-bottom: 1px solid #cbbf9e; padding-bottom: 0.35rem; margin: 1.8rem 0 0.8rem 0; }
.report-sheet h2:first-child { margin-top: 0; }
.report-sheet h3 { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.05rem; color: #262218; margin: 1.3rem 0 0.5rem 0; }
.report-sheet p { font-family: 'Newsreader', serif; font-size: 1.02rem; line-height: 1.75; margin: 0 0 0.9rem 0; }
.report-sheet p.lede::first-letter { font-family: 'Fraunces', serif; font-weight: 700; font-size: 3.1em; float: left; line-height: 0.8; padding: 0.04em 0.08em 0 0; color: var(--redline); }
.report-sheet ul, .report-sheet ol { margin: 0 0 1.1rem 0; padding-left: 1.4rem; }
.report-sheet li { font-family: 'Newsreader', serif; font-size: 1.0rem; line-height: 1.65; margin-bottom: 0.4rem; }
.report-sheet li::marker { color: var(--redline); }
.report-sheet li p { margin: 0; }
.report-sheet strong { font-weight: 700; color: #1d1a12; }
.report-sheet hr { border: none; border-top: 1px solid #cbbf9e; margin: 1.7rem 0; }
.report-sheet a { color: var(--redline); text-decoration-color: #c98d80; }
.report-sheet table { width: 100%; border-collapse: collapse; margin: 0.4rem 0 1.4rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }
.report-sheet th { text-align: left; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.68rem; color: #6b6248; border-bottom: 2px solid #cbbf9e; padding: 0.45rem 0.7rem; }
.report-sheet td { padding: 0.45rem 0.7rem; border-bottom: 1px solid #ddd3ba; vertical-align: top; }
.report-sheet tr:last-child td { border-bottom: none; }

/* Memo (desk / critic) */
.memo { background: var(--paper); color: #262218; padding: 1.6rem 1.9rem; border-radius: 2px; position: relative; }
.memo-head { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: #6b6248; margin: 1rem 0 0.4rem 0; }
.memo-head:first-of-type { margin-top: 0; }
.redline-list { list-style: none; padding: 0; margin: 0 0 0.4rem 0; }
.redline-list li { position: relative; padding-left: 1.1rem; font-family: 'Newsreader', serif; line-height: 1.6; margin-bottom: 0.3rem; }
.redline-list--up li::before { content: '+'; position: absolute; left: 0; color: #4a7a52; font-weight: 700; }
.redline-list--down li::before { content: '—'; position: absolute; left: 0; color: var(--redline); font-weight: 700; }
.verdict { font-family: 'Newsreader', serif; font-style: italic; border-left: 3px solid var(--redline); padding-left: 0.9rem; margin-top: 1.1rem; color: #3a3427; }
.stamp { position: absolute; top: 1.4rem; right: 1.6rem; width: 84px; height: 84px; border-radius: 50%; border: 3px dashed var(--redline); display: flex; flex-direction: column; align-items: center; justify-content: center; transform: rotate(-9deg); animation: stampIn 0.4s ease-out; }
.stamp-value { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.7rem; color: var(--redline); line-height: 1; }
.stamp-label { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: var(--redline); letter-spacing: 0.08em; }
@keyframes stampIn { from { opacity: 0; transform: rotate(-9deg) scale(1.6); } to { opacity: 1; transform: rotate(-9deg) scale(1); } }

/* Sidebar form */
.brief-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.12em; color: var(--brass); text-transform: uppercase; margin-bottom: 0.2rem; }
.brief-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.4rem; color: var(--paper); margin-bottom: 1rem; }
[data-testid="stSidebar"] .stTextInput input { background: var(--ink-950); color: var(--paper); border: 1px solid var(--ink-700); font-family: 'Newsreader', serif; }
[data-testid="stSidebar"] .stButton button { background: var(--brass); color: var(--ink-950); border: none; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.06em; text-transform: uppercase; font-size: 0.78rem; font-weight: 600; width: 100%; padding: 0.6rem 0; }
[data-testid="stSidebar"] .stButton button:hover { background: #e0a458; color: var(--ink-950); }
.stDownloadButton button { background: transparent; border: 1px solid var(--brass); color: var(--brass); font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase; }

/* Assignment room — a high-contrast operational layer over the editorial file. */
:root {
  --ink-950: #080D18;
  --ink-900: #0E1628;
  --ink-800: #131E33;
  --ink-700: #293751;
  --paper: #F3F0E8;
  --paper-dim: #CBC8BE;
  --brass: #FF8B61;
  --cyan: #66D6E8;
  --redline: #FF6D78;
  --slate: #95A3BD;
  --quiet: #65738D;
  --success: #83D6AB;
}

html { scroll-behavior: smooth; }
.stApp {
  background:
    radial-gradient(ellipse 70% 45% at 92% -8%, rgba(102, 214, 232, 0.10), transparent 65%),
    radial-gradient(ellipse 44% 30% at 8% 88%, rgba(255, 139, 97, 0.08), transparent 70%),
    var(--ink-950);
}
.block-container { max-width: 1220px; padding-top: 2.6rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] { background: var(--ink-900); border-right: 1px solid rgba(149, 163, 189, 0.18); }
[data-testid="stSidebar"] > div:first-child { padding-top: 2.1rem; }

/* The masthead behaves like a calm briefing header, not a decorative newspaper banner. */
.masthead {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1.25rem;
  align-items: end;
  padding: 0 0 1.5rem;
  margin: 0 0 1.5rem;
  border-bottom: 1px solid rgba(149, 163, 189, 0.30);
}
.masthead::after { content: ''; position: absolute; height: 2px; width: min(190px, 32%); bottom: -1px; left: 0; background: linear-gradient(90deg, var(--brass), var(--cyan)); }
.masthead-eyebrow { color: var(--cyan); font-size: 0.68rem; letter-spacing: 0.16em; }
.masthead-title { max-width: 780px; margin: 0.35rem 0 0.45rem; color: var(--paper); font-size: clamp(2.8rem, 6.5vw, 5.5rem); line-height: 0.94; letter-spacing: -0.055em; }
.masthead-sub { max-width: 500px; color: var(--paper-dim); font-size: 1.18rem; line-height: 1.35; }
.masthead-note { align-self: center; display: flex; align-items: center; gap: 0.55rem; color: var(--slate); font: 500 0.68rem/1.25 'JetBrains Mono', monospace; letter-spacing: 0.08em; text-transform: uppercase; }
.masthead-note::before { content: ''; width: 8px; height: 8px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 0 4px rgba(102, 214, 232, 0.12); }

/* The horizontal wire is the page's signature: it connects actual handoffs. */
.docket { position: relative; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0; overflow: visible; margin: 0 0 2.4rem; border: 1px solid rgba(149, 163, 189, 0.20); border-radius: 0; background: rgba(19, 30, 51, 0.62); }
.wire-rail { position: absolute; z-index: 0; top: 34px; right: 12.5%; left: 12.5%; height: 1px; background: linear-gradient(90deg, var(--quiet), var(--quiet) 72%, rgba(101, 115, 141, 0.12)); }
.docket-tab { position: relative; z-index: 1; min-height: 132px; padding: 1rem 1.1rem 1.05rem; background: transparent; border: 0; border-right: 1px solid rgba(149, 163, 189, 0.16); transition: background .22s ease, box-shadow .22s ease; }
.docket-tab:last-child { border-right: 0; }
.docket-tab:hover { background: rgba(255,255,255,0.035); }
.docket-tab.is-running { background: linear-gradient(180deg, rgba(255, 139, 97, 0.12), rgba(255, 139, 97, 0.025)); box-shadow: inset 0 -2px 0 var(--brass); }
.docket-tab.is-done { background: rgba(131, 214, 171, 0.035); }
.docket-tab.is-error { background: rgba(255, 109, 120, 0.08); box-shadow: inset 0 -2px 0 var(--redline); }
.docket-tab-top { position: relative; display: flex; min-height: 20px; align-items: center; gap: 0.45rem; margin: 0 0 0.72rem; }
.stage-code { font: 500 0.62rem/1 'JetBrains Mono', monospace; letter-spacing: 0.08em; color: var(--quiet); }
.dot { position: relative; width: 10px; height: 10px; border: 2px solid var(--ink-800); background: var(--quiet); box-shadow: 0 0 0 1px var(--quiet); }
.dot--running { background: var(--brass); border-color: var(--ink-800); box-shadow: 0 0 0 1px var(--brass), 0 0 18px rgba(255, 139, 97, 0.65); }
.dot--done { background: var(--success); box-shadow: 0 0 0 1px var(--success); }
.dot--error { background: var(--redline); box-shadow: 0 0 0 1px var(--redline); }
.docket-tab-status { color: var(--quiet); font-size: 0.61rem; letter-spacing: 0.1em; }
.docket-tab.is-running .docket-tab-status { color: var(--brass); }
.docket-tab.is-done .docket-tab-status { color: var(--success); }
.docket-tab.is-error .docket-tab-status { color: var(--redline); }
.docket-tab-label { display: flex; align-items: baseline; justify-content: space-between; gap: 0.75rem; color: var(--paper); font: 600 1.35rem/1 'Fraunces', Georgia, serif; }
.docket-tab-verb { color: var(--slate); font: 500 0.62rem/1 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.07em; }
.docket-tab-role { margin-top: 0.4rem; color: var(--slate); font-size: 0.68rem; line-height: 1.3; }

/* Files are intentionally roomy and read as distinct artifacts, not generic cards. */
.doc-card { position: relative; overflow: hidden; margin: 0 0 1.1rem; padding: 1.25rem 1.45rem 1.5rem; border: 1px solid rgba(149, 163, 189, 0.20); border-radius: 0; background: rgba(19, 30, 51, 0.75); box-shadow: none; animation: fadeUp .45s ease both; }
.doc-card::before { content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%; background: var(--quiet); }
.doc-card--lead::before { background: var(--cyan); }
.doc-card--source::before, .doc-card--draft::before { background: var(--brass); }
.doc-card--desk::before { background: var(--redline); }
.doc-card--running::before { background: var(--brass); animation: scan 1.2s ease-in-out infinite; }
.doc-card--error { border-color: rgba(255, 109, 120, 0.55); }
.doc-card--error::before { background: var(--redline); }
.doc-card-head { margin: 0 0 1.05rem; padding: 0 0 0.85rem; border-bottom: 1px solid rgba(149, 163, 189, 0.18); }
.doc-card-kicker { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.5rem; }
.doc-card-eyebrow { color: var(--slate); font-size: 0.65rem; letter-spacing: 0.11em; }
.doc-card-title-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.doc-card-title { margin: 0; color: var(--paper); font: 600 clamp(1.35rem, 2.5vw, 1.8rem)/1 'Fraunces', Georgia, serif; letter-spacing: -0.025em; }
.card-status { flex: 0 0 auto; color: var(--quiet); font: 500 0.61rem/1 'JetBrains Mono', monospace; letter-spacing: 0.1em; }
.card-status--done { color: var(--success); }.card-status--running { color: var(--brass); }.card-status--error { color: var(--redline); }
.doc-card-desc { max-width: 630px; margin: 0.5rem 0 0; color: var(--slate); font: italic 0.98rem/1.35 'Newsreader', Georgia, serif; }
.doc-card--empty { min-height: 168px; background: rgba(19, 30, 51, 0.38); }
.empty-state { display: flex; align-items: center; gap: 0.7rem; color: var(--quiet); font: 0.82rem/1.4 'JetBrains Mono', monospace; }
.empty-state span { color: var(--cyan); font-size: 1.1rem; }.empty-state p { margin: 0; }
.running-state { display: inline-flex; align-items: center; gap: 0.62rem; color: var(--brass); font: 500 0.75rem/1 'JetBrains Mono', monospace; letter-spacing: 0.05em; text-transform: uppercase; }
.signal-bars { display: inline-flex; height: 13px; align-items: end; gap: 2px; }.signal-bars i { display: block; width: 3px; background: currentColor; animation: levels .9s ease-in-out infinite alternate; }.signal-bars i:nth-child(1) { height: 40%; }.signal-bars i:nth-child(2) { height: 100%; animation-delay: -.3s; }.signal-bars i:nth-child(3) { height: 65%; animation-delay: -.55s; }
.error-state { display: grid; gap: 0.35rem; color: var(--redline); font: 0.8rem/1.45 'JetBrains Mono', monospace; }.error-state strong { text-transform: uppercase; letter-spacing: 0.07em; font-weight: 500; }
@keyframes scan { 0%,100% { opacity: 1; } 50% { opacity: .28; } } @keyframes levels { to { transform: scaleY(.35); } }

.telex-feed { color: #DBE5F1; border-left: 1px solid var(--cyan); font-size: 0.79rem; line-height: 1.78; }.telex-feed a { color: var(--cyan); text-underline-offset: 0.18em; }.telex-feed::-webkit-scrollbar-thumb { background: var(--quiet); }
.clip-sheet { background: #E9E5DC; color: #172033; padding: 1.25rem 1.35rem; border: 1px solid #C5C0B4; box-shadow: 7px 7px 0 rgba(8,13,24,.55); }.clip-tape { color: #627086; }
.report-sheet { background: #F3F0E8; color: #182033; padding: clamp(1.5rem, 4vw, 2.65rem); border: 1px solid #D1CCC0; box-shadow: 9px 9px 0 rgba(8,13,24,.50); }.report-sheet h2 { color: #D4535A; }.report-sheet h3, .report-sheet strong { color: #172033; }.report-sheet p.lede::first-letter, .report-sheet li::marker { color: #D4535A; }.report-sheet a { color: #B5444C; }.report-sheet table { font-size: .78rem; }
.memo { background: #F3F0E8; color: #182033; padding: 1.8rem 2rem; border: 1px solid #D1CCC0; box-shadow: 9px 9px 0 rgba(8,13,24,.50); }.memo-head { color: #647188; }.redline-list li { line-height: 1.65; }.stamp { border-color: #D4535A; }.stamp-value,.stamp-label { color: #D4535A; }.verdict { border-left-color: #D4535A; color: #273146; }

/* Assignment controls use familiar form behavior with stronger hierarchy. */
.brief-eyebrow { color: var(--cyan); font-size: .65rem; letter-spacing: .14em; }.brief-title { margin-bottom: .35rem; color: var(--paper); font-size: 1.8rem; letter-spacing: -.035em; }.brief-help { margin: 0 0 1.15rem; color: var(--slate); font: .88rem/1.45 'Newsreader', Georgia, serif; }
[data-testid="stSidebar"] .stTextInput input { min-height: 2.8rem; border: 1px solid rgba(149,163,189,.40); border-radius: 0; background: #080D18; color: var(--paper); font-size: .98rem; } [data-testid="stSidebar"] .stTextInput input:focus { border-color: var(--cyan); box-shadow: 0 0 0 3px rgba(102,214,232,.13); }
[data-testid="stSidebar"] .stButton button { min-height: 2.85rem; border-radius: 0; background: var(--brass); color: #111827; box-shadow: 4px 4px 0 rgba(255, 139, 97, .22); transition: transform .18s ease, box-shadow .18s ease, background .18s ease; }[data-testid="stSidebar"] .stButton button:hover { background: #FFAE7E; transform: translate(-2px,-2px); box-shadow: 6px 6px 0 rgba(255, 139, 97, .16); }[data-testid="stSidebar"] .stButton button:active { transform: translate(0); box-shadow: 2px 2px 0 rgba(255, 139, 97, .18); }
.stDownloadButton { margin-top: 1.1rem; }.stDownloadButton button { width: 100%; min-height: 2.5rem; border-color: rgba(102,214,232,.7); border-radius: 0; color: var(--cyan); }.stDownloadButton button:hover { border-color: var(--cyan); background: rgba(102,214,232,.08); color: var(--paper); }
button:focus-visible, input:focus-visible, a:focus-visible { outline: 2px solid var(--cyan) !important; outline-offset: 3px !important; }

@media (max-width: 760px) { .block-container { padding: 1.65rem 1rem 3rem; }.masthead { grid-template-columns: 1fr; gap: .75rem; }.masthead-title { font-size: clamp(2.65rem, 14vw, 4rem); }.masthead-note { align-self: start; }.docket { grid-template-columns: 1fr 1fr; }.wire-rail { display: none; }.docket-tab { min-height: 118px; padding: .85rem; }.docket-tab:nth-child(3) { border-right: 0; }.docket-tab:nth-child(-n+2) { border-bottom: 1px solid rgba(149,163,189,.16); }.docket-tab-label { font-size: 1.18rem; }.docket-tab-verb { display: none; }.doc-card { padding: 1.05rem 1rem 1.2rem; }.doc-card-title-row { align-items: flex-start; }.card-status { padding-top: .2rem; }.report-sheet,.memo { padding: 1.25rem; box-shadow: 5px 5px 0 rgba(8,13,24,.5); }.stamp { position: static; margin: 0 0 1.25rem auto; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: .01ms !important; } }
</style>
    """
)


# ---------------------------------------------------------------------------
# Sidebar — the assignment brief
# ---------------------------------------------------------------------------

with st.sidebar:
    st.html('<div class="brief-eyebrow">New assignment</div>')
    st.html('<div class="brief-title">Pitch the story</div>')
    st.html('<p class="brief-help">Give the desk a precise subject. It will scout, verify, write, and review one connected file.</p>')
    topic = st.text_input(
        "What's the story?",
        placeholder="e.g. Quantum computing breakthroughs in 2026",
        help="Use a specific question, person, event, or trend for more useful research.",
    )
    dispatch = st.button("Dispatch the desk")
    st.html(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.67rem;line-height:1.6;letter-spacing:.03em;color:#95A3BD;margin-top:1.15rem;">'
        "THE ROUTE: SCOUT → VERIFY → WRITE → REVIEW</p>"
    )
    if st.session_state.results.get("draft") and st.session_state.stage_status.get("draft") == "done":
        st.download_button(
            "Download filed report",
            st.session_state.results["draft"],
            file_name=f"{(st.session_state.topic or 'report').strip().replace(' ', '_')}.md",
            mime="text/markdown",
        )

# ---------------------------------------------------------------------------
# Main — masthead + docket + document cards
# ---------------------------------------------------------------------------

st.html(
    f"""
    <div class="masthead">
      <div>
        <div class="masthead-eyebrow">Global desk / {date.today().strftime('%B %d, %Y')}</div>
        <h1 class="masthead-title">The Assignment Desk</h1>
        <div class="masthead-sub">From first signal to finished file, with every handoff visible.</div>
      </div>
      <div class="masthead-note">Live research workflow</div>
    </div>
    """
)

stepper_slot = st.empty()
stage_slots = {s: st.empty() for s in STAGES}

stepper_slot.html(stepper_html())
for s in STAGES:
    stage_slots[s].html(stage_card_html(s))


def run_stage(key, fn):
    st.session_state.stage_status[key] = "running"
    stepper_slot.html(stepper_html())
    stage_slots[key].html(stage_card_html(key))
    try:
        result = fn()
        st.session_state.results[key] = result
        st.session_state.stage_status[key] = "done"
    except Exception as exc:  # noqa: BLE001 — surfaced to the user as a killed story
        st.session_state.errors[key] = str(exc)
        st.session_state.stage_status[key] = "error"
    stepper_slot.html(stepper_html())
    stage_slots[key].html(stage_card_html(key))
    return st.session_state.stage_status[key] == "done"


if dispatch:
    if not topic.strip():
        st.toast("Give the desk a topic before you dispatch it.", icon="🗞️")
    elif not (os.getenv("TAVILY_API_KEY") and os.getenv("MISTRAL_API_KEY")):
        st.error(
            "The wire is down — TAVILY_API_KEY and MISTRAL_API_KEY aren't set. "
            "Add them to the .env file in this project and restart the app."
        )
    else:
        from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain

        st.session_state.topic = topic
        reset_run()
        stepper_slot.html(stepper_html())
        for s in STAGES:
            stage_slots[s].html(stage_card_html(s))

        def do_lead():
            agent = build_search_agent()
            result = agent.invoke(
                {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
            )
            return extract_text(result["messages"][-1].content)

        if run_stage("lead", do_lead):

            def do_source():
                agent = build_reader_agent()
                result = agent.invoke(
                    {
                        "messages": [
                            (
                                "user",
                                f"Based on the following search results about '{topic}', "
                                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                                f"Search Results:\n{st.session_state.results['lead'][:800]}",
                            )
                        ]
                    }
                )
                return extract_text(result["messages"][-1].content)

            if run_stage("source", do_source):

                def do_draft():
                    combined = (
                        f"SEARCH RESULTS:\n{st.session_state.results['lead']}\n\n"
                        f"DETAILED SCRAPED CONTENT:\n{st.session_state.results['source']}"
                    )
                    return writer_chain.invoke({"topic": topic, "research": combined})

                if run_stage("draft", do_draft):

                    def do_desk():
                        return critic_chain.invoke({"report": st.session_state.results["draft"]})

                    run_stage("desk", do_desk)
