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
    "lead": {"label": "Lead", "role": "Search correspondent", "desc": "Wires in the first leads on the story."},
    "source": {"label": "Source", "role": "Field reader", "desc": "Pulls the fullest account from the strongest lead."},
    "draft": {"label": "Draft", "role": "Staff writer", "desc": "Files the story for the desk."},
    "desk": {"label": "Desk", "role": "Copy editor", "desc": "Reads it back, marks it up, stamps a grade."},
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
    return f'<span class="dot dot--{status}"></span>'


def stepper_html() -> str:
    tabs = []
    for s in STAGES:
        status = st.session_state.stage_status[s]
        meta = STAGE_META[s]
        tabs.append(
            f"""
            <div class="docket-tab is-{status}">
              <div class="docket-tab-top">{status_dot(status)}<span class="docket-tab-status">{STATUS_TEXT[status]}</span></div>
              <div class="docket-tab-label">{meta['label']}</div>
              <div class="docket-tab-role">{meta['role']}</div>
            </div>
            """
        )
    return f'<div class="docket">{"".join(tabs)}</div>'


def stage_card_html(stage: str) -> str:
    meta = STAGE_META[stage]
    status = st.session_state.stage_status[stage]
    header = f'<div class="doc-card-head"><span class="doc-card-eyebrow">{meta["label"].upper()} — {meta["desc"]}</span></div>'

    if status == "pending":
        return f'<div class="doc-card doc-card--empty">{header}<p class="empty-state">Awaiting assignment.</p></div>'
    if status == "error":
        err = esc(st.session_state.errors.get(stage, "Unknown error."))
        return f'<div class="doc-card doc-card--error">{header}<p class="error-state">Story killed &mdash; {err}</p></div>'
    if status == "running":
        return f'<div class="doc-card doc-card--running">{header}<p class="running-state">On the wire<span class="cursor">▍</span></p></div>'

    content = st.session_state.results.get(stage, "")
    if stage == "lead":
        inner = telex_html(content)
    elif stage == "source":
        inner = clip_html(content)
    elif stage == "draft":
        inner = f'<div class="report-sheet">{report_to_html(content)}</div>'
    else:
        inner = memo_html(content)
    return f'<div class="doc-card doc-card--{stage}">{header}{inner}</div>'


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
</style>
    """
)


# ---------------------------------------------------------------------------
# Sidebar — the assignment brief
# ---------------------------------------------------------------------------

with st.sidebar:
    st.html('<div class="brief-eyebrow">New assignment</div>')
    st.html('<div class="brief-title">Pitch the story</div>')
    topic = st.text_input(
        "What's the story?",
        placeholder="e.g. Quantum computing breakthroughs in 2026",
        label_visibility="collapsed",
    )
    dispatch = st.button("Dispatch the desk")
    st.html(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:#8A93A6;margin-top:0.8rem;">'
        "Four correspondents work the story in sequence: a search lead, a field reader, "
        "a staff writer, and a copy desk that grades the file.</p>"
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
      <div class="masthead-eyebrow">Vol. I — Global Desk — {date.today().strftime('%B %d, %Y')}</div>
      <h1 class="masthead-title">The Assignment Desk</h1>
      <div class="masthead-sub">Four correspondents. One story, filed start to finish.</div>
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
