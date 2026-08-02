"""A polished Streamlit interface for the Mistral persona chatbot.

Run from the project root with:
    streamlit run chatmodels/Uichatbot.py
"""

from __future__ import annotations

from collections.abc import Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI


load_dotenv()

st.set_page_config(
    page_title="Moodboard AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODES = {
    "Funny": {
        "emoji": "😂",
        "tagline": "Comedy with a clever edge",
        "color": "#ffb703",
        "prompt": (
            "You are a very funny AI assistant. Be genuinely helpful, witty, and "
            "playful. Use light jokes when they improve the answer, but never make "
            "fun of the user or sacrifice clarity."
        ),
        "ideas": [
            "Explain quantum computing using a pizza analogy.",
            "Give me a funny but useful plan for beating procrastination.",
        ],
    },
    "Angry": {
        "emoji": "😤",
        "tagline": "Blunt, fierce, and focused",
        "color": "#fb5607",
        "prompt": (
            "You are an intense, impatient AI assistant with a sharp, assertive "
            "voice. Be helpful and accurate, never abusive or insulting. Keep answers "
            "direct and decisive, with restrained dramatic flair."
        ),
        "ideas": [
            "Give me a no-excuses plan to organize my week.",
            "Tell me the hard truth about improving my study habits.",
        ],
    },
    "Sad": {
        "emoji": "🌧️",
        "tagline": "Gentle, thoughtful, and reflective",
        "color": "#8ecae6",
        "prompt": (
            "You are a gentle, wistful AI assistant. Reply with warmth and emotional "
            "sensitivity while remaining constructive and helpful. Do not encourage "
            "hopelessness, self-harm, or emotional dependency."
        ),
        "ideas": [
            "Write a quietly hopeful note for a difficult day.",
            "Help me make a calm evening reset routine.",
        ],
    },
}


def inject_styles() -> None:
    """Apply the app's visual system without requiring external assets."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 9% 4%, rgba(131, 56, 236, .22), transparent 25rem),
                radial-gradient(circle at 90% 12%, rgba(255, 183, 3, .17), transparent 22rem),
                #0b1020;
            color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background: rgba(10, 15, 33, .84);
            border-right: 1px solid rgba(255, 255, 255, .09);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.6rem; }
        .brand { font-size: 1.45rem; font-weight: 800; letter-spacing: -.04em; }
        .eyebrow { color: #9aa8c7; font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
        .hero { padding: 2.4rem 1.1rem 1.35rem; max-width: 780px; }
        .hero h1 { font-size: clamp(2.35rem, 5vw, 4.6rem); line-height: .96; letter-spacing: -.065em; margin: .35rem 0 .85rem; }
        .hero p { color: #b9c4db; font-size: 1.08rem; line-height: 1.6; margin: 0; }
        .mode-pill {
            display: inline-block; margin-top: 1.15rem; padding: .4rem .8rem;
            border: 1px solid rgba(255,255,255,.15); border-radius: 999px;
            background: rgba(255,255,255,.06); font-size: .86rem; font-weight: 650;
        }
        .mode-card {
            background: rgba(255, 255, 255, .055); border: 1px solid rgba(255, 255, 255, .1);
            border-radius: 18px; padding: 1rem 1.1rem; margin: .8rem 0 1.25rem;
        }
        .mode-card strong { font-size: 1.05rem; }
        .mode-card span { color: #b9c4db; display: block; font-size: .88rem; margin-top: .2rem; }
        .stButton > button { border-radius: 12px; font-weight: 650; transition: .2s ease; }
        .stButton > button:hover { border-color: rgba(255,255,255,.7); transform: translateY(-1px); }
        [data-testid="stChatMessage"] { border-radius: 18px; padding: .28rem .45rem; }
        [data-testid="stChatMessage"] p { line-height: 1.65; }
        [data-testid="stChatInput"] { border-radius: 16px; }
        .empty-note { color: #9aa8c7; padding: .8rem 0 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_model() -> ChatMistralAI:
    """Create the client once per Streamlit server process."""
    return ChatMistralAI(model="mistral-small-2506", temperature=0.9)


def reset_conversation() -> None:
    st.session_state.messages = []


def stream_response(messages: list[dict[str, str]], mode: str) -> Iterator[str]:
    """Stream the model output while preserving persona and conversation context."""
    history = [SystemMessage(content=MODES[mode]["prompt"])]
    for message in messages:
        message_class = HumanMessage if message["role"] == "user" else AIMessage
        history.append(message_class(content=message["content"]))

    for chunk in get_model().stream(history):
        if chunk.content:
            yield str(chunk.content)


def main() -> None:
    inject_styles()
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("mode", "Funny")

    with st.sidebar:
        st.markdown('<div class="brand">✦ Moodboard AI</div>', unsafe_allow_html=True)
        st.caption("One question. Three personalities.")
        st.divider()
        st.markdown("#### Choose your vibe")
        selected_mode = st.radio(
            "Persona",
            options=list(MODES),
            format_func=lambda key: f"{MODES[key]['emoji']}  {key}",
            label_visibility="collapsed",
        )
        if selected_mode != st.session_state.mode:
            st.session_state.mode = selected_mode
            reset_conversation()

        selected = MODES[st.session_state.mode]
        st.markdown(
            f'<div class="mode-card" style="border-color: {selected["color"]}66;">'
            f'<strong>{selected["emoji"]} {st.session_state.mode} mode</strong>'
            f'<span>{selected["tagline"]}</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("↺  New conversation", use_container_width=True):
            reset_conversation()
            st.rerun()
        st.divider()
        st.caption("Powered by Mistral Small · Your messages stay in this session.")

    mode = st.session_state.mode
    details = MODES[mode]
    st.markdown(
        f'''<section class="hero">
            <div class="eyebrow">Persona chat · choose a mood</div>
            <h1>Talk it out,<br>your way.</h1>
            <p>Ask anything. Moodboard AI answers with a point of view—switch the
            personality anytime for a fresh conversation.</p>
            <div class="mode-pill" style="color: {details["color"]};">{details["emoji"]} {mode}: {details["tagline"]}</div>
        </section>''',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown('<div class="empty-note">Need a starting point? Try one of these.</div>', unsafe_allow_html=True)
        suggestions = st.columns(2)
        for column, idea in zip(suggestions, details["ideas"]):
            if column.button(idea, key=f"suggestion-{mode}-{idea}", use_container_width=True):
                st.session_state.pending_prompt = idea

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else details["emoji"]):
            st.markdown(message["content"])

    prompt = st.chat_input(f"Message {mode} mode…")
    prompt = prompt or st.session_state.pop("pending_prompt", None)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=details["emoji"]):
            try:
                response = st.write_stream(stream_response(st.session_state.messages, mode))
            except Exception as error:
                st.error(
                    "I couldn't reach the model. Check that `MISTRAL_API_KEY` is set in your "
                    "environment or `.env` file, then try again."
                )
                st.caption(f"Technical detail: {error}")
                return

        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
