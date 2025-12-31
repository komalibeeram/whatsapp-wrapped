import os
from datetime import datetime
import json
import base64
import hashlib
import zlib
import urllib.request
from urllib.parse import urlencode, quote, unquote

import altair as alt
import pandas as pd
import streamlit as st

from analytics import compute_stats
from llm.ai_insights import (
    MessageRecord,
    _get_client,
    chat_evolution,
    chunk_messages,
    conversation_summary,
    emotional_tone,
    qa_answer,
    topic_intelligence,
    wrapped_beats,
)
from parser import parse_chat


st.set_page_config(page_title="WhatsApp Wrapped", layout="wide")

# WhatsApp Wrapped theme - WhatsApp green + dark mode
THEME = {
    "bg": "#0B141A",  # WhatsApp dark background
    "text": "#E9EDEF",  # WhatsApp light text
    "text_secondary": "#8696A0",  # WhatsApp secondary text
    "whatsapp_green": "#25D366",  # WhatsApp brand green
    "green_glow": "rgba(37, 211, 102, 0.15)",  # Subtle green glow
    "card_bg": "#1F2C34",  # WhatsApp message bubble dark
    "card_border": "rgba(37, 211, 102, 0.2)",  # Green-tinted border
    "hero_bg": "linear-gradient(135deg, #25D366, #128C7E, #075E54)",  # WhatsApp green gradient
    "share_bg": "linear-gradient(135deg, #25D366 0%, #34B7F1 100%)",  # Green to blue gradient for share
    "hero_text": "#ffffff",
    "divider": "rgba(134, 150, 160, 0.15)",
}
theme = THEME

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    body, .main {{
        background: {theme['bg']}; 
        color: {theme['text']}; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    h1, h2, h3, h4, h5, h6, p, div, span {{
        color: {theme['text']};
    }}
    
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }}
    
    /* Hero card - colorful gradient for celebration */
    .hero-card {{
        padding: 32px 24px;
        border-radius: 16px;
        background: {theme['hero_bg']};
        color: {theme['hero_text']};
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 32px;
    }}
    
    .hero-title {{
        font-size: 12px;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        opacity: 0.9;
        font-weight: 600;
        margin-bottom: 8px;
    }}
    
    .hero-number {{
        font-size: 28px;
        font-weight: 800;
        margin: 8px 0;
        line-height: 1.3;
    }}
    
    .hero-sub {{
        font-size: 14px;
        opacity: 0.9;
        margin-top: 8px;
    }}
    
    /* Message-style cards - WhatsApp bubble aesthetic */
    .mini-card {{
        flex: 1;
        padding: 18px;
        border-radius: 12px;
        background: {theme['card_bg']};
        border-left: 3px solid {theme['whatsapp_green']};
        color: {theme['text']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 200ms ease;
    }}
    
    .mini-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(37, 211, 102, 0.2);
        border-left-color: {theme['whatsapp_green']};
    }}
    
    /* Award cards */
    .award-row {{
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding-bottom: 8px;
        margin-top: 16px;
    }}
    
    .award-card {{
        min-width: 200px;
        padding: 16px;
        border-radius: 12px;
        background: {theme['card_bg']};
        border: 1px solid {theme['card_border']};
        color: {theme['text']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 200ms ease;
    }}
    
    .award-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(37, 211, 102, 0.15);
        border-color: {theme['whatsapp_green']};
    }}
    
    /* Section headers */
    .section-title {{
        font-size: 22px;
        font-weight: 700;
        margin: 32px 0 8px 0;
        letter-spacing: -0.3px;
        color: {theme['text']};
    }}
    
    .section-caption {{
        opacity: 0.7;
        margin-bottom: 16px;
        color: {theme['text_secondary']};
        font-size: 14px;
    }}
    
    /* Micro text */
    .micro {{
        font-size: 11px;
        opacity: 0.65;
        color: {theme['text']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }}
    
    /* Commentary lines - prominent callout blocks */
    .commentary {{
        text-align: center;
        opacity: 0;
        font-size: 16px;
        font-weight: 600;
        margin: 40px auto;
        padding: 20px 24px;
        max-width: 600px;
        color: {theme['whatsapp_green']};
        background: rgba(37, 211, 102, 0.08);
        border-radius: 12px;
        border-left: 3px solid {theme['whatsapp_green']};
        line-height: 1.5;
        animation: fadeInUp 0.8s ease-out 0.3s forwards;
    }}
    
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(15px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    /* Section transitions */
    .section-title {{
        animation: fadeInSlide 0.6s ease-out;
    }}
    
    .section-caption {{
        animation: fadeIn 0.7s ease-out 0.1s backwards;
    }}
    
    @keyframes fadeIn {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}
    
    @keyframes fadeInSlide {{
        from {{
            opacity: 0;
            transform: translateX(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    /* Card entrance animations */
    .mini-card {{
        animation: fadeInScale 0.5s ease-out backwards;
    }}
    
    @keyframes fadeInScale {{
        from {{
            opacity: 0;
            transform: scale(0.96) translateY(10px);
        }}
        to {{
            opacity: 1;
            transform: scale(1) translateY(0);
        }}
    }}
    
    .hero-card {{
        animation: heroEntrance 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}
    
    @keyframes heroEntrance {{
        0% {{
            opacity: 0;
            transform: scale(0.95) translateY(20px);
        }}
        60% {{
            transform: scale(1.01) translateY(-5px);
        }}
        100% {{
            opacity: 1;
            transform: scale(1) translateY(0);
        }}
    }}
    
    /* Award cards staggered entrance */
    .award-card {{
        animation: fadeInScale 0.5s ease-out backwards;
    }}
    
    .award-card:nth-child(1) {{ animation-delay: 0.1s; }}
    .award-card:nth-child(2) {{ animation-delay: 0.2s; }}
    .award-card:nth-child(3) {{ animation-delay: 0.3s; }}
    .award-card:nth-child(4) {{ animation-delay: 0.4s; }}
    
    /* Staggered column animations */
    [data-testid="column"]:nth-child(1) .mini-card {{
        animation-delay: 0.1s;
    }}
    
    [data-testid="column"]:nth-child(2) .mini-card {{
        animation-delay: 0.2s;
    }}
    
    [data-testid="column"]:nth-child(3) .mini-card {{
        animation-delay: 0.3s;
    }}
    
    [data-testid="column"]:nth-child(4) .mini-card {{
        animation-delay: 0.4s;
    }}
    
    /* Dividers */
    hr {{
        border: none;
        border-top: 1px solid {theme['divider']};
        margin: 32px 0;
    }}
    
    /* File uploader */
    div[data-testid="stFileUploader"] section {{
        border: 2px dashed {theme['card_border']};
        background: {theme['card_bg']};
        border-radius: 12px;
        padding: 32px;
    }}
    
    div[data-testid="stFileUploader"] section:hover {{
        border-color: {theme['whatsapp_green']};
        background: rgba(31, 44, 52, 0.8);
    }}
    
    /* Streamlit overrides */
    .stMetric {{
        background: transparent;
    }}
    
    /* Info boxes */
    .stAlert {{
        background: {theme['card_bg']};
        border-left: 3px solid {theme['whatsapp_green']};
        color: {theme['text']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper functions for data compression/decompression
def compress_data(data):
    """Compress data to reduce URL length (max compression)."""
    json_str = json.dumps(data, default=str)
    compressed = zlib.compress(json_str.encode(), level=9)
    return base64.urlsafe_b64encode(compressed).decode()

def decompress_data(encoded_data):
    """Decompress data from URL parameter"""
    try:
        compressed = base64.urlsafe_b64decode(encoded_data.encode())
        json_str = zlib.decompress(compressed).decode()
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Failed to load shared Wrapped: {str(e)}")
        return None


def shorten_url(long_url: str) -> str:
    """Shorten URLs via multiple free services; fallback to long URL on failure."""
    services = [
        "https://tinyurl.com/api-create.php?url=",
        "https://is.gd/create.php?format=simple&url=",
    ]
    for base in services:
        try:
            api_url = f"{base}{quote(long_url)}"
            with urllib.request.urlopen(api_url, timeout=6) as resp:
                short = resp.read().decode().strip()
                if short.startswith("http"):
                    return short
        except Exception:
            continue
    return long_url


def is_rate_limit_error(err: Exception | str) -> bool:
    """Detect rate limit / 429 messages."""
    msg = str(err).lower() if err is not None else ""
    return "rate limit" in msg or "429" in msg

# Initialize shared data cache in session state
if 'shared_wrappeds' not in st.session_state:
    st.session_state.shared_wrappeds = {}

if 'ai_rate_limited' not in st.session_state:
    st.session_state.ai_rate_limited = False

# Check for shared wrapped link
query_params = st.query_params
shared_data_param = query_params.get("d", None) or query_params.get("data", None)

# Determine if we're in shared view mode
if shared_data_param:
    # Decompress data from URL
    shared_data = decompress_data(shared_data_param)
    
    if shared_data:
        stats = shared_data.get("stats")
        ai = shared_data.get("ai")
        story_copy = shared_data.get("story_copy")
        
        st.session_state.is_shared_view = True
        uploaded = None
        
        # Show title for shared view
        st.title("WhatsApp Wrapped")
        st.markdown(
            f"<div style='text-align:center; opacity:0.5; font-size:12px; margin-bottom:32px;'>"
            f"🔗 Viewing shared Wrapped</div>",
            unsafe_allow_html=True
        )
    else:
        st.session_state.is_shared_view = False
        uploaded = None
else:
    st.session_state.is_shared_view = False
    
    st.title("WhatsApp Wrapped")

    # Only show upload UI if no file is uploaded yet and not viewing shared link
    if 'uploaded' not in st.session_state or st.session_state.uploaded is None:
        st.caption("Drop your exported chat text file to unlock the story.")
        uploaded = st.file_uploader("Upload WhatsApp chat (.txt)", type="txt", key="file_uploader")
        if uploaded:
            st.session_state.uploaded = uploaded
            st.rerun()
    else:
        uploaded = st.session_state.uploaded
        # Subtle indicator that file is loaded - doesn't compete with hero
        st.markdown(
            f"<div style='text-align:center; opacity:0.5; font-size:12px; margin-bottom:32px;'>"
            f"📊 Analyzing {uploaded.name}</div>",
            unsafe_allow_html=True
        )

# Helper to detect OpenAI key from env or st.secrets
def _has_api_key():
    if os.getenv("GROQ_API_KEY"):
        return True
    try:
        return bool(getattr(st, "secrets", {}).get("GROQ_API_KEY"))
    except Exception:
        return False


# Helper to generate playful Story tab copy using LLM
@st.cache_data(show_spinner=False)
def generate_story_copy(stats_dict):
    """Use LLM to generate Spotify Wrapped-style copy for Story tab sections"""
    try:
        client = _get_client()
        
        stats_context = f"""
        Based on this WhatsApp chat analysis, generate Spotify Wrapped-style playful copy:
        - Total messages: {stats_dict.get('total_messages', 0)}
        - Chat span: {stats_dict.get('chat_span_label', 'unknown')}
        - Most active day: {stats_dict.get('most_active_day', 'unknown')}
        - Most active hour: {stats_dict.get('most_active_hour', 'unknown')}
        - Top participant: {next(iter(stats_dict.get('messages_per_user', {}).items()), ('Unknown', 0))[0]}
        - Messages per day: {stats_dict.get('messages_per_day', 0):.1f}
        - Avg message length: {stats_dict.get('avg_length', 0)} chars
        - Emoji usage: {stats_dict.get('emoji_per_100', 0):.1f} per 100 messages
        - Links: {stats_dict.get('links_count', 0)} total
        - Peak month: {stats_dict.get('peak_month', 'unknown')}
        
        Generate ONLY the following as valid JSON (no explanation):
        {{
            "hero_identity": "A 2-3 word identity label (e.g., 'Late-Night Planners')",
            "hero_subtext": "One sentence that feels raw and emotional",
            "rhythm_description": "A playful description of when the chat comes alive",
            "energy_intro": "A short one-liner motivating the next section",
            "sound_intro": "The real tea on their texting style",
            "awards_intro": "Recognition of earned badges",
            "closing_message": "A reflective, emotional wrap-up about consistency"
        }}
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": stats_context}],
            temperature=0.7,
            max_tokens=500,
        )
        
        import json
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            copy_dict = json.loads(response_text)
            return copy_dict
        except json.JSONDecodeError:
            # If JSON fails, look for JSON block in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                copy_dict = json.loads(json_match.group())
                return copy_dict
            else:
                return None
    except Exception as e:
        if is_rate_limit_error(e):
            st.session_state.ai_rate_limited = True
        return None


if uploaded or st.session_state.is_shared_view:
    # Process uploaded file only if not in shared view
    if uploaded and not st.session_state.is_shared_view:
        raw_lines = uploaded.read().decode("utf-8", errors="ignore").splitlines()
        df = parse_chat(raw_lines)

        if df.empty:
            st.error("No messages were parsed. Please verify the export format.")
            st.stop()

        stats = compute_stats(df)
        # Reset rate-limit flag for a fresh run
        st.session_state.ai_rate_limited = False

        # Helper formatting
        def hour_label(h):
            if h is None:
                return "-"
            return datetime.strptime(str(int(h)), "%H").strftime("%I %p").lstrip("0")

        # Prepare messages for LLM (no deterministic recompute)
        records = [
            MessageRecord(
                sender=row["sender"],
                text=str(row["message"]),
                timestamp=str(row["datetime"]),
            )
            for _, row in df.iterrows()
        ]
        chunks = chunk_messages(records)

        @st.cache_resource
        def cached_client():
            return _get_client()

        @st.cache_data(show_spinner=False)
        def cached_insights(stats_snapshot, chunk_snapshot):
            client = cached_client()
            summary = conversation_summary(client, stats_snapshot, chunk_snapshot)
            topics = topic_intelligence(client, stats_snapshot, chunk_snapshot)
            tone = emotional_tone(client, stats_snapshot, chunk_snapshot)
            evolution = chat_evolution(client, stats_snapshot, chunk_snapshot)
            beats = wrapped_beats(client, stats_snapshot, chunk_snapshot)
            return {
                "summary": summary,
                "topics": topics,
                "tone": tone,
                "evolution": evolution,
                "beats": beats,
            }

        # Load AI insights if available
        chunk_snapshot = tuple(chunks) if chunks else None
        ai = None
        if _has_api_key() and chunk_snapshot:
            with st.spinner("✨ Analyzing your chat vibe..."):
                try:
                    ai = cached_insights(stats, chunk_snapshot)
                except Exception as e:
                    if is_rate_limit_error(e):
                        st.session_state.ai_rate_limited = True
                        # st.info("AI is cooling off for a few minutes (rate limit). Your stats still load fine.")

        # Generate AI copy for Story tab
        story_copy = generate_story_copy(stats) if _has_api_key() else None
        
        # Generate compressed data for sharing (no need for ID - data is in URL)
        if 'wrapped_data' not in st.session_state:
            wrapped_data = {
                "stats": stats,
                "ai": ai,
                "story_copy": story_copy,
            }
            st.session_state.wrapped_data = wrapped_data
    
    # If in shared view, stats/ai/story_copy are already loaded from session state
    # Define hour_label helper for both paths
    # if st.session_state.get("ai_rate_limited"):
        # st.info("AI is cooling off for a few minutes due to rate limits. Stats are still available; try AI sections again shortly.")

    def hour_label(h):
        if h is None:
            return "-"
        return datetime.strptime(str(int(h)), "%H").strftime("%I %p").lstrip("0")
    
    # HERO — bold opening with rhythm and emotion
    chat_span = stats.get("chat_span_label", "your timeline")
    hero = (
        f"Your chat traded {stats['total_messages']:,} messages over {chat_span}, "
        f"coming alive most on {stats.get('most_active_day', 'someday')} around {hour_label(stats.get('most_active_hour'))}."
    )
    
    # Use AI-generated identity or fallback
    if story_copy and story_copy.get('hero_identity'):
        identity_label = story_copy['hero_identity']
        hero_subtext = story_copy.get('hero_subtext', 'Raw, unfiltered, and totally you.')
    else:
        identity_label = "Late-Night Planners" if stats.get('most_active_hour', 0) >= 20 else "Morning Vibes" if stats.get('most_active_hour', 0) < 9 else "All-Day Squad"
        hero_subtext = 'Raw, unfiltered, and totally you.'
    
    st.markdown(
        f"""
        <div class='hero-card'>
            <div class='hero-title'>Your 2025 in Messages</div>
            <div class='hero-number' style='font-size:28px; margin-bottom:12px;'>{identity_label}</div>
            <div class='hero-number' style='font-size:30px;'>{hero}</div>
            <div class='micro'>{hero_subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Chat Personality — when the group comes alive
    st.markdown("<div class='section-title'>Your Rhythm</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>How this chat beats</div>", unsafe_allow_html=True)
    persona_col1, persona_col2 = st.columns(2)
    persona_col1.markdown(
        f"""
        <div class='mini-card'>
            <div style='font-weight:700;'>{stats.get('most_active_day','-')}</div>
            <div class='micro'>Peak night</div>
            <div style='font-size:18px; font-weight:800; margin-top:8px; opacity:0.9;'>When you come alive.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    persona_col2.markdown(
        f"""
        <div class='mini-card'>
            <div style='font-weight:700;'>{hour_label(stats.get('most_active_hour'))}</div>
            <div class='micro'>Power hour</div>
            <div style='font-size:18px; font-weight:800; margin-top:8px; opacity:0.9;'>The vibe peaks.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("<div class='commentary'>✨ The real question: who's actually sleeping?</div>", unsafe_allow_html=True)

    # Voices — ranked participant roles
    st.markdown("<div class='section-title'>🎭 The Cast</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Every chat has a lineup. This is you.</div>", unsafe_allow_html=True)
    mpu = pd.DataFrame(
        [
            {"sender": k, "count": v, "pct": (v / stats["total_messages"]) * 100}
            for k, v in stats["messages_per_user"].items()
        ]
    ).sort_values("count", ascending=False)

    roles = ["The Lead 🎤", "The Hype 🔥", "The Steady 🛡️", "The Rare Gem 💎"]
    cols = st.columns(min(4, len(mpu)) or 1)
    for (idx, row), col in zip(mpu.head(4).iterrows(), cols):
        role = roles[idx] if idx < len(roles) else "The Real One ⭐"
        col.markdown(
            f"""
            <div class='mini-card'>
                <div class='micro' style='font-size:12px; font-weight:700;'>{role}</div>
                <div style='font-weight:800; font-size:20px; margin-top:6px;'>{row['sender']}</div>
                <div class='micro' style='margin-top:8px;'>{row['pct']:.1f}% of the convo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(f"<div class='commentary'>💬 This is how {identity_label} communicate.</div>", unsafe_allow_html=True)

    # Message Style — personality traits, not metrics
    st.markdown("<div class='section-title'>💬 Your Sound</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>The real tea on your texting style</div>", unsafe_allow_html=True)

    style_cols = st.columns(3)
    tone = "direct" if stats["avg_length"] < 30 else "storytelling"
    emoji_vibe = "obsessed" if stats["emoji_per_100"] > 5 else "cool"
    link_vibe = "rarely" if stats["links_count"] == 0 else "always" if stats["links_count"] > 10 else "sometimes"

    style_messages = [
        f"Texts are {tone}—no fluff.",
        f"Emojis? You're {emoji_vibe}.",
        f"Links: {link_vibe}.",
    ]
    for msg, col in zip(style_messages, style_cols):
        col.markdown(
            f"""
            <div class='mini-card'>
                <div style='font-weight:700; font-size:18px;'>{msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    awards_intro_text = story_copy.get('awards_intro', "You've earned some serious recognition...") if story_copy else "You've earned some serious recognition..."
    st.markdown(f"<div class='commentary'>🏆 {awards_intro_text}</div>", unsafe_allow_html=True)

    # Fun Awards — celebrate the earned badges
    st.markdown("<div class='section-title'>🏅 The Badges</div>", unsafe_allow_html=True)
    awards = stats.get("awards", {})
    award_labels = {
        "night_owl": ("Night Owl 🦉", "Living for midnight."),
        "early_bird": ("Early Bird ☀️", "First one awake."),
        "emoji_king": ("Emoji Master 👑", "Speaks in reactions."),
        "marathon_chatter": ("Streaker 🔥", "Never misses a beat."),
    }
    shown = list(awards.items())[:3]
    st.markdown(
        "<div class='award-row'>"
        + "".join(
            [
                f"<div class='award-card'><div style='font-weight:800'>{award_labels.get(k, (k, ''))[0]}</div>"
                f"<div style='margin-top:6px; font-weight:700'>{v}</div>"
                f"<div style='margin-top:4px; opacity:0.8; font-size:13px;'>{award_labels.get(k, (k, ''))[1]}</div>"
                f"</div>"
                for k, v in shown
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    energy_intro_text = story_copy.get('energy_intro', 'The real MVPs of your chat:') if story_copy else 'The real MVPs of your chat:'
    st.markdown(f"<div class='commentary'>⚡ {energy_intro_text}</div>", unsafe_allow_html=True)

    # Vibe Check — the movers and shakers
    st.markdown("<div class='section-title'>⚙️ The Energy</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Who keeps the momentum going</div>", unsafe_allow_html=True)

    starters = pd.DataFrame(
        [
            {"sender": k, "starters": v}
            for k, v in stats.get("starters_per_user", {}).items()
        ]
    ).sort_values("starters", ascending=False)

    replies = pd.DataFrame(
        [
            {"sender": k, "avg_reply_sec": v}
            for k, v in stats.get("avg_reply_seconds", {}).items()
        ]
    ).sort_values("avg_reply_sec")
    replies["avg_reply_min"] = (replies["avg_reply_sec"] / 60).round(1) if not replies.empty else []

    vibe_cols = st.columns(2)

    if not starters.empty:
        top_starter = starters.iloc[0]
        vibe_cols[0].markdown(
            f"""
            <div class='mini-card'>
                <div class='micro'>The Initiator</div>
                <div style='font-weight:800; font-size:20px;'>{top_starter['sender']}</div>
                <div style='margin-top:6px; font-size:14px;'>Keeps things going. {top_starter['starters']} times.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        vibe_cols[0].info("No starter data yet.")

    if not replies.empty:
        fastest = replies.iloc[0]
        vibe_cols[1].markdown(
            f"""
            <div class='mini-card'>
                <div class='micro'>The Quick Reply</div>
                <div style='font-weight:800; font-size:20px;'>{fastest['sender']}</div>
                <div style='margin-top:6px; font-size:14px;'>In ~{fastest['avg_reply_min']:.1f} min. Always there.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        vibe_cols[1].info("No reply-time data yet.")
    st.markdown(" ")
    # Closing summary — reflective and emotional
    closing_msg = story_copy.get('closing_message', f"You showed up ~{stats['messages_per_day']:.1f} times today. That consistency? That's love.") if story_copy else f"You showed up ~{stats['messages_per_day']:.1f} times today. That consistency? That's love."
    st.markdown(
        f"""
        <div class='mini-card' style='background: {theme['hero_bg']}; color: #ffffff; text-align:center; border-left: none;'>
            <div style='font-size:16px; font-weight:700; line-height:1.6;'>{closing_msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ========================================
    # AI INSIGHTS SECTIONS (moved from tab 2)
    # ========================================
    
    if ai:

        # ========================================
        # EMOTIONAL VIBE (minimal)
        # ========================================
        
        # Extract main emotion word
        tone_words = ai['tone'].lower().split()
        emotion_keywords = ['playful', 'lively', 'excited', 'calm', 'supportive', 'chaotic', 'intense', 'casual', 'warm', 'energetic']
        main_emotion = next((word.capitalize() for word in tone_words if word in emotion_keywords), "Vibrant")
        
        # Generate context-aware one-liner from tone analysis
        tone_lower = ai['tone'].lower()
        if 'calm' in tone_lower and 'loud' in tone_lower:
            one_liner = "Calm days. Loud nights."
        elif 'supportive' in tone_lower or 'care' in tone_lower:
            one_liner = "Always there when it matters."
        elif 'playful' in tone_lower or 'joke' in tone_lower:
            one_liner = "Never taking it too seriously."
        elif 'intense' in tone_lower or 'passionate' in tone_lower:
            one_liner = "All in, all the time."
        else:
            one_liner = "Energy you can feel."
        
        st.markdown("<div class='commentary'>The feeling</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='mini-card' style='text-align:center; padding:24px;'>
                <div style='font-size:36px; font-weight:800; line-height:1.1; margin-bottom:8px; color:{theme['text']};'>
                    {main_emotion}
                </div>
                <div style='font-size:13px; opacity:0.75; line-height:1.4;'>
                    {one_liner}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ========================================
        # 3. CHAT EVOLUTION (before vs after comparison)
        # ========================================
        st.markdown("<div class='commentary'>How you evolved</div>", unsafe_allow_html=True)
        
        # Parse evolution with better extraction logic
        evolution_lower = ai['evolution'].lower()
        if 'start' in evolution_lower or 'began' in evolution_lower or 'early' in evolution_lower:
            # Extract before/after from AI text
            evo_sentences = [s.strip() for s in ai['evolution'].split('.') if s.strip()]
            before_text = "Short. Functional." if 'short' in evolution_lower or 'brief' in evolution_lower else "Casual exchanges."
            after_text = "Longer. Expressive." if 'long' in evolution_lower or 'detailed' in evolution_lower else "Rich conversations."
        else:
            before_text = "Getting started."
            after_text = "Going deep."
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class='mini-card' style='text-align:center; padding:20px;'>
                    <div style='font-size:28px; margin-bottom:8px;'>🌱</div>
                    <div class='micro'>EARLY DAYS</div>
                    <div style='margin-top:10px; font-weight:700; font-size:16px; line-height:1.4;'>{before_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class='mini-card' style='text-align:center; padding:20px;'>
                    <div style='font-size:28px; margin-bottom:8px;'>🚀</div>
                    <div class='micro'>NOW</div>
                    <div style='margin-top:10px; font-weight:700; font-size:16px; line-height:1.4;'>{after_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br><br>", unsafe_allow_html=True)

    # ========================================
    # SHAREABLE FINAL CARD (always shown, outside AI block)
    # ========================================
    st.markdown("<div class='section-title'>📸 Share Your Vibe</div>", unsafe_allow_html=True)
    
    # Use canonical identity from hero section
    share_statement = f"Your group vibe: {identity_label}"
    
    st.markdown(
        f"""
        <div style='padding:36px 24px; border-radius:16px; background:{theme['share_bg']}; 
             color:#ffffff; text-align:center; box-shadow: 0 8px 24px rgba(0,0,0,0.3);'>
            <div style='font-size:26px; font-weight:800; line-height:1.3; margin-bottom:12px;'>
                {share_statement}
            </div>
            <div style='font-size:13px; opacity:0.9; margin-top:16px; letter-spacing:0.5px; font-weight:600;'>
                WhatsApp Wrapped • AI-powered
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
        
    # WhatsApp share button
    import urllib.parse
    
    # Create shareable message
    share_text = f"""🎉 My WhatsApp Wrapped is here!

{share_statement}

📊 {stats['total_messages']:,} messages over {stats.get('chat_span_label', 'the year')}
⚡ Most active: {stats.get('most_active_day', 'weekends')} at {hour_label(stats.get('most_active_hour'))}

#WhatsAppWrapped"""
    
    # Generate shareable link with compressed data
    current_url = "https://kb-whatsapp-wrapped.streamlit.app"  # Update with your actual deployment URL
    
    # Compress and encode the wrapped data for the URL
    compressed_data = compress_data(st.session_state.wrapped_data)
    long_link = f"{current_url}?d={compressed_data}"
    wrapped_link = shorten_url(long_link)
    
    # URL encode the message with link
    share_text_with_link = share_text + f"\n\n🔗 View the full experience: {wrapped_link}"
    encoded_text = urllib.parse.quote(share_text_with_link)
    whatsapp_url = f"https://wa.me/?text={encoded_text}"
    
    # Share buttons with WhatsApp styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div style='text-align:center;'>
                <a href="{whatsapp_url}" target="_blank" style='text-decoration:none;'>
                    <div style='display:inline-block; padding:14px 32px; background:#25D366; color:white; 
                         border-radius:30px; font-weight:700; font-size:15px; box-shadow: 0 4px 12px rgba(37, 211, 102, 0.4);
                         transition: all 200ms ease; cursor:pointer;'>
                        <span style='font-size:18px; margin-right:8px;'>💬</span>
                        Share on WhatsApp
                    </div>
            </div>
            
            <style>
            a:hover div {{
                background: #20BA5A !important;
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(37, 211, 102, 0.5);
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        # Copy link button with JavaScript
        st.markdown(
            f"""
            <div style='text-align:center;'>
                <div id='copy-btn' onclick='copyToClipboard()' style='display:inline-block; padding:14px 32px; background:#1F2C34; color:#25D366; 
                     border: 2px solid #25D366; border-radius:30px; font-weight:700; font-size:15px; 
                     box-shadow: 0 4px 12px rgba(37, 211, 102, 0.2);
                     transition: all 200ms ease; cursor:pointer;'>
                    <span style='font-size:18px; margin-right:8px;'>🔗</span>
                    <span id='copy-text'>Copy Link</span>
                </div>
            </div>
            
            <script>
            function copyToClipboard() {{
                const link = "{wrapped_link}";
                navigator.clipboard.writeText(link).then(function() {{
                    const btn = document.getElementById('copy-btn');
                    const text = document.getElementById('copy-text');
                    text.textContent = 'Copied!';
                    btn.style.background = '#25D366';
                    btn.style.color = 'white';
                    btn.style.borderColor = '#25D366';
                    setTimeout(function() {{
                        text.textContent = 'Copy Link';
                        btn.style.background = '#1F2C34';
                        btn.style.color = '#25D366';
                        btn.style.borderColor = '#25D366';
                    }}, 2000);
                }});
            }}
            </script>
            
            <style>
            #copy-btn:hover {{
                background: #25D366 !important;
                color: white !important;
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(37, 211, 102, 0.5);
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    
    # ========================================
    # 5. INTERACTIVE Q&A (bottom of flow)
    # ========================================
    if ai:
        st.markdown("<div class='section-title'>🤔 Ask the Chat</div>", unsafe_allow_html=True)
        question = st.text_input("Ask anything about this chat (AI will answer from context)")
        if question:
            with st.spinner("🧠 Searching your chat..."):
                answer = qa_answer(cached_client(), stats, chunk_snapshot, question)
            st.markdown(
                f"""
                <div class='mini-card' style='margin-top:12px;'>
                    <div class='micro'>AI-GENERATED ANSWER</div>
                    <div style='margin-top:8px; line-height:1.6;'>{answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        if not _has_api_key():
            st.info("💡 Add GROQ_API_KEY to unlock AI-powered insights: Who You Really Are, The Feeling, Evolution, and more.")
        else:
            st.info("Need chat messages to generate AI insights.")

