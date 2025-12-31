"""AI Insights layer for WhatsApp Wrapped.

- Consumes raw chat messages + precomputed stats ONLY.
- Does NOT recompute deterministic metrics.
- Provides narrative, topic, tone, evolution, beats, and Q&A insights via LLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

from groq import Groq

from llm import prompts


@dataclass
class MessageRecord:
    sender: str
    text: str
    timestamp: str  # stringified timestamp to avoid tz issues in prompts


def _get_client() -> Groq:
    """Create a Groq client; requires GROQ_API_KEY in env or st.secrets.

    Reads from environment first; if streamlit secrets are present, uses them.
    No keys are hardcoded to keep credentials safe.
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st

            api_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
        except Exception:
            api_key = ""

    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY for AI insights.")

    return Groq(api_key=api_key)


def chunk_messages(messages: Sequence[MessageRecord], max_chars: int = 3200) -> List[str]:
    """Chunk raw messages into bounded strings for context windows.

    This is a simple size-based chunker to stay deterministic and fast.
    """

    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for msg in messages:
        line = f"[{msg.timestamp}] {msg.sender}: {msg.text}"
        if size + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _stats_as_context(stats: Dict) -> str:
    """Flatten key stats into text without recomputing them."""

    lines = []
    lines.append(f"Total messages: {stats.get('total_messages')}")
    if stats.get("most_active_day"):
        lines.append(f"Most active day: {stats['most_active_day']}")
    if stats.get("most_active_hour") is not None:
        lines.append(f"Most active hour: {stats['most_active_hour']}")
    if stats.get("chat_span_label"):
        lines.append(f"Chat span: {stats['chat_span_label']}")
    if stats.get("messages_per_user"):
        top_sender = max(stats["messages_per_user"], key=stats["messages_per_user"].get)
        lines.append(f"Top participant: {top_sender}")
    if stats.get("peak_month"):
        lines.append(f"Peak activity month: {stats['peak_month']}")
    return "\n".join(lines)


def _build_prompt(system_prompt: str, stats: Dict, chunks: Sequence[str]) -> str:
    stats_blob = _stats_as_context(stats)
    recent = "\n\n".join(chunks[:2])  # keep prompt lean; first couple chunks
    return (
        f"SYSTEM:\n{system_prompt}\n\n"
        f"STATS (do not recalc):\n{stats_blob}\n\n"
        f"MESSAGES (snippets):\n{recent}\n\n"
        "Answer in a short, human, Wrapped-style paragraph or bullets as instructed."
    )


def _complete(client: Groq, prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=350,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"AI call failed: {str(e)}")


def conversation_summary(client: Groq, stats: Dict, chunks: Sequence[str]) -> str:
    prompt = _build_prompt(prompts.CONVERSATION_SUMMARY, stats, chunks)
    return _complete(client, prompt)


def topic_intelligence(client: Groq, stats: Dict, chunks: Sequence[str]) -> str:
    prompt = _build_prompt(prompts.TOPIC_INTELLIGENCE, stats, chunks)
    return _complete(client, prompt)


def emotional_tone(client: Groq, stats: Dict, chunks: Sequence[str]) -> str:
    prompt = _build_prompt(prompts.EMOTIONAL_TONE, stats, chunks)
    return _complete(client, prompt)


def chat_evolution(client: Groq, stats: Dict, chunks: Sequence[str]) -> str:
    prompt = _build_prompt(prompts.CHAT_EVOLUTION, stats, chunks)
    return _complete(client, prompt)


def wrapped_beats(client: Groq, stats: Dict, chunks: Sequence[str]) -> str:
    prompt = _build_prompt(prompts.WRAPPED_BEATS, stats, chunks)
    return _complete(client, prompt)


def qa_answer(client: Groq, stats: Dict, chunks: Sequence[str], question: str) -> str:
    # Lightweight retrieval: pick first chunk; in production, plug in embeddings.
    qa_prompt = (
        f"{prompts.QA_SYSTEM}\n\nQUESTION: {question}\n\n"
        f"STATS (read-only):\n{_stats_as_context(stats)}\n\n"
        f"MESSAGES (snippet):\n{chunks[0] if chunks else ''}"
    )
    return _complete(client, qa_prompt, model="llama-3.3-70b-versatile")


__all__ = [
    "MessageRecord",
    "chunk_messages",
    "conversation_summary",
    "topic_intelligence",
    "emotional_tone",
    "chat_evolution",
    "wrapped_beats",
    "qa_answer",
    "_get_client",
]
