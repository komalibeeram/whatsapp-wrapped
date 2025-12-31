"""Prompt templates for AI Insights.

These prompts are designed to:
- Consume only provided context (raw chat snippets + precomputed stats).
- Avoid recomputing deterministic metrics.
- Steer the LLM to narrate and infer, not calculate.
"""

# Conversation Summary
CONVERSATION_SUMMARY = """
You are creating an upbeat, human, Wrapped-style summary of a WhatsApp chat.
Use only the provided messages and stats for grounding.
Do NOT invent numbers or counts that are not explicitly given.
Return a short paragraph (3-5 sentences) covering:
- What the chat is mostly about.
- How the tone feels overall.
- Any noticeable shifts over time.
"""

# Topic Intelligence
TOPIC_INTELLIGENCE = """
Identify 3-5 recurring themes in the chat.
Use high-level topics, not low-level keywords.
Avoid numbers or counts; describe themes briefly (one sentence each).
"""

# Emotional Tone Analysis
EMOTIONAL_TONE = """
Describe the overall emotional tone and notable shifts.
Mention periods that felt stressed, excited, calm, or celebratory based only on provided context.
Do not invent dates or counts; keep it narrative.
"""

# Chat Evolution Story
CHAT_EVOLUTION = """
Explain how communication style evolved (e.g., short to long messages, formal to casual).
Use the supplied stats and message snippets as evidence.
Do not compute new metrics; narrate changes you infer.
Keep it to 3-4 sentences.
"""

# Wrapped-style Storytelling
WRAPPED_BEATS = """
Craft 4-6 short, fun, human-readable beats using the provided stats.
Each beat should be one sentence, playful, and avoid raw numbers unless supplied.
Do not generate new counts—only reference stats already given.
"""

# Natural Language Q&A
QA_SYSTEM = """
You answer user questions about the chat using ONLY the provided snippets and stats.
If the answer is not clearly supported, say you are not sure.
Do not fabricate counts or dates. Keep answers concise and friendly.
"""
