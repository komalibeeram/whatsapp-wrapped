import re
from datetime import timedelta

import pandas as pd


EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]"
)


def _longest_consecutive_run(dates):
    longest = 0
    current = 0
    prev = None
    for d in sorted(dates):
        if prev and (d - prev).days == 1:
            current += 1
        else:
            current = 1
        prev = d
        longest = max(longest, current)
    return longest


def compute_stats(df):
    df = df.copy()

    # Drop WhatsApp system/service messages from analytics
    system_patterns = [
        r"Messages and calls are end-to-end encrypted",
        r"created group",
        r"added .*",  # additions
        r"left$",
        r"removed .*",
        r"changed (this group's|the group) (icon|subject|description|name)",
        r"missed (voice|video) call",
        r"ended the call",
        r"waiting for this message",
        r"you're now an admin",
    ]
    system_regex = "|".join(system_patterns)
    df = df[~df["message"].str.contains(system_regex, case=False, regex=True, na=False)]

    stats = {}
    stats["total_messages"] = len(df)
    stats["messages_per_user"] = df["sender"].value_counts().to_dict()

    if df.empty:
        # Keep the shape stable for the UI even when no data parses.
        stats.update(
            {
                "most_active_day": None,
                "most_active_hour": None,
                "chat_span_days": 0,
                "chat_span_label": "0 days",
                "daily_counts": pd.DataFrame(columns=["date", "count"]),
                "heatmap": pd.DataFrame(columns=["day", "hour", "count"]),
                "avg_length": 0,
                "emoji_per_100": 0,
                "links_count": 0,
                "starters_per_user": {},
                "avg_reply_seconds": {},
                "awards": {},
                "peak_month": None,
                "messages_per_day": 0,
            }
        )
        return stats

    # Precompute helpful series
    dt = df["datetime"]
    day_mode = dt.dt.day_name().mode()
    hour_mode = dt.dt.hour.mode()

    stats["most_active_day"] = day_mode.iloc[0] if not day_mode.empty else None
    stats["most_active_hour"] = int(hour_mode.iloc[0]) if not hour_mode.empty else None

    span_days = (dt.max() - dt.min()).days + 1
    stats["chat_span_days"] = span_days
    span_months = max(1, round(span_days / 30))
    stats["chat_span_label"] = f"{span_months} month" + ("s" if span_months != 1 else "")

    daily = (
        df.groupby(dt.dt.date)
        .size()
        .reset_index(name="count")
        .rename(columns={"datetime": "date"})
    )
    stats["daily_counts"] = daily

    # Heatmap data (day of week 0=Mon, hour 0-23)
    # Name the groupby keys to avoid duplicate column insertion on reset_index
    heatmap = (
        df.groupby([dt.dt.dayofweek.rename("day"), dt.dt.hour.rename("hour")])
        .size()
        .reset_index(name="count")
    )
    stats["heatmap"] = heatmap

    # Message style
    lengths = df["message"].astype(str).str.len()
    stats["avg_length"] = float(lengths.mean()) if not lengths.empty else 0

    emoji_counts = df["message"].astype(str).apply(lambda m: len(EMOJI_PATTERN.findall(m)))
    total_emojis = int(emoji_counts.sum())
    stats["emoji_per_100"] = round((total_emojis / len(df)) * 100, 2) if len(df) else 0

    link_mask = df["message"].astype(str).str.contains(r"https?://", regex=True)
    stats["links_count"] = int(link_mask.sum())

    # Conversation starters: gap > 2 hours from previous message
    df = df.sort_values("datetime")
    gaps = df["datetime"].diff().fillna(pd.Timedelta(days=999))
    starters = gaps > timedelta(hours=2)
    starters_per_user = df.loc[starters, "sender"].value_counts().to_dict()
    stats["starters_per_user"] = starters_per_user

    # Reply time per user (only when sender changes)
    prev_sender = df["sender"].shift(1)
    reply_deltas = df.loc[df["sender"] != prev_sender, "datetime"].diff().dt.total_seconds()
    reply_senders = df.loc[df["sender"] != prev_sender, "sender"]
    avg_reply = reply_deltas.groupby(reply_senders).mean().dropna()
    stats["avg_reply_seconds"] = avg_reply.to_dict()

    # Awards
    awards = {}
    # Night owl: 22-4
    night_mask = dt.dt.hour.isin(list(range(22, 24)) + list(range(0, 5)))
    night_user = df.loc[night_mask, "sender"].value_counts()
    if not night_user.empty:
        awards["night_owl"] = night_user.idxmax()

    # Early bird: 5-9
    early_mask = dt.dt.hour.isin(range(5, 10))
    early_user = df.loc[early_mask, "sender"].value_counts()
    if not early_user.empty:
        awards["early_bird"] = early_user.idxmax()

    # Emoji king: highest emoji rate per message
    user_emoji = df.groupby("sender")["message"].apply(
        lambda msgs: sum(len(EMOJI_PATTERN.findall(str(m))) for m in msgs)
    )
    user_counts = df["sender"].value_counts()
    emoji_rate = (user_emoji / user_counts).replace([pd.NA, pd.NaT], 0)
    if not emoji_rate.empty:
        awards["emoji_king"] = emoji_rate.idxmax()

    # Marathon chatter: longest consecutive-day streak per user
    streaks = {}
    for sender, group in df.groupby("sender"):
        dates = set(group["datetime"].dt.date)
        streaks[sender] = _longest_consecutive_run(dates)
    if streaks:
        awards["marathon_chatter"] = max(streaks, key=streaks.get)
    stats["awards"] = awards

    # Peak month/year and daily average
    daily_idx = daily.set_index("date")
    if not daily_idx.empty:
        peak_day = daily_idx["count"].idxmax()
        stats["peak_month"] = peak_day.strftime("%B %Y")
    else:
        stats["peak_month"] = None
    stats["messages_per_day"] = round(len(df) / span_days, 2) if span_days else 0

    return stats