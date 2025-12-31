import re
import pandas as pd

# Accept WhatsApp export lines like: [9/16/25, 9:55:36 AM] Name: Message
# Handles optional seconds, various spaces (including narrow no-break space) before AM/PM, and 12h or 24h times.
LINE_PATTERN = re.compile(
    r"^\s*\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]\s+([^:]+):\s+(.*)$"
)


def parse_chat(lines):
    rows = []
    current = None

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        line = line.lstrip("\ufeff")  # drop BOM if present
        match = LINE_PATTERN.match(line)

        if match:
            if current:
                rows.append(current)
            date, time, sender, message = match.groups()
            current = [date, time.strip(), sender.strip(), message]
        else:
            if current:
                current[3] += "\n" + line  # append multiline messages

    if current:
        rows.append(current)

    df = pd.DataFrame(rows, columns=["date", "time", "sender", "message"])
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    return df