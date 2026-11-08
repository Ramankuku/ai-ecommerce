import json
import time
from pathlib import Path

HISTORY_DIR = Path("data/history")
MAX_CHATS = 30


def _file(user_id):
    safe = "".join(c for c in str(user_id) if c.isalnum() or c in "-_")
    return HISTORY_DIR / f"{safe or 'guest'}.json"


def load(user_id):
    """All saved chats, newest first."""
    try:
        data = json.loads(_file(user_id).read_text(encoding="utf-8"))
        return [c for c in data if "messages" in c]   # ignore old-format entries
    except (OSError, ValueError):
        return []


def save(user_id, chat_id, messages):
    """Create or update one conversation and return the updated list."""
    if not messages:
        return load(user_id)

    title = next((m["content"] for m in messages if m["role"] == "user"), "New chat")
    chat = {
        "id": chat_id,
        "title": title[:40],
        "at": time.strftime("%d %b, %H:%M"),
        # image bytes cannot be stored as JSON
        "messages": [{k: v for k, v in m.items() if k != "image"} for m in messages],
    }
    chats = [chat] + [c for c in load(user_id) if c["id"] != chat_id]
    chats = chats[:MAX_CHATS]

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _file(user_id).write_text(
        json.dumps(chats, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return chats


def clear(user_id):
    _file(user_id).unlink(missing_ok=True)
    return []