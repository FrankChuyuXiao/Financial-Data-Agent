from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
from pathlib import Path
from datetime import datetime


@dataclass
class ConversationMemory:
    """Simple file-backed conversation memory."""

    path: str = "outputs/memory.json"
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        if self.path.exists():
            try:
                self.messages = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.messages = []

    def save(self):
        self.path.write_text(
            json.dumps(self.messages, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def add(self, role: str, content: str, metadata: Dict[str, Any] | None = None):
        self.messages.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(timespec="seconds")
        })
        self.save()

    def recent(self, n: int = 8) -> List[Dict[str, Any]]:
        return self.messages[-n:]

    def as_text(self, n: int = 8) -> str:
        recent_messages = self.recent(n)
        lines = []
        for msg in recent_messages:
            lines.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(lines)