"""
AinSeba - Conversation Memory
Manages chat history for multi-turn legal conversations.
Implements a sliding window buffer (last k exchanges).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single conversation message."""
    role: str          # "user" or "assistant"
    content: str
    sources: list = field(default_factory=list)  # Source citations (assistant only)

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.sources:
            d["sources"] = self.sources
        return d


class ConversationMemory:
    """
    Sliding-window conversation memory for the RAG chain.
    
    Keeps the last `k` user-assistant exchange pairs.
    This provides enough context for follow-up questions
    ("What section covers that?") without blowing up the token budget.
    
    Features:
    - Sliding window (configurable k)
    - Separate session tracking via session_id
    - Export/import for persistence
    - Token-aware truncation
    """

    def __init__(self, k: int = 5):
        """
        Args:
            k: Number of exchange pairs to keep (user + assistant = 1 pair).
               k=5 means last 10 messages (5 user + 5 assistant).
        """
        self.k = k
        self._sessions: dict[str, list[Message]] = {}
        self._default_session = "default"

    def add_user_message(
        self, content: str, session_id: Optional[str] = None
    ) -> None:
        """Add a user message to the conversation."""
        session = session_id or self._default_session
        self._ensure_session(session)
        self._sessions[session].append(Message(role="user", content=content))
        self._trim(session)

    def add_assistant_message(
        self,
        content: str,
        sources: list = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Add an assistant response to the conversation."""
        session = session_id or self._default_session
        self._ensure_session(session)
        self._sessions[session].append(
            Message(role="assistant", content=content, sources=sources or [])
        )
        self._trim(session)

    def get_history(
        self, session_id: Optional[str] = None
    ) -> list[dict]:
        """
        Get conversation history as a list of message dicts.

        Returns:
            List of {"role": str, "content": str} dicts.
        """
        session = session_id or self._default_session
        if session not in self._sessions:
            return []
        return [msg.to_dict() for msg in self._sessions[session]]

    def get_history_messages(
        self, session_id: Optional[str] = None
    ) -> list[Message]:
        """Get conversation history as Message objects."""
        session = session_id or self._default_session
        return list(self._sessions.get(session, []))

    def get_last_exchange(
        self, session_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Get the last user question and assistant answer.

        Returns:
            (last_user_message, last_assistant_message) or (None, None).
        """
        session = session_id or self._default_session
        messages = self._sessions.get(session, [])

        last_user = None
        last_assistant = None

        for msg in reversed(messages):
            if msg.role == "assistant" and last_assistant is None:
                last_assistant = msg.content
            elif msg.role == "user" and last_user is None:
                last_user = msg.content
            if last_user and last_assistant:
                break

        return last_user, last_assistant

    def clear(self, session_id: Optional[str] = None) -> None:
        """Clear conversation history for a session."""
        session = session_id or self._default_session
        self._sessions[session] = []
        logger.debug(f"Cleared memory for session '{session}'")

    def clear_all(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()

    @property
    def message_count(self) -> int:
        """Total messages in default session."""
        return len(self._sessions.get(self._default_session, []))

    @property
    def session_ids(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def export_session(self, session_id: Optional[str] = None) -> list[dict]:
        """Export session history for persistence."""
        return self.get_history(session_id)

    def import_session(
        self, messages: list[dict], session_id: Optional[str] = None
    ) -> None:
        """Import session history from saved data."""
        session = session_id or self._default_session
        self._sessions[session] = [
            Message(
                role=m["role"],
                content=m["content"],
                sources=m.get("sources", []),
            )
            for m in messages
        ]
        self._trim(session)

    def _ensure_session(self, session_id: str) -> None:
        """Create session if it doesn't exist."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

    def _trim(self, session_id: str) -> None:
        """Trim to keep only the last k exchange pairs (2k messages)."""
        messages = self._sessions[session_id]
        max_messages = self.k * 2  # Each exchange = user + assistant
        if len(messages) > max_messages:
            self._sessions[session_id] = messages[-max_messages:]
