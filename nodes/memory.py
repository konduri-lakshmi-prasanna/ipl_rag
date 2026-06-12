"""
memory.py  —  Conversational Memory Manager
Handles multi-turn conversation context so follow-up questions work correctly.
Stores chat history in-memory (for the session) with optional cache support.
"""

import time
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class Turn:
    role: str       # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CachedAnswer:
    answer: str
    query_type: str
    sources: List[str]
    conflict_detected: bool
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# Conversation Memory Store
# ─────────────────────────────────────────────

class ConversationMemory:
    """
    Per-session conversation history.
    Keeps the last N turns to stay within context window limits.
    """

    def __init__(self, max_turns: int = 10):
        self._sessions: Dict[str, List[Turn]] = {}
        self.max_turns = max_turns

    def add_turn(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(Turn(role=role, content=content))
        # Keep only last max_turns turns (each turn = 1 message)
        if len(self._sessions[session_id]) > self.max_turns * 2:
            self._sessions[session_id] = self._sessions[session_id][-(self.max_turns * 2):]

    def get_history(self, session_id: str) -> List[Dict]:
        """Returns history as a list of {role, content} dicts."""
        turns = self._sessions.get(session_id, [])
        return [{"role": t.role, "content": t.content} for t in turns]

    def get_last_n(self, session_id: str, n: int = 4) -> List[Dict]:
        history = self.get_history(session_id)
        return history[-n:]

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)

    def all_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# ─────────────────────────────────────────────
# Answer Cache
# ─────────────────────────────────────────────

class AnswerCache:
    """
    Simple in-memory cache that maps query fingerprints to answers.
    Avoids re-running the full graph for repeated identical questions.
    Cache entries expire after `ttl_seconds` (default 10 minutes).
    """

    def __init__(self, ttl_seconds: int = 600):
        self._cache: Dict[str, CachedAnswer] = {}
        self.ttl = ttl_seconds

    def _key(self, query: str) -> str:
        # Normalize: lowercase, strip whitespace, then hash
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[CachedAnswer]:
        key = self._key(query)
        entry = self._cache.get(key)
        if entry is None:
            return None
        # Check TTL
        if time.time() - entry.timestamp > self.ttl:
            del self._cache[key]
            return None
        return entry

    def set(self, query: str, answer: str, query_type: str,
            sources: List[str], conflict_detected: bool):
        key = self._key(query)
        self._cache[key] = CachedAnswer(
            answer=answer,
            query_type=query_type,
            sources=sources,
            conflict_detected=conflict_detected,
        )

    def invalidate(self, query: str):
        self._cache.pop(self._key(query), None)

    def clear_all(self):
        self._cache.clear()

    def stats(self) -> Dict:
        now = time.time()
        active = sum(1 for v in self._cache.values() if now - v.timestamp <= self.ttl)
        return {"total_entries": len(self._cache), "active_entries": active, "ttl_seconds": self.ttl}


# ─────────────────────────────────────────────
# Singleton instances (shared across requests)
# ─────────────────────────────────────────────

conversation_memory = ConversationMemory(max_turns=10)
answer_cache = AnswerCache(ttl_seconds=600)