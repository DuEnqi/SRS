#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STALE Benchmark - Core Data Types and Interfaces

STALE (State Tracking And Latent Evaluation) tests LLM Agent memory update
capability in implicit-conflict scenarios, where new observations invalidate
old memories without ever explicitly negating the old information.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str   # "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "ConversationTurn":
        return cls(role=d["role"], content=d["content"])


@dataclass
class Session:
    """A single conversation session (one of the 50 in a haystack)."""
    session_id: str
    timestamp: str
    turns: List[ConversationTurn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "turns": [t.to_dict() for t in self.turns],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Session":
        return cls(
            session_id=d["session_id"],
            timestamp=d["timestamp"],
            turns=[ConversationTurn.from_dict(t) for t in d.get("turns", [])],
        )


@dataclass
class STALEQuery:
    """A single probe query with its expected behaviour description."""
    question: str
    expected_behavior: str

    def to_dict(self) -> Dict[str, str]:
        return {"question": self.question, "expected_behavior": self.expected_behavior}

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "STALEQuery":
        return cls(question=d["question"], expected_behavior=d["expected_behavior"])


@dataclass
class STALEInstance:
    """
    A single STALE benchmark instance.

    Fields match the spec JSON schema exactly so instances can be loaded
    directly from a .jsonl dataset file.
    """
    uid: str
    conflict_type: str          # "I" or "II"
    attribute: str              # e.g. "location"
    m_old: str                  # Old memory claim
    m_new: str                  # New memory claim
    explanation: str
    time_gap: str               # e.g. "3 months"
    session_o: Session          # Session containing m_old
    session_n: Session          # Session containing m_new
    haystack_sessions: List[Session]   # All 50 sessions (includes session_o & session_n)
    queries: Dict[str, STALEQuery]    # Keys: "sr", "pr", "ipa"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "type": self.conflict_type,
            "attribute": self.attribute,
            "m_old": self.m_old,
            "m_new": self.m_new,
            "explanation": self.explanation,
            "time_gap": self.time_gap,
            "session_o": self.session_o.to_dict(),
            "session_n": self.session_n.to_dict(),
            "haystack": {"sessions": [s.to_dict() for s in self.haystack_sessions]},
            "queries": {k: v.to_dict() for k, v in self.queries.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "STALEInstance":
        haystack = d.get("haystack", {})
        sessions = [Session.from_dict(s) for s in haystack.get("sessions", [])]
        queries_raw = d.get("queries", {})
        queries = {k: STALEQuery.from_dict(v) for k, v in queries_raw.items()}
        return cls(
            uid=d["uid"],
            conflict_type=d.get("type", "I"),
            attribute=d.get("attribute", ""),
            m_old=d.get("m_old", ""),
            m_new=d.get("m_new", ""),
            explanation=d.get("explanation", ""),
            time_gap=d.get("time_gap", ""),
            session_o=Session.from_dict(d["session_o"]),
            session_n=Session.from_dict(d["session_n"]),
            haystack_sessions=sessions,
            queries=queries,
        )


# ---------------------------------------------------------------------------
# Adapter I/O types
# ---------------------------------------------------------------------------

@dataclass
class STALEInput:
    """Input passed to an adapter's process_query()."""
    uid: str
    sessions: List[Session]   # All 50 sessions, chronologically ordered
    query: str
    query_type: str           # "sr" | "pr" | "ipa"


@dataclass
class STALEOutput:
    """Output returned by an adapter's process_query()."""
    response: str
    retrieved_memories: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------

class STALEAdapter(ABC):
    """
    Abstract base class for STALE benchmark adapters.

    Subclass this and implement the three abstract methods to plug any
    LLM / memory system into the STALE evaluation pipeline.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialise the system (load models, connect to memory store, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def process_query(self, stale_input: STALEInput) -> STALEOutput:
        """
        Answer a single STALE probe query.

        Rules:
        - All three dimensions (sr / pr / ipa) must be called independently.
        - The system must base its answer on the full session history provided.
        - Return the raw generated text; scoring is done externally.
        """
        raise NotImplementedError

    @abstractmethod
    def get_context_window_size(self) -> int:
        """Return the model's maximum context window in tokens."""
        raise NotImplementedError

    def ingest_history(self, sessions: List[Session], session_n_id: Optional[str] = None) -> None:
        """
        [Optional] Write the conversation history into an external memory store.
        session_n_id: session_id of the session containing M_new (the updated state).
        Memory-augmented adapters must override this.
        """
        pass


# ---------------------------------------------------------------------------
# Utility: haystack formatter
# ---------------------------------------------------------------------------

def format_haystack(sessions: List[Session]) -> str:
    """Serialise 50 sessions into a single flat conversation-history string."""
    lines: List[str] = []
    for session in sessions:
        lines.append(f"[Session {session.session_id}]")
        lines.append(f"Timestamp: {session.timestamp}")
        for turn in session.turns:
            lines.append(f"{turn.role.upper()}: {turn.content}")
        lines.append("")   # blank line between sessions
    return "\n".join(lines)


def load_dataset(path: str) -> List[STALEInstance]:
    """Load a STALE dataset from a .jsonl file."""
    instances: List[STALEInstance] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                instances.append(STALEInstance.from_dict(json.loads(line)))
    return instances
