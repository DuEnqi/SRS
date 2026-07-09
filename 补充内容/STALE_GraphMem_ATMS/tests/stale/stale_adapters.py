#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STALE Benchmark - NPC Memory System Adapter

Integrates the project's NPCMemoryManager / NpcKnowledgeStore / MemorySynthesizer
with the STALE evaluation interface.  Two concrete adapters are provided:

  1. SimpleLLMAdapter  - raw LLM, no external memory; the full haystack is
                         passed as context in every call.
  2. NPCMemoryAdapter  - uses the project's NPCMemoryManager + NpcKnowledgeStore
                         to store sessions, then answers queries from retrieved
                         memories (mirrors MemoryFrameworkAdapter in the spec).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Any

# Ensure project root is on sys.path when running from tests/ directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests.stale.stale_types import (
    STALEAdapter,
    STALEInput,
    STALEOutput,
    Session,
    format_haystack,
)


# ---------------------------------------------------------------------------
# Helper: token approximation (4 chars ≈ 1 token)
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Helper: LLM 调用 + 429 自动重试
# ---------------------------------------------------------------------------

def _call_with_retry(fn, max_retries: int = 3, base_wait: float = 15.0):
    """
    调用 fn()，遇到 429 RateLimitError 时指数退避重试。
    超过 max_retries 次后抛出原始异常。
    base_wait: 第一次重试等待秒数（之后翻倍）。
    """
    import time
    wait = base_wait
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            status = getattr(e, "status_code", None)
            is_rate_limit = (
                status == 429
                or "429" in str(e)
                or "rate" in str(e).lower()
                or "RateLimitError" in type(e).__name__
            )
            if is_rate_limit and attempt < max_retries:
                print(f"[retry] 429 rate limit, waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                wait *= 2
                continue
            raise


# ---------------------------------------------------------------------------
# 1. SimpleLLMAdapter - full haystack in context, no external memory
# ---------------------------------------------------------------------------

class SimpleLLMAdapter(STALEAdapter):
    """
    Passes the complete conversation haystack as context on every call.
    Requires an OpenAI-compatible client (uses project LLMFactory if available,
    falls back to direct openai import otherwise).
    """

    def __init__(
        self,
        model_name: str = "",
        max_context_tokens: int = 128_000,
    ) -> None:
        self.model_name = model_name or os.getenv("STALE_MODEL", "openai/gpt-5.4-mini")
        self.max_context_tokens = max_context_tokens
        self._config: Dict[str, Any] = {}
        self._client: Optional[Any] = None
        self._llm: Optional[Any] = None
        self._use_langchain = False
        self._llm_ready = False

    # ------------------------------------------------------------------
    def initialize(self, config: Dict[str, Any]) -> None:
        """Store config; actual client creation is deferred to first call."""
        self._config = config or {}

    # ------------------------------------------------------------------
    def _ensure_llm(self) -> None:
        """Lazy-initialise the LLM client on first use."""
        if self._llm_ready:
            return
        try:
            from npc.llm_config.llm import LLMFactory
            from npc.utils.constants import LLMUsage
            self._llm = LLMFactory.create_chat_model(
                usage=LLMUsage.GENERAL,
                model_name=self.model_name,
                temperature=0.0,
            )
            self._use_langchain = True
            print(f"[SimpleLLMAdapter] Initialised via LLMFactory ({self.model_name})")
        except Exception as e:
            print(f"[SimpleLLMAdapter] LLMFactory unavailable ({e}), trying openai directly")
            from openai import OpenAI
            api_key = self._config.get("api_key") or os.getenv("OPENAI_API_KEY")
            base_url = self._config.get("base_url") or os.getenv("OPENAI_API_BASE")
            if not api_key:
                raise RuntimeError(
                    "No LLM backend available. Set OPENAI_API_KEY or install "
                    "langchain_openai. For no-key testing use MockSTALEAdapter."
                )
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self._use_langchain = False
        self._llm_ready = True

    # ------------------------------------------------------------------
    def get_context_window_size(self) -> int:
        return self.max_context_tokens

    # ------------------------------------------------------------------
    def _truncate_context(self, context: str) -> str:
        """Drop characters from the front if context exceeds the window."""
        max_chars = self.max_context_tokens * 4
        if len(context) > max_chars:
            context = context[-max_chars:]
            # Trim to next newline so we don't start mid-sentence
            newline_pos = context.find("\n")
            if newline_pos != -1:
                context = context[newline_pos + 1:]
        return context

    # ------------------------------------------------------------------
    def process_query(self, stale_input: STALEInput) -> STALEOutput:
        self._ensure_llm()
        context = format_haystack(stale_input.sessions)
        context = self._truncate_context(context)

        prompt = (
            "You are a helpful AI assistant with access to a user's complete "
            "conversation history below.\n\n"
            "Conversation History:\n"
            f"{context}\n\n"
            "Based on the conversation history above, answer the following question "
            "accurately. If a previous belief is contradicted by a later conversation, "
            "the later information takes precedence.\n\n"
            f"Question: {stale_input.query}\n"
            "Answer:"
        )

        def _do_call():
            if self._use_langchain:
                from langchain_core.messages import HumanMessage
                return self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
            else:
                comp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.0,
                )
                return comp.choices[0].message.content.strip()

        try:
            text = _call_with_retry(_do_call)
        except Exception as exc:
            print(f"[SimpleLLMAdapter] LLM call failed: {exc}")
            text = None

        return STALEOutput(response=text or "")


# ---------------------------------------------------------------------------
# 2. NPCMemoryAdapter - uses project NPCMemoryManager + NpcKnowledgeStore
# ---------------------------------------------------------------------------

class NPCMemoryAdapter(STALEAdapter):
    """
    Stores conversation sessions in the project's knowledge store and answers
    STALE queries using retrieved memories — mirroring how the NPC system
    recalls past-scene knowledge during gameplay.

    Lifecycle per STALE instance:
      1. ingest_history(sessions)  → writes sessions to NpcKnowledgeStore
      2. process_query(input)      → retrieves relevant memories, calls LLM
    """

    # NPC name used as a stable key inside NPCMemoryManager
    _NPC_KEY = "stale_eval_npc"

    def __init__(
        self,
        model_name: str = "",
        max_retrieved: int = 10,
    ) -> None:
        self.model_name = model_name or os.getenv("STALE_MODEL", "openai/gpt-5.4-mini")
        self.max_retrieved = max_retrieved

        self._knowledge_store: Optional[Any] = None
        self._memory_manager: Optional[Any] = None
        self._llm: Optional[Any] = None
        self._raw_client: Optional[Any] = None
        self._use_langchain = False
        self._config: Dict[str, Any] = {}
        self._llm_ready = False

    # ------------------------------------------------------------------
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialise the knowledge store; LLM client creation is deferred."""
        self._config = config or {}
        from npc.knowledge.knowledge_store import NpcKnowledgeStore
        from npc.knowledge.npc_memory_manager import NPCMemoryManager
        self._knowledge_store = NpcKnowledgeStore()
        self._memory_manager = NPCMemoryManager()

    # ------------------------------------------------------------------
    def _ensure_llm(self) -> None:
        """Lazy-initialise the LLM client on first use."""
        if self._llm_ready:
            return
        try:
            from npc.llm_config.llm import LLMFactory
            from npc.utils.constants import LLMUsage
            self._llm = LLMFactory.create_chat_model(
                usage=LLMUsage.GENERAL,
                model_name=self.model_name,
                temperature=0.0,
            )
            self._use_langchain = True
            print(f"[NPCMemoryAdapter] Initialised via LLMFactory ({self.model_name})")
        except Exception as e:
            print(f"[NPCMemoryAdapter] LLMFactory unavailable ({e}), trying openai directly")
            from openai import OpenAI
            api_key = self._config.get("api_key") or os.getenv("OPENAI_API_KEY")
            base_url = self._config.get("base_url") or os.getenv("OPENAI_API_BASE")
            if not api_key:
                raise RuntimeError(
                    "No LLM backend available. Set OPENAI_API_KEY or install "
                    "langchain_openai. For no-key testing use MockSTALEAdapter."
                )
            self._raw_client = OpenAI(api_key=api_key, base_url=base_url)
            self._use_langchain = False
        self._llm_ready = True

    # ------------------------------------------------------------------
    def get_context_window_size(self) -> int:
        return 128_000

    # ------------------------------------------------------------------
    def ingest_history(self, sessions: List[Session], session_n_id: Optional[str] = None) -> None:
        """
        Write each turn of every session into the knowledge store.
        session_n_id: session_id of the session containing M_new.
                      Used to pin those messages to the front of every retrieval.
        """
        if self._knowledge_store is None:
            raise RuntimeError("Call initialize() before ingest_history()")

        self._knowledge_store.knowledge_messages.clear()
        self._knowledge_store.current_scene_messages.clear()

        self._total_sessions = len(sessions)
        self._session_n_id = session_n_id  # 精确记录 M_new 所在 session

        for session_idx, session in enumerate(sessions):
            for turn in session.turns:
                self._knowledge_store.add_message(
                    speaker=turn.role,
                    listeners=[self._NPC_KEY],
                    content=turn.content,
                    to_current_scene=False,
                )
                record = self._knowledge_store.knowledge_messages[-1]
                record["_session_idx"] = session_idx
                record["_session_id"] = session.session_id

    # ------------------------------------------------------------------
    def _retrieve_relevant_messages(self, query: str, sessions: List[Session]) -> List[str]:
        """
        检索相关记忆，应用三项修复：
        1. M_new session 强制注入：精确用 session_n_id 标识，而非盲猜最后一个
        2. 时序权重：session 越新，TF-IDF 得分乘以更大系数
        3. max_retrieved=10 减少噪音
        """
        if self._knowledge_store is None:
            return []

        total_sessions = getattr(self, "_total_sessions", max(len(sessions), 1))
        session_n_id = getattr(self, "_session_n_id", None)
        query_words = set(query.lower().split())

        all_messages = (
            self._knowledge_store.knowledge_messages
            + self._knowledge_store.current_scene_messages
        )

        # --- 精确强制注入：M_new 所在 session 的全部消息 ---
        pinned: List[str] = []
        for record in all_messages:
            rid = record.get("_session_id", "")
            # 若知道 session_n_id 就精确匹配；否则退回用最高 session_idx
            if session_n_id:
                if rid == session_n_id:
                    pinned.append(record.get("content", ""))
            else:
                if record.get("_session_idx", -1) == total_sessions - 1:
                    pinned.append(record.get("content", ""))

        # --- TF-IDF + 时序权重（跳过已 pin 的 M_new session）---
        scored: List[tuple[float, str]] = []
        for record in all_messages:
            rid = record.get("_session_id", "")
            is_pinned = (rid == session_n_id) if session_n_id else (
                record.get("_session_idx", -1) == total_sessions - 1
            )
            if is_pinned:
                continue
            content: str = record.get("content", "")
            words = set(content.lower().split())
            overlap = len(query_words & words)
            if overlap > 0:
                session_idx = record.get("_session_idx", 0)
                recency = (session_idx + 1) / total_sessions
                score = (overlap / max(len(query_words), 1)) * (1.0 + recency)
                scored.append((score, content))

        scored.sort(key=lambda x: x[0], reverse=True)

        remaining_slots = max(self.max_retrieved - len(pinned), 0)
        ranked = [c for _, c in scored[:remaining_slots]]

        # M_new session 消息在最前（Memory 1~N），相关旧记忆跟在后面
        return pinned + ranked

    # ------------------------------------------------------------------
    def process_query(self, stale_input: STALEInput) -> STALEOutput:
        if self._knowledge_store is None:
            raise RuntimeError("Call initialize() before process_query()")
        self._ensure_llm()

        retrieved = self._retrieve_relevant_messages(stale_input.query, stale_input.sessions)

        if retrieved:
            # 列表顺序：最新 session（含 M_new）在最前，旧记忆在后
            context_block = "\n".join(
                f"[Memory {i+1}]: {mem}" for i, mem in enumerate(retrieved)
            )
        else:
            context_block = format_haystack(stale_input.sessions)[-16_000:]

        prompt = (
            "You are a helpful AI assistant. You have access to retrieved memories "
            "from a user's past conversation history.\n\n"
            "Retrieved Memories (ordered newest first — Memory 1 is the most recent):\n"
            f"{context_block}\n\n"
            "Important rules:\n"
            "- Memory 1 is the MOST RECENT. Lower-numbered memories always override "
            "higher-numbered ones if they contradict each other.\n"
            "- Before answering, check whether the question contains any assumptions "
            "about the user's past state. If those assumptions are contradicted by a "
            "recent memory, explicitly point that out before (or instead of) answering.\n\n"
            f"Question: {stale_input.query}\n"
            "Answer:"
        )

        def _do_call():
            if self._use_langchain:
                from langchain_core.messages import HumanMessage
                return self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
            else:
                comp = self._raw_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.0,
                )
                return comp.choices[0].message.content.strip()

        try:
            text = _call_with_retry(_do_call)
        except Exception as exc:
            print(f"[NPCMemoryAdapter] LLM call failed: {exc}")
            text = None

        return STALEOutput(
            response=text or "",
            retrieved_memories=retrieved,
        )
