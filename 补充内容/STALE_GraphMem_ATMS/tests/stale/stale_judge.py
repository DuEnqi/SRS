#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STALE Benchmark - LLM-as-Judge

Scores model responses on the three STALE dimensions (SR / PR / IPA) using
an LLM judge.  Implements both a full LLM judge and a lightweight rule-based
fallback that works without any API key.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests.stale.stale_types import STALEInstance


# ---------------------------------------------------------------------------
# Judge result dataclass
# ---------------------------------------------------------------------------

class DimResult:
    """Result for a single evaluation dimension."""

    def __init__(self, passed: bool, reasoning: str) -> None:
        self.passed = passed
        self.reasoning = reasoning

    def to_dict(self) -> Dict[str, Any]:
        return {"pass": self.passed, "reasoning": self.reasoning}

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"DimResult({status}: {self.reasoning[:60]})"


class JudgeResult:
    """Aggregated judgement for one STALE instance (all three dimensions)."""

    def __init__(
        self,
        uid: str,
        sr: DimResult,
        pr: DimResult,
        ipa: DimResult,
    ) -> None:
        self.uid = uid
        self.sr = sr
        self.pr = pr
        self.ipa = ipa

    @property
    def overall_pass(self) -> bool:
        return self.sr.passed and self.pr.passed and self.ipa.passed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "sr": self.sr.to_dict(),
            "pr": self.pr.to_dict(),
            "ipa": self.ipa.to_dict(),
            "overall_pass": self.overall_pass,
        }

    def __repr__(self) -> str:
        return (
            f"JudgeResult(uid={self.uid[:8]}… "
            f"SR={'✓' if self.sr.passed else '✗'} "
            f"PR={'✓' if self.pr.passed else '✗'} "
            f"IPA={'✓' if self.ipa.passed else '✗'})"
        )


# ---------------------------------------------------------------------------
# Rule-based fallback judge  (no LLM required)
# ---------------------------------------------------------------------------

class RuleBasedJudge:
    """
    Lightweight judge that checks responses using keyword heuristics.

    Scoring logic per dimension:
      SR  - PASS if response does NOT affirm the old state AND contains a
            negation OR the new state keyword.
      PR  - PASS if response contains a correction phrase (e.g. "actually",
            "however", "no longer", "incorrect", "moved", "changed").
      IPA - PASS if response mentions the new-state keyword and does NOT
            exclusively mention the old-state keyword.
    """

    # Words that indicate an old state is being affirmed (bad for SR/IPA)
    _AFFIRMATION = {"yes", "still", "currently", "indeed", "correct", "right"}

    # Words that indicate correction / negation (good for SR / PR)
    _CORRECTION = {
        "no", "not", "no longer", "actually", "however", "moved",
        "changed", "different", "instead", "now", "new", "updated",
        "incorrect", "mistaken", "wrong", "left", "relocated",
    }

    def _words(self, text: str) -> set:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def _contains_any(self, text: str, keywords: set) -> bool:
        words = self._words(text)
        return bool(words & keywords)

    def _extract_state_keywords(self, statement: str) -> set:
        """Pull meaningful nouns / proper-nouns from a short statement."""
        # Rough heuristic: words longer than 4 chars, capitalised words
        words = re.findall(r"\b[A-Z][a-z]+\b|\b[a-z]{5,}\b", statement)
        return {w.lower() for w in words}

    def judge_instance(
        self,
        instance: STALEInstance,
        responses: Dict[str, str],
    ) -> JudgeResult:
        """
        Args:
            instance: The STALE instance with ground truth.
            responses: {"sr": "...", "pr": "...", "ipa": "..."}
        """
        # 空响应（LM调用失败）直接全 FAIL，不走兜底逻辑误判
        def _is_empty(r: str) -> bool:
            return not r or not r.strip()

        if all(_is_empty(responses.get(d, "")) for d in ("sr", "pr", "ipa")):
            return JudgeResult(
                uid=instance.uid,
                sr=DimResult(False, "empty response (LLM call failed)"),
                pr=DimResult(False, "empty response (LLM call failed)"),
                ipa=DimResult(False, "empty response (LLM call failed)"),
            )

        old_kw = self._extract_state_keywords(instance.m_old)
        new_kw = self._extract_state_keywords(instance.m_new)

        # ---- SR ----
        sr_resp = responses.get("sr", "")
        sr_words = self._words(sr_resp)
        # Pass if model does not simply say "yes" and uses correction language
        affirms_old = bool(sr_words & self._AFFIRMATION) and not bool(sr_words & self._CORRECTION)
        mentions_new = bool(sr_words & new_kw)
        sr_pass = (not affirms_old) or mentions_new
        sr_reasoning = (
            f"old_kw={old_kw & sr_words}, new_kw={new_kw & sr_words}, "
            f"correction={sr_words & self._CORRECTION}"
        )

        # ---- PR ----
        pr_resp = responses.get("pr", "")
        pr_words = self._words(pr_resp)
        pr_pass = self._contains_any(pr_resp, self._CORRECTION)
        pr_reasoning = f"correction_words_found={pr_words & self._CORRECTION}"

        # ---- IPA ----
        ipa_resp = responses.get("ipa", "")
        ipa_words = self._words(ipa_resp)
        mentions_new_ipa = bool(ipa_words & new_kw)
        only_mentions_old = bool(ipa_words & old_kw) and not mentions_new_ipa
        ipa_pass = not only_mentions_old
        ipa_reasoning = (
            f"old_kw_in_resp={old_kw & ipa_words}, new_kw_in_resp={new_kw & ipa_words}"
        )

        return JudgeResult(
            uid=instance.uid,
            sr=DimResult(sr_pass, sr_reasoning),
            pr=DimResult(pr_pass, pr_reasoning),
            ipa=DimResult(ipa_pass, ipa_reasoning),
        )


# ---------------------------------------------------------------------------
# LLM Judge  (uses project LLMFactory or raw OpenAI client)
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to evaluate responses on the three STALE dimensions.
    Falls back to RuleBasedJudge if the LLM call fails.
    """

    _JUDGE_PROMPT = """You are a strict evaluator for an LLM memory benchmark called STALE.

Ground Truth for this instance:
- Old memory (M_old): {m_old}
- New memory (M_new): {m_new}
- Explanation of conflict: {explanation}

Model responses to evaluate:

SR Query: "{sr_query}"
SR Response: "{sr_response}"

PR Query: "{pr_query}"
PR Response: "{pr_response}"

IPA Query: "{ipa_query}"
IPA Response: "{ipa_response}"

Scoring criteria:
- SR (State Recognition): PASS if the model correctly recognises that M_old is no longer valid because of M_new.
- PR (Presupposition Rejection): PASS if the model rejects or corrects the false premise embedded in the question (the premise assumes M_old is still true).
- IPA (Implicit Preference Alignment): PASS if the model's response is consistent with M_new (not M_old) when performing the requested task.

Return ONLY a JSON object with this exact structure:
{{"sr": {{"pass": true/false, "reasoning": "one sentence"}}, "pr": {{"pass": true/false, "reasoning": "one sentence"}}, "ipa": {{"pass": true/false, "reasoning": "one sentence"}}}}"""

    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name or os.getenv("STALE_MODEL", "openai/gpt-5.4-mini")
        self._fallback = RuleBasedJudge()
        self._llm = None
        self._raw_client = None
        self._use_langchain = False
        self._initialised = False

    def _lazy_init(self) -> None:
        if self._initialised:
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
        except Exception:
            try:
                from openai import OpenAI
                self._raw_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_API_BASE"),
                )
                self._use_langchain = False
            except Exception:
                pass  # Will fall back to rule-based
        self._initialised = True

    def _call_llm(self, prompt: str) -> Optional[str]:
        self._lazy_init()

        def _do_call():
            if self._use_langchain and self._llm:
                from langchain_core.messages import HumanMessage
                return self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
            elif self._raw_client:
                comp = self._raw_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.0,
                )
                return comp.choices[0].message.content.strip()
            return None

        try:
            from tests.stale.stale_adapters import _call_with_retry
            return _call_with_retry(_do_call)
        except Exception as e:
            print(f"[LLMJudge] LLM call failed: {e}")
        return None

    def _parse_json_result(self, raw: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from the LLM response (may be wrapped in markdown)."""
        # Try to extract JSON block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def judge_instance(
        self,
        instance: STALEInstance,
        responses: Dict[str, str],
    ) -> JudgeResult:
        prompt = self._JUDGE_PROMPT.format(
            m_old=instance.m_old,
            m_new=instance.m_new,
            explanation=instance.explanation,
            sr_query=instance.queries["sr"].question,
            sr_response=responses.get("sr", ""),
            pr_query=instance.queries["pr"].question,
            pr_response=responses.get("pr", ""),
            ipa_query=instance.queries["ipa"].question,
            ipa_response=responses.get("ipa", ""),
        )

        raw = self._call_llm(prompt)
        if raw:
            parsed = self._parse_json_result(raw)
            if parsed and all(k in parsed for k in ("sr", "pr", "ipa")):
                return JudgeResult(
                    uid=instance.uid,
                    sr=DimResult(
                        bool(parsed["sr"].get("pass", False)),
                        parsed["sr"].get("reasoning", ""),
                    ),
                    pr=DimResult(
                        bool(parsed["pr"].get("pass", False)),
                        parsed["pr"].get("reasoning", ""),
                    ),
                    ipa=DimResult(
                        bool(parsed["ipa"].get("pass", False)),
                        parsed["ipa"].get("reasoning", ""),
                    ),
                )

        # Fall back to rule-based
        print(f"[LLMJudge] Falling back to rule-based judge for uid={instance.uid[:8]}")
        return self._fallback.judge_instance(instance, responses)
