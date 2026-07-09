#!/usr/bin/env python3
"""
action_handler.py — 玩家动作分析（参照 npc_consensus_v13_updated.analyze_action）

将 Talk / Inspect / Question / Accuse / Give Evidence / Continue
映射为：默认台词 → 命题抽取 → 信念/信任更新 → LLM 提示。

Phase 3: 无场景硬编码，规则层从 known_props + 通用语义推断命题。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# 与 Play.jsx ACTIONS 一致
ACTION_TYPES = ("Talk", "Inspect", "Question", "Accuse", "Give Evidence", "Continue")

ACTION_DEFAULT_UTTERANCE = {
    "Talk": "I'd like to talk with you about recent events.",
    "Inspect": "I carefully examine the surroundings and anything notable nearby.",
    "Question": "I have some questions - can you tell me what you know?",
    "Accuse": "I want to raise a serious accusation based on what I've seen.",
    "Give Evidence": "I present evidence about something suspicious I've noticed.",
    "Continue": "Please continue - tell me more about what you've seen.",
}

ACTION_NPC_HINT = {
    "Talk": "Respond naturally in conversation. Share what you know if relevant.",
    "Inspect": "Describe what the player notices from your perspective or the scene.",
    "Question": "Answer the player's questions based on your beliefs and memories.",
    "Accuse": "React to the accusation — defend, deny, or reconsider based on your beliefs.",
    "Give Evidence": "React to the evidence presented — update stance if convincing.",
    "Continue": "Continue the conversation thread; elaborate on prior topic.",
}

# 通用语义（无场景专有实体）
_NEG = re.compile(
    r"\b(fake|fraud|impostor|deserter|lie|liar|forgery|suspicious|wrong|false|guilty|stolen)\b",
    re.I,
)
_POS = re.compile(
    r"\b(trust|trustworthy|real|legitimate|hero|brave|honest|innocent|true|safe)\b",
    re.I,
)
_CHANGE = re.compile(
    r"\b(no longer|fake|fraud|impostor|revealed|evidence|accuse|markings|deserter|contradict)\b",
    re.I,
)
_WORD_TO_CLAIM = re.compile(r"[a-z][a-z0-9_]{2,}")


def effective_utterance(action_type: str, player_input: str, npc_name: str) -> str:
    """空输入时用动作默认台词（与原 SRS mock 行为一致）。"""
    text = (player_input or "").strip()
    if text:
        return text
    base = ACTION_DEFAULT_UTTERANCE.get(action_type, ACTION_DEFAULT_UTTERANCE["Talk"])
    if "{npc}" in base:
        return base.format(npc=npc_name)
    return base.replace("you", npc_name) if action_type == "Talk" else base


def _claim_from_text(text: str, known_props: Dict[str, str]) -> Optional[str]:
    """从文本 token 匹配 known_props 中的 claim_id。"""
    low = text.lower()
    for cid in known_props:
        if cid.replace("_", " ") in low or cid in low:
            return cid
    tokens = set(_WORD_TO_CLAIM.findall(low))
    for cid in known_props:
        parts = set(cid.split("_"))
        if parts & tokens:
            return cid
    return None


def _default_prop_key(text: str, action_type: str, known_props: Dict[str, str]) -> str:
    matched = _claim_from_text(text, known_props)
    if matched:
        return matched
    if action_type == "Inspect":
        return "scene_observation"
    if action_type == "Question":
        return "player_inquiry"
    if action_type in ("Accuse", "Give Evidence"):
        return "player_accusation" if action_type == "Accuse" else "player_evidence"
    return "player_misc"


def _target_facts(prop: str, polarity: float, known_props: Dict[str, str]) -> List[str]:
    """推断与 prop 可能矛盾的已有命题。"""
    if not known_props or prop in ("chitchat", "scene_observation", "player_inquiry", "player_misc"):
        return []
    if polarity < 0:
        return [cid for cid in known_props if cid != prop]
    if polarity > 0:
        return []
    return []


def _rule_analyze(
    text: str,
    action_type: str,
    actor: str,
    known_props: Optional[Dict[str, str]] = None,
) -> dict:
    """规则 analyze_action fallback（场景无关）。"""
    known_props = known_props or {}
    low = text.lower()
    chitchat = action_type in ("Talk", "Continue") and not _CHANGE.search(low) and len(text) < 40
    if chitchat and not _NEG.search(text) and not _POS.search(text):
        return {
            "proposition_key": "chitchat",
            "content_label": text[:80],
            "polarity": 0.0,
            "evidence_strength": 0.2,
            "is_chitchat": True,
            "category": "emotional",
            "target_facts": [],
        }

    polarity = -1.0 if _NEG.search(text) else (1.0 if _POS.search(text) else 0.0)
    prop = _default_prop_key(text, action_type, known_props)
    strength = 0.45
    targets: List[str] = []

    if action_type in ("Accuse", "Give Evidence"):
        strength = 0.88 if action_type == "Accuse" else 0.82
        if polarity == 0.0 and action_type == "Accuse":
            polarity = -1.0
        targets = _target_facts(prop, polarity, known_props)
    elif _NEG.search(text):
        polarity = -1.0
        prop = _claim_from_text(text, known_props) or prop
        targets = _target_facts(prop, polarity, known_props)
        strength = 0.65
    elif _POS.search(text):
        polarity = 1.0
        prop = _claim_from_text(text, known_props) or prop
        strength = 0.7
    elif action_type == "Inspect":
        strength = 0.55
    elif action_type == "Question":
        strength = 0.4

    if action_type == "Give Evidence":
        strength = min(0.95, strength + 0.15)

    return {
        "proposition_key": prop,
        "content_label": text[:120],
        "polarity": polarity,
        "evidence_strength": strength,
        "is_chitchat": False,
        "category": "fact_about_player" if "player" in prop else "task",
        "target_facts": targets,
    }


def analyze_player_action(
    text: str,
    action_type: str,
    actor: str = "Player",
    *,
    known_props: Optional[Dict[str, str]] = None,
    use_llm: bool = True,
) -> dict:
    """
    分析玩家动作 → 命题 + 关系（LLM 优先，规则 fallback）。
    参照 npc_consensus_v13_updated.ActionJudge.analyze_action。
    """
    action_type = action_type if action_type in ACTION_TYPES else "Talk"
    known_props = known_props or {}

    if use_llm:
        try:
            from srs_llm import analyze_action_llm  # noqa: WPS433
            llm_result = analyze_action_llm(text, actor, action_type, known_props)
            if llm_result and llm_result.get("proposition_key"):
                llm_result["action_type"] = action_type
                llm_result["source"] = "llm"
                return llm_result
        except Exception:
            pass

    fb = _rule_analyze(text, action_type, actor, known_props)
    fb["action_type"] = action_type
    fb["source"] = "rule"
    return fb


def action_belief_trust_deltas(analysis: dict, action_type: str) -> tuple[float, float]:
    """根据分析结果计算 belief/trust 变化量。"""
    if analysis.get("is_chitchat"):
        return 0.02, 0.01
    strength = float(analysis.get("evidence_strength", 0.5))
    pol = float(analysis.get("polarity", 0))
    belief = abs(pol) * strength * 0.12
    trust = 0.0
    if action_type in ("Accuse", "Give Evidence"):
        trust = 0.04 if pol != 0 else 0.02
    elif action_type == "Question":
        trust = 0.01
    return belief, trust
