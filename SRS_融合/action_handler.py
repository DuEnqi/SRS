#!/usr/bin/env python3
"""
action_handler.py — 玩家动作分析（参照 npc_consensus_v13_updated.analyze_action）

将 Talk / Inspect / Question / Accuse / Give Evidence / Continue
映射为：默认台词 → 命题抽取 → 信念/信任更新 → LLM 提示。
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
    "Accuse": "I accuse the knight of being a fraud. The evidence doesn't add up.",
    "Give Evidence": "I present evidence about the knight's suspicious armor markings.",
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

# 规则 fallback（npc_consensus _fallback 简化版）
_KNIGHT_POS = re.compile(r"\b(trust|trustworthy|real|legitimate|hero|brave|honest)\b", re.I)
_KNIGHT_NEG = re.compile(r"\b(fake|fraud|impostor|deserter|lie|liar|forgery|markings|suspicious)\b", re.I)
_CHANGE = re.compile(
    r"\b(no longer|fake|fraud|impostor|revealed|evidence|accuse|markings|deserter)\b", re.I)


def effective_utterance(action_type: str, player_input: str, npc_name: str) -> str:
    """空输入时用动作默认台词（与原 SRS mock 行为一致）。"""
    text = (player_input or "").strip()
    if text:
        return text
    base = ACTION_DEFAULT_UTTERANCE.get(action_type, ACTION_DEFAULT_UTTERANCE["Talk"])
    if "{npc}" in base:
        return base.format(npc=npc_name)
    return base.replace("you", npc_name) if action_type == "Talk" else base


def _rule_analyze(text: str, action_type: str, actor: str) -> dict:
    """规则 analyze_action fallback。"""
    low = text.lower()
    chitchat = action_type in ("Talk", "Continue") and not _CHANGE.search(low) and len(text) < 40
    if chitchat and not _KNIGHT_NEG.search(text) and not _KNIGHT_POS.search(text):
        return {
            "proposition_key": "chitchat",
            "content_label": text[:80],
            "polarity": 0.0,
            "evidence_strength": 0.2,
            "is_chitchat": True,
            "category": "emotional",
            "target_facts": [],
        }

    polarity = -1.0 if _KNIGHT_NEG.search(text) else (1.0 if _KNIGHT_POS.search(text) else 0.0)
    prop = "player_misc"
    strength = 0.45
    targets: List[str] = []

    if _KNIGHT_NEG.search(text) or action_type in ("Accuse",):
        prop = "knight_is_fake"
        polarity = -1.0
        strength = 0.88 if action_type in ("Accuse", "Give Evidence") else 0.65
        targets = ["knight_is_trustworthy"]
    elif _KNIGHT_POS.search(text):
        prop = "knight_is_trustworthy"
        polarity = 1.0
        strength = 0.7
        targets = ["knight_is_fake"]
    elif action_type == "Inspect":
        prop = "scene_observation"
        strength = 0.55
    elif action_type == "Question":
        prop = "player_inquiry"
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

    fb = _rule_analyze(text, action_type, actor)
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
