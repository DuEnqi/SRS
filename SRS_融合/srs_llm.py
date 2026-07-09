#!/usr/bin/env python3
"""LLM helper for SRS dialogue (formal evidence-block + character reply)."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_env() -> None:
    root = Path(__file__).resolve().parent.parent
    try:
        sys_path = str(root)
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        from llm_env import load_llm_env  # noqa: WPS433
        load_llm_env(root)
    except Exception:
        pass


def _chat(messages: List[dict], *, max_tokens: int = 180, temperature: float = 0.7) -> str:
    _load_env()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("YUNWU_API_KEY")
    base = (os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1" if "azure" not in base else base
    model = os.getenv("STALE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    if not api_key:
        return ""
    url = f"{base}/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()


def generate_npc_response(
    npc: dict,
    player_input: str,
    *,
    action_type: str = "Talk",
    action_hint: str = "",
    evidence_block: str = "",
    dialogue_history: Optional[List[dict]] = None,
    scenario: Optional[dict] = None,
    analysis: Optional[dict] = None,
) -> str:
    """Generate in-character NPC reply with action-aware prompt."""
    hist = dialogue_history or []
    hist_txt = "\n".join(f"- {d.get('speaker', '?')}: {d.get('text', '')}" for d in hist[-6:])
    beliefs = npc.get("beliefs") or []
    bel_txt = "\n".join(
        f"- {b.get('statement', b.get('id', '?'))} (conf={b.get('confidence', 0):.0%})"
        for b in beliefs[:6]
    )
    scen = scenario or {}
    hint = action_hint or ""
    anal = analysis or {}
    action_line = f"Player action type: {action_type}\nAction guidance: {hint}"

    system = (
        "You are a character in an interactive narrative mystery game (Greyford village, fake knight plot). "
        "Stay in character. Respond in 1-3 sentences. Match the player's action type "
        "(Talk=chat, Inspect=describe observations, Question=answer questions, "
        "Accuse=react to accusation, Give Evidence=react to proof, Continue=elaborate)."
    )
    user = f"""
Character: {npc.get('name')} ({npc.get('role')})
Personality: {npc.get('personality', '')}
Goal: {npc.get('currentGoal', '')}
Hidden motivation: {npc.get('hiddenMotivation', '')}

Scenario: {scen.get('name', 'Greyford')} — {scen.get('description', '')[:300]}

Your beliefs:
{bel_txt or '(none)'}

{evidence_block}

Memory analysis of player's move: {json.dumps({k: anal.get(k) for k in ('proposition_key', 'polarity', 'evidence_strength', 'is_chitchat') if k in anal}, ensure_ascii=False)}

{action_line}

Recent dialogue:
{hist_txt or '(none)'}

Player says/does: "{player_input}"

Reply as {npc.get('name')} in character. Do not break the fourth wall.
""".strip()
    text = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    if text:
        return text
    return _action_fallback_reply(npc, action_type, player_input)


def _action_fallback_reply(npc: dict, action_type: str, player_input: str) -> str:
    """LLM 不可用时的动作类型模板回复。"""
    name = npc.get("name", "NPC")
    templates = {
        "Talk": f"{name} nods thoughtfully. \"I've been watching the village closely lately.\"",
        "Inspect": f"{name} follows your gaze. \"From here you can see the knight's camp near the elder's hall.\"",
        "Question": f"{name} considers your question. \"Ask what you will — I'll share what I know honestly.\"",
        "Accuse": f"{name}'s expression hardens. \"That's a serious claim. I won't dismiss it lightly.\"",
        "Give Evidence": f"{name} studies what you show. \"This... changes how I see things.\"",
        "Continue": f"{name} continues, \"There's more you should know about the knight's arrival.\"",
    }
    return templates.get(action_type, templates["Talk"])


def analyze_action_llm(
    text: str,
    actor: str,
    action_type: str,
    known_props: Dict[str, str],
) -> Optional[dict]:
    """LLM 版 analyze_action（精简 npc_consensus 字段）。"""
    props_desc = "\n".join(f"- {k}: {v}" for k, v in list(known_props.items())[:12])
    system = (
        "Game memory engine. Given player action + utterance, output JSON only: "
        "proposition_key (snake_case), content_label, evidence_strength (0-1), "
        "polarity (+1 support / -1 contradict / 0 neutral), is_chitchat (bool), "
        "target_facts (array of fact ids to invalidate if contradicted). "
        "Knight-related: knight_is_trustworthy, knight_is_fake."
    )
    user = (
        f"Action type: {action_type}\nActor: {actor}\nUtterance: 「{text}」\n"
        f"Known facts:\n{props_desc or '- knight_is_trustworthy\n- knight_is_fake'}"
    )
    raw = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=200, temperature=0.3)
    if not raw:
        return None
    try:
        m = __import__("re").search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group(0))
            data.setdefault("target_facts", [])
            return data
    except Exception:
        pass
    return None


def generate_npc_dialogue(
    npc1: dict,
    npc2: dict,
    *,
    conflicts: Optional[List[dict]] = None,
    dialogue_history: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Generate short NPC-NPC exchange; parse JSON if model complies."""
    c_txt = json.dumps(conflicts or [], ensure_ascii=False)[:400]
    user = f"""
Generate a short dialogue between {npc1.get('name')} and {npc2.get('name')}.
Belief conflicts (if any): {c_txt}

Return JSON only:
{{"dialogue":[{{"speaker":"{npc1.get('name')}","text":"..."}},{{"speaker":"{npc2.get('name')}","text":"..."}}],
 "beliefChanges":{{"{npc1.get('id')}":0.02,"{npc2.get('id')}":-0.01}},
 "trustChanges":{{"{npc1.get('id')}->{npc2.get('id')}":0.01}}}}
"""
    raw = _chat(
        [{"role": "system", "content": "Narrative game dialogue generator. JSON only."},
         {"role": "user", "content": user}],
        max_tokens=320,
        temperature=0.8,
    )
    try:
        m = __import__("re").search(r"\{[\s\S]*\}", raw or "")
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    n1, n2 = npc1.get("name", "A"), npc2.get("name", "B")
    return {
        "dialogue": [
            {"speaker": n1, "text": f"{n1} shares what they've observed lately."},
            {"speaker": n2, "text": f"{n2} disagrees on some details."},
        ],
        "beliefChanges": {npc1.get("id", ""): 0.02, npc2.get("id", ""): -0.01},
        "trustChanges": {f"{npc1.get('id')}->{npc2.get('id')}": 0.01},
    }
