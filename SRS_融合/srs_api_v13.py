#!/usr/bin/env python3
"""
srs_api_v13.py — 统一 GraphMem-ATMS SRS 后端
=============================================
形式化层 + v13 图 + 组友时间戳 + LLM 对话，对接 ForCadia/SRS 前端。

启动: python srs_api_v13.py --port 8765
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_UPLOAD = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_UPLOAD))

from game_engine_v13 import GameEngineV13, get_engine  # noqa: E402
from srs_llm import generate_npc_dialogue, generate_npc_response  # noqa: E402

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    print("pip install fastapi uvicorn pydantic")
    raise

app = FastAPI(title="GraphMem-ATMS SRS Backend", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine: GameEngineV13 = get_engine()


# ── Models ──

class NPCRequest(BaseModel):
    npcId: str = ""
    playerInput: str = ""
    actionType: str = "Talk"
    context: dict = {}


class DialogueRequest(BaseModel):
    npcId1: str = ""
    npcId2: str = ""
    context: dict = {}


class BeliefUpdate(BaseModel):
    npcId: str = ""
    change: float = 0.0
    claim: str = ""
    confidence: float = 0.5
    evidence: str = ""


class MemoryUpdate(BaseModel):
    type: str = "event"
    title: str = ""
    description: str = ""
    source: str = ""
    confidence: float = 0.8
    relatedNPCs: list = []


class TrustUpdate(BaseModel):
    sourceId: str = ""
    targetId: str = ""
    change: float = 0.0


class TimeAdvance(BaseModel):
    days: float = 1.0


class TurnAdvance(BaseModel):
    steps: int = 1


class ConflictRequest(BaseModel):
    claimId1: str = "knight_is_trustworthy"
    claimId2: str = "knight_is_fake"


class EventRequest(BaseModel):
    description: str = ""
    participants: list = []
    fact_updates: list = []


def _apply_npc_belief_change(
    npc_id: str,
    delta: float,
    *,
    claim_id: Optional[str] = None,
    evidence: str = "NPC dialogue influence",
) -> None:
    """通过 supersede_fact 更新信念，避免 uniform mock delta。"""
    if abs(delta) < 1e-6:
        return
    npc = engine.npcs.get(npc_id)
    if not npc:
        return

    fid = claim_id
    if not fid:
        for b in npc.get("beliefs", []):
            fid = b.get("fact_id") or engine._statement_to_fact_id(b.get("statement", ""))
            if fid in engine.version_chains:
                break
    if not fid or fid not in engine.version_chains:
        return

    current = engine.get_belief(npc_id, fid)
    new_conf = max(0.0, min(1.0, current["confidence"] + delta))
    engine.supersede_fact(fid, evidence[:200], new_conf, holder=npc_id)


# ── Health / state ──

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "engine": "V13+ATMS+Hansson+Timestamp+Graph",
        "npcs": len(engine.npcs),
        "facts": len(engine.facts),
        "graph_nodes": len(engine.graph_nodes),
        "day": engine.current_day,
        "turn": engine.current_turn,
    }


@app.get("/api/state")
async def get_state():
    return engine.to_srs_state()


@app.get("/api/state/diagnostic")
async def diagnostic():
    from collections import Counter
    return {
        "version_chains": dict(engine.version_chains),
        "status_counts": dict(Counter(
            ub.status.value for ub in engine.beliefs.values())),
        "atms_assumptions": len(engine.atms.assumptions),
        "atms_nogoods": len(engine.atms.nogood_claims),
        "graph_nodes": len(engine.graph_nodes),
        "graph_edges": len(engine.graph_edges),
    }


# ── NPC dialogue ──

@app.post("/api/npc/generate")
async def generate_response(req: NPCRequest):
    """玩家→NPC：形式化 memory + 动作分析 + LLM 角色回复。"""
    npc = engine.npcs.get(req.npcId)
    if not npc:
        return {"response": "NPC not found.", "beliefChange": 0, "trustChange": 0}

    ctx = req.context or {}
    hist = ctx.get("dialogueHistory") or engine.dialogue_history
    action_result = engine.player_action(
        req.npcId, req.actionType, req.playerInput, dialogue_history=hist)

    utterance = action_result["utterance"]
    reply = generate_npc_response(
        npc,
        utterance,
        action_type=req.actionType,
        action_hint=action_result.get("actionHint", ""),
        evidence_block=action_result.get("evidence_block", ""),
        dialogue_history=hist,
        scenario=engine.scenario,
        analysis=action_result.get("analysis"),
    )

    ts = datetime.now().isoformat()
    dialogue_entry = {
        "id": f"dlg-{time.time()}",
        "speaker": npc.get("name", req.npcId),
        "text": reply,
        "playerAction": req.actionType,
        "playerInput": utterance,
        "timestamp": ts,
    }
    engine.dialogue_history.append(dialogue_entry)

    state = engine.to_srs_state()
    return {
        "response": reply,
        "text": reply,
        "effectiveUtterance": utterance,
        "dialogueEntry": dialogue_entry,
        "beliefChange": action_result.get("beliefChange", 0),
        "trustChange": action_result.get("trustChange", 0),
        "memoryUpdate": action_result.get("memoryUpdate"),
        "classification": action_result.get("classification"),
        "analysis": action_result.get("analysis"),
        "state": state,
    }


@app.post("/api/npc/dialogue")
async def npc_dialogue(req: DialogueRequest):
    """NPC↔NPC 对话。"""
    n1 = engine.npcs.get(req.npcId1, {})
    n2 = engine.npcs.get(req.npcId2, {})
    if not n1 or not n2:
        return {"dialogue": [], "beliefChanges": {}, "trustChanges": {}}

    b1 = engine.get_all_beliefs(req.npcId1)
    b2 = engine.get_all_beliefs(req.npcId2)
    c1 = {b["claim"]: b["confidence"] for b in b1}
    c2 = {b["claim"]: b["confidence"] for b in b2}
    conflicts = [
        {"claim": c, req.npcId1: round(c1[c], 3), req.npcId2: round(c2[c], 3),
         "diff": round(abs(c1[c] - c2[c]), 3)}
        for c in set(c1) & set(c2) if abs(c1[c] - c2[c]) > 0.25
    ]

    result = generate_npc_dialogue(n1, n2, conflicts=conflicts,
                                   dialogue_history=req.context.get("dialogueHistory"))
    for line in result.get("dialogue", []):
        engine.dialogue_history.append(line)

    primary_claim = conflicts[0]["claim"] if conflicts else None
    dialogue_evidence = f"NPC dialogue ({n1.get('name', req.npcId1)} ↔ {n2.get('name', req.npcId2)})"

    for nid, ch in (result.get("beliefChanges") or {}).items():
        _apply_npc_belief_change(
            nid, float(ch),
            claim_id=primary_claim,
            evidence=dialogue_evidence,
        )
    for pair, ch in (result.get("trustChanges") or {}).items():
        parts = pair.split("->")
        if len(parts) == 2:
            engine.set_trust(parts[0], parts[1],
                             engine.get_trust(parts[0], parts[1]) + float(ch))

    return {**result, "beliefConflicts": conflicts, "state": engine.to_srs_state()}


# ── Belief / memory / trust ──

@app.post("/api/belief/update")
async def update_belief(req: BeliefUpdate):
    if req.claim:
        engine.supersede_fact(req.claim, req.evidence or "manual_update",
                              req.confidence, holder=req.npcId)
    else:
        _apply_npc_belief_change(req.npcId, req.change, evidence="Manual belief adjustment")
    return {"ok": True, "state": engine.to_srs_state()}


@app.post("/api/memory/update")
async def update_memory(req: MemoryUpdate):
    fid = f"{req.source}_{req.type}_{int(time.time())}"
    vnid = engine.assert_fact(fid, req.description or req.title,
                              holder=req.source or "world", confidence=req.confidence)
    return {"version_node_id": vnid, "state": engine.to_srs_state()}


@app.post("/api/trust/update")
async def update_trust(req: TrustUpdate):
    old = engine.get_trust(req.sourceId, req.targetId)
    engine.set_trust(req.sourceId, req.targetId, max(0, min(1, old + req.change)))
    return {"source": req.sourceId, "target": req.targetId,
            "trust": engine.get_trust(req.sourceId, req.targetId),
            "state": engine.to_srs_state()}


# ── Events / time / conflict ──

@app.post("/api/event/propagate")
async def propagate_event(req: dict):
    desc = req.get("description", "player event")
    parts = req.get("participants", [])
    result = engine.process_event(desc, parts, req.get("fact_updates"))
    engine.propagation_queue = engine.propagation_queue[-10:]
    return {"propagated": True, **result, "state": engine.to_srs_state()}


@app.post("/api/event/process")
async def process_event(req: EventRequest):
    result = engine.process_event(req.description, req.participants, req.fact_updates)
    return {**result, "state": engine.to_srs_state()}


@app.post("/api/conflict/resolve")
async def resolve_conflict(req: ConflictRequest):
    result = engine.resolve_conflict(req.claimId1, req.claimId2)
    return {**result, "state": engine.to_srs_state()}


@app.post("/api/time/advance")
async def advance_time(req: TimeAdvance):
    changed = engine.advance_day(req.days)
    return {"day": engine.current_day, "turn": engine.current_turn,
            "status_changes": changed, "state": engine.to_srs_state()}


@app.post("/api/time/turn")
async def advance_turn(req: TurnAdvance):
    out = engine.advance_turn()
    return {**out, "state": engine.to_srs_state()}


@app.get("/api/time/now")
async def time_now():
    return {"timestamp": engine.now, "day": engine.current_day,
            "turn": engine.current_turn,
            "iso": datetime.fromtimestamp(engine.now).isoformat()}


# Legacy mock-compatible getters (frontend may call these)

@app.get("/api/mock/npcs")
async def mock_npcs():
    return engine.to_srs_state()["npcs"]


@app.get("/api/mock/scenarios")
async def mock_scenarios():
    return engine.scenarios


@app.get("/api/mock/memory")
async def mock_memory():
    st = engine.to_srs_state()
    return {"nodes": st["memoryNodes"], "edges": st["memoryEdges"]}


if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()
    print(f"GraphMem-ATMS SRS Backend → http://localhost:{args.port}/api/health")
    uvicorn.run("srs_api_v13:app" if args.reload else app,
                host="0.0.0.0", port=args.port, reload=args.reload)
