#!/usr/bin/env python3
"""
srs_api.py — SRS 兼容 FastAPI 后端
===================================
对接 SRS 前端 (https://github.com/ForCadia/SRS) 的 Zustand store。
所有端点映射到 SRS 的 api.js 调用，内部使用融合引擎。

启动:
  python srs_api.py                    # http://localhost:8765
  python srs_api.py --engine v2        # 使用实战版引擎
  python srs_api.py --port 8765
"""
from __future__ import annotations

import json, os, sys, time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    print("pip install fastapi uvicorn pydantic")
    raise

_HERE = Path(__file__).resolve().parent
_UPLOAD = _HERE.parent
sys.path.insert(0, str(_HERE))

from unified_belief import UnifiedBelief, BeliefTuple, BeliefStatus
from fusion_engine import FusionEngineV1, FusionEngineV2

app = FastAPI(title="GraphMem-ATMS SRS Backend", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 启动参数
ENGINE_MODE = "v2"  # v1=设计版, v2=实战高分数版
engine_v1 = FusionEngineV1()
engine_v2 = FusionEngineV2()


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic 模型
# ══════════════════════════════════════════════════════════════════════════════

class NPCRequest(BaseModel):
    npcId: str = ""
    playerInput: str = ""
    actionType: str = "talk"
    context: dict = {}

class DialogueRequest(BaseModel):
    npcId1: str = ""
    npcId2: str = ""
    context: dict = {}

class BeliefUpdate(BaseModel):
    npcId: str = ""
    change: float = 0.0

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
    reason: str = ""

class TimeAdvance(BaseModel):
    days: float = 1.0

class ConflictRequest(BaseModel):
    claimId1: str = ""
    claimId2: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# SRS 兼容端点
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": ENGINE_MODE, "beliefs": len(engine_v1.beliefs)}


# ── NPC 对话生成（替换 SRS 的 Qwen 调用）──

@app.post("/api/npc/generate")
async def generate_response(req: NPCRequest):
    """玩家→NPC 对话。"""
    context = req.context or {}
    context["sessions"] = context.get("dialogueHistory", context.get("sessions", []))

    if ENGINE_MODE == "v1":
        # V1: 纯形式化查询
        state = engine_v1.to_srs_state()
        beliefs = [engine_v1.query(req.npcId, claim)
                   for claim in engine_v1.versions.get(f"{req.npcId}_state", [])]
        return {
            "text": f"[V1 Formal] Current belief state: {json.dumps(beliefs[:3], ensure_ascii=False)}",
            "beliefChange": 0.0, "trustChange": 0.0,
            "state": state,
        }
    else:
        # V2: 检索 + 分类 + evidence block
        result = engine_v2.process_srs_action(req.npcId, req.playerInput, context)
        return {
            "text": result["response"],
            "beliefChange": result["beliefChange"],
            "trustChange": result["trustChange"],
            "memoryUpdate": result["memoryUpdate"],
            "classification": result["classification"],
            "state": result["belief_state"],
        }


@app.post("/api/npc/dialogue")
async def generate_npc_dialogue(req: DialogueRequest):
    """NPC→NPC 对话。"""
    context = req.context or {}
    npc1_state = engine_v1.query_all(req.npcId1)
    npc2_state = engine_v1.query_all(req.npcId2)

    # 找共同 claim 的信念差异
    common = set(c for c in engine_v1.versions)
    conflicts = []
    for claim in common:
        q1 = engine_v1.query(req.npcId1, claim)
        q2 = engine_v1.query(req.npcId2, claim)
        b1 = q1.get("belief", {}).get("belief", 0)
        b2 = q2.get("belief", {}).get("belief", 0)
        if abs(b1 - b2) > 0.3:
            conflicts.append({"claim": claim, "diff": abs(b1 - b2),
                              f"{req.npcId1}_belief": b1, f"{req.npcId2}_belief": b2})

    return {
        "dialogue": [
            {"speaker": req.npcId1, "text": f"[V{ENGINE_MODE}] I believe what I've witnessed."},
            {"speaker": req.npcId2, "text": f"[V{ENGINE_MODE}] My perspective differs."},
        ],
        "beliefChanges": {req.npcId1: 0.0, req.npcId2: 0.0},
        "trustChanges": {f"{req.npcId1}->{req.npcId2}": 0.0},
        "conflicts": conflicts,
    }


# ── 信念操作 ──

@app.post("/api/belief/update")
async def update_belief(req: BeliefUpdate):
    """更新 NPC 信念的置信度。"""
    changed = []
    for vnid, ub in engine_v1.beliefs.items():
        if ub.holder == req.npcId and ub.is_active:
            old_b = ub.belief_tuple.belief
            new_b = max(0.0, min(1.0, old_b + req.change))
            ub.belief_tuple = BeliefTuple(new_b, ub.belief_tuple.disbelief,
                                          max(0.0, 1.0 - new_b - ub.belief_tuple.disbelief))
            changed.append({"claim": ub.claim_id, "old": round(old_b, 3),
                            "new": round(new_b, 3)})
    return {"updated": len(changed), "changes": changed}


@app.post("/api/trust/update")
async def update_trust(req: TrustUpdate):
    """更新信任关系。SRS 中没有 Trust 数据模型，记录而不执行。"""
    return {"source": req.sourceId, "target": req.targetId, "change": req.change,
            "note": "trust update recorded"}


@app.post("/api/memory/update")
async def update_memory(req: MemoryUpdate):
    """写入记忆节点 → 创建 UnifiedBelief（对接 InMemoryGraphDatabase）。"""
    ub = engine_v1.assert_belief(
        claim=f"{req.source}_{req.type}_{int(time.time())}",
        holder=req.source,
        evidence=req.title[:200],
        confidence=req.confidence,
    )
    return {"version_node_id": ub.version_node_id, "status": ub.status.value}


# ── 事件传播 ──

@app.post("/api/event/propagate")
async def propagate_event(req: dict):
    """事件传播 → 触发 ATMS label 重算 + 时间衰减 + 状态跃迁。"""
    changed = engine_v1.advance_time(days=0.01)  # 微推进以触发状态检查
    return {"propagated": True, "status_changes": changed,
            "active_beliefs": sum(1 for ub in engine_v1.beliefs.values() if ub.is_active)}


# ── 冲突解决（替换 SRS 硬编码动画）──

@app.post("/api/conflict/resolve")
async def resolve_conflict(req: ConflictRequest):
    """真实 Hansson incision 替换 SRS 的 5 步动画。"""
    vnids1 = engine_v1.versions.get(req.claimId1, [])
    vnids2 = engine_v1.versions.get(req.claimId2, [])

    if not vnids1 or not vnids2:
        raise HTTPException(404, "claim not found")

    ub1 = engine_v1.beliefs[vnids1[-1]]
    ub2 = engine_v1.beliefs[vnids2[-1]]

    if not engine_v1.detect_conflict(ub1, ub2):
        return {"conflict": False, "message": "no conflict detected"}

    result = engine_v1.resolve_conflict(ub1, ub2)

    # SRS 5-step 格式
    return {
        "conflict": True,
        "steps": [
            {"step": 1, "name": "Receive", "detail": f"New evidence: {ub2.evidence_id}"},
            {"step": 2, "name": "Trust Check",
             "detail": f"Source credibility: {ub2.credibility:.2f} vs {ub1.credibility:.2f}"},
            {"step": 3, "name": "Conflict Detection",
             "detail": f"nogood: {ub1.claim_id} vs {ub2.claim_id}"},
            {"step": 4, "name": "Belief Revision",
             "detail": " → ".join(result["trace"])},
            {"step": 5, "name": "Consensus",
             "detail": f"Result: {ub1.claim_id} → {ub1.status.value}"},
        ],
        "incision_trace": result["trace"],
        "result": result["action"],
    }


# ── 时间 ──

@app.post("/api/time/advance")
async def advance_time(req: TimeAdvance):
    """推进模拟时间。"""
    changed = engine_v1.advance_time(req.days)
    return {"now": datetime.fromtimestamp(engine_v1.now).isoformat(),
            "days_advanced": req.days, "status_changes": changed}


@app.get("/api/time/now")
async def get_time():
    return {"timestamp": engine_v1.now, "iso": datetime.fromtimestamp(engine_v1.now).isoformat()}


# ── 全状态导出 ──

@app.get("/api/state")
async def get_full_state():
    """导出完整信念状态（SRS 前端同步）。"""
    return engine_v1.to_srs_state() if ENGINE_MODE == "v1" else engine_v2.to_srs_state()


@app.get("/api/state/diagnostic")
async def get_diagnostic():
    """诊断视图：版本链、incision traces、置信度分布。"""
    result = {"beliefs": {}, "version_chains": {}, "status_counts": {}}
    for vnid, ub in engine_v1.beliefs.items():
        result["beliefs"][vnid] = ub.to_dict()
        result["beliefs"][vnid]["incision_trace"] = ub.incision_trace
    for claim, vnids in engine_v1.versions.items():
        result["version_chains"][claim] = vnids
    from collections import Counter
    result["status_counts"] = dict(Counter(
        ub.status.value for ub in engine_v1.beliefs.values()))
    result["engine_mode"] = ENGINE_MODE
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="v2", choices=["v1", "v2"])
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    ENGINE_MODE = args.engine
    print(f"GraphMem-ATMS SRS Backend (engine={ENGINE_MODE})")
    print(f"  http://localhost:{args.port}/api/health")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
