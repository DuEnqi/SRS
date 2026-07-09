#!/usr/bin/env python3
"""
fusion_engine.py — 融合信念修正引擎
=====================================
V1（设计版）: ATMS + Hansson + 版本链 + 时间戳的统一信念生命周期
V2（实战版）: v1 简单 retriever + GraphMemATMS 裁决 + v3 置信度 + evidence-block prompt

两个版本共享同一个 UnifiedBelief 数据模型和 SRS API 接口。
"""
from __future__ import annotations

import json, os, re, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# ── 加载 unified_stale ──
_UPLOAD = Path(__file__).resolve().parent.parent
if str(_UPLOAD) not in sys.path:
    sys.path.insert(0, str(_UPLOAD))
import unified_stale as _U

from unified_belief import (
    UnifiedBelief, BeliefTuple, BeliefStatus,
    Justification, JustificationOp,
)

CORE = _U.CORE
V9 = _U.V9


# ══════════════════════════════════════════════════════════════════════════════
# V1 — 新融合引擎（设计版：完整 blend lifecycle）
# ══════════════════════════════════════════════════════════════════════════════

class FusionEngineV1:
    """
    设计版融合引擎。

    信念生命周期:
      1. assert_belief() → Fact 创建, ATMS label 注册
      2. receive_evidence() → 可能触发 incision
      3. advance_time(days) → 时间衰减, 状态自动跃迁
      4. query(npc, claim, now) → 时间感知的 label 计算
    """

    def __init__(self):
        self.beliefs: Dict[str, UnifiedBelief] = {}
        self.atms_assumptions: Dict[str, dict] = {}
        self.versions: Dict[str, List[str]] = defaultdict(list)  # claim_id → [vnid...]
        self.now: float = datetime.now().timestamp()

    # ── 信念生命周期 ──────────────────────────────────────────────────

    def assert_belief(self, claim: str, holder: str = "world",
                      evidence: str = "", confidence: float = 0.8,
                      valid_from: float = 0.0, valid_to: Optional[float] = None,
                      justifications: Optional[List[Justification]] = None,
                      nogood_with: Optional[Set[str]] = None) -> UnifiedBelief:
        """创建一个新信念（或返回已有版本）。"""
        now_ts = datetime.now().timestamp()
        if valid_from == 0.0:
            valid_from = now_ts

        ub = UnifiedBelief(
            claim_id=claim, evidence_id=evidence, holder=holder,
            belief_tuple=BeliefTuple.from_confidence(confidence),
            asserted_at=datetime.now().isoformat(),
            valid_from=valid_from, valid_to=valid_to,
            justifications=justifications or [],
            nogood_with=nogood_with or set(),
            credibility=confidence,
        )
        self.beliefs[ub.version_node_id] = ub
        self.versions[claim].append(ub.version_node_id)
        return ub

    def supersede(self, claim: str, new_evidence: str,
                  new_confidence: float, holder: str = "world") -> UnifiedBelief:
        """创建新版本，旧版本标记为 SUPERSEDED。"""
        old_vnids = self.versions.get(claim, [])
        for vnid in old_vnids:
            old = self.beliefs.get(vnid)
            if old and old.is_active:
                old.is_active = False
                old.superseded_by = f"{claim}_v{len(old_vnids) + 1}"
                old.status = BeliefStatus.SUPERSEDED

        new_version = len(old_vnids) + 1
        ub = self.assert_belief(claim=claim, holder=holder,
                                evidence=new_evidence, confidence=new_confidence)
        ub.version = new_version
        ub.version_node_id = f"{claim}_v{new_version}"
        self.beliefs.pop(f"{claim}_v1", None)  # replace old key
        self.beliefs[ub.version_node_id] = ub
        self.versions[claim].append(ub.version_node_id)
        return ub

    # ── 时间驱动 ─────────────────────────────────────────────────────

    def advance_time(self, days: float) -> List[dict]:
        """推进时间，返回所有状态发生变化的信念。"""
        self.now += days * 86400.0
        changed = []
        for vnid, ub in self.beliefs.items():
            if not ub.is_active:
                continue
            old_status = ub.status
            new_status = ub.compute_status(self.now)
            if new_status != old_status:
                ub.status = new_status
                ub.last_updated = datetime.now().isoformat()
                changed.append({
                    "claim": ub.claim_id, "holder": ub.holder,
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "temporal_weight": ub.temporal_weight(self.now),
                })
        return changed

    # ── 冲突检测与修正 ────────────────────────────────────────────────

    def detect_conflict(self, ub1: UnifiedBelief, ub2: UnifiedBelief) -> bool:
        """检测两条信念是否冲突（nogood 或 active 版本矛盾）。"""
        if ub1.claim_id in ub2.nogood_with or ub2.claim_id in ub1.nogood_with:
            return True
        if ub1.claim_id == ub2.claim_id and ub1.version != ub2.version:
            return ub1.is_active and ub2.is_active
        return False

    def resolve_conflict(self, ub_old: UnifiedBelief, ub_new: UnifiedBelief) -> dict:
        """
        Hansson 4-step incision（时间感知版）。
        返回 incise trace。
        """
        trace = []
        now = self.now

        # Step 1: 切 defeasible + 时间陈旧
        if ub_old.justifications:
            for j in ub_old.justifications:
                if j.operator == JustificationOp.DEFEASIBLE:
                    elapsed = (now - ub_old.valid_from) / 86400.0
                    if elapsed > 60 or ub_old.credibility < 0.4:
                        ub_old.status = BeliefStatus.REFUTED
                        ub_old.is_active = False
                        trace.append(f"STEP1: cut defeasible {j.jid} " +
                                     f"(elapsed={elapsed:.0f}d, cred={ub_old.credibility:.2f})")
                        return {"action": "refute_old", "trace": trace, "refuted": ub_old.claim_id}

        # Step 2: 切低可信度 + 时间新鲜度
        if ub_new.credibility < ub_old.credibility * 0.5:
            elapsed = (now - ub_new.asserted_at if isinstance(ub_new.asserted_at, float)
                       else (now - datetime.fromisoformat(ub_new.asserted_at).timestamp())) / 86400.0
            if elapsed < 7:  # 最近7天的新事件
                ub_old.status = BeliefStatus.WEAK
                trace.append(f"STEP2: low-cred new event is recent ({elapsed:.0f}d), weaken old")
                return {"action": "weaken_old", "trace": trace, "weakened": ub_old.claim_id}

        # Step 3: 保留 + 时间标记
        ub_old.is_active = False
        ub_old.status = BeliefStatus.SUPERSEDED
        ub_old.last_updated = datetime.now().isoformat()
        trace.append(f"STEP3: old version superseded, retained as historical")

        # Step 4: 新版本激活
        ub_new.is_active = True
        ub_new.status = BeliefStatus.ACTIVE
        trace.append(f"STEP4: new version {ub_new.version_node_id} activated")

        return {"action": "supersede", "trace": trace, "old": ub_old.version_node_id,
                "new": ub_new.version_node_id}

    # ── 查询 ──────────────────────────────────────────────────────────

    def query(self, holder: str, claim: str, now: Optional[float] = None) -> dict:
        """查询某个 holder 对 claim 的当前信念状态。"""
        now = now or self.now
        vnids = self.versions.get(claim, [])
        active = None
        for vnid in reversed(vnids):  # 最新版本优先
            ub = self.beliefs.get(vnid)
            if ub and ub.is_active:
                active = ub
                break

        if active is None:
            return {"claim": claim, "status": "unknown", "belief": BeliefTuple.uncertain().to_dict()}

        tw = active.temporal_weight(now)
        discounted = BeliefTuple(
            active.belief_tuple.belief * tw,
            active.belief_tuple.disbelief * tw,
            max(0.0, 1.0 - (active.belief_tuple.belief + active.belief_tuple.disbelief) * tw),
        )

        return {
            "claim": claim,
            "holder": active.holder,
            "belief": discounted.to_dict(),
            "status": active.status.value,
            "version": active.version,
            "confidence_level": active.confidence_level,
            "temporal_weight": round(tw, 4),
            "version_chain": [f"v{i+1}" for i in range(len(vnids))],
            "incision_trace": active.incision_trace,
        }

    def query_all(self, holder: str) -> List[dict]:
        """查询 holder 所知的所有信念。"""
        return [self.query(holder, claim) for claim in self.versions]

    # ── SRS 适配输出 ─────────────────────────────────────────────────

    def to_srs_state(self) -> dict:
        """导出为 SRS 前端兼容的 Zustand store 片段。"""
        beliefs = []
        memory_nodes = []
        trust_edges = []
        for vnid, ub in self.beliefs.items():
            if ub.is_active:
                beliefs.append(ub.to_srs_belief())
            memory_nodes.append(ub.to_srs_memory_node())
            trust_edges.append({
                "source": ub.holder, "target": ub.claim_id,
                "trust": ub.credibility,
                "reason": f"Evidence: {ub.evidence_id}",
            })

        return {
            "beliefs": beliefs,
            "memoryNodes": memory_nodes,
            "memoryEdges": [],
            "trustNetwork": trust_edges,
            "scenarioTime": datetime.fromtimestamp(self.now).isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# V2 — 实战版引擎（v1 retriever + GraphMemATMS 裁决 + v3 置信度 + evidence-block）
# ══════════════════════════════════════════════════════════════════════════════

# 复用 v1 的简单 slot lexicon（来源：新版融合/adapter v1）
_SLOT_LEXICON = {
    "location": {"live", "living", "lives", "based", "home", "city", "moved", "relocated",
        "settled", "seattle", "denver", "austin", "chicago", "portland", "boston",
        "miami", "coast", "town", "neighborhood", "downtown"},
    "occupation": {"job", "work", "works", "working", "teacher", "teach", "teaching",
        "accountant", "nurse", "nursing", "developer", "engineer", "manager",
        "company", "role", "career", "employed", "retired", "hired", "promoted"},
    "diet": {"diet", "vegetarian", "vegan", "meat", "fish", "poultry", "keto",
        "omnivore", "eat", "eating", "food", "meals", "cooking"},
    "health": {"health", "injury", "injured", "broke", "fracture", "pain", "hospital",
        "doctor", "surgery", "recovery", "healing", "cast", "crutches"},
    "commute": {"commute", "commuting", "bike", "biking", "bicycle", "cycle", "cycling",
        "ride", "riding", "walk", "walking", "drive", "driving", "train", "bus"},
}

_STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
         "have", "has", "had", "do", "does", "did", "will", "would", "could",
         "should", "may", "might", "can", "shall", "to", "of", "in", "for",
         "on", "with", "at", "by", "from", "as", "into", "through", "during",
         "and", "but", "or", "nor", "not", "so", "yet", "it", "its", "my",
         "your", "his", "her", "our", "their", "me", "you", "him", "us", "them",
         "i", "we", "he", "she", "they", "that", "this", "these", "those",
         "about", "up", "out", "if", "then", "now", "here", "there", "when"}


def _tokens(text: str) -> set:
    return {w.lower() for w in re.findall(r"[a-z]+", (text or "").lower())
            if w not in _STOP and len(w) > 2}


def _infer_slot(text: str) -> str:
    toks = set(re.findall(r"[a-z]+", (text or "").lower()))
    best, best_s = "", 0
    for attr, lex in _SLOT_LEXICON.items():
        s = len(toks & lex)
        if s > best_s:
            best_s, best = s, attr
    return best


_CHANGE_RE = re.compile(
    r"\b(?:no longer|not (?:still |anymore|valid)|changed|switched|moved|relocated|"
    r"shifted|updated|replaced|quit|left|stopped|resigned|retired|passed away|died|"
    r"got (?:a |an )?new|recovering|healed|back to|resumed|restored|reverted|"
    r"instead|now lives|now works|now eat|started|became|promoted|divorced)\b",
    re.IGNORECASE,
)


class FusionEngineV2:
    """
    实战版引擎（高分数配置）。

    组合:
      - v1 简单 slot-aware retriever (best new_hit ~60%)
      - GraphMemATMS 裁决 (best SR 95.5%)
      - v3 置信度校准 (FIX-3 only)
      - evidence-block prompt 注入（非命令式）
    """

    def __init__(self):
        self.engine_v1 = FusionEngineV1()
        self.confidences: Dict[str, str] = {}  # claim_id → HIGH|MEDIUM|LOW

    def retrieve_old_new(self, sessions: List[dict]) -> dict:
        """v1 简单 retriever: 前70% old, 后30% new, slot lexicon 打分。"""
        turns = []
        for si, sess in enumerate(sessions):
            items = sess.get("turns", sess) if isinstance(sess, dict) else sess
            if isinstance(items, list):
                for ti, turn in enumerate(items):
                    content = turn.get("content", str(turn)) if isinstance(turn, dict) else str(turn)
                    turns.append({"content": content, "session": si, "turn": ti})

        if len(turns) < 2:
            return {"old": "", "new": "", "slot": ""}

        n = len(turns)
        split = max(1, int(n * 0.7))
        new_cands = turns[split:]
        best_new, best_score = new_cands[-1], -1
        for t in new_cands:
            for attr, lex in _SLOT_LEXICON.items():
                s = len(set(re.findall(r"[a-z]+", t["content"].lower())) & lex)
                if s > best_score:
                    best_score, best_new = s, t
        slot = _infer_slot(best_new["content"])

        old_cands = turns[:split]
        new_toks = _tokens(best_new["content"])
        scored = [(len(_tokens(t["content"]) & _SLOT_LEXICON.get(slot, set())) * 3 +
                   len(_tokens(t["content"]) & new_toks), t) for t in old_cands]
        scored.sort(key=lambda x: (x[0], x[1]["session"]), reverse=True)
        best_old = scored[0][1] if scored and scored[0][0] > 0 else old_cands[0]

        return {"old": best_old["content"], "new": best_new["content"], "slot": slot}

    def classify(self, old_text: str, new_text: str, slot: str) -> dict:
        """简化的冲突判断：change cue ∈ new_text → invalidate。"""
        change = bool(_CHANGE_RE.search((new_text or "").lower()))
        diff_tokens = bool(_tokens(new_text) - _tokens(old_text))
        same_slot = _infer_slot(old_text) == slot or _infer_slot(new_text) == slot

        should_invalidate = (change or diff_tokens) and same_slot
        # v3 置信度校准
        if not old_text or not new_text:
            confidence = "LOW"
        elif change and diff_tokens:
            confidence = "HIGH"
        elif change or diff_tokens:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {"invalidate": should_invalidate, "confidence": confidence,
                "slot": slot, "has_change_cue": change}

    def build_evidence_block(self, old_text: str, new_text: str, classification: dict) -> str:
        """Evidence-block prompt 注入（非命令式）。"""
        conf = classification["confidence"]
        if conf == "HIGH":
            prefix = "The system has high confidence that the user's situation has changed:"
        elif conf == "MEDIUM":
            prefix = "There is some evidence that the user's situation may have changed:"
        else:
            prefix = "It is unclear whether the user's situation has changed; please verify:"

        return (
            f"[Memory system evidence — {conf} confidence]\n"
            f"{prefix}\n"
            f"  Earlier: \"{old_text[:200]}\"\n"
            f"  More recent: \"{new_text[:200]}\"\n"
            f"Based on this, determine whether the earlier information is still current, "
            f"and answer accordingly. Cite evidence from the conversation history."
        )

    def process_srs_action(self, npc_id: str, action: str, context: dict) -> dict:
        """处理 SRS 前端发来的玩家动作。"""
        sessions = context.get("sessions", context.get("dialogueHistory", []))
        query = context.get("query", action)

        # v1 检索
        retrieved = self.retrieve_old_new(sessions)

        # 分类
        classification = self.classify(
            retrieved["old"], retrieved["new"], retrieved["slot"])

        # 信念版本管理
        if classification["invalidate"] and retrieved["slot"]:
            claim = f"{npc_id}_{retrieved['slot']}_state"
            old_belief = self.engine_v1.assert_belief(
                claim=claim, holder=npc_id,
                evidence=retrieved["old"][:100], confidence=0.7)
            new_belief = self.engine_v1.supersede(
                claim=claim, holder=npc_id,
                new_evidence=retrieved["new"][:100], new_confidence=0.8)
            self.engine_v1.resolve_conflict(old_belief, new_belief)

        # evidence block
        evidence = self.build_evidence_block(
            retrieved["old"], retrieved["new"], classification)

        return {
            "response": f"[GraphMem-ATMS V2] {evidence}",
            "beliefChange": 0.1 if classification["invalidate"] else 0.0,
            "trustChange": 0.0,
            "memoryUpdate": retrieved,
            "classification": classification,
            "belief_state": self.engine_v1.to_srs_state(),
        }

    def to_srs_state(self) -> dict:
        return self.engine_v1.to_srs_state()
