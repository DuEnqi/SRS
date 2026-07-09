#!/usr/bin/env python3
"""
game_engine_v13.py — V13 游戏引擎 + GraphMem-ATMS + 组友时间戳
================================================================
v6  ATMSKernelV2 + Hansson incision
v7-v9  classify_fine_v9
v13 GraphMemory 节点/边 + ConsensusEngine
组友 Fact / NPCKnows / NPCTrusts + SubjectiveLogicEngine

Phase 1 (per-NPC belief model):
  - npc_beliefs[(npc_id, claim_id)] 替代全局 version_chains
  - get_belief() NPCKnows gate
  - get_all_beliefs() 仅返回有 knowledge 的 claim
  - Consensus propagate/compute 按 NPC 独立写入与统计
Phase 2:
  - npc_private_memory 私有记忆
  - engine_checkpoint LLM failure rollback
  - Scenario CRUD API (srs_api_v13)
Phase 3:
  - stateVersion + idempotency cache
  - action_handler 无场景硬编码
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_HERE = Path(__file__).resolve().parent
_UPLOAD = _HERE.parent
_GRAPHMEM = _UPLOAD / "补充内容" / "STALE_GraphMem_ATMS"
_MOCK = _UPLOAD / "tmp_SRS" / "src" / "mock"
for p in [str(_UPLOAD), str(_GRAPHMEM), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import unified_stale as _U  # noqa: E402

CORE = _U.CORE
V9 = _U._BUILT["stale_experiments_v9ext"]

from npc.memory.core_types import Fact, FactScope, NPCKnows, NPCTrusts  # noqa: E402
from npc.memory.belief_engine import SubjectiveLogicEngine  # noqa: E402
from unified_belief import (  # noqa: E402
    UnifiedBelief,
    BeliefStatus,
    BeliefTuple,
    JustificationOp,
)

ATMS = CORE.ATMSKernelV2
IncisionStep = CORE.IncisionStep

from action_handler import (  # noqa: E402
    ACTION_NPC_HINT,
    ACTION_TYPES,
    analyze_player_action,
    action_belief_trust_deltas,
    effective_utterance,
)
from engine_checkpoint import checkpoint as _checkpoint, rollback as _rollback  # noqa: E402

_CHANGE_RE = re.compile(
    r"\b(?:no longer|not (?:still |anymore|valid)|changed|switched|moved|relocated|"
    r"shifted|updated|replaced|quit|left|stopped|resigned|retired|passed away|died|"
    r"got (?:a |an )?new|recovering|healed|back to|resumed|restored|reverted|"
    r"instead|now lives|now works|fake|fraud|impostor|deserter|revealed)\b",
    re.IGNORECASE,
)


def hansson_incise(old_belief: UnifiedBelief, new_belief: UnifiedBelief,
                   now: float, trust_network: dict) -> dict:
    trace: List[str] = []
    elapsed_old = (now - old_belief.valid_from) / 86400.0 if old_belief.valid_from > 0 else 0
    if any(j.operator == JustificationOp.DEFEASIBLE for j in old_belief.justifications):
        if elapsed_old > 60 or old_belief.credibility < 0.4:
            old_belief.status = BeliefStatus.REFUTED
            old_belief.is_active = False
            trace.append(f"STEP1: refute {old_belief.claim_id} (stale/defeasible)")
            return {"action": "refute_old", "trace": trace, "refuted": old_belief.claim_id}

    new_elapsed = (now - new_belief.valid_from) / 86400.0
    if new_belief.credibility < old_belief.credibility * 0.5 and new_elapsed < 7:
        old_belief.status = BeliefStatus.WEAK
        trace.append(f"STEP2: weaken {old_belief.claim_id}")
        return {"action": "weaken_old", "trace": trace, "weakened": old_belief.claim_id}

    if len(old_belief.justifications) > 1:
        trace.append(f"STEP3: alt support for {old_belief.claim_id}")
    else:
        trace.append(f"STEP3: no alt support for {old_belief.claim_id}")

    old_belief.is_active = False
    old_belief.status = BeliefStatus.SUPERSEDED
    old_belief.last_updated = datetime.now().isoformat()
    trace.append(f"STEP4: superseded by {new_belief.version_node_id}")

    new_belief.is_active = True
    new_belief.status = BeliefStatus.ACTIVE
    return {"action": "supersede", "trace": trace,
            "old": old_belief.version_node_id, "new": new_belief.version_node_id}


def classify_game_event(old_text: str, new_text: str, claim_id: str,
                        confidence_delta: float) -> dict:
    try:
        rec = {"M_old": old_text or "no prior evidence", "M_new": new_text,
               "conflict_type": "", "graph_hint": {"attribute_b": claim_id}}
        op_type = V9.classify_fine_v9(rec)
        if op_type == "NO_EFFECT" and abs(confidence_delta) > 0.3:
            op_type = "UPDATE"
    except Exception:
        op_type = "UPDATE" if abs(confidence_delta) > 0.1 else "NO_EFFECT"
    return {"op_type": op_type,
            "is_invalidation": op_type in ("UPDATE", "REVERT"),
            "is_keep": op_type in ("REINFORCE", "SUPPLEMENT", "NO_EFFECT")}


class GraphMemoryStatus(str, Enum):
    ACTIVE = "active"
    WEAK = "weak"
    STALE = "stale"
    SUPERSEDED = "superseded"
    HISTORICAL = "historical"
    REFUTED = "refuted"

    @classmethod
    def compute(cls, ub: UnifiedBelief, now: float) -> "GraphMemoryStatus":
        if ub.status == BeliefStatus.REFUTED:
            return cls.REFUTED
        if ub.superseded_by:
            return cls.SUPERSEDED
        try:
            last = datetime.fromisoformat(ub.last_updated).timestamp()
        except Exception:
            last = ub.valid_from if ub.valid_from > 0 else now
        elapsed = (now - last) / 86400.0
        if elapsed > 180:
            return cls.HISTORICAL
        if elapsed > 90:
            return cls.STALE
        if elapsed > 30:
            return cls.WEAK
        return cls.ACTIVE


class ConsensusEngine:
    def __init__(self, engine: "GameEngineV13"):
        self.eng = engine
        self.consensus_history: List[dict] = []

    def propagate(self, claim: str, new_evidence: str, source_npc: str,
                  confidence: float, target_npcs: Optional[List[str]] = None) -> dict:
        targets = target_npcs or [n for n in self.eng.npcs if n != source_npc]
        results = {"claim": claim, "source": source_npc, "propagations": []}
        for tnpc in targets:
            trust = self.eng.get_trust(tnpc, source_npc)
            if trust < 0.15:
                results["propagations"].append({
                    "target": tnpc, "accepted": False,
                    "reason": f"premise_resistance trust={trust:.2f}",
                })
                continue
            had_knowledge = self.eng._has_knowledge(tnpc, claim)
            current = self.eng.get_belief(tnpc, claim)
            origins = {e.npc_id_from for k, e in self.eng.trust_edges.items()
                       if e.npc_id_to == tnpc and e.weight >= 0.3}
            delta = confidence * trust * 0.5
            if len(origins) <= 1 and source_npc != "world":
                delta *= 0.3
            old_conf = current["confidence"] if had_knowledge else 0.0
            new_conf = max(0.0, min(1.0, old_conf + delta))
            if had_knowledge:
                new_conf = old_conf * 0.7 + new_conf * 0.3
            accepted = abs(new_conf - old_conf) > 0.02
            if accepted:
                # Per-NPC write: each target gets its own version chain
                self.eng.supersede_fact(
                    claim, new_evidence, new_conf, holder=tnpc,
                    source_npc=source_npc, direct=False,
                )
            results["propagations"].append({
                "target": tnpc, "accepted": accepted,
                "trust": round(trust, 3), "old_conf": round(old_conf, 3),
                "new_conf": round(new_conf, 3),
                "had_knowledge": had_knowledge,
            })
        self.consensus_history.append(results)
        return results

    def compute_consensus(self, claim: str) -> dict:
        holders = [n for n in self.eng.npcs if self.eng._has_knowledge(n, claim)]
        if not holders:
            return {"consensus": 0.0, "variance": 0.0, "convergence": 0.0, "holders": 0}
        confs = [self.eng.get_belief(n, claim)["confidence"] for n in holders]
        mean = sum(confs) / len(confs)
        var = sum((c - mean) ** 2 for c in confs) / len(confs)
        return {
            "consensus": round(mean, 4), "variance": round(var, 4),
            "convergence": round(max(0, 1 - math.sqrt(var)), 4),
            "holders": len(holders),
        }


class GameEngineV13:
    """Greyford 多 NPC 引擎 — 单一 source of truth。"""

    def __init__(self, scenario_name: str = "Fake Knight Incident"):
        self.atms = ATMS()
        self.atms.now = time.time()
        self.consensus = ConsensusEngine(self)
        self.sl_engine = SubjectiveLogicEngine()

        self.facts: Dict[str, Fact] = {}
        self.knows_edges: Dict[str, NPCKnows] = {}
        self.trust_edges: Dict[str, NPCTrusts] = {}
        # Per-NPC version chains: (npc_id, claim_id) -> [vnid, ...]
        self.npc_beliefs: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.beliefs: Dict[str, UnifiedBelief] = {}
        self.graph_nodes: List[dict] = []
        self.graph_edges: List[dict] = []

        self.npcs: Dict[str, dict] = {}
        self.scenario: dict = {}
        self.scenarios: List[dict] = []
        self.now = datetime.now().timestamp()
        self.scenario_name = scenario_name
        self.current_day = 3
        self.current_turn = 8
        self.events_log: List[dict] = []
        self.activity_feed: List[dict] = []
        self.dialogue_history: List[dict] = []
        self.pending_memory_updates: List[dict] = []
        self.propagation_queue: List[dict] = []
        self.npc_private_memory: Dict[str, List[dict]] = defaultdict(list)
        self.state_version: int = 0
        self._idempotency_cache: Dict[str, dict] = {}

        self._load_mock_data()
        self._seed_scenario_facts()

    def _load_mock_data(self) -> None:
        npc_path = _MOCK / "npc.json"
        scen_path = _MOCK / "scenario.json"
        mem_path = _MOCK / "memory.json"
        if npc_path.exists():
            agents = json.loads(npc_path.read_text(encoding="utf-8")).get("agents", [])
            for a in agents:
                self.npcs[a["id"]] = dict(a)
        if scen_path.exists():
            self.scenarios = json.loads(scen_path.read_text(encoding="utf-8")).get("scenarios", [])
            self.scenario = self.scenarios[0] if self.scenarios else {}
            self.current_day = int(self.scenario.get("currentDay", 3))
        if mem_path.exists():
            mem = json.loads(mem_path.read_text(encoding="utf-8"))
            self.graph_nodes = list(mem.get("nodes", []))
            self.graph_edges = list(mem.get("edges", []))
        self._ensure_player_trust_edges()

    def _ensure_player_trust_edges(self) -> None:
        """确保每个 NPC 对 Player 有显式信任边（mock 缺省时补 0.5）。"""
        for npc_id, npc in self.npcs.items():
            network = npc.setdefault("trustNetwork", [])
            has_player = any(
                str(t.get("target", "")).lower() == "player" for t in network
            )
            if not has_player:
                network.append({
                    "target": "Player",
                    "trust": 0.5,
                    "reason": "Default acquaintance",
                })
            trust = next(
                (float(t["trust"]) for t in network
                 if str(t.get("target", "")).lower() == "player"),
                0.5,
            )
            self.set_trust(npc_id, "Player", trust)

    def _add_activity(self, message: str, typ: str = "system") -> None:
        now = datetime.now()
        t = f"{now.hour:02d}:{now.minute:02d}"
        self.activity_feed.insert(0, {"time": t, "message": message, "type": typ})
        self.activity_feed = self.activity_feed[:30]

    def _graph_add(self, node: dict) -> str:
        nid = node.get("id") or f"g{len(self.graph_nodes)+1}"
        node = {**node, "id": nid}
        self.graph_nodes.append(node)
        self.pending_memory_updates.append(node)
        return nid

    def _graph_link(self, source: str, target: str, label: str) -> None:
        self.graph_edges.append({"source": source, "target": target, "label": label})

    def _atms_register(self, claim: str, evidence: str, holder: str,
                       confidence: float, valid_from: float) -> str:
        self.atms.now = self.now
        return self.atms.assert_evidence(
            claim, evidence, holder=holder, trust=confidence, valid_from=valid_from)

    def _statement_to_fact_id(self, statement: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", (statement or "").lower()).strip("_")
        return s[:48] or "unknown_claim"

    # ── Per-NPC belief index (Phase 1) ──

    def _npc_claim_key(self, npc_id: str, claim_id: str) -> Tuple[str, str]:
        return (npc_id, claim_id)

    def _get_npc_chain(self, npc_id: str, claim_id: str) -> List[str]:
        return list(self.npc_beliefs.get(self._npc_claim_key(npc_id, claim_id), []))

    def _has_knowledge(self, npc_id: str, claim_id: str) -> bool:
        """NPCKnows gate: NPC must hold a non-stale KNOWS edge to an active version."""
        chain = self._get_npc_chain(npc_id, claim_id)
        if not chain:
            return False
        for vnid in reversed(chain):
            edge = self.knows_edges.get(f"{npc_id}->{vnid}")
            if not edge or edge.is_stale:
                continue
            fact = self.facts.get(vnid)
            ub = self.beliefs.get(vnid)
            if fact and ub and fact.is_active and ub.is_active:
                return True
        return False

    def _known_claims(self, npc_id: str) -> List[str]:
        return [
            claim for (nid, claim) in self.npc_beliefs
            if nid == npc_id and self._has_knowledge(npc_id, claim)
        ]

    def _all_known_claim_ids(self) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for (nid, claim) in self.npc_beliefs:
            if claim not in seen and self._has_knowledge(nid, claim):
                seen.add(claim)
                out.append(claim)
        return out

    def _mark_knows_stale(self, npc_id: str, vnid: str) -> None:
        key = f"{npc_id}->{vnid}"
        if key in self.knows_edges:
            self.knows_edges[key].is_stale = True

    @property
    def version_chains(self) -> Dict[str, List[str]]:
        """Diagnostic merge view (legacy); canonical store is npc_beliefs."""
        merged: Dict[str, List[str]] = defaultdict(list)
        for (_nid, claim), chain in self.npc_beliefs.items():
            for vnid in chain:
                if vnid not in merged[claim]:
                    merged[claim].append(vnid)
        return dict(merged)

    def bump_state_version(self) -> int:
        self.state_version += 1
        return self.state_version

    def checkpoint(self) -> dict:
        return _checkpoint(self)

    def rollback(self, snap: dict) -> None:
        _rollback(self, snap)

    def get_idempotent(self, key: str) -> Optional[dict]:
        return self._idempotency_cache.get(key)

    def store_idempotent(self, key: str, payload: dict) -> None:
        if not key:
            return
        self._idempotency_cache[key] = payload
        if len(self._idempotency_cache) > 200:
            oldest = list(self._idempotency_cache.keys())[:50]
            for k in oldest:
                self._idempotency_cache.pop(k, None)

    def add_private_memory(self, npc_id: str, entry: dict) -> None:
        """NPC 私有记忆（不写入全局 graph，仅 holder 可见）。"""
        mem = {
            **entry,
            "id": entry.get("id") or f"pm-{npc_id}-{time.time()}",
            "npc_id": npc_id,
            "scope": "private",
            "timestamp": entry.get("timestamp") or datetime.now().isoformat(),
        }
        self.npc_private_memory[npc_id].append(mem)
        self.npc_private_memory[npc_id] = self.npc_private_memory[npc_id][-50:]
        self.bump_state_version()

    def get_private_memory(self, npc_id: str) -> List[dict]:
        return list(self.npc_private_memory.get(npc_id, []))

    def get_known_props_for_npc(self, npc_id: str) -> Dict[str, str]:
        props: Dict[str, str] = {}
        for fid in self._known_claims(npc_id):
            props[fid] = fid.replace("_", " ")
        return props

    def get_tracked_claims(self) -> List[str]:
        sc = self.scenario or {}
        tracked = sc.get("trackedClaims") or sc.get("conflictClaims")
        if isinstance(tracked, list) and tracked:
            return [c for c in tracked if any(self._has_knowledge(n, c) for n in self.npcs)]
        return self._all_known_claim_ids()[:8]

    def get_conflict_pair(self) -> Tuple[str, str]:
        sc = self.scenario or {}
        pair = sc.get("conflictPair") or sc.get("conflictClaims")
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            return str(pair[0]), str(pair[1])
        claims = self._all_known_claim_ids()
        if len(claims) >= 2:
            return claims[0], claims[1]
        return (claims[0] if claims else "claim_a",
                claims[1] if len(claims) > 1 else "claim_b")

    # ── Scenario CRUD (Phase 2) ──

    def list_scenarios(self) -> List[dict]:
        return list(self.scenarios)

    def get_scenario(self, scenario_id: str) -> Optional[dict]:
        return next((s for s in self.scenarios if s.get("id") == scenario_id), None)

    def create_scenario(self, data: dict) -> dict:
        sid = data.get("id") or f"scenario-{int(time.time())}"
        scen = {**data, "id": sid}
        self.scenarios.append(scen)
        self.bump_state_version()
        self._add_activity(f"Scenario created: {scen.get('name', sid)}", "system")
        return scen

    def update_scenario(self, scenario_id: str, data: dict) -> dict:
        for i, s in enumerate(self.scenarios):
            if s.get("id") == scenario_id:
                updated = {**s, **data, "id": scenario_id}
                self.scenarios[i] = updated
                if (self.scenario or {}).get("id") == scenario_id:
                    self.scenario = updated
                    self.scenario_name = updated.get("name", self.scenario_name)
                self.bump_state_version()
                return updated
        raise KeyError(f"scenario not found: {scenario_id}")

    def delete_scenario(self, scenario_id: str) -> bool:
        before = len(self.scenarios)
        self.scenarios = [s for s in self.scenarios if s.get("id") != scenario_id]
        if len(self.scenarios) == before:
            return False
        if (self.scenario or {}).get("id") == scenario_id:
            self.scenario = self.scenarios[0] if self.scenarios else {}
            self.scenario_name = self.scenario.get("name", self.scenario_name)
        self.bump_state_version()
        return True

    def activate_scenario(self, scenario_id: str) -> dict:
        scen = self.get_scenario(scenario_id)
        if not scen:
            raise KeyError(f"scenario not found: {scenario_id}")
        self.scenario = scen
        self.scenario_name = scen.get("name", scenario_id)
        self.current_day = int(scen.get("currentDay", self.current_day))
        self.bump_state_version()
        self._add_activity(f"Activated scenario: {self.scenario_name}", "system")
        return scen

    def _seed_scenario_facts(self) -> None:
        for npc_id, npc in self.npcs.items():
            for b in npc.get("beliefs", []):
                fid = self._statement_to_fact_id(b.get("statement", b.get("id", "")))
                conf = float(b.get("confidence", 0.7))
                ev = str(b.get("evidence", ""))[:120]
                self.assert_fact(fid, ev, holder=npc_id, confidence=conf, valid_from=self.now)
                # sync NPC belief view
                b["fact_id"] = fid

    def get_trust(self, src: str, dst: str) -> float:
        for key in (f"{src}->{dst}", f"{src}->{dst.capitalize()}"):
            e = self.trust_edges.get(key)
            if e:
                return e.weight
        npc = self.npcs.get(src, {})
        for t in npc.get("trustNetwork", []):
            tgt = str(t.get("target", "")).lower()
            if tgt == dst.lower() or tgt == dst:
                return float(t.get("trust", 0.5))
        return 0.5

    def set_trust(self, src: str, dst: str, w: float) -> None:
        k = f"{src}->{dst}"
        if k in self.trust_edges:
            self.trust_edges[k].weight = w
        else:
            self.trust_edges[k] = NPCTrusts(npc_id_from=src, npc_id_to=dst, weight=w)
        npc = self.npcs.get(src)
        if npc:
            for t in npc.get("trustNetwork", []):
                if str(t.get("target", "")).lower() in (dst.lower(), dst):
                    t["trust"] = w
        self.bump_state_version()

    def assert_fact(self, fact_id: str, evidence: str, holder: str = "world",
                    confidence: float = 0.8, valid_from: Optional[float] = None,
                    version: Optional[int] = None, *,
                    source_npc: Optional[str] = None,
                    direct: bool = True) -> str:
        if valid_from is None:
            valid_from = self.now
        npc_key = self._npc_claim_key(holder, fact_id)
        ver = version or (len(self.npc_beliefs.get(npc_key, [])) + 1)
        vnid = f"{fact_id}_v{ver}@{holder}"
        isots = datetime.fromtimestamp(valid_from).isoformat()

        fact = Fact(
            fact_id=fact_id, subject=holder, predicate="claims", obj=evidence[:200],
            confidence=confidence, scope=FactScope.GLOBAL,
            version=ver, version_node_id=vnid, created_at=isots, is_active=True,
            source_npc=source_npc or holder,
        )
        self.facts[vnid] = fact
        chain = self.npc_beliefs[npc_key]
        if vnid not in chain:
            chain.append(vnid)

        self._atms_register(fact_id, evidence, holder, confidence, valid_from)

        ub = UnifiedBelief(
            claim_id=fact_id, evidence_id=evidence[:200], holder=holder,
            belief_tuple=BeliefTuple.from_confidence(confidence),
            valid_from=valid_from, version=ver, version_node_id=vnid,
            credibility=confidence,
        )
        self.beliefs[vnid] = ub

        self.knows_edges[f"{holder}->{vnid}"] = NPCKnows(
            npc_id=holder, fact_id=fact_id,
            opinion=BeliefTuple.from_confidence(confidence),
            version_node_id=vnid, direct=direct,
            last_updated=isots, is_stale=False,
        )

        ev_nid = self._graph_add({
            "type": "evidence", "title": evidence[:40] or fact_id,
            "description": evidence[:200], "timestamp": isots,
            "source": holder, "confidence": confidence, "relatedNPCs": [holder],
        })
        cl_nid = self._graph_add({
            "type": "claim", "title": fact_id.replace("_", " "),
            "description": f"{holder} claims: {evidence[:100]}",
            "timestamp": isots, "source": holder, "confidence": confidence,
            "relatedNPCs": [holder],
        })
        self._graph_link(ev_nid, cl_nid, "supported by")

        self.bump_state_version()
        return vnid

    def supersede_fact(self, fact_id: str, new_evidence: str, new_confidence: float,
                       holder: str = "world", *,
                       source_npc: Optional[str] = None,
                       direct: bool = True) -> dict:
        npc_key = self._npc_claim_key(holder, fact_id)
        chain = self.npc_beliefs.get(npc_key, [])
        new_v = len(chain) + 1
        new_vnid = f"{fact_id}_v{new_v}@{holder}"

        old_text, old_conf = "", 0.0
        if chain:
            old_ub = self.beliefs.get(chain[-1])
            if old_ub:
                old_text = old_ub.evidence_id
                old_conf = old_ub.belief_tuple.belief

        op_info = classify_game_event(old_text, new_evidence, fact_id, new_confidence - old_conf)
        incision_traces = []

        for old_vnid in chain:
            old_fact = self.facts.get(old_vnid)
            old_ub = self.beliefs.get(old_vnid)
            if not old_fact or not old_ub or not old_fact.is_active:
                continue
            if op_info["is_invalidation"]:
                new_ub = UnifiedBelief(
                    claim_id=fact_id, evidence_id=new_evidence[:200], holder=holder,
                    belief_tuple=BeliefTuple.from_confidence(new_confidence),
                    valid_from=self.now, version=new_v, version_node_id=new_vnid,
                    credibility=new_confidence,
                )
                incision_traces.append(hansson_incise(old_ub, new_ub, self.now, self.trust_edges))
            else:
                old_ub.status = BeliefStatus.WEAK
            old_fact.is_active = False
            old_fact.superseded_by = new_vnid
            old_ub.is_active = False
            if not old_ub.superseded_by:
                old_ub.superseded_by = new_vnid
                old_ub.status = BeliefStatus.SUPERSEDED
            self._mark_knows_stale(holder, old_vnid)

        vnid = self.assert_fact(
            fact_id, new_evidence, holder, new_confidence,
            valid_from=self.now, version=new_v,
            source_npc=source_npc, direct=direct,
        )
        if chain:
            self._graph_link(chain[-1], vnid, "superseded by")

        self._add_activity(f"Fact {fact_id} → v{new_v} ({op_info['op_type']}) [{holder}]", "belief")
        self.bump_state_version()
        return {"vnid": vnid, "version": new_v, "operation": op_info,
                "incision_traces": incision_traces, "holder": holder}

    def get_belief(self, npc_id: str, fact_id: str) -> dict:
        if not self._has_knowledge(npc_id, fact_id):
            return {
                "claim": fact_id, "holder": npc_id,
                "status": "no_knowledge", "confidence": 0.0,
            }

        chain = self._get_npc_chain(npc_id, fact_id)
        active_vnid = None
        for vnid in reversed(chain):
            edge = self.knows_edges.get(f"{npc_id}->{vnid}")
            if edge and edge.is_stale:
                continue
            f = self.facts.get(vnid)
            if f and f.is_active:
                active_vnid = vnid
                break
        if not active_vnid:
            return {
                "claim": fact_id, "holder": npc_id,
                "status": "no_active", "confidence": 0.0,
            }
        ub = self.beliefs.get(active_vnid)
        if not ub:
            return {
                "claim": fact_id, "holder": npc_id,
                "status": "no_belief", "confidence": 0.0,
            }
        trust = self.get_trust(npc_id, ub.holder)
        tw = ub.temporal_weight(self.now)
        discounted = self.sl_engine.discount_opinion(ub.belief_tuple, trust)
        final_b = discounted.belief * tw
        gm = GraphMemoryStatus.compute(ub, self.now)
        return {
            "claim": fact_id, "holder": npc_id, "confidence": round(final_b, 4),
            "status": gm.value, "version": ub.version,
            "source_holder": ub.holder, "source_trust": round(trust, 3),
            "temporal_weight": round(tw, 4),
            "version_chain": [f"v{i+1}" for i in range(len(chain))],
            "direct": (self.knows_edges.get(f"{npc_id}->{active_vnid}") or NPCKnows(
                npc_id=npc_id, fact_id=fact_id, opinion=BeliefTuple.from_confidence(0.5),
            )).direct,
        }

    def get_all_beliefs(self, npc_id: str) -> List[dict]:
        results: List[dict] = []
        for fid in self._known_claims(npc_id):
            b = self.get_belief(npc_id, fid)
            if b.get("status") != "no_knowledge":
                results.append(b)
        return results

    def _active_ub_for_claim(self, claim_id: str,
                             prefer_npc: Optional[str] = None) -> Optional[UnifiedBelief]:
        candidates: List[str] = []
        if prefer_npc:
            candidates.append(prefer_npc)
        candidates.extend(
            nid for (nid, claim) in self.npc_beliefs
            if claim == claim_id and nid not in candidates
        )
        for nid in candidates:
            if not self._has_knowledge(nid, claim_id):
                continue
            for vnid in reversed(self._get_npc_chain(nid, claim_id)):
                ub = self.beliefs.get(vnid)
                if ub and ub.is_active:
                    return ub
        return None

    def resolve_conflict(self, claim_id1: str = "", claim_id2: str = "") -> dict:
        """两相关 claim 的冲突消解（claim 来自 scenario 或引擎已知命题）。"""
        if claim_id1 and claim_id2:
            c1, c2 = claim_id1, claim_id2
        else:
            c1, c2 = self.get_conflict_pair()

        ub1 = self._active_ub_for_claim(c1)
        ub2 = self._active_ub_for_claim(c2)
        if not ub1 and not ub2:
            return {"conflict": False, "message": "claims not found"}
        if ub1 and ub2:
            self.atms.add_nogood_claims({c1, c2})
            result = hansson_incise(ub1, ub2, self.now, self.trust_edges)
            self.bump_state_version()
        else:
            result = {"action": "no_op", "trace": ["missing belief nodes"]}

        cons = self.consensus.compute_consensus(c1)
        steps = [
            {"step": 1, "name": "Receive", "detail": f"Conflicting claims: {c1} vs {c2}"},
            {"step": 2, "name": "Trust Check",
             "detail": f"Active holders: {ub1.holder if ub1 else '?'} / {ub2.holder if ub2 else '?'}"},
            {"step": 3, "name": "Conflict Detection",
             "detail": f"ATMS nogood registered for {c1}/{c2}"},
            {"step": 4, "name": "Belief Revision",
             "detail": " → ".join(result.get("trace", []))},
            {"step": 5, "name": "Consensus",
             "detail": f"Convergence={cons.get('convergence', 0):.0%} (holders={cons.get('holders', 0)})"},
        ]
        return {"conflict": True, "steps": steps, "incision_trace": result.get("trace", []),
                "result": result.get("action"), "consensus": cons, "claims": [c1, c2]}

    def player_action(self, npc_id: str, action_type: str, player_input: str,
                      dialogue_history: Optional[List[dict]] = None) -> dict:
        npc = self.npcs.get(npc_id, {})
        npc_name = npc.get("name", npc_id)
        action_type = action_type if action_type in ACTION_TYPES else "Talk"
        utterance = effective_utterance(action_type, player_input, npc_name)
        hist = dialogue_history or self.dialogue_history

        known_props = self.get_known_props_for_npc(npc_id)

        analysis = analyze_player_action(
            utterance, action_type, "Player", known_props=known_props)
        belief_change, trust_change = action_belief_trust_deltas(analysis, action_type)

        prop_key = analysis.get("proposition_key", "")
        skip_props = {
            "chitchat", "player_inquiry", "scene_observation", "player_misc",
        }
        classification = {
            "invalidate": bool(analysis.get("target_facts")),
            "confidence": "HIGH" if float(analysis.get("evidence_strength", 0)) > 0.7 else "LOW",
            "proposition_key": prop_key,
            "source": analysis.get("source", "rule"),
        }

        if not analysis.get("is_chitchat") and prop_key not in skip_props:
            conf = float(analysis.get("evidence_strength", 0.75))
            sr = self.supersede_fact(prop_key, utterance[:200], conf, holder=npc_id)
            classification["operation"] = sr["operation"]["op_type"]
            belief_change = max(belief_change, abs(float(analysis.get("polarity", 0))) * conf * 0.12)
            if sr["operation"]["is_invalidation"]:
                prop = self.consensus.propagate(prop_key, utterance[:200], npc_id, conf)
                self.propagation_queue.append({
                    "id": f"prop-{time.time()}", "claim": prop_key, "prop": prop,
                })
                self.propagation_queue = self.propagation_queue[-10:]

        if trust_change:
            self.set_trust(
                npc_id, "Player",
                min(1.0, self.get_trust(npc_id, "Player") + trust_change),
            )

        ev_block = ""
        beliefs = self.get_all_beliefs(npc_id)
        if beliefs:
            lines = [
                f"- {b['claim']}: conf={b['confidence']:.2f} status={b['status']}"
                for b in beliefs[:5]
            ]
            ev_block = "[Memory system evidence]\n" + "\n".join(lines)

        mem_node = self._graph_add({
            "type": "event",
            "title": f"{action_type} with {npc_name}",
            "description": utterance[:200],
            "timestamp": datetime.now().isoformat(),
            "source": "Player", "confidence": 1.0, "relatedNPCs": [npc_id],
        })

        self._add_activity(f"Player {action_type} with {npc_name}", "player")
        self.events_log.append({
            "time": datetime.now().isoformat(), "event": utterance[:200],
            "participants": ["Player", npc_id], "day": self.current_day,
        })

        self.add_private_memory(npc_id, {
            "type": "player_interaction",
            "action": action_type,
            "utterance": utterance,
            "proposition": prop_key,
            "classification": classification,
            "participants": ["Player", npc_id],
        })

        return {
            "utterance": utterance,
            "analysis": analysis,
            "evidence_block": ev_block,
            "beliefChange": belief_change,
            "trustChange": trust_change,
            "classification": classification,
            "memoryUpdate": mem_node,
            "beliefs": beliefs[:5],
            "npc": npc,
            "actionHint": ACTION_NPC_HINT.get(action_type, ""),
        }

    def process_event(self, description: str, participants: List[str],
                      fact_updates: Optional[List[dict]] = None) -> dict:
        results = {"event": description, "belief_changes": [], "propagations": []}
        if fact_updates:
            for fu in fact_updates:
                fid = fu["fact_id"]
                old_b = self.get_belief(participants[0] if participants else "world", fid)
                sr = self.supersede_fact(fid, fu.get("new_evidence", description[:100]),
                                         fu.get("confidence", 0.8), holder=fu.get("holder", "world"))
                new_b = self.get_belief(participants[0] if participants else "world", fid)
                results["belief_changes"].append({
                    "fact": fid, "old": old_b["confidence"], "new": new_b["confidence"],
                    "operation": sr["operation"]["op_type"],
                })
                if sr["operation"]["is_invalidation"]:
                    results["propagations"].append(
                        self.consensus.propagate(fid, fu.get("new_evidence", ""), fu.get("holder", "world"),
                                                 fu.get("confidence", 0.8)))
        evt = {"id": f"evt-{time.time()}", "type": "event", "description": description,
               "participants": participants, "timestamp": datetime.now().isoformat()}
        self.propagation_queue.append(evt)
        self.propagation_queue = self.propagation_queue[-10:]
        self.events_log.append({"time": datetime.now().isoformat(), "event": description,
                                "participants": participants, "day": self.current_day})
        self._add_activity(f"Event: {description[:60]}", "system")
        return results

    def advance_day(self, days: float = 1) -> List[dict]:
        self.now += days * 86400.0
        self.atms.now = self.now
        self.current_day += int(days)
        self.current_turn = 1
        changed = []
        for vnid, ub in self.beliefs.items():
            if not ub.is_active:
                continue
            old = ub.status
            ub.status = ub.compute_status(self.now)
            if ub.status != old:
                changed.append({"claim": ub.claim_id, "old": old.value, "new": ub.status.value})
        self._add_activity(f"Day advanced to {self.current_day}", "system")
        return changed

    def advance_turn(self) -> dict:
        self.current_turn += 1
        if self.current_turn > 24:
            self.advance_day(1)
        elif self.current_turn % 6 == 0:
            self._add_activity("NPC auto-dialogue window", "system")
        return {"day": self.current_day, "turn": self.current_turn}

    def sync_npc_beliefs_from_engine(self) -> None:
        for npc_id, npc in self.npcs.items():
            synced = []
            for i, b in enumerate(npc.get("beliefs", [])):
                fid = b.get("fact_id") or self._statement_to_fact_id(b.get("statement", ""))
                gb = self.get_belief(npc_id, fid)
                synced.append({
                    **b,
                    "confidence": gb.get("confidence", b.get("confidence", 0.5)),
                    "statement": b.get("statement", fid.replace("_", " ")),
                })
            npc["beliefs"] = synced

    def to_srs_state(self) -> dict:
        self.sync_npc_beliefs_from_engine()
        tracked = self.get_tracked_claims()

        consensus_metrics = {
            fid: self.consensus.compute_consensus(fid) for fid in tracked
        }
        cons_vals = [
            m.get("convergence", 0) for m in consensus_metrics.values()
        ]
        current_consensus = sum(cons_vals) / len(cons_vals) if cons_vals else 0.65

        npcs_out = []
        for npc_id, info in self.npcs.items():
            priv = self.get_private_memory(npc_id)
            priv_summaries = [
                m.get("utterance") or m.get("summary") or m.get("action", "")
                for m in priv[-5:]
            ]
            npcs_out.append({
                **info,
                "trustNetwork": info.get("trustNetwork", []),
                "shortTermMemory": priv_summaries or [
                    e["event"][:100] for e in self.events_log[-5:]
                    if npc_id in e.get("participants", [])
                ],
                "longTermMemory": [
                    m.get("utterance", "")[:100] for m in priv[-20:]
                ] or [e["event"][:100] for e in self.events_log[-20:]],
                "privateMemoryCount": len(priv),
            })

        return {
            "stateVersion": self.state_version,
            "npcs": npcs_out,
            "scenarios": self.scenarios,
            "currentScenario": self.scenario,
            "scenarioName": self.scenario_name,
            "currentDay": self.current_day,
            "currentTurn": self.current_turn,
            "memoryNodes": self.graph_nodes,
            "memoryEdges": self.graph_edges,
            "events": self.events_log[-20:],
            "activityFeed": self.activity_feed[:20],
            "pendingMemoryUpdates": self.pending_memory_updates[-10:],
            "propagationQueue": self.propagation_queue[-10:],
            "currentConsensus": round(current_consensus, 3),
            "consensusMetrics": consensus_metrics,
            "consensusHistory": self.consensus.consensus_history[-20:],
            "dialogueHistory": self.dialogue_history[-50:],
        }


_engine: Optional[GameEngineV13] = None


def get_engine() -> GameEngineV13:
    global _engine
    if _engine is None:
        _engine = GameEngineV13()
    return _engine


if __name__ == "__main__":
    eng = GameEngineV13()
    print("NPCs:", list(eng.npcs.keys()), "graph nodes:", len(eng.graph_nodes))
    c1, c2 = eng.get_conflict_pair()
    r = eng.resolve_conflict(c1, c2)
    print("conflict steps:", len(r.get("steps", [])))
    pa = eng.player_action("duran", "Accuse", "The markings on the armor look forged.")
    print("player_action beliefs:", len(pa.get("beliefs", [])))
