#!/usr/bin/env python3
"""
game_engine_v13.py — V13 游戏引擎 + GraphMem-ATMS + 组友时间戳
================================================================
v6  ATMSKernelV2 + Hansson incision
v7-v9  classify_fine_v9
v13 GraphMemory 节点/边 + ConsensusEngine
组友 Fact / NPCKnows / NPCTrusts + SubjectiveLogicEngine
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
            current = self.eng.get_belief(tnpc, claim)
            origins = {e.npc_id_from for k, e in self.eng.trust_edges.items()
                       if e.npc_id_to == tnpc and e.weight >= 0.3}
            delta = confidence * trust * 0.5
            if len(origins) <= 1 and source_npc != "world":
                delta *= 0.3
            old_conf = current["confidence"]
            new_conf = max(0.0, min(1.0, old_conf + delta))
            ub = self.eng.beliefs.get(f"{tnpc}_believes_{claim}")
            if ub and ub.holder == tnpc:
                new_conf = old_conf * 0.7 + new_conf * 0.3
            accepted = abs(new_conf - old_conf) > 0.02
            if accepted:
                self.eng.supersede_fact(claim, new_evidence, new_conf, holder=tnpc)
            results["propagations"].append({
                "target": tnpc, "accepted": accepted,
                "trust": round(trust, 3), "old_conf": round(old_conf, 3),
                "new_conf": round(new_conf, 3),
            })
        self.consensus_history.append(results)
        return results

    def compute_consensus(self, claim: str) -> dict:
        confs = [self.eng.get_belief(n, claim)["confidence"] for n in self.eng.npcs]
        if not confs:
            return {"consensus": 0.0, "variance": 0.0, "convergence": 0.0}
        mean = sum(confs) / len(confs)
        var = sum((c - mean) ** 2 for c in confs) / len(confs)
        return {"consensus": round(mean, 4), "variance": round(var, 4),
                "convergence": round(max(0, 1 - math.sqrt(var)), 4)}


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
        self.version_chains: Dict[str, List[str]] = defaultdict(list)
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

    def assert_fact(self, fact_id: str, evidence: str, holder: str = "world",
                    confidence: float = 0.8, valid_from: Optional[float] = None,
                    version: Optional[int] = None) -> str:
        if valid_from is None:
            valid_from = self.now
        ver = version or (len(self.version_chains.get(fact_id, [])) + 1)
        vnid = f"{fact_id}_v{ver}"
        isots = datetime.fromtimestamp(valid_from).isoformat()

        fact = Fact(
            fact_id=fact_id, subject=holder, predicate="claims", obj=evidence[:200],
            confidence=confidence, scope=FactScope.GLOBAL,
            version=ver, version_node_id=vnid, created_at=isots, is_active=True,
        )
        self.facts[vnid] = fact
        if vnid not in self.version_chains[fact_id]:
            self.version_chains[fact_id].append(vnid)

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
            version_node_id=vnid, direct=True,
            last_updated=isots, is_stale=False,
        )

        self._atms_register(fact_id, evidence[:200], holder, confidence, valid_from)

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

        return vnid

    def supersede_fact(self, fact_id: str, new_evidence: str, new_confidence: float,
                       holder: str = "world") -> dict:
        chain = self.version_chains.get(fact_id, [])
        new_v = len(chain) + 1
        new_vnid = f"{fact_id}_v{new_v}"

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

        vnid = self.assert_fact(fact_id, new_evidence, holder, new_confidence,
                                valid_from=self.now, version=new_v)
        if chain:
            self._graph_link(chain[-1], vnid, "superseded by")

        self._add_activity(f"Fact {fact_id} → v{new_v} ({op_info['op_type']})", "belief")
        return {"vnid": vnid, "version": new_v, "operation": op_info,
                "incision_traces": incision_traces}

    def get_belief(self, npc_id: str, fact_id: str) -> dict:
        chain = self.version_chains.get(fact_id, [])
        if not chain:
            return {"claim": fact_id, "status": "unknown", "confidence": 0.0}
        active_vnid = None
        for vnid in reversed(chain):
            f = self.facts.get(vnid)
            if f and f.is_active:
                active_vnid = vnid
                break
        if not active_vnid:
            return {"claim": fact_id, "status": "no_active", "confidence": 0.0}
        ub = self.beliefs.get(active_vnid)
        if not ub:
            return {"claim": fact_id, "status": "no_belief", "confidence": 0.0}
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
        }

    def get_all_beliefs(self, npc_id: str) -> List[dict]:
        return [self.get_belief(npc_id, fid) for fid in self.version_chains]

    def resolve_conflict(self, claim_id1: str, claim_id2: str) -> dict:
        """Thomas vs Duran style conflict: two related claims."""
        c1 = claim_id1 or "knight_is_trustworthy"
        c2 = claim_id2 or "knight_is_fake"
        chain1 = self.version_chains.get(c1, [])
        chain2 = self.version_chains.get(c2, [])
        if not chain1 and not chain2:
            return {"conflict": False, "message": "claims not found"}
        ub1 = self.beliefs.get(chain1[-1]) if chain1 else None
        ub2 = self.beliefs.get(chain2[-1]) if chain2 else None
        if ub1 and ub2:
            self.atms.add_nogood_claims({c1, c2})
            result = hansson_incise(ub1, ub2, self.now, self.trust_edges)
        else:
            result = {"action": "no_op", "trace": ["missing belief nodes"]}

        cons = self.consensus.compute_consensus(c1)
        steps = [
            {"step": 1, "name": "Receive", "detail": f"Conflicting claims: {c1} vs {c2}"},
            {"step": 2, "name": "Trust Check",
             "detail": f"Thomas→Duran trust={self.get_trust('thomas','duran'):.2f}"},
            {"step": 3, "name": "Conflict Detection",
             "detail": f"ATMS nogood registered for {c1}/{c2}"},
            {"step": 4, "name": "Belief Revision",
             "detail": " → ".join(result.get("trace", []))},
            {"step": 5, "name": "Consensus",
             "detail": f"Convergence={cons.get('convergence', 0):.0%}"},
        ]
        return {"conflict": True, "steps": steps, "incision_trace": result.get("trace", []),
                "result": result.get("action"), "consensus": cons}

    def player_action(self, npc_id: str, action_type: str, player_input: str,
                      dialogue_history: Optional[List[dict]] = None) -> dict:
        npc = self.npcs.get(npc_id, {})
        npc_name = npc.get("name", npc_id)
        action_type = action_type if action_type in ACTION_TYPES else "Talk"
        utterance = effective_utterance(action_type, player_input, npc_name)
        hist = dialogue_history or self.dialogue_history

        known_props: Dict[str, str] = {}
        for fid in ("knight_is_trustworthy", "knight_is_fake"):
            b = self.get_belief(npc_id, fid)
            if b:
                known_props[fid] = b.get("claim", fid)

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
        tracked = [
            fid for fid in (
                "knight_is_trustworthy", "knight_is_fake",
                "village_is_safe", "dark_forces_lurk_nearby",
            )
            if fid in self.version_chains
        ]
        if not tracked:
            tracked = list(self.version_chains.keys())[:6]

        consensus_metrics = {
            fid: self.consensus.compute_consensus(fid) for fid in tracked
        }
        cons_vals = [
            m.get("convergence", 0) for m in consensus_metrics.values()
        ]
        current_consensus = sum(cons_vals) / len(cons_vals) if cons_vals else 0.65

        npcs_out = []
        for npc_id, info in self.npcs.items():
            npcs_out.append({
                **info,
                "trustNetwork": info.get("trustNetwork", []),
                "shortTermMemory": [e["event"][:100] for e in self.events_log[-5:]],
                "longTermMemory": [e["event"][:100] for e in self.events_log[-20:]],
            })

        return {
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
    r = eng.resolve_conflict("knight_is_trustworthy", "knight_is_fake")
    print("conflict steps:", len(r.get("steps", [])))
    pa = eng.player_action("duran", "Accuse", "The knight's armor markings prove he is a fake.")
    print("player_action beliefs:", len(pa.get("beliefs", [])))
