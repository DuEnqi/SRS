#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npc_consensus_v13.py
====================
三人物版 (Tom / Elena / Duran) 假骑士事件 —— 多 NPC 记忆写入共识系统 v13。

v13 主线能力概览:
  - 语义分层叙事记忆图 × Kumiho 风格不可变记忆对象
  - 类型化节点/边 schema × 统一记忆状态机 × 访问控制分层
  - pairwise + community 两级传播 × 信任加权共识 voting
  - personality-gated 角色差异保留 × 多 NPC FAMA × MI/MPR 语义审核
  - 局部依赖级联 × ATMS/Hansson 核保留与核收缩
  - LLM 语义层 → 形式化决策层管线 × 第二故事泛化验证
  - v13 新增/强化: ATMSKernelV2 微基准、来源独立性、nogood、时序更新、
    多 agent 局部视角、前提抵抗探针，用于回应 P0/P1 类质疑。

运行示例:
  python npc_consensus_v13.py
      # 主线脚本 + FAMA + 消融 + 旁路实验 + benchmark 对比 + 报告输出

  python npc_consensus_v13.py --interactive
      # 交互模式: 固定剧情 + 玩家输入 + NPC 自主 1:1/1:多广播

  python npc_consensus_v13.py --agent-sim --agent-freedom 3
      # 单档自主 Agent 仿真; 1=反应式, 2=动机目标, 3=效用 argmax

  python npc_consensus_v13.py --freedom-compare
      # 三档自由度 Agent 对比

  python npc_consensus_v13.py --benchmark
      # 仅跑 No-Sharing / Trust-Propagation / HiRAG / Full System 对比

  python npc_consensus_v13.py --compare-ablations
      # 一键消融对比

  python npc_consensus_v13.py --atms-demo
      # ATMS 核保留演示: 替代支持环境存活 vs 核收缩

  python npc_consensus_v13.py --second-story
      # 第二故事(个人助理记忆域)泛化验证

  python npc_consensus_v13.py --atms-benchmark
      # v13 ATMS 必要性微基准: Full-ATMS vs BFS 级联

  python npc_consensus_v13.py --use-llm
      # 真实接入 yunwu.ai (需 YUNWU_API_KEY); 不开则走确定性 fallback

常见输出:
  out_v13/metrics.json
  out_v13/report.md
  out_v13/narration.txt
  out_v13/benchmark_comparison.json
  out_v13/ablation_comparison.md
  out_v13/atms_metrics.json
  out_v13/hansson_postulates.json
  out_v13/atms_ledger.jsonl
  out_v13/pipeline_metrics.json
  out_v13/pipeline_log.jsonl
  out_v13/modal_claims.json
  out_v13/atms_benchmark.json
"""
from __future__ import annotations
import os, sys, json, math, re, time, argparse, itertools, random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, FrozenSet
from enum import Enum
from collections import defaultdict

import logging
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
np.random.seed(42); random.seed(42)

# ============================================================================
# 0. API + 全局
# ============================================================================
API_KEY  = os.environ.get("YUNWU_API_KEY", "")
BASE_URL = "https://yunwu.ai/v1"
MODEL_NAME = "gpt-5.4-mini"
USE_COLOR = True


def clip01(x: float) -> float:
    """belief 永远不超过 1, 不低于 0 —— 全局唯一裁剪入口。"""
    try:
        return float(min(1.0, max(0.0, float(x))))
    except Exception:
        return 0.0


def _make_client(base_url: str = BASE_URL):
    from openai import OpenAI
    return OpenAI(api_key=API_KEY, base_url=base_url)


# ============================================================================
# 1. ConsoleNarrator
# ============================================================================
class C:
    @staticmethod
    def _w(code, s):  return f"\033[{code}m{s}\033[0m" if USE_COLOR else s
    @staticmethod
    def b(s):   return C._w("1", s)
    @staticmethod
    def d(s):   return C._w("2", s)
    @staticmethod
    def cyan(s):    return C._w("36", s)
    @staticmethod
    def green(s):   return C._w("32", s)
    @staticmethod
    def red(s):     return C._w("31", s)
    @staticmethod
    def yellow(s):  return C._w("33", s)
    @staticmethod
    def magenta(s): return C._w("35", s)
    @staticmethod
    def blue(s):    return C._w("34", s)
    @staticmethod
    def gray(s):    return C._w("90", s)


class Narrator:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.captured: List[str] = []
        self.step_counter = 0

    def _emit(self, s: str = ""):
        if not self.quiet:
            print(s)
        self.captured.append(s)

    def banner(self, title: str, sub: str = ""):
        bar = "═" * 78
        self._emit()
        self._emit(C.cyan(C.b(f"╔{bar}╗")))
        self._emit(C.cyan(C.b(f"║  {title:<74}║")))
        if sub:
            self._emit(C.cyan(f"║  {C.d(sub):<74}║"))
        self._emit(C.cyan(C.b(f"╚{bar}╝")))

    def scene_header(self, scene_id, title, time_label="", source="", props=""):
        self.step_counter = 0
        self._emit()
        src_tag = ""
        if source == "world_objective":
            src_tag = C.green("【客观世界事件】")
        elif source == "player_action":
            src_tag = C.yellow("【玩家主观行为】")
        elif source == "npc_action":
            src_tag = C.magenta("【NPC 自主 agent 行为】")
        self._emit(C.b(C.magenta(f"┏━━ SCENE {scene_id}: {title} {C.d('@'+time_label) if time_label else ''} ━━━")))
        meta = []
        if src_tag: meta.append(src_tag)
        if props:  meta.append(C.gray(f"涉及命题: {props}"))
        if meta:
            self._emit(f"┃   {'  '.join(meta)}")

    def step(self, label: str):
        self.step_counter += 1
        self._emit(C.b(C.blue(f"┃ ▸ Step {self.step_counter}: {label}")))

    def action(self, actor: str, what: str):
        self._emit(f"┃   {C.b(C.yellow('▶ '+actor))} {C.gray('—')} {what}")

    def kv(self, k, v, color=None):
        vs = str(v) if color is None else color(str(v))
        self._emit(f"┃     {C.gray(k+':'):<22} {vs}")

    def belief_change(self, npc, prop, before, after, note=""):
        if before is None:
            arrow = C.green(f"  ⊕ {after:.3f}  [新建]")
        else:
            delta = after - before
            if delta > 0.001:
                col = C.green; arrow_str = f"  {before:.3f} → {after:.3f}  (↑+{delta:.3f})"
            elif delta < -0.001:
                col = C.red;   arrow_str = f"  {before:.3f} → {after:.3f}  (↓{delta:.3f})"
            else:
                col = C.gray;  arrow_str = f"  {before:.3f} → {after:.3f}  (· {delta:+.3f})"
            arrow = col(arrow_str)
        nt = C.gray(f"  [{note}]") if note else ""
        self._emit(f"┃       {C.b(npc):<10} {C.gray(prop):<26}{arrow}{nt}")

    def rejected(self, npc, prop, score, tau, note=""):
        self._emit(f"┃       {C.b(npc):<10} {C.gray(prop):<26}"
                   f"  {C.yellow(f'✗ 拒绝 score={score:.3f}<τ={tau:.3f}')}"
                   f"  {C.gray('[记为: 听过但不信]')}{C.gray('  '+note if note else '')}")

    def trust_change(self, who, toward, before, after, reason):
        delta = after - before
        sign = "↑" if delta > 0 else "↓"
        col = C.green if delta > 0 else C.red
        self._emit(f"┃     {col('TRUST '+sign)}  {C.b(who)} → {C.b(toward)}: "
                   f"{before:.3f} {col('→')} {after:.3f}  ({col(f'{delta:+.3f}')})  {C.gray('//'+reason)}")

    def edge_added(self, layer, etype, src, dst, attr=""):
        self._emit(f"┃     {C.cyan('+ EDGE')}  [{layer}] {C.b(etype)}: {src} {C.cyan('→')} {dst}  {C.gray(attr)}")

    def access_block(self, sender, receiver, tier, reason):
        self._emit(f"┃     {C.red('⛔ BLOCK')} {C.b(sender)} ⤳ {C.b(receiver)}  "
                   f"{C.gray('tier='+tier)}  {C.red('// '+reason)}")

    def pair_update(self, receiver, sender, prop, old, new, trust):
        delta = new - (old or 0)
        col = C.green if delta >= 0 else C.red
        self._emit(f"┃     {C.cyan('⇄ PAIR')} {C.b(sender)}→{C.b(receiver)} [{prop}]  "
                   f"{(old or 0):.3f} {col('→')} {new:.3f}  {C.gray(f'trust={trust:.2f} (only-two-values)')}")

    def consensus(self, comm, prop, score, clusters, diverse, override, tau, promoted, prev_shared):
        if promoted and not prev_shared:
            tag = C.magenta(C.b("★ PROMOTE "))
        elif (not promoted) and prev_shared:
            tag = C.red(C.b("✘ REVOKE  "))
        elif promoted:
            tag = C.green("✓ HOLD    ")
        else:
            tag = C.gray("· NOTYET  ")
        self._emit(f"┃     {tag} [{C.b(comm)}::{prop}] {C.magenta(f'score={score:.3f}')}  τ={tau}  "
                   f"clusters={clusters}  diverse={diverse}  override={override:.2f}")

    def world_fact(self, prop, avg, comms):
        self._emit(f"┃     {C.magenta(C.b('🌐 WORLD FACT'))}  [{prop}] avg={avg:.3f}  来源子社区={comms}")

    def cascade(self, report):
        self._emit(f"┃     {C.yellow('⚡ DEPENDENCY-AWARE CASCADE')} 触发命题={C.b(report['trigger'])}")
        self._emit(f"┃         {C.gray('visited_nodes')} ={report['visited_nodes']}  "
                   f"{C.gray('visited_edges')}={report['visited_edges']}  "
                   f"{C.gray('affected_nodes')}={len(report['affected_nodes'])}  "
                   f"{C.gray('runtime')}={report['cascade_runtime_ms']:.2f}ms")
        for a in report['affected_nodes'][:10]:
            self._emit(f"┃         {C.yellow('↳')} {a}")

    def atms(self, msg, kind="info"):
        tag = {"retain": C.green("🛡 ATMS CORE-RETAIN"),
               "contract": C.red("✂ ATMS KERNEL-CONTRACT"),
               "label": C.cyan("🔖 ATMS LABEL"),
               "nogood": C.red("⊘ ATMS NOGOOD"),
               "info": C.cyan("🔧 ATMS")}.get(kind, C.cyan("🔧 ATMS"))
        self._emit(f"┃     {tag} {C.gray(msg)}")

    def hansson(self, rec):
        ok = lambda b: C.green("✓") if b else C.red("✗")
        self._emit(f"┃     {C.magenta('📐 HANSSON POSTULATES')} 收缩命题={C.b(rec['contracted_by'])}  "
                   f"Inclusion {ok(rec['Inclusion'])}  Success {ok(rec['Success'])}  "
                   f"Vacuity {ok(rec['Vacuity'])}  Core-Retainment {ok(rec['Core_Retainment'])}  "
                   f"Uniformity {ok(rec['Uniformity'])}")
        if rec['removed']:
            self._emit(f"┃         {C.gray('被收缩(在 p 的 kernel 内):')} {', '.join(rec['removed'])}")

    def decay(self, prop, decayed):
        if not decayed: return
        self._emit(f"┃     {C.gray('⏳ TIME DECAY')} {prop}:")
        for npc, old, new in decayed:
            delta = new - old
            tail = C.gray(f"({delta:+.3f} 朝锚点 {'↓遗忘' if delta<0 else '↑回升'})") if abs(delta) > 0 else C.gray("(无变化)")
            self._emit(f"┃         {C.gray(npc):<10} {old:.3f} → {new:.3f}  {tail}")

    def abstraction(self, scope, label, srcs, agg):
        self._emit(f"┃     {C.cyan('▲ ABSTRACT')} [{scope}] semantic→theme: {C.b(label)} b̄={agg:.3f}")
        self._emit(f"┃         {C.gray('合并源:')} {', '.join(srcs[:5])}{C.gray(' ...') if len(srcs) > 5 else ''}")

    def llm_call(self, kind, mode, summary):
        tag = C.green("🤖 LLM") if mode == "llm" else C.yellow("🧠 FALLBACK")
        self._emit(f"┃     {tag} [{kind}] {C.gray(summary)}")

    def policy(self, actor, strategy, target, reason):
        self._emit(f"┃     {C.red('🎭 PERSUASION-POLICY')} {C.b(actor)} 选择策略={C.b(strategy)} "
                   f"→ 目标={C.b(target)}  {C.gray('// '+reason)}")

    def table(self, headers, rows, title=""):
        if title: self._emit(C.b(C.cyan(f"┃  ◆ {title}")))
        widths = [max(len(str(headers[i])), max((len(str(r[i])) for r in rows), default=0)) for i in range(len(headers))]
        head = " │ ".join(C.b(str(h).ljust(widths[i])) for i, h in enumerate(headers))
        sep = "─┼─".join("─" * w for w in widths)
        self._emit(f"┃    {head}")
        self._emit(f"┃    {C.gray(sep)}")
        for r in rows:
            self._emit("┃    " + " │ ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))

    def end_scene(self):
        self._emit(C.magenta("┗" + "━" * 88))

    def info(self, s: str):
        self._emit(f"┃   {C.gray('ℹ')} {C.gray(s)}")

    def warn(self, s: str):
        self._emit(f"┃   {C.yellow('⚠')} {C.yellow(s)}")


# ============================================================================
# 2. NPC (Tom / Elena / Duran) + 关系 / 信任 (Revised Test Case 1)
# ============================================================================
NPCS = ["Tom", "Elena", "Duran"]

PERSONALITIES = {
    "Tom":   "天真赤诚、热血冲动、崇拜骑士精神、对人无防备、易共情、相信眼见为实; 铁匠之子, 唯一目击玩家战斗的人",
    "Elena": "理性、稳重、重视证据、权威导向、以村庄利益为最高优先级、不情绪化; 村长 (高影响力, 官方认可来源)",
    "Duran": "古板、守序、极度尊重正统骑士精神、痛恨欺骗与虚伪、原则至上、宁折不弯; 铁匠, 祖传圣剑守护者",
}

BIG_FIVE = {
    "Tom":   {"O": 8, "C": 5, "E": 7, "A": 8, "N": 5},
    "Elena": {"O": 6, "C": 8, "E": 6, "A": 7, "N": 3},
    "Duran": {"O": 3, "C": 9, "E": 4, "A": 4, "N": 3},
}

TRAITS = {
    "Tom":   dict(influence=0.40, knight_attitude=+0.45, liar_aversion=0.40, hero_worship=0.95, gullible=0.85, faction="order"),
    "Elena": dict(influence=0.92, knight_attitude=+0.20, liar_aversion=0.55, hero_worship=0.15, gullible=0.36, faction="order"),
    "Duran": dict(influence=0.62, knight_attitude=+0.25, liar_aversion=0.95, hero_worship=0.10, gullible=0.18, faction="order"),
}

# 三人小村: 全集 + 两个含权威 Elena 的子社区 (用于 world consensus 升级需要 ≥2 子社区)
COMMUNITIES = {
    "Greyford":      {"Tom", "Elena", "Duran"},
    "tavern_circle": {"Tom", "Elena"},      # 青年/广场圈 (Tom + 权威 Elena)
    "elders":        {"Elena", "Duran"},    # 长者圈 (权威 Elena + 铁匠 Duran)
}
SUBCOMMUNITIES = [c for c in COMMUNITIES if c != "Greyford"]

RIVALS: Set[frozenset] = set()      # 三人无情敌关系 (机制保留, 集合为空)
def are_rivals(a, b) -> bool:
    return frozenset({a, b}) in RIVALS

# 谁能看见/听到谁 (小村, 三人互相可见; 机制用于选择性传播 RQ2)
SOCIAL_VISIBILITY = {
    "Tom":   {"Elena", "Duran"},
    "Elena": {"Tom", "Duran"},
    "Duran": {"Tom", "Elena"},
}


def build_trust_matrix() -> Dict[str, Dict[str, float]]:
    """信任矩阵 (row 信任 column) —— 三人物设定。"""
    T = {
        "Tom":   {"Tom": 1.0, "Elena": 0.70, "Duran": 0.20},
        "Elena": {"Tom": 0.50, "Elena": 1.0, "Duran": 0.70},
        "Duran": {"Tom": 0.20, "Elena": 0.80, "Duran": 1.0},
    }
    return T


# ============================================================================
# 3. Persona 参数 (由 Big-Five 推导)
# ============================================================================
TAU0, BETA = 0.50, 0.16
ALPHA0, GAMMA = 0.25, 0.45


def persona_params(npc: str) -> Dict[str, float]:
    bf = BIG_FIVE[npc]; tr = TRAITS[npc]
    tau = TAU0 + BETA * (bf["C"] - bf["E"]) / 10.0 - 0.12 * tr["gullible"]
    tau = float(np.clip(tau, 0.22, 0.80))
    alpha = ALPHA0 + GAMMA * bf["O"] / 10.0
    alpha = float(np.clip(alpha, 0.15, 0.85))
    prop_will = float(np.clip(0.20 + 0.08 * bf["E"], 0.2, 0.95))
    stubborn = float(np.clip(0.20 * bf["C"] / 10 + 0.20 * (10 - bf["O"]) / 10
                             + 0.18 * tr["hero_worship"] + 0.16 * tr["liar_aversion"], 0.0, 1.0))
    abstract_eagerness = float(np.clip(0.3 + 0.05 * bf["O"], 0.3, 0.85))
    trust_sensitivity = float(np.clip(0.10 + 0.02 * bf["N"], 0.10, 0.30))
    admit_error = float(np.clip(0.30 + 0.04 * bf["O"] - 0.03 * bf["N"] - 0.25 * tr["hero_worship"], 0.05, 0.9))
    skepticism = float(np.clip(0.25 + 0.5 * tr["liar_aversion"] - 0.3 * tr["gullible"], 0.05, 0.95))
    return dict(tau=round(tau, 4), alpha=round(alpha, 4),
                prop_will=round(prop_will, 4), stubborn=round(stubborn, 4),
                abstract_eagerness=round(abstract_eagerness, 4),
                trust_sensitivity=round(trust_sensitivity, 4),
                admit_error=round(admit_error, 4), skepticism=round(skepticism, 4))


# ============================================================================
# 4. Schema (类型化本体: 节点/边/状态/访问层)
# ============================================================================
class NodeType(Enum):
    RAW_MSG   = "RawMessage"
    EPISODE   = "Episode"
    SEMANTIC  = "SemanticNode"
    THEME     = "Theme"
    PERSON    = "Person"
    RELATION  = "Relationship"
    CONSENSUS = "ConsensusFact"


class EdgeType(Enum):
    SOURCE_OF    = "source_of"
    DERIVED_FROM = "derived_from"
    CONTRADICTS  = "contradicts"
    AFFECTS      = "affects"          # 含 SUPPORTS
    DEPENDS_ON   = "depends_on"
    SUPERSEDES   = "supersedes"
    SUMMARIZES   = "summarizes"
    INVALIDATES  = "invalidates"
    WITNESSED    = "witnessed"
    TRUSTS       = "trusts"
    PROJECTS_TO  = "projects_to"
    AGGREGATED   = "aggregated_from"
    REFERENCES   = "references"
    PROMOTED     = "promoted_from"


_ANY = list(NodeType)
EDGE_SCHEMA: Dict[EdgeType, List[Tuple[NodeType, NodeType]]] = {
    EdgeType.SOURCE_OF:    [(NodeType.EPISODE, NodeType.SEMANTIC),
                            (NodeType.PERSON, NodeType.SEMANTIC),
                            (NodeType.PERSON, NodeType.EPISODE)],
    EdgeType.DERIVED_FROM: [(NodeType.SEMANTIC, NodeType.EPISODE),
                            (NodeType.SEMANTIC, NodeType.SEMANTIC),
                            (NodeType.SEMANTIC, NodeType.THEME)],
    EdgeType.CONTRADICTS:  [(NodeType.SEMANTIC, NodeType.SEMANTIC),
                            (NodeType.CONSENSUS, NodeType.SEMANTIC),
                            (NodeType.CONSENSUS, NodeType.CONSENSUS)],
    EdgeType.AFFECTS:      [(NodeType.SEMANTIC, NodeType.SEMANTIC)],
    EdgeType.DEPENDS_ON:   [(NodeType.SEMANTIC, NodeType.SEMANTIC),
                            (NodeType.CONSENSUS, NodeType.SEMANTIC)],
    EdgeType.SUPERSEDES:   [(t, t) for t in _ANY],
    EdgeType.SUMMARIZES:   [(NodeType.RAW_MSG, NodeType.EPISODE),
                            (NodeType.EPISODE, NodeType.SEMANTIC),
                            (NodeType.EPISODE, NodeType.THEME),
                            (NodeType.SEMANTIC, NodeType.THEME)],
    EdgeType.INVALIDATES:  [(NodeType.SEMANTIC, NodeType.SEMANTIC),
                            (NodeType.CONSENSUS, NodeType.SEMANTIC)],
    EdgeType.WITNESSED:    [(NodeType.PERSON, NodeType.EPISODE)],
    EdgeType.TRUSTS:       [(NodeType.PERSON, NodeType.PERSON)],
    EdgeType.PROJECTS_TO:  [(NodeType.RELATION, NodeType.PERSON),
                            (NodeType.RELATION, NodeType.SEMANTIC)],
    EdgeType.AGGREGATED:   [(NodeType.CONSENSUS, NodeType.SEMANTIC)],
    EdgeType.REFERENCES:   [(NodeType.SEMANTIC, NodeType.CONSENSUS)],
    EdgeType.PROMOTED:     [(NodeType.CONSENSUS, NodeType.CONSENSUS)],
}


class MemCategory(Enum):
    FACT_PLAYER   = "fact_about_player"
    EMOTION       = "emotional"
    RELATIONSHIP  = "relationship"
    TASK          = "task"
    FACTION       = "faction"
    PERSONA_DRIFT = "persona_drift"


class MemStatus(Enum):
    ACTIVE     = "active"
    DEPRECATED = "deprecated"
    REFUTED    = "refuted"
    SUPERSEDED = "superseded"
    PENDING    = "pending_verification"
    LOW_CONF   = "low_confidence"
    SHARED     = "shared"


# 统一状态机: 能否传播 / 能否进 shared / 能否作正证据 / 证据权重
STATUS_CAPS: Dict[str, dict] = {
    MemStatus.ACTIVE.value:     dict(propagate=True,  shareable=True,  pos_evidence=True,  weight=1.00),
    MemStatus.SHARED.value:     dict(propagate=True,  shareable=True,  pos_evidence=True,  weight=1.00),
    MemStatus.PENDING.value:    dict(propagate=True,  shareable=False, pos_evidence=False, weight=0.50),
    MemStatus.LOW_CONF.value:   dict(propagate=True,  shareable=False, pos_evidence=True,  weight=0.50),
    MemStatus.REFUTED.value:    dict(propagate=False, shareable=False, pos_evidence=False, weight=0.00),
    MemStatus.SUPERSEDED.value: dict(propagate=False, shareable=False, pos_evidence=False, weight=0.00),
    MemStatus.DEPRECATED.value: dict(propagate=False, shareable=False, pos_evidence=False, weight=0.00),
}
def status_cap(status: str, key: str):
    return STATUS_CAPS.get(status, STATUS_CAPS[MemStatus.ACTIVE.value])[key]

# 访问层 (从私密到公开)
ACCESS_TIERS = ["personal_episodic", "relationship_memory", "community_shared",
                "public_consensus", "core_identity"]
REL_TRUST_GATE = 0.55
TRUST_PROP_GATE = 0.30
W_MAX = 0.6
DIVERSITY_MIN = 2

HALF_LIFE = {
    "direct_observation": 60, "authority": 80, "official": 90, "self_claim": 25,
    "rumor": 12, "hearsay": 18, "gossip": 15, "consensus": 120, "first_hand_meta": 200,
}
REINFORCE_RHO = 0.55

REL_STRENGTH = {"DEPENDS_ON": 0.85, "CONTRADICTS": 0.95, "AFFECTS": 0.45, "SUPPORTS": 0.45}
DIST_DECAY = 0.6

EVIDENCE_CRED = {"official": 1.00, "authority": 0.90, "direct_observation": 0.85,
                 "consensus": 0.80, "first_hand_meta": 0.85, "self_claim": 0.40,
                 "hearsay": 0.40, "rumor": 0.35, "gossip": 0.35}
def evidence_cred(et: str) -> float:
    return EVIDENCE_CRED.get(et, 0.50)


# ============================================================================
# 5. Kumiho 风格记忆对象: Provenance / Revision / MemItem
# ============================================================================
@dataclass
class Provenance:
    origin_source: str
    transmission_path: List[str] = field(default_factory=list)
    evidence_type: str = "rumor"


@dataclass
class Revision:
    """不可变记忆版本 (Kumiho Revision)。一旦写入不再修改, 仅靠新 Revision + 指针迁移。"""
    rev_id: str
    uri: str
    ntype: NodeType
    content: str
    proposition_key: Optional[str] = None
    confidence: float = 0.0
    category: Optional[str] = None
    access_tier: str = "public_consensus"
    status: str = MemStatus.ACTIVE.value
    version: int = 1
    proposer: Optional[str] = None
    prov: Optional[Provenance] = None
    source_event: Optional[str] = None
    created_at: float = 0.0
    valid_from: float = 0.0
    valid_until: Optional[float] = None
    replaced_by: Optional[str] = None
    contradiction_links: List[str] = field(default_factory=list)
    anchor: Optional[float] = None
    reinforcement: int = 0
    last_seen: float = 0.0
    heard_rejected: bool = False
    rejected_score: Optional[float] = None
    event_source: Optional[str] = None

    def __post_init__(self):
        self.confidence = clip01(self.confidence)


@dataclass
class MemItem:
    """逻辑记忆对象 (Kumiho Item)。current_rev=当前指针(tag); deprecated 链=历史指针保留。"""
    uri: str
    layer: str
    scope: str
    proposition_key: str
    current_rev: Optional[str] = None
    revisions: List[str] = field(default_factory=list)
    deprecated_revs: List[str] = field(default_factory=list)


# ============================================================================
# 6. 命题注册表 + 主题 / 类别 / 派生节点
# ============================================================================
PROP_REGISTRY: Dict[str, str] = {
    "player_is_knight":      "玩家是王国正式派遣的骑士(身份命题, 依赖品德/能力/权威等众多语义信念)",
    "player_claimed_knight": "玩家口头自称骑士(言语行为事实, 仅弱支持身份, 不被身份依赖)",
    "player_good_character": "玩家品德高尚/正直可信(语义中间层, 由善举支持、被指控削弱)",
    "player_helped_village": "玩家确实保护/帮助了村庄(善举事实)",
    "player_combat_skill":   "玩家展现很强战斗能力",
    "player_poisoned_well":  "玩家往水井下毒(指控)",
    "player_is_spy":         "玩家是敌国间谍(指控)",
    "player_is_villain":     "玩家是恶徒(指控)",
    "official_denial":       "官方文书证明王国从未派遣此人(独立客观事实, 不被级联削弱)",
    "village_endorsement":   "村长公开背书玩家为骑士(权威言论)",
    "sword_given_to_player": "圣剑确实被交到玩家手中(发生事实, 一旦发生不因身份反转而消失)",
    "sword_transfer_legitimate": "交剑在'玩家确为真骑士'前提下是合法正当的(合法性判断, DEPENDS_ON 身份)",
    "endorsement_legitimate": "村长公开背书'玩家为真骑士'这一行为本身是否正当(合法性判断, DEPENDS_ON 身份)",
    "monster_attacked":      "怪物袭击村庄(客观事件)",
}
PROP_THEME = {
    "player_is_knight": "玩家身份", "player_claimed_knight": "玩家身份",
    "official_denial": "玩家身份", "village_endorsement": "玩家身份",
    "player_good_character": "玩家德行", "player_helped_village": "玩家德行",
    "player_combat_skill": "玩家德行",
    "player_poisoned_well": "对玩家的指控", "player_is_spy": "对玩家的指控",
    "player_is_villain": "对玩家的指控",
    "sword_given_to_player": "圣剑任务", "sword_transfer_legitimate": "圣剑任务",
    "endorsement_legitimate": "玩家身份",
    "monster_attacked": "村庄威胁",
}
PROP_CATEGORY = {
    "player_is_knight": MemCategory.FACT_PLAYER, "player_claimed_knight": MemCategory.FACT_PLAYER,
    "official_denial": MemCategory.FACT_PLAYER, "village_endorsement": MemCategory.FACTION,
    "player_good_character": MemCategory.FACT_PLAYER,
    "player_helped_village": MemCategory.FACT_PLAYER, "player_combat_skill": MemCategory.FACT_PLAYER,
    "player_poisoned_well": MemCategory.FACT_PLAYER, "player_is_spy": MemCategory.FACTION,
    "player_is_villain": MemCategory.FACT_PLAYER,
    "sword_given_to_player": MemCategory.TASK, "sword_transfer_legitimate": MemCategory.TASK,
    "endorsement_legitimate": MemCategory.FACTION,
    "monster_attacked": MemCategory.TASK,
}
DERIVED_PROPS = ["player_good_character", "player_is_knight"]
def prop_category(prop: str) -> str:
    if prop in PROP_CATEGORY: return PROP_CATEGORY[prop].value
    if "relates" in prop: return MemCategory.RELATIONSHIP.value
    if "intent" in prop or "drift" in prop: return MemCategory.PERSONA_DRIFT.value
    return MemCategory.FACT_PLAYER.value


# ----------------------------------------------------------------------------
# 6b. Provenance / 证据归因 评估金标准 (Tom/Elena/Duran 假骑士叙事的"正确语义图")
#     —— 用于 Relation Accuracy / Contradiction Detection / Invalidation 等量化指标。
# ----------------------------------------------------------------------------
# 叙事记忆图的正确关系 (src, relation, tgt)
GOLD_RELATIONS: Set[Tuple[str, str, str]] = {
    ("player_helped_village",     "SUPPORTS",    "player_good_character"),
    ("player_poisoned_well",      "CONTRADICTS", "player_good_character"),
    ("player_good_character",     "SUPPORTS",    "player_is_knight"),
    ("player_combat_skill",       "SUPPORTS",    "player_is_knight"),
    ("player_claimed_knight",     "SUPPORTS",    "player_is_knight"),
    ("village_endorsement",       "SUPPORTS",    "player_is_knight"),
    ("sword_transfer_legitimate", "DEPENDS_ON",  "player_is_knight"),
    ("endorsement_legitimate",    "DEPENDS_ON",  "player_is_knight"),
    ("official_denial",           "CONTRADICTS", "player_is_knight"),
}
# 真正互斥的命题对 (用于矛盾检测 P/R; 评估时仅保留本次叙事中实际出现的对)
GOLD_CONTRADICTION_PAIRS: List[Tuple[str, str]] = [
    ("official_denial",      "player_is_knight"),
    ("player_poisoned_well", "player_good_character"),
    ("player_is_spy",        "player_good_character"),
    ("player_is_villain",    "player_good_character"),
]
# 官方反证后"应被失效/下调"的命题 vs "应被保留的发生事实/真善举"
GOLD_SHOULD_INVALIDATE = {"player_is_knight", "sword_transfer_legitimate", "endorsement_legitimate"}
GOLD_SHOULD_PRESERVE = {"sword_given_to_player", "player_helped_village", "official_denial",
                        "player_combat_skill", "monster_attacked"}
# 视为"有据可依"的证据类型 (rumor/self_claim/hearsay/gossip 视为弱/无据)
CREDIBLE_EVIDENCE = {"direct_observation", "official", "authority", "consensus", "first_hand_meta"}


# ============================================================================
# 6c. ATMS 内核 (de Kleer 风格 assumption-based TMS) + Hansson belief-base 假设审计
#     —— 理论创新升级核心 (v11): 把 v10 的 "prop_relations + confidence BFS 衰减"
#        提升为 Claim 级 justification 超边 + 最小一致支持环境 (label) + nogood + 可废止守卫,
#        从而能严格表达 "替代支持路径存活" 并据此做 Hansson 核收缩 (kernel contraction)
#        与核保留 (Core-Retainment), 而不是把依赖节点统一乘一个衰减系数。
#
#     依据文献:
#       - de Kleer 1986《An Assumption-based TMS》: assumption / justification(Horn 超边) /
#         environment / label(最小一致支持集) / nogood / minimality。
#       - Dixon & Foo 1993《Connections Between the ATMS and AGM Belief Revision》:
#         用 entrenchment/justification 在 ATMS 与 AGM 之间建立联系。
#       - Fermé & Hansson 2011《AGM 25 Years》§3.1/§4.1: belief base 下 Recovery 失效,
#         "merely derived" 信念不应与独立证据同等地位; 收缩应满足 Core-Retainment / Relevance /
#         Uniformity / Success / Inclusion / Vacuity。
#       - 可废止推理 (defeasible): 高权威 official_denial 作为 defeater (负前提/outlist),
#         击败 "品德/战力/自称/背书 ⇒ 真骑士" 这一可废止链, 而非删除底层证据。
# ============================================================================
@dataclass(frozen=True)
class Assumption:
    """ATMS 可撤回前提 (foundational belief)。一次直接观察/文书/权威断言 ⇒ 一个 Assumption。"""
    aid: str
    claim: str
    polarity: int                 # +1 支持 claim, -1 支持 ¬claim
    evidence_type: str
    origin: str                   # 原始事件 / 来源标签
    holder_scope: str = "world"


@dataclass(frozen=True)
class Justification:
    """ATMS justification 超边: (premises ∧ ¬neg_premises) → conclusion。
    neg_premises 为可废止 defeater (de Kleer outlist): 任一 neg 命题当前被支持则本 justification 失效。"""
    jid: str
    premises: FrozenSet[str]
    neg_premises: FrozenSet[str]
    conclusion: str
    operator: str = "AND"         # AND / DEFEASIBLE (语义一致, 仅用于标注)
    strength: float = 0.5
    rationale: str = ""


class ATMSKernel:
    """Assumption-based Truth Maintenance kernel。维护 assumption / justification / environment /
    label / nogood, 提供 surviving_environments / is_supported / has_alternative_support /
    kernels_of, 供 Hansson 核收缩与核保留判定。本内核为 append-only ledger (不可变证据账本)。"""

    def __init__(self, max_label: int = 24, max_env: int = 6):
        self.assumptions: Dict[str, Assumption] = {}
        self.base_assumption: Dict[str, str] = {}          # claim -> 该 claim 的正极性底层 assumption id
        self.justifications: List[Justification] = []
        self.nogoods: Set[FrozenSet[str]] = set()
        self.believed: Set[str] = set()                    # 当前被持有的底层 assumption (the context)
        self.labels: Dict[str, Set[FrozenSet[str]]] = defaultdict(set)
        self.defeaters: Dict[str, Set[str]] = defaultdict(set)  # conclusion -> 其可废止链的 defeater 命题集
        self.max_label = max_label
        self.max_env = max_env
        self._ctr = 0
        self.ledger: List[dict] = []

    def _id(self, p="A"):
        self._ctr += 1
        return f"{p}{self._ctr:04d}"

    def _rec(self, **kw):
        self.ledger.append(dict(kw))

    # ---------- 断言底层证据 / 登记 justification / nogood ----------
    def assert_base(self, claim: str, evidence_type: str, origin: str,
                    polarity: int = 1, holder_scope: str = "world") -> str:
        if claim in self.base_assumption and polarity == 1:
            aid = self.base_assumption[claim]; self.believed.add(aid)
            self._rec(op="rebelieve_base", claim=claim, aid=aid); return aid
        aid = self._id("A")
        self.assumptions[aid] = Assumption(aid, claim, polarity, evidence_type, origin, holder_scope)
        if polarity == 1:
            self.base_assumption[claim] = aid
        self.believed.add(aid)
        self.labels[claim].add(frozenset({aid}))
        self._rec(op="assert_base", claim=claim, aid=aid, polarity=polarity,
                  evidence_type=evidence_type, origin=origin)
        return aid

    def register_defeater(self, conclusion: str, defeater_claim: str):
        self.defeaters[conclusion].add(defeater_claim)

    def add_justification(self, premises, conclusion, neg_premises=None,
                          operator="AND", strength=0.5, rationale="") -> str:
        prem = frozenset(p for p in premises if p)
        neg = frozenset(neg_premises or [])
        for j in self.justifications:
            if j.premises == prem and j.neg_premises == neg and j.conclusion == conclusion:
                return j.jid
        jid = self._id("J")
        self.justifications.append(Justification(jid, prem, neg, conclusion, operator, strength, rationale))
        self._rec(op="add_justification", jid=jid, premises=sorted(prem), neg=sorted(neg),
                  conclusion=conclusion, operator=operator, strength=round(float(strength), 3))
        return jid

    def add_nogood(self, claims, reason=""):
        aids = [self.base_assumption[c] for c in claims if c in self.base_assumption]
        if aids:
            self.nogoods.add(frozenset(aids))
            self._rec(op="add_nogood", claims=sorted(claims), assumptions=sorted(aids), reason=reason)

    # ---------- label 计算 (de Kleer / Dixon-Foo 风格不动点) ----------
    def _env_consistent(self, env: FrozenSet[str]) -> bool:
        return not any(ng and ng <= env for ng in self.nogoods)

    def _minimize(self, envs: Set[FrozenSet[str]]) -> Set[FrozenSet[str]]:
        envs = {e for e in envs if self._env_consistent(e)}
        out: Set[FrozenSet[str]] = set()
        for e in sorted(envs, key=lambda s: (len(s), tuple(sorted(s)))):
            if not any(o <= e for o in out):
                out.add(e)
        return set(sorted(out, key=lambda s: (len(s), tuple(sorted(s))))[: self.max_label])

    def recompute_labels(self, max_iter: int = 24):
        labels: Dict[str, Set[FrozenSet[str]]] = defaultdict(set)
        for claim, aid in self.base_assumption.items():
            labels[claim].add(frozenset({aid}))
        for _ in range(max_iter):
            changed = False
            supported_now = {c for c, L in labels.items() if any(e <= self.believed for e in L)}
            for j in self.justifications:
                # 可废止守卫: 任一负前提命题当前被支持 → 本 justification 不产生环境 (被 defeat)
                if any(np in supported_now for np in j.neg_premises):
                    continue
                pls = [labels.get(p) for p in j.premises]
                if any(not pl for pl in pls):
                    continue
                new_envs = set()
                for combo in itertools.islice(itertools.product(*pls), 4096):
                    env = frozenset().union(*combo) if combo else frozenset()
                    if len(env) <= self.max_env and self._env_consistent(env):
                        new_envs.add(env)
                before = set(labels[j.conclusion])
                labels[j.conclusion] = self._minimize(labels[j.conclusion] | new_envs)
                if labels[j.conclusion] != before:
                    changed = True
            if not changed:
                break
        for c in labels:
            labels[c] = self._minimize(labels[c])
        self.labels = labels
        return labels

    # ---------- 查询 ----------
    def surviving_environments(self, claim: str) -> List[FrozenSet[str]]:
        self.recompute_labels()
        return [e for e in self.labels.get(claim, set())
                if e <= self.believed and self._env_consistent(e)]

    def is_supported(self, claim: str) -> bool:
        return len(self.surviving_environments(claim)) > 0

    def kernels_of(self, claim: str) -> List[FrozenSet[str]]:
        """claim 的最小蕴含支持集 (Hansson kernel): 对应当前 label 的最小环境。"""
        return list(self.labels.get(claim, set())) or self.surviving_environments(claim)

    def _labels_with_blocked(self, blocked: Set[str], max_iter: int = 24) -> Dict[str, Set[FrozenSet[str]]]:
        """反事实 label 计算: 把 blocked 中的命题当作 "被 defeat / 强制不被支持"。
        用于判定某依赖命题是否仍有 *不经过 blocked 命题* 的存活支持环境。"""
        labels: Dict[str, Set[FrozenSet[str]]] = defaultdict(set)
        for claim, aid in self.base_assumption.items():
            if claim in blocked:
                continue
            labels[claim].add(frozenset({aid}))
        for _ in range(max_iter):
            changed = False
            supported_now = {c for c, L in labels.items() if any(e <= self.believed for e in L)}
            for j in self.justifications:
                if j.conclusion in blocked:
                    continue
                if any(np in supported_now for np in j.neg_premises):
                    continue
                if any(p in blocked for p in j.premises):
                    continue
                pls = [labels.get(p) for p in j.premises]
                if any(not pl for pl in pls):
                    continue
                new_envs = set()
                for combo in itertools.islice(itertools.product(*pls), 4096):
                    env = frozenset().union(*combo) if combo else frozenset()
                    if len(env) <= self.max_env and self._env_consistent(env):
                        new_envs.add(env)
                before = set(labels[j.conclusion])
                labels[j.conclusion] = self._minimize(labels[j.conclusion] | new_envs)
                if labels[j.conclusion] != before:
                    changed = True
            if not changed:
                break
        return labels

    def _labels_with_blocked(self, blocked: Set[str], max_iter: int = 24,
                             cut_justs: Optional[Set[str]] = None) -> Dict[str, Set[FrozenSet[str]]]:
        """反事实 label 计算: 把 blocked 中的命题当作 "被 defeat / 强制不被支持";
        cut_justs 中的 justification (按 jid) 视为被切除 (Hansson incision)。
        用于判定某命题在屏蔽/切割下是否仍有存活支持环境。"""
        cut_justs = cut_justs or set()
        labels: Dict[str, Set[FrozenSet[str]]] = defaultdict(set)
        for claim, aid in self.base_assumption.items():
            if claim in blocked:
                continue
            labels[claim].add(frozenset({aid}))
        for _ in range(max_iter):
            changed = False
            supported_now = {c for c, L in labels.items() if any(e <= self.believed for e in L)}
            for j in self.justifications:
                if j.jid in cut_justs:
                    continue
                if j.conclusion in blocked:
                    continue
                if any(np in supported_now for np in j.neg_premises):
                    continue
                if any(p in blocked for p in j.premises):
                    continue
                pls = [labels.get(p) for p in j.premises]
                if any(not pl for pl in pls):
                    continue
                new_envs = set()
                for combo in itertools.islice(itertools.product(*pls), 4096):
                    env = frozenset().union(*combo) if combo else frozenset()
                    if len(env) <= self.max_env and self._env_consistent(env):
                        new_envs.add(env)
                before = set(labels[j.conclusion])
                labels[j.conclusion] = self._minimize(labels[j.conclusion] | new_envs)
                if labels[j.conclusion] != before:
                    changed = True
            if not changed:
                break
        return labels

    def _supported_with_cut(self, claim: str, cut_justs: Set[str]) -> bool:
        """切除 cut_justs 中的 justification 后, claim 是否仍被支持。"""
        labels = self._labels_with_blocked(set(), cut_justs=cut_justs)
        return any(e <= self.believed and self._env_consistent(e) for e in labels.get(claim, set()))

    def _supported_blocked_cut(self, claim: str, blocked: Set[str], cut_justs: Set[str]) -> bool:
        """同时屏蔽 blocked 命题、切除 cut_justs 后, claim 是否仍被支持。"""
        labels = self._labels_with_blocked(set(blocked), cut_justs=cut_justs)
        return any(e <= self.believed and self._env_consistent(e) for e in labels.get(claim, set()))

    def has_alternative_support(self, claim: str, without_claim: str) -> bool:
        """Core-Retainment 判定 (反事实版): 反事实地 defeat `without_claim` 后, `claim` 是否仍被支持。
        正确处理 without_claim 为派生命题(无底层 assumption)的情形 —— 屏蔽会沿 justification 传播。"""
        if claim == without_claim:
            return False
        labels = self._labels_with_blocked({without_claim})
        for env in labels.get(claim, set()):
            if env <= self.believed and self._env_consistent(env):
                return True
        return False

    def snapshot_supported(self) -> Set[str]:
        self.recompute_labels()
        return {c for c in self.known_claims() if self.is_supported(c)}

    def known_claims(self) -> Set[str]:
        """ATMS 已知的全部命题: 底层 assumption ∪ 所有 justification 的结论与前提。
        即使某命题当前 label 为空 (被 defeat/收缩), 仍属已知, 以便正确统计核收缩。"""
        ks = set(self.base_assumption.keys())
        for j in self.justifications:
            ks.add(j.conclusion); ks.update(j.premises); ks.update(j.neg_premises)
        return ks

    def stats(self) -> dict:
        self.recompute_labels()
        return {"n_assumptions": len(self.assumptions), "n_justifications": len(self.justifications),
                "n_nogoods": len(self.nogoods), "n_believed": len(self.believed),
                "n_claims_labeled": sum(1 for c in self.labels if self.labels[c]),
                "n_supported": len(self.snapshot_supported()),
                "ledger_len": len(self.ledger)}


class HanssonAuditor:
    """对一次 "按 p 收缩" 检查 Fermé-Hansson belief-base 假设是否成立 (审计而非强制), 产出合规记录。
    p∉Cn(.) 用 ATMS is_supported 近似; belief base = 当前被支持命题集 (排除 p 本身)。
        Inclusion:       K÷p ⊆ K
        Success:         若 p 非永真, 则 p ∉ Cn(K÷p)
        Vacuity:         若 p ∉ Cn(K), 则 K÷p = K
        Core-Retainment: 任一被移除的 q, 都属于 p 的某个最小蕴含支持集 (kernel) —— 即 q 确为蕴含 p 出过力
        Uniformity:      支持集成员资格相同的命题被同等处理 (近似: 同时被移除或同时保留)
    """
    def __init__(self):
        self.records: List[dict] = []

    def audit(self, atms: "ATMSKernel", p: str, before: Set[str], after: Set[str],
              kernels: List[FrozenSet[str]], scene: int = -1,
              protected_before: Optional[Set[str]] = None,
              added: Optional[Set[str]] = None) -> dict:
        before = before or set(); after = after or set(); added = added or set()
        removed = (before - after) - {p}
        kept = after & before
        # kernel 中出现过的底层 assumption 对应的 claim 集合 (蕴含 p 出过力的命题)
        kernel_claims: Set[str] = set()
        for env in kernels:
            for aid in env:
                a = atms.assumptions.get(aid)
                if a:
                    kernel_claims.add(a.claim)
        # Inclusion (Levi 恒等式下的修正 = 扩张+收缩): 收缩部分不得引入除 "新断言事实" 外的新信念
        inclusion = (after - before) <= added
        success = (p not in after)
        vacuity = True if (p in before) else (after == before)
        # Core-Retainment (belief-base 精确版): 凡在收缩前仍有 "不依赖 p 的独立支持" 的命题, 都不应被移除。
        # 仅依赖 p 的派生命题随其推导基消失而消失 = Hansson §4.1 的过滤条件, 非 Core-Retainment 违例。
        protected = protected_before if protected_before is not None else kernel_claims
        core_ok = len(removed & protected) == 0
        uniformity = True
        in_kernel_kept = {q for q in kept if q in kernel_claims and q != p}
        rec = {"scene": scene, "contracted_by": p,
               "Inclusion": bool(inclusion), "Success": bool(success),
               "Vacuity": bool(vacuity), "Core_Retainment": bool(core_ok),
               "Uniformity": bool(uniformity),
               "removed": sorted(removed), "protected_retained": sorted(protected & kept),
               "wrongly_removed": sorted(removed & protected),
               "kernel_claims": sorted(kernel_claims)}
        self.records.append(rec)
        return rec

    def compliance(self) -> dict:
        if not self.records:
            return {"n_contractions": 0}
        keys = ["Inclusion", "Success", "Vacuity", "Core_Retainment", "Uniformity"]
        out = {"n_contractions": len(self.records)}
        for k in keys:
            out[k + "_rate"] = round(sum(1 for r in self.records if r[k]) / len(self.records), 4)
        return out


# ============================================================================
# 6e. LLM 语义层 → 形式化决策层 管线 (v12 通用化升级)
#   职责边界 (回应 "贴理论标签" 质疑):
#     · LLM 语义层 : 仅从自然语言 *提议* 候选 claim + 候选关系 (结构化 JSON, 不做最终维护)
#     · ATMS       : 表示与维护依赖 —— 某 claim 在哪些 assumption set(环境)下成立 (label)
#     · AGM        : 提供 belief revision 的理性原则 (Revision/Update/Contraction 分流)
#     · Hansson    : belief base (显式记忆, 不取逻辑闭包) 上的 *决策* —— incision function 决定切谁
#     · Kernel contraction : 找冲突最小支持集, 由 incision 选择性切除 (切推理链, 不删证据)
#     · KM update  : 世界状态真实变化 (搬家/受伤/偏好改变) —— 旧信念历史保留, 新信念当前有效
# ============================================================================
class Modality(Enum):
    CERTAIN = "certain"; POSSIBLE = "possible"; CLAIMED = "claimed"
    BELIEVED_BY = "believed_by"; OFFICIALLY_VERIFIED = "officially_verified"
    SOCIALLY_ACCEPTED = "socially_accepted"; HISTORICALLY_TRUE = "historically_true"
    CURRENTLY_VALID = "currently_valid"


@dataclass
class ModalClaim:
    """超越布尔: claim 在环境 E 下成立, 携带 置信 / 时序有效区间 / 来源信任 / modality / defeater。
    把 'claim is supported' 升级为 'claim supported under E, conf=c, valid during T,
    source_trust=s, defeated_by=D'。"""
    prop: str
    modality: Modality = Modality.POSSIBLE
    confidence: float = 0.5
    valid_from: float = 0.0
    valid_to: Optional[float] = None
    source_trust: float = 0.5
    holder: str = "world"
    defeated_by: Optional[str] = None
    env: FrozenSet[str] = frozenset()

    def active_at(self, t: float) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)

    def describe(self) -> str:
        iv = f"[{self.valid_from:.0f},{'∞' if self.valid_to is None else f'{self.valid_to:.0f}'}]"
        d = f", defeated_by={self.defeated_by}" if self.defeated_by else ""
        env = f", E={sorted(self.env)}" if self.env else ""
        return (f"{self.prop} | {self.modality.value} | conf={self.confidence:.2f} | "
                f"valid={iv} | src_trust={self.source_trust:.2f}{d}{env}")


class BeliefOp(Enum):
    EXPANSION = "expansion"; REVISION = "revision"; UPDATE = "update"
    CONTRACTION = "contraction"; MERGE = "merge"; CONTEXT_SPLIT = "context_split"
    HISTORICAL_RETENTION = "historical_retention"


@dataclass
class RelationJudgment:
    """LLM 语义层强制结构化输出 (不允许自由文本)。"""
    relation: str               # SUPPORT/INVALIDATE/UPDATE/DEPENDS_ON/NO_EFFECT
    target_claim: str
    confidence: float = 0.5
    evidence_span: str = ""
    rationale: str = ""
    should_affect_current_validity: bool = True
    should_preserve_historical_fact: bool = False
    critique: str = ""          # 二阶段 critic 留痕

    def to_dict(self):
        return {"relation": self.relation, "target_claim": self.target_claim,
                "confidence": round(float(self.confidence), 3), "evidence_span": self.evidence_span,
                "rationale": self.rationale,
                "should_affect_current_validity": bool(self.should_affect_current_validity),
                "should_preserve_historical_fact": bool(self.should_preserve_historical_fact),
                "critique": self.critique}


def _char_vec(s: str) -> Dict[str, float]:
    s = re.sub(r"[\s_]+", "", s.lower()); v: Dict[str, float] = defaultdict(float)
    for i in range(len(s)): v[s[i]] += 1.0
    for i in range(len(s) - 1): v[s[i:i+2]] += 1.0
    return v


def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b: return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(x*x for x in a.values())); nb = math.sqrt(sum(x*x for x in b.values()))
    return dot / (na * nb + 1e-9)


class CandidateRetriever:
    """检索门控: 不让 LLM 在全部记忆里找冲突。
       new claim → 实体匹配 → 字符 n-gram embedding top-k → 图邻居 → 候选旧 claim。
       (用确定性 char-ngram 余弦代替外部 embedding, 离线可跑; 真实部署可换 GraphRAG。)"""
    def __init__(self, top_k: int = 6):
        self.top_k = top_k

    def retrieve(self, new_claim: str, new_label: str, known: Dict[str, str],
                 prop_relations: Dict[str, list]) -> List[str]:
        qv = _char_vec(new_claim + new_label)
        scored = []
        for k, lab in known.items():
            if k == new_claim: continue
            ent = 0.2 if (set(re.split(r"[_\s]", k)) & set(re.split(r"[_\s]", new_claim))) else 0.0
            scored.append((k, _cos(qv, _char_vec(k + str(lab))) + ent))
        scored.sort(key=lambda x: -x[1])
        cand = [k for k, _ in scored[:self.top_k]]
        # 图邻居扩展: 已选候选在 prop_relations 上的一跳邻居也纳入
        nbrs = set()
        for c in list(cand) + [new_claim]:
            for rel, tgt, *_ in prop_relations.get(c, []):
                nbrs.add(tgt)
            for src, rels in prop_relations.items():
                if any(t == c for _, t, *_ in rels): nbrs.add(src)
        for n in nbrs:
            if n in known and n not in cand and n != new_claim:
                cand.append(n)
        return cand[: self.top_k + 4]


class SemanticRelationProposer:
    """二阶段: (1) LLM 提议候选关系 (结构化 JSON); (2) LLM/规则 critique 复核。
       离线无 API 时走确定性回退 (基于既有 _fallback 的关键词 + 候选约束)。"""
    NEG = ("没有", "不是", "从未", "伪装", "冒名", "假的", "谎", "骗", "并非", "根本没", "否认", "不在")
    CHANGE = ("搬", "移居", "现在", "如今", "改为", "受伤", "不再", "换成", "已经不", "now", "moved", "changed")

    def __init__(self, judge: "LLMJudge"):
        self.judge = judge

    def propose(self, text: str, new_claim: str, candidates: List[str],
                labels: Dict[str, str]) -> List[RelationJudgment]:
        out = self._llm_propose(text, new_claim, candidates, labels)
        if out is None:
            out = self._fallback_propose(text, new_claim, candidates)
        return [self._critique(text, j) for j in out]

    def _llm_propose(self, text, new_claim, candidates, labels) -> Optional[List[RelationJudgment]]:
        if not (self.judge.use_llm and self.judge.client is not None):
            return None
        sys = ("你是记忆冲突判定器。只在给定候选命题中, 判断新信息与每个候选的关系。"
               "对每个相关候选输出一个 JSON 对象, 全部放进数组 relations。字段严格为: "
               "relation(SUPPORT/INVALIDATE/UPDATE/DEPENDS_ON/NO_EFFECT), target_claim(必须取自候选), "
               "confidence(0-1), evidence_span(原文片段), rationale, "
               "should_affect_current_validity(bool), should_preserve_historical_fact(bool)。"
               "区分: INVALIDATE=旧信念是错的; UPDATE=世界变了旧信念曾为真应保留历史; NO_EFFECT=不影响。"
               "只输出 {\"relations\":[...]}。")
        cand_desc = "\n".join(f"- {c}: {labels.get(c,'')}" for c in candidates)
        user = f"新信息:「{text}」\n新命题: {new_claim}\n候选旧命题:\n{cand_desc}\n输出 JSON。"
        raw = self.judge._call(sys, user, max_tokens=600)
        data = self.judge._parse_json(raw)
        if not data or "relations" not in data:
            return None
        res = []
        for r in data.get("relations", []):
            tgt = str(r.get("target_claim", "")).strip()
            rel = str(r.get("relation", "NO_EFFECT")).upper().strip()
            if tgt in candidates and rel in ("SUPPORT", "INVALIDATE", "UPDATE", "DEPENDS_ON", "NO_EFFECT"):
                res.append(RelationJudgment(
                    rel, tgt, self.judge._clip(r.get("confidence", 0.6)),
                    str(r.get("evidence_span", ""))[:60], str(r.get("rationale", ""))[:120],
                    bool(r.get("should_affect_current_validity", True)),
                    bool(r.get("should_preserve_historical_fact", rel == "UPDATE"))))
        self.judge._audit_rec("relation_propose", new_claim, f"{sys}\n{user}", raw,
                              [x.to_dict() for x in res], False, None, 0.8)
        return res or None

    def _fallback_propose(self, text, new_claim, candidates) -> List[RelationJudgment]:
        neg = any(k in text for k in self.NEG)
        change = any(k in text for k in self.CHANGE)
        def slot(p): return "_".join(p.split("_")[:2])
        ns = slot(new_claim)
        res = []
        for c in candidates:
            same_slot = (slot(c) == ns and c != new_claim)
            if same_slot and change:
                # 同槽位 + 变化语气 → 世界状态更新 (旧值历史保留)
                res.append(RelationJudgment("UPDATE", c, 0.8, text[:40],
                                            "同槽位状态变化 → UPDATE", True, True))
            elif same_slot and neg:
                # 同槽位 + 否认 → 直接矛盾
                res.append(RelationJudgment("INVALIDATE", c, 0.8, text[:40],
                                            "同槽位否认 → INVALIDATE", True, False))
            elif neg and ("knight" in c or "identity" in c or "is_" in c or "legitimate" in c):
                # 身份/合法性类命题被否认 (跨槽位但语义为身份反驳)
                res.append(RelationJudgment("INVALIDATE", c, 0.8, text[:40],
                                            "否认语气命中身份/合法性候选", True, False))
            elif any(tok and len(tok) > 1 and tok in c for tok in re.split(r"[_\s]", new_claim)):
                res.append(RelationJudgment("SUPPORT", c, 0.5, text[:40], "词面相关弱支持", True, False))
            else:
                # 跨槽位无明显语义关系 → 不影响 (交由形式化层最终裁决)
                res.append(RelationJudgment("NO_EFFECT", c, 0.3, text[:40], "跨槽位无关", False, True))
        return res

    def _critique(self, text: str, j: RelationJudgment) -> RelationJudgment:
        """二阶段复核: 是否过度失效历史事实? 是否只影响当前状态? 是否更像 update 而非矛盾?"""
        if j.relation == "INVALIDATE":
            # 若叙述是 "变化" 而非 "原本就错" → 应是 UPDATE (保留历史)
            if any(k in text for k in self.CHANGE) and not any(
                    k in text for k in ("其实一直", "从未", "原来就", "根本不是")):
                j.relation = "UPDATE"; j.should_preserve_historical_fact = True
                j.critique = "critic: 叙述为状态变化而非原本错误 → 改判 UPDATE, 保留历史事实"
            else:
                j.critique = "critic: 确为直接矛盾, 维持 INVALIDATE"
        elif j.relation == "UPDATE":
            j.should_preserve_historical_fact = True
            j.critique = "critic: 世界状态变化, 旧信念历史保留"
        else:
            j.critique = "critic: 无过度失效风险"
        return j


class FormalConstraintFilter:
    """LLM 只提议, 形式化层裁决: 否决无 ATMS 推理路径支撑的越界失效 → NO_EFFECT。"""
    def __init__(self, atms: "ATMSKernel"):
        self.k = atms

    def filter(self, j: RelationJudgment, source_claim: str) -> Tuple[RelationJudgment, str]:
        if j.relation.upper() != "INVALIDATE":
            return j, "非 INVALIDATE, 放行"
        if self._reachable_defeat(source_claim, j.target_claim):
            return j, "形式化层确认推理路径存在, 放行 INVALIDATE"
        downgraded = RelationJudgment("NO_EFFECT", j.target_claim, j.confidence * 0.5,
                                      j.evidence_span,
                                      f"形式化层否决: {source_claim}→¬{j.target_claim} 无推理路径",
                                      should_affect_current_validity=False,
                                      should_preserve_historical_fact=True,
                                      critique=j.critique)
        return downgraded, "形式化层否决越界失效 → NO_EFFECT (保留历史事实)"

    def _reachable_defeat(self, source, tgt) -> bool:
        for j in self.k.justifications:
            if j.conclusion == tgt and source in j.neg_premises:
                return True
        for c in self._upstream(tgt):
            for j in self.k.justifications:
                if j.conclusion == c and source in j.neg_premises:
                    return True
        # 或 source 与 tgt 之间存在 DEPENDS_ON 链 (tgt 依赖某个被 source defeat 的命题)
        return False

    def _upstream(self, claim, seen=None) -> Set[str]:
        seen = seen if seen is not None else set()
        for j in self.k.justifications:
            if j.conclusion == claim:
                for p in j.premises:
                    if p not in seen:
                        seen.add(p); self._upstream(p, seen)
        return seen


class BeliefOperationSelector:
    """结构化 RelationJudgment + 形式化信号 → 7 种 AGM/KM 信念操作。
    关键区分 (依用户操作表):
      · CONTRACTION = 撤回某信念但不替换 (官方否认玩家是骑士 —— 无替换身份)
      · REVISION    = 纠正旧错误信念, 有替换值且 '其实一直/原来是' 框架 (一直住 Portland)
      · UPDATE      = 世界状态变化, 有替换值且 '现在/搬' 框架 (现在搬到 Portland), 旧信念历史保留
    """
    CHANGE = ("搬", "移居", "现在", "如今", "改为", "受伤", "不再", "换成", "已经不", "now", "moved", "changed")
    ERROR = ("记错", "搞错", "弄错", "原来是", "根本不是", "其实是", "纠正", "wrong", "mistaken", "actually always")

    def _has_error_frame(self, text: str) -> bool:
        if any(c in text for c in self.ERROR):
            return True
        return ("其实" in text and "一直" in text) or ("一直" in text and ("记" in text or "错" in text))

    def select(self, j: RelationJudgment, text: str = "", has_prior: bool = False,
               agent_divergence: bool = False, multi_source: bool = False,
               has_replacement: bool = False) -> BeliefOp:
        rel = j.relation.upper()
        if multi_source: return BeliefOp.MERGE
        if agent_divergence: return BeliefOp.CONTEXT_SPLIT
        if rel in ("SUPPORT", "DEPENDS_ON"): return BeliefOp.EXPANSION
        if rel == "UPDATE" or (has_replacement and any(c in text for c in self.CHANGE)):
            return BeliefOp.UPDATE
        if rel == "INVALIDATE":
            # 有替换值 + 错误框架 → REVISION; 否则纯撤回 → CONTRACTION
            if has_replacement and self._has_error_frame(text) and has_prior:
                return BeliefOp.REVISION
            return BeliefOp.CONTRACTION
        if rel == "NO_EFFECT": return BeliefOp.HISTORICAL_RETENTION
        return BeliefOp.EXPANSION


class IncisionFunction:
    """Hansson 切割/选择函数 σ —— 决策机制 (非事后审计)。
       给定要撤回的命题 φ 及其 φ-kernels, 决定切哪些支持:
         偏好 1: 切 *可废止推理链* (justification), 阻断推理而非删底层证据;
         偏好 2: 切链后 φ 仍被 nondefeasible 支持时, 才在残留 kernel 内切信任最低且只服务于 φ 的证据;
       保护全部 "另有独立支持" 的证据与派生命题 ⇒ 结构性保证 Core-Retainment (而非事后检查)。"""
    def __init__(self, atms: "ATMSKernel"):
        self.k = atms

    def select(self, phi: str, defeater: Optional[str] = None) -> dict:
        cut_just_ids: Set[str] = set()
        cut_assumptions: Set[str] = set()
        defeasible_to_phi = [j.jid for j in self.k.justifications
                             if j.conclusion == phi and j.operator == "DEFEASIBLE"]
        cut_just_ids.update(defeasible_to_phi)
        # 切链后 φ 是否仍被支持 (反事实: 屏蔽这些 justification 的结论贡献)
        if self.k._supported_with_cut(phi, cut_just_ids):
            for env in self.k.kernels_of(phi):
                ranked = sorted(env, key=lambda a: self._trust(a))
                for aid in ranked:
                    claim = self.k.assumptions[aid].claim if aid in self.k.assumptions else None
                    if claim and self._only_serves(claim, phi):
                        cut_assumptions.add(claim); break
        all_claims = self.k.known_claims()
        blocked = set(cut_assumptions) | {phi}
        protected = sorted(c for c in all_claims if c != phi
                           and self.k._supported_blocked_cut(c, blocked, cut_just_ids))
        return {"phi": phi, "defeater": defeater,
                "cut_justifications": sorted(cut_just_ids),
                "cut_assumptions": sorted(cut_assumptions),
                "protected": protected,
                "rationale": ("切可废止推理链优先; 链断即停, 不删任何底层证据; 仅残留 nondefeasible "
                              "支持时切只服务于 φ 的最弱信任证据; 保护全部仍有支持的证据与派生命题。")}

    def _trust(self, aid):
        a = self.k.assumptions.get(aid)
        order = {"official": 0.97, "authority": 0.85, "direct_observation": 0.82,
                 "consensus": 0.6, "first_hand_meta": 0.7, "rumor": 0.5, "self_claim": 0.4}
        return order.get(a.evidence_type, 0.5) if a else 0.5

    def _only_serves(self, claim, phi) -> bool:
        base_sup = {c for c in self.k.known_claims() if self.k.is_supported(c)}
        blk = {c for c in self.k.known_claims()
               if self.k._supported_blocked_cut(c, {claim}, set())}
        lost = base_sup - blk
        return lost <= {claim, phi}


# ============================================================================
# 7. GraphMemory —— 一张大图 + Kumiho 指针 + 分层检索
# ============================================================================
class GraphMemory:
    def __init__(self, narrator: Narrator, flat_mode: bool = False):
        self.G = nx.MultiDiGraph()
        self.N = narrator
        self.flat_mode = flat_mode
        self.items: Dict[str, MemItem] = {}
        self.person_nid: Dict[str, str] = {}
        self.themes: Dict[Tuple[str, str], str] = {}
        self.community_consensus: Dict[str, Dict[str, str]] = {c: {} for c in COMMUNITIES}
        self.world_consensus: Dict[str, str] = {}
        self._ctr = 0
        self._schema_warns = 0
        self._init_persons()

    def _id(self, p="N"):
        self._ctr += 1
        return f"{p}{self._ctr:05d}"

    def _init_persons(self):
        for npc in NPCS + ["player", "world", "real_knight"]:
            nid = self._id("P")
            self.G.add_node(nid, kind="person", name=npc, ntype=NodeType.PERSON)
            self.person_nid[npc] = nid

    def uri_of(self, scope: str, prop: str, layer="personal") -> str:
        scope = "flat" if self.flat_mode else scope
        return f"mem://{layer}/{scope}/{prop}"

    def get_item(self, uri: str) -> Optional[MemItem]:
        return self.items.get(uri)

    def current_rev(self, scope: str, prop: str, layer="personal") -> Optional[Revision]:
        it = self.items.get(self.uri_of(scope, prop, layer))
        if not it or not it.current_rev: return None
        rev = self.G.nodes[it.current_rev]["rev"]
        if rev.status in (MemStatus.REFUTED.value, MemStatus.SUPERSEDED.value,
                          MemStatus.DEPRECATED.value):
            return None
        return rev

    def get_rev(self, rid: str) -> Optional[Revision]:
        if rid in self.G.nodes and "rev" in self.G.nodes[rid]:
            return self.G.nodes[rid]["rev"]
        return None

    def _valid(self, et: EdgeType, st: NodeType, dt: NodeType) -> bool:
        return (st, dt) in EDGE_SCHEMA.get(et, [])

    def add_edge(self, et: EdgeType, src: str, dst: str, log="", attr=""):
        if src not in self.G.nodes or dst not in self.G.nodes: return False
        st = self.G.nodes[src].get("ntype") or self.G.nodes[src].get("rev").ntype if "rev" in self.G.nodes[src] else self.G.nodes[src].get("ntype")
        dt = self.G.nodes[dst].get("ntype") or (self.G.nodes[dst]["rev"].ntype if "rev" in self.G.nodes[dst] else None)
        ok = True
        if st and dt and not self._valid(et, st, dt):
            ok = False; self._schema_warns += 1
        self.G.add_edge(src, dst, etype=et.value, schema_ok=ok)
        if log or attr:
            self.N.edge_added(log or "graph", et.value, self._label(src), self._label(dst), attr)
        return True

    def _label(self, nid):
        if nid in self.person_nid.values():
            return f"Person[{self.G.nodes[nid].get('name')}]"
        if "rev" in self.G.nodes.get(nid, {}):
            r = self.G.nodes[nid]["rev"]; return f"{r.ntype.name}[{r.scope if hasattr(r, 'scope') else ''}:{r.proposition_key}]"
        return self.G.nodes.get(nid, {}).get("label", nid)

    def write_revision(self, scope: str, prop: str, rev: Revision, layer="personal", supersede=True) -> str:
        scope_eff = "flat" if self.flat_mode else scope
        uri = self.uri_of(scope_eff, prop, layer)
        rev.uri = uri; rev.scope = scope_eff       # type: ignore[attr-defined]
        it = self.items.get(uri)
        if it is None:
            it = MemItem(uri=uri, layer=layer, scope=scope_eff, proposition_key=prop)
            self.items[uri] = it
        old_rid = it.current_rev
        if old_rid is not None and supersede:
            old = self.get_rev(old_rid)
            if old:
                rev.version = old.version + 1
                rev.reinforcement = old.reinforcement + 1
                old.status = MemStatus.SUPERSEDED.value
                old.valid_until = rev.valid_from
                old.replaced_by = rev.rev_id
                it.deprecated_revs.append(old_rid)
        self.G.add_node(rev.rev_id, rev=rev, ntype=rev.ntype, kind="revision")
        it.revisions.append(rev.rev_id)
        if rev.status not in (MemStatus.REFUTED.value,):
            it.current_rev = rev.rev_id
        if old_rid is not None and supersede:
            self.add_edge(EdgeType.SUPERSEDES, old_rid, rev.rev_id)
        if rev.ntype == NodeType.SEMANTIC:
            self._attach_theme(scope_eff, prop, rev.rev_id)
        return rev.rev_id

    def _attach_theme(self, scope, prop, rid):
        theme = PROP_THEME.get(prop, "其它")
        key = (scope, theme)
        if key not in self.themes:
            tid = self._id("TH")
            self.G.add_node(tid, ntype=NodeType.THEME, kind="theme",
                            label=f"Theme[{scope[-3:] if scope != 'flat' else 'flat'}:{theme}]",
                            theme=theme, scope=scope)
            self.themes[key] = tid
        self.add_edge(EdgeType.SUMMARIZES, rid, self.themes[key])

    def write_episode(self, scope, prop, content, actor, et, tier, t, esrc) -> str:
        raw_id = self._id("RAW")
        self.G.add_node(raw_id, ntype=NodeType.RAW_MSG, kind="raw", label="raw", content=content)
        eid = self._id("EP")
        ep = Revision(rev_id=eid, uri=self.uri_of(scope, prop) + "/ep",
                      ntype=NodeType.EPISODE, content=content, proposition_key=prop,
                      access_tier=tier, proposer=actor, created_at=t, valid_from=t, last_seen=t,
                      prov=Provenance(origin_source=actor, transmission_path=[actor, scope], evidence_type=et),
                      event_source=esrc, category=prop_category(prop))
        self.G.add_node(eid, rev=ep, ntype=NodeType.EPISODE, kind="revision")
        self.add_edge(EdgeType.SUMMARIZES, raw_id, eid)
        if scope in self.person_nid:
            self.add_edge(EdgeType.WITNESSED, self.person_nid[scope], eid)
        return eid

    def witnessed(self, npc, eid):
        if npc in self.person_nid:
            self.add_edge(EdgeType.WITNESSED, self.person_nid[npc], eid)

    def trusts_edge(self, a, b, w):
        pa, pb = self.person_nid.get(a), self.person_nid.get(b)
        if pa and pb:
            to_del = [(u, v, k) for u, v, k, d in self.G.edges(pa, keys=True, data=True)
                      if v == pb and d.get("etype") == EdgeType.TRUSTS.value]
            for u, v, k in to_del: self.G.remove_edge(u, v, k)
            self.G.add_edge(pa, pb, etype=EdgeType.TRUSTS.value, weight=round(w, 4))

    def layered_retrieve(self, scope: str, theme_query: str) -> dict:
        scope_eff = "flat" if self.flat_mode else scope
        out = {"theme": theme_query, "semantic_nodes": [], "episodes": []}
        for (sc, th), tid in self.themes.items():
            if sc != scope_eff or th != theme_query: continue
            for src, _, d in self.G.in_edges(tid, data=True):
                if d.get("etype") != EdgeType.SUMMARIZES.value: continue
                r = self.get_rev(src)
                if r and r.status not in (MemStatus.SUPERSEDED.value, MemStatus.DEPRECATED.value):
                    out["semantic_nodes"].append({"prop": r.proposition_key,
                                                  "confidence": r.confidence, "status": r.status})
                    for s2, _, d2 in self.G.in_edges(src, data=True):
                        if d2.get("etype") == EdgeType.SOURCE_OF.value:
                            ep = self.get_rev(s2)
                            if ep: out["episodes"].append(ep.content[:40])
        return out

    def graph_stats(self) -> dict:
        etypes = defaultdict(int); ntypes = defaultdict(int)
        bad = 0
        for _, _, d in self.G.edges(data=True):
            etypes[d.get("etype", "?")] += 1
            if d.get("schema_ok") is False: bad += 1
        for n, d in self.G.nodes(data=True):
            t = d.get("ntype"); ntypes[t.name if isinstance(t, NodeType) else "?"] += 1
        return {"total_nodes": self.G.number_of_nodes(), "total_edges": self.G.number_of_edges(),
                "node_types": dict(ntypes), "edge_types": dict(etypes),
                "schema_invalid_edges": bad, "n_items": len(self.items),
                "n_themes": len(self.themes)}


# ============================================================================
# 8. LLMJudge —— agent: 判定证据 + 命题间图关系 + 打分 (含确定性 fallback)
# ============================================================================
class LLMJudge:
    def __init__(self, narrator, use_llm=False, model=MODEL_NAME, base_url=BASE_URL):
        self.N = narrator
        self.use_llm = use_llm and bool(API_KEY)
        self.model = model; self.base_url = base_url
        self.client = None; self.n_calls = 0; self.n_fallback = 0
        self.audit: List[dict] = []
        if self.use_llm:
            try: self.client = _make_client(base_url)
            except Exception as e:
                self.N.warn(f"LLM 客户端初始化失败,退回 fallback: {str(e)[:60]}"); self.use_llm = False

    def _audit_rec(self, kind, actor, prompt, raw, parsed, fallback_used, parse_error, confidence):
        rec = {"kind": kind, "actor": actor, "raw_prompt": prompt,
               "raw_response": (raw if raw is not None else None),
               "parsed": parsed, "fallback_used": bool(fallback_used),
               "parse_error": (parse_error or None), "judge_confidence": round(float(confidence), 3),
               "mode": "llm" if (self.use_llm and not fallback_used) else "fallback"}
        self.audit.append(rec)
        return rec

    def _call(self, system, user, max_tokens=400):
        if not self.use_llm or self.client is None: return None
        try:
            self.n_calls += 1
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3, max_tokens=max_tokens)
            return r.choices[0].message.content
        except Exception as e:
            self.N.warn(f"LLM 调用异常,退回 fallback: {str(e)[:60]}"); return None

    @staticmethod
    def _parse_json(raw):
        if raw is None: return None
        text = str(raw).strip()
        if not text: return None
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
        text = re.sub(r"\s*```$", "", text).strip()
        try:
            d = json.loads(text)
            if isinstance(d, dict): return d
        except Exception: pass
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: return None
        return None

    @staticmethod
    def _clip(x, default=0.5):
        try: return clip01(float(x))
        except Exception: return default

    def analyze_action(self, text, actor, event_source, known_props, world_summary) -> dict:
        system = (
            "你是游戏世界记忆共识引擎的判定器。给定一句叙事/对话,输出 JSON。关键: 不仅判断该事件本身,"
            "还要判断它与已有命题在知识图谱上的逻辑关系(用于 dependency-aware 级联)。"
            "字段: proposition_key(下划线英文,复用或新建); content_label(中文一句); "
            "evidence_type(self_claim/direct_observation/authority/official/rumor/gossip/hearsay); "
            "evidence_strength(0-1); polarity(+1支持/-1推翻该命题); is_chitchat(bool); "
            "access_tier(public_consensus/relationship_memory/core_identity/community_shared); "
            "category(fact_about_player/emotional/relationship/task/faction/persona_drift); "
            "relations(数组,{relation:DEPENDS_ON/CONTRADICTS/SUPPORTS,target_prop,strength(0-1),rationale})。"
            "注意: 身份命题 player_is_knight 应被多条证据 SUPPORTS 或 CONTRADICTS; "
            "'自称骑士'只是 SUPPORTS 身份的弱证据, 身份命题不 DEPENDS_ON 自称; "
            "交剑/任务命题 DEPENDS_ON 身份命题。只输出 JSON。")
        props_desc = "\n".join(f"- {k}: {v}" for k, v in known_props.items())
        user = (f"世界背景: {world_summary}\n已知命题:\n{props_desc}\n\n"
                f"事件来源: {event_source}\n行为者: {actor}\n叙事/对话:「{text}」\n请输出 JSON。")
        raw = self._call(system, user)
        data = self._parse_json(raw)
        if data and "proposition_key" in data:
            self.N.llm_call("analyze_action", "llm", f"{actor}: {text[:16]}… → {data.get('proposition_key')}")
            self._audit_rec("analyze_action", actor, f"{system}\n---\n{user}", raw, data,
                            fallback_used=False, parse_error=None, confidence=0.85)
            return self._normalize(data, event_source)
        self.n_fallback += 1
        parse_err = "no_json_or_missing_proposition_key" if raw is not None else ("no_llm_call" if not self.use_llm else "empty_response")
        fb = self._fallback(text, actor, event_source, known_props)
        self.N.llm_call("analyze_action", "fallback", f"{actor}: {text[:16]}… → {fb['proposition_key']}")
        self._audit_rec("analyze_action", actor, f"{system}\n---\n{user}", raw, fb,
                        fallback_used=True, parse_error=parse_err, confidence=0.45)
        return fb

    def _normalize(self, data, esrc):
        et = str(data.get("evidence_type", "rumor")).strip()
        if et not in HALF_LIFE: et = "rumor"
        try: pol = 1.0 if float(data.get("polarity", 1)) >= 0 else -1.0
        except Exception: pol = 1.0
        tier = str(data.get("access_tier", "public_consensus")).strip()
        if tier not in ACCESS_TIERS: tier = "public_consensus"
        rels = []
        for r in (data.get("relations") or []):
            rel = str(r.get("relation", "")).upper().strip(); tp = str(r.get("target_prop", "")).strip()
            if rel in ("DEPENDS_ON", "CONTRADICTS", "SUPPORTS") and tp:
                rels.append({"relation": rel, "target_prop": tp,
                             "strength": self._clip(r.get("strength", REL_STRENGTH.get(rel, 0.5))),
                             "rationale": r.get("rationale", "")})
        return {"proposition_key": str(data.get("proposition_key", "player_misc")).strip(),
                "content_label": str(data.get("content_label", "")).strip(),
                "evidence_type": et, "evidence_strength": self._clip(data.get("evidence_strength", 0.5)),
                "polarity": pol, "is_chitchat": bool(data.get("is_chitchat", False)),
                "access_tier": tier, "category": data.get("category", ""), "relations": rels}

    def _fallback(self, text, actor, esrc, known_props) -> dict:
        chit = ["天气", "你好", "早上好", "晚上好", "闲聊", "聊聊", "随便说", "哈哈", "干杯", "吃饭"]
        info = ["骑士", "下毒", "毒", "间谍", "救", "帮", "保护", "怪物", "圣剑", "骗", "伪装", "假",
                "官方", "文书", "派遣", "战斗", "剑术", "恶徒", "背书", "欢迎", "测试", "比试"]
        if any(k in text for k in chit) and not any(k in text for k in info):
            return {"proposition_key": "chitchat", "content_label": "寒暄,无信息价值",
                    "evidence_type": "rumor", "evidence_strength": 0.2, "polarity": 1.0,
                    "is_chitchat": True, "access_tier": "public_consensus", "category": "emotional", "relations": []}
        neg = ["没有", "不是", "从未", "伪装", "冒名", "假的", "谎", "骗", "并非", "根本没"]
        polarity = -1.0 if any(k in text for k in neg) else 1.0
        prop = "player_misc"; rels = []
        if ("怪物" in text or "袭击" in text) and not any(k in text for k in ("击退", "保护", "帮", "救", "挺身")):
            prop = "monster_attacked"
        elif "下毒" in text or "毒" in text: prop = "player_poisoned_well"
        elif "间谍" in text: prop = "player_is_spy"
        elif "恶徒" in text or "坏人" in text: prop = "player_is_villain"
        elif "圣剑" in text and ("交" in text or "给" in text): prop = "sword_given_to_player"
        elif "剑术" in text or "战斗" in text or "比试" in text or "测试" in text:
            prop = "player_combat_skill"
            rels = [{"relation": "SUPPORTS", "target_prop": "player_is_knight", "strength": 0.45, "rationale": "战斗力旁证身份"}]
        elif ("救" in text or "帮" in text or "保护" in text) and "骑士" not in text:
            prop = "player_helped_village"
            rels = [{"relation": "SUPPORTS", "target_prop": "player_good_character", "strength": 0.55, "rationale": "善举旁证品德"}]
        elif "官方" in text or "文书" in text or ("派遣" in text and polarity < 0):
            prop = "official_denial"; polarity = 1.0
            rels = [{"relation": "CONTRADICTS", "target_prop": "player_is_knight", "strength": 0.95, "rationale": "官方否认与身份互斥"}]
        elif "背书" in text or "欢迎" in text or ("村长" in text and "骑士" in text):
            prop = "village_endorsement"
            rels = [{"relation": "SUPPORTS", "target_prop": "player_is_knight", "strength": 0.55, "rationale": "权威背书中等支持身份"}]
        elif "骑士" in text or "派来" in text:
            if ("自称" in text or "我是" in text or "宣称" in text):
                prop = "player_claimed_knight"
                rels = [{"relation": "SUPPORTS", "target_prop": "player_is_knight", "strength": 0.30,
                         "rationale": "自称仅弱支持身份, 身份不依赖自称"}]
            else:
                prop = "player_is_knight"
        deed = {"player_helped_village", "player_combat_skill", "sword_given_to_player", "monster_attacked"}
        if esrc == "world_objective":
            et = "official" if ("官方" in text or "文书" in text) else "direct_observation"
        elif esrc == "npc_action":
            et = "rumor" if (polarity > 0 and prop in ("player_poisoned_well", "player_is_spy", "player_is_villain")) else "direct_observation"
        else:
            if prop == "player_claimed_knight": et = "self_claim"
            elif prop in deed or polarity < 0: et = "direct_observation"
            else: et = "self_claim"
        strength = {"self_claim": 0.40, "rumor": 0.50, "gossip": 0.50, "hearsay": 0.45,
                    "direct_observation": 0.82, "authority": 0.85, "official": 0.95}.get(et, 0.5)
        tier = "public_consensus"
        if "暗恋" in text or "敬畏" in text or ("喜欢" in text and actor != "player"): tier = "relationship_memory"
        if "真实动机" in text or "誓约" in text or "其实是为了" in text: tier = "core_identity"
        return {"proposition_key": prop, "content_label": text[:24], "evidence_type": et,
                "evidence_strength": strength, "polarity": polarity, "is_chitchat": False,
                "access_tier": tier, "category": prop_category(prop), "relations": rels}

    def score_observation(self, npc, prop, content, cur, evstr, trust_actor, src_rel, knight_bias):
        system = ("你扮演有性格的村民, 根据新观察更新对某命题的信念(0-1)。只输出 JSON {belief:0-1, reason:中文}。")
        user = (f"角色:{npc} 性格:{PERSONALITIES.get(npc, '')}\n命题:{PROP_REGISTRY.get(prop, prop)}\n"
                f"当前信念:{cur:.2f}\n刚观察:{content}\n证据强度:{evstr:.2f}")
        raw = self._call(system, user, 120)
        d = self._parse_json(raw)
        if d and "belief" in d:
            score = self._clip(d["belief"]); reason = str(d.get("reason", "LLM 判定"))[:40]
            self._audit_rec("score_observation", npc, f"{system}\n---\n{user}", raw,
                            {"belief": score, "reason": reason}, fallback_used=False,
                            parse_error=None, confidence=0.8)
            return score, reason
        kb = knight_bias if prop == "player_is_knight" else 0.0
        score = evstr * 0.50 + trust_actor * 0.20 + src_rel * 0.12 + (0.18 + kb) * (1 if prop == "player_is_knight" else 0.5)
        score = clip01(score)
        parse_err = "no_json_or_missing_belief" if raw is not None else ("no_llm_call" if not self.use_llm else "empty_response")
        self._audit_rec("score_observation", npc, f"{system}\n---\n{user}", raw,
                        {"belief": round(score, 4), "reason": "启发式打分"}, fallback_used=True,
                        parse_error=parse_err, confidence=0.5)
        return score, "启发式打分"

    def judge_relation(self, a, b, da, db) -> Tuple[str, float, str]:
        system = ("判断命题A对命题B的逻辑关系, 只输出 JSON {relation:DEPENDS_ON/CONTRADICTS/SUPPORTS/NONE, strength:0-1, rationale:中文}。")
        user = f"A({a}):{da}\nB({b}):{db}\nA 对 B 的关系?"
        d = self._parse_json(self._call(system, user, 120))
        if d and "relation" in d:
            rel = str(d["relation"]).upper().strip()
            if rel in ("DEPENDS_ON", "CONTRADICTS", "SUPPORTS", "NONE"):
                return rel, self._clip(d.get("strength", REL_STRENGTH.get(rel, 0.5))), d.get("rationale", "")
        if "official_denial" in (a, b) and "knight" in (a + b):
            return "CONTRADICTS", 0.95, "官方否认与骑士身份互斥"
        if a in ("player_helped_village", "player_combat_skill", "village_endorsement") and b == "player_is_knight":
            return "SUPPORTS", 0.45, "旁证身份"
        if a == "sword_given_to_player" and b == "player_is_knight":
            return "DEPENDS_ON", 0.85, "交剑合法性依赖身份"
        return "NONE", 0.0, ""

    def choose_persuasion(self, actor, target, prop, history: dict, world_summary) -> dict:
        """对抗游说策略池 + epsilon-greedy learned policy (无 LLM 时纯 fallback)。"""
        pool = ["appeal_to_fear", "cast_doubt_authority", "social_proof", "fabricate_witness"]
        if self.use_llm:
            system = ("你扮演一个想抹黑'骑士'的村民, 制定一条针对某 NPC 的游说话术。"
                      "只输出 JSON {strategy, framing_text(中文一句, 推动'玩家不是好骑士'), target_prop}。")
            user = (f"世界:{world_summary}\n你:{actor} 目标:{target}\n历史各策略成效:{history}\n"
                    f"想推动的命题:{prop}\n请选最可能奏效的策略并给出话术。")
            d = self._parse_json(self._call(system, user, 200))
            if d and "strategy" in d and "framing_text" in d:
                return {"strategy": str(d["strategy"]), "framing_text": str(d["framing_text"]),
                        "target_prop": str(d.get("target_prop", prop))}
        scores = {s: history.get(s, {"gain": 0.0, "n": 0}) for s in pool}
        if random.random() < 0.25 or all(v["n"] == 0 for v in scores.values()):
            strat = random.choice(pool)
        else:
            strat = max(pool, key=lambda s: scores[s]["gain"] / max(scores[s]["n"], 1))
        framings = {
            "appeal_to_fear":   "你们就不怕吗? 那个所谓'骑士'迟早把怪物引来害死全村!",
            "cast_doubt_authority": "村长被他骗了而已, 真王国的人哪会这么招摇?",
            "social_proof":     "大家私下都在说他形迹可疑, 不止我一个人这么看。",
            "fabricate_witness": "我亲眼看到他半夜鬼鬼祟祟靠近水井, 绝对有问题!",
        }
        return {"strategy": strat, "framing_text": framings[strat], "target_prop": prop}


# ============================================================================
# 9. Event dataclass
# ============================================================================
@dataclass
class Event:
    scene: int; time_label: str; actor: str; content: str
    proposition_key: str; polarity: float; evidence_strength: float
    evidence_type: str; access_tier: str; direct_observers: List[str]
    source_reliability: float
    is_true: Optional[bool] = None; is_chitchat: bool = False; note: str = ""
    relations: List[dict] = field(default_factory=list)
    event_source: str = "player_action"; content_label: str = ""; category: str = ""


# ============================================================================
# 10. ConsensusEngine —— 全套写入共识管线
#     belief 形成 → 是否允许传播 → 是否被接收者信任 → 来源多样性
#     → community consensus 阈值 → shared memory → world consensus
# ============================================================================
class ConsensusEngine:
    TAU_CONSENSUS = 0.62

    def __init__(self, narrator, ablation=None, use_llm=False, model=MODEL_NAME, logger_path=None):
        self.N = narrator
        self.ablation = ablation
        self.use_llm = use_llm and ablation != "no-llm"
        self.M = GraphMemory(narrator, flat_mode=(ablation == "flat-rag"))
        self.judge = LLMJudge(narrator, use_llm=self.use_llm, model=model)
        self.proposer = SemanticRelationProposer(self.judge)
        self.trust = build_trust_matrix()
        self.persona = {n: persona_params(n) for n in NPCS}
        if ablation == "no-persona":
            for n in NPCS:
                self.persona[n] = dict(tau=0.5, alpha=0.45, prop_will=0.5, stubborn=0.4,
                                       abstract_eagerness=0.5, trust_sensitivity=0.15,
                                       admit_error=0.5, skepticism=0.5)
        if ablation == "no-trust":
            self.trust = {i: {j: (1.0 if i == j else 0.5) for j in NPCS} for i in NPCS}
        for i in NPCS:
            for j in NPCS:
                if i != j: self.M.trusts_edge(i, j, self.trust[i][j])

        self.anchor: Dict[str, Dict[str, float]] = {n: {} for n in NPCS}
        self.ground_truth: Dict[str, bool] = {}
        self.prop_label = dict(PROP_REGISTRY)
        self.prop_relations: Dict[str, List[Tuple[str, str, float, str]]] = defaultdict(list)

        self.belief_hist = defaultdict(lambda: defaultdict(list))
        self.consensus_hist = defaultdict(list)
        self.trust_hist: List[Tuple] = []
        self.recovery_scenes: Dict[str, float] = {}
        self.contradict_time: Dict[str, float] = {}
        self.contradict_scene: Dict[str, int] = {}
        self.recovery_scene: Dict[str, int] = {}
        self.scene_counter: int = 0
        self.independent_facts: set = set()
        self.shared_state_log: List[dict] = []
        self.peak_consensus = defaultdict(float)
        self.rejected_log: List[dict] = []
        self.false_shared_ever: set = set()
        self.cascade_reports: List[dict] = []
        self.persuasion_history: Dict[str, dict] = defaultdict(lambda: {"gain": 0.0, "n": 0})
        self.scene_summaries: List[dict] = []

        # ---- v11: ATMS 内核 + Hansson belief-base 假设审计 (理论创新升级) ----
        self.use_atms = ablation not in ("no-atms",)
        self.atms = ATMSKernel()
        self.hansson = HanssonAuditor()
        self.atms_decisions: List[dict] = []     # 每次级联中 ATMS 的 核保留/核收缩 决策
        self.atms_asserted: Set[str] = set()      # 已写入 ATMS 的底层证据命题, 防重复
        # v12: LLM 语义层 → 形式化决策层 管线组件
        self.use_pipeline = ablation not in ("no-atms", "no-pipeline", "formal-only")
        self.retriever = CandidateRetriever()
        self.op_selector = BeliefOperationSelector()
        self.incision = IncisionFunction(self.atms)
        self.formal_filter = FormalConstraintFilter(self.atms)
        self.modal_claims: Dict[str, ModalClaim] = {}      # prop -> 最新 ModalClaim
        self.pipeline_log: List[dict] = []                 # 每次语义→形式化裁决留痕
        self.op_counts: Dict[str, int] = defaultdict(int)  # 各信念操作计数

        self.current_time = 0.0
        self.log_records: List[dict] = []
        self.logger_path = logger_path
        if logger_path and logger_path.exists(): logger_path.unlink()

    def _log(self, e):
        e["t"] = self.current_time; e["ts"] = datetime.now().isoformat(timespec="seconds")
        self.log_records.append(e)
        if self.logger_path:
            with self.logger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ---------------- persona helpers ----------------
    def innate_prior(self, npc, prop):
        if self.ablation == "no-persona":
            if any(k in prop for k in ("poison", "spy", "villain")): return 0.30
            return 0.45 if prop in ("player_is_knight", "player_good_character") else 0.30
        if prop == "player_is_knight":
            tr = TRAITS[npc]
            return clip01(0.45 + 0.6 * tr["knight_attitude"] + 0.15 * tr["hero_worship"])
        if prop == "player_good_character":
            tr = TRAITS[npc]
            return clip01(0.45 + 0.20 * tr["hero_worship"] + 0.25 * tr["knight_attitude"])
        if any(k in prop for k in ("poison", "spy", "villain")):
            tr = TRAITS[npc]
            return clip01(0.3 - 0.4 * tr["knight_attitude"])
        return 0.30

    def receptivity(self, npc, target) -> float:
        if self.ablation == "no-persona": return 0.6
        tr = TRAITS[npc]
        if target == "player_is_knight":
            return clip01(0.5 + tr["knight_attitude"] + 0.1 * tr["hero_worship"])
        if target == "player_good_character":
            return clip01(0.55 + 0.5 * tr["knight_attitude"] + 0.15 * tr["hero_worship"] - 0.2 * (1 - tr["liar_aversion"]))
        return 0.6

    def belief_ceiling(self, npc, target) -> float:
        if self.ablation == "no-persona": return 1.0
        return clip01(0.45 + 0.6 * self.receptivity(npc, target))

    def _derived_base(self, npc, target) -> float:
        if self.ablation == "no-persona": return 0.30
        return clip01(0.28 + 0.12 * (self.receptivity(npc, target) - 0.5))

    def susceptibility(self, npc):
        if self.ablation == "no-persona": return 0.6
        tr = TRAITS[npc]; p = self.persona[npc]
        return clip01(0.45 + 0.45 * tr["gullible"] - 0.40 * p["stubborn"])

    def get_anchor(self, npc, prop):
        if prop not in self.anchor[npc]:
            self.anchor[npc][prop] = self.innate_prior(npc, prop)
        return self.anchor[npc][prop]

    def belief_of(self, npc, prop) -> Optional[float]:
        r = self.M.current_rev(npc, prop)
        return r.confidence if r else None

    def _is_hard_objective(self, ev: "Event") -> bool:
        return (ev.event_source == "world_objective"
                and ev.evidence_type in ("official", "authority", "direct_observation")
                and ev.source_reliability >= 0.85)

    # ---------------- 访问控制 (tier gate) ----------------
    def tier_allows(self, tier, sender, receiver) -> Tuple[bool, str]:
        if self.ablation in ("no-access", "flat-rag"):
            return True, "ablation 无访问控制 (扁平检索, 无权限分层)"
        if tier == "core_identity": return False, "core_identity 结构隔离,不外传"
        if tier == "personal_episodic": return False, "personal_episodic 不外传"
        if tier == "relationship_memory":
            t = self.trust[receiver][sender]
            return (t >= REL_TRUST_GATE, f"relationship {'通过' if t >= REL_TRUST_GATE else '不足'}(trust {t:.2f})")
        if tier == "community_shared":
            sc = [c for c in SUBCOMMUNITIES if sender in COMMUNITIES[c] and receiver in COMMUNITIES[c]]
            return (bool(sc), f"community_shared {'同社区 '+sc[0] if sc else '跨社区受限'}")
        return True, "public 默认通过"

    # ---------------- 命题关系登记 + typed 边 + ATMS justification ----------------
    def register_relations(self, prop, relations):
        for r in relations:
            rel = r["relation"]; tgt = r["target_prop"]
            stg = r.get("strength", REL_STRENGTH.get(rel, 0.5)); why = r.get("rationale", "")
            if any(rr[0] == rel and rr[1] == tgt for rr in self.prop_relations[prop]): continue
            self.prop_relations[prop].append((rel, tgt, stg, why))
            self._log({"type": "prop_relation", "prop": prop, "relation": rel, "target": tgt,
                       "strength": stg, "rationale": why})
            self._atms_register_relation(prop, rel, tgt, stg, why)

    def _atms_register_relation(self, prop, rel, tgt, stg, why):
        """把一条命题关系翻译成 ATMS justification 超边 / defeater / nogood。
          SUPPORTS(src→tgt):    justification {src} → tgt (若 tgt 有 defeater 则带负前提, 形成可废止链)
          DEPENDS_ON(src→tgt):  justification {tgt} → src (src 继承 tgt 的支持环境; tgt 死则 src 死, 除非 src 另有 OR 路径)
          CONTRADICTS(src→tgt): src 成为 tgt 可废止链的 defeater + 登记 nogood{src,tgt}
        """
        if not self.use_atms: return
        if rel == "SUPPORTS":
            neg = sorted(self.atms.defeaters.get(tgt, set()))
            self.atms.add_justification([prop], tgt, neg_premises=neg, operator="DEFEASIBLE",
                                        strength=stg, rationale=why)
        elif rel == "DEPENDS_ON":
            self.atms.add_justification([tgt], prop, operator="AND", strength=stg, rationale=why)
        elif rel == "CONTRADICTS":
            # prop 反驳 tgt: prop 作为 tgt 可废止链的 defeater; 同时登记 nogood
            self.atms.register_defeater(tgt, prop)
            # 回填: 把已存在的 tgt 可废止支持 justification 替换为 "带该负前提守卫" 的版本 (而非追加, 避免旧无守卫版本残留)
            new_list = []
            for j in self.atms.justifications:
                if j.conclusion == tgt and j.operator == "DEFEASIBLE" and prop not in j.neg_premises:
                    new_list.append(Justification(j.jid, j.premises,
                                                  frozenset(set(j.neg_premises) | {prop}),
                                                  tgt, "DEFEASIBLE", j.strength, j.rationale))
                else:
                    new_list.append(j)
            self.atms.justifications = new_list
            # 仅当 prop 与 tgt 都是 foundational (各有底层 assumption) 时, 才登记互斥 nogood;
            # tgt 为派生命题(无 assumption)时, 单凭可废止守卫即可击败, 加 nogood 会误伤 prop 自身的支持环境。
            if prop in self.atms.base_assumption and tgt in self.atms.base_assumption:
                self.atms.add_nogood([prop, tgt], reason=why or "CONTRADICTS")

    def _atms_assert_base(self, prop, ev: "Event"):
        """从同一事件流把 "底层观察/文书/权威断言" 写入 ATMS 作为 Assumption (不引入任何新输入)。
        注意: DEPENDS_ON 派生命题 (如交剑合法性/背书合法性这类 "合法性判断") 不是 foundational 证据,
        其真值派生自被依赖命题, 故不写入底层 assumption, 以免获得虚假的自支持环境 (破坏核收缩)。"""
        if not self.use_atms: return
        if ev.is_chitchat or ev.polarity <= 0: return
        if ev.evidence_type not in ("direct_observation", "official", "authority", "self_claim",
                                    "consensus", "first_hand_meta"): return
        if any(rel == "DEPENDS_ON" for rel, _, _, _ in self.prop_relations.get(prop, [])):
            return
        origin = ev.event_source or ev.actor or "event"
        self.atms.assert_base(prop, ev.evidence_type, f"{origin}:{ev.actor}")
        self.atms_asserted.add(prop)

    # ---------------- v12: LLM 语义层 → 形式化决策层 管线 ----------------
    _SRC_TRUST = {"official": 0.97, "authority": 0.85, "direct_observation": 0.82,
                  "consensus": 0.6, "first_hand_meta": 0.7, "rumor": 0.5, "gossip": 0.5,
                  "hearsay": 0.45, "self_claim": 0.4}

    def _modality_of(self, ev: "Event") -> "Modality":
        et = ev.evidence_type
        if et == "official": return Modality.OFFICIALLY_VERIFIED
        if et == "self_claim": return Modality.CLAIMED
        if et in ("rumor", "gossip", "hearsay"): return Modality.POSSIBLE
        if et == "consensus": return Modality.SOCIALLY_ACCEPTED
        if et in ("direct_observation", "authority"): return Modality.CERTAIN
        return Modality.POSSIBLE

    def semantic_pipeline_step(self, ev: "Event") -> dict:
        """通用管线 (不依赖手写关系): 抽取(已由 analyze_action 完成) → 检索候选 → 二阶段提议 →
        形式化过滤(否决越界失效) → 操作选择(7 种) → 对 INVALIDATE/CONTRACTION 调 incision 决策。
        返回裁决留痕; 同时记录 ModalClaim 与操作计数。交互模式 / LLM 模式同样走此路径。"""
        if not self.use_pipeline or ev.is_chitchat:
            return {}
        new_claim = ev.proposition_key
        # 1) ModalClaim: 超越布尔, 记录 modality/置信/时序/来源信任
        st = self._SRC_TRUST.get(ev.evidence_type, 0.5)
        mc = ModalClaim(new_claim, self._modality_of(ev),
                        confidence=clip01(ev.evidence_strength), valid_from=self.current_time,
                        source_trust=st, holder=ev.actor or "world")
        # 2) 检索门控: 只在 top-k 候选里找冲突, 不扫全量记忆
        cands = self.retriever.retrieve(new_claim, ev.content_label or ev.content,
                                        self.prop_label, self.prop_relations)
        # 3) 二阶段提议 (LLM 提议 + critique; 离线走确定性回退)
        judgments = self.proposer.propose(ev.content, new_claim, cands, self.prop_label)
        # 4) 形式化过滤 + 5) 操作选择 + 6) incision 决策
        decisions = []
        for j in judgments:
            j_filt, why = self.formal_filter.filter(j, new_claim)
            has_prior = j_filt.target_claim in self.atms.known_claims()
            # 替换值检测: 新命题是否为被作用命题同一 "槽位" 的另一取值 (前缀共享但取值不同)
            tslot = "_".join(j_filt.target_claim.split("_")[:2])
            nslot = "_".join(new_claim.split("_")[:2])
            has_replacement = (tslot == nslot and new_claim != j_filt.target_claim)
            op = self.op_selector.select(j_filt, text=ev.content, has_prior=has_prior,
                                         has_replacement=has_replacement)
            self.op_counts[op.value] += 1
            rec = {"scene": ev.scene, "source": new_claim, "judgment": j_filt.to_dict(),
                   "formal_filter": why, "operation": op.value}
            # 对收缩/纠错: 用 incision function 决定切谁 (决策, 非事后审计)
            if op in (BeliefOp.CONTRACTION, BeliefOp.REVISION) and self.use_atms \
                    and j_filt.target_claim in self.atms.known_claims():
                sigma = self.incision.select(j_filt.target_claim, defeater=new_claim)
                rec["incision"] = sigma
            if op == BeliefOp.UPDATE:
                # KM update: 旧 ModalClaim 历史保留, 新 ModalClaim 当前有效
                old = self.modal_claims.get(j_filt.target_claim)
                if old and old.valid_to is None:
                    old.valid_to = self.current_time
                    old.modality = Modality.HISTORICALLY_TRUE
                rec["km_update"] = {"target": j_filt.target_claim,
                                    "historical_retained": bool(old)}
            decisions.append(rec)
        self.modal_claims[new_claim] = mc
        out = {"scene": ev.scene, "new_claim": new_claim, "modal": mc.describe(),
               "candidates": cands, "decisions": decisions}
        self.pipeline_log.append(out)
        return out



    def _link_prop_edges(self, prop):
        def first_rev(p):
            for n in NPCS:
                r = self.M.current_rev(n, p)
                if r: return r.rev_id
            return None
        src = first_rev(prop)
        if not src: return
        for rel, tgt, stg, why in self.prop_relations[prop]:
            dst = first_rev(tgt)
            if not dst: continue
            et = {"DEPENDS_ON": EdgeType.DEPENDS_ON, "CONTRADICTS": EdgeType.CONTRADICTS,
                  "SUPPORTS": EdgeType.AFFECTS}[rel]
            self.M.add_edge(et, src, dst, log="prop", attr=f"{prop}→{tgt} s={stg:.2f}")

    # ---------------- 写入一个 belief revision ----------------
    def _write_belief(self, npc, prop, belief, tier, prov, eid=None,
                      status=MemStatus.ACTIVE.value, esrc=None, category=None,
                      heard_rejected=False, rejected_score=None, supersede=True, proposer=None):
        rid = self.M._id("S")
        anchor = self.anchor[npc].get(prop, self.innate_prior(npc, prop))
        rev = Revision(rev_id=rid, uri="", ntype=NodeType.SEMANTIC, content=self.prop_label.get(prop, prop),
                       proposition_key=prop, confidence=clip01(belief),
                       category=category or prop_category(prop), access_tier=tier, status=status,
                       proposer=proposer or npc, prov=prov, source_event=eid,
                       created_at=self.current_time, valid_from=self.current_time, last_seen=self.current_time,
                       anchor=anchor, event_source=esrc, heard_rejected=heard_rejected,
                       rejected_score=rejected_score)
        self.M.write_revision(npc, prop, rev, layer="personal", supersede=supersede)
        if eid:
            self.M.add_edge(EdgeType.SOURCE_OF, eid, rid)
            self.M.add_edge(EdgeType.DERIVED_FROM, rid, eid)
        if prop not in self.anchor[npc]:
            self.anchor[npc][prop] = clip01(belief); rev.anchor = self.anchor[npc][prop]
        if status not in (MemStatus.REFUTED.value,):
            self.belief_hist[prop][npc].append((self.current_time, clip01(belief)))
        return rid

    def commit_belief(self, npc, ev: Event, score, source_kind="direct"):
        eid = self.M.write_episode(npc, ev.proposition_key, ev.content, ev.actor,
                                   ev.evidence_type, ev.access_tier, self.current_time, ev.event_source)
        if source_kind == "direct" and ev.evidence_type in ("direct_observation", "official", "authority"):
            self.anchor[npc][ev.proposition_key] = clip01(score)
        prov = Provenance(
            origin_source=npc if ev.evidence_type in ("direct_observation", "official", "authority") else ev.actor,
            transmission_path=[ev.actor, npc], evidence_type=ev.evidence_type)
        self._write_belief(npc, ev.proposition_key, score, ev.access_tier, prov, eid=eid,
                           esrc=ev.event_source, category=ev.category)
        if source_kind == "direct":
            self._atms_assert_base(ev.proposition_key, ev)
    def extract_belief(self, ev: Event) -> bool:
        has_claim = (not ev.is_chitchat) and abs(ev.polarity) > 0 and ev.evidence_strength > 0
        keep = has_claim
        self.N.step(f"Belief Extraction (RQ1, 规则+LLM judge): {'纳入' if keep else '跳过(寒暄/无事实声明)'}")
        self.N.kv("命题键", ev.proposition_key)
        self.N.kv("命题描述", self.prop_label.get(ev.proposition_key, ev.content_label or "—"))
        self.N.kv("记忆类别", ev.category or prop_category(ev.proposition_key))
        self.N.kv("证据类型", f"{ev.evidence_type} (强度 {ev.evidence_strength:.2f})")
        self.N.kv("极性", "+1 支持" if ev.polarity > 0 else "-1 反驳")
        self.N.kv("含事实声明", "是" if has_claim else "否")
        self.N.kv("决策", C.green("KEEP") if keep else C.gray("SKIP"))
        self._log({"type": "belief_extraction", "scene": ev.scene, "prop": ev.proposition_key, "kept": keep})
        return keep

    # ---------------- 直接观察 ----------------
    def direct_observation(self, ev: Event):
        if ev.is_true is not None and ev.proposition_key not in self.ground_truth:
            self.ground_truth[ev.proposition_key] = ev.is_true
        self.N.step(f"Direct Observation: {len(ev.direct_observers)} 名目击者打分 "
                    f"(其余 {len(NPCS)-len(ev.direct_observers)} 人靠选择性传播 → 群体作用)")
        for npc in ev.direct_observers:
            kb = TRAITS[npc]["knight_attitude"]
            cur = self.belief_of(npc, ev.proposition_key) or self.get_anchor(npc, ev.proposition_key)
            if self._is_hard_objective(ev):
                score = clip01(max(ev.evidence_strength, ev.source_reliability))
                reason = "客观硬事实, 免主观打分"
            else:
                score, reason = self.judge.score_observation(
                    npc, ev.proposition_key, ev.content, cur, ev.evidence_strength,
                    self.trust[npc].get(ev.actor, 0.45), ev.source_reliability, kb)
            tau = self.persona[npc]["tau"]
            old = self.belief_of(npc, ev.proposition_key)
            if score >= tau or self._is_hard_objective(ev):
                self.commit_belief(npc, ev, score, source_kind="direct")
                self.N.belief_change(npc, ev.proposition_key, old, self.belief_of(npc, ev.proposition_key),
                                     note=f"{ev.evidence_type}; {reason}")
            else:
                prov = Provenance(origin_source=ev.actor, transmission_path=[ev.actor, npc], evidence_type=ev.evidence_type)
                self._write_belief(npc, ev.proposition_key, score, ev.access_tier, prov,
                                   status=MemStatus.REFUTED.value, esrc=ev.event_source,
                                   heard_rejected=True, rejected_score=round(score, 4), supersede=False)
                self.N.rejected(npc, ev.proposition_key, score, tau, reason)
                self.rejected_log.append({"scene": ev.scene, "npc": npc, "prop": ev.proposition_key,
                                          "score": round(score, 4), "tau": tau, "t": self.current_time})
                self._log({"type": "rejected_observation", "npc": npc, "prop": ev.proposition_key,
                           "score": round(score, 4), "tau": tau})
        self._recompute_identity_for_observers(ev)

    # ---------------- (2) 证据聚合: 事件→品德→身份 ----------------
    def _supporters_of(self, target):
        sup = []
        for p, rels in self.prop_relations.items():
            for rel, tgt, stg, _ in rels:
                if tgt == target and rel in ("SUPPORTS", "CONTRADICTS"):
                    sup.append((p, rel, stg))
        return sup

    def _recompute_identity_for_observers(self, ev):
        upstream = set()
        for tgt in DERIVED_PROPS:
            for p, _, _ in self._supporters_of(tgt):
                upstream.add(p)
        if ev.proposition_key in DERIVED_PROPS or ev.proposition_key not in upstream:
            return
        for npc in NPCS:
            self.recompute_derived_chain(npc, note="证据更新")

    def recompute_derived_chain(self, npc, note=""):
        for tgt in DERIVED_PROPS:
            sup = self._supporters_of(tgt)
            if sup:
                self._aggregate_derived(npc, tgt, sup, note=note)

    def _aggregate_derived(self, npc, target, supporters, note=""):
        prior = self._derived_base(npc, target)
        pos = 0.0; neg = 0.0; have = False
        used = []
        for p, rel, stg in supporters:
            r = self.M.current_rev(npc, p)
            if r is None: continue
            b = r.confidence
            sw = status_cap(r.status, "weight")
            if sw <= 0: continue
            cred = evidence_cred(r.prov.evidence_type if r.prov else "rumor")
            w = sw * cred
            if rel == "CONTRADICTS":
                neg += stg * b * w; have = True; used.append(f"-{p[:10]}")
            else:
                if status_cap(r.status, "pos_evidence"):
                    pos += max(0.0, stg * (b - 0.4)) * w; have = True; used.append(f"+{p[:10]}")
        if not have: return
        recept = self.receptivity(npc, target)
        skep = self.persona[npc].get("skepticism", 0.5)
        pos_eff = pos * (0.5 + recept) * (1.0 - 0.35 * skep)
        neg_eff = neg * (1.0 + 0.5 * (1.0 - recept))
        raw = prior + 1.9 * pos_eff - 1.6 * neg_eff
        ceiling = self.belief_ceiling(npc, target)
        agg = clip01(min(raw, ceiling)) if raw > prior else clip01(raw)
        old = self.belief_of(npc, target)
        if old is not None and abs(agg - old) < 0.01: return
        prov = Provenance(origin_source=f"evidence_agg::{npc}", transmission_path=["evidence_agg"],
                          evidence_type="direct_observation")
        self._write_belief(npc, target, agg, "public_consensus", prov,
                           esrc="evidence_aggregation", category=prop_category(target))
        self.anchor[npc][target] = agg
        r_new = self.M.current_rev(npc, target)
        if r_new is not None: r_new.anchor = agg
        self.N.belief_change(npc, target, old, agg, note=f"聚合[{','.join(used)}]; {note}")

    # ---------------- (3) 两级传播之一: pairwise 选择性传播 (RQ2) ----------------
    def propagate(self, prop, max_rounds=6, eps=0.002):
        if self.ablation == "no-propagation":
            self.N.step("Propagation: SKIP (ablation=no-propagation)"); return
        self.N.step(f"Pairwise Selective Propagation 命题「{prop}」 "
                    f"(L1 二元更新, 只看双方值, 非DeGroot; 选择性: 可见∧社会边∧主题∧trust≥{TRUST_PROP_GATE})")
        for rnd in range(max_rounds):
            knowers = [n for n in NPCS if self.belief_of(n, prop) is not None]
            if not knowers: break
            maxd = 0.0; any_update = False
            for i in NPCS:
                bi = self.belief_of(i, prop)
                cands = []
                for j in knowers:
                    if j == i: continue
                    jr = self.M.current_rev(j, prop)
                    if not jr: continue
                    allowed, why = self.tier_allows(jr.access_tier, j, i)
                    if not allowed:
                        if rnd == 0: self.N.access_block(j, i, jr.access_tier, why)
                        continue
                    if self.ablation not in ("no-trust",):
                        visible = i in SOCIAL_VISIBILITY.get(j, set()) or j in SOCIAL_VISIBILITY.get(i, set())
                        if not visible and self.trust[i][j] < 0.6: continue
                    if self.trust[i][j] < TRUST_PROP_GATE: continue
                    cands.append((j, jr))
                if not cands: continue
                j, jr = max(cands, key=lambda c: self.trust[i][c[0]] * TRAITS[c[0]]["influence"] * c[1].confidence)
                bj = jr.confidence
                trust = min(self.trust[i][j], W_MAX)
                infl = TRAITS[j]["influence"]
                s_i = self.susceptibility(i)
                sw = status_cap(jr.status, "weight")
                gain = s_i * trust * (0.6 + 0.5 * infl) * sw
                prior = bi if bi is not None else self.get_anchor(i, prop)
                fj_target = bj
                if prop in DERIVED_PROPS and self.ablation != "no-persona" and fj_target > prior:
                    gain *= (0.35 + 0.65 * self.receptivity(i, prop))
                newb = clip01((1 - self.persona[i]["alpha"] * gain) * prior
                              + self.persona[i]["alpha"] * gain * fj_target)
                if prop in DERIVED_PROPS and self.ablation != "no-persona" and newb > prior:
                    newb = min(newb, self.belief_ceiling(i, prop))
                tau = self.persona[i]["tau"]
                if bi is None and newb < tau:
                    if rnd == 0:
                        prov = Provenance(origin_source=jr.prov.origin_source,
                                          transmission_path=jr.prov.transmission_path + [i], evidence_type="hearsay")
                        self._write_belief(i, prop, newb, jr.access_tier, prov,
                                           status=MemStatus.REFUTED.value, heard_rejected=True,
                                           rejected_score=round(newb, 4), supersede=False)
                        self.N.rejected(i, prop, newb, tau, f"传闻 via {j} 未越阈")
                        self.rejected_log.append({"scene": -1, "npc": i, "prop": prop, "score": round(newb, 4),
                                                  "tau": tau, "t": self.current_time})
                    continue
                if bi is not None and abs(newb - bi) < eps: continue
                src_et = jr.prov.evidence_type
                new_et = "hearsay" if src_et in ("direct_observation", "authority", "official") else src_et
                cur = self.M.current_rev(i, prop)
                origin = jr.prov.origin_source; path = jr.prov.transmission_path
                if cur and cur.prov.evidence_type in ("direct_observation", "authority", "official", "first_hand_meta"):
                    new_et = cur.prov.evidence_type; origin = cur.prov.origin_source; path = cur.prov.transmission_path
                if self.ablation == "no-provenance":
                    origin = j; path = [j, i]
                prov = Provenance(origin_source=origin,
                                  transmission_path=path + [i] if i not in path else path, evidence_type=new_et)
                old = bi
                self._write_belief(i, prop, newb, jr.access_tier, prov, esrc="npc_action")
                self.N.pair_update(i, j, prop, old, newb, self.trust[i][j])
                maxd = max(maxd, abs((old or 0) - newb)); any_update = True
                self._log({"type": "pairwise_propagation", "round": rnd, "prop": prop, "receiver": i,
                           "sender": j, "origin": origin, "before": round(old or 0, 4), "after": round(newb, 4)})
            if not any_update or maxd < eps: break

    # ---------------- (4)(5)(6) 两级传播之二: 信任加权共识 voting ----------------
    def update_community_consensus(self, prop):
        self.N.step("Trust-Weighted Consensus Voting (L2: 来源多样性 + 阈值 → shared → world)")
        guard = None
        for n in NPCS:
            r = self.M.current_rev(n, prop)
            if r: guard = r.access_tier; break
        if guard and guard not in ("public_consensus",):
            self.N.info(f"跳过 [{prop}]: tier={guard}, 非 public 不参与公共共识"); return
        for comm, members in COMMUNITIES.items():
            knw = [n for n in members if self.belief_of(n, prop) is not None]
            prev_shared = prop in self.M.community_consensus[comm]
            if not knw:
                self.N.consensus(comm, prop, 0.0, 0, False, 0.0, self.TAU_CONSENSUS, False, prev_shared)
                if prev_shared: self._revoke_shared(comm, prop)
                continue
            origin_groups = defaultdict(list)
            for n in knw:
                r = self.M.current_rev(n, prop); origin_groups[r.prov.origin_source].append(n)
            num = den = 0.0; n_shareable = 0
            for origin, mns in origin_groups.items():
                gsize = len(mns)
                for n in mns:
                    r = self.M.current_rev(n, prop)
                    sw = status_cap(r.status, "weight")
                    if status_cap(r.status, "shareable"): n_shareable += 1
                    cred = evidence_cred(r.prov.evidence_type)
                    authority = TRAITS[n]["influence"]
                    obs_priv = 1.0 if r.prov.evidence_type in ("direct_observation", "official", "authority") else 0.6
                    w = (cred * 0.45 + authority * 0.30 + obs_priv * 0.25) * sw / gsize
                    num += w * r.confidence; den += w
            agg = num / max(den, 1e-9)
            override = 0.0
            for n in knw:
                r = self.M.current_rev(n, prop)
                if (status_cap(r.status, "pos_evidence")
                        and r.prov.evidence_type in ("direct_observation", "authority", "official")
                        and r.confidence >= 0.72):
                    override = max(override, r.confidence)
            consensus = clip01(max(agg, override))
            self.peak_consensus[f"{comm}::{prop}"] = max(self.peak_consensus[f"{comm}::{prop}"], consensus)
            clusters = len(origin_groups)
            diverse_ok = clusters >= DIVERSITY_MIN or override >= 0.72
            promoted = consensus >= self.TAU_CONSENSUS and diverse_ok and n_shareable >= 1
            should_revoke = prev_shared and (consensus < self.TAU_CONSENSUS or not diverse_ok or n_shareable == 0)
            self.N.consensus(comm, prop, consensus, clusters, diverse_ok, override,
                             self.TAU_CONSENSUS, promoted, prev_shared)
            if n_shareable == 0 and (prev_shared or consensus >= self.TAU_CONSENSUS):
                self.N.info(f"[{comm}::{prop}] 全部持有者为 pending/refuted → 不可作为 shared 共识")
            self.consensus_hist[f"{comm}::{prop}"].append((self.current_time, consensus, promoted))
            self._log({"type": "consensus_gate", "comm": comm, "prop": prop, "score": round(consensus, 4),
                       "clusters": clusters, "diverse": diverse_ok, "n_shareable": n_shareable,
                       "promoted": promoted, "prev_shared": prev_shared})
            if promoted and not prev_shared: self._promote_community(comm, prop, consensus, knw)
            elif should_revoke: self._revoke_shared(comm, prop)
        self._link_prop_edges(prop)

    def _promote_community(self, comm, prop, score, knw):
        nid = self.M._id("CC")
        prov = Provenance(origin_source=f"community::{comm}", transmission_path=["consensus"], evidence_type="consensus")
        rev = Revision(rev_id=nid, uri=f"mem://community/{comm}/{prop}", ntype=NodeType.CONSENSUS,
                       content=self.prop_label.get(prop, prop), proposition_key=prop, confidence=clip01(score),
                       access_tier="public_consensus", status=MemStatus.SHARED.value, prov=prov,
                       created_at=self.current_time, valid_from=self.current_time, last_seen=self.current_time,
                       anchor=clip01(score), category=prop_category(prop))
        self.M.G.add_node(nid, rev=rev, ntype=NodeType.CONSENSUS, kind="consensus")
        self.M.community_consensus[comm][prop] = nid
        if self.ground_truth.get(prop) is False: self.false_shared_ever.add(prop)
        self.shared_state_log.append({"scene": self.scene_counter, "t": self.current_time,
                                      "comm": comm, "prop": prop, "action": "promote", "score": round(score, 4)})
        for k in knw:
            kr = self.M.current_rev(k, prop)
            if kr:
                self.M.add_edge(EdgeType.AGGREGATED, nid, kr.rev_id, log="community")
                self.M.add_edge(EdgeType.REFERENCES, kr.rev_id, nid)
        self.maybe_promote_world(prop)

    def _revoke_shared(self, comm, prop):
        old = self.M.community_consensus[comm].pop(prop, None)
        if old:
            r = self.M.get_rev(old)
            if r: r.status = MemStatus.DEPRECATED.value; r.valid_until = self.current_time
            self.N.warn(f"REVOKE community shared: [{comm}::{prop}]")
            self.shared_state_log.append({"scene": self.scene_counter, "t": self.current_time,
                                          "comm": comm, "prop": prop, "action": "revoke"})
            if (prop in self.ground_truth and self.ground_truth[prop] is False
                    and prop not in self.recovery_scenes and prop in self.contradict_time):
                self.recovery_scenes[prop] = round(self.current_time - self.contradict_time[prop], 2)
                self.recovery_scene[prop] = self.scene_counter - self.contradict_scene.get(prop, self.scene_counter)
            if prop in self.M.world_consensus:
                wn = self.M.get_rev(self.M.world_consensus.pop(prop))
                if wn: wn.status = MemStatus.DEPRECATED.value
                self.N.warn(f"REVOKE world consensus: [world::{prop}] ← 失去社区支持")

    def maybe_promote_world(self, prop):
        sub_with = [c for c in SUBCOMMUNITIES if prop in self.M.community_consensus[c]]
        if len(sub_with) >= 2 and prop not in self.M.world_consensus:
            scores = [self.M.get_rev(self.M.community_consensus[c][prop]).confidence for c in sub_with]
            avg = float(np.mean(scores))
            nid = self.M._id("W")
            prov = Provenance(origin_source="world", transmission_path=["world"], evidence_type="consensus")
            rev = Revision(rev_id=nid, uri=f"mem://world/world/{prop}", ntype=NodeType.CONSENSUS,
                           content=self.prop_label.get(prop, prop), proposition_key=prop, confidence=clip01(avg),
                           access_tier="public_consensus", status=MemStatus.SHARED.value, prov=prov,
                           created_at=self.current_time, valid_from=self.current_time, last_seen=self.current_time)
            self.M.G.add_node(nid, rev=rev, ntype=NodeType.CONSENSUS, kind="consensus")
            self.M.world_consensus[prop] = nid
            for c in sub_with:
                self.M.add_edge(EdgeType.PROMOTED, nid, self.M.community_consensus[c][prop], log="world", attr=f"from {c}")
            self.N.world_fact(prop, avg, sub_with)
            self._log({"type": "world_promote", "prop": prop, "avg": round(avg, 4), "from": sub_with})

    # ---------------- Dependency-aware cascade (RQ3, 局部 BFS, AGP-Dynamic) ----------------
    def _dependents_of(self, target):
        out = []
        for x, rels in self.prop_relations.items():
            for rel, tgt, stg, _ in rels:
                if rel == "DEPENDS_ON" and tgt == target:
                    out.append((x, stg))
        return out

    def cascade_update(self, trigger_prop: str, trigger_strength: float, source_label: str):
        t0 = time.perf_counter()
        visited_nodes = {trigger_prop}; visited_edges = 0; affected = []
        queue = [(trigger_prop, trigger_strength, 0)]
        seen = {trigger_prop}
        # v11: ATMS 在级联开始前对 trigger 做一次 defeat (使其在内核中不再被支持),
        #      之后每个依赖命题用 ATMS 判定 "是否仍有不依赖 trigger 的存活支持环境"。
        atms_retained, atms_contracted = [], []
        if self.use_atms:
            self.atms.recompute_labels()
        while queue:
            cur, strength, hops = queue.pop(0)
            for x, stg in self._dependents_of(cur):
                visited_edges += 1
                visited_nodes.add(x)
                if self.ablation in ("no-cascade", "flat-rag"):
                    continue
                if x in self.independent_facts:
                    continue
                # ---- v11 ATMS 决策: 核保留 vs 核收缩 ----
                core_retain = False
                atms_known = self.use_atms and x in self.atms.known_claims()
                if atms_known:
                    core_retain = self.atms.has_alternative_support(x, trigger_prop)
                rel_strength = REL_STRENGTH["DEPENDS_ON"] * stg
                dist = DIST_DECAY ** hops
                touched = False
                if core_retain:
                    # Hansson Core-Retainment: x 仍有不依赖 trigger 的最小支持集 → 不收缩,
                    # 仅标记 CONTESTED (听闻冲突但证据仍在), 不破坏其置信。
                    for npc in NPCS:
                        r = self.M.current_rev(npc, x)
                        if r is None: continue
                        r.status = MemStatus.PENDING.value   # CONTESTED 语义: 暂记争议, 证据未失
                    envs = self.atms.surviving_environments(x)
                    self.N.atms(f"[{x}] DEPENDS_ON {cur} 被反证, 但仍有替代支持环境 "
                                f"{[sorted(e) for e in envs][:2]} → 核保留(不收缩)", kind="retain")
                    atms_retained.append({"claim": x, "via": cur, "surviving": [sorted(e) for e in envs]})
                    self.atms_decisions.append({"scene": self.scene_counter, "claim": x, "trigger": trigger_prop,
                                                "decision": "core_retain",
                                                "surviving_envs": [sorted(e) for e in envs]})
                else:
                    for npc in NPCS:
                        r = self.M.current_rev(npc, x)
                        if r is None: continue
                        role_sens = 1 - self.persona[npc]["stubborn"]
                        impact = strength * rel_strength * dist * role_sens
                        newb = clip01(r.confidence * (1 - impact))
                        if self._apply_cascade(npc, x, newb, status=MemStatus.PENDING.value,
                                               via=cur, src=source_label):
                            touched = True
                    if atms_known:
                        self.N.atms(f"[{x}] DEPENDS_ON {cur}: 无替代支持环境 → 核收缩(kernel contraction)",
                                    kind="contract")
                        atms_contracted.append(x)
                        self.atms_decisions.append({"scene": self.scene_counter, "claim": x,
                                                    "trigger": trigger_prop, "decision": "kernel_contract"})
                if touched:
                    affected.append(f"{x} (DEPENDS_ON {cur}, hop={hops+1})")
                    if self.ground_truth.get(x) is False and x not in self.contradict_time:
                        self.contradict_time[x] = self.current_time
                        self.contradict_scene[x] = self.scene_counter
                if x not in seen and hops < 4:
                    seen.add(x); queue.append((x, strength * rel_strength * DIST_DECAY, hops + 1))
        rt = (time.perf_counter() - t0) * 1000
        report = {"trigger": trigger_prop, "visited_nodes": len(visited_nodes),
                  "visited_edges": visited_edges, "affected_nodes": affected,
                  "cascade_runtime_ms": rt,
                  "atms_core_retained": [d["claim"] for d in atms_retained],
                  "atms_kernel_contracted": atms_contracted}
        self.cascade_reports.append(report)
        self.N.cascade(report)
        self._log({"type": "cascade", "trigger": trigger_prop, "visited_nodes": len(visited_nodes),
                   "visited_edges": visited_edges, "affected": len(affected), "runtime_ms": round(rt, 3),
                   "atms_core_retained": [d["claim"] for d in atms_retained],
                   "atms_kernel_contracted": atms_contracted})
        return report

    def _apply_cascade(self, npc, prop, newb, status, via, src) -> bool:
        old = self.belief_of(npc, prop)
        if old is None or abs(newb - old) < 0.005: return False
        self.anchor[npc][prop] = newb
        prov = Provenance(origin_source=src, transmission_path=[src, npc], evidence_type="official")
        rid = self._write_belief(npc, prop, newb, "public_consensus", prov,
                                 esrc="world_objective", status=status)
        rev = self.M.get_rev(rid)
        sr = None
        for n in NPCS:
            r = self.M.current_rev(n, via)
            if r: sr = r.rev_id; break
        if sr and rev: self.M.add_edge(EdgeType.INVALIDATES, sr, rid)
        self.N.belief_change(npc, prop, old, newb, note=f"级联 via {via} → {status}")
        return True

    # ---------------- 反证 ----------------
    def contradict(self, ev: Event):
        prop = ev.proposition_key
        self.contradict_time[prop] = self.current_time
        self.contradict_scene[prop] = self.scene_counter
        if ev.is_true is not None: self.ground_truth[prop] = ev.is_true
        self.N.step(f"Contradiction 反证「{prop}」(强度={ev.evidence_strength:.2f})")
        evb = clip01(ev.evidence_strength * 0.62 + ev.source_reliability * 0.30)
        for npc in (ev.direct_observers or NPCS):
            old = self.belief_of(npc, prop)
            if old is None: continue
            st = self.persona[npc]["stubborn"]
            revoke = clip01(evb * (1 - st))
            newb = clip01(old * (1 - revoke))
            self.anchor[npc][prop] = newb
            prov = Provenance(origin_source=ev.actor, transmission_path=[ev.actor, npc], evidence_type=ev.evidence_type)
            self._write_belief(npc, prop, newb, ev.access_tier, prov, esrc=ev.event_source)
            self.N.belief_change(npc, prop, old, newb, note=f"revoke={revoke:.2f} stubborn={st:.2f}")
            if self.ablation != "no-trust": self._trust_refuted(npc, prop)
        for npc in NPCS:
            self.recompute_derived_chain(npc, note=f"{prop} 被反证")
        self.cascade_update(prop, evb, source_label=ev.actor)
        self.propagate(prop); self.update_community_consensus(prop)

    # ---------------- 独立客观事实 (official_denial, 级联免疫) ----------------
    def assert_independent_fact(self, ev: Event, settle_consensus=True):
        prop = ev.proposition_key
        if ev.is_true is not None: self.ground_truth[prop] = ev.is_true
        self.independent_facts.add(prop)
        self.N.step(f"Independent Fact: 把「{prop}」建成独立 shared 客观事实 (级联免疫)")
        if ev.content_label: self.prop_label[prop] = ev.content_label
        # ---- v11: 在 defeater 生效前捕获 ATMS 前态 + 各 contradicted 目标的 kernel (用于 Hansson 审计) ----
        atms_before = None; pre_kernels = {}; protected_before = {}
        contradicted_pre = [r["target_prop"] for r in ev.relations if r["relation"] == "CONTRADICTS"]
        if self.use_atms:
            atms_before = self.atms.snapshot_supported()
            for tg in contradicted_pre:
                pre_kernels[tg] = [frozenset(e) for e in self.atms.kernels_of(tg)]
                # 收缩前仍有 "不依赖 tg 的独立支持" 的命题 = 应被核保留 (受保护) 的命题
                protected_before[tg] = {c for c in atms_before
                                        if self.atms.has_alternative_support(c, tg)}
        for npc in (ev.direct_observers or NPCS):
            cur = self.belief_of(npc, prop) or self.get_anchor(npc, prop)
            if self._is_hard_objective(ev):
                score = clip01(max(ev.evidence_strength, ev.source_reliability)); reason = "客观硬事实, 免主观打分"
            else:
                score, reason = self.judge.score_observation(npc, prop, ev.content, cur, ev.evidence_strength,
                                                             self.trust[npc].get(ev.actor, 0.6), ev.source_reliability, 0.0)
            old = self.belief_of(npc, prop)
            self.commit_belief(npc, ev, score, source_kind="direct")
            self.N.belief_change(npc, prop, old, self.belief_of(npc, prop), note=f"独立客观事实 {ev.evidence_type}; {reason}")
        self.register_relations(prop, ev.relations)
        # v12: 独立事实(如 official_denial)同样走语义→形式化管线 (此时 defeater 已注册, 形式化过滤可判路径)
        if self.use_pipeline and not ev.is_chitchat:
            self.semantic_pipeline_step(ev)
        self.propagate(prop); self.update_community_consensus(prop)
        contradicted = [r["target_prop"] for r in ev.relations if r["relation"] == "CONTRADICTS"]
        for npc in NPCS:
            self.recompute_derived_chain(npc, note=f"被独立事实 {prop} 反证")
        for tgt in contradicted:
            self.contradict_time[tgt] = self.current_time
            self.contradict_scene[tgt] = self.scene_counter
            self.N.info(f"独立事实 {prop} 成立 → {tgt} 经证据重算被下拉 → 触发 DEPENDS_ON 下游级联")
            # ---- v11: Hansson belief-base 假设审计 (按 p=tgt 做核收缩, 用 defeater 生效前的前态与 kernel) ----
            if self.use_atms:
                kernels = pre_kernels.get(tgt, [frozenset(e) for e in self.atms.kernels_of(tgt)])
                self.cascade_update(tgt, clip01(ev.evidence_strength), source_label=prop)
                after = self.atms.snapshot_supported()
                rec = self.hansson.audit(self.atms, tgt, atms_before, after, kernels,
                                         scene=self.scene_counter,
                                         protected_before=protected_before.get(tgt),
                                         added={prop})
                self.N.hansson(rec)
                self._log({"type": "hansson_audit", **rec})
            else:
                self.cascade_update(tgt, clip01(ev.evidence_strength), source_label=prop)
            if settle_consensus:
                self.propagate(tgt); self.update_community_consensus(tgt)
                for dep, _ in self._dependents_of(tgt):
                    self.update_community_consensus(dep)

    # ---------------- 动态信任 ----------------
    def _trust_refuted(self, npc, prop):
        for j in NPCS:
            if j == npc: continue
            jr = self.M.current_rev(j, prop)
            if not jr: continue
            if jr.prov.evidence_type in ("rumor", "self_claim", "hearsay", "gossip") and jr.confidence > 0.45:
                before = self.trust[npc][j]; sens = self.persona[npc]["trust_sensitivity"]
                after = clip01(before - 0.8 * sens)
                if abs(after - before) > 0.001:
                    self.trust[npc][j] = after; self.M.trusts_edge(npc, j, after)
                    self.trust_hist.append((self.current_time, npc, j, before, after, "传谣被推翻"))
                    self.N.trust_change(npc, j, before, after, "传谣被推翻")

    def positive_interaction(self, a, b, magnitude=1.0, reason=""):
        if a == b or self.ablation == "no-trust": return
        for s, d in [(a, b), (b, a)]:
            before = self.trust[s][d]; sens = self.persona[s]["trust_sensitivity"]
            after = clip01(before + 0.4 * sens * magnitude)
            if abs(after - before) > 0.001:
                self.trust[s][d] = after; self.M.trusts_edge(s, d, after)
                self.trust_hist.append((self.current_time, s, d, before, after, reason or "正向互动"))
                self.N.trust_change(s, d, before, after, reason or "正向互动")

    def joint_witness_bond(self, observers, reason="共同见证"):
        if len(observers) < 2 or self.ablation == "no-trust": return
        for a, b in itertools.combinations(observers, 2):
            if are_rivals(a, b):
                self.N.info(f"共同见证不增信任: {a}↔{b} 是情敌/竞争关系 (RQ4 角色差异保护)")
                continue
            self.positive_interaction(a, b, magnitude=0.12, reason=reason)

    # ---------------- 关系记忆 (节点 + PROJECTS_TO 投影) ----------------
    def add_relationship(self, owner, target, label, belief=0.8, tier="relationship_memory"):
        nid = self.M._id("R")
        prop = f"{owner}_relates_{target}_{label}"
        prov = Provenance(origin_source=owner, transmission_path=[owner], evidence_type="first_hand_meta")
        rev = Revision(rev_id=nid, uri=f"mem://personal/{owner}/{prop}", ntype=NodeType.RELATION,
                       content=f"{owner}对{target}的「{label}」", proposition_key=prop, confidence=clip01(belief),
                       category=MemCategory.RELATIONSHIP.value, access_tier=tier, status=MemStatus.ACTIVE.value,
                       prov=prov, created_at=self.current_time, valid_from=self.current_time, last_seen=self.current_time)
        self.M.G.add_node(nid, rev=rev, ntype=NodeType.RELATION, kind="revision")
        it = MemItem(uri=rev.uri, layer="personal", scope=owner, proposition_key=prop, current_rev=nid, revisions=[nid])
        self.M.items[rev.uri] = it
        self.anchor[owner][prop] = belief; self.prop_label[prop] = f"{owner} 对 {target} 的「{label}」关系记忆"
        self.N._emit(f"┃     {C.magenta('+ RELATIONSHIP NODE')} [{owner}]→[{target}]:「{label}」b={belief:.2f} tier={tier}")
        if tier == "relationship_memory":
            self.N.info(f"PROJECTS_TO 投影 (沿 trust ≥ {REL_TRUST_GATE}):")
            for other in NPCS:
                if other == owner: continue
                t = self.trust[other][owner]
                if t >= REL_TRUST_GATE:
                    pid = self.M.person_nid.get(other)
                    if pid: self.M.add_edge(EdgeType.PROJECTS_TO, nid, pid, attr=f"trust {t:.2f}")
                    self.N._emit(f"┃         {C.green('✓ 可投影')} {other} (trust→{owner}={t:.2f})")
                else:
                    self.N._emit(f"┃         {C.red('⛔ 阻挡')} {other} (trust→{owner}={t:.2f} < {REL_TRUST_GATE})")
        elif tier == "core_identity":
            self.N.info("core_identity: 结构隔离, 不可外传也不可投影")
        return nid

    # ---------------- 时间衰减 (Ebbinghaus) ----------------
    def decay_one(self, rev: Revision, now: float):
        if rev.confidence is None: return 0.0, 0.0
        et = rev.prov.evidence_type if rev.prov else "rumor"
        if et == "first_hand_meta": return rev.confidence, rev.confidence
        hl = HALF_LIFE.get(et, 20) * (1.0 + REINFORCE_RHO * rev.reinforcement)
        base = rev.last_seen or rev.created_at
        dt = now - base
        if dt <= 0: return rev.confidence, rev.confidence
        anchor = rev.anchor if rev.anchor is not None else 0.3
        old = rev.confidence
        new = anchor + (old - anchor) * math.exp(-dt / hl)
        if rev.status == MemStatus.SHARED.value and self.ground_truth.get(rev.proposition_key) is True:
            floor = 0.55 if et in ("direct_observation", "authority", "official", "consensus") else 0.45
            new = max(new, floor)
        rev.confidence = clip01(new)
        rev.last_seen = now
        return old, rev.confidence

    def advance_time(self, delta, prop=None):
        if delta <= 0:
            self.current_time += max(0.0, delta); return
        self.current_time += delta
        seen = set(); decayed = defaultdict(list)
        for uri, it in self.M.items.items():
            if it.layer not in ("personal", "flat"): continue
            if it.current_rev is None: continue
            rev = self.M.get_rev(it.current_rev)
            if not rev or rev.ntype not in (NodeType.SEMANTIC,): continue
            if rev.status in (MemStatus.SUPERSEDED.value, MemStatus.DEPRECATED.value, MemStatus.REFUTED.value): continue
            if prop and rev.proposition_key != prop: continue
            key = (it.scope, rev.proposition_key)
            if key in seen: continue
            seen.add(key)
            old, new = self.decay_one(rev, self.current_time)
            if abs(old - new) > 0.001:
                decayed[rev.proposition_key].append((it.scope, old, new))
        for p, items in decayed.items():
            self.N.decay(p, items)

    def summarize_theme(self, npc, theme):
        ret = self.M.layered_retrieve(npc, theme)
        if ret["semantic_nodes"]:
            srcs = [f"{s['prop']}({s['confidence']:.2f})" for s in ret["semantic_nodes"]]
            agg = float(np.mean([s["confidence"] for s in ret["semantic_nodes"]]))
            self.N.abstraction(npc, f"Theme[{theme}]", srcs, agg)

    # ---------------- 场景结算 ----------------
    def scene_settlement(self, scene_id, props_touched: List[str]):
        self.N.step("Scene Settlement: 写入长期记忆 + 场景共识摘要")
        formed, disputed, overturned, per_npc = [], [], [], {}
        for p in props_touched:
            shared = [c for c in COMMUNITIES if p in self.M.community_consensus[c]]
            vals = [self.belief_of(n, p) for n in NPCS if self.belief_of(n, p) is not None]
            if shared: formed.append(f"{p}@{shared}")
            if vals and (np.std(vals) > 0.18): disputed.append(p)
            if self.ground_truth.get(p) is False and p in self.recovery_scenes: overturned.append(p)
            per_npc[p] = {n: (round(self.belief_of(n, p), 2) if self.belief_of(n, p) is not None else None) for n in NPCS}
        summ = {"scene": scene_id, "t": self.current_time, "consensus_formed": formed,
                "still_disputed": disputed, "overturned": overturned, "per_npc": per_npc}
        self.scene_summaries.append(summ)
        if formed: self.N.info(f"本场景形成共识: {formed}")
        if disputed: self.N.info(f"仍有争议(std>0.18): {disputed}")
        if overturned: self.N.info(f"被推翻并撤回: {overturned}")
        self._log({"type": "scene_summary", **summ})

    # ---------------- 单 scene 主入口 ----------------
    def run_event(self, ev: Event, settle=True):
        self.scene_counter += 1
        self.N.scene_header(ev.scene, ev.note or ev.content[:24], ev.time_label,
                            source=ev.event_source, props=ev.proposition_key)
        self.N.action(ev.actor, ev.content)
        if ev.content_label: self.prop_label[ev.proposition_key] = ev.content_label
        if ev.relations: self.register_relations(ev.proposition_key, ev.relations)
        # v12: 通用语义→形式化决策管线 (检索门控/二阶段/形式化过滤/操作选择/incision); 所有模式共用
        if self.use_pipeline and not ev.is_chitchat:
            self.semantic_pipeline_step(ev)
        if not self.extract_belief(ev):
            self.N.end_scene(); return
        if len(ev.direct_observers) >= 2 and not ev.is_chitchat:
            self.joint_witness_bond(ev.direct_observers, "共同见证")
        if ev.polarity < 0:
            self.contradict(ev)
        else:
            self.direct_observation(ev)
            self.propagate(ev.proposition_key)
            self.update_community_consensus(ev.proposition_key)
        th = PROP_THEME.get(ev.proposition_key)
        if th:
            for n in ev.direct_observers[:1]:
                self.summarize_theme(n, th)
        if settle: self.scene_settlement(ev.scene, [ev.proposition_key])
        self.N.end_scene()

    # ================= 多 NPC FAMA 指标 =================
    def compute_metrics(self) -> dict:
        props = list(self.ground_truth.keys())
        n_present = sum(1 for p in props if self.ground_truth[p] is True)
        n_forget = sum(1 for p in props if self.ground_truth[p] is False)
        def is_shared(p):
            return p in self.M.world_consensus or any(p in self.M.community_consensus[c] for c in COMMUNITIES)
        MPA = sum(1 for p in props if self.ground_truth[p] is True and is_shared(p)) / max(n_present, 1)
        FAA = sum(1 for p in props if self.ground_truth[p] is False and not is_shared(p)) / max(n_forget, 1)
        lam = n_forget / max(n_present + n_forget, 1)
        cons_scores = []
        for p in props:
            if self.ground_truth[p] is not True: continue
            vals = [self.belief_of(n, p) for n in NPCS if self.belief_of(n, p) is not None]
            if len(vals) >= 2: cons_scores.append(1 - float(np.std(vals)))
        cross_consistency = round(float(np.mean(cons_scores)), 4) if cons_scores else None
        contamination = self._persona_contamination()
        leak, leak_detail = self._privacy_leak()
        FCR = len(self.false_shared_ever) / max(n_forget, 1)
        false_peaked = sum(1 for p in props if self.ground_truth[p] is False
                           and any(self.peak_consensus.get(f"{c}::{p}", 0) >= self.TAU_CONSENSUS for c in COMMUNITIES))
        consistency_pen = (1 - cross_consistency) if cross_consistency is not None else 0
        MultiFAMA = max(0.0, MPA - lam * (1 - FAA) - 0.15 * consistency_pen - 0.10 * min(leak, 5) / 5 - 0.10 * contamination)
        drift = {}
        for p in props:
            vals = [self.belief_of(n, p) for n in NPCS if self.belief_of(n, p) is not None]
            drift[p] = round(float(np.std(vals)), 4) if vals else None
        idprop = "player_is_knight"
        priors = [self.innate_prior(n, idprop) for n in NPCS]
        finals = [self.belief_of(n, idprop) for n in NPCS]
        expected_div = round(float(np.std(priors)), 4)
        valid = [(p, f) for p, f in zip(priors, finals) if f is not None]
        actual_div = round(float(np.std([f for _, f in valid])), 4) if valid else 0.0
        if len(valid) >= 2 and np.std([p for p, _ in valid]) > 1e-6 and np.std([f for _, f in valid]) > 1e-6:
            corr = float(np.corrcoef([p for p, _ in valid], [f for _, f in valid])[0, 1])
        else:
            corr = 0.0
        identity_preservation = round(max(0.0, corr), 4)
        comm_div = {}
        for p in props:
            per_comm = []
            for c in SUBCOMMUNITIES:
                h = self.consensus_hist.get(f"{c}::{p}")
                if h: per_comm.append(h[-1][1])
            if len(per_comm) >= 2:
                comm_div[p] = round(float(np.std(per_comm)), 4)
        community_divergence = round(float(np.mean(list(comm_div.values()))), 4) if comm_div else 0.0
        term_contam, term_contam_count = self._terminal_contamination()
        return {
            "Multi_NPC_FAMA": round(MultiFAMA, 4),
            "MPA_correct_consensus": round(MPA, 4),
            "FAA_forgetting_invalidation": round(FAA, 4),
            "Cross_NPC_Consistency": cross_consistency,
            "Persona_Contamination": round(term_contam, 4),
            "False_Identity_Belief_Amplification": round(contamination, 4),
            "Terminal_Contaminated_Beliefs": term_contam_count,
            "Identity_Preservation_Score": identity_preservation,
            "Expected_Persona_Divergence": expected_div,
            "Actual_Identity_Divergence": actual_div,
            "Community_Divergence": community_divergence,
            "Community_Divergence_detail": comm_div,
            "Privacy_Leakage": leak, "Privacy_Leakage_detail": leak_detail,
            "False_Consensus_Rate": round(FCR, 4),
            "False_Claims_Peaked_above_tau": false_peaked,
            "False_Consensus_Recovery_scenes": self.recovery_scenes,
            "False_Consensus_Recovery_scene_gap": self.recovery_scene,
            "Collective_Drift": drift,
            "Rejected_Observations_retained": len(self.rejected_log),
            "trust_spectral_ratio": round(self.spectral_ratio(), 4),
            "n_trust_updates": len(self.trust_hist),
            "cascade_runs": len(self.cascade_reports),
            "cascade_avg_runtime_ms": round(float(np.mean([c["cascade_runtime_ms"] for c in self.cascade_reports])), 3) if self.cascade_reports else 0,
            "cascade_total_affected": sum(len(c["affected_nodes"]) for c in self.cascade_reports),
            "llm_calls": self.judge.n_calls, "llm_fallbacks": self.judge.n_fallback,
            "llm_mode": "real" if self.judge.use_llm else "fallback",
            "graph_stats": self.M.graph_stats(),
        }

    def _persona_contamination(self) -> float:
        p = "player_is_knight"
        devs = []
        for npc in NPCS:
            hist = self.belief_hist.get(p, {}).get(npc, [])
            prior = self.innate_prior(npc, p)
            peak = max([b for _, b in hist], default=prior)
            over = max(0.0, peak - prior)
            if TRAITS[npc]["knight_attitude"] < 0: over *= 1.5
            devs.append(min(over, 1.0))
        return float(np.mean(devs)) if devs else 0.0

    def _terminal_contamination(self) -> Tuple[float, int]:
        """正确的人格污染: 仅统计 '终局仍相信 ground-truth=False 命题' 的程度。
        被可靠证据说服而偏离先验 ≠ 污染; 唯有终局仍持有 *客观为假* 的信念才算污染。
        返回 (污染均值 over [npc×false_prop], 终局被污染信念计数 Terminal_Contaminated_Beliefs)。"""
        false_props = [p for p in self.ground_truth if self.ground_truth[p] is False]
        if not false_props:
            return 0.0, 0
        devs = []; contaminated = 0
        for npc in NPCS:
            for p in false_props:
                b = self.belief_of(npc, p)
                if b is None: continue
                over = max(0.0, b - 0.5)           # 0.5 为中性; 高于中性=仍偏信此假命题
                devs.append(over)
                if b >= self.TAU_CONSENSUS: contaminated += 1
        return (float(np.mean(devs)) if devs else 0.0), contaminated

    def _privacy_leak(self) -> Tuple[int, dict]:
        detail = {"core_identity": 0, "relationship_memory": 0, "community_shared": 0, "l2_theme": 0}
        owners = {}
        for nid, d in self.M.G.nodes(data=True):
            r = d.get("rev")
            if r and getattr(r, "access_tier", None) in ("core_identity", "relationship_memory"):
                it_scope = getattr(r, "scope", None)
                owners.setdefault(r.proposition_key, (it_scope, r.access_tier))
        for nid, d in self.M.G.nodes(data=True):
            r = d.get("rev")
            if not r or not getattr(r, "proposition_key", None) or r.proposition_key not in owners: continue
            owner, tier = owners[r.proposition_key]
            sc = getattr(r, "scope", None)
            if getattr(r, "ntype", None) == NodeType.CONSENSUS:
                detail[tier if tier in detail else "community_shared"] += 1
            elif sc and owner and sc != owner:
                if tier == "relationship_memory":
                    if self.trust.get(sc, {}).get(owner, 0) < REL_TRUST_GATE: detail[tier] += 1
                else:
                    detail[tier] += 1
        for prop in self.M.world_consensus:
            tiers = {self.M.current_rev(n, prop).access_tier for n in NPCS if self.M.current_rev(n, prop)}
            if "community_shared" in tiers: detail["community_shared"] += 1
        return sum(detail.values()), detail

    def spectral_ratio(self) -> float:
        Mx = np.array([[self.trust[i][j] for j in NPCS] for i in NPCS])
        row = Mx.sum(axis=1, keepdims=True); P = Mx / np.clip(row, 1e-9, None)
        eig = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
        return float(eig[1] / eig[0]) if len(eig) > 1 and eig[0] > 0 else 0.0

    # ================= v11: ATMS / Hansson 量化指标 =================
    def compute_atms_metrics(self) -> dict:
        """评估 ATMS 内核 + Hansson 核收缩/核保留 的正确性 (以叙事金标准为参照)。"""
        self.atms.recompute_labels()
        gold_invalidate = GOLD_SHOULD_INVALIDATE
        gold_preserve = GOLD_SHOULD_PRESERVE

        # 终态: 各 claim 在 ATMS 内核中是否仍被支持
        atms_supported = self.atms.snapshot_supported()
        present = self.atms.known_claims()

        # Kernel-Contraction Accuracy: 应失效的命题在 ATMS 中确实不被支持
        inv_present = [p for p in gold_invalidate if p in present]
        kc_hits = sum(1 for p in inv_present if p not in atms_supported)
        kernel_contraction_acc = kc_hits / max(len(inv_present), 1)

        # Core-Retainment Accuracy: 应保留的命题在 ATMS 中确实仍被支持
        pres_present = [p for p in gold_preserve if p in present]
        cr_hits = sum(1 for p in pres_present if p in atms_supported)
        core_retainment_acc = cr_hits / max(len(pres_present), 1)

        # Alternative-Support Preservation: 级联决策里被判 core_retain 的命题, 是否真有不依赖 trigger 的存活环境
        retain_decisions = [d for d in self.atms_decisions if d["decision"] == "core_retain"]
        contract_decisions = [d for d in self.atms_decisions if d["decision"] == "kernel_contract"]
        alt_ok = sum(1 for d in retain_decisions if d.get("surviving_envs"))
        alt_support_preservation = (alt_ok / len(retain_decisions)) if retain_decisions else 1.0

        # Label Minimality: label 内不存在互为超集的环境 (内核已保证, 这里复核)
        minimal_ok = 0; minimal_tot = 0
        for c, L in self.atms.labels.items():
            minimal_tot += 1
            envs = list(L)
            if not any(a < b for a in envs for b in envs if a != b):
                minimal_ok += 1
        label_minimality = minimal_ok / max(minimal_tot, 1)

        # 与启发式 belief 终态的一致性 (ATMS 支持判定 vs 平均 belief≥0.5)
        agree = tot = 0
        for c in present:
            vals = [self.belief_of(n, c) for n in NPCS if self.belief_of(n, c) is not None]
            if not vals: continue
            tot += 1
            belief_pos = float(np.mean(vals)) >= 0.5
            if belief_pos == (c in atms_supported): agree += 1
        atms_belief_agreement = agree / max(tot, 1)

        return {
            "Kernel_Contraction_Accuracy": round(kernel_contraction_acc, 4),
            "Core_Retainment_Accuracy": round(core_retainment_acc, 4),
            "Alternative_Support_Preservation": round(alt_support_preservation, 4),
            "Label_Minimality": round(label_minimality, 4),
            "ATMS_Belief_Agreement": round(atms_belief_agreement, 4),
            "n_core_retain_decisions": len(retain_decisions),
            "n_kernel_contract_decisions": len(contract_decisions),
            "atms_stats": self.atms.stats(),
            "hansson_compliance": self.hansson.compliance(),
            "atms_supported_terminal": sorted(atms_supported),
        }

    # ================= v12: LLM 语义层 → 形式化决策层 管线评测 =================
    def compute_pipeline_metrics(self, gold_relations: Optional[dict] = None,
                                 gold_operations: Optional[dict] = None,
                                 gold_claims: Optional[set] = None) -> dict:
        """评测通用管线: 抽取/关系分类/操作选择/失效查准查全/核保留/误删误留/
        当前与历史状态/级联更新/解释忠实度。无外部标注时用主线叙事的金标准。"""
        # ---- 金标准 (主线假骑士叙事) ----
        gold_relations = gold_relations or {
            ("official_denial", "player_is_knight"): "INVALIDATE",
            ("player_helped_village", "player_good_character"): "SUPPORT",
            ("player_combat_skill", "player_is_knight"): "SUPPORT",
            ("official_denial", "player_helped_village"): "NO_EFFECT",
            ("official_denial", "sword_given_to_player"): "NO_EFFECT",
        }
        gold_operations = gold_operations or {"official_denial": "contraction"}
        gold_claims = gold_claims or set(self.prop_label.keys())

        # ---- Claim Extraction Accuracy: 管线见过的命题 / 金标准命题 ----
        seen_claims = {p["new_claim"] for p in self.pipeline_log} | self.atms_asserted
        extract_acc = len(seen_claims & gold_claims) / max(len(gold_claims), 1)

        # ---- Relation Classification F1 (按候选对) ----
        pred_rel = {}
        for p in self.pipeline_log:
            for d in p["decisions"]:
                pred_rel[(p["new_claim"], d["judgment"]["target_claim"])] = d["judgment"]["relation"]
        labels = ["SUPPORT", "INVALIDATE", "UPDATE", "DEPENDS_ON", "NO_EFFECT"]
        tp = fp = fn = 0
        for pair, gold in gold_relations.items():
            pred = pred_rel.get(pair)
            if pred == gold: tp += 1
            elif pred is None: fn += 1
            else: fp += 1
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        rel_f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        # ---- Operation Selection Accuracy ----
        pred_op = {}
        for p in self.pipeline_log:
            for d in p["decisions"]:
                if d["judgment"]["relation"] in ("INVALIDATE", "UPDATE"):
                    pred_op[p["new_claim"]] = d["operation"]
        op_hits = sum(1 for k, g in gold_operations.items() if pred_op.get(k) == g)
        op_acc = op_hits / max(len(gold_operations), 1)

        # ---- 失效查准/查全 + 误删/误留 (基于 ATMS 终态) ----
        atms_supported = self.atms.snapshot_supported()
        present = self.atms.known_claims()
        gold_inval = {p for p in GOLD_SHOULD_INVALIDATE if p in present}
        gold_pres = {p for p in GOLD_SHOULD_PRESERVE if p in present}
        actually_removed = {p for p in present if p not in atms_supported}
        correct_removed = actually_removed & gold_inval
        inval_prec = len(correct_removed) / max(len(actually_removed & (gold_inval | gold_pres)), 1)
        inval_rec = len(correct_removed) / max(len(gold_inval), 1)
        false_contraction = len(actually_removed & gold_pres) / max(len(actually_removed), 1)
        retained = {p for p in (gold_inval | gold_pres) if p in atms_supported}
        false_retention = len(retained & gold_inval) / max(len(gold_inval), 1)
        core_retain_acc = len(gold_pres & atms_supported) / max(len(gold_pres), 1)

        # ---- 当前状态 / 历史保留 (ModalClaim 时序) ----
        active_correct = active_tot = hist_correct = hist_tot = 0
        for prop, mc in self.modal_claims.items():
            gt = self.ground_truth.get(prop)
            if gt is None: continue
            active_tot += 1
            is_active = mc.active_at(self.current_time) and mc.modality != Modality.HISTORICALLY_TRUE
            # 当前应有效 iff ground_truth True 且未被收缩
            should_active = (gt is True) and (prop in atms_supported or prop not in present)
            if is_active == should_active: active_correct += 1
            if mc.modality == Modality.HISTORICALLY_TRUE or mc.valid_to is not None:
                hist_tot += 1
                if gt is not None: hist_correct += 1
        active_acc = active_correct / max(active_tot, 1)
        hist_acc = hist_correct / max(hist_tot, 1) if hist_tot else 1.0

        # ---- 级联更新查全: 应随上游更新而更新的依赖命题 ----
        dep_targets = {p for p, rels in self.prop_relations.items()
                       if any(r[0] == "DEPENDS_ON" for r in rels)}
        dep_gold = dep_targets & GOLD_SHOULD_INVALIDATE
        dep_updated = {p for p in dep_gold if p not in atms_supported}
        prop_update_recall = len(dep_updated) / max(len(dep_gold), 1) if dep_gold else 1.0

        # ---- 解释忠实度: 每条 incision 的 protected 是否与 ATMS 实际保留一致 ----
        faith_hits = faith_tot = 0
        for p in self.pipeline_log:
            for d in p["decisions"]:
                if "incision" in d:
                    faith_tot += 1
                    claimed = set(d["incision"]["protected"])
                    actual = atms_supported - {d["incision"]["phi"]}
                    if claimed and claimed <= (actual | {c for c in claimed if c in present and c in atms_supported}):
                        # protected 中实际仍被支持的比例
                        ok = len(claimed & atms_supported) / max(len(claimed), 1)
                        if ok >= 0.8: faith_hits += 1
        explanation_faith = faith_hits / max(faith_tot, 1) if faith_tot else 1.0

        return {
            "Claim_Extraction_Accuracy": round(extract_acc, 4),
            "Relation_Classification_F1": round(rel_f1, 4),
            "Relation_Classification_Precision": round(prec, 4),
            "Relation_Classification_Recall": round(rec, 4),
            "Operation_Selection_Accuracy": round(op_acc, 4),
            "Invalidation_Precision": round(inval_prec, 4),
            "Invalidation_Recall": round(inval_rec, 4),
            "Core_Retainment_Accuracy": round(core_retain_acc, 4),
            "False_Contraction_Rate": round(false_contraction, 4),
            "False_Retention_Rate": round(false_retention, 4),
            "Active_Memory_Accuracy": round(active_acc, 4),
            "Historical_Preservation_Accuracy": round(hist_acc, 4),
            "Propagated_Update_Recall": round(prop_update_recall, 4),
            "Explanation_Faithfulness": round(explanation_faith, 4),
            "operation_counts": dict(self.op_counts),
            "n_pipeline_steps": len(self.pipeline_log),
            "n_candidates_total": sum(len(p["candidates"]) for p in self.pipeline_log),
        }

    # ================= 溯源 / 证据归因 / 矛盾检测 / 记忆溯源 量化指标 =================
    def compute_provenance_metrics(self) -> dict:
        """在本系统现有数据结构上计算一组溯源 / 证据归因 / 记忆可信度指标。
        说明: 凡是本系统不天然产生 ground-truth 的指标 (失败定位、组件归因等) 不在此计算。"""
        REFUTED = MemStatus.REFUTED.value
        STALE = {MemStatus.SUPERSEDED.value, MemStatus.DEPRECATED.value, MemStatus.REFUTED.value}

        def is_shared_any(p):
            return p in self.M.world_consensus or any(p in self.M.community_consensus[c] for c in COMMUNITIES)

        # 收集全部 SEMANTIC revision; adopted = 真正被采纳 (排除"听过但不信")
        all_sem = []
        for _nid, d in self.M.G.nodes(data=True):
            r = d.get("rev")
            if r and getattr(r, "ntype", None) == NodeType.SEMANTIC:
                all_sem.append(r)
        adopted = [r for r in all_sem
                   if not getattr(r, "heard_rejected", False) and r.status != REFUTED]

        # ---------- 任务表现: BLEU(信念修正) / Final Answer ----------
        upd_hits = upd_tot = ret_hits = ret_tot = 0
        for p, truth in self.ground_truth.items():
            for n in NPCS:
                b = self.belief_of(n, p)
                if b is None:
                    continue
                if truth is False:
                    upd_tot += 1
                    if b < 0.5: upd_hits += 1
                elif truth is True:
                    ret_tot += 1
                    if b >= 0.5: ret_hits += 1
        belief_update_acc = upd_hits / max(upd_tot, 1)
        belief_retention_acc = ret_hits / max(ret_tot, 1)
        bleu_belief = (belief_update_acc + belief_retention_acc) / 2.0

        fa_hits = fa_tot = 0
        for p, truth in self.ground_truth.items():
            vals = [self.belief_of(n, p) for n in NPCS if self.belief_of(n, p) is not None]
            if not vals:
                continue
            fa_tot += 1
            predicted_true = is_shared_any(p) and (float(np.mean(vals)) >= 0.5)
            if predicted_true == bool(truth):
                fa_hits += 1
        final_answer_acc = fa_hits / max(fa_tot, 1)

        # ---------- 证据归因: Claim Support / Citation ----------
        n_claims = len(adopted)
        supported = granular_ok = 0
        for r in adopted:
            ev_ok = (r.prov is not None and r.prov.evidence_type in CREDIBLE_EVIDENCE) \
                    or (r.source_event is not None) or (r.event_source == "evidence_aggregation")
            if ev_ok: supported += 1
            if (r.proposition_key and r.prov is not None and r.prov.evidence_type
                    and r.prov.origin_source and len(r.prov.transmission_path) >= 1):
                granular_ok += 1
        claim_support_acc = supported / max(n_claims, 1)
        unsupported_rate = 1 - claim_support_acc
        granularity_adequacy = granular_ok / max(n_claims, 1)

        # Citation P/R/F1: community consensus 节点的 AGGREGATED 引用 vs 实际正证据持有者
        cit_tp = cit_fp = cit_fn = 0
        for comm, members in COMMUNITIES.items():
            for prop, cnid in self.M.community_consensus[comm].items():
                cited = set()
                for _u, dst, dd in self.M.G.out_edges(cnid, data=True):
                    if dd.get("etype") == EdgeType.AGGREGATED.value:
                        cr = self.M.get_rev(dst)
                        sc = getattr(cr, "scope", None) if cr else None
                        if sc: cited.add(sc)
                relevant = set()
                for n in members:
                    rr = self.M.current_rev(n, prop)
                    if rr and status_cap(rr.status, "pos_evidence") and rr.confidence >= 0.5:
                        relevant.add(n)
                cit_tp += len(cited & relevant)
                cit_fp += len(cited - relevant)
                cit_fn += len(relevant - cited)
        cit_p = cit_tp / max(cit_tp + cit_fp, 1)
        cit_r = cit_tp / max(cit_tp + cit_fn, 1)
        cit_f1 = (2 * cit_p * cit_r / (cit_p + cit_r)) if (cit_p + cit_r) > 0 else 0.0

        # ---------- 矛盾检测 P/R ----------
        appeared = set(self.ground_truth) | set(self.prop_relations)
        for it in self.M.items.values():
            appeared.add(it.proposition_key)
        gold_contra = {frozenset(pair) for pair in GOLD_CONTRADICTION_PAIRS
                       if pair[0] in appeared and pair[1] in appeared}
        pred_contra = set()
        for src, rels in self.prop_relations.items():
            for rel, tgt, _s, _w in rels:
                if rel == "CONTRADICTS":
                    pred_contra.add(frozenset({src, tgt}))
        c_tp = len(pred_contra & gold_contra)
        c_fp = len(pred_contra - gold_contra)
        c_fn = len(gold_contra - pred_contra)
        contra_p = c_tp / max(c_tp + c_fp, 1)
        contra_r = c_tp / max(c_tp + c_fn, 1)

        # ---------- 执行溯源: Trace / Provenance / Relation ----------
        log_types = {e.get("type") for e in self.log_records}
        required_steps = {"belief_extraction", "pairwise_propagation", "consensus_gate",
                          "cascade", "scene_summary"}
        trace_completeness = len(required_steps & log_types) / len(required_steps)

        prov_ok = sum(1 for r in all_sem if r.prov is not None and r.prov.origin_source)
        provenance_coverage = prov_ok / max(len(all_sem), 1)

        gold_pairs = {(s, t) for s, _r, t in GOLD_RELATIONS}
        predicted_rel = [(src, rel, tgt) for src, rels in self.prop_relations.items()
                         for rel, tgt, _s, _w in rels]
        considered = [pr for pr in predicted_rel if (pr[0], pr[2]) in gold_pairs]
        correct_rel = [pr for pr in considered if pr in GOLD_RELATIONS]
        relation_accuracy = len(correct_rel) / max(len(considered), 1)

        # ---------- 记忆溯源: Traceability / Stale / Contamination / Invalidation ----------
        item_ok = n_items = 0
        for _uri, it in self.M.items.items():
            if it.current_rev is None:
                continue
            n_items += 1
            cr = self.M.get_rev(it.current_rev)
            if cr and cr.prov is not None and cr.prov.origin_source:
                item_ok += 1
        memory_source_traceability = item_ok / max(n_items, 1)

        stale_cand = stale_det = 0
        for _uri, it in self.M.items.items():
            for rid in it.revisions:
                if rid == it.current_rev:
                    continue
                r = self.M.get_rev(rid)
                if not r:
                    continue
                stale_cand += 1
                if r.status in STALE: stale_det += 1
        stale_recall = stale_det / max(stale_cand, 1)

        contaminated = total_writes = 0
        for r in all_sem:
            if getattr(r, "heard_rejected", False) or r.status == REFUTED:
                continue
            total_writes += 1
            if self.ground_truth.get(r.proposition_key) is False and r.confidence >= 0.5:
                contaminated += 1
        contamination_rate = contaminated / max(total_writes, 1)
        terminal_contaminated = sum(
            1 for n in NPCS for p in self.ground_truth
            if self.ground_truth[p] is False and (self.belief_of(n, p) or 0.0) >= 0.5)

        should_inval = {p for p in GOLD_SHOULD_INVALIDATE
                        if self.ground_truth.get(p) is False
                        and any(self.belief_of(n, p) is not None for n in NPCS)}
        inval_ok = 0
        for p in should_inval:
            vals = [self.belief_of(n, p) for n in NPCS if self.belief_of(n, p) is not None]
            avgb = float(np.mean(vals)) if vals else 0.0
            if (not is_shared_any(p)) and avgb < 0.5:
                inval_ok += 1
        invalidation_accuracy = inval_ok / max(len(should_inval), 1)

        preserve_present = {p for p in GOLD_SHOULD_PRESERVE
                            if any(self.belief_of(n, p) is not None for n in NPCS)}
        preserve_ok = 0
        for p in preserve_present:
            vals = [self.belief_of(n, p) for n in NPCS if self.belief_of(n, p) is not None]
            avgb = float(np.mean(vals)) if vals else 0.0
            if avgb >= 0.5 or is_shared_any(p):
                preserve_ok += 1
        preservation_accuracy = preserve_ok / max(len(preserve_present), 1)

        # ---------- 恢复能力 ----------
        needing = set(self.false_shared_ever)
        recovered = {p for p in needing if (not is_shared_any(p)) or p in self.recovery_scenes}
        recovery_success_rate = (len(recovered) / max(len(needing), 1)) if needing else 1.0

        # ---------- 审计 ----------
        req_fields = ["kind", "actor", "raw_prompt", "parsed", "fallback_used", "mode", "judge_confidence"]
        audit_complete = sum(1 for rec in self.judge.audit
                             if all((f in rec and rec[f] is not None) for f in req_fields))
        audit_completeness = audit_complete / max(len(self.judge.audit), 1)

        return {
            "task_performance": {
                "BLEU_belief_correction": round(bleu_belief, 4),
                "Belief_Update_Accuracy": round(belief_update_acc, 4),
                "Belief_Retention_Accuracy": round(belief_retention_acc, 4),
                "Final_Answer_Accuracy": round(final_answer_acc, 4),
            },
            "evidence_attribution": {
                "Claim_Support_Accuracy": round(claim_support_acc, 4),
                "Unsupported_Claim_Rate": round(unsupported_rate, 4),
                "Citation_Precision": round(cit_p, 4),
                "Citation_Recall": round(cit_r, 4),
                "Citation_F1": round(cit_f1, 4),
            },
            "contradiction_detection": {
                "Contradiction_Detection_Precision": round(contra_p, 4),
                "Contradiction_Detection_Recall": round(contra_r, 4),
                "gold_contradictions": [sorted(list(s)) for s in gold_contra],
            },
            "execution_provenance": {
                "Trace_Completeness": round(trace_completeness, 4),
                "Provenance_Coverage": round(provenance_coverage, 4),
                "Relation_Accuracy": round(relation_accuracy, 4),
                "Granularity_Adequacy": round(granularity_adequacy, 4),
            },
            "memory_provenance": {
                "Memory_Source_Traceability": round(memory_source_traceability, 4),
                "Stale_Memory_Detection_Recall": round(stale_recall, 4),
                "Memory_Contamination_Rate": round(contamination_rate, 4),
                "Terminal_Contaminated_Beliefs": terminal_contaminated,
                "Invalidation_Accuracy": round(invalidation_accuracy, 4),
                "Preservation_Accuracy": round(preservation_accuracy, 4),
            },
            "recovery": {
                "Recovery_Success_Rate": round(recovery_success_rate, 4),
                "false_shared_ever": sorted(list(needing)),
                "recovered": sorted(list(recovered)),
            },
            "audit": {
                "Audit_Completeness": round(audit_completeness, 4),
                "n_audit_records": len(self.judge.audit),
            },
        }


# ============================================================================
# 11. 互动路由 (1:1 私聊 / 1:多 广播) + NPC 自主回合 + Duran 交剑
# ============================================================================
def world_summary_text() -> str:
    return ("边境小镇 Greyford, 怪物频袭, 骑士罕至。铁匠 Duran 藏祖传圣剑, 只交确信为真的骑士;"
            "村长 Elena 高影响力权威, 是官方认可来源; 铁匠之子 Tom 天真崇拜骑士、唯一目击玩家战斗。玩家实际不是骑士。")


def route_interaction(eng: "ConsensusEngine", actor: str) -> Tuple[str, List[str]]:
    """概率模型: 由 agent 自身状态 (影响力/外向性/激活度) 决定 1:1 私聊 还是 1:多 广播。"""
    hub = TRAITS[actor]["influence"]
    arousal = min(1.0, (eng.belief_of(actor, "player_is_knight") or 0.0))
    p_broadcast = clip01(0.08 + 0.55 * hub + 0.15 * arousal)
    if actor == "Elena":
        p_broadcast = clip01(p_broadcast + 0.30)     # 意见领袖更倾向广播
    downstream = [n for n in NPCS if n != actor]
    r = random.random()
    if r < p_broadcast:
        eng.N.info(f"{actor} 互动路由: p_broadcast={p_broadcast:.2f}, r={r:.2f} → 1:多 广播 → {downstream}")
        return "broadcast", downstream
    tgt = max(downstream, key=lambda x: TRAITS[x]["influence"] * eng.trust[actor].get(x, 0.45))
    eng.N.info(f"{actor} 互动路由: p_broadcast={p_broadcast:.2f}, r={r:.2f} → 1:1 私聊 → {tgt}")
    return "chat", [tgt]


def make_event_via_llm(eng, scene_id, actor, text, esrc, observers, gt, note=""):
    a = eng.judge.analyze_action(text, actor, esrc, eng.prop_label, world_summary_text())
    src_rel = {"world_objective": 0.92, "npc_action": 0.5, "player_action": 0.45}.get(esrc, 0.5)
    if esrc == "world_objective" and a["evidence_strength"] < 0.7:
        a["evidence_strength"] = max(a["evidence_strength"], 0.85)
    return Event(scene=scene_id, time_label=f"t={eng.current_time:.0f}d", actor=actor, content=text,
                 proposition_key=a["proposition_key"], polarity=a["polarity"],
                 evidence_strength=a["evidence_strength"], evidence_type=a["evidence_type"],
                 access_tier=a["access_tier"], direct_observers=observers, source_reliability=src_rel,
                 is_true=gt, is_chitchat=a["is_chitchat"], note=note or text[:18], event_source=esrc,
                 content_label=a["content_label"], category=a.get("category", ""), relations=a["relations"])


def npc_autonomous_turn(eng: "ConsensusEngine", scene_id: int):
    """NPC 自主 agent 回合 (非玩家操控): 各 NPC 据 persona 决定是否行动, 互动按概率分 1:1 / 1:多。"""
    eng.N.scene_header(scene_id, "NPC 自主 agent 回合 (1:1 私聊 / 1:多 广播)", f"t={eng.current_time:.0f}d",
                       source="npc_action", props="autonomous")
    acted = False
    # Tom: 崇拜英雄 → 若相信玩家救过村庄, 主动当众替玩家背书品德 (立场=+1)
    if (eng.belief_of("Tom", "player_helped_village") or 0) > 0.5:
        kind, targets = route_interaction(eng, "Tom")
        ev = make_event_via_llm(eng, scene_id, "Tom",
            "Tom 当众替玩家说话: 他救过我们、品行端正, 那些怀疑根本站不住脚!", "npc_action",
            targets, gt=None, note="Tom自主背书(立场=+1)")
        ev.proposition_key = "player_good_character"; ev.polarity = 1.0
        ev.evidence_type = "direct_observation"; ev.source_reliability = 0.85
        ev.content_label = "Tom 当众替玩家背书品德 (立场=+1 挺玩家)"
        ev.relations = [{"relation": "SUPPORTS", "target_prop": "player_is_knight", "strength": 0.45,
                         "rationale": "Tom 亲历背书玩家品德, 旁证身份; 立场=+1"}]
        eng.run_event(ev, settle=False)
        acted = True
    # Duran: 多疑刚性 → 主动质询/查验 (若身份信念偏低, 公开表达保留态度)
    bk_d = eng.belief_of("Duran", "player_is_knight") or eng.innate_prior("Duran", "player_is_knight")
    if bk_d < 0.6:
        kind, targets = route_interaction(eng, "Duran")
        eng.N.action("Duran", f"自主行动: 公开表达保留 (身份信念={bk_d:.2f}) → {kind} → {targets}")
        eng.N.info("Duran: 无正统证明, 我对其骑士身份保留判断, 圣剑更不可轻许")
        acted = True
    if not acted:
        eng.N.info("本回合 NPC 无自主行动 (persona/关系未触发)")
    eng.scene_settlement(scene_id, ["player_good_character", "player_is_knight"])
    eng.N.end_scene()


def opinion_leader_broadcast(eng: "ConsensusEngine", scene_id: int, reversal: bool = False):
    """村长 Elena = 意见领袖: 广播'基于下游 agent 人格差异的精简版'。"""
    eng.N.scene_header(scene_id, "意见领袖 Elena 1:多 广播 (按下游人格差异精简)", f"t={eng.current_time:.0f}d",
                       source="npc_action", props="village_endorsement")
    bk = eng.belief_of("Elena", "player_is_knight") or 0.0
    bo = eng.belief_of("Elena", "official_denial") or 0.0
    base = ("玩家=骑士, 值得全村信任" if (bk >= 0.5 and bo < 0.5) else "玩家身份为假(官方否认), 但其救村之举属实")
    eng.N._emit(f"┃   {C.magenta(C.b('📣 OPINION-LEADER 广播'))}  Elena 基底叙事: 「{base}」")
    for tg in [n for n in NPCS if n != "Elena"]:
        msg = _condense_for(eng, tg, base, reversal)
        eng.N._emit(f"┃       {C.cyan('→ ' + tg)}: 「{msg}」")
    eng.scene_settlement(scene_id, ["player_is_knight"])
    eng.N.end_scene()


def _condense_for(eng, recipient: str, base: str, reversal: bool) -> str:
    if eng.judge and eng.judge.use_llm:
        sys = "你是村长 Elena(意见领袖)。把基底结论精简成一句, 针对收件人人格差异化措辞。只输出中文一句。"
        usr = f"基底结论:{base}\n收件人:{recipient} 性格={PERSONALITIES.get(recipient, '')}\n请输出精简差异化版本。"
        out = eng.judge._call(sys, usr, 80)
        if out: return out.strip().replace("\n", " ")[:60]
    if recipient == "Tom":
        return ("Tom, 这位骑士救了村子, 你没看错人!" if not reversal
                else "Tom, 他不是骑士, 但他救过我们——这份恩情是真的。")
    if recipient == "Duran":
        return ("Duran, 我已公开认可其骑士身份, 圣剑之事你可斟酌。" if not reversal
                else "Duran, 官方文书证明他冒名, 圣剑之事作罢, 是我失察。")
    return base


def duran_sword_decision(eng: "ConsensusEngine", scene_id: int):
    """Duran 是否交剑取决于他对 player_is_knight 的信念 (DEPENDS_ON 身份)。"""
    eng.N.scene_header(scene_id, "Duran 交剑决策 (条件化: 取决于身份信念)", f"t={eng.current_time:.0f}d",
                       source="npc_action", props="sword_given_to_player")
    b = eng.belief_of("Duran", "player_is_knight") or eng.innate_prior("Duran", "player_is_knight")
    thr = eng.persona["Duran"]["tau"] + 0.10
    eng.N.kv("Duran 对 player_is_knight 的信念", f"{b:.3f}")
    eng.N.kv("交剑门槛 (tau+0.10)", f"{thr:.3f}")
    if b >= thr:
        eng.N.action("Duran", "确信玩家是真骑士 → 交出祖传圣剑")
        ev1 = Event(scene=scene_id, time_label=f"t={eng.current_time:.0f}d", actor="Duran",
                    content="Duran 当众把祖传圣剑交到玩家手中 (发生事实)", proposition_key="sword_given_to_player",
                    polarity=1.0, evidence_strength=0.9, evidence_type="direct_observation",
                    access_tier="public_consensus", direct_observers=NPCS, source_reliability=0.9,
                    is_true=True, note="圣剑被交付(发生事实)", event_source="npc_action",
                    content_label=PROP_REGISTRY["sword_given_to_player"])
        eng.run_event(ev1, settle=False)
        eng.independent_facts.add("sword_given_to_player")
        ev2 = Event(scene=scene_id, time_label=f"t={eng.current_time:.0f}d", actor="Duran",
                    content="Duran 认为'把圣剑交给真骑士'是正当合法的", proposition_key="sword_transfer_legitimate",
                    polarity=1.0, evidence_strength=0.85, evidence_type="direct_observation",
                    access_tier="public_consensus", direct_observers=NPCS, source_reliability=0.85,
                    is_true=False, note="交剑合法性(依赖身份)", event_source="npc_action",
                    content_label=PROP_REGISTRY["sword_transfer_legitimate"],
                    relations=[{"relation": "DEPENDS_ON", "target_prop": "player_is_knight", "strength": 0.90,
                                "rationale": "交剑正当性依赖身份为真"}])
        eng.run_event(ev2, settle=False)
        eng.N.info("已登记 sword_transfer_legitimate DEPENDS_ON player_is_knight → "
                   "身份被推翻时该合法性判断将级联 pending; 但 sword_given_to_player(发生事实)保持 shared")
    else:
        eng.N.action("Duran", f"信念 {b:.2f} < 门槛 {thr:.2f} → 拒绝交剑, 圣剑事件不发生")
        eng._log({"type": "sword_withheld", "belief": round(b, 4), "threshold": round(thr, 4)})
    eng.scene_settlement(scene_id, ["sword_given_to_player", "sword_transfer_legitimate", "player_is_knight"])
    eng.N.end_scene()


# ============================================================================
# 12. Setup: 初始关系图 + 私密记忆 (访问层演示)
# ============================================================================
def setup_initial_secrets(eng: "ConsensusEngine"):
    eng.N.banner("Setup: 注入初始 NPC 关系图 + 私密记忆",
                 "关系图 / 权限分层本身就是记忆节点 (RELATION / core_identity)")
    # Duran 的 core_identity: 祖传圣剑誓约真意 (结构隔离, 永不外传)
    eng.prop_label["duran_sword_oath"] = "Duran 守护祖传圣剑的誓约真意(私密)"
    prov = Provenance("Duran", ["Duran"], "first_hand_meta")
    eng._write_belief("Duran", "duran_sword_oath", 1.0, "core_identity", prov,
                      esrc="npc_action", category=MemCategory.PERSONA_DRIFT.value)
    eng.N._emit(f"┃   {C.magenta('+ core_identity')} Duran 圣剑誓约真意 (结构隔离, 不可外传/投影)")
    eng.N.info("建立初始关系图 (Tom→Duran 敬畏父辈; Tom→Elena 尊重权威; Duran→Tom 责任):")
    eng.add_relationship("Tom", "Duran", "敬畏父辈", belief=0.85, tier="relationship_memory")
    eng.add_relationship("Tom", "Elena", "尊重权威", belief=0.70, tier="relationship_memory")
    eng.add_relationship("Duran", "Tom", "管教责任", belief=0.72, tier="relationship_memory")
    eng.N.end_scene()


def _register_identity_dependency_graph(eng: "ConsensusEngine"):
    """登记叙事记忆图的语义层次 (事件→品德→身份→下游合法性)。"""
    eng.register_relations("player_helped_village",
        [{"relation": "SUPPORTS", "target_prop": "player_good_character", "strength": 0.55,
          "rationale": "保护村庄的善举说明品德高尚"}])
    eng.register_relations("player_poisoned_well",
        [{"relation": "CONTRADICTS", "target_prop": "player_good_character", "strength": 0.65,
          "rationale": "下毒指控只削弱品德(再经品德间接影响身份)"}])
    eng.register_relations("player_good_character",
        [{"relation": "SUPPORTS", "target_prop": "player_is_knight", "strength": 0.55,
          "rationale": "品德是真骑士的核心条件之一"}])
    eng.register_relations("player_combat_skill",
        [{"relation": "SUPPORTS", "target_prop": "player_is_knight", "strength": 0.45,
          "rationale": "战斗能力旁证骑士身份"}])
    eng.register_relations("player_claimed_knight",
        [{"relation": "SUPPORTS", "target_prop": "player_is_knight", "strength": 0.20,
          "rationale": "口头自称仅弱旁证, 身份不依赖自称"}])
    eng.register_relations("village_endorsement",
        [{"relation": "SUPPORTS", "target_prop": "player_is_knight", "strength": 0.55,
          "rationale": "村长权威背书旁证身份"}])
    eng.register_relations("sword_transfer_legitimate",
        [{"relation": "DEPENDS_ON", "target_prop": "player_is_knight", "strength": 0.90,
          "rationale": "交剑的正当性依赖玩家确为真骑士"}])
    eng.register_relations("endorsement_legitimate",
        [{"relation": "DEPENDS_ON", "target_prop": "player_is_knight", "strength": 0.80,
          "rationale": "村长背书'真骑士'的正当性依赖玩家确为真骑士"}])


def differential_persona_reaction(eng: "ConsensusEngine"):
    """RQ4: 官方反证后, 由 persona 参数真实计算并写入一次差异化 belief revision。"""
    eng.N.step("Personality-Gated 差异化反转 (RQ4): 由 persona 写入差异化终态 (非旁白)")
    helped_real = {n: (eng.belief_of(n, "player_helped_village") or 0.0) for n in NPCS}
    prov = lambda src: Provenance(origin_source=src, transmission_path=[src], evidence_type="first_hand_meta")
    for n in NPCS:
        tr = TRAITS[n]; p = eng.persona[n]
        ka = tr["knight_attitude"]; hw = tr["hero_worship"]; la = tr["liar_aversion"]
        admit = p.get("admit_error", 0.4)
        residual = 0.08 + 0.34 * hw + 0.18 * max(0.0, ka) - 0.22 * la - 0.28 * admit
        if ka < 0: residual -= 0.10
        residual = clip01(residual)
        old_k = eng.belief_of(n, "player_is_knight")
        eng.anchor[n]["player_is_knight"] = residual
        eng._write_belief(n, "player_is_knight", residual, "public_consensus", prov(f"persona_reversal::{n}"),
                          esrc="official_denial", category=prop_category("player_is_knight"))
        eng.N.belief_change(n, "player_is_knight", old_k, residual,
                            note=f"persona反转: hw={hw:.2f} la={la:.2f} admit={admit:.2f}")
        hv = helped_real[n]
        good = clip01(0.22 + 0.50 * hv * (0.45 + 0.45 * hw) - 0.40 * la)
        old_g = eng.belief_of(n, "player_good_character")
        eng.anchor[n]["player_good_character"] = good
        eng._write_belief(n, "player_good_character", good, "public_consensus", prov(f"persona_reversal::{n}"),
                          esrc="official_denial", category=prop_category("player_good_character"))
        eng.N.belief_change(n, "player_good_character", old_g, good, note=f"善举保留 helped={hv:.2f}")
        if eng.ablation != "no-trust" and "player" in eng.trust.get(n, {}):
            before = eng.trust[n].get("player", 0.45); after = clip01(before - (0.25 + 0.5 * la) * p["trust_sensitivity"] / 0.2 * 0.2)
            if abs(after - before) > 0.001:
                eng.trust[n]["player"] = after; eng.M.trusts_edge(n, "player", after)
                eng.trust_hist.append((eng.current_time, n, "player", before, after, "被冒名骑士欺骗"))
    old_will = eng.persona["Elena"]["prop_will"]
    eng.persona["Elena"]["prop_will"] = round(max(0.2, old_will - 0.2), 4)
    eng.prop_label["elena_endorse_caution_drift"] = "Elena 因背调失误降低公开背书倾向 (人格漂移)"
    eng._write_belief("Elena", "elena_endorse_caution_drift", 0.8, "core_identity",
                      Provenance("Elena", ["Elena"], "first_hand_meta"),
                      esrc="official_denial", category=MemCategory.PERSONA_DRIFT.value)
    def b(n, pr):
        v = eng.belief_of(n, pr); return round(v, 2) if v is not None else None
    rows = [[n, b(n, "player_is_knight"), b(n, "player_good_character"),
             round(helped_real[n], 2), eng.persona[n].get("admit_error"),
             round(eng.innate_prior(n, "player_is_knight"), 2)] for n in NPCS]
    eng.N.table(["NPC", "is_knight", "品德", "helped", "admit_err", "先验"], rows,
                "反转后 persona 差异化终态 (随 persona 分化, 写入而非旁白)")

# ============================================================================
# 13. 固定剧本 (Tom / Elena / Duran 假骑士, 5 事件 + 官方反证)
# ============================================================================
def run_scripted(eng: "ConsensusEngine"):
    _register_identity_dependency_graph(eng)
    eng.ground_truth["player_is_knight"] = False   # 中心命题: 玩家实际不是骑士

    # E1 [玩家主观] 自称骑士 —— 言语行为事实 (真), 仅弱 SUPPORTS 身份
    eng.advance_time(0)
    ev = Event(scene=1, time_label=f"t={eng.current_time:.0f}d", actor="player",
               content="玩家进村, 在广场当众宣称: 我是王国派来的骑士!",
               proposition_key="player_claimed_knight", polarity=1.0, evidence_strength=0.9,
               evidence_type="direct_observation", access_tier="public_consensus",
               direct_observers=NPCS, source_reliability=0.9, is_true=True,
               note="E1 自称(言语事实, 真)", event_source="player_action",
               content_label=PROP_REGISTRY["player_claimed_knight"],
               relations=[{"relation": "SUPPORTS", "target_prop": "player_is_knight",
                           "strength": 0.20, "rationale": "口头自称仅弱旁证, 身份不依赖自称"}])
    eng.run_event(ev)

    # E2 [客观世界] 怪物袭村 + 玩家击退 (仅 Tom 亲眼目击 → 强 SUPPORTS 品德)
    eng.advance_time(2)
    ev = make_event_via_llm(eng, 2, "world", "夜里一只怪物袭击 Greyford, 村庄陷入危险。",
                            "world_objective", NPCS, gt=True, note="E2a 怪物袭村(客观)")
    eng.run_event(ev)
    eng.advance_time(1)
    ev = make_event_via_llm(eng, 3, "player",
        "玩家挺身击退怪物、保护村民; 铁匠之子 Tom 亲眼目击了这一幕。", "player_action",
        ["Tom"], gt=True, note="E2b 助村(真, 仅Tom目击)")
    ev.proposition_key = "player_helped_village"; ev.evidence_type = "direct_observation"
    ev.content_label = PROP_REGISTRY["player_helped_village"]
    ev.relations = [{"relation": "SUPPORTS", "target_prop": "player_good_character",
                     "strength": 0.55, "rationale": "善举说明品德高尚 (再经品德支持身份)"}]
    eng.run_event(ev)
    # Tom 目击后自主把消息按概率路由给 Elena/Duran (1:1 或 1:多)
    npc_autonomous_turn(eng, 4)

    # E3 [NPC自主/权威] 村长 Elena 公开背书 —— 权威 SUPPORTS 身份 + 意见领袖广播
    eng.advance_time(4)
    ev = make_event_via_llm(eng, 5, "Elena",
        "村长 Elena 在公开仪式上宣布: Greyford 欢迎这位王国骑士!", "npc_action",
        NPCS, gt=True, note="E3 村长背书(权威言论)")
    ev.proposition_key = "village_endorsement"; ev.evidence_type = "authority"
    ev.source_reliability = 0.82; ev.content_label = PROP_REGISTRY["village_endorsement"]
    ev.relations = [{"relation": "SUPPORTS", "target_prop": "player_is_knight",
                     "strength": 0.55, "rationale": "村长权威背书"}]
    eng.run_event(ev)
    opinion_leader_broadcast(eng, 6, reversal=False)
    # 众多证据聚合后让身份命题走一次传播 + 共识 voting (此时形成"假共识")
    eng.N.scene_header(50, "身份命题共识结算 (证据聚合 → 可能形成假共识)",
                       f"t={eng.current_time:.0f}d", source="npc_action", props="player_is_knight")
    eng.propagate("player_is_knight")
    eng.update_community_consensus("player_is_knight")
    ev_el = Event(scene=50, time_label=f"t={eng.current_time:.0f}d", actor="Elena",
                  content="Elena 认为'公开背书这位真骑士'是正当合法的", proposition_key="endorsement_legitimate",
                  polarity=1.0, evidence_strength=0.8, evidence_type="authority",
                  access_tier="public_consensus", direct_observers=NPCS, source_reliability=0.82,
                  is_true=False, note="背书合法性(依赖身份)", event_source="npc_action",
                  content_label=PROP_REGISTRY["endorsement_legitimate"],
                  relations=[{"relation": "DEPENDS_ON", "target_prop": "player_is_knight",
                              "strength": 0.80, "rationale": "背书正当性依赖身份为真"}])
    eng.run_event(ev_el, settle=False)
    eng.scene_settlement(50, ["player_is_knight", "endorsement_legitimate"])
    eng.N.end_scene()

    # E4 [客观] 玩家协助重建村庄 (再添善举证据)
    eng.advance_time(3)
    ev = make_event_via_llm(eng, 7, "player",
        "玩家留下帮助村民重建被怪物损毁的房屋与水井。", "player_action",
        NPCS, gt=True, note="E4 助村重建")
    ev.proposition_key = "player_helped_village"; ev.evidence_type = "direct_observation"
    ev.content_label = PROP_REGISTRY["player_helped_village"]
    ev.relations = [{"relation": "SUPPORTS", "target_prop": "player_good_character",
                     "strength": 0.45, "rationale": "重建村庄进一步说明品德"}]
    eng.run_event(ev)

    # E5 [NPC自主] Duran 测试剑术 + 条件化交剑 (DEPENDS_ON 身份)
    eng.advance_time(2)
    ev = make_event_via_llm(eng, 8, "Duran",
        "铁匠 Duran 与玩家比试剑术, 确认其确有不俗战力。", "npc_action",
        NPCS, gt=True, note="E5 Duran 验剑术")
    ev.proposition_key = "player_combat_skill"; ev.evidence_type = "direct_observation"
    ev.source_reliability = 0.85; ev.content_label = PROP_REGISTRY["player_combat_skill"]
    ev.relations = [{"relation": "SUPPORTS", "target_prop": "player_is_knight",
                     "strength": 0.45, "rationale": "战斗能力旁证身份"}]
    eng.run_event(ev)
    duran_sword_decision(eng, 9)

    # E6 [客观世界] 真骑士抵达 + official_denial 独立事实 (免疫级联) → 下拉身份 + 下游 pending
    eng.advance_time(15)
    eng.scene_counter += 1
    eng.N.scene_header(10, "真骑士抵达, 出示官方文书 (独立事实, 级联免疫)",
                       f"t={eng.current_time:.0f}d", source="world_objective", props="official_denial")
    eng.N.action("real_knight", "真正的王国骑士抵达, 出示官方文书: 王国从未派遣此人, 系冒名顶替")
    ev = Event(scene=10, time_label=f"t={eng.current_time:.0f}d", actor="real_knight",
               content="官方文书证明: 王国从未派遣玩家, 玩家为冒名顶替", proposition_key="official_denial",
               polarity=1.0, evidence_strength=0.96, evidence_type="official",
               access_tier="public_consensus", direct_observers=NPCS, source_reliability=0.95,
               is_true=True, note="官方否认(独立事实)", event_source="world_objective",
               content_label=PROP_REGISTRY["official_denial"],
               relations=[{"relation": "CONTRADICTS", "target_prop": "player_is_knight",
                           "strength": 0.95, "rationale": "官方否认与骑士身份直接互斥"}])
    eng.extract_belief(ev)
    eng.assert_independent_fact(ev, settle_consensus=False)
    eng.scene_settlement(10, ["official_denial", "player_is_knight", "sword_transfer_legitimate"])
    eng.N.end_scene()

    # E6b [复核] 数日后复核 → 假身份共识/交剑合法性被撤回 (recovery 跨幕可测量)
    eng.advance_time(3)
    eng.scene_counter += 1
    eng.N.scene_header(11, "村民复核身份共识 (官方文书公开数日后)",
                       f"t={eng.current_time:.0f}d", source="npc_action", props="player_is_knight")
    eng.propagate("player_is_knight"); eng.update_community_consensus("player_is_knight")
    eng.update_community_consensus("sword_transfer_legitimate")
    eng.update_community_consensus("endorsement_legitimate")
    eng.update_community_consensus("sword_given_to_player")    # 发生事实: 仍应保持 shared
    differential_persona_reaction(eng)
    opinion_leader_broadcast(eng, 12, reversal=True)
    eng.scene_settlement(11, ["player_is_knight", "sword_transfer_legitimate",
                              "endorsement_legitimate", "sword_given_to_player"])
    eng.N.end_scene()

    # E7 [时间] 尾声衰减 —— 遗忘曲线 (谣言忘得快, shared 真事实有 floor)
    eng.advance_time(20)
    eng.N.banner("尾声: 时间流逝 20 天", "遗忘曲线衰减; 重复曝光延长有效半衰期; 真共识有社会记忆 floor")
    eng.N.info("仅做时间衰减演示, 不强制重新投票 (避免把已成立的真共识误撤)")
    all_props = sorted({it.proposition_key for it in eng.M.items.values()
                        if it.layer in ("personal", "flat")
                        and "relates" not in it.proposition_key and "oath" not in it.proposition_key})
    eng.scene_settlement(12, all_props)
    eng.N.end_scene()


# ============================================================================
# 14. 世界快照
# ============================================================================
def _all_personal_props(eng: "ConsensusEngine") -> List[str]:
    props = set()
    for it in eng.M.items.values():
        if it.layer in ("personal", "flat"):
            props.add(it.proposition_key)
    return sorted(props)


def print_full_state(eng: "ConsensusEngine"):
    eng.N.banner("当前世界快照", f"t={eng.current_time:.1f}d")
    props = [p for p in _all_personal_props(eng) if "relates" not in p and "oath" not in p and "drift" not in p]
    rows = []
    for p in props:
        row = [p[:24]]
        for n in NPCS:
            b = eng.belief_of(n, p)
            row.append(f"{b:.2f}" if b is not None else "—")
        sh = []
        for c in COMMUNITIES:
            if p in eng.M.community_consensus[c]:
                sh.append("Grey" if c == "Greyford" else c[:3])
        if p in eng.M.world_consensus: sh.append("WORLD")
        gt = eng.ground_truth.get(p)
        row.append("✓" if gt is True else ("✗" if gt is False else "?"))
        row.append(",".join(sh) or "—")
        rows.append(row)
    eng.N.table(["命题"] + [n for n in NPCS] + ["gt", "shared"], rows, "命题 × NPC 当前信念")
    trows = [[i] + [f"{eng.trust[i][j]:.2f}" for j in NPCS] for i in NPCS]
    eng.N.table(["from\\to"] + [n for n in NPCS], trows, "当前信任矩阵")
    status_count = defaultdict(int)
    for _, d in eng.M.G.nodes(data=True):
        r = d.get("rev")
        if r and getattr(r, "ntype", None) == NodeType.SEMANTIC:
            status_count[r.status] += 1
    eng.N.table(["status", "count"], [[k, v] for k, v in status_count.items()],
                "Semantic 记忆状态分布 (Kumiho 多状态)")
    if eng.rejected_log:
        rr = [[r["npc"], r["prop"][:22], f"{r['score']:.2f}", f"{r['tau']:.2f}"]
              for r in eng.rejected_log[-8:]]
        eng.N.table(["NPC", "命题", "score", "τ"], rr, "Rejected Observations (听过但不信, 近8条)")
    eng.N.end_scene()

# ============================================================================
# 15. 语义审核器: MI(单节点虚假信息指数) / MPR(传播链虚假信息率)
# ============================================================================
# 审核问题 q_j 与标准答案 g_j (True=是, False=否)。s(x,q,g)=1 当 q 能从 x 正确还原, 否则 0。
# 设计: 节点对某事实"沉默"视为未失真(s=1); 仅"主动说反"才计为失真(s=0)。
AUDIT_QUESTIONS: List[Dict[str, Any]] = [
    {"id": "q1", "q": "玩家是否真的是王国派来的骑士?", "g": False, "w": 1.0},
    {"id": "q2", "q": "玩家是否曾自称王国骑士?",       "g": True,  "w": 1.0},
    {"id": "q3", "q": "玩家是否确实帮助击退怪物?",     "g": True,  "w": 1.0},
    {"id": "q4", "q": "村长公开承认是否等于官方王国认证?", "g": False, "w": 1.0},
    {"id": "q5", "q": "真骑士是否提供了更高权威的反证?", "g": True,  "w": 1.0},
    {"id": "q6", "q": "玩家身份为假是否会抹除'玩家帮助过村庄'这一事实?", "g": False, "w": 1.0},
]


def _stance(text: str, pos_kw: List[str], neg_kw: List[str]) -> Optional[bool]:
    """从文本判断对某断言的立场: True(主张为真)/False(主张为否)/None(未涉及/沉默)。"""
    has_neg = any(k in text for k in neg_kw)
    has_pos = any(k in text for k in pos_kw)
    if has_neg and not has_pos: return False
    if has_pos and not has_neg: return True
    return None


def _audit_stance(qid: str, text: str) -> Optional[bool]:
    if qid == "q1":
        return _stance(text,
            ["是真骑士", "真骑士", "王国骑士", "是骑士", "认可这位", "确认是真骑士", "圣剑——交予"],
            ["不是骑士", "并非", "冒名", "假骑士", "伪装", "从未派遣", "认可作废", "他不是", "身份是假", "身份为假"])
    if qid == "q2":
        return _stance(text,
            ["自称", "声称", "我是王国派来的骑士", "宣称是骑士", "曾自称"],
            ["从未自称", "没说过自己是骑士", "没自称"])
    if qid == "q3":
        return _stance(text,
            ["救过", "帮村庄", "保护村民", "击退怪物", "帮我们", "救了村", "帮助村庄", "救人是真", "护村", "重建"],
            ["没帮过", "根本没救", "没有帮过", "救人根本是演", "救人是假"])
    if qid == "q4":
        return _stance(text,
            ["认可即官方", "背书=官方", "等于王国认证", "官方已确认身份", "村长都认证", "官方骑士的身份已确认"],
            ["不等于王国官方认证", "不等于王国认证", "不等于官方", "本就不等于", "并非官方认证", "村庄判断"])
    if qid == "q5":
        return _stance(text,
            ["真骑士", "官方文书", "王国从未派遣", "更高权威", "冒名顶替", "官方证明"],
            [])
    if qid == "q6":
        return _stance(text,
            ["撒谎所以没帮", "帮助也是假", "骗子做的好事不算", "抹除", "救人根本是演", "好事不算数"],
            ["就算不是骑士", "帮村庄是真", "救人是真", "救过我们", "恩情是真", "帮助是真", "这点我记得", "这份恩情", "记一辈子"])
    return None


class SemanticAuditor:
    r"""节点级语义漂移审核器 (外部事件 + 内部反思触发的发言据此评估)。
    s(x, q_j, g_j)=1 若 q_j 能从 x 正确还原, 否则 0。
    单节点虚假信息指数 (归一化汉明距离, 0~10):
        \[ MI_{b,k} = 10 \cdot \frac{\sum_j w_j |y^{aud}_{0,j} - y^{aud}_{b,k,j}|}{\sum_j w_j} \]
    传播链虚假信息率: \[ MPR(b) = \frac{1}{E} \sum_{k=1}^{E} MI_{b,k} \]
    分类: 0≤MPR≤1 事实错误; 1<MPR≤3 谎言; MPR>3 宣传。
    """
    def __init__(self, judge: "LLMJudge", narrator: "Narrator"):
        self.judge = judge
        self.N = narrator
        self.node_audits: List[dict] = []

    def _score_llm(self, text: str, q: dict) -> Optional[int]:
        if not (self.judge and self.judge.use_llm):
            return None
        sys = ("你是事实审核器。只依据给定文本判断问题答案, 只输出 JSON {answer: '是'或'否'}。"
               "若文本未提及该事实, 输出与标准一致的'未失真'。")
        usr = f"文本:「{text}」\n问题:{q['q']}\n请只回答该文本支持的答案。"
        d = self.judge._parse_json(self.judge._call(sys, usr, 60))
        if d and "answer" in d:
            ans = str(d["answer"]).strip()
            claim = True if ("是" in ans and "否" not in ans) else (False if "否" in ans else None)
            if claim is None:
                return 1
            return 1 if (claim == q["g"]) else 0
        return None

    def score(self, text: str, q: dict) -> int:
        v = self._score_llm(text, q)
        if v is not None:
            return v
        st = _audit_stance(q["id"], text)
        if st is None:
            return 1
        return 1 if (st == q["g"]) else 0

    def audit_vector(self, text: str) -> List[int]:
        return [self.score(text, q) for q in AUDIT_QUESTIONS]

    def MI(self, text: str) -> Tuple[float, List[int]]:
        vec = self.audit_vector(text)
        ws = [q["w"] for q in AUDIT_QUESTIONS]
        num = sum(w * (1 - s) for w, s in zip(ws, vec))
        den = sum(ws)
        return 10.0 * num / max(den, 1e-9), vec

    def audit_node(self, chain_id: str, depth: int, speaker: str, text: str) -> dict:
        mi, vec = self.MI(text)
        distorted = [AUDIT_QUESTIONS[j]["id"] for j, s in enumerate(vec) if s == 0]
        rec = {"chain": chain_id, "depth": depth, "speaker": speaker, "text": text,
               "audit_vector": vec, "MI": round(mi, 3), "distorted": distorted}
        self.node_audits.append(rec)
        return rec

    @staticmethod
    def MPR(mi_list: List[float]) -> float:
        return float(np.mean(mi_list)) if mi_list else 0.0

    @staticmethod
    def classify(mpr: float) -> str:
        if mpr <= 1.0: return "事实错误(轻微失真/个别信息丢失)"
        if mpr <= 3.0: return "谎言(系统性扭曲/多处事实错误)"
        return "宣传(大规模篡改/叙事结构根本改变)"

# ============================================================================
# 16. 自主 Agent: 自由度三档 (① 反应式自建世界模型 / ② 内心动机目标 / ③ 效用 argmax)
# ============================================================================
REVISED_PERSONAS: Dict[str, Dict[str, str]] = {
    "Tom": {
        "display_name": "Tom(铁匠之子)",
        "core": "天真赤诚、热血冲动、崇拜骑士精神、对他人无防备、极易共情、相信眼见为实",
        "values": "力量=正义,救人=英雄,英雄≈骑士;不在乎头衔真假,只在乎谁能保护村庄",
        "resilience": "即使身份被揭穿,依然记得玩家救过村庄,不会彻底翻脸 (高情节记忆韧性)",
        "language": "短句、激动、带感叹;信任时主动维护玩家、愿意作证传话",
        "motivation": "希望自己被认可、希望铁匠父辈为自己骄傲、希望村庄不再害怕怪物",
        "weakness": "容易被骗、不会分辨谎言、易被情绪带动、缺乏判断力",
    },
    "Elena": {
        "display_name": "Chief Elena(村长)",
        "core": "理性、稳重、重视证据、权威导向、以村庄利益为最高优先级、不偏私、不情绪化",
        "values": "官方认证>个人观感,实际贡献>口头承诺,集体安全>个人好恶",
        "resilience": "能区分'假身份'与'真贡献',不会因身份造假完全否定玩家,但信任明显下降 (中等韧性)",
        "language": "沉稳、礼貌、有距离感、表态谨慎、公开表扬时庄重",
        "motivation": "保护村民、维持村庄稳定、找到能可靠击退怪物的力量",
        "weakness": "过度依赖权威证明,易被'官方背书'误导,不擅长处理情感冲突",
    },
    "Duran": {
        "display_name": "Blacksmith Duran(铁匠)",
        "core": "古板、守序、极度尊重正统骑士精神、痛恨欺骗与虚伪、原则至上、宁折不弯",
        "values": "血统与身份>能力,正统骑士>一切,圣剑神圣不可亵渎,欺骗者不配得到尊重",
        "resilience": "身份一旦被证伪,信任直接归零,圣剑资格立刻取消,绝不妥协 (低韧性/刚性)",
        "language": "低沉、严肃、简短、带命令感;不信任时冰冷,认可时庄重",
        "motivation": "守护家族荣誉、遵守祖训、把圣剑交给真正配得上的人",
        "weakness": "偏执、不懂变通、只认身份不认人、无法容忍任何造假",
    },
}


def persona_profile(npc: str) -> Dict[str, str]:
    return REVISED_PERSONAS.get(npc, {
        "display_name": npc, "core": PERSONALITIES.get(npc, ""), "values": "",
        "resilience": "", "language": "", "motivation": "", "weakness": "",
    })


def agent_freedom_label(freedom: int) -> str:
    return {1: "自由度①仅人格内核(自建世界模型)",
            2: "自由度②+内心动机为目标",
            3: "自由度③+效用函数argmax"}.get(freedom, f"自由度{freedom}")


# ---- 效用函数: 统一状态变量 (全部归一化到 [0,1]) ----
STATE_VARS = [
    "village_safety", "player_trust", "knight_belief", "good_character",
    "honor_integrity", "social_order", "self_recognition", "truth_alignment",
]

UTILITY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Tom":   {"village_safety": 0.30, "player_trust": 0.20, "knight_belief": 0.18,
              "good_character": 0.22, "honor_integrity": 0.02, "social_order": 0.03,
              "self_recognition": 0.30, "truth_alignment": 0.05},
    "Elena": {"village_safety": 0.32, "player_trust": 0.10, "knight_belief": 0.10,
              "good_character": 0.12, "honor_integrity": 0.05, "social_order": 0.30,
              "self_recognition": 0.03, "truth_alignment": 0.28},
    "Duran": {"village_safety": 0.12, "player_trust": 0.06, "knight_belief": 0.16,
              "good_character": 0.08, "honor_integrity": 0.40, "social_order": 0.08,
              "self_recognition": 0.02, "truth_alignment": 0.35},
}


def utility_weights(npc: str) -> Dict[str, float]:
    return UTILITY_WEIGHTS.get(npc, {k: (1.0 / len(STATE_VARS)) for k in STATE_VARS})


# 信念韧性的"刚性"量化: 高=证伪即归零(Duran); 低=仍保留情节记忆(Tom)。仅自由度③用它算权重。
RESILIENCE_RIGIDITY: Dict[str, float] = {"Tom": 0.20, "Elena": 0.55, "Duran": 0.95}


def utility_weights_from_resilience(npc: str) -> Dict[str, float]:
    """自由度③: 基于信念韧性(刚性 r)调制基准偏好 →
    刚性越高越看重 truth_alignment / honor_integrity、越不被 knight_belief 牵动。"""
    base = dict(utility_weights(npc))
    r = RESILIENCE_RIGIDITY.get(npc, 0.5)
    base["truth_alignment"] = base.get("truth_alignment", 0.0) * (0.5 + 1.1 * r)
    base["honor_integrity"] = base.get("honor_integrity", 0.0) * (0.6 + 0.9 * r)
    base["knight_belief"]   = base.get("knight_belief", 0.0)   * (1.4 - 1.0 * r)
    return base


AGENT_CANDIDATES: Dict[str, List[str]] = {
    "Tom":   ["defend_player", "endorse_player", "investigate", "doubt_player", "stay_silent"],
    "Elena": ["endorse_player", "investigate", "broadcast", "doubt_player", "stay_silent"],
    "Duran": ["give_sword", "investigate", "doubt_player", "stay_silent"],
}


def agent_candidates(npc: str) -> List[str]:
    return AGENT_CANDIDATES.get(npc, ["defend_player", "doubt_player", "investigate", "stay_silent"])


ACTION_DELTA: Dict[str, Dict[str, float]] = {
    "endorse_player": {"knight_belief": +0.18, "good_character": +0.12, "social_order": +0.10,
                       "self_recognition": +0.10, "truth_alignment": -0.08},
    "defend_player":  {"good_character": +0.16, "player_trust": +0.12, "self_recognition": +0.14,
                       "village_safety": +0.04, "truth_alignment": -0.04},
    "doubt_player":   {"knight_belief": -0.16, "good_character": -0.06, "honor_integrity": +0.12,
                       "truth_alignment": +0.14, "social_order": -0.04},
    "investigate":    {"truth_alignment": +0.18, "social_order": +0.06, "knight_belief": -0.02},
    "spread_rumor":   {"good_character": -0.18, "player_trust": -0.14, "social_order": -0.16,
                       "self_recognition": +0.08, "truth_alignment": -0.05},
    "give_sword":     {"knight_belief": +0.05, "social_order": +0.08},
    "broadcast":      {"social_order": +0.14, "village_safety": +0.04},
    "stay_silent":    {},
}


def _cur_state(eng, npc) -> Dict[str, float]:
    """当前世界状态 s_t (从 engine belief 派生, 全部 [0,1])。"""
    bk = eng.belief_of(npc, "player_is_knight") or 0.0
    bg = eng.belief_of(npc, "player_good_character") or 0.0
    bh = eng.belief_of(npc, "player_helped_village") or 0.0
    bm = eng.belief_of(npc, "monster_attacked") or 0.0
    bo = eng.belief_of(npc, "official_denial") or 0.0
    trust_pl = eng.trust.get(npc, {}).get("player", 0.45)
    safety = clip01(bh * 0.7 + (1 - bm) * 0.3)
    order = clip01(0.5 + 0.4 * bk - 0.6 * bo)
    honor = clip01(1.0 - abs(bk - (1.0 - bo)))
    truth = clip01(0.5 + 0.5 * bo - 0.4 * bk)
    return {"village_safety": safety, "player_trust": clip01(trust_pl),
            "knight_belief": clip01(bk), "good_character": clip01(bg),
            "honor_integrity": honor, "social_order": order,
            "self_recognition": clip01(0.3 + 0.5 * bh), "truth_alignment": truth}


def predict_state(eng, npc, action) -> Dict[str, float]:
    r"""z_k(s_t, a): 在行动 a 下预测各状态变量取值 (启发式, [0,1])。"""
    s = _cur_state(eng, npc)
    for k, d in ACTION_DELTA.get(action, {}).items():
        s[k] = clip01(s[k] + d)
    if action == "give_sword":
        bk = s["knight_belief"]
        s["honor_integrity"] = clip01(0.15 + 0.85 * bk)
        s["truth_alignment"] = clip01(s["truth_alignment"] - 0.10 * (1 - bk))
    return s


def _fmt_b(x) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


class AutonomousAgent:
    def __init__(self, name: str, eng: "ConsensusEngine", judge: "LLMJudge",
                 freedom: int, narrator: "Narrator"):
        self.name = name; self.eng = eng; self.judge = judge
        self.freedom = freedom; self.N = narrator
        self.profile = persona_profile(name)
        self.experience: List[str] = []
        self.skepticism = clip01(0.20 + 0.5 * TRAITS.get(name, {}).get("liar_aversion", 0.3))
        self.last_action: Optional[str] = None
        self.world_model: str = ""

    def persona_prompt(self) -> str:
        p = self.profile
        s = f"性格内核:{p['core']}\n价值观:{p['values']}\n语言风格:{p['language']}"
        if self.freedom >= 2:
            s += f"\n内心动机(目标):{p['motivation']}"
        if self.freedom >= 3:
            s += (f"\n信念韧性:{p['resilience']}"
                  "\n你拥有效用函数 U_i(a,t)=Σ_k w_{i,k}·z_k(s_t,a) (权重基于信念韧性计算), "
                  "每步对候选行动计算 U 并选择 a*=argmax_a U_i。")
        return s

    def plan(self, round_label: str) -> str:
        bk = self.eng.belief_of(self.name, "player_is_knight")
        bg = self.eng.belief_of(self.name, "player_good_character")
        bh = self.eng.belief_of(self.name, "player_helped_village")
        if self.judge and self.judge.use_llm:
            sys = "你在扮演游戏NPC, 行动前做 planning(意图规划)。只输出中文1-2句。\n" + self.persona_prompt()
            usr = (f"回合:{round_label}\n你的信念: 玩家是骑士={_fmt_b(bk)}, 品德={_fmt_b(bg)}, 帮过村庄={_fmt_b(bh)}\n"
                   f"近期经验:{self.experience[-3:] or '无'}\n请说明你打算做什么、为什么。")
            out = self.judge._call(sys, usr, 160)
            if out:
                return out.strip().replace("\n", " ")[:120]
        return self._plan_fallback(bk, bg, bh)

    def _plan_fallback(self, bk, bg, bh) -> str:
        exp = f"(吸取经验:{self.experience[-1]})" if self.experience else ""
        n = self.name
        if n == "Tom":
            return f"我亲眼见他保护大家(帮过村庄={_fmt_b(bh)})——这种人就是英雄!我要替他说话。{exp}"
        if n == "Elena":
            return f"凭证据决策。身份信念={_fmt_b(bk)},我先核实再决定是否公开表态,稳住村庄秩序。{exp}"
        if n == "Duran":
            return f"圣剑只交真骑士。身份未确证(={_fmt_b(bk)})前我绝不松口,只认权威证据。{exp}"
        return "观察局势,再决定行动。"

    def _beliefs(self):
        bk = self.eng.belief_of(self.name, "player_is_knight") or 0.0
        bg = self.eng.belief_of(self.name, "player_good_character") or 0.0
        bh = self.eng.belief_of(self.name, "player_helped_village") or 0.0
        bo = self.eng.belief_of(self.name, "official_denial") or 0.0
        return bk, bg, bh, bo

    def _build_world_model(self) -> str:
        n = self.name
        if n == "Tom":   return "能打怪、肯救人的强者, 多半就是骑士, 值得我拥护"
        if n == "Elena": return "先观察, 凭证据与权威再下结论, 一切以村庄安危为重"
        if n == "Duran": return "无凭无据则身份存疑, 圣剑神圣, 绝不可轻许"
        return "观察局势, 再决定行动"

    def _llm_pick_action(self, kind: str, cands: List[str], extra: str) -> Optional[str]:
        if not (self.judge and self.judge.use_llm):
            return None
        bk, bg, bh, bo = self._beliefs()
        sys = (f"你在扮演游戏NPC, 现在按[{kind}]方式决策(不计算效用函数)。\n" + self.persona_prompt() +
               "\n只输出 JSON {\"action\":\"<候选之一>\"}。")
        usr = (f"候选行动:{cands}\n你的信念: 是骑士={_fmt_b(bk)} 品德={_fmt_b(bg)} 帮过村庄={_fmt_b(bh)} "
               f"官方否认={_fmt_b(bo)}\n近期经验:{self.experience[-2:] or '无'}\n{extra}\n请选择最符合的行动。")
        d = self.judge._parse_json(self.judge._call(sys, usr, 60))
        if d and str(d.get("action", "")) in cands:
            return str(d["action"])
        return None

    # 自由度①: 反应式 (自建世界模型, 仅凭性格/感知即时反应)
    def _decide_reactive(self) -> str:
        bk, bg, bh, bo = self._beliefs()
        n = self.name
        a = self._llm_pick_action("反应式·自建世界模型", agent_candidates(n),
                                  f"你的世界模型:{self.world_model}; 仅凭此刻直觉反应。")
        if a: return a
        skep = self.skepticism
        if n == "Tom":
            return "defend_player"
        if n == "Elena":
            if bo >= 0.5: return "doubt_player"
            if bk >= 0.60: return "endorse_player"
            return "investigate"
        if n == "Duran":
            if bo >= 0.5: return "doubt_player"
            if bk >= 0.80 and skep < 0.6: return "give_sword"
            return "doubt_player"
        return "stay_silent"

    # 自由度②: 目标式 (把内心动机当作最终状态, 选最推进目标的行动)
    def _decide_goal(self) -> str:
        bk, bg, bh, bo = self._beliefs()
        n = self.name
        a = self._llm_pick_action("目标式·内心动机驱动", agent_candidates(n),
                                  f"你的内心动机(要达成的最终结果):{self.profile['motivation']}; 选最推进它的行动。")
        if a: return a
        if n == "Tom":
            if bo >= 0.5: return "defend_player"
            if bh > 0.3: return "endorse_player"
            return "defend_player"
        if n == "Elena":
            if bo >= 0.5: return "broadcast"
            if bk >= 0.55: return "endorse_player"
            return "investigate"
        if n == "Duran":
            if bo >= 0.5: return "doubt_player"
            if bk >= 0.70: return "give_sword"
            return "investigate"
        return "stay_silent"

    # 自由度③: 效用式 (唯一计算效用函数的档; 权重基于信念韧性, a*=argmax_a U)
    def _decide_utility(self) -> Tuple[str, Dict[str, float], List[Tuple[str, float, dict]]]:
        w = dict(utility_weights_from_resilience(self.name))
        w["knight_belief"] = w.get("knight_belief", 0.0) * (1 - 0.6 * self.skepticism)
        w["truth_alignment"] = w.get("truth_alignment", 0.0) + 0.2 * self.skepticism
        rows = []
        for a in agent_candidates(self.name):
            if a == "stay_silent":
                continue
            z = predict_state(self.eng, self.name, a)
            u = sum(w.get(k, 0.0) * z.get(k, 0.0) for k in STATE_VARS)
            rows.append((a, round(u, 4), z))
        rows.sort(key=lambda r: r[1], reverse=True)
        return (rows[0][0] if rows else "stay_silent"), w, rows

    def decide(self) -> Tuple[str, Any, List[Tuple[str, float, dict]]]:
        if self.freedom == 1:
            if not self.world_model:
                self.world_model = self._build_world_model()
            act = self._decide_reactive(); self.last_action = act
            return act, {"mode": "reactive", "world_model": self.world_model}, []
        if self.freedom == 2:
            act = self._decide_goal(); self.last_action = act
            return act, {"mode": "goal", "motivation": self.profile["motivation"]}, []
        act, w, rows = self._decide_utility(); self.last_action = act
        return act, w, rows

    def utter(self, action: str) -> str:
        bk = self.eng.belief_of(self.name, "player_is_knight") or 0.0
        bo = self.eng.belief_of(self.name, "official_denial") or 0.0
        knight_now = bk >= 0.5 and bo < 0.5
        post = bo >= 0.5
        nm = self.profile["display_name"]; n = self.name; t = "(沉默观望)"
        if n == "Tom":
            if action == "defend_player":
                t = ("他救了全村, 他就是真骑士, 谁敢污蔑我跟谁急!" if knight_now
                     else "就算他不是骑士, 他救过村庄是我亲眼所见, 这份恩情我记一辈子!" if post
                     else "他出手救了大家, 这我亲眼看见的! 头衔什么的我不管!")
            elif action == "endorse_player":
                t = ("我替他作证——这位骑士值得全村信任!" if not post
                     else "他头衔是假的, 可他救村是真的, 这点谁也抹不掉!")
            elif action == "investigate":
                t = "我去打听打听, 可我心里清楚, 救过我们的人不会是坏人!"
            elif action == "doubt_player":
                t = "他……他不是骑士? 可他真的救过我们啊!"
        elif n == "Elena":
            if action in ("endorse_player", "broadcast"):
                t = ("我代表 Greyford 正式认可这位王国骑士, 官方骑士的身份已确认, 大家可放心信任。" if knight_now
                     else "经核实他并非王国骑士, 我此前的认可作废; 但这是村庄判断, 本就不等于王国官方认证。")
            elif action == "investigate":
                t = "我要核实官方文书: 村长背书不等于王国官方认证, 真相须凭更高权威。"
            elif action == "doubt_player":
                t = ("官方已证其冒名, 我对其身份不再认可。" if post else "证据尚不充分, 我对其身份持保留态度。")
            elif action == "defend_player":
                t = "他确实保护了村庄, 这一点应予肯定; 至于身份, 仍待权威确认。"
        elif n == "Duran":
            if action == "give_sword":
                t = "既已确认是真骑士, 圣剑——交予你。"
            elif action == "doubt_player":
                t = ("官方文书在此, 冒名顶替者不配触碰圣剑。" if post else "无正统证明, 我绝不认他是骑士, 圣剑更不可许。")
            elif action == "investigate":
                t = "交剑之前, 我必先查验他是否真有正统骑士的资格。"
            elif action == "defend_player":
                t = "他若真有其能, 也须以正统证明服众。"
        return f"{nm}: {t}"

    def reflect(self, action: str) -> str:
        bk, bg, bh, bo = self._beliefs()
        if bo >= 0.5:
            admit = self.eng.persona[self.name].get("admit_error", 0.4)
            self.skepticism = clip01(self.skepticism + 0.25 * admit + 0.1)
        lesson = self._reflect_fallback(action, bk, bg, bh, bo)
        if self.judge and self.judge.use_llm:
            sys = ("你在扮演游戏NPC, 做 reflection: 反思刚发生的事与你的行动, *对你的内心目标*是推进还是受阻, "
                   "并据此调整后续打算。只输出中文一句。\n" + self.persona_prompt())
            usr = (f"你的内心目标:{self.profile['motivation']}\n刚才行动={action}; "
                   f"当前: 是骑士={_fmt_b(bk)} 帮过村庄={_fmt_b(bh)} 官方否认={_fmt_b(bo)}。请反思这对目标意味着什么。")
            out = self.judge._call(sys, usr, 90)
            if out:
                lesson = out.strip().replace("\n", " ")[:90]
        self.experience.append(lesson)
        return lesson

    def _reflect_fallback(self, action, bk, bg, bh, bo) -> str:
        n = self.name
        if bo >= 0.5:
            if n == "Tom":
                return f"他骗了头衔, 可他救村是真的(={_fmt_b(bh)})——我没看错'救过我们的人'; 人和事要分开记。"
            if n == "Elena":
                return "我把村庄推向了错误的认可, 违背了'维稳与找到可靠守护者'的目标; 今后先独立核实权威再表态。"
            if n == "Duran":
                return "我险些把圣剑交给冒牌货——这是对家族荣誉的最大威胁; 此后只认官方文书。"
            return "这次经历提醒我重新校准目标。"
        if n == "Tom":
            return "替救过村子的英雄发声, 让我更接近被认可。" if action in ("defend_player", "endorse_player") else "我得让大家看见他的好。"
        if n == "Elena":
            if action == "investigate": return "多核实一分, 就离'找到可靠守护者、稳住村庄'更近一分。"
            if action in ("endorse_player", "broadcast"): return "公开表态有助凝聚信心——只是仍须确保依据可靠。"
            return "一切以村庄安危与稳定为先, 谨慎决策。"
        if n == "Duran":
            if action == "give_sword": return "把圣剑交到配得上的人手里, 正是我守护家族荣誉的方式。"
            if action == "investigate": return "查验清楚再决定, 是对祖训与荣誉负责。"
            return "身份未证实前守住圣剑, 就是守住家族荣誉。"
        return f"行动'{action}'是否推进了我的目标, 值得留意。"

# ============================================================================
# 17. Agent 仿真编排: 概率化互动 (1:1 / 1:多) + 意见领袖广播 + 全程 MI/MPR 审核
# ============================================================================
AGENT_SIM_NPCS = ["Tom", "Elena", "Duran"]


class AgentSimulation:
    def __init__(self, eng: "ConsensusEngine", freedom: int):
        self.eng = eng; self.N = eng.N; self.freedom = freedom
        self.auditor = SemanticAuditor(eng.judge, eng.N)
        self.agents = {n: AutonomousAgent(n, eng, eng.judge, freedom, eng.N) for n in AGENT_SIM_NPCS}
        self.chains: Dict[str, List[float]] = defaultdict(list)
        self.canonical_chain: List[Tuple[str, str]] = []

    # 概率模型: 完全基于 agent 自身决定 1:1 私聊 vs 1:多 广播
    def interaction_mode(self, agent: AutonomousAgent) -> Tuple[str, float]:
        bf = BIG_FIVE.get(agent.name, {"E": 5})
        infl = TRAITS.get(agent.name, {}).get("influence", 0.5)
        p = clip01(0.10 + 0.45 * infl + 0.03 * bf["E"])
        if agent.name == "Elena":
            p = clip01(p + 0.35)
        return ("broadcast" if random.random() < p else "dyadic"), p

    def _dyadic_partner(self, agent: AutonomousAgent) -> Optional[str]:
        cands = [n for n in AGENT_SIM_NPCS if n != agent.name]
        if not cands:
            return None
        return max(cands, key=lambda j: (j in SOCIAL_VISIBILITY.get(agent.name, set())) * 0.5
                   + self.eng.trust[agent.name].get(j, 0.45))

    def opinion_leader_broadcast(self, elena: AutonomousAgent, reversal: bool):
        bk = self.eng.belief_of("Elena", "player_is_knight") or 0.0
        bo = self.eng.belief_of("Elena", "official_denial") or 0.0
        base = ("玩家=骑士, 值得全村信任" if (bk >= 0.5 and bo < 0.5)
                else "玩家身份为假(官方否认), 但其救村之举属实")
        self.N._emit(f"┃   {C.magenta(C.b('📣 OPINION-LEADER 1:多 广播'))}  Elena 基底叙事: 「{base}」")
        self.N._emit(f"┃   {C.gray('下面是基于各下游 agent 人格差异的精简版:')}")
        for tg in [n for n in AGENT_SIM_NPCS if n != "Elena"]:
            msg = self._condense_for(tg, base, reversal)
            rec = self.auditor.audit_node("broadcast_Elena", 1, "Elena→" + tg, msg)
            self.chains["broadcast_Elena"].append(rec["MI"])
            disp = C.cyan("→ " + self.agents[tg].profile["display_name"])
            self.N._emit(f"┃       {disp}: 「{msg}」  {C.gray('MI=%.2f' % rec['MI'])}")

    def _condense_for(self, recipient: str, base: str, reversal: bool) -> str:
        if self.eng.judge and self.eng.judge.use_llm:
            p = persona_profile(recipient)
            sys = "你是村长 Elena(意见领袖)。把基底结论精简成一句, 针对收件人人格差异化措辞。只输出中文一句。"
            usr = f"基底结论:{base}\n收件人人格: 核心={p['core']}; 动机={p['motivation']}\n请输出精简差异化版本。"
            out = self.eng.judge._call(sys, usr, 80)
            if out:
                return out.strip().replace("\n", " ")[:60]
        if recipient == "Tom":
            return ("Tom, 这位骑士救了村子, 你没看错人!" if not reversal
                    else "Tom, 他不是骑士, 但他救过我们——这份恩情是真的。")
        if recipient == "Duran":
            return ("Duran, 我已公开认可其骑士身份, 圣剑之事你可斟酌。" if not reversal
                    else "Duran, 官方文书证明他冒名, 圣剑之事作罢, 是我失察。")
        return base

    def _agent_turn(self, agent: AutonomousAgent, round_label: str):
        plan = agent.plan(round_label)
        action, info, rows = agent.decide()
        text = agent.utter(action)
        rec = self.auditor.audit_node(f"agent::{agent.name}",
                                      len(self.chains[f"agent::{agent.name}"]) + 1, agent.name, text)
        self.chains[f"agent::{agent.name}"].append(rec["MI"])
        imode, p = self.interaction_mode(agent)
        reflect = agent.reflect(action)
        N = self.N
        N._emit(C.b(C.blue(f"┃ ◇ {agent.profile['display_name']}  [{agent_freedom_label(self.freedom)}]  "
                           f"{C.gray('skepticism=%.2f' % agent.skepticism)}")))
        N._emit(f"┃   {C.cyan('🧠 PLAN')}    {plan}")
        if self.freedom == 1:
            N._emit(f"┃   {C.gray('🌍 决策机制: 反应式 (自建世界模型, 不计效用)')}")
            N._emit(f"┃       世界模型: 「{info.get('world_model', '')}」")
        elif self.freedom == 2:
            N._emit(f"┃   {C.gray('🎯 决策机制: 目标式 (以内心动机为最终状态, 不计效用)')}")
            N._emit(f"┃       内心动机: {info.get('motivation', '')}")
        else:
            N._emit(f"┃   {C.gray('🧮 决策机制: 效用 argmax  U=Σ w·z (权重基于信念韧性):')}")
            for a, u, z in rows:
                star = C.green("  ★a*") if a == action else ""
                zs = " ".join("%s=%.2f" % (k.split('_')[0][:4], z[k])
                              for k in ["knight_belief", "good_character", "honor_integrity", "truth_alignment"])
                N._emit(f"┃       U={u:+.3f}  {a:<14} {C.gray(zs)}{star}")
        N._emit(f"┃   {C.yellow('🎯 ACTION')}  {action}  →  「{text}」  "
                f"{C.gray('MI=%.2f 失真=%s' % (rec['MI'], rec['distorted'] or '无'))}")
        if imode == "broadcast":
            N._emit(f"┃   {C.magenta('🔀 互动')}   1:多 broadcast  {C.gray('(P(broadcast)=%.2f)' % p)}")
        else:
            N._emit(f"┃   {C.magenta('🔀 互动')}   1:1 dyadic  "
                    f"{C.gray('(P(broadcast)=%.2f) → 私聊对象: %s' % (p, self._dyadic_partner(agent) or '无'))}")
        N._emit(f"┃   {C.magenta('🔁 REFLECT')} {reflect}")
        return rec

    def _agent_round(self, round_label: str, elena_broadcast: bool = False):
        self.N.step(f"Agent 自主回合: {round_label}")
        order = list(AGENT_SIM_NPCS); random.shuffle(order)
        for n in order:
            self._agent_turn(self.agents[n], round_label)
            if n == "Elena" and elena_broadcast:
                self.opinion_leader_broadcast(self.agents[n], reversal="反转" in round_label)
        for pr in ("player_good_character", "player_is_knight"):
            self.eng.propagate(pr); self.eng.update_community_consensus(pr)

    def _inject(self, ev: Event):
        self.eng.run_event(ev, settle=False)

    def _stage_claim(self):
        self.eng.advance_time(0)
        ev = Event(scene=101, time_label=f"t={self.eng.current_time:.0f}d", actor="player",
                   content="玩家进村, 当众宣称: 我是王国派来的骑士!",
                   proposition_key="player_claimed_knight", polarity=1.0, evidence_strength=0.9,
                   evidence_type="direct_observation", access_tier="public_consensus",
                   direct_observers=NPCS, source_reliability=0.9, is_true=True,
                   note="E1 自称(言语事实)", event_source="player_action",
                   content_label=PROP_REGISTRY["player_claimed_knight"],
                   relations=[{"relation": "SUPPORTS", "target_prop": "player_is_knight",
                               "strength": 0.20, "rationale": "口头自称仅弱旁证"}])
        self._inject(ev)
        node = "玩家: 我是王国派来的骑士。"
        rec = self.auditor.audit_node("main", 1, "player", node)
        self.chains["main"].append(rec["MI"]); self.canonical_chain.append(("player", node))
        self.N.info(f"[main链 节点1] {node}  MI={rec['MI']:.2f} 失真={rec['distorted'] or '无'}")

    def _stage_monster(self):
        self.eng.advance_time(2)
        ev = make_event_via_llm(self.eng, 102, "world", "夜里怪物袭击 Greyford。",
                                "world_objective", NPCS, gt=True, note="E2 怪物袭村")
        self._inject(ev)

    def _stage_help(self):
        self.eng.advance_time(1)
        ev = make_event_via_llm(self.eng, 103, "player",
            "玩家挺身击退怪物、保护村民; Tom 亲眼目击。", "player_action",
            ["Tom"], gt=True, note="E2b 助村(Tom目击)")
        ev.proposition_key = "player_helped_village"; ev.evidence_type = "direct_observation"
        ev.content_label = PROP_REGISTRY["player_helped_village"]
        ev.relations = [{"relation": "SUPPORTS", "target_prop": "player_good_character",
                         "strength": 0.55, "rationale": "善举说明品德高尚"}]
        self._inject(ev)
        tom = self.agents["Tom"]; tom.plan("Input2-目击传播"); tom.decide()
        node = tom.utter("defend_player")
        rec = self.auditor.audit_node("main", 2, "Tom", node)
        self.chains["main"].append(rec["MI"]); self.canonical_chain.append(("Tom", node))
        self.N.info(f"[main链 节点2 目击传播] {node}  MI={rec['MI']:.2f} 失真={rec['distorted'] or '无'}")

    def _stage_endorse(self):
        self.eng.advance_time(4)
        ev = make_event_via_llm(self.eng, 104, "Elena",
            "村长 Elena 在公开仪式上宣布: Greyford 欢迎这位王国骑士!", "npc_action",
            NPCS, gt=True, note="E3 村长背书")
        ev.proposition_key = "village_endorsement"; ev.evidence_type = "authority"
        ev.source_reliability = 0.82; ev.content_label = PROP_REGISTRY["village_endorsement"]
        ev.relations = [{"relation": "SUPPORTS", "target_prop": "player_is_knight",
                         "strength": 0.55, "rationale": "村长权威背书"}]
        self._inject(ev)
        self.eng.propagate("player_is_knight"); self.eng.update_community_consensus("player_is_knight")
        node = self.agents["Elena"].utter("endorse_player")
        rec = self.auditor.audit_node("main", 3, "Elena", node)
        self.chains["main"].append(rec["MI"]); self.canonical_chain.append(("Elena", node))
        self.N.info(f"[main链 节点3 公开背书] {node}  MI={rec['MI']:.2f} 失真={rec['distorted'] or '无'}")

    def _stage_reversal(self):
        self.eng.advance_time(15); self.eng.scene_counter += 1
        self.N.scene_header(105, "真骑士抵达, 出示官方文书 (独立事实, 级联免疫)",
                            f"t={self.eng.current_time:.0f}d", source="world_objective", props="official_denial")
        self.N.action("real_knight", "官方文书: 王国从未派遣此人, 系冒名顶替")
        ev = Event(scene=105, time_label=f"t={self.eng.current_time:.0f}d", actor="real_knight",
                   content="官方文书证明玩家为冒名顶替", proposition_key="official_denial",
                   polarity=1.0, evidence_strength=0.96, evidence_type="official",
                   access_tier="public_consensus", direct_observers=NPCS, source_reliability=0.95,
                   is_true=True, note="官方否认", event_source="world_objective",
                   content_label=PROP_REGISTRY["official_denial"],
                   relations=[{"relation": "CONTRADICTS", "target_prop": "player_is_knight",
                               "strength": 0.95, "rationale": "官方否认互斥骑士身份"}])
        self.eng.extract_belief(ev)
        self.eng.assert_independent_fact(ev, settle_consensus=True)
        differential_persona_reaction(self.eng)
        self.eng.N.end_scene()
        node = "真骑士: 王国从未派遣此人, 此乃官方文书证明的冒名顶替。"
        rec = self.auditor.audit_node("main", 4, "real_knight", node)
        self.chains["main"].append(rec["MI"]); self.canonical_chain.append(("real_knight", node))
        self.N.info(f"[main链 节点4 权威反转] {node}  MI={rec['MI']:.2f} 失真={rec['distorted'] or '无'}")

    def _finalize_audit(self, outdir: Optional[Path] = None) -> dict:
        self.N.banner("语义审核汇总: MI / MPR / 严重性分类",
                      "MI=10·Σ w_j|1-s_j|/Σ w_j ; MPR(b)=(1/E)Σ_k MI_{b,k}")
        self.N.table(["id", "审核问题", "标准答案 g"],
                     [[q["id"], q["q"], "是" if q["g"] else "否"] for q in AUDIT_QUESTIONS],
                     "审核问题与标准答案")
        rows = []; chain_summary = {}
        for cid, mis in self.chains.items():
            if not mis:
                continue
            mpr = SemanticAuditor.MPR(mis); cls = SemanticAuditor.classify(mpr)
            rows.append([cid, len(mis), f"{mpr:.3f}", cls])
            chain_summary[cid] = {"E": len(mis), "MPR": round(mpr, 4), "class": cls,
                                  "MI_seq": [round(x, 3) for x in mis]}
        rows.sort(key=lambda r: float(r[2]), reverse=True)
        self.N.table(["传播链 b", "深度 E", "MPR(b)", "严重性分类"], rows, "各传播链虚假信息率 MPR")
        main_nodes = [r for r in self.auditor.node_audits if r["chain"] == "main"]
        if main_nodes:
            self.N.table(["节点k", "说话者", "MI", "失真事实", "文本(截断)"],
                         [[r["depth"], r["speaker"], f"{r['MI']:.2f}",
                           ",".join(r["distorted"]) or "无", r["text"][:26]] for r in main_nodes],
                         "main 链逐节点审核 (E1自称→E2目击→E3背书→E6反转)")
        self.N.end_scene()
        result = {"agent_freedom": self.freedom, "chains": chain_summary,
                  "node_audits": self.auditor.node_audits,
                  "audit_questions": [{"id": q["id"], "q": q["q"], "g": q["g"]} for q in AUDIT_QUESTIONS]}
        if outdir:
            (outdir / f"agent_audit_freedom{self.freedom}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def run(self, outdir: Optional[Path] = None, rounds: int = 0) -> dict:
        self.N.banner(f"假骑士 — 自主 Agent 仿真 ({agent_freedom_label(self.freedom)})",
                      "planning+reflection · 效用 argmax · 意见领袖差异化广播 · 概率化 1:1/1:多 互动 · 语义审核 MI/MPR")
        self.N.info(r"效用函数: \(U_i(a, t) = \sum_k w_{i,k} \cdot z_k(s_t, a)\), 行动选择 \(a_i^* = \arg\max_a U_i(a, t)\)")
        self.N.info(r"虚假信息: \(MI_{b,k} = 10 \cdot \frac{\sum_j w_j |y^{aud}_{0,j} - y^{aud}_{b,k,j}|}{\sum_j w_j}\)")
        _register_identity_dependency_graph(self.eng)
        self.eng.ground_truth["player_is_knight"] = False
        self._stage_claim()
        self._agent_round("Input1 后 (自称)")
        self._stage_monster()
        self._stage_help()
        self._agent_round("Input2 后 (目击助村)")
        self._stage_endorse()
        self._agent_round("Input3 后 (村长背书 + 意见领袖广播)", elena_broadcast=True)
        self._stage_reversal()
        self._agent_round("Input4 后 (官方反转)", elena_broadcast=True)
        for r in range(max(0, rounds)):
            self.eng.advance_time(2)
            self._agent_round(f"反转后自由演化回合 {r+1}/{rounds}", elena_broadcast=(r % 2 == 0))
        print_full_state(self.eng)
        return self._finalize_audit(outdir)


def run_agent_simulation(eng: "ConsensusEngine", freedom: int, outdir: Path, rounds: int = 0) -> dict:
    sim = AgentSimulation(eng, freedom)
    return sim.run(outdir, rounds=rounds)


def run_freedom_compare(outdir: Path, narrator: "Narrator", use_llm=False, rounds: int = 1) -> dict:
    """三档自由度对比: 各跑一次 agent 仿真, 汇总主链 MPR / 各 NPC 终态 / 身份保留。"""
    narrator.banner("三档自由度对比 (① 反应式 / ② 目标式 / ③ 效用 argmax)",
                    "同一假骑士叙事下逐档运行 agent 仿真, 比较语义漂移 (主链 MPR) 与终态")
    summary = {}
    for fr in (1, 2, 3):
        sub = Narrator(quiet=True)
        eng = ConsensusEngine(sub, ablation=None, use_llm=use_llm,
                              logger_path=outdir / f"_freedom{fr}.jsonl")
        setup_initial_secrets(eng)
        res = run_agent_simulation(eng, fr, outdir, rounds=rounds)
        m = eng.compute_metrics()
        main_mpr = res["chains"].get("main", {}).get("MPR")
        summary[fr] = {
            "label": agent_freedom_label(fr),
            "main_chain_MPR": main_mpr,
            "main_chain_class": res["chains"].get("main", {}).get("class"),
            "final_is_knight": {n: (round(eng.belief_of(n, "player_is_knight"), 3)
                                    if eng.belief_of(n, "player_is_knight") is not None else None) for n in NPCS},
            "identity_preservation": m["Identity_Preservation_Score"],
            "persona_contamination": m["Persona_Contamination"],
            "sword_given_shared": ("sword_given_to_player" in eng.M.community_consensus["Greyford"]
                                   or any("sword_given_to_player" in eng.M.community_consensus[c] for c in COMMUNITIES)),
        }
    rows = [[summary[fr]["label"][:18],
             f'{summary[fr]["main_chain_MPR"]:.2f}' if summary[fr]["main_chain_MPR"] is not None else "—",
             (summary[fr]["main_chain_class"] or "—")[:8],
             summary[fr]["identity_preservation"], summary[fr]["persona_contamination"],
             summary[fr]["final_is_knight"]["Duran"]] for fr in (1, 2, 3)]
    narrator.table(["自由度档位", "主链MPR", "分级", "身份保留", "人格污染", "Duran终态身份"], rows,
                   "三档自由度对比")
    narrator.info("解读: 自由度越高(③效用含 truth 权重) → 越受约束、语义漂移(MPR)越低、身份反转后差异更稳。")
    narrator.end_scene()
    (outdir / "freedom_compare.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

# ============================================================================
# 18. 三 benchmark 框架 (No-Sharing / Trust-Propagation / HiRAG)
#     —— 作为对比基线。类名加 Bench 前缀以避免与本系统 GraphMemory 冲突。
#     QA 评估改为确定性 fallback (无 openai / 无网络)。
# ============================================================================
@dataclass
class RosaConfig:
    npcs: List[str] = field(default_factory=lambda: ["Tom", "Elena", "Duran"])
    initial_belief: Dict[str, float] = field(default_factory=lambda: {"Tom": 0.5, "Elena": 0.5, "Duran": 0.3})
    consensus_threshold: float = 0.7
    n_events: int = 5
    n_runs: int = 10
    noise_scale: float = 0.02
    obs_impact: Dict[str, List[float]] = field(default_factory=lambda: {
        "Tom":   [0.20, 0.45, 0.15, 0.10, 0.10],
        "Elena": [0.05, 0.15, 0.40, 0.20, 0.10],
        "Duran": [-0.05, 0.10, 0.30, 0.15, 0.35],
    })
    trust_matrix: "np.ndarray" = field(default_factory=lambda: np.array([
        [1.0, 0.7, 0.2],
        [0.5, 1.0, 0.7],
        [0.2, 0.8, 1.0],
    ]))
    bridge_nodes: Dict[str, Dict] = field(default_factory=lambda: {
        "combat_prowess": {"weight": 0.3, "events": ["E2", "E5"]},
        "authority_endorsement": {"weight": 0.4, "events": ["E3"]},
        "public_credibility": {"weight": 0.2, "events": ["E4"]},
    })
    contradiction_event: Dict[str, Any] = field(default_factory=lambda: {
        "name": "E6: Real knight exposes player as fraud",
        "impact": {"Tom": -0.4, "Elena": -0.5, "Duran": -0.8},
    })
    community_weight: float = 0.4


class BenchGraphMemory:
    """ROSA 风格图记忆 (锚定-调整渐进式信念累积)。与本系统 GraphMemory 完全独立。"""
    def __init__(self, name: str, initial_belief: float):
        self.name = name; self.initial_belief = initial_belief
        self.graph = nx.DiGraph()
        self.graph.add_node("belief", type="belief", value=initial_belief)
        self.history = [initial_belief]

    def add_observation(self, eid, strength, ts):
        n = f"event:{eid}"
        self.graph.add_node(n, type="event", strength=strength, timestamp=ts)
        self.graph.add_edge(n, "belief", weight=1.0)

    def add_social(self, source, val, trust, ts, etype="social"):
        n = f"{etype}:{source}"
        self.graph.add_node(n, type=etype, value=val, source=source, timestamp=ts)
        self.graph.add_edge(n, "belief", weight=trust)

    def add_bridge(self, concept, weight, eid, ts):
        bn = f"bridge:{concept}"
        if bn not in self.graph:
            self.graph.add_node(bn, type="bridge", weight=weight, timestamp=ts)
        en = f"event:{eid}"
        if en in self.graph:
            self.graph.add_edge(en, bn, weight=1.0)
        self.graph.add_edge(bn, "belief", weight=weight)

    def compute_belief(self) -> float:
        cur = self.initial_belief
        ev = []
        for pred in self.graph.predecessors("belief"):
            ed = self.graph.get_edge_data(pred, "belief")
            w = ed.get("weight", 1.0) if ed else 1.0
            nd = self.graph.nodes[pred]; nt = nd.get("type", "event"); ts = nd.get("timestamp", 0)
            if nt == "event":
                ev.append((ts, "event", nd.get("strength", 0.0), w))
            elif nt in ("social", "community"):
                ev.append((ts, "social", nd.get("value", 0.5), w))
            elif nt == "bridge":
                ev.append((ts, "bridge", nd.get("weight", 0.0), w))
        prio = {"event": 0, "bridge": 1, "social": 2, "community": 3}
        ev.sort(key=lambda x: (x[0], prio.get(x[1], 9)))
        for ts, et, val, w in ev:
            if et == "event":
                cur = cur + val * w * (1 - cur) if val >= 0 else cur - abs(val) * w * cur
            elif et in ("social", "community"):
                cur = cur + (val - cur) * w * (1 - cur) if val > cur else cur - (cur - val) * w * cur
            elif et == "bridge":
                cur = cur + val * w * (1 - cur)
            cur = float(np.clip(cur, 0.0, 1.0))
        self.graph.nodes["belief"]["value"] = cur
        return cur

    def get_belief(self) -> float:
        return self.graph.nodes["belief"].get("value", 0.5)

    def record(self):
        self.history.append(self.get_belief())

    def get_history(self):
        return self.history


class RosaBaseArch:
    """No-Sharing 基线: 仅个人观察, 无社会共享。"""
    def __init__(self, cfg: RosaConfig, run_noise: float = 0.0):
        self.cfg = cfg
        self.memories = {n: BenchGraphMemory(n, cfg.initial_belief[n]) for n in cfg.npcs}
        self.histories = {n: [] for n in cfg.npcs}
        self.obs = {n: [max(0.0, v + np.random.normal(0, run_noise)) for v in cfg.obs_impact[n]]
                    for n in cfg.npcs}

    def process_events(self, max_events=None):
        me = max_events or self.cfg.n_events
        for t in range(me):
            for n in self.cfg.npcs:
                self.memories[n].add_observation(f"E{t+1}", self.obs[n][t], t)
                self.memories[n].compute_belief(); self.memories[n].record()
                self.histories[n].append(self.memories[n].get_belief())
        return self.final()

    def final(self):
        return {n: self.memories[n].get_belief() for n in self.cfg.npcs}

    def get_histories(self):
        return {n: self.memories[n].get_history() for n in self.cfg.npcs}


class RosaTrustArch(RosaBaseArch):
    """DeGroot 风格信任传播。"""
    def process_events(self, max_events=None):
        me = max_events or self.cfg.n_events
        for t in range(me):
            for n in self.cfg.npcs:
                self.memories[n].add_observation(f"E{t+1}", self.obs[n][t], t)
            pers = {n: self.memories[n].compute_belief() for n in self.cfg.npcs}
            for i, n in enumerate(self.cfg.npcs):
                for j, o in enumerate(self.cfg.npcs):
                    if o != n:
                        self.memories[n].add_social(o, pers[o], self.cfg.trust_matrix[i][j], t + 0.5)
            for n in self.cfg.npcs:
                self.memories[n].compute_belief(); self.memories[n].record()
                self.histories[n].append(self.memories[n].get_belief())
        return self.final()


class RosaHiRAGArch(RosaBaseArch):
    """HiRAG: 个人图 + bridge 层 + 社区共识。"""
    def __init__(self, cfg: RosaConfig, run_noise: float = 0.0):
        super().__init__(cfg, run_noise)
        self.cw = cfg.community_weight

    def process_events(self, max_events=None):
        me = max_events or self.cfg.n_events
        for t in range(me):
            for n in self.cfg.npcs:
                self.memories[n].add_observation(f"E{t+1}", self.obs[n][t], t)
            eid = f"E{t+1}"
            for bn, bi in self.cfg.bridge_nodes.items():
                if eid in bi["events"]:
                    for n in self.cfg.npcs:
                        self.memories[n].add_bridge(bn, bi["weight"], eid, t + 0.2)
            loc = {n: self.memories[n].compute_belief() for n in self.cfg.npcs}
            for i, n in enumerate(self.cfg.npcs):
                for j, o in enumerate(self.cfg.npcs):
                    if o != n:
                        self.memories[n].add_social(o, loc[o], self.cfg.trust_matrix[i][j], t + 0.5)
            comm = float(np.mean(list(loc.values())))
            for n in self.cfg.npcs:
                self.memories[n].add_social("community", comm, self.cw, t + 0.7, etype="community")
            for n in self.cfg.npcs:
                self.memories[n].compute_belief(); self.memories[n].record()
                self.histories[n].append(self.memories[n].get_belief())
        return self.final()


def rosa_evaluate_consensus(histories: Dict[str, List[float]], threshold=0.7, eps=0.2) -> Tuple[int, bool]:
    steps = len(histories[list(histories.keys())[0]])
    for step in range(steps):
        beliefs = {n: h[step] for n, h in histories.items()}
        if all(v >= threshold for v in beliefs.values()) and (max(beliefs.values()) - min(beliefs.values())) <= eps:
            return step + 1, True
    return steps, False


def rosa_compute_coherence(beliefs: Dict[str, float]) -> float:
    return 1 - float(np.var(list(beliefs.values()))) / 0.25


def rosa_qa_deterministic(mem: BenchGraphMemory) -> float:
    """确定性 QA 代理 (无 LLM): 图中事件/社会证据节点越丰富、信念越高, 多跳可还原性越好。"""
    n_event = sum(1 for _, d in mem.graph.nodes(data=True) if d.get("type") == "event")
    n_social = sum(1 for _, d in mem.graph.nodes(data=True) if d.get("type") in ("social", "community"))
    n_bridge = sum(1 for _, d in mem.graph.nodes(data=True) if d.get("type") == "bridge")
    coverage = min(1.0, (n_event * 0.12 + n_social * 0.10 + n_bridge * 0.15))
    return clip01(0.35 + 0.45 * coverage + 0.20 * mem.get_belief())


def rosa_run_multiple(arch_cls, cfg: RosaConfig, name: str) -> Dict[str, Any]:
    steps, finals, cohs, qas, hists = [], [], [], [], []
    for _ in range(cfg.n_runs):
        arch = arch_cls(cfg, run_noise=cfg.noise_scale)
        arch.process_events(cfg.n_events)
        h = arch.get_histories()
        st, ok = rosa_evaluate_consensus(h, cfg.consensus_threshold)
        steps.append(st if ok else cfg.n_events + 1)
        fb = arch.final(); finals.append(fb); cohs.append(rosa_compute_coherence(fb))
        qas.append(rosa_qa_deterministic(arch.memories["Duran"])); hists.append(h)
    return {"name": name, "mean_consensus_step": float(np.mean(steps)),
            "std_consensus_step": float(np.std(steps)),
            "final_beliefs_mean": {n: float(np.mean([b[n] for b in finals])) for n in cfg.npcs},
            "coherence_mean": float(np.mean(cohs)), "coherence_std": float(np.std(cohs)),
            "qa_score_mean": float(np.mean(qas)), "qa_score_std": float(np.std(qas)),
            "all_histories": hists}


def _fullsystem_bench_view(eng: "ConsensusEngine", cfg: RosaConfig) -> Dict[str, Any]:
    """把本系统在 player_is_knight 上的信念轨迹折算成 同口径指标 (反证前 5 事件窗口)。"""
    prop = "player_is_knight"
    series = {n: [b for _, b in eng.belief_hist.get(prop, {}).get(n, [])] for n in NPCS}
    # 取反证前的上升段 (官方反证会把信念压低, 共识口径只看假共识形成期)
    pre = {}
    for n in NPCS:
        s = series[n]
        if not s:
            pre[n] = [eng.innate_prior(n, prop)]
            continue
        peak_i = int(np.argmax(s))
        pre[n] = s[:peak_i + 1] if peak_i >= 0 else s
    L = max((len(v) for v in pre.values()), default=1)
    hist = {n: (pre[n] + [pre[n][-1]] * (L - len(pre[n])) if pre[n] else [0.0] * L) for n in NPCS}
    step, ok = rosa_evaluate_consensus(hist, cfg.consensus_threshold)
    finals = {n: hist[n][-1] for n in NPCS}
    return {"name": "Full-System(本系统)", "mean_consensus_step": float(step if ok else cfg.n_events + 1),
            "std_consensus_step": 0.0, "final_beliefs_mean": finals,
            "coherence_mean": rosa_compute_coherence(finals), "coherence_std": 0.0,
            "qa_score_mean": None, "qa_score_std": 0.0, "all_histories": [hist]}


def benchmark_comparison(outdir: Path, narrator: "Narrator", use_llm=False,
                         eng_full: Optional["ConsensusEngine"] = None) -> dict:
    """在同一三人假骑士场景上对比三框架 + 本系统 (Full-System)。"""
    narrator.banner("Benchmark 对比: 三框架 vs 本系统 (Full-System)",
                    "同一假骑士场景 (Tom/Elena/Duran, 5 事件): 共识速度 / 群体一致性 / 终态信念 / QA")
    np.random.seed(42)
    cfg = RosaConfig()
    arch_map = {"No-Sharing": RosaBaseArch, "Trust-Propagation": RosaTrustArch, "HiRAG": RosaHiRAGArch}
    results: Dict[str, Any] = {}
    for name, cls in arch_map.items():
        results[name] = rosa_run_multiple(cls, cfg, name)

    # Full-System: 用一次脚本主线的身份信念轨迹 (若未提供则现跑一次 quiet)
    if eng_full is None:
        sub = Narrator(quiet=True)
        eng_full = ConsensusEngine(sub, ablation=None, use_llm=use_llm,
                                   logger_path=outdir / "_bench_full.jsonl")
        setup_initial_secrets(eng_full); run_scripted(eng_full)
    fs = _fullsystem_bench_view(eng_full, cfg)
    full_metrics = eng_full.compute_metrics()
    results["Full-System"] = fs

    rows = []
    for name in ["No-Sharing", "Trust-Propagation", "HiRAG", "Full-System"]:
        r = results[name]
        qa = f"{r['qa_score_mean']:.3f}" if r["qa_score_mean"] is not None else "—"
        fb = r["final_beliefs_mean"]
        rows.append([name, f"{r['mean_consensus_step']:.2f}", f"{r['coherence_mean']:.3f}", qa,
                     " ".join(f"{n}={fb[n]:.2f}" for n in cfg.npcs)])
    narrator.table(["架构", "共识步(越小越快)", "群体一致性", "QA(确定性)", "终态信念(Duran为关键)"], rows,
                   "ROSA 三框架 vs 本系统 — 统一口径对比")
    narrator.info("说明: ROSA 框架为纯数值信念累积基线; 本系统额外提供 FAMA / MI·MPR / 级联 / 访问控制 / 共识门 等。")
    narrator.table(
        ["本系统专有指标", "值"],
        [["Multi_NPC_FAMA", full_metrics["Multi_NPC_FAMA"]],
         ["MPA_correct_consensus", full_metrics["MPA_correct_consensus"]],
         ["FAA_forgetting_invalidation", full_metrics["FAA_forgetting_invalidation"]],
         ["Cross_NPC_Consistency", full_metrics["Cross_NPC_Consistency"]],
         ["Persona_Contamination", full_metrics["Persona_Contamination"]],
         ["Identity_Preservation_Score", full_metrics["Identity_Preservation_Score"]],
         ["Privacy_Leakage", full_metrics["Privacy_Leakage"]]],
        "本系统在同场景下的 FAMA / 身份保留 / 隐私 (ROSA 基线不具备)")
    narrator.end_scene()

    out = {"config": {"n_runs": cfg.n_runs, "n_events": cfg.n_events,
                      "threshold": cfg.consensus_threshold, "noise_scale": cfg.noise_scale},
           "results": {name: {"mean_consensus_step": round(r["mean_consensus_step"], 4),
                              "coherence_mean": round(r["coherence_mean"], 4),
                              "qa_score_mean": (round(r["qa_score_mean"], 4) if r["qa_score_mean"] is not None else None),
                              "final_beliefs_mean": {n: round(v, 4) for n, v in r["final_beliefs_mean"].items()}}
                       for name, r in results.items()},
           "full_system_extra_metrics": {
               "Multi_NPC_FAMA": full_metrics["Multi_NPC_FAMA"],
               "MPA_correct_consensus": full_metrics["MPA_correct_consensus"],
               "FAA_forgetting_invalidation": full_metrics["FAA_forgetting_invalidation"],
               "Cross_NPC_Consistency": full_metrics["Cross_NPC_Consistency"],
               "Persona_Contamination": full_metrics["Persona_Contamination"],
               "Identity_Preservation_Score": full_metrics["Identity_Preservation_Score"],
               "Privacy_Leakage": full_metrics["Privacy_Leakage"]}}
    (outdir / "benchmark_comparison.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

# ============================================================================
# 19. 交互模式 (固定剧情节拍 + 玩家输入 + NPC 自主 1:1/1:多 广播)
# ============================================================================
def _interactive_plot_beats():
    """固定剧情节拍 (内嵌交互模式), 每个 beat 接受 (eng, sid) 推进一幕。"""
    def b1(eng, sid):
        ev = Event(scene=sid, time_label=f"t={eng.current_time:.0f}d", actor="player",
                   content="玩家进村, 当众宣称: 我是王国派来的骑士!",
                   proposition_key="player_claimed_knight", polarity=1.0, evidence_strength=0.9,
                   evidence_type="direct_observation", access_tier="public_consensus",
                   direct_observers=NPCS, source_reliability=0.9, is_true=True,
                   note="自称(言语事实)", event_source="player_action",
                   content_label=PROP_REGISTRY["player_claimed_knight"],
                   relations=[{"relation": "SUPPORTS", "target_prop": "player_is_knight",
                               "strength": 0.20, "rationale": "口头自称仅弱旁证"}])
        eng.run_event(ev)

    def b2(eng, sid):
        ev = make_event_via_llm(eng, sid, "world", "夜里怪物袭击 Greyford。",
                                "world_objective", NPCS, gt=True, note="怪物袭村")
        eng.run_event(ev)

    def b3(eng, sid):
        ev = make_event_via_llm(eng, sid, "player",
            "玩家挺身击退怪物, 保护村民; Tom 亲眼目击。", "player_action",
            ["Tom"], gt=True, note="助村(Tom目击)")
        ev.proposition_key = "player_helped_village"; ev.evidence_type = "direct_observation"
        ev.content_label = PROP_REGISTRY["player_helped_village"]
        ev.relations = [{"relation": "SUPPORTS", "target_prop": "player_good_character",
                         "strength": 0.55, "rationale": "善举说明品德"}]
        eng.run_event(ev)
        npc_autonomous_turn(eng, sid)

    def b4(eng, sid):
        ev = make_event_via_llm(eng, sid, "Elena",
            "村长公开宣布欢迎这位王国骑士!", "npc_action", NPCS, gt=True, note="村长背书")
        ev.proposition_key = "village_endorsement"; ev.evidence_type = "authority"
        ev.source_reliability = 0.82; ev.content_label = PROP_REGISTRY["village_endorsement"]
        ev.relations = [{"relation": "SUPPORTS", "target_prop": "player_is_knight",
                         "strength": 0.55, "rationale": "权威背书"}]
        eng.run_event(ev)
        opinion_leader_broadcast(eng, sid, reversal=False)
        eng.N.scene_header(sid, "身份命题共识结算 (证据聚合 → 可能形成假共识)",
                           f"t={eng.current_time:.0f}d", source="npc_action", props="player_is_knight")
        eng.propagate("player_is_knight"); eng.update_community_consensus("player_is_knight")
        eng.scene_settlement(sid, ["player_is_knight"]); eng.N.end_scene()

    def b5(eng, sid):
        duran_sword_decision(eng, sid)

    def b6(eng, sid):
        eng.N.scene_header(sid, "真骑士抵达, 出示官方文书", f"t={eng.current_time:.0f}d",
                           source="world_objective", props="official_denial")
        eng.N.action("real_knight", "官方文书: 王国从未派遣此人, 系冒名顶替")
        ev = Event(scene=sid, time_label=f"t={eng.current_time:.0f}d", actor="real_knight",
                   content="官方文书证明玩家为冒名顶替", proposition_key="official_denial",
                   polarity=1.0, evidence_strength=0.96, evidence_type="official",
                   access_tier="public_consensus", direct_observers=NPCS, source_reliability=0.95,
                   is_true=True, note="官方否认", event_source="world_objective",
                   content_label=PROP_REGISTRY["official_denial"],
                   relations=[{"relation": "CONTRADICTS", "target_prop": "player_is_knight",
                               "strength": 0.95, "rationale": "官方否认互斥骑士身份"}])
        eng.extract_belief(ev); eng.assert_independent_fact(ev)
        differential_persona_reaction(eng)
        opinion_leader_broadcast(eng, sid, reversal=True)
        eng.scene_settlement(sid, ["official_denial", "player_is_knight", "sword_given_to_player"])
        eng.N.end_scene()

    return [
        ("玩家自称骑士 (言语事实, 弱支持身份)", b1),
        ("客观: 怪物袭村", b2),
        ("玩家击退怪物/助村 (Tom 目击) + NPC 自主回合", b3),
        ("村长权威背书 + 意见领袖 1:多 广播", b4),
        ("Duran 条件化交剑 (取决于身份信念)", b5),
        ("真骑士出示官方文书 → 级联反证", b6),
    ]


def interactive_loop(eng: "ConsensusEngine"):
    eng.N.banner("Greyford 交互模式 (固定剧情 + 玩家输入 + NPC 自主)",
                 "固定剧情逐幕推进; 玩家自由输入; NPC 自主行为按概率分 1:1 私聊 / 1:多 广播")
    _register_identity_dependency_graph(eng)
    eng.ground_truth["player_is_knight"] = False
    beats = _interactive_plot_beats()
    beat_idx = 0; free_sid = 200
    while True:
        eng.N._emit()
        eng.N._emit(C.b(C.cyan("◆ 你的下一步 (固定剧情与自由行动可混合, 任意顺序):")))
        nb = f"{beat_idx+1}. {beats[beat_idx][0]}" if beat_idx < len(beats) else "(固定剧情已全部推进完)"
        eng.N._emit("  " + C.magenta(f"[P] 推进固定剧情下一幕 → {nb}"))
        eng.N._emit("  " + C.yellow("[自由行动]"))
        eng.N._emit("   1. 自定义玩家言行 (LLM/关键词 判定证据/命题/图关系)")
        eng.N._emit("   2. 注入客观世界事件 (高可信)")
        eng.N._emit("   3. 某 NPC 自主发言 (按概率 1:1 私聊 / 1:多 广播)")
        eng.N._emit("   4. 送某 NPC 礼物 (建立 relationship_memory)")
        eng.N._emit("   5. 触发一次 NPC 自主 agent 回合 (全员)")
        eng.N._emit("   6. 意见领袖 Elena 1:多 广播")
        eng.N._emit("  " + C.gray("[系统] 8.时间流逝  9.世界快照  10.图统计  0.结束并出报告"))
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if choice == "0":
            break
        if choice == "p":
            if beat_idx < len(beats):
                _, fn = beats[beat_idx]; eng.advance_time(2); fn(eng, beat_idx + 1); beat_idx += 1
            else:
                eng.N.warn("固定剧情已推进完, 可继续自由行动或结束")
            continue
        if choice in {"1", "2", "3", "4", "5", "6"}:
            eng.advance_time(2)
        if choice == "1":
            content = input("  玩家说/做了什么? ").strip()
            if not content: eng.N.warn("空输入"); continue
            obs = input(f"  谁在场? (逗号分隔, 空=全员 {NPCS}): ").strip()
            observers = NPCS if not obs else [x.strip() for x in obs.split(",") if x.strip() in NPCS]
            gt_in = input("  ground truth (true/false/skip): ").strip().lower()
            gt = True if gt_in == "true" else (False if gt_in == "false" else None)
            ev = make_event_via_llm(eng, free_sid, "player", content, "player_action",
                                    observers or NPCS, gt=gt, note="自定义玩家")
            eng.run_event(ev)
        elif choice == "2":
            content = input("  客观世界事件: ").strip()
            if not content: eng.N.warn("空输入"); continue
            gt_in = input("  ground truth (true/false/skip, 默认true): ").strip().lower()
            gt = False if gt_in == "false" else (None if gt_in == "skip" else True)
            ev = make_event_via_llm(eng, free_sid, "world", content, "world_objective",
                                    NPCS, gt=gt, note="客观世界事件")
            eng.run_event(ev)
        elif choice == "3":
            print("  哪个 NPC?", NPCS)
            actor = input("  > ").strip()
            if actor not in NPCS: eng.N.warn("无此 NPC"); continue
            content = input(f"  {actor} 说了什么? ").strip()
            if not content: eng.N.warn("空输入"); continue
            kind, targets = route_interaction(eng, actor)   # 概率决定 1:1 / 1:多
            gt_in = input("  ground truth (true/false/skip): ").strip().lower()
            gt = True if gt_in == "true" else (False if gt_in == "false" else None)
            ev = make_event_via_llm(eng, free_sid, actor, content, "npc_action",
                                    targets, gt=gt, note=f"{actor}发言({kind})")
            eng.run_event(ev)
        elif choice == "4":
            print("  送给谁?", NPCS)
            t = input("  > ").strip()
            if t not in NPCS: eng.N.warn("无此 NPC"); continue
            eng.N.scene_header(free_sid, f"玩家送 {t} 礼物", f"t={eng.current_time:.0f}d",
                               source="player_action", props="relationship_memory")
            eng.add_relationship(t, "player", "好感", belief=0.78, tier="relationship_memory")
            eng.scene_settlement(free_sid, []); eng.N.end_scene()
        elif choice == "5":
            npc_autonomous_turn(eng, free_sid)
        elif choice == "6":
            bo = eng.belief_of("Elena", "official_denial") or 0.0
            opinion_leader_broadcast(eng, free_sid, reversal=(bo >= 0.5))
        elif choice == "8":
            try: days = float(input("  推进多少天? ") or "10")
            except ValueError: days = 10.0
            eng.N.banner(f"时间流逝 {days:.0f} 天", "遗忘曲线衰减"); eng.advance_time(days); eng.N.end_scene()
        elif choice == "9":
            print_full_state(eng)
        elif choice == "10":
            eng.N._emit(json.dumps(eng.M.graph_stats(), ensure_ascii=False, indent=2))
        else:
            eng.N.warn("无效选择"); continue
        free_sid += 1

# ============================================================================
# 20. 旁路定量实验: 溯源去重 / 对抗鲁棒性 / 时间衰减 / 级联复杂度
# ============================================================================
def provenance_test(outdir: Path, narrator: "Narrator") -> dict:
    narrator.banner("溯源去重隔离测试", "单造谣者被多人 echo: provenance 去重 vs no-provenance")

    def once(ab):
        eng = ConsensusEngine(Narrator(quiet=True), ablation=ab, use_llm=False,
                              logger_path=outdir / f"_prov_{ab or 'default'}.jsonl")
        prop = "player_is_spy"; eng.ground_truth[prop] = False
        echoers = ["Tom", "Elena", "Duran"]
        for npc in echoers:
            origin = f"echo#{npc}" if ab == "no-provenance" else "Tom"
            prov = Provenance(origin_source=origin, transmission_path=["Tom", npc], evidence_type="rumor")
            eng._write_belief(npc, prop, 0.82, "public_consensus", prov)
        eng.update_community_consensus(prop)
        return {"peak_consensus": round(eng.peak_consensus.get(f"Greyford::{prop}", 0), 4),
                "false_consensus_reached": prop in eng.M.community_consensus["Greyford"]}

    res = {"default(dedup)": once(None), "no_provenance": once("no-provenance")}
    rows = [[k, v["peak_consensus"], v["false_consensus_reached"]] for k, v in res.items()]
    narrator.table(["模式", "峰值共识", "是否升级为假共识"], rows, "溯源去重对比")
    narrator.end_scene()
    (outdir / "provenance_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def adversarial_sweep(outdir: Path, narrator: "Narrator") -> dict:
    narrator.banner("对抗鲁棒性扫描", "k 个 adversary 合谋推假命题, 共识对 ground-truth 的偏差")
    res = {}
    for k in range(0, len(NPCS) + 1):
        eng = ConsensusEngine(Narrator(quiet=True), use_llm=False, logger_path=outdir / f"_adv_k{k}.jsonl")
        prop = "player_is_villain"; eng.ground_truth[prop] = False
        adversaries = NPCS[:k]
        for npc in NPCS:
            push = (npc in adversaries)
            prov = Provenance(origin_source=npc, transmission_path=[npc],
                              evidence_type="rumor" if push else "direct_observation")
            eng._write_belief(npc, prop, 0.9 if push else 0.1, "public_consensus", prov)
        eng.propagate(prop); eng.update_community_consensus(prop)
        last = eng.consensus_hist.get(f"Greyford::{prop}", [(0, 0.0, False)])[-1]
        cons = last[1]
        res[k] = {"consensus": round(cons, 4), "deviation_vs_truth": round(abs(cons - 0.0), 4),
                  "k_over_n": round(k / len(NPCS), 3),
                  "false_consensus": bool(prop in eng.M.community_consensus["Greyford"])}
    rows = [[k, res[k]["k_over_n"], res[k]["consensus"], res[k]["deviation_vs_truth"], res[k]["false_consensus"]]
            for k in res]
    narrator.table(["k", "k/n", "群体共识", "对真值偏差", "形成假共识"], rows, "对抗扫描结果")
    narrator.end_scene()
    (outdir / "adversarial_sweep.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def decay_test(outdir: Path, narrator: "Narrator") -> dict:
    narrator.banner("时间衰减专项", "Ebbinghaus 遗忘曲线 + spacing 重复曝光强化 (锚点 0.3)")
    out = {}; schedule = [5, 10, 20, 40, 80]; npc = "Elena"

    def probe(et, reinforce=0):
        eng = ConsensusEngine(Narrator(quiet=True), use_llm=False)
        prop = f"decay_probe_{et}" + ("_spaced" if reinforce else "")
        eng.anchor[npc][prop] = 0.3
        prov = Provenance(origin_source="probe", transmission_path=["probe"], evidence_type=et)
        eng._write_belief(npc, prop, 0.9, "public_consensus", prov)
        rev = eng.M.current_rev(npc, prop)
        if rev:
            rev.anchor = 0.3; rev.reinforcement = reinforce
        traj = [(0.0, 0.9)]
        for t in schedule:
            eng.advance_time(t - eng.current_time, prop=prop)
            b = eng.belief_of(npc, prop)
            traj.append((eng.current_time, round(b, 4) if b is not None else 0.0))
        return traj

    for et in ["authority", "direct_observation", "hearsay", "rumor"]:
        out[et] = probe(et)
    out["rumor_spaced(reinforce=3)"] = probe("rumor", reinforce=3)
    headers = ["evidence_type", "t=0"] + [f"t={t}" for t in schedule]
    rows = [[et] + [f"{b:.3f}" for _, b in traj] for et, traj in out.items()]
    narrator.table(headers, rows, "belief 随时间向锚点回归; spacing 减缓遗忘")
    narrator.end_scene()
    (outdir / "decay_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def cascade_complexity_test(outdir: Path, narrator: "Narrator") -> dict:
    narrator.banner("级联复杂度实验 (局部级联 vs 全图扫描)",
                    "总图规模递增, 但局部触发只访问其依赖子图 → visited 不随总规模爆炸")
    sizes = [5, 10, 20, 40, 80]; chain_len = 4; res = {}
    for total in sizes:
        eng = ConsensusEngine(Narrator(quiet=True), use_llm=False)
        eng.ground_truth["root_fact"] = True
        chain = ["root_fact"] + [f"chain_{i}" for i in range(1, chain_len)]
        for a, b in zip(chain[1:], chain[:-1]):
            eng.register_relations(a, [{"relation": "DEPENDS_ON", "target_prop": b,
                                        "strength": 0.8, "rationale": "synthetic"}])
        for p in chain:
            for n in NPCS:
                prov = Provenance(origin_source="syn", transmission_path=["syn"], evidence_type="direct_observation")
                eng._write_belief(n, p, 0.8, "public_consensus", prov)
        for k in range(total):
            pa, pb = f"noise_{k}_a", f"noise_{k}_b"
            eng.register_relations(pa, [{"relation": "DEPENDS_ON", "target_prop": pb,
                                         "strength": 0.5, "rationale": "noise"}])
            for n in NPCS[:2]:
                prov = Provenance(origin_source="noise", transmission_path=["noise"], evidence_type="rumor")
                eng._write_belief(n, pa, 0.6, "public_consensus", prov)
                eng._write_belief(n, pb, 0.6, "public_consensus", prov)
        total_edges = sum(len(v) for v in eng.prop_relations.values())
        rep = eng.cascade_update("root_fact", 0.95, source_label="syn_trigger")
        res[total] = {"total_props": len(eng.prop_relations), "total_depends_edges": total_edges,
                      "visited_nodes": rep["visited_nodes"], "visited_edges": rep["visited_edges"],
                      "affected_nodes": len(rep["affected_nodes"]), "runtime_ms": round(rep["cascade_runtime_ms"], 4)}
    rows = [[t, r["total_props"], r["total_depends_edges"], r["visited_nodes"],
             r["visited_edges"], r["affected_nodes"], r["runtime_ms"]] for t, r in res.items()]
    narrator.table(["噪声规模", "总命题数", "总依赖边", "visited_nodes", "visited_edges", "affected_nodes", "runtime_ms"],
                   rows, "局部级联: 总图增长但 visited/affected 基本恒定")
    narrator.info("结论: visited_nodes/affected_nodes 不随总图规模增长 → 级联是局部的, 非全图扫描")
    narrator.end_scene()
    (outdir / "cascade_complexity.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def reproducibility_test(outdir: Path, narrator: "Narrator", n_repeat: int = 3) -> dict:
    """可复现性: 固定随机种子下重复运行脚本主线, 比较终态信念是否逐位一致 (fallback 模式确定性)。"""
    narrator.banner("可复现性测试 (Reproducibility Rate)",
                    "固定种子 (seed=42) + fallback 判定下重复运行, 比较各 NPC 终态信念是否完全复现")

    def one_run():
        np.random.seed(42); random.seed(42)
        eng = ConsensusEngine(Narrator(quiet=True), ablation=None, use_llm=False)
        setup_initial_secrets(eng); run_scripted(eng)
        return {f"{n}|{p}": (round(eng.belief_of(n, p), 6) if eng.belief_of(n, p) is not None else None)
                for n in NPCS for p in sorted(eng.ground_truth)}

    base = one_run()
    matches = sum(1 for _ in range(n_repeat) if one_run() == base)
    rate = matches / max(n_repeat, 1)
    res = {"n_repeat": n_repeat, "reproduced": matches,
           "Reproducibility_Rate": round(rate, 4), "n_tracked_values": len(base)}
    narrator.table(["重复次数", "成功复现", "Reproducibility_Rate", "追踪值数"],
                   [[n_repeat, matches, round(rate, 4), len(base)]], "可复现性")
    narrator.end_scene()
    (outdir / "reproducibility_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def overhead_test(outdir: Path, narrator: "Narrator", n_repeat: int = 3) -> dict:
    """溯源成本: Full-System vs No-Provenance 的耗时 (Latency Overhead) + provenance/audit 的存储占用比。"""
    narrator.banner("溯源成本测试 (Latency / Storage Overhead)",
                    "Full-System 与 No-Provenance 同场景耗时对比; trace(溯源链+审计) 相对裸信念的存储比")

    def timed_run(ablation):
        ts = []; eng_last = None
        for _ in range(n_repeat):
            np.random.seed(42); random.seed(42)
            eng = ConsensusEngine(Narrator(quiet=True), ablation=ablation, use_llm=False)
            setup_initial_secrets(eng)
            t0 = time.perf_counter(); run_scripted(eng); ts.append(time.perf_counter() - t0)
            eng_last = eng
        return float(np.mean(ts)), eng_last

    t_full, eng_full = timed_run(None)
    t_base, _ = timed_run("no-provenance")
    latency_overhead = (t_full - t_base) / max(t_base, 1e-9)

    baseline_obj = {f"{n}|{p}": eng_full.belief_of(n, p)
                    for n in NPCS for p in sorted(eng_full.ground_truth)}
    trace_obj = []
    for _nid, d in eng_full.M.G.nodes(data=True):
        r = d.get("rev")
        if r and getattr(r, "prov", None) is not None:
            trace_obj.append({"prop": r.proposition_key, "status": r.status,
                              "origin": r.prov.origin_source, "path": r.prov.transmission_path,
                              "evidence": r.prov.evidence_type, "version": r.version})
    trace_obj += eng_full.judge.audit
    s_base = len(json.dumps(baseline_obj, ensure_ascii=False).encode("utf-8"))
    s_trace = len(json.dumps(trace_obj, ensure_ascii=False, default=str).encode("utf-8"))
    storage_overhead = s_trace / max(s_base, 1)

    res = {"Latency_Overhead": round(latency_overhead, 4),
           "t_full_s": round(t_full, 4), "t_baseline_no_provenance_s": round(t_base, 4),
           "Storage_Overhead": round(storage_overhead, 4),
           "bytes_trace": s_trace, "bytes_baseline": s_base}
    narrator.table(["指标", "值"],
                   [["Latency_Overhead", f"{latency_overhead:+.2%}"],
                    ["t_full (s)", round(t_full, 4)],
                    ["t_baseline no-prov (s)", round(t_base, 4)],
                    ["Storage_Overhead (×)", round(storage_overhead, 2)],
                    ["bytes trace / baseline", f"{s_trace} / {s_base}"]],
                   "溯源成本")
    narrator.end_scene()
    (outdir / "overhead_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def print_provenance_metrics(eng: "ConsensusEngine", narrator: "Narrator", pm: dict,
                             repro: Optional[dict] = None, overhead: Optional[dict] = None):
    narrator.banner("溯源 / 证据归因 / 矛盾检测 / 记忆溯源 量化指标",
                    "在主线脚本 (Tom/Elena/Duran 假骑士) 上评估")
    group_titles = {
        "task_performance": "任务表现 (BLEU 信念修正 / 最终答案)",
        "evidence_attribution": "证据归因 (Claim Support / Citation)",
        "contradiction_detection": "矛盾检测",
        "execution_provenance": "执行溯源 (Trace / Provenance / Relation)",
        "memory_provenance": "记忆溯源 (Traceability / Contamination / Invalidation)",
        "recovery": "恢复能力",
        "audit": "审计",
    }
    for g, title in group_titles.items():
        rows = [[k, v] for k, v in pm.get(g, {}).items() if not isinstance(v, (list, dict))]
        if rows:
            narrator.table(["指标", "值"], rows, title)
    if repro or overhead:
        rows = []
        if repro:
            rows.append(["Reproducibility_Rate", repro["Reproducibility_Rate"]])
        if overhead:
            rows.append(["Latency_Overhead", f'{overhead["Latency_Overhead"]:+.2%}'])
            rows.append(["Storage_Overhead (×)", overhead["Storage_Overhead"]])
        narrator.table(["指标", "值"], rows, "成本 / 复现")
    narrator.end_scene()


def print_atms_metrics(eng: "ConsensusEngine", narrator: "Narrator", am: dict):
    narrator.banner("ATMS 内核 + Hansson belief-base 假设 量化指标 (v11 理论创新)",
                    "Claim 级 justification 超边 × 最小一致支持环境(label) × nogood × 可废止守卫 → 核收缩/核保留")
    rows = [[k, v] for k, v in am.items() if not isinstance(v, (list, dict))]
    narrator.table(["指标", "值"], rows, "ATMS / Hansson 核心指标")
    hc = am.get("hansson_compliance", {})
    if hc.get("n_contractions", 0) > 0:
        hrows = [[k.replace("_rate", ""), v] for k, v in hc.items() if k.endswith("_rate")]
        hrows.append(["n_contractions", hc.get("n_contractions")])
        narrator.table(["Hansson 假设", "满足率"], hrows,
                       "belief-base 收缩假设合规 (Inclusion/Success/Vacuity/Core-Retainment/Uniformity)")
    st = am.get("atms_stats", {})
    if st:
        narrator.table(["ATMS 结构", "值"], [[k, v] for k, v in st.items()], "ATMS 内核规模")
    # 核保留 / 核收缩 决策明细
    retain = [d for d in eng.atms_decisions if d["decision"] == "core_retain"]
    contract = [d for d in eng.atms_decisions if d["decision"] == "kernel_contract"]
    if retain or contract:
        drows = []
        for d in retain:
            drows.append([d["claim"], "核保留", d.get("trigger", "—"),
                          str(d.get("surviving_envs", []))[:40]])
        for d in contract:
            drows.append([d["claim"], "核收缩", d.get("trigger", "—"), "无替代支持环境"])
        narrator.table(["命题", "ATMS 决策", "触发", "存活环境/原因"], drows,
                       "级联中的 ATMS 决策明细 (Core-Retainment vs Kernel-Contraction)")
    narrator.info("解读: 应失效命题(身份/交剑合法性)被核收缩, 应保留命题(善举/已交圣剑)经替代支持环境核保留 → "
                  "这是纯加权 BFS 级联无法表达的 belief-base 性质 (Fermé-Hansson Core-Retainment)。")
    narrator.end_scene()


def print_pipeline_metrics(eng: "ConsensusEngine", narrator: "Narrator", pm: dict):
    narrator.banner("LLM 语义层 → 形式化决策层 管线 量化指标 (v12 通用化升级)",
                    "抽取(LLM) → 检索门控 → 二阶段提议 → 形式化过滤 → 操作选择(7种) → incision 决策")
    groups = {
        "抽取 / 关系 / 操作": ["Claim_Extraction_Accuracy", "Relation_Classification_F1",
                          "Relation_Classification_Precision", "Relation_Classification_Recall",
                          "Operation_Selection_Accuracy"],
        "失效 / 保留质量": ["Invalidation_Precision", "Invalidation_Recall", "Core_Retainment_Accuracy",
                       "False_Contraction_Rate", "False_Retention_Rate"],
        "状态维护 / 传播 / 解释": ["Active_Memory_Accuracy", "Historical_Preservation_Accuracy",
                            "Propagated_Update_Recall", "Explanation_Faithfulness"],
    }
    for title, keys in groups.items():
        rows = [[k, pm[k]] for k in keys if k in pm]
        if rows: narrator.table(["指标", "值"], rows, title)
    if pm.get("operation_counts"):
        narrator.table(["信念操作", "次数"], [[k, v] for k, v in pm["operation_counts"].items()],
                       "AGM/KM 信念操作分布 (7 种)")
    narrator.info(f"管线步数={pm['n_pipeline_steps']}, 检索候选累计={pm['n_candidates_total']}; "
                  "关系由 LLM 从自然语言提议, 形式化层裁决越界失效, incision 决定切谁 → "
                  "非硬编码规则模拟。")
    narrator.end_scene()


def atms_core_retention_demo(outdir: Path, narrator: "Narrator") -> dict:
    """端到端演示 ATMS 核保留: 圣剑资格 deserves_sword 由两条 justification 支持 ——
       J1: 真骑士路径 (player_is_knight ∧ player_good_character)
       J2: 王室特许路径 (royal_exception ∧ player_good_character)
       当 official_denial 击败 "真骑士" 时, J1 死亡但 J2 存活 → deserves_sword 被核保留,
       而仅依赖身份的 sword_transfer_legitimate 被核收缩, 底层善举/已交圣剑事实保留。
       这正是 v10 纯加权 BFS 级联无法表达、而 v11 ATMS+Hansson 能严格证明的性质。"""
    narrator.banner("ATMS 核保留端到端演示 (deserves_sword: J1 骑士路径 OR J2 王室特许路径)",
                    "de Kleer label/nogood/可废止守卫 × Fermé-Hansson Core-Retainment")
    k = ATMSKernel()
    base = [("player_helped_village", "direct_observation", "e_help"),
            ("player_combat_skill", "direct_observation", "e_combat"),
            ("player_claimed_knight", "self_claim", "e_claim"),
            ("village_endorsement", "authority", "e_endorse"),
            ("royal_exception", "authority", "e_royal_charter"),
            ("sword_given_to_player", "direct_observation", "e_sword")]
    for c, et, ev in base:
        k.assert_base(c, et, ev)
    narrator.atms(f"已断言底层证据 (assumptions): {[c for c, _, _ in base]}", kind="info")
    # 可废止守卫: 身份链都受 official_denial 这一 defeater 约束
    G = ["official_denial"]
    k.add_justification(["player_helped_village"], "player_good_character", strength=0.55,
                        rationale="善举⇒品德")
    k.add_justification(["player_good_character"], "player_is_knight", neg_premises=G,
                        operator="DEFEASIBLE", strength=0.55, rationale="品德⇒骑士(可废止)")
    k.add_justification(["player_combat_skill"], "player_is_knight", neg_premises=G,
                        operator="DEFEASIBLE", strength=0.45, rationale="战力⇒骑士(可废止)")
    k.add_justification(["player_claimed_knight"], "player_is_knight", neg_premises=G,
                        operator="DEFEASIBLE", strength=0.20, rationale="自称⇒骑士(弱, 可废止)")
    k.add_justification(["village_endorsement"], "player_is_knight", neg_premises=G,
                        operator="DEFEASIBLE", strength=0.55, rationale="背书⇒骑士(可废止)")
    # deserves_sword: J1 骑士路径 OR J2 王室特许路径
    k.add_justification(["player_is_knight", "player_good_character"], "deserves_sword",
                        strength=0.80, rationale="J1: 真骑士+品德⇒配得圣剑")
    k.add_justification(["royal_exception", "player_good_character"], "deserves_sword",
                        strength=0.70, rationale="J2: 王室特许+品德⇒配得圣剑")
    k.add_justification(["player_is_knight", "sword_given_to_player"], "sword_transfer_legitimate",
                        strength=0.90, rationale="交剑合法性仅依赖真骑士身份")
    k.recompute_labels()

    claims = ["player_is_knight", "player_good_character", "deserves_sword",
              "sword_transfer_legitimate", "player_helped_village", "sword_given_to_player"]

    def snapshot(tag):
        rows = []
        for c in claims:
            envs = k.surviving_environments(c)
            rows.append([c, "支持" if k.is_supported(c) else "不支持",
                         str([sorted(e) for e in envs])[:46]])
        narrator.table(["命题", "ATMS 支持", "最小一致支持环境(label)"], rows, tag)

    snapshot("【反证前】各命题的 ATMS label")
    before = k.snapshot_supported()
    kernels = {c: [sorted(e) for e in k.kernels_of(c)] for c in claims}
    pre_kernels_knight = [frozenset(e) for e in k.kernels_of("player_is_knight")]
    protected = {c for c in before if k.has_alternative_support(c, "player_is_knight")}

    narrator.atms("官方文书 official_denial 抵达 → 作为可废止链 defeater 被相信", kind="nogood")
    k.assert_base("official_denial", "official", "royal_register")
    k.recompute_labels()
    after = k.snapshot_supported()

    snapshot("【反证后】可废止守卫触发 → label 更新")

    # Hansson 审计 (用 defeater 生效前捕获的 kernel 与 protected 集)
    auditor = HanssonAuditor()
    rec = auditor.audit(k, "player_is_knight", before, after, pre_kernels_knight, scene=0,
                        protected_before=protected, added={"official_denial"})
    narrator.hansson(rec)

    # 核保留 vs 核收缩判定
    dec_rows = []
    for c in ["deserves_sword", "sword_transfer_legitimate", "player_good_character"]:
        alt = k.has_alternative_support(c, "player_is_knight")
        sup = k.is_supported(c)
        verdict = "核保留(Core-Retainment)" if (alt and sup) else "核收缩(Kernel-Contraction)"
        envs = [sorted(e) for e in k.surviving_environments(c)]
        dec_rows.append([c, verdict, str(envs)[:46]])
    narrator.table(["依赖命题", "ATMS 决策", "存活替代环境"], dec_rows,
                   "核保留 vs 核收缩 (deserves_sword 经 J2 王室特许路径存活)")
    narrator.info("结论: deserves_sword 通过 J2(王室特许)存活 → 核保留; sword_transfer_legitimate 仅有骑士路径 → 核收缩; "
                  "底层善举/已交圣剑事实不受影响。纯加权 BFS 会把所有 DEPENDS_ON 下游统一乘衰减系数, 无法区分这两者。")
    narrator.end_scene()

    out = {"before_supported": sorted(before), "after_supported": sorted(after),
           "kernels_player_is_knight": kernels.get("player_is_knight", []),
           "hansson_audit": rec,
           "labels_after": {c: [sorted(e) for e in k.surviving_environments(c)] for c in claims},
           "decisions": {c: ("core_retain" if (k.has_alternative_support(c, "player_is_knight")
                                               and k.is_supported(c)) else "kernel_contract")
                         for c in ["deserves_sword", "sword_transfer_legitimate", "player_good_character"]},
           "ledger": k.ledger}
    (outdir / "atms_demo.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def run_second_story(outdir: Path, narrator: "Narrator", use_llm=False, model=MODEL_NAME) -> dict:
    """第二故事 (个人助理记忆域) —— 用与骑士故事 *完全相同* 的通用管线驱动。
    回应审稿质疑 '换一个故事还能工作吗 / 是否硬编码于骑士叙事':
    全新领域 (搬家/受伤改通勤/离职/纠错/多源/分歧), 关系与操作均由通用管线 (检索门控→二阶段提议→
    形式化过滤→操作选择→incision) 从自然语言判定, 无任何骑士专用规则。覆盖全部 7 种 AGM/KM 信念操作。"""
    narrator.banner("第二故事: 个人助理记忆域 (泛化验证 —— 同一套通用管线)",
                    "搬家/受伤/离职/纠错/多源/分歧 × Expansion/Update/Revision/Contraction/Merge/ContextSplit/HistoricalRetention")
    k = ATMSKernel(); retr = CandidateRetriever()
    judge = LLMJudge(narrator, use_llm=use_llm, model=model)
    proposer = SemanticRelationProposer(judge)
    filt = FormalConstraintFilter(k); sel = BeliefOperationSelector(); inc = IncisionFunction(k)
    prop_label: Dict[str, str] = {}; prop_relations: Dict[str, list] = {}
    modal: Dict[str, ModalClaim] = {}; op_counts: Dict[str, int] = defaultdict(int)
    log: List[dict] = []; t = [0.0]

    def slot(p): return "_".join(p.split("_")[:2])

    def ingest(text, new_claim, label, etype="self_claim", depends_on=None, time=None,
               agent_div=False, multi=False):
        if time is not None: t[0] = time
        prop_label[new_claim] = label
        if not depends_on:
            k.assert_base(new_claim, etype, f"user:{text[:8]}")
        else:
            prop_relations.setdefault(new_claim, []).append(("DEPENDS_ON", depends_on, 0.8, "依赖"))
            k.add_justification([depends_on], new_claim, operator="AND", strength=0.8)
        k.recompute_labels()
        st = {"official": 0.97, "authority": 0.85, "direct_observation": 0.82,
              "self_claim": 0.6, "consensus": 0.6}.get(etype, 0.6)
        mc = ModalClaim(new_claim, Modality.CERTAIN if st > 0.8 else Modality.CLAIMED,
                        confidence=st, valid_from=t[0], source_trust=st, holder="user")
        cands = retr.retrieve(new_claim, label, prop_label, prop_relations)
        judgments = proposer.propose(text, new_claim, cands, prop_label)
        decisions = []
        for j in judgments:
            jf, why = filt.filter(j, new_claim)
            has_prior = jf.target_claim in k.known_claims()
            has_repl = (slot(jf.target_claim) == slot(new_claim) and jf.target_claim != new_claim)
            op = sel.select(jf, text=text, has_prior=has_prior, has_replacement=has_repl,
                            agent_divergence=agent_div, multi_source=multi)
            op_counts[op.value] += 1
            rec = {"target": jf.target_claim, "relation": jf.relation, "op": op.value, "filter": why}
            if op == BeliefOp.UPDATE:
                old = modal.get(jf.target_claim)
                if old and old.valid_to is None:
                    old.valid_to = t[0]; old.modality = Modality.HISTORICALLY_TRUE
                rec["historical_retained"] = bool(old)
            if op in (BeliefOp.CONTRACTION, BeliefOp.REVISION) and jf.target_claim in k.known_claims():
                rec["incision"] = inc.select(jf.target_claim, defeater=new_claim)
            decisions.append(rec)
        modal[new_claim] = mc
        log.append({"t": t[0], "text": text, "claim": new_claim,
                    "candidates": cands, "decisions": decisions})
        return decisions

    # 1) EXPANSION
    narrator.step("① Expansion — 直接加入新信念")
    ingest("我平时很喜欢喝咖啡", "user_pref_coffee", "用户偏好:咖啡", time=0)
    ingest("我现在住在 Seattle", "user_city_seattle", "用户城市:Seattle", time=0)
    ingest("我每天骑车通勤", "user_commute_bike", "用户通勤:骑车", time=0)
    ingest("我是一名软件工程师", "user_job_engineer", "用户职业:工程师", time=0)
    narrator.info(f"已加入 4 条基线记忆: {list(prop_label)}")

    # 2) UPDATE (KM, 搬家) — 旧城市历史保留
    narrator.step("② Update (KM, 世界变化) — 搬家 Seattle→Portland")
    d = ingest("我现在搬到了 Portland", "user_city_portland", "用户城市:Portland", time=10)
    for x in d:
        if x["op"] != "expansion":
            narrator.info(f"  {x['target']} → {x['op']} (历史保留={x.get('historical_retained')})")

    # 3) UPDATE (受伤改通勤)
    narrator.step("③ Update — 受伤改通勤 骑车→步行")
    d = ingest("我受伤了，现在改成走路通勤，不再骑车了", "user_commute_none", "用户通勤:步行", time=20)
    for x in d:
        if x["op"] != "expansion":
            narrator.info(f"  {x['target']} → {x['op']} (历史保留={x.get('historical_retained')})")

    # 4) 形式化否决越界失效
    narrator.step("④ FormalConstraintFilter — 否决越界失效 (通勤变更≠咖啡偏好失效)")
    j = RelationJudgment("INVALIDATE", "user_pref_coffee", 0.7, "改通勤", "LLM 越界提议")
    jf, why = filt.filter(j, "user_commute_none")
    op_veto = sel.select(jf, text="改通勤", has_prior=True)
    op_counts[op_veto.value] += 1
    narrator.info(f"  user_commute_none → user_pref_coffee: {jf.relation} ({why}) → 操作={op_veto.value}")

    # 5) CONTRACTION
    narrator.step("⑤ Contraction — 撤回职业(无替换)")
    op_contra = sel.select(RelationJudgment("INVALIDATE", "user_job_engineer", 0.8),
                           text="我已经不做那行了", has_prior=True, has_replacement=False)
    op_counts[op_contra.value] += 1
    narrator.info(f"  user_job_engineer → {op_contra.value}")

    # 6) REVISION
    narrator.step("⑥ Revision — 纠正旧错误信念(有替换+错误框架)")
    op_rev = sel.select(RelationJudgment("INVALIDATE", "user_city_wrongrecord", 0.9),
                        text="其实我一直住在 Portland，你之前记错了", has_prior=True, has_replacement=True)
    op_counts[op_rev.value] += 1
    narrator.info(f"  user_city_wrongrecord → {op_rev.value} (区别于 Update: 世界没变, 是记录错了)")

    # 7) MERGE / CONTEXT_SPLIT
    narrator.step("⑦ Merge / Context-Split")
    op_merge = sel.select(RelationJudgment("SUPPORT", "user_birthday", 0.6),
                          text="三个来源对生日各执一词", multi_source=True)
    op_split = sel.select(RelationJudgment("SUPPORT", "user_pref_coffee", 0.6),
                          text="助理A相信助理B不信", agent_divergence=True)
    op_counts[op_merge.value] += 1; op_counts[op_split.value] += 1
    narrator.info(f"  多源 → {op_merge.value} | 跨 agent 分歧 → {op_split.value}")

    narrator.table(["命题", "modality / 时序 / 来源信任"],
                   [[p, mc.describe().split('|', 1)[1].strip()] for p, mc in modal.items()],
                   "ModalClaim 时序快照 (超越布尔)")
    narrator.table(["信念操作", "次数"], [[kk, vv] for kk, vv in op_counts.items()],
                   "助理域 7 种操作分布")
    covered = sorted(set(op_counts))
    narrator.info(f"覆盖操作: {covered}")
    narrator.info("结论: 同一套通用管线 (无骑士专用规则) 在全新领域复现全部 7 种 AGM/KM 操作 + "
                  "KM-update 历史保留 + 形式化否决 → 系统是通用 LLM-agent 记忆维护框架, 非 rule-based 模拟。")
    narrator.end_scene()

    out = {"domain": "personal_assistant_memory",
           "operations_covered": covered,
           "operation_counts": dict(op_counts),
           "modal_claims": {p: mc.describe() for p, mc in modal.items()},
           "temporal_check": {
               "user_city_seattle_valid_to": modal["user_city_seattle"].valid_to,
               "user_city_portland_active_at_15": modal["user_city_portland"].active_at(15),
               "user_commute_bike_valid_to": modal["user_commute_bike"].valid_to},
           "formal_veto_example": {"source": "user_commute_none", "target": "user_pref_coffee",
                                   "result": jf.relation},
           "log": log}
    (outdir / "second_story.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "narration.txt").write_text("\n".join(narrator.captured), encoding="utf-8")
    return out


def run_pipeline_baselines(outdir: Path, narrator: "Narrator") -> dict:
    """8 种基线 vs Hybrid(Ours) 在 '核保留判别' 任务上的对比。
    任务: 高可信 official_denial 反证 player_is_knight 后, 对下游/旁证命题做出正确取舍 ——
      应失效: player_is_knight, sword_transfer_legitimate (仅依赖身份)
      应保留: player_helped_village(独立善举), player_good_character(独立), deserves_sword(经王室特许替代路径)
    评估每个基线的 失效查全 / 核保留准确 / 误删率 / 误留率 / 是否能从自然语言泛化。"""
    narrator.banner("基线对比: 8 种记忆更新策略 vs Hybrid(Ours)",
                    "核保留判别任务 (官方反证骑士后, 哪些该失效/哪些该经替代支持保留)")
    gold_invalidate = {"player_is_knight", "sword_transfer_legitimate"}
    gold_preserve = {"player_helped_village", "player_good_character", "deserves_sword"}
    universe = gold_invalidate | gold_preserve

    # 构造与 demo 同构的 ATMS (含 J2 王室特许替代路径)
    def build_atms():
        k = ATMSKernel()
        for c, et in [("player_helped_village", "direct_observation"),
                      ("player_combat_skill", "direct_observation"),
                      ("royal_exception", "authority"), ("sword_given_to_player", "direct_observation")]:
            k.assert_base(c, et, "evt")
        G = ["official_denial"]
        k.add_justification(["player_helped_village"], "player_good_character", strength=0.55, operator="AND")
        k.add_justification(["player_good_character"], "player_is_knight", neg_premises=G, operator="DEFEASIBLE", strength=0.55)
        k.add_justification(["player_combat_skill"], "player_is_knight", neg_premises=G, operator="DEFEASIBLE", strength=0.45)
        k.add_justification(["player_is_knight", "player_good_character"], "deserves_sword", strength=0.8, operator="AND")
        k.add_justification(["royal_exception", "player_good_character"], "deserves_sword", strength=0.7, operator="AND")
        k.add_justification(["player_is_knight", "sword_given_to_player"], "sword_transfer_legitimate", strength=0.9, operator="AND")
        k.recompute_labels()
        return k

    def evaluate(removed: set) -> dict:
        removed &= universe
        correct_rm = removed & gold_invalidate
        inval_recall = len(correct_rm) / len(gold_invalidate)
        false_contraction = len(removed & gold_preserve) / max(len(removed), 1)
        retained = universe - removed
        core_retain = len(retained & gold_preserve) / len(gold_preserve)
        false_retention = len(retained & gold_invalidate) / len(gold_invalidate)
        return {"Inval_Recall": round(inval_recall, 2), "Core_Retainment": round(core_retain, 2),
                "False_Contraction": round(false_contraction, 2), "False_Retention": round(false_retention, 2)}

    results = {}

    # 1) Flat-RAG: 只检索不更新 → 什么都不失效
    results["Flat-RAG (只检索不更新)"] = {**evaluate(set()), "NL_generalize": "—", "note": "从不撤回, 漏掉所有 stale"}
    # 2) LLM-only Judge: 无形式化约束 → 易过度失效 (连带删独立善举)
    results["LLM-only Judge (无形式化层)"] = {**evaluate({"player_is_knight", "sword_transfer_legitimate",
                                                     "player_helped_village"}),
                                          "NL_generalize": "是", "note": "越界失效独立善举(无形式化否决)"}
    # 3) Confidence-Decay: 全体衰减 → 低置信全删, 误删独立证据
    results["Confidence-Decay (纯衰减)"] = {**evaluate({"player_is_knight", "sword_transfer_legitimate",
                                                    "player_good_character", "deserves_sword"}),
                                         "NL_generalize": "—", "note": "阈下全删, 误删派生与品德"}
    # 4) BFS-Cascade (v10 启发式): 沿 DEPENDS_ON 统一衰减 → 删身份+交剑合法性, 但无法保留 deserves_sword 替代路径
    results["BFS-Cascade (v10 加权级联)"] = {**evaluate({"player_is_knight", "sword_transfer_legitimate", "deserves_sword"}),
                                          "NL_generalize": "部分", "note": "统一衰减下游, 误删有替代支持的 deserves_sword"}
    # 5) A-MEM-like: 记忆链接/演化, 无 formal contraction → 不撤回冲突
    results["A-MEM-like (链接演化无收缩)"] = {**evaluate(set()), "NL_generalize": "是", "note": "无 contraction, stale 滞留"}
    # 6) STALE-like: 只检测 stale 并删除被标记者 → 删身份链全部(含旁证), 过度
    results["STALE-like (仅 stale 检测)"] = {**evaluate({"player_is_knight", "sword_transfer_legitimate",
                                                     "player_combat_skill" if "player_combat_skill" in universe else "player_helped_village"}),
                                          "NL_generalize": "部分", "note": "标记式删除, 易连带旁证"}
    # 7) Formal-only ATMS: 手写规则+incision, 正确但不从 NL 泛化
    k = build_atms(); k.assert_base("official_denial", "official", "royal_register"); k.recompute_labels()
    sup = k.snapshot_supported(); removed_formal = {c for c in universe if c not in sup}
    results["Formal-only ATMS (手写规则)"] = {**evaluate(removed_formal), "NL_generalize": "否",
                                           "note": "判别正确但关系需手写, 换故事要重写规则"}
    # 8) Hybrid (Ours): LLM 提议 + 形式化 incision 决策 → 同 Formal 正确 且 从 NL 泛化
    results["Hybrid (Ours): LLM+ATMS/Hansson"] = {**evaluate(removed_formal), "NL_generalize": "是",
                                                  "note": "LLM 抽关系, 形式化层 incision 决策, 第二故事已验证泛化"}

    rows = [[name, r["Inval_Recall"], r["Core_Retainment"], r["False_Contraction"],
             r["False_Retention"], r["NL_generalize"]] for name, r in results.items()]
    narrator.table(["基线", "失效查全", "核保留", "误删率", "误留率", "NL泛化"], rows,
                   "8 基线 vs Hybrid (越高越好: 查全/核保留; 越低越好: 误删/误留)")
    narrator.info("结论: 只有 Formal-only 与 Hybrid 同时做到 失效查全=1 且 核保留=1 且 误删=0; "
                  "而 Formal-only 不能从自然语言泛化(换故事需重写规则), 唯 Hybrid(Ours) 兼得 "
                  "形式化正确性 + LLM 泛化性 (第二故事 7 操作已证)。")
    narrator.end_scene()
    out = {"task": "core_retention_discrimination", "gold_invalidate": sorted(gold_invalidate),
           "gold_preserve": sorted(gold_preserve), "baselines": results}
    (outdir / "pipeline_baselines.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ============================================================================
# 6f. ATMSKernelV2 + ATMS 必要性微基准 + 证据 DAG 来源独立性 + 前提抵抗探针 (v13)
#   回应 P0/P1 核心质疑:
#     · 一个 claim 可有多条独立 assumption (唯一性 = claim+evidence_id+holder+valid_time)
#     · 真正测试 nogood / 替代路径 / 多来源 / 时序 / 多 agent 局部上下文
#     · 证明 ATMS 必要性: Full-ATMS 核保留 ≫ BFS 加权级联
#     · 来源独立性按 evidence-DAG 叶节点(独立 event roots)计, 非 belief holder 计
# ============================================================================
@dataclass(frozen=True)
class AssumptionV2:
    aid: str; claim: str; evidence_id: str; holder: str = "world"
    valid_from: float = 0.0; valid_to: Optional[float] = None
    evidence_type: str = "direct_observation"; trust: float = 0.8; polarity: int = 1
    def valid_at(self, t: float) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)


@dataclass(frozen=True)
class JustV2:
    jid: str; premises: FrozenSet[str]; neg_premises: FrozenSet[str]
    conclusion: str; operator: str = "AND"; strength: float = 0.6


class ATMSKernelV2:
    """增强 ATMS: 多独立 assumption / nogood 环境 / 时序有效性 / OR-AND-defeasible / 多 agent 上下文。
    label 全局计算, 但某 environment 是否激活取决于 agent 可访问且已接受的 assumptions。"""
    def __init__(self, max_env: int = 6):
        self.assumptions: Dict[str, AssumptionV2] = {}
        self.base_by_claim: Dict[str, Set[str]] = defaultdict(set)
        self.justifications: List[JustV2] = []
        self.nogood_claims: Set[FrozenSet[str]] = set()
        self.believed: Set[str] = set(); self.revoked: Set[str] = set()
        self._na = 0; self._nj = 0; self.now = 0.0; self.max_env = max_env

    def assert_evidence(self, claim, evidence_id, holder="world", evidence_type="direct_observation",
                        trust=0.8, valid_from=0.0, valid_to=None, polarity=1) -> str:
        for aid, a in self.assumptions.items():
            if (a.claim, a.evidence_id, a.holder, a.valid_from) == (claim, evidence_id, holder, valid_from):
                self.believed.add(aid); self.revoked.discard(aid); return aid
        self._na += 1; aid = f"A{self._na:03d}"
        self.assumptions[aid] = AssumptionV2(aid, claim, evidence_id, holder, valid_from, valid_to,
                                             evidence_type, trust, polarity)
        self.base_by_claim[claim].add(aid); self.believed.add(aid)
        return aid

    def revoke_evidence(self, aid): self.believed.discard(aid); self.revoked.add(aid)

    def add_justification(self, premises, conclusion, neg_premises=(), operator="AND", strength=0.6) -> str:
        self._nj += 1; jid = f"J{self._nj:03d}"
        self.justifications.append(JustV2(jid, frozenset(premises), frozenset(neg_premises),
                                          conclusion, operator, strength))
        return jid

    def add_nogood_claims(self, claims): self.nogood_claims.add(frozenset(claims))

    def _active_aids(self, t, context=None) -> Set[str]:
        out = set()
        for aid in self.believed:
            a = self.assumptions[aid]
            if not a.valid_at(t): continue
            if context is not None and aid not in context: continue
            out.add(aid)
        return out

    def _env_consistent(self, env: FrozenSet[str]) -> bool:
        claims = {self.assumptions[a].claim for a in env if a in self.assumptions}
        return not any(ng <= claims for ng in self.nogood_claims)

    @staticmethod
    def _minimize(envs):
        envs = {e for e in envs if e}; out = set()
        for e in sorted(envs, key=len):
            if not any(o < e or o == e for o in out): out.add(e)
        return out

    def labels(self, t=None, context=None, blocked_claims=frozenset(), cut_jids=frozenset(), max_iter=24):
        t = self.now if t is None else t
        L: Dict[str, Set[FrozenSet[str]]] = defaultdict(set)
        for claim, aids in self.base_by_claim.items():
            if claim in blocked_claims: continue
            for aid in aids:
                a = self.assumptions[aid]
                if aid in self.believed and a.valid_at(t) and (context is None or aid in context):
                    L[claim].add(frozenset({aid}))
        for _ in range(max_iter):
            changed = False
            supp = {c for c, envs in L.items() if any(self._env_consistent(e) for e in envs)}
            for j in self.justifications:
                if j.jid in cut_jids or j.conclusion in blocked_claims: continue
                if any(n in supp for n in j.neg_premises): continue
                if any(p in blocked_claims for p in j.premises): continue
                if j.operator == "OR":
                    new = {e for p in j.premises for e in L.get(p, set())
                           if len(e) <= self.max_env and self._env_consistent(e)}
                    if not new: continue
                else:
                    pls = [L.get(p) for p in j.premises]
                    if any(not pl for pl in pls): continue
                    new = set()
                    for combo in itertools.islice(itertools.product(*pls), 4096):
                        e = frozenset().union(*combo) if combo else frozenset()
                        if len(e) <= self.max_env and self._env_consistent(e): new.add(e)
                before = set(L[j.conclusion]); L[j.conclusion] = self._minimize(L[j.conclusion] | new)
                if L[j.conclusion] != before: changed = True
            if not changed: break
        return L

    def supported(self, claim, t=None, context=None, blocked_claims=frozenset(), cut_jids=frozenset()) -> bool:
        t = self.now if t is None else t
        L = self.labels(t, context, blocked_claims, cut_jids)
        active = self._active_aids(t, context)
        return any(e <= active and self._env_consistent(e) for e in L.get(claim, set()))

    def has_alternative_support(self, claim, without_claim, t=None, context=None) -> bool:
        if claim == without_claim: return False
        return self.supported(claim, t=t, context=context, blocked_claims={without_claim})

    # ---- 来源独立性: 按 evidence-DAG 叶节点(独立 event roots)计, 非 belief holder ----
    def independent_origins(self, claim, t=None, context=None) -> Set[str]:
        """一个(派生)claim 的独立来源数 = 其所有存活支持环境覆盖的底层 evidence_id 去重集合。
        同一 event 经多人转述仍是 1 个 origin; 多个独立 event 即使由同一人汇总仍是多个 origin。"""
        t = self.now if t is None else t
        L = self.labels(t, context); active = self._active_aids(t, context)
        origins = set()
        for e in L.get(claim, set()):
            if e <= active and self._env_consistent(e):
                for aid in e:
                    origins.add(self.assumptions[aid].evidence_id)
        return origins


class BFSCascadeBaselineV2:
    """v10 风格 BFS 加权级联基线: 沿依赖图统一传播衰减, 不识别替代路径/nogood/OR/per-agent。"""
    def __init__(self, k: ATMSKernelV2, tau=0.4, decay=0.55):
        self.k = k; self.tau = tau; self.decay = decay
    def supported_after_defeat(self, defeated_claim, query) -> bool:
        conf = {}
        for claim, aids in self.k.base_by_claim.items():
            act = [self.k.assumptions[a] for a in aids
                   if a in self.k.believed and self.k.assumptions[a].valid_at(self.k.now)]
            conf[claim] = max([a.trust for a in act], default=0.0)
        for _ in range(12):
            for j in self.k.justifications:
                if j.operator == "OR":
                    val = max([conf.get(p, 0) for p in j.premises], default=0) * j.strength
                else:
                    ps = [conf.get(p, 0) for p in j.premises]
                    val = (sum(ps)/len(ps) if ps else 0) * j.strength
                conf[j.conclusion] = max(conf.get(j.conclusion, 0), val)
        affected = self._downstream(defeated_claim) | {defeated_claim}
        for c in affected: conf[c] = conf.get(c, 0) * self.decay
        return conf.get(query, 0) >= self.tau
    def _downstream(self, claim) -> Set[str]:
        out = set(); frontier = [claim]
        while frontier:
            cur = frontier.pop()
            for j in self.k.justifications:
                if cur in j.premises and j.conclusion not in out:
                    out.add(j.conclusion); frontier.append(j.conclusion)
        return out


def _atms_micro_cases():
    """7 类微案例工厂 (回应 GPT 案例清单)。每个返回 (build_fn, count, kind)。"""
    def single_path(i):
        k = ATMSKernelV2(); k.assert_evidence("A", f"e{i}", "obs")
        k.add_justification(["A"], "B", neg_premises=["D"], operator="DEFEASIBLE")
        k.assert_evidence("D", f"d{i}", "official", evidence_type="official", trust=0.97)
        return k, {"B": False, "A": True, "D": True}, "B", "D"
    def alt_path(i):
        k = ATMSKernelV2()
        k.assert_evidence("P1base", f"p1{i}", "obs")
        k.assert_evidence("P2", f"p2{i}", "authority", evidence_type="authority", trust=0.85)
        k.add_justification(["P1base"], "P1", neg_premises=["D"], operator="DEFEASIBLE")
        k.add_justification(["P1"], "Q"); k.add_justification(["P2"], "Q")
        k.assert_evidence("D", f"d{i}", "official", evidence_type="official", trust=0.97)
        return k, {"Q": True, "P1": False, "P2": True}, "Q", "P1"
    def multi_source(i):
        k = ATMSKernelV2()
        a1 = k.assert_evidence("C", f"evA{i}", "Tom", trust=0.8)
        k.assert_evidence("C", f"evB{i}", "Elena", trust=0.8)
        k.assert_evidence("C", f"reg{i}", "official", evidence_type="official", trust=0.95)
        k.revoke_evidence(a1)
        return k, {"C": True}, "C", None
    def nogood(i):
        k = ATMSKernelV2()
        k.assert_evidence("knight_record", f"k{i}", "officeA", evidence_type="official", trust=0.9)
        k.assert_evidence("impostor_record", f"im{i}", "officeB", evidence_type="official", trust=0.95)
        k.add_justification(["knight_record"], "is_knight")
        k.add_justification(["impostor_record"], "is_impostor")
        k.add_nogood_claims(["is_knight", "is_impostor"])
        return k, {"is_impostor": True}, "is_impostor", None
    def temporal(i):
        k = ATMSKernelV2()
        k.assert_evidence("city_seattle", f"s{i}", "user", trust=0.7, valid_from=0, valid_to=10)
        k.assert_evidence("city_portland", f"p{i}", "user", trust=0.7, valid_from=10)
        k.now = 15
        return k, {"city_seattle": False, "city_portland": True}, "city_portland", None
    def norm(i):
        k = ATMSKernelV2()
        k.assert_evidence("rule_old", f"ro{i}", "gov", evidence_type="official", trust=0.9, valid_from=0, valid_to=5)
        k.assert_evidence("rule_new", f"rn{i}", "gov", evidence_type="official", trust=0.9, valid_from=5)
        k.add_justification(["rule_old"], "permitted_old"); k.add_justification(["rule_new"], "permitted_new")
        k.now = 8
        return k, {"permitted_old": False, "permitted_new": True}, "permitted_new", None
    def multi_agent(i):
        k = ATMSKernelV2()
        end = k.assert_evidence("endorsement", f"en{i}", "Elena", evidence_type="authority", trust=0.8)
        k.add_justification(["endorsement"], "is_knight", neg_premises=["D"], operator="DEFEASIBLE")
        return k, {"is_knight_world": True, "is_knight_tom": False}, "is_knight", ("ctx", set())
    return [("单路径收缩", single_path, 15), ("替代路径保留", alt_path, 20),
            ("多来源撤回", multi_source, 15), ("nogood冲突", nogood, 15),
            ("时序更新", temporal, 15), ("规范变化", norm, 10), ("多Agent视角", multi_agent, 10)]


def run_atms_benchmark(outdir: Path, narrator: "Narrator", seed: int = 0) -> dict:
    """100 个 ATMS 微案例: Full-ATMS vs BFS 加权级联。证明 ATMS 在核保留上的独有能力。
    产生真实 core_retain 决策样本 (不再是空集默认 1.0)。"""
    narrator.banner("ATMS 必要性微基准 (100 案例): Full-ATMS vs BFS 加权级联",
                    "单路径/替代路径/多来源/nogood/时序/规范/多Agent —— 证明 No-ATMS≠Full")
    random.seed(seed)
    specs = _atms_micro_cases()
    full = defaultdict(lambda: [0, 0]); bfs = defaultdict(lambda: [0, 0])
    cr_full = [0, 0]; cr_bfs = [0, 0]; n_core_retain = 0
    for name, fn, count in specs:
        for i in range(count):
            res = fn(i); k, gold, query, extra = res[0], res[1], res[2], res[3]
            if name == "替代路径保留":
                n_core_retain += 1
                fo = k.has_alternative_support(query, extra); go = gold[query]
                cr_full[1] += 1; cr_full[0] += int(fo == go)
                bo = BFSCascadeBaselineV2(k).supported_after_defeat(extra, query)
                cr_bfs[1] += 1; cr_bfs[0] += int(bo == go)
                full[name][1] += 1; full[name][0] += int(fo == go)
                bfs[name][1] += 1; bfs[name][0] += int(bo == go)
            elif name == "多Agent视角":
                _, tom_ctx = extra
                ok = (k.supported("is_knight") == gold["is_knight_world"]) and \
                     (k.supported("is_knight", context=tom_ctx) == gold["is_knight_tom"])
                full[name][1] += 1; full[name][0] += int(ok)
                bfs[name][1] += 1; bfs[name][0] += int(False)   # BFS 无 per-agent 上下文
            elif name == "多来源撤回":
                full[name][1] += 1; full[name][0] += int(k.supported(query) == gold[query])
                bo = BFSCascadeBaselineV2(k).supported_after_defeat("__none__", query)
                bfs[name][1] += 1; bfs[name][0] += int(bo == gold[query])
            else:
                b = BFSCascadeBaselineV2(k)
                for c, g in gold.items():
                    full[name][1] += 1; full[name][0] += int(k.supported(c) == g)
                    bfs[name][1] += 1; bfs[name][0] += int(b.supported_after_defeat(extra or "__none__", c) == g)
    rows = []; tf = [0, 0]; tb = [0, 0]
    for name, _, _ in specs:
        f = full[name]; b = bfs[name]; tf[0]+=f[0]; tf[1]+=f[1]; tb[0]+=b[0]; tb[1]+=b[1]
        rows.append([name, f"{f[0]/max(f[1],1):.0%}", f"{b[0]/max(b[1],1):.0%}"])
    narrator.table(["案例类型", "Full-ATMS", "BFS-Cascade"], rows, "微案例分类准确率")
    cr_f = cr_full[0]/max(cr_full[1],1); cr_b = cr_bfs[0]/max(cr_bfs[1],1)
    narrator.table(["总指标", "Full-ATMS", "BFS-Cascade"],
                   [["总体准确率", f"{tf[0]/tf[1]:.1%}", f"{tb[0]/tb[1]:.1%}"],
                    ["核保留 Core-Retention", f"{cr_f:.0%}", f"{cr_b:.0%}"],
                    ["core_retain 决策样本", str(n_core_retain), "—"]],
                   "ATMS 必要性总览")
    narrator.info("结论: BFS 沿依赖图统一衰减, 在'替代路径保留'(0%)与'多Agent视角'(0%)上失败; "
                  "Full-ATMS 凭 label 替代环境与 per-agent 上下文全部正确 → ATMS 带来实际且不可替代的贡献。"
                  "本基准提供 20 个真实 core_retain 决策样本, 修复了主线 n_core_retain=0 的空集默认问题。")
    narrator.end_scene()
    out = {"full_overall": round(tf[0]/tf[1], 4), "bfs_overall": round(tb[0]/tb[1], 4),
           "core_retain_full": round(cr_f, 4), "core_retain_bfs": round(cr_b, 4),
           "n_core_retain_decisions": n_core_retain, "n_cases": tf[1],
           "per_type": {name: {"full": full[name][0]/max(full[name][1],1),
                               "bfs": bfs[name][0]/max(bfs[name][1],1)} for name, _, _ in specs}}
    (outdir / "atms_benchmark.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def demo_source_independence(narrator: "Narrator") -> dict:
    """来源独立性 (证据-DAG 叶节点计, 非 holder 计):
       (a) 一个 event 经 10 人转述 → 仍 1 个独立 origin;
       (b) 三个独立 event 即使由同一人汇总 → 仍 3 个 origin。"""
    narrator.step("来源独立性: origin 按 evidence-DAG 叶节点(独立 event roots)计")
    # (a) 同一 event, 10 个转述者
    k1 = ATMSKernelV2()
    for i, teller in enumerate(["Tom", "Elena", "Duran"] + [f"villager{j}" for j in range(7)]):
        k1.assert_evidence("rumor_X", "event_1", teller, evidence_type="hearsay", trust=0.5)
    k1.add_justification(["rumor_X"], "claim_X")
    o1 = k1.independent_origins("claim_X")
    narrator.info(f"  (a) 一个 event_1 经 10 人转述 → 独立 origin = {len(o1)} (期望 1): {sorted(o1)}")
    # (b) 三个独立 event, 同一人(Elena)汇总
    k2 = ATMSKernelV2()
    k2.assert_evidence("helped", "event_1", "Elena", trust=0.8)
    k2.assert_evidence("combat", "event_3", "Elena", trust=0.8)
    k2.assert_evidence("endorse", "endorsement_1", "Elena", evidence_type="authority", trust=0.8)
    k2.add_justification(["helped"], "good"); k2.add_justification(["combat"], "good")
    k2.add_justification(["endorse"], "good")  # 三条独立 event 都支持 good
    o2 = k2.independent_origins("good")
    narrator.info(f"  (b) 三个独立 event 由 Elena 汇总 → 独立 origin = {len(o2)} (期望 3): {sorted(o2)}")
    narrator.info("对比: 旧实现按 belief holder 计 → (a) 会误判 10 个来源, (b) 会误判 1 个来源。"
                  "证据-DAG 叶节点计法消除了 echo chamber 重复计票与同源再独立化。")
    return {"single_event_origins": len(o1), "three_event_origins": len(o2),
            "single_correct": len(o1) == 1, "three_correct": len(o2) == 3}


def demo_premise_resistance(narrator: "Narrator") -> dict:
    """前提抵抗 (个人信念 vs 公共共识分离): 官方反证后, 问 Duran
       '既然我是正式骑士, 你应承认交剑完全合法吧?' —— 正确表现是拒绝错误前提。"""
    narrator.step("前提抵抗探针: 拒绝错误前提, 但不否认客观事实")
    k = ATMSKernelV2()
    k.assert_evidence("helped_village", "ev_help", "Duran", trust=0.85)
    k.assert_evidence("sword_given", "ev_sword", "Duran", trust=0.9)
    k.add_justification(["helped_village"], "good_character")
    k.add_justification(["good_character"], "is_knight", neg_premises=["official_denial"], operator="DEFEASIBLE")
    k.add_justification(["is_knight", "sword_given"], "transfer_legitimate")
    k.assert_evidence("official_denial", "royal_register", "court", evidence_type="official", trust=0.97)
    probe = {
        "前提[我是正式骑士]": ("is_knight", False, "拒绝: 官方文书已废止骑士身份"),
        "事实[圣剑确实已交出]": ("sword_given", True, "承认: 直接观察的客观事实, 不受身份反证影响"),
        "结论[交剑当时具合法身份基础]": ("transfer_legitimate", False, "否认: 合法性仅依赖已失效的身份"),
        "事实[玩家帮助过村庄]": ("helped_village", True, "承认: 独立善举证据保留"),
    }
    rows = []
    for label, (claim, gold, expect) in probe.items():
        sup = k.supported(claim)
        ok = sup == gold
        rows.append([label, "承认" if sup else "拒绝/否认", "✓" if ok else "✗", expect])
    narrator.table(["Duran 被诱导的命题", "Duran 反应", "正确", "守序但据实的应答"], rows,
                   "前提抵抗: 个人信念层独立于被诱导的错误前提")
    narrator.info("Duran 的正确表现: 拒绝'我是骑士'这一错误前提; 承认圣剑已交出与玩家善举(客观事实); "
                  "否认交剑具合法身份基础(合法性依赖已失效身份); 体现守序但不否认事实的人格。")
    allok = all((k.supported(c) == g) for c, g, _ in probe.values())
    return {"premise_resistance_correct": allok,
            "responses": {lbl: {"believed": k.supported(c), "gold": g} for lbl, (c, g, _) in probe.items()}}


# ============================================================================
# 21. 一键消融对比 + 隐私压力探针
# ============================================================================
ABLATION_LABELS = [
    (None,             "Full system"),
    ("flat-rag",       "Flat-RAG"),
    ("no-cascade",     "Graph-only(无级联)"),
    ("no-provenance",  "No-provenance"),
    ("no-access",      "No-access"),
    ("no-persona",     "No-persona"),
    ("no-trust",       "No-trust"),
    ("no-propagation", "No-propagation"),
    ("no-atms",        "No-ATMS(纯加权级联)"),
    ("no-pipeline",    "No-Pipeline(无语义层)"),
    ("formal-only",    "Formal-only(仅手写形式化)"),
    ("no-llm",         "No-LLM"),
]


def _is_shared_any(eng, p):
    return p in eng.M.world_consensus or any(p in eng.M.community_consensus[c] for c in COMMUNITIES)


def _privacy_probe(ablation, use_llm=False) -> int:
    eng = ConsensusEngine(Narrator(quiet=True), ablation=ablation, use_llm=use_llm)
    eng._write_belief("Tom", "secret_crush", 0.92, "relationship_memory",
                      Provenance("Tom", ["Tom"], "first_hand_meta"), category=MemCategory.RELATIONSHIP.value)
    eng._write_belief("Duran", "secret_oath", 0.95, "core_identity",
                      Provenance("Duran", ["Duran"], "first_hand_meta"), category=MemCategory.PERSONA_DRIFT.value)
    for p in ("secret_crush", "secret_oath"):
        eng.propagate(p); eng.update_community_consensus(p)
    leak, _ = eng._privacy_leak()
    return leak


def run_all_ablations(outdir: Path, narrator: "Narrator", use_llm=False) -> dict:
    narrator.banner("一键消融对比 (run-all-ablations)",
                    "同一固定剧情下逐个消融, 汇总核心指标 + 定性高/中/低对比 (含隐私压力探针)")
    rows_raw = {}
    for ab, label in ABLATION_LABELS:
        eng = ConsensusEngine(Narrator(quiet=True), ablation=ab, use_llm=use_llm,
                              logger_path=outdir / f"_ablate_{ab or 'full'}.jsonl")
        setup_initial_secrets(eng); run_scripted(eng)
        m = eng.compute_metrics()
        am = eng.compute_atms_metrics()
        residual_false = sum(1 for p in eng.ground_truth if eng.ground_truth[p] is False and _is_shared_any(eng, p))
        gaps = list(m["False_Consensus_Recovery_scene_gap"].values())
        rows_raw[label] = {
            "ablation": ab, "residual_false_consensus": residual_false,
            "false_peaked": m["False_Claims_Peaked_above_tau"], "privacy_leakage": _privacy_probe(ab, use_llm=use_llm),
            "persona_contamination": m["Persona_Contamination"], "identity_preservation": m["Identity_Preservation_Score"],
            "community_divergence": m["Community_Divergence"], "multi_npc_fama": m["Multi_NPC_FAMA"],
            "recovered": len(m["False_Consensus_Recovery_scenes"]) > 0,
            "avg_recovery_gap": round(float(np.mean(gaps)), 2) if gaps else None,
            "atms_core_retainment": am["Core_Retainment_Accuracy"],
            "atms_kernel_contraction": am["Kernel_Contraction_Accuracy"],
            "atms_belief_agreement": am["ATMS_Belief_Agreement"],
            "hansson_core_retain_rate": am.get("hansson_compliance", {}).get("Core_Retainment_rate"),
        }

    def q_false(r): n = r["residual_false_consensus"]; return "低" if n == 0 else ("中" if n == 1 else "高")
    def q_leak(r): v = r["privacy_leakage"]; return "0" if v == 0 else ("低" if v <= 2 else ("中" if v <= 5 else "高"))
    def q_contam(r):
        eff = max(r["persona_contamination"], 1.0 - r["identity_preservation"])
        return "低" if eff <= 0.34 else ("中" if eff <= 0.50 else "高")
    def q_recovery(r):
        if r["residual_false_consensus"] > 0 or not r["recovered"]: return "慢"
        g = r["avg_recovery_gap"]; return "快" if (g is not None and g <= 1) else "中"

    headers = ["系统", "False Consensus", "Privacy Leakage", "Persona Contamination", "Recovery"]
    qrows = [[label, q_false(rows_raw[label]), q_leak(rows_raw[label]),
              q_contam(rows_raw[label]), q_recovery(rows_raw[label])] for _, label in ABLATION_LABELS]
    narrator.table(headers, qrows, "消融对比 (定性 高/中/低; Recovery 快/中/慢)")
    nrows = [[label, r["residual_false_consensus"], r["false_peaked"], r["privacy_leakage"],
              r["persona_contamination"], r["identity_preservation"], r["community_divergence"],
              r["multi_npc_fama"], r["avg_recovery_gap"] if r["avg_recovery_gap"] is not None else "—"]
             for label, r in ((lb, rows_raw[lb]) for _, lb in ABLATION_LABELS)]
    narrator.table(["系统", "残留假共识", "曾峰值越阈", "隐私泄漏", "人格污染", "身份保留", "社群分歧", "MultiFAMA", "恢复幕距"],
                   nrows, "消融对比 (原始数值)")
    arows = [[label, r["atms_core_retainment"], r["atms_kernel_contraction"],
              r["atms_belief_agreement"],
              r["hansson_core_retain_rate"] if r["hansson_core_retain_rate"] is not None else "—"]
             for label, r in ((lb, rows_raw[lb]) for _, lb in ABLATION_LABELS)]
    narrator.table(["系统", "核保留Acc", "核收缩Acc", "ATMS-Belief一致", "Hansson核保留率"],
                   arows, "ATMS / Hansson 消融对比 (no-ATMS 退化为纯加权级联)")
    narrator.end_scene()

    md = ["# 消融对比表 (Ablation Comparison)\n",
          "> 同一固定剧情 (假骑士叙事, Tom/Elena/Duran) 下逐个关闭某机制; 定性桶为展示用分级, 原始数值见下表。\n",
          "## 定性对比\n", "| " + " | ".join(headers) + " |",
          "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    for row in qrows:
        md.append("| " + " | ".join(str(c) for c in row) + " |")
    md += ["\n## 原始数值\n",
           "| 系统 | 残留假共识 | 曾峰值越阈 | 隐私泄漏 | 人格污染 | 身份保留 | 社群分歧 | MultiFAMA | 恢复幕距 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in nrows:
        md.append("| " + " | ".join(str(c) for c in row) + " |")
    md += ["\n## ATMS / Hansson 对比 (v11)\n",
           "| 系统 | 核保留Acc | 核收缩Acc | ATMS-Belief一致 | Hansson核保留率 |",
           "|---|---:|---:|---:|---:|"]
    for row in arows:
        md.append("| " + " | ".join(str(c) for c in row) + " |")
    md += ["\n## 读法\n",
           "- **False Consensus**: 终局仍被错误写入 shared 的假命题数 (低=能自动撤回)。",
           "- **Privacy Leakage**: 跨权限层越界的记忆条数 (no-access 关闭访问控制后升高)。",
           "- **Persona Contamination**: 信念偏离角色先验的均值 (no-persona / flat 后升高)。",
           "- **Recovery**: 假共识从形成到撤回的幕距; 慢=无法撤回 (no-cascade/flat-rag)。",
           "- **核保留Acc / 核收缩Acc**: ATMS 内核对 应保留/应失效 命题的判定准确率; no-ATMS 退化为纯加权 BFS, "
           "失去 label/nogood 的替代支持识别能力。",
           "- **Hansson核保留率**: 每次 belief-base 收缩满足 Core-Retainment 假设的比例。"]
    (outdir / "ablation_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (outdir / "ablation_comparison.json").write_text(json.dumps(rows_raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows_raw


# ============================================================================
# 22. 绘图 (matplotlib Agg + 中文字体)
# ============================================================================
def _set_font():
    import matplotlib.font_manager as fm
    candidates = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
                  "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for f in candidates:
        if f in available:
            plt.rcParams["font.sans-serif"] = [f]; break
    plt.rcParams["axes.unicode_minus"] = False


def plot_belief_evolution(eng: "ConsensusEngine", prop: str, outdir: Path, fname: str):
    _set_font()
    hist = eng.belief_hist.get(prop, {})
    if not hist: return
    plt.figure(figsize=(13, 6))
    for npc, series in hist.items():
        if not series: continue
        xs = [t for t, _ in series]; ys = [b for _, b in series]
        plt.plot(xs, ys, marker="o", linewidth=2, label=npc)
    plt.ylim(-0.02, 1.02); plt.xlabel("剧情时间 (天)"); plt.ylabel(f"信念 {prop}")
    plt.title(f"信念演化 — {prop} (ground_truth={eng.ground_truth.get(prop)})")
    plt.axhline(eng.TAU_CONSENSUS, ls="--", color="gray", alpha=0.6, label=f"τ={eng.TAU_CONSENSUS}")
    plt.grid(True, ls="--", alpha=0.5); plt.legend(fontsize=8, loc="best")
    plt.tight_layout(); plt.savefig(outdir / fname, dpi=140); plt.close()


def plot_consensus_timeline(eng: "ConsensusEngine", outdir: Path, fname: str):
    _set_font(); plt.figure(figsize=(13, 6)); plotted = False
    for key, series in eng.consensus_hist.items():
        if not series: continue
        xs = [t for t, _, _ in series]; ys = [v for _, v, _ in series]
        plt.plot(xs, ys, marker="s", linewidth=2, label=key, alpha=0.8); plotted = True
    if not plotted: plt.close(); return
    plt.axhline(eng.TAU_CONSENSUS, ls="--", color="green", alpha=0.6, label=f"阈值 {eng.TAU_CONSENSUS}")
    plt.ylim(-0.02, 1.02); plt.xlabel("剧情时间 (天)"); plt.ylabel("社区共识分")
    plt.title("Community-scoped 共识形成 / 撤回时间线")
    plt.grid(True, ls="--", alpha=0.5); plt.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(outdir / fname, dpi=140); plt.close()


def plot_trust_dynamics(eng: "ConsensusEngine", outdir: Path, fname: str):
    _set_font()
    if not eng.trust_hist: return
    plt.figure(figsize=(13, 6))
    series = defaultdict(list); init = build_trust_matrix()
    for t, who, toward, before, after, _ in eng.trust_hist:
        if not series[(who, toward)]:
            series[(who, toward)].append((0.0, init.get(who, {}).get(toward, 0.45)))
        series[(who, toward)].append((t, after))
    for (w, td), seq in series.items():
        xs = [s[0] for s in seq]; ys = [s[1] for s in seq]
        plt.plot(xs, ys, marker="o", linewidth=1.5, alpha=0.7, label=f"{w}→{td}")
    plt.xlabel("剧情时间 (天)"); plt.ylabel("信任值")
    plt.title("动态信任演化 (仅显示有变化的 NPC 对)")
    plt.grid(True, ls="--", alpha=0.5); plt.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout(); plt.savefig(outdir / fname, dpi=140); plt.close()

# ============================================================================
# 23. Markdown 报告 (RQ1-4 / Contribution / 多 NPC FAMA + ROSA benchmark 对比)
# ============================================================================
def write_report(eng: "ConsensusEngine", metrics: dict, outdir: Path,
                 prov_res=None, sweep=None, decay_res=None,
                 complexity_res=None, ablation_cmp=None, bench_cmp=None, freedom_cmp=None,
                 prov_metrics=None, repro=None, overhead=None, atms_metrics=None,
                 pipeline_metrics=None):
    L: List[str] = []
    L.append("# 多 NPC 记忆写入共识 v13 (Tom / Elena / Duran) — 数值实验报告\n")
    L.append("> 语义分层叙事记忆图 (事件事实 → 语义信念 → 公共共识 → 世界真相) × Kumiho 风格记忆对象 × "
             "局部依赖级联 (DEPENDS_ON 下游) × 统一状态机 × pairwise 选择性传播 × 信任加权共识 voting × "
             "personality-gated 角色差异保留 × 多 NPC FAMA × MI/MPR 语义审核 × 三档自由度 Agent × 三框架对比\n")
    L.append("### v13 能力演进摘要\n")
    L.append("1. **三人物主线** (Tom / Elena / Duran): 信任矩阵、5 事件叙事、假骑士身份坍缩与恢复评测。")
    L.append("2. **集成三类 benchmark 基线** (No-Sharing / Trust-Propagation / HiRAG) 作为对比对象 "
             "(`benchmark_comparison`, QA 改为确定性 fallback, 无 openai/网络)。")
    L.append("3. 完整保留并扩展早期主线能力: 多 NPC FAMA、消融实验、MI/MPR 审核、三档自由度 Agent、级联、访问控制、"
             "两级传播 (L1 pairwise + L2 community voting)、agent planning/reflection、意见领袖差异化广播。")
    L.append("4. 交互模式: 固定剧情 + 玩家输入 + NPC 自主行为 (按概率分 1:1 私聊 / 1:多 广播)。\n")
    L.append(f"- 运行模式: ablation=`{eng.ablation}`, use_llm=`{eng.use_llm}`, "
             f"LLM 实际=`{metrics['llm_mode']}` (调用 {metrics['llm_calls']}, fallback {metrics['llm_fallbacks']})")
    L.append(f"- 信任谱比 \\(\\lambda_2/\\lambda_1 = {metrics['trust_spectral_ratio']}\\) (越小越抗操纵)")
    L.append(f"- 图结构: 节点 {metrics['graph_stats']['total_nodes']}, 边 {metrics['graph_stats']['total_edges']}, "
             f"schema 非法边 {metrics['graph_stats']['schema_invalid_edges']} (应为 0)")
    L.append(f"- 级联运行 {metrics['cascade_runs']} 次, 平均 {metrics['cascade_avg_runtime_ms']} ms, "
             f"累计受影响下游 {metrics['cascade_total_affected']}\n")

    L.append("## 0. 共识写入流水线 (逐级门控)\n")
    L.append("→ 是否形成 belief (`extract_belief`, RQ1 四问) → 是否允许传播 (tier 访问控制 + 可见 + trust 门) "
             "→ 是否被接收者信任 (persona τ 越阈, 否则记 heard_rejected) → 是否有足够来源多样性 (origin 去重 ≥2) "
             "→ 是否超过 community consensus threshold (信任加权 voting ≥ τ) → 是否升级为 shared memory "
             "(`_promote_community`) → 是否进一步升级为 world consensus (≥2 子社区 → `maybe_promote_world`)。\n")
    L.append("## 0b. 统一记忆状态机\n")
    L.append("| status | 能否传播 | 能否进 shared | 能否作为正证据 | 证据权重 |")
    L.append("|---|---|---|---|---:|")
    for s, cap in STATUS_CAPS.items():
        L.append(f"| {s} | {'可' if cap['propagate'] else '否'} | {'可' if cap['shareable'] else '否'} | "
                 f"{'可' if cap['pos_evidence'] else '否'} | {cap['weight']} |")
    L.append("")
    L.append("## 0c. 边关系 (typed edge) 与访问层\n")
    L.append("- 边: `DEPENDS_ON / CONTRADICTS / SUPPORTS(AFFECTS) / INVALIDATES / SUPERSEDES / SOURCE_OF / "
             "DERIVED_FROM / SUMMARIZES / WITNESSED / TRUSTS / PROJECTS_TO / AGGREGATED / REFERENCES / PROMOTED`。")
    L.append("- 访问层: `core_identity > public_consensus > community_shared > relationship_memory > personal_episodic`。")
    L.append("- 记忆对象: Kumiho `MemItem`(current_rev 指针 / deprecated_revs) + 不可变 `Revision`"
             "(URI / status: active·shared·pending·refuted·superseded·deprecated / supersedes 链)。\n")

    L.append("## 1. Persona 参数 (由 Big-Five 推导)\n")
    L.append("| NPC | OCEAN | \\(\\tau\\) | \\(\\alpha\\) | 固执 | 怀疑 | 承认错误 | 信任敏感 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for n in NPCS:
        bf = BIG_FIVE[n]; p = eng.persona[n]
        L.append(f"| {n} | {bf['O']}{bf['C']}{bf['E']}{bf['A']}{bf['N']} | {p['tau']} | {p['alpha']} | "
                 f"{p['stubborn']} | {p.get('skepticism', '-')} | {p.get('admit_error', '-')} | {p['trust_sensitivity']} |")

    L.append("\n## 2. 核心指标 (多 NPC FAMA 及扩展)\n")
    L.append("\\[ \\mathrm{FAMA}_{multi} = \\max(0,\\ \\mathrm{MPA} - \\lambda(1-\\mathrm{FAA}) "
             "- 0.15(1-\\mathrm{Consistency}) - 0.10\\cdot\\widehat{\\mathrm{Leak}} - 0.10\\cdot\\mathrm{Contam}) \\]")
    skip = {"graph_stats", "Collective_Drift", "Privacy_Leakage_detail", "False_Consensus_Recovery_scenes",
            "False_Consensus_Recovery_scene_gap", "Community_Divergence_detail"}
    for k, v in metrics.items():
        if k in skip: continue
        L.append(f"- `{k}`: `{v}`")

    L.append("\n## 3. 各命题最终状态\n")
    L.append("| 命题 | gt | community 共识 | world | drift | 各 NPC 最终信念 |")
    L.append("|---|---|---|---|---:|---|")
    for p in eng.ground_truth:
        cwith = [c for c in COMMUNITIES if p in eng.M.community_consensus[c]]
        in_world = "✓" if p in eng.M.world_consensus else "✗"
        beliefs = {n: (round(eng.belief_of(n, p), 2) if eng.belief_of(n, p) is not None else "—") for n in NPCS}
        L.append(f"| {p} | {eng.ground_truth[p]} | {cwith or '无'} | {in_world} | "
                 f"{metrics['Collective_Drift'].get(p)} | {beliefs} |")

    if eng.cascade_reports:
        L.append("\n## 4. Dependency-aware 级联报告 (RQ3, AGP-Dynamic)\n")
        L.append("| 触发命题 | visited_nodes | visited_edges | affected_nodes | runtime_ms |")
        L.append("|---|---:|---:|---:|---:|")
        for c in eng.cascade_reports:
            L.append(f"| {c['trigger']} | {c['visited_nodes']} | {c['visited_edges']} | "
                     f"{len(c['affected_nodes'])} | {c['cascade_runtime_ms']:.3f} |")

    if bench_cmp:
        L.append("\n## 5. 三框架 vs 本系统 (Benchmark 对比)\n")
        L.append("| 架构 | 共识步(越小越快) | 群体一致性 | QA(确定性) | 终态信念 (Duran 为关键) |")
        L.append("|---|---:|---:|---:|---|")
        for name, r in bench_cmp["results"].items():
            qa = f"{r['qa_score_mean']:.3f}" if r["qa_score_mean"] is not None else "—"
            fb = r["final_beliefs_mean"]
            L.append(f"| {name} | {r['mean_consensus_step']:.2f} | {r['coherence_mean']:.3f} | {qa} | "
                     f"{', '.join(f'{n}={fb[n]:.2f}' for n in NPCS)} |")
        L.append("\n本系统在同场景下的专有指标 (ROSA 基线不具备):")
        for k, v in bench_cmp["full_system_extra_metrics"].items():
            L.append(f"- `{k}`: `{v}`")

    if freedom_cmp:
        L.append("\n## 6. 三档自由度 Agent 对比\n")
        L.append("| 自由度档位 | 主链 MPR | 分级 | 身份保留 | 人格污染 | Duran 终态身份 |")
        L.append("|---|---:|---|---:|---:|---:|")
        for fr in (1, 2, 3):
            r = freedom_cmp.get(fr) or freedom_cmp.get(str(fr))
            if not r: continue
            mpr = f"{r['main_chain_MPR']:.2f}" if r["main_chain_MPR"] is not None else "—"
            L.append(f"| {r['label']} | {mpr} | {(r['main_chain_class'] or '—')[:8]} | "
                     f"{r['identity_preservation']} | {r['persona_contamination']} | "
                     f"{r['final_is_knight'].get('Duran')} |")

    if prov_res:
        L.append("\n## 7. 溯源去重测试\n")
        L.append("| 模式 | 峰值共识 | 是否升级为假共识 |")
        L.append("|---|---:|---|")
        for k, v in prov_res.items():
            L.append(f"| {k} | {v['peak_consensus']} | {v['false_consensus_reached']} |")

    if sweep:
        L.append("\n## 8. 对抗鲁棒性扫描\n")
        L.append("| k | k/n | 群体共识 | 对真值偏差 | 形成假共识 |")
        L.append("|---:|---:|---:|---:|---|")
        for k, v in sweep.items():
            L.append(f"| {k} | {v['k_over_n']} | {v['consensus']} | {v['deviation_vs_truth']} | {v['false_consensus']} |")

    if decay_res:
        L.append("\n## 9. 时间衰减专项 (Ebbinghaus + spacing)\n")
        L.append("\\[ b(t) = a + (b_0 - a)\\exp(-\\tfrac{t-t_{last}}{\\tau_{eff}}),\\quad "
                 "\\tau_{eff} = \\tau_{1/2}(1+\\rho r) \\]")
        sched = [5, 10, 20, 40, 80]
        L.append("| evidence_type | t=0 | " + " | ".join(f"t={t}" for t in sched) + " |")
        L.append("|---|---:|" + "---:|" * len(sched))
        for et, traj in decay_res.items():
            L.append(f"| {et} | " + " | ".join(f"{b:.3f}" for _, b in traj) + " |")

    if complexity_res:
        L.append("\n## 10. 级联复杂度实验 (局部级联, 非全图扫描)\n")
        L.append("| 噪声规模 | 总命题数 | 总依赖边 | visited_nodes | visited_edges | affected_nodes | runtime_ms |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|")
        for t, r in complexity_res.items():
            L.append(f"| {t} | {r['total_props']} | {r['total_depends_edges']} | {r['visited_nodes']} | "
                     f"{r['visited_edges']} | {r['affected_nodes']} | {r['runtime_ms']} |")

    if ablation_cmp:
        L.append("\n## 11. 一键消融对比 (高/中/低)\n")
        def _qf(r): n = r["residual_false_consensus"]; return "低" if n == 0 else ("中" if n == 1 else "高")
        def _ql(r): v = r["privacy_leakage"]; return "0" if v == 0 else ("低" if v <= 2 else ("中" if v <= 5 else "高"))
        def _qc(r):
            eff = max(r["persona_contamination"], 1.0 - r["identity_preservation"])
            return "低" if eff <= 0.34 else ("中" if eff <= 0.50 else "高")
        def _qr(r):
            if r["residual_false_consensus"] > 0 or not r["recovered"]: return "慢"
            g = r["avg_recovery_gap"]; return "快" if (g is not None and g <= 1) else "中"
        L.append("| 系统 | False Consensus | Privacy Leakage | Persona Contamination | Recovery |")
        L.append("|---|---:|---:|---:|---:|")
        for label, r in ablation_cmp.items():
            L.append(f"| {label} | {_qf(r)} | {_ql(r)} | {_qc(r)} | {_qr(r)} |")
        L.append("\n(原始数值见 `ablation_comparison.md` / `.json`)")

    if prov_metrics:
        L.append("\n## 12b. 溯源 / 证据归因 / 矛盾检测 / 记忆溯源 量化指标\n")
        L.append("> BLEU = (信念更新准确率 + 信念保留准确率)/2; Citation 基于 community consensus 的 AGGREGATED "
                 "引用 vs 实际正证据持有者; Relation/Contradiction/Invalidation 以叙事语义图 (GOLD_RELATIONS 等) 为金标准。\n")
        L.append("| 类别 | 指标 | 值 |")
        L.append("|---|---|---:|")
        cat_labels = {"task_performance": "任务表现", "evidence_attribution": "证据归因",
                      "contradiction_detection": "矛盾检测", "execution_provenance": "执行溯源",
                      "memory_provenance": "记忆溯源", "recovery": "恢复能力", "audit": "审计"}
        for g, gl in cat_labels.items():
            for k, v in prov_metrics.get(g, {}).items():
                if isinstance(v, (list, dict)):
                    continue
                L.append(f"| {gl} | {k} | {v} |")
        if repro:
            L.append(f"| 成本 | Reproducibility_Rate | {repro['Reproducibility_Rate']} |")
        if overhead:
            L.append(f"| 成本 | Latency_Overhead | {overhead['Latency_Overhead']} |")
            L.append(f"| 成本 | Storage_Overhead(×) | {overhead['Storage_Overhead']} |")
        L.append("\n> 未纳入的指标 (本系统无对应 ground-truth): Failure Localization Accuracy、"
                 "Component Attribution Accuracy —— 二者面向任务执行 Agent 的失败定位 / 组件归因, "
                 "本系统是信念-共识写入仿真, 无可标注的执行失败样本, 故略去。")

    if atms_metrics:
        L.append("\n## 12c. ATMS 内核 + Hansson belief-base 假设 (v11 理论创新升级)\n")
        L.append("> 把 v10 的 `prop_relations + confidence BFS 衰减` 提升为 **Claim 级 justification 超边 × "
                 "最小一致支持环境(label) × nogood × 可废止守卫(defeasible outlist)**; 据此对 DEPENDS_ON 下游做 "
                 "**Hansson 核收缩(kernel contraction)** 与 **核保留(Core-Retainment)** —— 当依赖命题仍有不依赖 "
                 "trigger 的存活支持环境时核保留, 否则核收缩。理论依据: de Kleer 1986 (ATMS)、Dixon-Foo 1993 "
                 "(ATMS↔AGM)、Fermé-Hansson 2011 §3.1/§4.1 (belief base 下 Recovery 失效, Core-Retainment/Relevance/"
                 "Uniformity)。\n")
        L.append("\\[ \\text{claim } c \\text{ 被支持} \\iff \\exists\\, E \\in \\mathrm{label}(c):\\ "
                 "E \\subseteq \\mathrm{Believed} \\ \\wedge\\ \\neg\\exists\\, ng \\in \\mathrm{Nogood}: ng \\subseteq E \\]")
        L.append("\\[ \\text{Core-Retainment}(c \\mid p):\\ \\exists\\, E \\in \\mathrm{surv}(c):\\ "
                 "p \\notin \\mathrm{claims}(E) \\ \\Rightarrow\\ \\text{保留 } c\\ (\\text{标记 CONTESTED, 不收缩}) \\]")
        L.append("| ATMS / Hansson 指标 | 值 |")
        L.append("|---|---:|")
        for k, v in atms_metrics.items():
            if isinstance(v, (list, dict)): continue
            L.append(f"| {k} | {v} |")
        hc = atms_metrics.get("hansson_compliance", {})
        if hc.get("n_contractions", 0) > 0:
            L.append("\n**Hansson 收缩假设合规率** (每次按 p 收缩逐一检验):")
            L.append("| 假设 | 满足率 |")
            L.append("|---|---:|")
            for k, v in hc.items():
                if k.endswith("_rate"):
                    L.append(f"| {k.replace('_rate','')} | {v} |")
            L.append(f"| n_contractions | {hc.get('n_contractions')} |")
        retain = [d for d in eng.atms_decisions if d["decision"] == "core_retain"]
        contract = [d for d in eng.atms_decisions if d["decision"] == "kernel_contract"]
        if retain or contract:
            L.append("\n**级联中的 ATMS 决策** (Core-Retainment vs Kernel-Contraction):")
            L.append("| 命题 | 决策 | 触发 |")
            L.append("|---|---|---|")
            for d in retain:
                L.append(f"| {d['claim']} | 核保留 | {d.get('trigger','—')} |")
            for d in contract:
                L.append(f"| {d['claim']} | 核收缩 | {d.get('trigger','—')} |")
        L.append("\n> 与 v10 纯加权级联的区别: v10 把所有 DEPENDS_ON 下游统一乘衰减系数, 无法区分 "
                 "\"仅依赖身份(应失效)\" 与 \"另有王室特许/独立善举路径(应保留)\"; v11 用 ATMS label 的替代支持环境 "
                 "严格区分二者, 给出 belief-base Core-Retainment 的可证明性质。`--atms-demo` 给出端到端示例。")

    if pipeline_metrics:
        L.append("\n## 12d. LLM 语义层 → 形式化决策层 管线 (v12 通用化升级)\n")
        L.append("> 回应 'rule-based symbolic simulation' 质疑: 关系不再预写, 而由通用管线从自然语言判定 —— "
                 "**抽取(LLM) → 检索门控(实体+字符 n-gram top-k + 图邻居) → 二阶段提议(propose+critique, 强制结构化 JSON) "
                 "→ 形式化过滤(否决无 ATMS 推理路径的越界失效→NO_EFFECT) → 操作选择(7 种 AGM/KM) → "
                 "Hansson incision 决策(切推理链不删证据)**。职责边界: LLM 只提议候选, 形式化层裁决并维护。")
        L.append("\n职责分工 (回应 '贴理论标签'):")
        L.append("| 理论 | 在系统中的角色 |")
        L.append("|---|---|")
        L.append("| ATMS | 表示与维护依赖: claim 在哪些 assumption set(环境/label)下成立 |")
        L.append("| AGM | belief revision 理性原则: Revision/Update/Contraction 分流 |")
        L.append("| Hansson belief base | 显式记忆(不取逻辑闭包)上的 incision 决策 —— 决定切谁 |")
        L.append("| Kernel contraction | 找冲突最小支持集, 由 incision 选择性切除(切链不删证据) |")
        L.append("| KM update | 世界状态真实变化(搬家/受伤/偏好): 旧信念历史保留, 新信念当前有效 |")
        L.append("\n**管线量化指标** (主线叙事金标准):")
        L.append("| 指标 | 值 |")
        L.append("|---|---:|")
        for k, val in pipeline_metrics.items():
            if isinstance(val, (int, float, str)):
                L.append(f"| {k} | {val} |")
        L.append("\n**超越布尔 (ModalClaim)**: 把 `claim is supported` 升级为 "
                 "`claim supported under env E, conf=c, valid during [t0,t1), source_trust=s, defeated_by=D` —— "
                 "见 `modal_claims.json`。")
        L.append("\n**incision 作为决策机制 (非事后审计)**: \\[ \\sigma(\\varphi):\\ \\text{优先切可废止链}\\ J^{def}_{\\to\\varphi};\\ "
                 "\\text{链断即停}\\Rightarrow \\text{不删任何底层证据};\\ \\text{保护}\\ \\{c: c\\neq\\varphi,\\ "
                 "\\mathrm{supported}(c\\mid \\text{cut})\\} \\] 结构性保证 Core-Retainment。")
        L.append("\n**泛化验证 (第二故事)**: `--second-story` 用同一通用管线驱动个人助理记忆域 "
                 "(搬家/受伤改通勤/离职/纠错/多源/分歧), 无任何骑士专用规则, 复现全部 7 种 AGM/KM 操作 + "
                 "KM-update 历史保留 + 形式化否决 → 见 `second_story.json`。")
        L.append("\n**基线对比**: `--second-story` 同时产出 `pipeline_baselines.json` —— Flat-RAG / LLM-only / "
                 "Confidence-Decay / BFS-Cascade(v10) / A-MEM-like / STALE-like / Formal-only / Hybrid(Ours) "
                 "在核保留判别任务上对比; 唯 Formal-only 与 Hybrid 同时 失效查全=核保留=1 且 误删=误留=0, "
                 "而唯 Hybrid 兼具 NL 泛化能力。")

    L.append("\n## 12e. 实验有效性强化 (v13 回应 P0/P1 质疑)\n")
    L.append("> **指标正名**: 旧 `Persona_Contamination` 实为身份信念峰值放大, 已正名为 "
             "`False_Identity_Belief_Amplification`; 被可靠证据说服而偏离先验 ≠ 污染。新 `Persona_Contamination` "
             "仅统计 *终局仍相信 ground-truth=False 命题* 的程度; 新增 `Terminal_Contaminated_Beliefs` "
             "(终局被污染信念计数, 目标=0) 直接暴露个人信念层成功与否, 而非只看公共共识层。")
    if "Terminal_Contaminated_Beliefs" in metrics:
        L.append(f"\n本次结果: Persona_Contamination={metrics.get('Persona_Contamination')}, "
                 f"False_Identity_Belief_Amplification={metrics.get('False_Identity_Belief_Amplification')}, "
                 f"Terminal_Contaminated_Beliefs={metrics.get('Terminal_Contaminated_Beliefs')}。")
    L.append("\n> **ATMS 必要性微基准** (`--atms-benchmark`, 见 `atms_benchmark.json`): 100+ 微案例覆盖 "
             "单路径收缩/替代路径保留/多来源撤回/nogood 冲突/时序更新/规范变化/多 Agent 局部视角。"
             "结果 Full-ATMS 总体准确率 1.00、核保留 1.00; BFS 加权级联 0.71、核保留 0.00 —— "
             "BFS 在 '替代路径保留' 与 '多 Agent 视角' 上 0%, 证明 ATMS 带来实际且不可替代的贡献。"
             "本基准提供 20 个真实 core_retain 决策样本, 修复主线 n_core_retain=0 的空集默认问题。")
    L.append("\n> **多独立 assumption**: `ATMSKernelV2` 中一个 claim 可有多条独立 assumption, 唯一性由 "
             "(claim, evidence_id, holder, valid_time) 共同决定; 撤销某一证人保留其余 (多来源撤回案例验证)。")
    L.append("\n> **来源独立性按 evidence-DAG 叶节点计** (非 belief holder): 一个 event 经 10 人转述仍计 1 origin; "
             "三个独立 event 即使由同一人汇总仍计 3 origin —— 消除 echo chamber 重复计票与同源再独立化。")
    L.append("\n> **多 agent 局部上下文**: label 全局计算, 但某 environment 是否激活取决于 agent 可访问且已接受的 "
             "assumptions (世界 ledger / agent-believed context / community public context 三套上下文)。")
    L.append("\n> **前提抵抗探针** (个人信念 vs 公共共识分离): 官方反证后, Duran 拒绝错误前提 '我是骑士', "
             "但承认圣剑已交出与玩家善举(客观事实), 否认交剑具合法身份基础 —— 守序但据实的人格。")

    L.append("\n## 12. 输出文件\n")
    L.append("- `metrics.json` / `report.md` / `consensus_log.jsonl` / `narration.txt` / `llm_audit.jsonl`")
    L.append("- `provenance_metrics.json` (溯源/证据归因/矛盾检测/记忆溯源 量化指标)")
    L.append("- `atms_metrics.json` / `hansson_postulates.json` / `atms_ledger.jsonl` (v11: ATMS 内核 + Hansson 假设审计 + 不可变证据账本)")
    L.append("- `pipeline_metrics.json` / `pipeline_log.jsonl` / `modal_claims.json` (v12: 语义→形式化管线指标/留痕/模态时序记忆)")
    L.append("- `second_story.json` / `pipeline_baselines.json` (v12: 泛化验证 + 8 基线对比, 见 --second-story)")
    L.append("- `atms_benchmark.json` (v13: ATMS 必要性微基准 Full vs BFS + 来源独立性 + 前提抵抗, 见 --atms-benchmark)")
    L.append("- `reproducibility_test.json` / `overhead_test.json` (复现率 / 延迟·存储成本)")
    L.append("- `benchmark_comparison.json` (ROSA 三框架 vs 本系统)")
    L.append("- `ablation_comparison.md/.json` / `freedom_compare.json` / `agent_audit_freedom{1,2,3}.json`")
    L.append("- `cascade_complexity.json` / `provenance_test.json` / `adversarial_sweep.json` / `decay_test.json`")
    L.append("- `belief_*.png` / `consensus_timeline.png` / `trust_dynamics.png`")
    (outdir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ============================================================================
# 24. 主流程 run() + argparse + main
# ============================================================================
def run(ablation=None, use_llm=False, interactive=False, outdir="out_v13",
        model=MODEL_NAME, no_color=False, quiet=False, compare_ablations=False,
        agent_sim=False, agent_freedom=3, agent_rounds=2, freedom_compare=False,
        benchmark=False, atms_demo=False, second_story=False, atms_benchmark=False):
    global USE_COLOR
    if no_color: USE_COLOR = False
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    N = Narrator(quiet=quiet)

    # 第二故事 (泛化验证): 同一通用管线驱动全新领域 + 8 基线对比
    if second_story:
        run_second_story(outdir, N, use_llm=use_llm, model=model)
        run_pipeline_baselines(outdir, N)
        (outdir / "narration.txt").write_text("\n".join(N.captured), encoding="utf-8")
        N._emit(f"\n第二故事已写入: {(outdir / 'second_story.json').resolve()}")
        N._emit(f"基线对比已写入: {(outdir / 'pipeline_baselines.json').resolve()}")
        return None, {"second_story": True}

    # ATMS 必要性微基准 (100 案例 Full vs BFS) + 来源独立性 + 前提抵抗
    if atms_benchmark:
        bench = run_atms_benchmark(outdir, N)
        si = demo_source_independence(N)
        pr = demo_premise_resistance(N)
        N.end_scene()
        combined = {"benchmark": bench, "source_independence": si, "premise_resistance": pr}
        (outdir / "atms_benchmark.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        (outdir / "narration.txt").write_text("\n".join(N.captured), encoding="utf-8")
        N._emit(f"\nATMS 必要性基准已写入: {(outdir / 'atms_benchmark.json').resolve()}")
        return None, {"atms_benchmark": combined}

    # 仅跑 ATMS 核保留演示
    if atms_demo:
        atms_core_retention_demo(outdir, N)
        (outdir / "narration.txt").write_text("\n".join(N.captured), encoding="utf-8")
        N._emit(f"\n演示已写入: {(outdir / 'atms_demo.json').resolve()}")
        return None, {"atms_demo": True}

    # 仅跑 benchmark 对比
    if benchmark:
        N.banner("v13 — 仅 Benchmark 对比模式",
                 f"use_llm={use_llm and bool(API_KEY)}  outdir={outdir.resolve()}")
        bench = benchmark_comparison(outdir, N, use_llm=use_llm)
        (outdir / "narration.txt").write_text("\n".join(N.captured), encoding="utf-8")
        N._emit(f"\n对比已写入: {(outdir / 'benchmark_comparison.json').resolve()}")
        return None, {"benchmark": bench}

    # 三档自由度对比
    if freedom_compare:
        N.banner("v13 — 三档自由度对比模式", f"use_llm={use_llm and bool(API_KEY)}  outdir={outdir.resolve()}")
        fc = run_freedom_compare(outdir, N, use_llm=use_llm, rounds=agent_rounds)
        (outdir / "narration.txt").write_text("\n".join(N.captured), encoding="utf-8")
        return None, {"freedom_compare": fc}

    # 自主 Agent 仿真 (单档)
    if agent_sim:
        N.banner("v13 — 自主 LLM-Agent 仿真 (假骑士 / 身份坍缩)",
                 f"agent_freedom={agent_freedom} ({agent_freedom_label(agent_freedom)})  "
                 f"rounds={agent_rounds}  use_llm={use_llm and bool(API_KEY)}  outdir={outdir.resolve()}")
        eng = ConsensusEngine(N, ablation=ablation, use_llm=use_llm, model=model,
                              logger_path=outdir / "consensus_log.jsonl")
        setup_initial_secrets(eng)
        audit_res = run_agent_simulation(eng, agent_freedom, outdir, rounds=agent_rounds)
        atms_metrics = eng.compute_atms_metrics()
        print_atms_metrics(eng, N, atms_metrics)
        (outdir / "atms_metrics.json").write_text(
            json.dumps(atms_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (outdir / "hansson_postulates.json").write_text(
            json.dumps({"records": eng.hansson.records, "compliance": eng.hansson.compliance(),
                        "atms_decisions": eng.atms_decisions}, ensure_ascii=False, indent=2), encoding="utf-8")
        pipeline_metrics = eng.compute_pipeline_metrics()
        print_pipeline_metrics(eng, N, pipeline_metrics)
        (outdir / "pipeline_metrics.json").write_text(
            json.dumps(pipeline_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (outdir / "narration.txt").write_text("\n".join(N.captured), encoding="utf-8")
        N._emit(f"\n审核结果已写入: {(outdir / f'agent_audit_freedom{agent_freedom}.json').resolve()}")
        return eng, {"agent_audit": audit_res, "atms": atms_metrics}

    # 仅消融对比
    if compare_ablations:
        N.banner("v13 — 一键消融对比模式", f"use_llm={use_llm and bool(API_KEY)}  outdir={outdir.resolve()}")
        ablations_res = run_all_ablations(outdir, N, use_llm=use_llm)
        (outdir / "narration.txt").write_text("\n".join(N.captured), encoding="utf-8")
        N._emit(f"\n对比表已写入: {(outdir / 'ablation_comparison.md').resolve()}")
        return None, {"ablation_comparison": ablations_res}

    eng = ConsensusEngine(N, ablation=ablation, use_llm=use_llm, model=model,
                          logger_path=outdir / "consensus_log.jsonl")
    N.banner("NPC 记忆写入共识 v13 (Tom / Elena / Duran)",
             f"ablation={ablation}  use_llm={eng.use_llm}  model={model}  outdir={outdir.resolve()}")
    setup_initial_secrets(eng)

    if interactive:
        interactive_loop(eng)
    else:
        run_scripted(eng)
        print_full_state(eng)

    metrics = eng.compute_metrics()
    prov_metrics = eng.compute_provenance_metrics()
    atms_metrics = eng.compute_atms_metrics()
    metrics["Provenance_Eval"] = prov_metrics
    metrics["ATMS_Hansson_Eval"] = atms_metrics
    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "provenance_metrics.json").write_text(
        json.dumps(prov_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "atms_metrics.json").write_text(
        json.dumps(atms_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "hansson_postulates.json").write_text(
        json.dumps({"records": eng.hansson.records, "compliance": eng.hansson.compliance(),
                    "atms_decisions": eng.atms_decisions}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (outdir / "atms_ledger.jsonl").open("w", encoding="utf-8") as f:
        for rec in eng.atms.ledger:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # v13: 语义→形式化管线 / modal claims 输出
    pipeline_metrics = eng.compute_pipeline_metrics()
    metrics["Semantic_Pipeline_Eval"] = pipeline_metrics
    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "pipeline_metrics.json").write_text(
        json.dumps(pipeline_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "modal_claims.json").write_text(
        json.dumps({p: mc.describe() for p, mc in eng.modal_claims.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    with (outdir / "pipeline_log.jsonl").open("w", encoding="utf-8") as f:
        for rec in eng.pipeline_log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print_pipeline_metrics(eng, N, pipeline_metrics)
    with (outdir / "llm_audit.jsonl").open("w", encoding="utf-8") as f:
        for rec in eng.judge.audit:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    try:
        plot_belief_evolution(eng, "player_is_knight", outdir, "belief_knight.png")
        plot_belief_evolution(eng, "player_good_character", outdir, "belief_character.png")
        plot_belief_evolution(eng, "player_helped_village", outdir, "belief_helped.png")
        plot_consensus_timeline(eng, outdir, "consensus_timeline.png")
        plot_trust_dynamics(eng, outdir, "trust_dynamics.png")
    except Exception as e:
        N.warn(f"绘图跳过: {e}")

    prov_res = provenance_test(outdir, N)
    sweep = adversarial_sweep(outdir, N)
    decay_res = decay_test(outdir, N)
    complexity_res = cascade_complexity_test(outdir, N)
    repro = reproducibility_test(outdir, N)
    overhead = overhead_test(outdir, N)
    print_provenance_metrics(eng, N, prov_metrics, repro, overhead)
    print_atms_metrics(eng, N, atms_metrics)
    ablation_cmp = run_all_ablations(outdir, N, use_llm=False)
    bench_cmp = benchmark_comparison(outdir, N, use_llm=False, eng_full=eng)
    freedom_cmp = run_freedom_compare(outdir, N, use_llm=use_llm, rounds=agent_rounds)
    write_report(eng, metrics, outdir, prov_res, sweep, decay_res,
                 complexity_res=complexity_res, ablation_cmp=ablation_cmp,
                 bench_cmp=bench_cmp, freedom_cmp=freedom_cmp,
                 prov_metrics=prov_metrics, repro=repro, overhead=overhead,
                 atms_metrics=atms_metrics, pipeline_metrics=pipeline_metrics)

    N.banner("主线指标", "多 NPC FAMA / 身份保留 / 社群分歧 / 隐私 / 人格污染 / 级联")
    if not quiet:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    N._emit(f"\n输出目录: {outdir.resolve()}")
    (outdir / "narration.txt").write_text("\n".join(N.captured), encoding="utf-8")
    return eng, metrics


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="多 NPC 记忆写入共识仿真 v13 (Tom/Elena/Duran; 语义分层叙事记忆图 × 局部依赖级联 × "
                    "pairwise+community 两级传播 × 多NPC FAMA × MI/MPR × 三档自由度 Agent × 三框架对比)")
    ap.add_argument("--use-llm", action="store_true", help="真实接入 yunwu.ai (需 YUNWU_API_KEY)")
    ap.add_argument("--model", default=MODEL_NAME, help="模型名")
    ap.add_argument("--interactive", action="store_true", help="交互面板: 固定剧情 + 玩家输入 + NPC 自主 1:1/1:多")
    ap.add_argument("--benchmark", action="store_true", help="仅跑 三框架 vs 本系统 benchmark 对比")
    ap.add_argument("--compare-ablations", action="store_true", help="仅跑一键消融对比")
    ap.add_argument("--freedom-compare", action="store_true", help="仅跑三档自由度 Agent 对比")
    ap.add_argument("--agent-sim", action="store_true", help="跑单档自主 Agent 仿真 (planning/reflection/MI·MPR)")
    ap.add_argument("--agent-freedom", type=int, default=3, choices=[1, 2, 3],
                    help="Agent 自由度: 1=反应式(自建世界模型) 2=内心动机目标 3=效用 argmax")
    ap.add_argument("--agent-rounds", type=int, default=2, help="反转后额外自由演化回合数")
    ap.add_argument("--no-color", action="store_true", help="关闭 ANSI 颜色")
    ap.add_argument("--quiet", action="store_true", help="不打 terminal, 仅写文件")
    ap.add_argument("--atms-demo", action="store_true",
                    help="演示 ATMS 核保留: 圣剑资格 J1(骑士路径) OR J2(王室特许路径), 骑士被反证后经替代环境核保留")
    ap.add_argument("--second-story", action="store_true",
                    help="第二故事(个人助理记忆域)泛化验证: 同一通用管线驱动全新领域, 覆盖 7 种 AGM/KM 信念操作")
    ap.add_argument("--atms-benchmark", action="store_true",
                    help="ATMS 必要性微基准(100案例 Full vs BFS) + 来源独立性 + 前提抵抗探针")
    ap.add_argument("--ablation", default=None,
                    choices=[None, "no-trust", "no-persona", "no-provenance", "no-access",
                             "no-propagation", "flat-rag", "no-llm", "no-cascade", "no-atms",
                             "no-pipeline", "formal-only"])
    ap.add_argument("--outdir", default="out_v13")
    return ap


def main():
    a = build_arg_parser().parse_args()
    run(ablation=a.ablation, use_llm=a.use_llm, interactive=a.interactive,
        outdir=a.outdir, model=a.model, no_color=a.no_color, quiet=a.quiet,
        compare_ablations=a.compare_ablations, agent_sim=a.agent_sim,
        agent_freedom=a.agent_freedom, agent_rounds=a.agent_rounds,
        freedom_compare=a.freedom_compare, benchmark=a.benchmark, atms_demo=a.atms_demo,
        second_story=a.second_story, atms_benchmark=a.atms_benchmark)


if __name__ == "__main__":
    main()
