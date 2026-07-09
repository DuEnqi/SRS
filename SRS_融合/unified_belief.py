#!/usr/bin/env python3
"""
unified_belief.py — 统一信念数据模型
=====================================
融合 ATMS Assumption2 + 组友 Fact/NPCKnows/NPCTrusts/BeliefTuple + 时间戳
+ GraphMemory 状态机 + Hansson incision trace。

单一数据类 UnifiedBelief 承载完整信念生命周期。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, FrozenSet, Tuple


class BeliefStatus(str, Enum):
    """信念状态机（v13 GraphMemory）"""
    ACTIVE = "active"           # 当前有效
    WEAK = "weak"               # 弱化（30天未确认）
    STALE = "stale"             # 失效（90天未确认）
    SUPERSEDED = "superseded"   # 被新版本取代
    HISTORICAL = "historical"   # 历史存档（180天）
    REFUTED = "refuted"         # 被直接反驳
    UNCERTAIN = "uncertain"     # 检索不连贯→弃权


class JustificationOp(str, Enum):
    AND = "AND"
    OR = "OR"
    DEFEASIBLE = "DEFEASIBLE"


@dataclass
class BeliefTuple:
    """主观逻辑信念三元组 (b, d, u)，b+d+u=1.0"""
    belief: float
    disbelief: float
    uncertainty: float

    def __post_init__(self):
        total = self.belief + self.disbelief + self.uncertainty
        if not (0.99 < total < 1.01):
            self.belief = round(self.belief / max(total, 0.001), 6)
            self.disbelief = round(self.disbelief / max(total, 0.001), 6)
            self.uncertainty = round(self.uncertainty / max(total, 0.001), 6)

    @staticmethod
    def from_confidence(c: float) -> "BeliefTuple":
        return BeliefTuple(c, 0.0, 1.0 - c)

    @staticmethod
    def uncertain() -> "BeliefTuple":
        return BeliefTuple(0.0, 0.0, 1.0)

    def to_dict(self) -> dict:
        return {"belief": self.belief, "disbelief": self.disbelief, "uncertainty": self.uncertainty}


@dataclass
class Justification:
    """ATMS justification 适配版"""
    jid: str
    premises: FrozenSet[str]         # 前提 claim IDs
    neg_premises: FrozenSet[str]     # 废止条件 claim IDs
    conclusion: str                  # 结论 claim ID
    operator: JustificationOp = JustificationOp.AND
    strength: float = 0.6
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UnifiedBelief:
    """
    融合后的统一信念数据模型。

    ATMS 视角: 一个带 justifications 和环境的 assumption
    组友视角: 一个带版本链和时间戳的 Fact
    v13 视角: 一个有状态机和访问层的 GraphMemory 节点
    """
    # ── 身份 ──
    claim_id: str                    # "Tom_is_knight"
    evidence_id: str                 # "user_statement_42"
    holder: str = "world"            # "world" / "Tom" / "Elena"

    # ── 信念 ──
    belief_tuple: BeliefTuple = field(default_factory=BeliefTuple.uncertain)

    # ── 时间戳 ──
    asserted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    valid_from: float = 0.0          # ATMS 有效开始
    valid_to: Optional[float] = None # ATMS 有效截止
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    # ── 支持结构 ──
    justifications: List[Justification] = field(default_factory=list)
    nogood_with: Set[str] = field(default_factory=set)  # 互斥 claim IDs

    # ── 版本控制 ──
    version: int = 1
    version_node_id: str = ""
    superseded_by: Optional[str] = None
    is_active: bool = True

    # ── 状态 + 访问 ──
    status: BeliefStatus = BeliefStatus.ACTIVE
    access_tier: str = "episodic"    # volatile|episodic|profile|core

    # ── 质量 ──
    credibility: float = 0.8         # Hansson 来源可信度
    confidence_level: str = "MEDIUM" # HIGH|MEDIUM|LOW (from v3)

    # ── 审计 ──
    incision_trace: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.version_node_id:
            self.version_node_id = f"{self.claim_id}_v{self.version}"

    def valid_at(self, t: float) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)

    def compute_status(self, now: float) -> BeliefStatus:
        """时间驱动的状态自动跃迁"""
        if self.status == BeliefStatus.REFUTED:
            return BeliefStatus.REFUTED
        if self.superseded_by:
            return BeliefStatus.SUPERSEDED

        try:
            last = datetime.fromisoformat(self.last_updated).timestamp()
        except Exception:
            last = self.asserted_at
            try:
                last = datetime.fromisoformat(str(last)).timestamp()
            except Exception:
                last = now - 1

        elapsed_days = (now - last) / 86400.0
        if elapsed_days > 180:
            return BeliefStatus.HISTORICAL
        if elapsed_days > 90:
            return BeliefStatus.STALE
        if elapsed_days > 30:
            return BeliefStatus.WEAK
        return BeliefStatus.ACTIVE

    def temporal_weight(self, now: float, half_life_days: float = 30.0) -> float:
        """时间衰减权重：30天半衰期"""
        try:
            last = datetime.fromisoformat(self.last_updated).timestamp()
        except Exception:
            last = now - 0.01
        elapsed = max(0.0, (now - last) / 86400.0)
        return 0.5 ** (elapsed / half_life_days)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "holder": self.holder,
            "belief": self.belief_tuple.to_dict(),
            "status": self.status.value,
            "version": self.version,
            "credibility": self.credibility,
            "confidence": self.confidence_level,
            "asserted_at": str(self.asserted_at),
        }

    def to_srs_belief(self) -> dict:
        """转换为 SRS 前端期望的 Belief 格式"""
        return {
            "id": self.claim_id,
            "statement": self.claim_id.replace("_", " "),
            "confidence": self.belief_tuple.belief,
            "evidence": self.evidence_id,
            "source": self.holder,
            "timestamp": str(self.valid_from),
        }

    def to_srs_memory_node(self) -> dict:
        """转换为 SRS 前端期望的 Memory Node 格式"""
        return {
            "id": self.version_node_id,
            "type": "claim",
            "title": self.claim_id.replace("_", " "),
            "description": f"Belief by {self.holder}, confidence={self.belief_tuple.belief:.2f}",
            "timestamp": str(self.valid_from),
            "source": self.holder,
            "confidence": self.belief_tuple.belief,
            "relatedNPCs": [self.holder],
        }
