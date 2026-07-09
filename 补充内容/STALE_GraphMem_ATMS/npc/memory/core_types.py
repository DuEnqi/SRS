#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core data types for global consensus architecture.

Defines belief tuples, fact structures, and consensus-related types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum


class FactScope(Enum):
    """Visibility scope of a fact."""
    GLOBAL = "global"
    SCENE = "scene"
    PRIVATE = "private"


class BeliefStatus(Enum):
    """Status of a belief derivation."""
    DIRECT = "direct"
    DERIVED = "derived"
    INFERRED = "inferred"


@dataclass
class BeliefTuple:
    """
    Subjective logic belief tuple (b, d, u).
    
    - b (belief): Degree of belief [0, 1]
    - d (disbelief): Degree of disbelief [0, 1]
    - u (uncertainty): Degree of uncertainty [0, 1]
    
    Constraint: b + d + u = 1.0
    """
    belief: float
    disbelief: float
    uncertainty: float
    
    def __post_init__(self):
        """Validate belief tuple sums to 1.0."""
        total = self.belief + self.disbelief + self.uncertainty
        if not (0.99 < total < 1.01):  # Allow small floating point error
            raise ValueError(
                f"BeliefTuple components must sum to 1.0, got {total}. "
                f"b={self.belief}, d={self.disbelief}, u={self.uncertainty}"
            )
        # Normalize to ensure exact sum of 1.0
        self.belief = round(self.belief, 6)
        self.disbelief = round(self.disbelief, 6)
        self.uncertainty = round(self.uncertainty, 6)
    
    @staticmethod
    def from_confidence(confidence: float, uncertainty: float = None) -> 'BeliefTuple':
        """Create BeliefTuple from confidence score."""
        if uncertainty is None:
            uncertainty = 1.0 - confidence
        belief = confidence
        disbelief = 0.0
        return BeliefTuple(belief, disbelief, uncertainty)
    
    @staticmethod
    def uncertain() -> 'BeliefTuple':
        """Create maximum uncertainty belief tuple."""
        return BeliefTuple(0.0, 0.0, 1.0)
    
    def consensus(self, other: 'BeliefTuple', my_trust: float = 0.8) -> 'BeliefTuple':
        """
        Combine two belief tuples using trust-weighted consensus.
        
        Args:
            other: The other belief tuple
            my_trust: How much we trust the other belief [0, 1]
        
        Returns:
            Consensus belief tuple
        """
        if not (0 <= my_trust <= 1):
            raise ValueError(f"Trust weight must be in [0, 1], got {my_trust}")
        
        # Apply trust weight to other belief
        weighted_other_b = my_trust * other.belief
        weighted_other_d = my_trust * other.disbelief
        weighted_other_u = 1 - my_trust + my_trust * other.uncertainty
        
        # Average with our belief
        consensus_b = (self.belief + weighted_other_b) / 2
        consensus_d = (self.disbelief + weighted_other_d) / 2
        consensus_u = (self.uncertainty + weighted_other_u) / 2
        
        # Normalize
        total = consensus_b + consensus_d + consensus_u
        return BeliefTuple(
            consensus_b / total,
            consensus_d / total,
            consensus_u / total
        )
    
    def is_high_uncertainty(self, threshold: float = 0.5) -> bool:
        """Check if uncertainty exceeds threshold."""
        return self.uncertainty > threshold
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "belief": self.belief,
            "disbelief": self.disbelief,
            "uncertainty": self.uncertainty
        }
    
    def __repr__(self) -> str:
        return f"BeliefTuple(b={self.belief:.3f}, d={self.disbelief:.3f}, u={self.uncertainty:.3f})"


@dataclass
class Fact:
    """
    Represents a world state fact in the global graph.

    版本控制扩展（方案A时间戳版本机制）：
    - fact_id 标识"同一客观事实"的所有版本（如 princess_status）
    - version_node_id 唯一标识该具体版本节点（fact_id + "_v" + version）
    - is_active 同一 fact_id 下只有一个为 True
    - superseded_by 指向替代此版本的新版本节点 ID，形成单向链表
    - source_event 产生此版本的事件节点 ID
    - created_at 精确创建时间，用于新旧比对
    """
    fact_id: str
    subject: str
    predicate: str
    obj: str
    confidence: float
    scope: FactScope
    active: bool = True          # 保留向后兼容
    is_active: bool = True       # 版本控制：同 fact_id 下仅一个为 True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_npc: Optional[str] = None
    scene_id: Optional[str] = None
    version: int = 1
    version_node_id: str = ""    # 唯一版本节点 ID：fact_id + "_v" + version
    superseded_by: Optional[str] = None   # 指向新版本 version_node_id
    source_event: Optional[str] = None    # 产生此版本的事件节点 ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        # 自动生成 version_node_id（如未手动指定）
        if not self.version_node_id:
            self.version_node_id = f"{self.fact_id}_v{self.version}"
        # active 与 is_active 保持同步（向后兼容）
        self.active = self.is_active

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "confidence": self.confidence,
            "scope": self.scope.value,
            "active": self.active,
            "is_active": self.is_active,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
            "source_npc": self.source_npc,
            "scene_id": self.scene_id,
            "version": self.version,
            "version_node_id": self.version_node_id,
            "superseded_by": self.superseded_by,
            "source_event": self.source_event,
            "metadata": self.metadata,
        }

    def natural_language(self) -> str:
        """Convert to natural language summary."""
        return f"{self.subject} {self.predicate} {self.obj}"

    def is_stale(self) -> bool:
        """此版本是否已被更新版本取代。"""
        return not self.is_active


@dataclass
class NPCKnows:
    """
    Edge data for (NPC)-[:KNOWS]->(Fact) relationship.

    版本控制扩展：
    - version_node_id  指向具体版本节点（Fact.version_node_id）
    - last_updated     该信念最后刷新时间；与 Fact.created_at 比对可判断是否陈旧
    - is_stale         True 表示 NPC 持有的是旧版本信念，需要惰性刷新
    """
    npc_id: str
    fact_id: str
    opinion: BeliefTuple
    direct: bool = True
    derived: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # ---- 版本控制新增字段 ----
    version_node_id: str = ""          # 对应的 Fact.version_node_id
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    is_stale: bool = False             # NPC 是否持有陈旧版本信念

    def to_dict(self) -> Dict[str, Any]:
        return {
            "npc_id": self.npc_id,
            "fact_id": self.fact_id,
            "opinion": self.opinion.to_dict(),
            "direct": self.direct,
            "derived": self.derived,
            "timestamp": self.timestamp,
            "version_node_id": self.version_node_id,
            "last_updated": self.last_updated,
            "is_stale": self.is_stale,
        }


@dataclass
class NPCTrusts:
    """
    Edge data for (NPC)-[:TRUSTS]->(NPC) relationship.
    """
    npc_id_from: str
    npc_id_to: str
    weight: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        if not (0 <= self.weight <= 1):
            raise ValueError(f"Trust weight must be in [0, 1], got {self.weight}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.npc_id_from,
            "to": self.npc_id_to,
            "weight": self.weight,
            "timestamp": self.timestamp
        }


@dataclass
class FactUpdatedEvent:
    """
    Event published when a fact is updated in the global graph.
    """
    fact_id: str
    fact: Fact
    affected_scenes: List[str]
    event_type: str  # "created", "updated", "deleted"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    npc_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact": self.fact.to_dict(),
            "affected_scenes": self.affected_scenes,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "npc_id": self.npc_id
        }


@dataclass
class SceneConsensus:
    """
    Scene-filtered facts with precomputed beliefs for all NPCs.
    Used as cached context for efficient retrieval.
    """
    scene_id: str
    facts: List[Fact] = field(default_factory=list)
    npc_beliefs: Dict[str, Dict[str, BeliefTuple]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    ttl_seconds: int = 1800
    
    def is_expired(self) -> bool:
        """Check if cache has expired."""
        cached_time = datetime.fromisoformat(self.timestamp)
        elapsed = (datetime.now() - cached_time).total_seconds()
        return elapsed > self.ttl_seconds
    
    def get_npc_consensus_summary(self, npc_id: str, max_facts: int = 10) -> List[Dict[str, Any]]:
        """
        Get natural language summary of scene consensus for an NPC.
        
        Returns:
            List of {"fact": str, "confidence": float, "uncertainty": float}
        """
        if npc_id not in self.npc_beliefs:
            return []
        
        beliefs = self.npc_beliefs[npc_id]
        summaries = []
        
        for fact in self.facts[:max_facts]:
            if fact.fact_id not in beliefs:
                continue
            
            belief = beliefs[fact.fact_id]
            summaries.append({
                "fact": fact.natural_language(),
                "confidence": belief.belief,
                "uncertainty": belief.uncertainty,
                "belief_tuple": belief.to_dict()
            })
        
        return summaries
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "facts": [f.to_dict() for f in self.facts],
            "npc_beliefs": {
                npc_id: {fact_id: belief.to_dict() for fact_id, belief in beliefs.items()}
                for npc_id, beliefs in self.npc_beliefs.items()
            },
            "timestamp": self.timestamp
        }


@dataclass
class ExtractionResult:
    """
    Result from fact extraction from dialogue.
    """
    facts: List[Fact] = field(default_factory=list)
    confidence: float = 0.8
    dialogue_turn: int = 0
    source_npcs: List[str] = field(default_factory=list)
    scene_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": [f.to_dict() for f in self.facts],
            "confidence": self.confidence,
            "dialogue_turn": self.dialogue_turn,
            "source_npcs": self.source_npcs,
            "scene_id": self.scene_id,
            "timestamp": self.timestamp
        }
