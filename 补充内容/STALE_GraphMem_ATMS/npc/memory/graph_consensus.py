#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global graph consensus database wrapper.

Provides unified interface to the global knowledge graph storing all world state facts.
Handles fact storage, retrieval, and basic graph operations.
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime
import json
import os
from abc import ABC, abstractmethod

from npc.memory.core_types import (
    Fact, FactScope, FactUpdatedEvent, NPCKnows, NPCTrusts, 
    BeliefTuple, ExtractionResult
)


class GraphDatabaseBackend(ABC):
    """Abstract base class for graph database implementations."""
    
    @abstractmethod
    def create_fact(self, fact: Fact) -> str:
        """Create a new fact node. Returns fact_id."""
        pass
    
    @abstractmethod
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by ID."""
        pass
    
    @abstractmethod
    def update_fact(self, fact: Fact) -> bool:
        """Update an existing fact."""
        pass
    
    @abstractmethod
    def query_facts_by_scene(self, scene_id: str) -> List[Fact]:
        """Query all facts that occur in a scene."""
        pass
    
    @abstractmethod
    def query_global_facts(self) -> List[Fact]:
        """Query all global scope facts."""
        pass
    
    @abstractmethod
    def create_knows_edge(self, edge: NPCKnows) -> bool:
        """Create (NPC)-[:KNOWS]->(Fact) edge."""
        pass
    
    @abstractmethod
    def get_knows_edge(self, npc_id: str, fact_id: str) -> Optional[NPCKnows]:
        """Get (NPC)-[:KNOWS]->(Fact) edge."""
        pass
    
    @abstractmethod
    def create_trusts_edge(self, edge: NPCTrusts) -> bool:
        """Create (NPC)-[:TRUSTS]->(NPC) edge."""
        pass
    
    @abstractmethod
    def get_trust_weight(self, npc_from: str, npc_to: str) -> float:
        """Get trust weight between NPCs."""
        pass


class InMemoryGraphDatabase(GraphDatabaseBackend):
    """
    In-memory graph database implementation for testing and development.

    版本控制扩展：
    - version_chains: fact_id -> List[version_node_id]，按版本号升序排列
    - facts 字典的 key 改为 version_node_id（每个版本独立存储）
    - fact_id_index: fact_id -> 当前活跃的 version_node_id（快速查活跃版本）
    """

    def __init__(self):
        # key: version_node_id，每个版本独立存储
        self.facts: Dict[str, Fact] = {}
        # key: fact_id，value: 按版本号升序的 version_node_id 列表
        self.version_chains: Dict[str, List[str]] = {}
        # key: fact_id，value: 当前活跃版本的 version_node_id
        self.fact_id_index: Dict[str, str] = {}
        self.knows_edges: Dict[Tuple[str, str], "NPCKnows"] = {}
        self.trusts_edges: Dict[Tuple[str, str], "NPCTrusts"] = {}
        self._fact_counter = 0

    # ------------------------------------------------------------------
    # 版本链内部工具
    # ------------------------------------------------------------------

    def _deactivate_current_version(self, fact_id: str, new_version_node_id: str) -> None:
        """将 fact_id 当前活跃版本标记为非活跃，并设置 superseded_by。"""
        active_vnid = self.fact_id_index.get(fact_id)
        if active_vnid and active_vnid in self.facts:
            old_fact = self.facts[active_vnid]
            old_fact.is_active = False
            old_fact.active = False
            old_fact.superseded_by = new_version_node_id

    def get_active_fact(self, fact_id: str) -> Optional[Fact]:
        """按 fact_id 获取当前活跃版本。"""
        vnid = self.fact_id_index.get(fact_id)
        return self.facts.get(vnid) if vnid else None

    def get_version_chain(self, fact_id: str) -> List[Fact]:
        """返回 fact_id 的完整版本链（按版本号升序）。"""
        chain_ids = self.version_chains.get(fact_id, [])
        return [self.facts[vid] for vid in chain_ids if vid in self.facts]

    def find_active_facts_by_subject_predicate(
        self, subject: str, predicate: str
    ) -> List[Fact]:
        """查找 subject+predicate 相同且 is_active=True 的事实（版本匹配用）。"""
        results = []
        for vnid in self.fact_id_index.values():
            f = self.facts.get(vnid)
            if f and f.subject == subject and f.predicate == predicate and f.is_active:
                results.append(f)
        return results

    def gc_version_chain(self, fact_id: str, max_versions: int = 5) -> int:
        """
        垃圾回收：当版本数超过 max_versions 时，移除最旧的非活跃版本。
        返回实际删除的版本数。
        """
        chain = self.version_chains.get(fact_id, [])
        if len(chain) <= max_versions:
            return 0

        removed = 0
        to_remove = chain[: len(chain) - max_versions]
        for vnid in to_remove:
            fact = self.facts.get(vnid)
            if fact and not fact.is_active:
                del self.facts[vnid]
                removed += 1

        self.version_chains[fact_id] = [
            vid for vid in chain if vid in self.facts
        ]
        return removed

    # ------------------------------------------------------------------
    # GraphDatabaseBackend 接口实现
    # ------------------------------------------------------------------

    def create_fact(self, fact: Fact) -> str:
        """
        创建新事实版本节点。

        规则：
        - 若未设置 fact_id，自动生成。
        - version_node_id 格式：fact_id_v{version}。
        - 同一 fact_id 下前一个活跃版本自动标记为非活跃并链接 superseded_by。
        - 更新版本链索引和活跃版本索引。
        """
        if not fact.fact_id:
            self._fact_counter += 1
            fact.fact_id = f"fact_{self._fact_counter}"

        # 推算版本号：当前链长度 + 1
        existing_chain = self.version_chains.get(fact.fact_id, [])
        fact.version = len(existing_chain) + 1
        fact.version_node_id = f"{fact.fact_id}_v{fact.version}"
        fact.is_active = True
        fact.active = True

        # 旧活跃版本退位
        self._deactivate_current_version(fact.fact_id, fact.version_node_id)

        # 写入
        self.facts[fact.version_node_id] = fact
        self.version_chains.setdefault(fact.fact_id, []).append(fact.version_node_id)
        self.fact_id_index[fact.fact_id] = fact.version_node_id

        # 自动 GC（保留最近 10 个版本）
        self.gc_version_chain(fact.fact_id, max_versions=10)

        return fact.fact_id

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """获取 fact_id 的当前活跃版本。"""
        return self.get_active_fact(fact_id)

    def get_fact_by_version_node_id(self, version_node_id: str) -> Optional[Fact]:
        """按完整版本节点 ID 获取特定版本（含历史版本）。"""
        return self.facts.get(version_node_id)

    def update_fact(self, fact: Fact) -> bool:
        """
        更新事实（创建新版本）。
        直接复用 create_fact 的版本化写入逻辑。
        """
        if fact.fact_id not in self.fact_id_index:
            return False
        self.create_fact(fact)
        return True

    def query_facts_by_scene(self, scene_id: str) -> List[Fact]:
        """查询场景内所有 is_active=True 的事实。"""
        return [
            f for f in self.facts.values()
            if f.scene_id == scene_id and f.is_active
        ]

    def query_global_facts(self) -> List[Fact]:
        """查询所有 global scope 且 is_active=True 的事实。"""
        return [
            f for f in self.facts.values()
            if f.scope == FactScope.GLOBAL and f.is_active
        ]

    def create_knows_edge(self, edge: "NPCKnows") -> bool:
        """创建或覆盖 (NPC)-[:KNOWS]->(Fact) 边。"""
        key = (edge.npc_id, edge.fact_id)
        self.knows_edges[key] = edge
        return True

    def get_knows_edge(self, npc_id: str, fact_id: str) -> Optional["NPCKnows"]:
        """获取 (NPC)-[:KNOWS]->(Fact) 边。"""
        return self.knows_edges.get((npc_id, fact_id))

    def get_knows_edges_by_npc(self, npc_id: str) -> List["NPCKnows"]:
        """获取某个 NPC 的所有 KNOWS 边。"""
        return [e for (n, _), e in self.knows_edges.items() if n == npc_id]

    def create_trusts_edge(self, edge: "NPCTrusts") -> bool:
        """创建 (NPC)-[:TRUSTS]->(NPC) 边。"""
        key = (edge.npc_id_from, edge.npc_id_to)
        self.trusts_edges[key] = edge
        return True

    def get_trust_weight(self, npc_from: str, npc_to: str) -> float:
        """获取信任权重，不存在时返回默认值 0.5。"""
        edge = self.trusts_edges.get((npc_from, npc_to))
        return edge.weight if edge else 0.5


class GlobalGraphConsensus:
    """
    Main interface to the global consensus graph.
    
    Provides high-level operations for fact management and graph queries.
    """
    
    def __init__(self, backend: Optional[GraphDatabaseBackend] = None):
        """
        Initialize the global graph consensus.
        
        Args:
            backend: Graph database backend. Defaults to InMemoryGraphDatabase.
        """
        self.backend = backend or InMemoryGraphDatabase()
        self.event_listeners: List[callable] = []
    
    def register_event_listener(self, callback: callable) -> None:
        """Register a callback for FactUpdatedEvent."""
        self.event_listeners.append(callback)
    
    def _publish_event(self, event: FactUpdatedEvent) -> None:
        """Publish a fact update event to all listeners."""
        for listener in self.event_listeners:
            try:
                listener(event)
            except Exception as e:
                print(f"Error in event listener: {e}")
    
    def write_fact(self, fact: Fact) -> Tuple[str, Optional[FactUpdatedEvent]]:
        """
        版本化写入：将事实写入全局图，自动管理版本链。

        匹配逻辑（按工程指南第3节）：
        1. 在活跃事实中查找 subject+predicate 相同的事实：
           - 若 object 相同 → 置信度更高时覆盖更新，否则跳过（幂等）
           - 若 object 不同 → 判定为同一事实的新版本，创建新版本并将旧版本
             标记为 is_active=False、superseded_by 指向新版本
        2. 若完全没有匹配 → 创建全新 fact_id（version=1）
        3. 直接目击者的 KNOWS 边由调用方写入（write_npc_knows_edge）

        Returns:
            (fact_id, FactUpdatedEvent if published)
        """
        affected_scenes = [fact.scene_id] if fact.scene_id else []

        # 查找 subject+predicate 相同的活跃版本
        if isinstance(self.backend, InMemoryGraphDatabase):
            same_sp = self.backend.find_active_facts_by_subject_predicate(
                fact.subject, fact.predicate
            )
        else:
            # 非内存后端：遍历全量查询兜底
            all_active = (
                self.backend.query_global_facts()
                + self.backend.query_facts_by_scene(fact.scene_id or "")
            )
            same_sp = [
                f for f in all_active
                if f.subject == fact.subject and f.predicate == fact.predicate
            ]

        event_type = "created"

        if same_sp:
            existing = same_sp[0]

            if existing.obj == fact.obj:
                # 同一三元组：仅在置信度更高时更新（幂等保护）
                if fact.confidence <= existing.confidence:
                    return existing.fact_id, None
                # 保持 fact_id 一致，创建置信度更高的新版本
                fact.fact_id = existing.fact_id
                event_type = "updated"
            else:
                # object 不同 → 新版本替代旧版本
                fact.fact_id = existing.fact_id
                event_type = "updated"
        # else: 全新 fact_id，event_type 保持 "created"

        fact_id = self.backend.create_fact(fact)

        event = FactUpdatedEvent(
            fact_id=fact_id,
            fact=fact,
            affected_scenes=affected_scenes,
            event_type=event_type,
        )
        self._publish_event(event)
        return fact_id, event

    def get_version_chain(self, fact_id: str) -> List[Fact]:
        """
        返回 fact_id 的完整版本历史（按版本号升序）。
        最后一个元素即当前活跃版本。
        """
        if isinstance(self.backend, InMemoryGraphDatabase):
            return self.backend.get_version_chain(fact_id)
        # 其他后端：仅返回活跃版本
        active = self.backend.get_fact(fact_id)
        return [active] if active else []

    def get_active_fact(self, fact_id: str) -> Optional[Fact]:
        """获取 fact_id 的当前活跃版本。"""
        if isinstance(self.backend, InMemoryGraphDatabase):
            return self.backend.get_active_fact(fact_id)
        return self.backend.get_fact(fact_id)
    
    def write_npc_knows_edge(
        self,
        npc_id: str,
        fact_id: str,
        opinion: Optional[BeliefTuple] = None,
        direct: bool = True,
        version_node_id: str = "",
    ) -> bool:
        """
        创建或更新 (NPC)-[:KNOWS]->(Fact) 边。

        版本控制扩展：
        - version_node_id 为空时，自动取 fact_id 当前活跃版本的 version_node_id。
        - last_updated 设为当前时间。
        - 若 NPC 已有同 fact_id 的旧版本信念边，将其标记为 is_stale=True
          并用新版本边覆盖。
        """
        if opinion is None:
            opinion = BeliefTuple(0.9, 0.0, 0.1)

        # 自动解析当前活跃版本节点 ID
        if not version_node_id:
            active_fact = self.get_active_fact(fact_id)
            version_node_id = active_fact.version_node_id if active_fact else f"{fact_id}_v1"

        now = datetime.now().isoformat()
        edge = NPCKnows(
            npc_id=npc_id,
            fact_id=fact_id,
            opinion=opinion,
            direct=direct,
            derived=False,
            version_node_id=version_node_id,
            last_updated=now,
            is_stale=False,
        )
        return self.backend.create_knows_edge(edge)
    
    def set_trust_relationship(
        self, 
        npc_from: str, 
        npc_to: str, 
        weight: float
    ) -> bool:
        """
        Set trust weight between NPCs.
        
        Args:
            npc_from: Trusting NPC
            npc_to: Trusted NPC
            weight: Trust weight [0, 1]
        
        Returns:
            True if created/updated
        """
        edge = NPCTrusts(
            npc_id_from=npc_from,
            npc_id_to=npc_to,
            weight=weight
        )
        return self.backend.create_trusts_edge(edge)
    
    def get_scene_facts(
        self, 
        scene_id: str,
        include_global: bool = True
    ) -> List[Fact]:
        """
        Get all active facts for a scene.
        
        Args:
            scene_id: Scene identifier
            include_global: Whether to include global scope facts
        
        Returns:
            List of facts
        """
        facts = self.backend.query_facts_by_scene(scene_id)
        
        if include_global:
            facts.extend(self.backend.query_global_facts())
        
        return facts
    
    def get_npc_belief(self, npc_id: str, fact_id: str) -> Optional[BeliefTuple]:
        """
        Get NPC's belief about a fact.
        
        Returns the belief tuple from (NPC)-[:KNOWS]->(Fact) edge if exists.
        
        Args:
            npc_id: NPC identifier
            fact_id: Fact identifier
        
        Returns:
            BeliefTuple if edge exists, None otherwise
        """
        edge = self.backend.get_knows_edge(npc_id, fact_id)
        return edge.opinion if edge else None
    
    def get_trust_weight(self, npc_from: str, npc_to: str) -> float:
        """Get trust weight between NPCs."""
        return self.backend.get_trust_weight(npc_from, npc_to)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export graph state for serialization."""
        return {
            "facts": [f.to_dict() for f in self.backend.facts.values()],
            "knows_edges": [e.to_dict() for e in self.backend.knows_edges.values()],
            "trusts_edges": [e.to_dict() for e in self.backend.trusts_edges.values()],
            "timestamp": datetime.now().isoformat()
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save graph state to JSON file."""
        data = self.export_to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def clear(self) -> None:
        """Clear all facts and edges from the graph."""
        self.backend.facts.clear()
        self.backend.knows_edges.clear()
        self.backend.trusts_edges.clear()
