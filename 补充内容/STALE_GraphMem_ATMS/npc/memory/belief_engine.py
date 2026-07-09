#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Belief engine for subjective logic consensus computation.

Implements trust-weighted opinion discounting and consensus operators
for lazy belief calculation from the global graph.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math

from npc.memory.core_types import BeliefTuple, Fact, FactScope


class SubjectiveLogicEngine:
    """
    Implements subjective logic operators for belief computation.
    
    Supports trust-weighted opinion aggregation and consensus operations
    following Jøsang's subjective logic framework.
    """
    
    CONFIDENCE_THRESHOLD = 0.3
    MAX_TRUST_DEPTH = 3
    
    @staticmethod
    def discount_opinion(
        opinion: BeliefTuple,
        trust_weight: float
    ) -> BeliefTuple:
        """
        Apply trust weight discount to an opinion.
        
        Formula: ω(i←j) = (t_ij * b_j, t_ij * d_j, 1 - t_ij + t_ij * u_j)
        
        Args:
            opinion: Original belief tuple from source
            trust_weight: Trust weight in [0, 1]
        
        Returns:
            Discounted belief tuple
        """
        if not (0 <= trust_weight <= 1):
            raise ValueError(f"Trust weight must be in [0, 1], got {trust_weight}")
        
        discounted_b = trust_weight * opinion.belief
        discounted_d = trust_weight * opinion.disbelief
        discounted_u = 1 - trust_weight + trust_weight * opinion.uncertainty
        
        return BeliefTuple(discounted_b, discounted_d, discounted_u)
    
    @staticmethod
    def consensus_operator(
        opinions: List[BeliefTuple],
        weights: Optional[List[float]] = None
    ) -> BeliefTuple:
        """
        Compute consensus from multiple opinions using weighted average.
        
        Implements a simplified consensus that handles:
        - Multiple opinion sources
        - Weighted contribution
        - Uncertainty preservation
        
        Args:
            opinions: List of belief tuples
            weights: Optional weights for each opinion (normalized to sum to 1)
        
        Returns:
            Consensus belief tuple
        """
        if not opinions:
            return BeliefTuple.uncertain()
        
        if len(opinions) == 1:
            return opinions[0]
        
        # Default: equal weights
        if weights is None:
            weights = [1.0 / len(opinions)] * len(opinions)
        else:
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1.0 / len(opinions)] * len(opinions)
            else:
                weights = [w / total_weight for w in weights]
        
        # Compute weighted average
        consensus_b = sum(op.belief * w for op, w in zip(opinions, weights))
        consensus_d = sum(op.disbelief * w for op, w in zip(opinions, weights))
        consensus_u = sum(op.uncertainty * w for op, w in zip(opinions, weights))
        
        # Normalize to ensure sum = 1
        total = consensus_b + consensus_d + consensus_u
        if total > 0:
            consensus_b /= total
            consensus_d /= total
            consensus_u /= total
        
        return BeliefTuple(consensus_b, consensus_d, consensus_u)
    
    @staticmethod
    def cumulative_consensus(
        opinions: List[BeliefTuple]
    ) -> BeliefTuple:
        """
        Compute cumulative consensus over multiple opinions.
        
        Iteratively combines opinions pairwise, useful for belief aggregation
        from multiple independent sources.
        
        Args:
            opinions: List of belief tuples
        
        Returns:
            Cumulative consensus belief tuple
        """
        if not opinions:
            return BeliefTuple.uncertain()
        
        result = opinions[0]
        for opinion in opinions[1:]:
            # Combine pairwise using equal weights
            result = SubjectiveLogicEngine.consensus_operator([result, opinion])
        
        return result
    
    @staticmethod
    def apply_confidence_threshold(
        opinion: BeliefTuple,
        threshold: float = CONFIDENCE_THRESHOLD
    ) -> BeliefTuple:
        """
        Apply confidence threshold to preserve uncertainty for low-confidence facts.
        
        If (b + d) < threshold, boost uncertainty to maintain "don't know" signal.
        
        Args:
            opinion: Original belief tuple
            threshold: Confidence threshold [0, 1]
        
        Returns:
            Thresholded belief tuple
        """
        confidence = opinion.belief + opinion.disbelief
        
        if confidence < threshold:
            # Boost uncertainty while preserving b:d ratio
            new_u = min(1.0, opinion.uncertainty + (threshold - confidence))
            
            # Redistribute b and d to maintain ratio with remaining mass
            remaining = 1.0 - new_u
            if confidence > 0:
                ratio = opinion.belief / confidence
                new_b = remaining * ratio
                new_d = remaining * (1 - ratio)
            else:
                new_b = 0
                new_d = 0
            
            return BeliefTuple(new_b, new_d, new_u)
        
        return opinion


class BeliefEngine:
    """
    High-level engine for computing NPC beliefs from the global graph.

    版本控制扩展（方案A工程指南第4、5节）：
    - compute_belief 实现三路版本感知：
        A. NPC 已持有活跃版本信念且不陈旧 → 直接读取
        B. NPC 持有同 fact_id 的旧版本信念 → 沿版本链惰性刷新
        C. NPC 无任何 KNOWS 边 → 信任网络惰性传播（陈旧来源加不确定性惩罚）
    - 陈旧来源惩罚：若 witness 持有的是旧版本，信任权重额外打折 STALE_TRUST_PENALTY
    """

    STALE_TRUST_PENALTY = 0.5   # 陈旧来源信任折扣系数

    def __init__(self, graph_backend):
        self.graph = graph_backend
        self.belief_cache: Dict[Tuple[str, str], BeliefTuple] = {}
        self.trust_network_cache: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def compute_belief(
        self,
        npc_id: str,
        fact_id: str,
        use_cache: bool = True,
    ) -> BeliefTuple:
        """
        版本感知的信念计算入口（惰性）。

        按情况A→B→C依次尝试，命中即返回。
        缓存仅存储"已刷新"的结果，保证下次直接命中情况A。
        """
        cache_key = (npc_id, fact_id)
        if use_cache and cache_key in self.belief_cache:
            return self.belief_cache[cache_key]

        belief = self._compute_belief_versioned(npc_id, fact_id)

        if use_cache:
            self.belief_cache[cache_key] = belief
        return belief

    # ------------------------------------------------------------------
    # 版本感知核心逻辑
    # ------------------------------------------------------------------

    def _compute_belief_versioned(self, npc_id: str, fact_id: str) -> BeliefTuple:
        """三路版本感知信念计算（不使用缓存）。"""
        backend = self.graph.backend

        # 获取当前活跃事实版本
        active_fact = self._get_active_fact(fact_id)

        # 获取 NPC 已有的信念边
        direct_edge = backend.get_knows_edge(npc_id, fact_id)

        # ----------------------------------------------------------------
        # 情况 A：NPC 已有信念边且指向当前活跃版本（新鲜）→ 直接读取
        # ----------------------------------------------------------------
        if direct_edge and active_fact:
            edge_is_fresh = (
                direct_edge.version_node_id == active_fact.version_node_id
                and not direct_edge.is_stale
                and direct_edge.last_updated >= active_fact.created_at
            )
            if edge_is_fresh:
                return direct_edge.opinion

        # ----------------------------------------------------------------
        # 情况 B：NPC 有旧版本信念边 → 沿版本链刷新
        # ----------------------------------------------------------------
        if direct_edge and active_fact and (
            direct_edge.version_node_id != active_fact.version_node_id
            or direct_edge.is_stale
        ):
            refreshed = self._refresh_belief_along_chain(
                npc_id, fact_id, direct_edge, active_fact
            )
            # 将刷新后的信念写回图（更新 KNOWS 边）
            self._persist_refreshed_belief(npc_id, fact_id, refreshed, active_fact)
            return refreshed

        # ----------------------------------------------------------------
        # 情况 C：无 KNOWS 边 → 信任网络惰性传播
        # ----------------------------------------------------------------
        return self._propagate_via_trust_network(npc_id, fact_id, active_fact)

    # ------------------------------------------------------------------
    # 情况 B：沿版本链刷新信念
    # ------------------------------------------------------------------

    def _refresh_belief_along_chain(
        self,
        npc_id: str,
        fact_id: str,
        stale_edge: "NPCKnows",
        active_fact: Fact,
    ) -> BeliefTuple:
        """
        从旧版本信念出发，沿 SUPERSEDED_BY 链累积刷新到最新版本。

        每经过一个新版本，将新版本的置信度变化作为"新证据"，
        结合版本间 confidence 差值调整信念。
        """
        backend = self.graph.backend
        chain = self._get_version_chain(fact_id)

        if not chain or len(chain) < 2:
            # 链只有一个版本，无需刷新，使用当前置信度重建信念
            return BeliefTuple.from_confidence(active_fact.confidence)

        # 找到 NPC 持有旧版本在链中的位置
        old_vnid = stale_edge.version_node_id
        try:
            old_idx = next(
                i for i, f in enumerate(chain)
                if f.version_node_id == old_vnid
            )
        except StopIteration:
            old_idx = 0  # 找不到则从头刷新

        # 从旧版本信念出发，逐步刷新到链末（当前活跃版本）
        current_belief = stale_edge.opinion
        for fact_version in chain[old_idx + 1:]:
            delta_confidence = fact_version.confidence - chain[max(0, chain.index(fact_version) - 1)].confidence
            strength = abs(delta_confidence) * 0.8   # 版本跨度衰减
            if delta_confidence >= 0:
                # 新证据增强信念
                new_b = current_belief.belief + strength * (1 - current_belief.belief)
                new_b = min(1.0, new_b)
                new_d = max(0.0, current_belief.disbelief - strength * current_belief.disbelief)
            else:
                # 新证据削弱信念
                new_b = max(0.0, current_belief.belief - strength * current_belief.belief)
                new_d = min(1.0, current_belief.disbelief + strength * (1 - current_belief.disbelief))
            new_u = max(0.0, 1.0 - new_b - new_d)
            total = new_b + new_d + new_u
            if total > 0:
                current_belief = BeliefTuple(new_b / total, new_d / total, new_u / total)

        return current_belief

    def _persist_refreshed_belief(
        self,
        npc_id: str,
        fact_id: str,
        opinion: BeliefTuple,
        active_fact: Fact,
    ) -> None:
        """将刷新后的信念写回图（更新 KNOWS 边指向活跃版本）。"""
        from datetime import datetime as _dt
        from npc.memory.core_types import NPCKnows
        edge = NPCKnows(
            npc_id=npc_id,
            fact_id=fact_id,
            opinion=opinion,
            direct=False,
            derived=True,
            version_node_id=active_fact.version_node_id,
            last_updated=_dt.now().isoformat(),
            is_stale=False,
        )
        self.graph.backend.create_knows_edge(edge)
        # 清除缓存，下次直接命中情况A
        self.belief_cache.pop((npc_id, fact_id), None)

    # ------------------------------------------------------------------
    # 情况 C：信任网络惰性传播（含陈旧来源惩罚）
    # ------------------------------------------------------------------

    def _propagate_via_trust_network(
        self,
        npc_id: str,
        fact_id: str,
        active_fact: Optional[Fact],
    ) -> BeliefTuple:
        """
        通过信任网络计算间接信念。

        陈旧来源惩罚（工程指南第5.3节）：
        若 witness 持有的 KNOWS 边指向旧版本（is_stale=True 或
        version_node_id != active_fact.version_node_id），
        其信任权重额外乘以 STALE_TRUST_PENALTY。
        """
        backend = self.graph.backend
        active_vnid = active_fact.version_node_id if active_fact else ""

        witness_opinions: List[BeliefTuple] = []
        witness_weights: List[float] = []

        for (w_npc, w_fact), edge in backend.knows_edges.items():
            if w_fact != fact_id or w_npc == npc_id:
                continue

            base_trust = backend.get_trust_weight(npc_id, w_npc)

            # 陈旧来源惩罚
            is_stale_source = (
                edge.is_stale
                or (active_vnid and edge.version_node_id != active_vnid)
            )
            effective_trust = (
                base_trust * self.STALE_TRUST_PENALTY
                if is_stale_source
                else base_trust
            )

            discounted = SubjectiveLogicEngine.discount_opinion(edge.opinion, effective_trust)
            witness_opinions.append(discounted)
            witness_weights.append(effective_trust)

        if not witness_opinions:
            return BeliefTuple.uncertain()

        consensus = SubjectiveLogicEngine.consensus_operator(witness_opinions, witness_weights)
        return SubjectiveLogicEngine.apply_confidence_threshold(consensus)

    # ------------------------------------------------------------------
    # 辅助工具
    # ------------------------------------------------------------------

    def _get_active_fact(self, fact_id: str) -> Optional[Fact]:
        """获取 fact_id 当前活跃版本，兼容新旧后端。"""
        from npc.memory.graph_consensus import InMemoryGraphDatabase
        if isinstance(self.graph.backend, InMemoryGraphDatabase):
            return self.graph.backend.get_active_fact(fact_id)
        return self.graph.backend.get_fact(fact_id)

    def _get_version_chain(self, fact_id: str) -> List[Fact]:
        """获取版本链（含全部历史版本）。"""
        from npc.memory.graph_consensus import InMemoryGraphDatabase
        if isinstance(self.graph.backend, InMemoryGraphDatabase):
            return self.graph.backend.get_version_chain(fact_id)
        active = self._get_active_fact(fact_id)
        return [active] if active else []

    def compute_beliefs_for_npc(
        self,
        npc_id: str,
        facts: List[Fact],
    ) -> Dict[str, BeliefTuple]:
        """批量计算 NPC 对一组事实的信念。"""
        return {
            fact.fact_id: self.compute_belief(npc_id, fact.fact_id)
            for fact in facts
        }

    def invalidate_cache(self, npc_id: Optional[str] = None) -> None:
        """清除信念缓存。"""
        if npc_id:
            keys = [k for k in self.belief_cache if k[0] == npc_id]
            for k in keys:
                del self.belief_cache[k]
        else:
            self.belief_cache.clear()

    def invalidate_trust_cache(self, npc_id: Optional[str] = None) -> None:
        """清除信任网络缓存。"""
        if npc_id:
            self.trust_network_cache.pop(npc_id, None)
        else:
            self.trust_network_cache.clear()

    def get_trust_network(self, npc_id: str) -> Dict[str, float]:
        """获取 NPC 的信任网络（带缓存）。"""
        if npc_id not in self.trust_network_cache:
            network = {}
            for edge in self.graph.backend.trusts_edges.values():
                if edge.npc_id_from == npc_id:
                    network[edge.npc_id_to] = edge.weight
            self.trust_network_cache[npc_id] = network
        return self.trust_network_cache[npc_id]


class BeliefMetrics:
    """Utility class for belief analysis and metrics."""
    
    @staticmethod
    def belief_variance(beliefs: List[BeliefTuple]) -> float:
        """
        Compute variance in belief tuples across multiple beliefs.
        
        Measures disagreement in (b, d, u) components.
        
        Args:
            beliefs: List of belief tuples
        
        Returns:
            Variance measure [0, 1]
        """
        if len(beliefs) < 2:
            return 0.0
        
        # Compute mean belief
        mean_b = sum(b.belief for b in beliefs) / len(beliefs)
        mean_d = sum(b.disbelief for b in beliefs) / len(beliefs)
        mean_u = sum(b.uncertainty for b in beliefs) / len(beliefs)
        
        # Compute variance
        var_b = sum((b.belief - mean_b) ** 2 for b in beliefs) / len(beliefs)
        var_d = sum((b.disbelief - mean_d) ** 2 for b in beliefs) / len(beliefs)
        var_u = sum((b.uncertainty - mean_u) ** 2 for b in beliefs) / len(beliefs)
        
        # Total variance
        return math.sqrt(var_b + var_d + var_u)
    
    @staticmethod
    def belief_distance(belief1: BeliefTuple, belief2: BeliefTuple) -> float:
        """
        Compute distance between two belief tuples.
        
        Uses Euclidean distance in (b, d, u) space.
        
        Args:
            belief1: First belief tuple
            belief2: Second belief tuple
        
        Returns:
            Distance [0, sqrt(2)]
        """
        return math.sqrt(
            (belief1.belief - belief2.belief) ** 2 +
            (belief1.disbelief - belief2.disbelief) ** 2 +
            (belief1.uncertainty - belief2.uncertainty) ** 2
        )
    
    @staticmethod
    def convergence_score(beliefs: List[BeliefTuple]) -> float:
        """
        Compute convergence score (inverse of variance).
        
        Higher score means more agreement across beliefs.
        
        Args:
            beliefs: List of belief tuples
        
        Returns:
            Convergence score [0, 1]
        """
        if len(beliefs) < 2:
            return 1.0
        
        variance = BeliefMetrics.belief_variance(beliefs)
        # Normalize variance to [0, 1] and invert
        return max(0, 1 - variance)
