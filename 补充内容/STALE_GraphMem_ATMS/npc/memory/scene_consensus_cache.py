#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene consensus cache for efficient scene-aware retrieval.

Pre-computes and caches world consensus facts for each scene,
reducing repeated graph queries and belief computations.
"""

from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from npc.memory.core_types import Fact, FactScope, SceneConsensus, BeliefTuple
from npc.memory.belief_engine import BeliefEngine


class SceneConsensusCacheManager:
    """
    Manages scene-level consensus caching with TTL and invalidation.

    版本控制扩展（工程指南第4、6节）：
    - get_or_compute_consensus 调用前先过滤 is_active=True 的事实
    - invalidate_by_fact_update 收到 FactUpdatedEvent 时同时清除信念引擎缓存
    - on_scene_switch 预热时强制使用活跃版本
    """

    def __init__(self, belief_engine: BeliefEngine, default_ttl_seconds: int = 1800):
        self.belief_engine = belief_engine
        self.default_ttl_seconds = default_ttl_seconds
        self.cache: Dict[str, SceneConsensus] = {}
        self.active_scenes: Set[str] = set()

    # ------------------------------------------------------------------

    def get_or_compute_consensus(
        self,
        scene_id: str,
        facts: List[Fact],
        npc_ids: List[str],
        force_recompute: bool = False,
    ) -> SceneConsensus:
        """
        获取场景共识，必要时重新计算。

        版本控制：传入的 facts 列表在计算前过滤为 is_active=True，
        保证缓存内容永远基于最新活跃事实版本。
        """
        if not force_recompute and scene_id in self.cache:
            cached = self.cache[scene_id]
            if not cached.is_expired():
                return cached

        # 只使用活跃版本事实
        active_facts = [f for f in facts if f.is_active]
        consensus = self._compute_consensus(scene_id, active_facts, npc_ids)

        self.cache[scene_id] = consensus
        self.active_scenes.add(scene_id)
        return consensus

    def _compute_consensus(
        self,
        scene_id: str,
        facts: List[Fact],
        npc_ids: List[str],
    ) -> SceneConsensus:
        """内部：计算场景共识（仅接受已过滤的活跃事实）。"""
        consensus = SceneConsensus(scene_id=scene_id, facts=facts)
        for npc_id in npc_ids:
            npc_beliefs = {}
            for fact in facts:
                belief = self.belief_engine.compute_belief(
                    npc_id=npc_id,
                    fact_id=fact.fact_id,
                    use_cache=True,
                )
                npc_beliefs[fact.fact_id] = belief
            consensus.npc_beliefs[npc_id] = npc_beliefs
        return consensus

    def invalidate_scene(self, scene_id: str) -> None:
        """使某个场景的缓存失效。"""
        self.cache.pop(scene_id, None)
        self.active_scenes.discard(scene_id)

    def invalidate_by_fact_update(
        self,
        fact_id: str,
        affected_scenes: List[str],
    ) -> None:
        """
        收到事实更新事件时，同时：
        1. 使涉及场景的缓存失效（场景缓存标记脏数据）
        2. 清除信念引擎中该 fact_id 相关的所有 NPC 缓存（工程指南第9节）
        """
        for scene_id in affected_scenes:
            if scene_id in self.active_scenes:
                self.invalidate_scene(scene_id)

        # 清除信念引擎缓存中所有 (*, fact_id) 条目
        stale_keys = [
            k for k in self.belief_engine.belief_cache if k[1] == fact_id
        ]
        for k in stale_keys:
            del self.belief_engine.belief_cache[k]

    def on_scene_switch(self, old_scene: Optional[str], new_scene: str) -> None:
        """场景切换：移除旧场景活跃标记，注册新场景。"""
        if old_scene:
            self.active_scenes.discard(old_scene)
        self.active_scenes.add(new_scene)

    def mark_scene_active(self, scene_id: str) -> None:
        self.active_scenes.add(scene_id)

    def get_cached_consensus(self, scene_id: str) -> Optional[SceneConsensus]:
        cached = self.cache.get(scene_id)
        if cached and not cached.is_expired():
            return cached
        return None

    def clear_all(self) -> None:
        self.cache.clear()
        self.active_scenes.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "total_cached_scenes": len(self.cache),
            "active_scenes": len(self.active_scenes),
            "expired_caches": sum(1 for c in self.cache.values() if c.is_expired()),
        }


class SceneAwareMemoryRetriever:
    """
    Retrieves scene-filtered facts with precomputed beliefs.
    
    Core component for NPC response generation - provides
    scene consensus as natural language summaries.
    """
    
    def __init__(
        self,
        graph_consensus,
        belief_engine: BeliefEngine,
        cache_manager: SceneConsensusCacheManager
    ):
        """
        Initialize retriever.
        
        Args:
            graph_consensus: GlobalGraphConsensus instance
            belief_engine: BeliefEngine for belief computation
            cache_manager: SceneConsensusCacheManager for caching
        """
        self.graph = graph_consensus
        self.belief_engine = belief_engine
        self.cache_manager = cache_manager
    
    def get_scene_consensus_for_npc(
        self,
        npc_id: str,
        scene_id: str,
        facts: List[Fact],
        max_facts: int = 10,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Get consensus facts for NPC in scene.
        
        Args:
            npc_id: NPC identifier
            scene_id: Scene identifier
            facts: All facts in scene
            max_facts: Maximum facts to return
            force_recompute: Skip cache
        
        Returns:
            Dictionary with consensus facts and beliefs
        """
        # Get list of NPCs in scene (extract from facts)
        npc_ids = self._extract_npcs_from_facts(facts)
        
        # Get or compute consensus
        consensus = self.cache_manager.get_or_compute_consensus(
            scene_id=scene_id,
            facts=facts,
            npc_ids=npc_ids,
            force_recompute=force_recompute
        )
        
        # Get summary for this NPC
        summary = consensus.get_npc_consensus_summary(npc_id, max_facts)
        
        return {
            "npc_id": npc_id,
            "scene_id": scene_id,
            "consensus_facts": summary,
            "timestamp": consensus.timestamp,
            "is_cached": not force_recompute
        }
    
    def get_consensus_text_for_npc(
        self,
        npc_id: str,
        scene_id: str,
        facts: List[Fact],
        max_facts: int = 10
    ) -> str:
        """
        Get natural language consensus text for NPC.
        
        This is the text injected into system prompts.
        
        Args:
            npc_id: NPC identifier
            scene_id: Scene identifier
            facts: All facts in scene
            max_facts: Maximum facts to include
        
        Returns:
            Formatted consensus text
        """
        consensus_data = self.get_scene_consensus_for_npc(
            npc_id=npc_id,
            scene_id=scene_id,
            facts=facts,
            max_facts=max_facts
        )
        
        facts_list = consensus_data["consensus_facts"]
        
        if not facts_list:
            return "[当前世界共识]\n无已知信息\n[共识结束]"
        
        lines = ["[当前世界共识]"]
        
        for item in facts_list:
            fact_text = item["fact"]
            confidence = item["confidence"]
            uncertainty = item["uncertainty"]
            
            # Format with confidence indicator
            if uncertainty > 0.5:
                confidence_str = f"可能，但不确定"
            else:
                confidence_percent = int(confidence * 100)
                confidence_str = f"确信度 {confidence_percent}%"
            
            lines.append(f"- {fact_text} ({confidence_str})")
        
        lines.append("[共识结束]")
        
        return "\n".join(lines)
    
    def _extract_npcs_from_facts(self, facts: List[Fact]) -> List[str]:
        """Extract unique NPC identifiers from facts."""
        npcs = set()
        for fact in facts:
            if fact.source_npc:
                npcs.add(fact.source_npc)
        
        # If no NPCs found in facts, return empty list
        # (caller should provide NPC IDs separately)
        return list(npcs)


class EagerScenePreheater:
    """
    Proactively pre-computes scene consensus on scene switch.
    
    Runs async preheating to warm cache before NPCs generate responses.
    """
    
    def __init__(
        self,
        cache_manager: SceneConsensusCacheManager,
        retriever: SceneAwareMemoryRetriever
    ):
        """
        Initialize preheater.
        
        Args:
            cache_manager: Cache manager for storage
            retriever: Memory retriever for facts
        """
        self.cache_manager = cache_manager
        self.retriever = retriever
    
    async def preheat_scene_async(
        self,
        scene_id: str,
        facts: List[Fact],
        npc_ids: List[str],
    ) -> None:
        """
        异步预热场景缓存。

        版本控制：只传入 is_active=True 的事实，确保预热内容不含陈旧版本。
        """
        import asyncio
        active_facts = [f for f in facts if f.is_active]
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._preheat_worker,
            scene_id,
            active_facts,
            npc_ids,
        )

    def _preheat_worker(
        self,
        scene_id: str,
        facts: List[Fact],
        npc_ids: List[str],
    ) -> None:
        """预热工作线程。"""
        try:
            self.cache_manager.get_or_compute_consensus(
                scene_id=scene_id,
                facts=facts,
                npc_ids=npc_ids,
                force_recompute=True,
            )
        except Exception as e:
            print(f"Error preheating scene {scene_id}: {e}")


class ConsensusFactSummarizer:
    """
    Converts facts and beliefs to natural language summaries.
    
    Handles confidence formatting and uncertainty expression.
    """
    
    @staticmethod
    def summarize_fact(
        fact: Fact,
        belief: BeliefTuple,
        include_confidence: bool = True
    ) -> str:
        """
        Summarize a fact with belief into natural language.
        
        Args:
            fact: Fact to summarize
            belief: NPC's belief about fact
            include_confidence: Whether to include confidence
        
        Returns:
            Natural language summary
        """
        base_text = fact.natural_language()
        
        if not include_confidence:
            return base_text
        
        if belief.uncertainty > 0.5:
            return f"{base_text}（但不确定）"
        else:
            confidence_pct = int(belief.belief * 100)
            return f"{base_text}（确信度 {confidence_pct}%）"
    
    @staticmethod
    def summarize_facts(
        facts: List[Fact],
        beliefs: Dict[str, BeliefTuple],
        max_facts: int = 10
    ) -> List[str]:
        """
        Summarize multiple facts.
        
        Args:
            facts: Facts to summarize
            beliefs: Dictionary mapping fact_id -> belief
            max_facts: Maximum facts to include
        
        Returns:
            List of natural language summaries
        """
        summaries = []
        
        for fact in facts[:max_facts]:
            if fact.fact_id in beliefs:
                belief = beliefs[fact.fact_id]
                summary = ConsensusFactSummarizer.summarize_fact(fact, belief)
                summaries.append(summary)
        
        return summaries
