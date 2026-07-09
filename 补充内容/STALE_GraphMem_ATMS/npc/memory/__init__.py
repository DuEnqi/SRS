#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global consensus memory architecture for MultiNPC.

Provides unified world state knowledge graph with scene-aware filtering,
belief computation through subjective logic, and efficient NPC retrieval.
"""

from npc.memory.core_types import (
    Fact,
    FactScope,
    BeliefTuple,
    NPCKnows,
    NPCTrusts,
    FactUpdatedEvent,
    SceneConsensus,
    ExtractionResult,
    BeliefStatus,
)

from npc.memory.graph_consensus import (
    GlobalGraphConsensus,
    GraphDatabaseBackend,
    InMemoryGraphDatabase,
)

from npc.memory.belief_engine import (
    SubjectiveLogicEngine,
    BeliefEngine,
    BeliefMetrics,
)

from npc.memory.fact_extractor import (
    FactExtractor,
    FactExtractionManager,
    DialogueMemoryBuffer,
)

from npc.memory.scene_consensus_cache import (
    SceneConsensusCacheManager,
    SceneAwareMemoryRetriever,
    EagerScenePreheater,
    ConsensusFactSummarizer,
)

__all__ = [
    # Core types
    "Fact",
    "FactScope",
    "BeliefTuple",
    "NPCKnows",
    "NPCTrusts",
    "FactUpdatedEvent",
    "SceneConsensus",
    "ExtractionResult",
    "BeliefStatus",
    # Graph
    "GlobalGraphConsensus",
    "GraphDatabaseBackend",
    "InMemoryGraphDatabase",
    # Belief engine
    "SubjectiveLogicEngine",
    "BeliefEngine",
    "BeliefMetrics",
    # Fact extraction
    "FactExtractor",
    "FactExtractionManager",
    "DialogueMemoryBuffer",
    # Scene caching
    "SceneConsensusCacheManager",
    "SceneAwareMemoryRetriever",
    "EagerScenePreheater",
    "ConsensusFactSummarizer",
]
