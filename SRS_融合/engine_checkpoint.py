#!/usr/bin/env python3
"""Engine state snapshot / rollback for LLM failure recovery."""
from __future__ import annotations

import copy
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from game_engine_v13 import GameEngineV13


def checkpoint(engine: "GameEngineV13") -> Dict[str, Any]:
    """Deep-copy mutable engine state before a risky mutation."""
    return {
        "state_version": engine.state_version,
        "npc_beliefs": copy.deepcopy(dict(engine.npc_beliefs)),
        "facts": copy.deepcopy(engine.facts),
        "beliefs": copy.deepcopy(engine.beliefs),
        "knows_edges": copy.deepcopy(engine.knows_edges),
        "trust_edges": copy.deepcopy(engine.trust_edges),
        "graph_nodes": copy.deepcopy(engine.graph_nodes),
        "graph_edges": copy.deepcopy(engine.graph_edges),
        "events_log": copy.deepcopy(engine.events_log),
        "activity_feed": copy.deepcopy(engine.activity_feed),
        "propagation_queue": copy.deepcopy(engine.propagation_queue),
        "pending_memory_updates": copy.deepcopy(engine.pending_memory_updates),
        "npc_private_memory": copy.deepcopy(dict(engine.npc_private_memory)),
        "npcs": copy.deepcopy(engine.npcs),
        "consensus_history": copy.deepcopy(engine.consensus.consensus_history),
    }


def rollback(engine: "GameEngineV13", snap: Dict[str, Any]) -> None:
    """Restore engine from a checkpoint."""
    engine.state_version = snap["state_version"]
    engine.npc_beliefs.clear()
    engine.npc_beliefs.update(snap["npc_beliefs"])
    engine.facts = snap["facts"]
    engine.beliefs = snap["beliefs"]
    engine.knows_edges = snap["knows_edges"]
    engine.trust_edges = snap["trust_edges"]
    engine.graph_nodes = snap["graph_nodes"]
    engine.graph_edges = snap["graph_edges"]
    engine.events_log = snap["events_log"]
    engine.activity_feed = snap["activity_feed"]
    engine.propagation_queue = snap["propagation_queue"]
    engine.pending_memory_updates = snap["pending_memory_updates"]
    engine.npc_private_memory.clear()
    engine.npc_private_memory.update(snap["npc_private_memory"])
    engine.npcs = snap["npcs"]
    engine.consensus.consensus_history = snap["consensus_history"]
