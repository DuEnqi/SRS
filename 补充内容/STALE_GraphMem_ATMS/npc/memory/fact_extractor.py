#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fact extraction from dialogue using LLM.

Extracts structured facts from conversation turns and writes them to the global graph.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import re

from npc.memory.core_types import Fact, FactScope, ExtractionResult


class FactExtractor:
    """
    Extracts structured facts from dialogue using LLM.
    
    Takes dialogue context and produces structured fact triplets
    (subject, predicate, object) with confidence scores.
    """
    
    EXTRACTION_PROMPT = """请从以下对话中提取事实信息。

对话记录：
{dialogue}

请提取所有重要的事实，包括：
- NPC的状态变化
- 发生的事件
- 设定的目标或任务
- 人物关系的变化
- 环境或场景的变化
- 其他重要信息

对于每个事实，请提供：
1. 主语 (subject): 事实涉及的对象
2. 谓语 (predicate): 事实的属性或动作
3. 宾语 (object): 属性值或结果
4. 置信度 (confidence): 0.0-1.0
5. 范围 (scope): "global"（全局可见）或 "scene"（仅当前场景可见）

返回JSON格式的事实列表：
[
  {
    "subject": "...",
    "predicate": "...",
    "object": "...",
    "confidence": 0.95,
    "scope": "global"
  }
]

只返回JSON数组，不要添加其他文本。"""
    
    def __init__(self):
        try:
            from npc.llm_config.llm import get_llm
            self.llm = get_llm()
        except Exception:
            self.llm = None
    
    def extract_facts(
        self,
        dialogue_turns: List[Dict[str, str]],
        scene_id: Optional[str] = None,
        npc_ids: Optional[List[str]] = None
    ) -> ExtractionResult:
        """
        Extract facts from dialogue turns.
        
        Args:
            dialogue_turns: List of dialogue turns with {"speaker": str, "content": str}
            scene_id: Scene where dialogue occurred
            npc_ids: NPCs participating in dialogue
        
        Returns:
            ExtractionResult with extracted facts
        """
        if not self.llm:
            return ExtractionResult(
                facts=[],
                confidence=0.0,
                dialogue_turn=len(dialogue_turns),
                source_npcs=npc_ids or [],
                scene_id=scene_id
            )
        
        # Format dialogue for LLM
        dialogue_text = self._format_dialogue(dialogue_turns)
        
        # Call LLM to extract facts
        prompt = self.EXTRACTION_PROMPT.format(dialogue=dialogue_text)
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            facts_data = self._parse_extraction_response(response_text)
            
            # Convert to Fact objects
            facts = []
            for fact_data in facts_data:
                fact = self._create_fact_from_extraction(
                    fact_data,
                    scene_id=scene_id,
                    source_npcs=npc_ids or []
                )
                if fact:
                    facts.append(fact)
            
            # Create result
            result = ExtractionResult(
                facts=facts,
                confidence=0.8,
                dialogue_turn=len(dialogue_turns),
                source_npcs=npc_ids or [],
                scene_id=scene_id
            )
            
            return result
        
        except Exception as e:
            print(f"Error extracting facts: {e}")
            return ExtractionResult(
                facts=[],
                confidence=0.0,
                dialogue_turn=len(dialogue_turns),
                source_npcs=npc_ids or [],
                scene_id=scene_id
            )
    
    def _format_dialogue(self, dialogue_turns: List[Dict[str, str]]) -> str:
        """Format dialogue for LLM."""
        lines = []
        for turn in dialogue_turns:
            speaker = turn.get("speaker", "Unknown")
            content = turn.get("content", "")
            lines.append(f"{speaker}: {content}")
        
        return "\n".join(lines)
    
    def _parse_extraction_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract facts."""
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
            
            # Parse JSON
            facts = json.loads(response_text)
            
            if isinstance(facts, list):
                return facts
            else:
                return []
        
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response_text[:200]}")
            return []
    
    def _create_fact_from_extraction(
        self,
        fact_data: Dict[str, Any],
        scene_id: Optional[str] = None,
        source_npcs: Optional[List[str]] = None
    ) -> Optional[Fact]:
        """Convert extracted fact data to Fact object."""
        try:
            subject = fact_data.get("subject", "").strip()
            predicate = fact_data.get("predicate", "").strip()
            obj = fact_data.get("object", "").strip()
            confidence = float(fact_data.get("confidence", 0.8))
            scope_str = fact_data.get("scope", "global").lower()
            
            if not (subject and predicate and obj):
                return None
            
            # Convert scope
            scope = FactScope.GLOBAL if scope_str == "global" else FactScope.SCENE
            
            # Create fact ID
            fact_id = self._generate_fact_id(subject, predicate, obj)
            
            fact = Fact(
                fact_id=fact_id,
                subject=subject,
                predicate=predicate,
                obj=obj,
                confidence=confidence,
                scope=scope,
                scene_id=scene_id,
                source_npc=source_npcs[0] if source_npcs else None
            )
            
            return fact
        
        except Exception as e:
            print(f"Error creating fact: {e}")
            return None
    
    def _generate_fact_id(self, subject: str, predicate: str, obj: str) -> str:
        """Generate unique fact ID."""
        import hashlib
        combined = f"{subject}_{predicate}_{obj}"
        hash_val = hashlib.md5(combined.encode()).hexdigest()[:8]
        return f"fact_{hash_val}"


class DialogueMemoryBuffer:
    """
    Buffer for storing recent dialogue for fact extraction.
    
    Maintains a sliding window of dialogue turns for context-aware fact extraction.
    """
    
    def __init__(self, max_turns: int = 20):
        """
        Initialize dialogue buffer.
        
        Args:
            max_turns: Maximum dialogue turns to keep
        """
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []
    
    def add_turn(self, speaker: str, content: str) -> None:
        """Add a dialogue turn to buffer."""
        self.turns.append({
            "speaker": speaker,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
    
    def get_recent_turns(self, num_turns: int = 10) -> List[Dict[str, str]]:
        """Get the most recent turns for extraction."""
        return self.turns[-num_turns:] if num_turns > 0 else self.turns
    
    def clear(self) -> None:
        """Clear buffer."""
        self.turns.clear()
    
    def get_all(self) -> List[Dict[str, str]]:
        """Get all turns in buffer."""
        return self.turns.copy()


class FactExtractionManager:
    """
    High-level manager for fact extraction and graph writing.

    版本控制扩展（工程指南第3节）：
    _extract_and_write_facts 在写入前执行版本匹配逻辑：
    - subject+predicate 相同、object 不同 → 新版本替代旧版本
    - 完全相同三元组 → 幂等写入（write_fact 内部处理）
    - 直接目击者的 KNOWS 边携带当前活跃版本节点 ID
    """

    def __init__(self, graph_consensus):
        self.extractor = FactExtractor()
        self.graph = graph_consensus
        self.dialogue_buffer = DialogueMemoryBuffer()

    def process_dialogue_turn(
        self,
        speaker: str,
        content: str,
        scene_id: str,
        npc_ids: List[str],
    ) -> List[str]:
        """处理单轮对话：加入缓冲区，每3轮触发一次提取。"""
        self.dialogue_buffer.add_turn(speaker, content)
        if len(self.dialogue_buffer.turns) % 3 == 0:
            return self._extract_and_write_facts(scene_id, npc_ids)
        return []

    def _extract_and_write_facts(
        self,
        scene_id: str,
        npc_ids: List[str],
    ) -> List[str]:
        """
        从最近对话中提取事实并版本化写入全局图。

        版本匹配流程（工程指南第3节）：
        1. 对每个提取到的事实，在活跃事实中查找
           subject+predicate 相同的版本。
        2. 若 object 不同 → 新版本写入（write_fact 内部创建版本链）。
        3. 若完全相同但置信度更高 → write_fact 内部处理覆盖。
        4. 为直接目击 NPC 写入携带当前活跃版本节点 ID 的 KNOWS 边。
        5. 若 NPC 之前有同 fact_id 的旧版本 KNOWS 边，标记为 is_stale。
        """
        recent_turns = self.dialogue_buffer.get_recent_turns(num_turns=10)
        if not recent_turns:
            return []

        result = self.extractor.extract_facts(
            recent_turns,
            scene_id=scene_id,
            npc_ids=npc_ids,
        )

        written_ids: List[str] = []
        backend = self.graph.backend

        for fact in result.facts:
            # ---- 版本匹配：检查是否已存在 subject+predicate 相同的活跃事实 ----
            if hasattr(backend, "find_active_facts_by_subject_predicate"):
                existing = backend.find_active_facts_by_subject_predicate(
                    fact.subject, fact.predicate
                )
                if existing:
                    old_fact = existing[0]
                    if old_fact.obj != fact.obj:
                        # object 不同 → 延续旧 fact_id，write_fact 创建新版本
                        fact.fact_id = old_fact.fact_id
                    # else: object 相同，write_fact 按置信度决定是否覆盖

            # ---- 版本化写入全局图 ----
            fact_id, event = self.graph.write_fact(fact)
            if not event:
                # 无变更（置信度更低），跳过
                continue
            written_ids.append(fact_id)

            # ---- 获取刚写入的活跃版本节点 ID ----
            active_fact = self.graph.get_active_fact(fact_id)
            active_vnid = active_fact.version_node_id if active_fact else f"{fact_id}_v1"

            # ---- 为直接目击 NPC 写入 KNOWS 边 ----
            for npc_id in npc_ids:
                # 检查 NPC 是否已有该 fact_id 的旧版本信念边，若有则标记 stale
                old_edge = backend.get_knows_edge(npc_id, fact_id)
                if old_edge and old_edge.version_node_id != active_vnid:
                    old_edge.is_stale = True   # 原地标记陈旧（下次信念计算触发 B 路刷新）

                # 写入指向当前活跃版本的新 KNOWS 边
                self.graph.write_npc_knows_edge(
                    npc_id=npc_id,
                    fact_id=fact_id,
                    direct=True,
                    version_node_id=active_vnid,
                )

        return written_ids

    def force_extraction(self, scene_id: str, npc_ids: List[str]) -> List[str]:
        """强制立即提取当前缓冲区内的事实。"""
        return self._extract_and_write_facts(scene_id, npc_ids)
