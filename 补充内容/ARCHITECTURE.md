# GraphMem-ATMS 全栈融合架构 — Coding Agent 参考手册

## 一、总览

本文件夹 + `E:\STALE_upload\实验文件\` 合在一起 = 完整代码 + 完整结果。

本文件夹：核心代码（v6-v9 形式化层、v13 多Agent、组友图存储+时间戳、组友 STALE 接口）
实验文件：v3 检索器/回答引擎、融合适配器各版本、所有 judge 结果

---

## 二、已有资产清单

| 层 | 文件 | 核心内容 |
|------|------|---------|
| **v6 形式化内核** | `STALE_GraphMem_ATMS/core/stale_graphmem_atms_v6.py` | DPLL SAT、ATMS、AGM P1-P6、Hansson kernel contraction、DP1-DP4、BeliefBase |
| **v6ext** | `stale_experiments_v6ext.py` | Still-Valid suite、prompt ablation、formal core ablation、20-turn stream |
| **v7ext** | `stale_experiments_v7ext.py` | 7-class classifier、metric separation、ATMS stress |
| **v8ext** | `stale_experiments_v8ext.py` | Negation-aware polarity、ensemble classifier、generalization split |
| **v9ext** | `stale_experiments_v9ext.py` | W1 retrieval fairness、W2 7-class F1、W4 9-baselines |
| **v13** | `multi_agent/npc_consensus_v13_updated.py` (6416行) | ATMSKernelV2、ConsensusEngine、GraphMemory、HanssonAuditor |
| **proto_atms** | `multi_agent/proto_atms_bench.py` | 7类 case × Full-ATMS vs BFS-Cascade |
| **agm_memory** | `multi_agent/agm_memory.py` | AGM/KM/credibility-limited revision |
| **组友 图+时间戳** | `npc/memory/` | InMemoryGraphDatabase、BeliefEngine(3路A/B/C)、SubjectiveLogic、Fact/NPCKnows/NPCTrusts |
| **组友 STALE 接口** | `tests/stale/` | STALEAdapter ABC、SimpleLLM、NPCMemory、STALEEvaluator、LLMJudge |

---

## 三、核心融合点

### 3.1 两组信念修正需要互通

```
ATMSKernelV2.label(claim)              BeliefEngine.compute_belief(npc, fact)
Hansson 4-step incision                 Version chain: v1→v2→v3

目标: BeliefEngine B/C 路改为调用 ATMS label 计算
```

### 3.2 时间戳互不相认

```
ATMS Assumption2.valid_from/valid_to    Fact.created_at / NPCKnows.last_updated
二元过滤（valid_at 是/否）              离散判断（is_active True/False）

目标: 统一时间模型 → 连续衰减 + 状态机自动跃迁
```

### 3.3 融合后的信念生命周期

```
1. Fact 创建 → created_at=now, version=1
2. NPC 获知 → NPCKnows edge, last_updated=now
3. 新证据 → 新版本 version=2, ATMS label 重算, Hansson incision
4. 时间流逝 → 30天 WEAK → 90天 STALE → 180天 HISTORICAL
5. 被 supersede → 立即 SUPERSEDED
```

### 3.4 关键缺失（需要实现）

- **时间衰减函数**: `temporal_weight(elapsed_days, half_life=30d)`
- **ATMS ↔ BeliefEngine 桥接**: assumption → Fact 互转
- **时间感知 incision**: 4步中加入时间陈旧度/新鲜度
- **自动状态跃迁**: 时间驱动 WEAK→STALE→HISTORICAL

---

## 四、与 实验文件 对照

| 本文件夹有 | 实验文件补充 |
|-----------|------------|
| v6-v9 内核 | v3 检索器+回答引擎 (`code/改版/`) |
| v13 多Agent | 融合适配器 (`code/新版融合/`, `code/融合/`) |
| 组友图存储+时间戳 | v2 baseline 适配器 |
| 组友 STALE 接口 | 所有 judge 结果 (`results/`) |
| — | v1/v2/v3 各版本跑分 |
