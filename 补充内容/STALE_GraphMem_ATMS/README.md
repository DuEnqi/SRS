# STALE × GraphMem-ATMS / HyperBase

**Auditable Belief-Base Revision for Stale Memory in LLM Agents**

面向 LLM Agent 长期记忆的**可审计信念修正层**：将 ATMS（Assumption-based Truth Maintenance System）、AGM 信念修正理论、Hansson kernel contraction 和 Darwiche-Pearl iterated revision 引入记忆失效推理，在 STALE benchmark 上构建了完整的评估体系。

---

## 项目概览

原始 STALE benchmark（[arXiv:2605.06527](https://arxiv.org/abs/2605.06527)）提出一个问题：*LLM agent 能否知道自己的记忆何时不再有效？* 它提供了 400 个 T1（同属性改写）/ T2（依赖传播失效）场景和 CUP-Mem 基线。

本项目的核心贡献是：

1. **GraphMem-ATMS / HyperBase**：使用 provenance graph + ATMS justification labels + belief base + kernel contraction 的形式化记忆修正内核
2. **10 大类新增测试**：将原始 STALE 从单一的"能不能识别失效"扩展为覆盖失效识别、有效保留、机制必要性、理论正确性、检索公平性、多 agent 传播的完整评估体系

---

## 版本架构

```
stale_graphmem_atms_v6.py          ← 形式化内核（不变的基础）
         │
         ├── stale_experiments_v6ext.py   ← v6 实验层（Still-Valid / 消融 / Stream）
         │         │
         │         └── stale_experiments_v7ext.py  ← v7 实验层（操作分类 / 指标分离 / ATMS stress）
         │                   │
         │                   └── stale_experiments_v8ext.py  ← v8 实验层（否定判断 / 集成分类 / 泛化拆分）
         │                             │
         │                             └── stale_experiments_v9ext.py  ← v9 实验层（检索公平 / 细粒度F1 / 基线对比）
```

**每一层 `import` 前一层，不修改前一层代码。** v9 运行时自动加载完整 v6→v7→v8→v9 链路。

---

## 目录结构

```
STALE/
│
├── README.md                              ← 本文件
│
├── # === 形式化内核 ===
├── stale_graphmem_atms_v6.py              # 3984行完整 bundle
│   └── DPLL SAT / ATMS / AGM P1-P6 / Hansson kernel contraction
│       / DP1-DP4 iterated revision / multi-agent trust matrix
│       / belief base + postulate harness
│
├── # === 实验扩展层（v6→v7→v8→v9）===
├── stale_experiments_v6ext.py             # v6: Still-Valid / 消融 / Stream
├── stale_experiments_v7ext.py             # v7: 操作分类 / 指标分离 / ATMS stress
├── stale_experiments_v8ext.py             # v8: 否定判断 / 集成分类 / 泛化拆分
├── stale_experiments_v9ext.py             # v9: 检索公平 / 细粒度F1 / 基线对比
│
├── # === 多Agent信念修正（v13 完整系统）===
├── multi_agent/
│   ├── npc_consensus_v13_updated.py       # v13 完整系统（6416行）：多NPC叙事记忆
│   │   └── ConsensusEngine / ATMSKernelV2 / GraphMemory / HanssonAuditor
│   │       / SemanticAuditor / AgentSimulation / ROSA架构对比
│   │       / trust propagation / community consensus / source independence
│   │       / echo chamber collapse / premise resistance / decay / privacy
│   ├── proto_atms_bench.py                # ATMS vs BFS-Cascade 微基准
│   ├── agm_memory.py                      # AGM/KM/credibility-limited revision
│   └── README_STALE_GraphMem.md           # 设计迁移映射
│
├── # === 运行脚本 ===
├── run_stream_benchmark.py                # 流式 benchmark 运行
├── run_stream_eval.py                     # 流式 LLM judge 评估
├── run_v9_llm_n100.py                     # v9 LLM 实验 n=100 启动器
│
├── # === 实验结果（离线可复现）===
├── results/
│   ├── ext_v9_llm_n100/                   # 最新 LLM 全量结果 (n=100)
│   ├── ext_v7_llm_n100/                   # v7 全量结果 (n=100)
│   └── ext_llm/                           # v6 全量结果
│
├── # === 文档 ===
├── docs/
│   ├── README_STALE_hyperbase.md
│   ├── README_STALE_hyperbase v3.md
│   ├── README_v5.md
│   ├── README_v6.md
│   ├── README_answer_pipeline.md
│   ├── README_promptonly_variant.md
│   ├── RESEARCH_REVIEW_STALE.md
│   ├── SUPPLEMENT_RESEARCH_REVIEW_STALE.md
│   └── ANALYSIS_SUMMARY.json
│
├── # === 参考基线 ===
└── STALE-main/                            # 官方 STALE benchmark + CUP-Mem
    ├── STALE/                             # 官方 STALE 实现
    └── cup_mem/                           # CUP-Mem 实现
```

---

## 快速开始

### 离线自检（无需 API）

```bash
# 形式化内核自检 (31/31)
python stale_graphmem_atms_v6.py --self-test

# v9 实验层自检 (15/15)
python stale_experiments_v9ext.py --self-test

# ATMS 必要性微基准
python multi_agent/proto_atms_bench.py

# AGM/DP 公理测试
python stale_graphmem_atms_v6.py --postulates
```

### 离线实验（确定性，无需 API）

```bash
# 运行所有 v9 模块（离线 floor）
python stale_experiments_v9ext.py --all --out runs/ext_v9 --n-per-family 60

# 单独运行
python stale_experiments_v9ext.py --retrieval-fairness     # W1
python stale_experiments_v9ext.py --op-f1                   # W2
python stale_experiments_v9ext.py --baselines               # W4
```

### LLM 实验（需要 API key）

```bash
# 设置环境变量
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://yunwu.ai/v1"

# v9 全量 LLM 实验
python run_v9_llm_n100.py

# 或单独运行
python stale_experiments_v9ext.py --retrieval-fairness --use-llm --n-per-family 100

# 流式 STALE 全量 benchmark
python run_stream_benchmark.py --bundle v6 --use-llm \
    --icds-path STALE-main/STALE/outputs/MAIN.json --out runs/my_run
```

---

## 在原始 STALE 之上新增的 10 大类测试

> 原始 STALE benchmark 仅测试"能否识别失效"（400 个 T1/T2 场景 + 三维度评分）。
> 本项目将其扩展为覆盖 10 个维度的完整评估体系。

---

### 1. Still-Valid / 假阳性控制

**测什么**：系统会不会把仍然有效的记忆误删？

原始 STALE 只关心"能不能检测出 stale"，没有测试"会不会过度失效化仍然有效的记忆"。

| 测试场景 | 含义 |
|----------|------|
| M_new IRRELEVANT to M_old | 不相关的新信息不应该让旧记忆失效 |
| M_new SUPPORTS / RESTATES M_old | 强化旧记忆的新信息不应该让它失效 |
| M_new is only SUPPLEMENTARY | 补充信息不是取代旧记忆 |
| M_new from LOW-CREDIBILITY source | 低可信度来源不应覆盖高可信度旧记忆 |
| Alternative-support still valid | 其他独立支持路径仍在，结论不应失效 |

**指标**：Retention accuracy、False invalidation rate、Old-valid precision

**运行**：
```bash
python stale_experiments_v6ext.py --still-valid
```

---

### 2. 7类细粒度操作分类

**测什么**：不只是 keep/update 二元判断，而是精确区分 7 种信念修正操作。

原始 STALE 只区分 T1（同属性改写）和 T2（依赖传播失效），没有细粒度的操作类型标签。

| 操作类型 | 含义 | 示例 |
|----------|------|------|
| **UPDATE** | 旧值被新值取代 | "我现在开车通勤" → 旧值"骑车"失效 |
| **REINFORCE** | 新信息强化旧值 | "我仍然每天骑车" → 旧值更可信 |
| **SUPPLEMENT** | 新信息补充旧值 | "我周末也骑车" → 旧值不变，只是多了周末 |
| **NO_EFFECT** | 新信息与旧值无关 | "我换了手机" → 通勤方式不变 |
| **TEMPORARY** | 临时改变 | "这周因为下雨坐公交" → 临时替代 |
| **RECOVERY** | 从临时状态恢复 | "天晴了，恢复骑车" → 回到旧值 |
| **REVERT** | 直接回退 | "放弃了开车，重新骑车" → 明确回退 |

**指标**：Macro-F1、Per-class precision/recall/F1、Confusion matrix

**v7 → v9 的关键修复**：
- v7 deterministic: macro-F1 = 0.719，SUPPLEMENT F1 = 0.000（全部塌缩为 REINFORCE）
- v9 improved: macro-F1 = 0.990，SUPPLEMENT F1 = 1.000（通过 additive connective + topical overlap 门控修复）

**运行**：
```bash
python stale_experiments_v9ext.py --op-f1
```

---

### 3. ATMS 必要性微基准

**测什么**：证明 ATMS justification labels 不可被简单的 BFS 依赖级联替代。

原始 STALE 没有任何机制层面的测试——无法区分是 ATMS 在起作用，还是随便一个图算法也能做到。

| 测试用例 | Full-ATMS | BFS-Cascade | 说明 |
|----------|:---:|:---:|------|
| 单路径收缩 | ✅ 100% | ✅ 100% | 基础 case，两者都能处理 |
| **替代路径保留** | ✅ 100% | ❌ 0% | BFS 沿着一条路径全部删除，ATMS 识别其他支持路径 |
| 多来源撤回 | ✅ 100% | ✅ 100% | 撤回一条证据，其他来源仍在 |
| nogood 冲突 | ✅ 100% | ✅ 100% | 互斥声明标记冲突 |
| 时序更新 | ✅ 100% | ✅ 100% | 时间有效性过期 |
| 规范变化 | ✅ 100% | ✅ 100% | 规则本身改变 |
| **多 Agent 局部视角** | ✅ 100% | ❌ 0% | BFS 无法建模 agent-local 的支持环境 |
| **Core-Retention** | ✅ 100% | ❌ 0% | 20 个真实 core_retain 决策，非空集默认 |

**关键结论**：替代路径保留和多 agent 视角是 BFS 级联的**原理性盲区**——这不是调参能解决的，而是 ATMS label semantics 独有的能力。

**运行**：
```bash
python multi_agent/proto_atms_bench.py
# 或在 v6 中
python stale_graphmem_atms_v6.py --atms-bench
```

---

### 4. 检索公平性三层对比 🔥

**测什么**：在纯 retrieval-library 设定下（不给系统暴露 privileged M_old/M_new 字段），系统的真实性能是多少？

这是对 reviewer 核心质疑的直接回应：如果系统在 answer prompt 中显式传入 M_old/M_new，那就不是真正的 memory system——因为 real-world deployment 中不存在"标注好的哪条记忆被挑战了"。

| Tier | 检索方式 | 角色 |
|------|----------|------|
| **ORACLE** | 偷看 gold turn ids | 性能**上界**：如果检索完美，系统能到多少？ |
| **SEMANTIC** | 从 3-session 对话日志中检索（old+new 埋入 distractors），离线是 lexical retriever，`--use-llm` 升级为 LLM retriever | **真实系统**：这是你应该引用的数字 |
| **KEYWORD** | 固定 last-turn + overlap 规则 | 离线**下界**：最朴素的规则能到多少？ |

**每条记录的结构**：
```
[Session 0]  ... distractor turns ...
             "I bike 10 miles to work every day."  ← 旧事实（埋入）
             ... distractor turns ...
[Session 1]  ... distractor turns ...
             "I've started driving since I moved."  ← 新事件（埋入）
             ... distractor turns ...
[Session 2]  ... more distractors ...
             (有时有 trailing distractor 让 "最后一轮" 不是新事件)
Query: "What's my current commute mode?"
```

**双层指标**：
- **检索质量**：old recall、new recall、joint recall（系统是否找到了正确的会话轮？）
- **端到端决策**：retention accuracy、false invalidation rate、invalidation recall（检索到的 pair 上裁决表现如何？）

**离线 floor (n=60/family)**：

| Tier | Joint Retrieval | Decision Recall | False-Inval |
|------|:---:|:---:|:---:|
| ORACLE | 1.00 | 0.85 | 0.017 |
| **SEMANTIC** | 0.60 | 0.77 | 0.103 |
| KEYWORD | 0.25 | 0.48 | 0.106 |

**关键结论**：决策层随检索质量下降而**平滑退化**——错误可分解为"检索没找到"vs"找到了但裁决错了"，两个问题是可分离的。

**运行**：
```bash
# 离线 floor
python stale_experiments_v9ext.py --retrieval-fairness --n-per-family 60

# LLM 版（SEMANTIC tier 升级为 LLM retriever）
python stale_experiments_v9ext.py --retrieval-fairness --use-llm --n-per-family 100
```

---

### 5. 9 种记忆基线策略对比

**测什么**：你的系统 vs 所有主流记忆策略的 head-to-head 对比。

原始 STALE 只比了 CUP-Mem 一个基线。这里新增了 8 种额外策略，每种都是对**已发表决策策略的忠实复现**（明确标注不是原作者代码）。

| # | 策略 | 核心逻辑 | Retention | Recall | False-Inval | 点评 |
|---|------|---------|:---:|:---:|:---:|------|
| 1 | **flat-RAG** | 平铺检索，不做 staleness 推理 | 1.000 | 0.000 | 0.000 | 永远不会发现失效 |
| 2 | **recency-priority** | 时序优先，最新即当前 | 0.833 | 0.783 | 0.167 | recovery 被误读为更新 |
| 3 | **credibility-decay** | 可信度随时间衰减 | 0.833 | 0.783 | 0.167 | 与 recency 类似 |
| 4 | **naive-overwrite** | 直接覆盖，有变化就更新 | 0.317 | 1.000 | **0.683** | 召回完美但保留崩溃 |
| 5 | **BFS-cascade** | 沿依赖图 BFS 级联失效 | 0.819 | 0.783 | 0.181 | 在 transient+recovery 上误触发 |
| 6 | **A-MEM graph** | 图笔记风格 | 0.833 | 0.717 | 0.167 | supplement 被误覆盖 |
| 7 | **Zep bi-temporal KG** | 双时序知识图谱 | 1.000 | 0.633 | 0.000 | 精确但召回有限 |
| 8 | **CUP-Mem (STALE)** | STALE 原始基线 | 1.000 | 0.717 | 0.000 | 同上：用精确关键词门控 |
| 9 | **pure-LLM** | 无结构化记忆 | 0.500 | 0.783 | 0.500 | 对任何表面线索做出反应 |
| — | **Oracle upper bound** | 完美抽取上界 | 1.000 | 1.000 | 0.000 | 天花板 |
| — | **OURS (hybrid)** | 形式化+集成全系统 | **0.983** | **0.883** | **0.017** | 唯一在 Pareto 前沿上 |

**关键结论**：Zep 和 CUP-Mem 保持干净保留但用精确关键词限制了召回(0.63-0.72)；naive-overwrite 用摧毁保留换召回。**Ours 是唯一在 recall × retention Pareto 前沿上的方法**——通用更新检测器捕获了关键词基线错过的改写更新，集成保护了 keep family。

**运行**：
```bash
python stale_experiments_v9ext.py --baselines --n-per-family 60
```

---

### 6. 形式化内核消融

**测什么**：每个形式化模块的边际贡献。不是"都差不多"的空消融。

| 消融 arm | 去掉什么 | 在什么 case 上掉分 |
|----------|---------|-------------------|
| **no_formal_heuristic** | ATMS+SAT+incision+multi-agent 全部去掉，只用 surface-cue | T2 传播失效全挂 |
| **raw_history_only** | 形式化层，只扫描近期历史 | 隐式冲突识别不到 |
| **M_new_only** | 形式化层，只扫描 M_new | 无法对比 old vs new |
| **no_SAT_core** | DPLL SAT 证明引擎 | disjunctive defeat 无法处理 |
| **no_multi_agent** | 多 agent 层 | echo-chamber 无法检测 |
| **formal_only** | LLM 自由阅读，只用形式化模板 | 自然语言变体无法处理 |

**运行**：
```bash
python stale_experiments_v6ext.py --formal-ablation
```

---

### 7. Prompt 消融

**测什么**：v6 的 surgical PR fix 和 structured IPA 各自贡献多少？

| 消融 arm | 说明 |
|----------|------|
| **full** | 完整 v6 prompt |
| **-PR-detect-then-reject** | 去掉 Dim2 的 detect-then-reject，回退到 v2 的自由生成 |
| **-structured-IPA** | 去掉 Dim3 的结构化 "(1) old invalid → (2) new state → (3) 2-3 actions" |
| **-recent-emphasis** | 去掉近期信息的强调 |
| **-explicit-M_old/M_new-fields** | 去掉显式标注旧值/新值的字段 |
| **+force-formal-verdict** | 强制注入形式化判决（类似 v3 的错误方向） |

**诚实声明**：`-recent-emphasis`、`-explicit-fields`、`+force-verdict` 在离线模式下是 inert 的（只影响 LLM prompt 措辞，不影响确定性模板），表中标记为 "LLM-only"。

**运行**：
```bash
python stale_experiments_v6ext.py --v6-ablation
```

---

### 8. 多轮流式记忆

**测什么**：系统能否在连续时间线中维护一致记忆状态？原始 STALE 是单次快照。

20 轮记忆流：
```
set → update(stale) → no_effect → T2-break → recovery → reconfirm → revert
```

末尾混合探测：
- 当前状态查询
- 历史状态查询（"at that time?" 时间性探针）
- 跨轮一致性检查

**运行**：
```bash
python stale_experiments_v6ext.py --stream
```

---

### 9. AGM/DP 公理可证伪测试

**测什么**：记忆修正操作是否符合信念修正理论的基本理性约束？不是贴标签，而是**能失败的测试**。

| 公理 | 含义 | 测试方式 |
|------|------|---------|
| **AGM P1 (Closure)** | 修正后应保持逻辑闭包 | 检查修正后 base 是否 entail 期望的结论 |
| **AGM P2 (Inclusion)** | K\*φ ⊆ K+φ | 修正结果不应超出 expansion |
| **AGM P3 (Vacuity)** | 如果 ¬φ∉K，则 K\*φ = K+φ | 不矛盾时不收缩 |
| **AGM P4 (Consistency)** | 如果 φ 一致，则 K\*φ 一致 | 修正后一致性保持 |
| **AGM P5 (Extensionality)** | 如果 ⊢ φ↔ψ，则 K\*φ = K\*ψ | 等价命题修正结果相同 |
| **AGM P6 (Preservation)** | 如果 ¬φ∉K，则 K ⊆ K\*φ | 不矛盾时保留所有旧信念（含 ¬φ∉K guard） |
| **DP1-DP4** | Darwiche-Pearl iterated revision | 迭代修正的理性约束 |
| **Broken operator control** | 故意构造一个违反公理的操作符 | **验证 harness 能抓住** |

**关键**：Broken operator control 的 Vacuity 只有 13%——harness 成功识别了有问题的操作符，证明这套测试不是 `or True` 式的虚假通过。

**运行**：
```bash
python stale_graphmem_atms_v6.py --postulates
python stale_graphmem_atms_v6.py --theory-bench    # DP1-DP4 + DEL + extensions
```

---

### 10. 多 Agent 记忆传播专项（v13 完整系统）

**测什么**：当记忆跨越多个 agent 传播时，能否正确处理来源独立性、回声室效应和局部视角？

原始 STALE 仅考虑单 agent 场景。v13（`multi_agent/npc_consensus_v13_updated.py`，**6416 行完整系统**）提供了完整的多 agent 信念修正实现。

| 测试 | 含义 | Full-ATMS | BFS |
|------|------|:---:|:---:|
| **Source independence** | 同源多次转述 ≠ 多独立来源 | ✅ | ❌ |
| **Echo chamber collapse** | 多人重复同一来源不应强化信任 | ✅ | ❌ |
| **Local perspective** | agent 局部视角 ≠ community 共识 ≠ world ledger | ✅ | ❌ |
| **False consensus** | 错误信念在群体中传播 | 可检测 | 不可检测 |
| **Privacy leakage** | 私有信息不应泄露到公共共识 | access tier 控制 | 无 |
| **Trust-weighted propagation** | 低信任 agent 的信息应被门控 | ✅ | ❌ |
| **Premise resistance** | agent 不应被虚假前提操控 | ✅ | ❌ |

**v13 核心模块一览**：

| 模块 | 类/函数 | 功能 |
|------|---------|------|
| **ATMSKernelV2** | `ATMSKernelV2` (多独立 assumption / nogood / 时序 / OR-AND-defeasible) | 核心 justification 引擎，比 v6 的 ATMSKernel 增强了多 agent 上下文支持 |
| **ConsensusEngine** | `ConsensusEngine` (~1500行) | 多 agent 共识引擎：trust matrix、community consensus、personality-gated propagation、FAMA |
| **GraphMemory** | `GraphMemory` | 类型化节点/边 schema：Session / Claim / Attribute / Premise / Policy / ModalClaim |
| **HanssonAuditor** | `HanssonAuditor` | AGM/Hansson post-hoc 合规审计：Success / Inclusion / Vacuity / Recovery / Cut / Core-Retainment |
| **SemanticAuditor** | `SemanticAuditor` | 12 类语义审计：stale retraction / retain keep / false invalid / echo collapse / consensus drift 等 |
| **AgentSimulation** | `AgentSimulation` + `AutonomousAgent` | 自治 agent 多轮模拟：5 级 freedom、效用函数驱动决策、状态预测 |
| **ROSA 架构对比** | `RosaTrustArch` / `RosaHiRAGArch` / `RosaBaseArch` | 三种多 agent 架构的 head-to-head 基准对比 |
| **Pipeline baselines** | `run_pipeline_baselines()` | FAMA / semantic / graph / hybrid 四种 pipeline 对比 |
| **BeliefOperationSelector** | `BeliefOperationSelector` | 根据证据和冲突类型选择 UPDATE / REVISE / CONTRACT / IGNORE / SUPPLEMENT / REVERT 操作 |
| **IncisionFunction** | `IncisionFunction` | Hansson 4 步 incision：切 defeasible → 切低可信度 → 保护替代支持 → 保留历史禁用当前 |
| **Second-story** | `run_second_story()` | 泛化测试：在全新的叙事领域（非骑士故事）验证多 agent 机制的领域无关性 |
| **Decay / Adversarial / Reproducibility / Overhead** | `decay_test()`, `adversarial_sweep()`, `reproducibility_test()`, `overhead_test()` | 鲁棒性四件套 |

**v13 已有的验证结果**：
- `proto_atms_bench.py`：Full-ATMS 100% vs BFS-Cascade 70.97%，alternative path 100% vs 0%，multi-agent 100% vs 0%
- `agm_memory.py`：所有 sanity / AGM / vacuity / recovery failure / update-vs-revision / non-prioritized write / kernel consolidation checks 通过
- `npc_consensus_v13_updated.py` 自身含完整自检和 demo 模式

**运行**：
```bash
# ATMS 微基准
python multi_agent/proto_atms_bench.py

# AGM 基础测试
python multi_agent/agm_memory.py

# v13 完整系统（离线 demo）
python multi_agent/npc_consensus_v13_updated.py --demo

# v13 含 LLM
python multi_agent/npc_consensus_v13_updated.py --use-llm --outdir out_v13

# v13 消融矩阵
python multi_agent/npc_consensus_v13_updated.py --ablation all
```

---

## 实验结果汇总

### STALE 全量 benchmark（400 样本，gpt-4o-mini judge）

| 方法 | T1 Overall | T2 Overall | Combined |
|------|:---:|:---:|:---:|
| **stale_graphmem v6** | **98.5%** | **92.3%** | **95.4%** |
| CUP-Mem (官方基线) | 84.7% | 86.7% | 85.7% |
| stale_graphmem v2 | 68.0% | 62.5% | 65.2% |
| v6 formal-only baseline | 95.8% | 89.0% | 92.4% |

> ⚠️ **重要**：v6 的 95.4% 是在 answer prompt 中明示 M_old/M_new 的条件下取得的（与 STALE 官方 target-model 设定不完全等价）。v9 W1 已解决此问题——见检索公平性三层对比。

### v9 离线 floor (n=60/family)

| 模块 | 指标 | 数值 |
|------|------|:---:|
| **W1 检索公平性** | SEMANTIC tier decision recall | 0.77 |
| | SEMANTIC tier false-inval | 0.103 |
| **W2 操作分类** | v7 deterministic macro-F1 | 0.719 |
| | **v9 improved macro-F1** | **0.990** |
| | v9 LLM macro-F1 (n=100) | 0.868 |
| **W4 基线对比** | Ours retention | 0.983 |
| | Ours recall | **0.883** |
| | Ours false-inval | **0.017** |
| | Best baseline binary-F1 (CUP-Mem) | 0.906 |
| | **Ours binary-F1** | **0.936** |

### ATMS 微基准

| 测试 | Full-ATMS | BFS-Cascade |
|------|:---:|:---:|
| Overall | **100%** | 70.97% |
| Alternative path retention | **100%** | 0% |
| Multi-agent perspective | **100%** | 0% |
| Core-Retention | **100%** | 0% |

---

## 诚实边界 / 已知局限

| 局限 | 说明 |
|------|------|
| **离线 SEMANTIC retriever 是 lexical 的** | 离线模式使用 embedding-free 的词汇匹配，非真实 dense embedding。`--use-llm` 升级为 LLM retriever，但生产级 embedding retriever 介于 offline lexical floor 和 oracle ceiling 之间 |
| **W2 label 是 gold-by-construction** | 7 类操作标签由模板构造，非人工标注。macro-F1 应在人工标注集上重新测量后再做强声明 |
| **v6 answer prompt 仍暴露 M_old/M_new** | v9 W1 在检索层解决了公平性，但 v6 主实验在 answer 层仍暴露 privileged info。v9 的 SEMANTIC tier 是真正公平的数字 |
| **baselines 是策略复现，非原作者代码** | W4 的 9 种基线是对已发表决策策略的忠实复现（用于受控对比），不是原作者的系统/代码 |
| **LLM judge 单一** | 当前只用 gpt-4o-mini 作为 judge。完整验证需第二 judge 模型 + 人工盲审校准 |
| **更新改写召回仍是共享瓶颈** | 所有方法（包括 ours）在 novel UPDATE paraphrase 上的召回仍有限。通用更新检测器缩小了但未关闭这个缺口 |

---

## 引用

本项目建立在以下工作之上：

- **STALE Benchmark**: [arXiv:2605.06527](https://arxiv.org/abs/2605.06527) — "STALE: State-Tracking And Long-Context Evaluation for LLM Agent Memory"
- **CUP-Mem**: STALE 官方基线 — structured memory consolidation with propagation-aware search
- **SSGM**: [arXiv:2603.11768](https://arxiv.org/abs/2603.11768) — Self-Governed Memory with verification/decay/access control
- **NeuSymMS**: [arXiv:2605.17596](https://arxiv.org/abs/2605.17596) — Neuro-Symbolic Subject-Relation-Value Memory with lifecycle governance
- **de Kleer ATMS** (1986): Assumption-based Truth Maintenance System
- **Hansson** (1994): Kernel Contraction
- **Darwiche & Pearl** (1997): Iterated Belief Revision (DP1-DP4)
- **AGM** (Alchourrón, Gärdenfors, Makinson, 1985): Belief Revision postulates

---

## 运行环境

- Python 3.10+
- `numpy`（SAT 求解器使用）
- `ijson`（流式加载 STALE 数据集）
- LLM 模式额外需要 `openai` 和有效的 API key

---

## License

本项目基于 STALE-main（MIT License）构建。新增代码遵循相同许可证。
