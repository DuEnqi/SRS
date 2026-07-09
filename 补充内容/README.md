# 补充内容

与 `E:\STALE_upload\实验文件\` 互补。合集覆盖 v6/v7/v8/v9 形式化内核、v13 多Agent、组友图存储+时间戳、组友 STALE 接口。

## 目录

```
STALE_GraphMem_ATMS/
├── core/                                    # v6 → v9 形式化内核 + 实验层
│   ├── stale_graphmem_atms_v6.py            # DPLL SAT / ATMS / AGM / Hansson
│   ├── stale_experiments_v6ext.py           # Still-Valid / ablation / stream
│   ├── stale_experiments_v7ext.py           # Operation classifier / metric separation
│   ├── stale_experiments_v8ext.py           # Negation-aware / ensemble / generalization
│   └── stale_experiments_v9ext.py           # W1 retrieval fairness / W2 F1 / W4 baselines
├── multi_agent/                             # v13 多Agent 信念修正
│   ├── npc_consensus_v13_updated.py         # ATMSKernelV2 / ConsensusEngine / GraphMemory
│   ├── proto_atms_bench.py                  # ATMS necessity micro-benchmark
│   └── agm_memory.py                        # AGM/KM revision
├── npc/memory/                              # 组友 图存储 + 时间戳
│   ├── core_types.py                        # Fact / NPCKnows / NPCTrusts / BeliefTuple
│   ├── belief_engine.py                     # 3路信念计算 (A/B/C)
│   ├── graph_consensus.py                   # InMemoryGraphDatabase + 版本链
│   ├── fact_extractor.py
│   └── scene_consensus_cache.py
├── tests/stale/                             # 组友 STALE 标准化接口
│   ├── stale_types.py                       # STALEAdapter ABC
│   ├── stale_adapters.py                    # SimpleLLM / NPCMemory 基线
│   ├── stale_evaluator.py
│   ├── stale_judge.py
│   └── stale_dataset_generator.py
└── runners/                                 # 运行脚本
    ├── run_stream_benchmark.py
    ├── run_stream_eval.py
    └── run_v9_llm_n100.py
```

## 与 实验文件 的分工

| 本文件夹 | 实验文件 |
|---------|---------|
| 形式化内核 (v6-v9) | v3 检索器+回答引擎 |
| 多Agent 信念修正 (v13) | 融合适配器各版本 |
| 组友图存储+时间戳 | 全部 judge 结果 |
| 组友 STALE 接口 | v1/v2/v3 跑分汇总 |

## Coding Agent 任务

见 `ARCHITECTURE.md`。
