# STALE × GraphMem-ATMS

把 `npc_consensus_v12.py` 的整套设计迁移到 **STALE** benchmark
(<https://github.com/icedreamc/STALE>) 上, 测试 LLM agent 的「记忆失效推理」能力。

单文件实现: `stale_graphmem_atms.py` (无外部依赖即可离线运行; numpy 可选, LLM 可选)。

## 为什么 STALE 是这套机制的天然落点

STALE 考的是「agent 能否知道自己的记忆何时不再有效」。每个样本:

- `M_old` : 旧用户状态 (例: 我每天骑车 10 英里通勤)
- `M_new` : 以 *隐式* 方式让 M_old 失效的新状态 (无显式否定)
- `haystack_session` : 把 M_old / M_new 埋进一长串噪声 session

三个探测维度: **SR**(dim1 旧值是否仍有效) / **PR**(dim2 虚假前提抵抗) / **IPA**(dim3 任务遵循新状态);
两种冲突: **T1** 同属性改写 / **T2** 经 `A→B` 依赖传播失效。

## 设计迁移映射 (npc_consensus_v12 → STALE)

| v12 机制 | STALE 落点 |
|---|---|
| 类型化节点/边 schema | `Session/Claim/Attribute/Premise/Policy` 节点 + 白名单校验边 |
| 统一记忆状态机 `STATUS_CAPS` | `ACTIVE/WEAK/STALE/SUPERSEDED/UNKNOWN_CURRENT/REFUTED` + 能力表 |
| 访问控制分层 | `volatile/episodic/profile/core` 控制覆盖门槛 |
| 人格化选择性传播 | 证据/来源信任加权覆盖 |
| 局部依赖级联 cascade | 常识 `A→B` 依赖图 → T2 传播失效局部 BFS |
| ATMS 内核 | Claim 级 justification 超边 × 最小一致支持环境(label) × nogood × 可废止守卫 |
| AGM/KM 操作 | T1→UPDATE(KM, 历史保留); T2→CONTRACTION(无替换) |
| Hansson postulates + incision | 核收缩 / 核保留(Core-Retainment) / σ incision function |

三维作答由形式化层给出 (确定性, 无需 LLM):

- **SR**  : `atms.is_supported(B_old)` 是否仍成立
- **PR**  : 虚假前提命题在内核里是否仍被支持, 不支持即识破
- **IPA** : 取当前 valid 的 `ModalClaim` 作为新状态依据

## 用法 (全部离线可跑)

```bash
python stale_graphmem_atms.py --self-test       # 16 项单元自检 (核收缩/核保留/三维/状态机)
python stale_graphmem_atms.py --core-demo       # ATMS 核保留 vs 核收缩 端到端演示
python stale_graphmem_atms.py --demo            # 6 个合成样本 (T1+T2) 端到端演示
python stale_graphmem_atms.py --synth --n 24 --out runs/synth   # 合成 STALE 数据集 + 跑 + 本地判分

# 直接消费真实 STALE MAIN.json (确定性形式化作答)
python stale_graphmem_atms.py --icds-path STALE/outputs/demo_T1_MAIN.json --out runs/real
# 形式化层 + LLM 润色 (需 OPENAI_API_KEY)
python stale_graphmem_atms.py --icds-path ... --use-llm
```

## 输出 (与 STALE 兼容)

- `answers.json` — STALE `run_target_model` 格式 (`uid` + `target_model_responses{dim1/2/3}`),
  可直接喂官方 `Evaluation/full_eval_performance.py` 做 LLM judge 复核。
- `eval.json` — 本地确定性判分 (SR/PR/IPA per type + Overall + Hansson 合规率)。
- `traces.jsonl` — 每样本的图/ATMS/incision/cascade/Hansson 决策留痕。
- `synth_MAIN.json` — (`--synth` 时) 生成的 STALE 格式数据集, 可喂官方 `run_target_model.py`。

## 已验证

- `--self-test`: 16/16 通过。
- 合成集 SR/PR/IPA = 100%, Hansson 五条假设合规率 = 1.0。
- 去掉 `graph_hint`/`conflict_type` 的 real-like 路径仍能从原文正确推断 T1/T2 并作答。

## 重要说明 (诚实边界)

1. **确定性抽取是关键词驱动的**, 对真正隐式的冲突并非万能; 真实 STALE 数据的推荐路径是 `--use-llm`
   (LLM 抽取属性/取值/冲突类型, 形式化层做最终维护与作答)。
2. **本地判分是 cue-based 代理**, 用于离线快速回归; 权威分数应把 `answers.json` 交给 STALE 官方
   LLM judge (`full_eval_performance.py`) 评定。
3. 形式化结论 (旧值是否仍有效) 始终以 ATMS/Hansson 为准; `--use-llm` 模式下 LLM 仅润色, 不得把
   已失效的旧假设重新当作当前为真。
