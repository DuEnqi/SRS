# SRS 融合 — GraphMem-ATMS × ForCadia/SRS

完整对接 [ForCadia/SRS](https://github.com/ForCadia/SRS) 前端 + v6–v9 形式化层 + v13 图结构 + 组友时间戳后端。

## 架构

```
tmp_SRS 前端 (React/Zustand)
    ↓ fetch localhost:8765
srs_api_v13.py  (FastAPI 统一 API)
    ↓
game_engine_v13.py
    ├── ATMSKernelV2 (assert_evidence, nogood)
    ├── V9 classify_fine_v9
    ├── Hansson 4-step incision
    ├── GraphMemory nodes/edges (live sync)
    ├── ConsensusEngine (propagation)
    └── 组友 Fact / NPCKnows / NPCTrusts / SubjectiveLogicEngine
srs_llm.py  (证据块 + LLM 角色台词)
```

## 启动

```powershell
# 1. 后端（项目根 .env 配置 LLM）
cd E:\STALE_upload\SRS_融合
pip install fastapi uvicorn pydantic
python srs_api_v13.py --port 8765

# 2. 前端
cd E:\STALE_upload\tmp_SRS
npm install
npm run dev
# 浏览器打开 http://localhost:5173
```

## API 端点

| 端点 | SRS 功能 |
|------|----------|
| `GET /api/state` | 启动加载：npcs + memory graph + day/turn + monitor |
| `POST /api/npc/generate` | Play 玩家对话（形式化 + LLM） |
| `POST /api/npc/dialogue` | NPC↔NPC 自动对话 |
| `POST /api/conflict/resolve` | Memory Graph Conflict Simulator |
| `POST /api/event/propagate` | 事件传播 |
| `POST /api/time/advance` | 换天 |
| `POST /api/time/turn` | 推进 turn |
| `POST /api/belief/update` | 信念更新 |
| `POST /api/memory/update` | 记忆节点写入 |
| `POST /api/trust/update` | 信任更新 |

## 文件

| 文件 | 说明 |
|------|------|
| `game_engine_v13.py` | 唯一引擎（Fact 版本链 + 图 + ATMS + 共识） |
| `srs_api_v13.py` | 统一 FastAPI 后端 |
| `srs_llm.py` | LLM 对话（读取项目 `.env`） |
| `unified_belief.py` | UnifiedBelief 数据模型 |
| `fusion_engine.py` | STALE 评测用 V1/V2 引擎（保留） |
| `srs_api.py` | 旧版 API（已被 v13 取代，保留参考） |

## NPC 数据

引擎从 `tmp_SRS/src/mock/npc.json` 加载，与 GitHub 前端 mock 一致（Thomas/Duran/Mila/Gareth…）。

## 测试

```bash
cd E:\STALE_upload\SRS_融合
python game_engine_v13.py
curl http://localhost:8765/api/health
curl http://localhost:8765/api/state
```
