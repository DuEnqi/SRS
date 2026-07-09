# SRS — GraphMem-ATMS × ForCadia/SRS

完整可运行的 Multi-NPC 叙事平台：React 前端 + FastAPI 后端 + 形式化信念图。

## Vercel 部署

仓库已包含 `vercel.json` + `pyproject.toml` + `api/index.py`（FastAPI 入口 shim）。

1. 在 [Vercel](https://vercel.com) 导入 `DuEnqi/SRS`，**Root Directory 留空**（仓库根目录）
2. Framework Preset 选 **Other**（或自动识别）
3. 环境变量（Settings → Environment Variables）：
   - `OPENAI_API_KEY` / `AZURE_OPENAI_API_KEY`（可选，LLM 对话）
   - `LLM_PROVIDER=azure` 等（见 `.env.example`）
4. Deploy

构建流程：编译 `tmp_SRS` → 复制到 `public/` → FastAPI 处理 `/api/*` → 静态 SPA 处理其余路由。

本地预览生产构建：

```powershell
cd tmp_SRS
npm run build
npm run preview
```

**终端 1 — 后端**

```powershell
cd SRS_融合
pip install -r ../requirements.txt
python srs_api_v13.py --port 8765
```

**终端 2 — 前端**

```powershell
cd tmp_SRS
npm install
npm run dev
```

浏览器打开 **http://localhost:5173** → Play / Memory Graph / Dashboard 等全部页面可用。

> 后端未启动时前端自动降级为 mock 模式；配置 `.env` 后 LLM 对话可用。

## 配置 LLM（可选）

```powershell
copy .env.example .env
# 编辑 .env 填入 AZURE_OPENAI_API_KEY 或 OPENAI_API_KEY
```

## 单文件静态版（无需 dev server）

```powershell
cd tmp_SRS
npm install
npm run build
npm run preview
```

打开 **http://localhost:4173**（需同时运行后端以使用 GraphMem API）。

## 仓库结构

| 路径 | 说明 |
|------|------|
| `SRS_融合/` | FastAPI 后端 (`srs_api_v13.py`) |
| `tmp_SRS/` | React + Vite + Zustand 前端 |
| `补充内容/STALE_GraphMem_ATMS/` | 信念/图核心类型 |
| `unified_stale.py` | ATMS / v9 形式化层 |
| `llm_env.py` | LLM 环境变量加载 |

详细 API 见 `SRS_融合/README.md`。
