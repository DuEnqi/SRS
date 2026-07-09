#!/usr/bin/env python3
"""Stream-run STALE full benchmark without loading the 300MB JSON into memory."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import ijson

ROOT = Path(__file__).resolve().parent
STALE_BUNDLES = {
    "v2": ROOT / "stale_graphmem_atms_v2.py",
    "v3": ROOT / "stale_graphmem_atms_v3.py",
    "v4": ROOT / "stale_graphmem_atms_v4.py",
    "v5": ROOT / "stale_graphmem_atms_v5.py",
    "v6": ROOT / "stale_graphmem_atms_v6.py",
    "promptonly": ROOT / "stale_graphmem_atms_promptonly.py",
}
NATIVE_ANSWER_BUNDLES = frozenset({"v4", "v5", "v6", "promptonly"})
CUP_MEM_ROOT = ROOT / "STALE-main"
DEFAULT_ENV_FILE = CUP_MEM_ROOT / "STALE" / ".env"
DEFAULT_OPENAI_BASE_URL = "https://yunwu.ai/v1"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_key:
            continue
        if (
            normalized_value.startswith(("'", '"'))
            and normalized_value.endswith(("'", '"'))
            and len(normalized_value) >= 2
        ):
            normalized_value = normalized_value[1:-1]
        os.environ.setdefault(normalized_key, normalized_value)


def load_default_env() -> None:
    load_env_file(DEFAULT_ENV_FILE)


def get_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def select_sessions(rec: Dict[str, Any], session_mode: str) -> Dict[str, Any]:
    if session_mode != "relevant_only":
        return rec
    sessions = list(rec.get("haystack_session", []) or [])
    timestamps = list(rec.get("timestamps", []) or [])
    rel = rec.get("relevant_session_index", [])
    if not (isinstance(rel, list) and rel and sessions):
        return rec
    indices = [int(i) for i in rel if 0 <= int(i) < len(sessions)]
    out = dict(rec)
    out["haystack_session"] = [sessions[i] for i in indices]
    out["timestamps"] = [timestamps[i] if i < len(timestamps) else "" for i in indices]
    return out


def format_haystack(haystack_sessions: List[Any], timestamps: Optional[List[str]] = None) -> str:
    formatted_history = ""
    for idx, session in enumerate(haystack_sessions or []):
        if not session:
            continue
        time_str = f" [Time: {timestamps[idx]}]" if timestamps and idx < len(timestamps) else ""
        formatted_history += f"\n=== Session {idx + 1}{time_str} ===\n"
        for turn in session:
            role = "User" if turn.get("role") == "user" else "Assistant"
            formatted_history += f"{role}: {turn.get('content', '')}\n"
    return formatted_history


def build_target_llm(
    *,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Callable[[str, str], str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    def _call(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    return _call


def build_llm_runtime() -> Dict[str, str]:
    load_default_env()
    api_key = get_env("OPENAI_API_KEY", "YUNWU_API_KEY")
    base_url = get_env("OPENAI_BASE_URL", default=DEFAULT_OPENAI_BASE_URL)
    model = get_env("STALE_GRAPHMEM_MODEL", "TARGET_MODEL", default="gpt-4o-mini")
    if not api_key:
        raise ValueError("No API key found. Set OPENAI_API_KEY in STALE-main/STALE/.env")
    return {"api_key": api_key, "base_url": base_url, "model": model}


def formal_context(res: Any) -> str:
    b_old = f"{getattr(res, 'attribute_b', '')}={getattr(res, 'value_old', '')}"
    return (
        f"conflict_type={getattr(res, 'conflict_type', '')}\n"
        f"attribute_b={getattr(res, 'attribute_b', '')}\n"
        f"value_old={getattr(res, 'value_old', '')}\n"
        f"value_new={getattr(res, 'value_new', '')}\n"
        f"upstream_a={getattr(res, 'upstream_a', '')}\n"
        f"operation={getattr(res, 'operation', '')}\n"
        f"old_claim={b_old}\n"
        f"old_supported_terminal={getattr(res, 'old_supported_terminal', '')}\n"
        f"sr_should_invalidate={getattr(res, 'sr_should_invalidate', '')}"
    )


def answer_with_llm(
    rec: Dict[str, Any],
    llm: Callable[[str, str], str],
    *,
    answer_mode: str,
    formal_res: Any = None,
) -> Dict[str, str]:
    history_text = format_haystack(rec.get("haystack_session", []), rec.get("timestamps", []))
    queries = rec.get("probing_queries", {}) or {}
    formal_block = ""
    if answer_mode == "hybrid" and formal_res is not None:
        formal_block = (
            "\n\n[Formal memory adjudication — authoritative for whether the old state still holds]\n"
            f"{formal_context(formal_res)}"
        )

    responses: Dict[str, str] = {}
    for dim_key, query_key in (
        ("dim1", "dim1_query"),
        ("dim2", "dim2_query"),
        ("dim3", "dim3_query"),
    ):
        query_text = str(queries.get(query_key, "") or "")
        if dim_key == "dim3":
            system_prompt = (
                "You are a helpful assistant with long-term memory of the user. "
                "Read the conversation history carefully, then respond to the user's latest query directly. "
                "Structure your answer in three steps: (1) state the old assumption is no longer valid; "
                "(2) describe the new state or upstream change; (3) give concrete actionable advice based on the new state."
            )
            user_prompt = (
                f"[Conversation History]\n{history_text}"
                f"{formal_block}\n\n[Latest Query]\n{query_text}"
            )
        elif dim_key == "dim2":
            system_prompt = (
                "You are a helpful assistant with long-term memory of the user. "
                "If the question contains an outdated premise, you MUST reject it explicitly and "
                "must not build recommendations on the stale assumption."
            )
            user_prompt = (
                f"[Conversation History]\n{history_text}"
                f"{formal_block}\n\n[Question]\n{query_text}"
            )
        else:
            system_prompt = (
                "You are a helpful assistant with long-term memory of the user. "
                "Read the conversation history carefully, then accurately answer the question."
            )
            user_prompt = (
                f"[Conversation History]\n{history_text}"
                f"{formal_block}\n\n[Question]\n{query_text}"
            )
        responses[f"{dim_key}_response"] = llm(system_prompt, user_prompt)
    return responses


def iter_records(path: Path, *, filter_type: str = "") -> Iterator[Dict[str, Any]]:
    want = (filter_type or "").strip().upper()
    with path.open("rb") as f:
        for item in ijson.items(f, "item"):
            rec = dict(item)
            rec["conflict_type"] = rec.get("conflict_type") or rec.get("type") or ""
            if want and str(rec.get("conflict_type", "")).upper() != want:
                continue
            yield rec


def load_answers(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    return {str(row["uid"]): row for row in rows if row.get("uid")}


def save_answers(path: Path, answers: List[Dict[str, Any]], *, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {"target_model": model_name, "num_items": len(answers)},
        "data": answers,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_stale_bundle(name: str) -> Path:
    key = (name or "v2").strip().lower()
    if key not in STALE_BUNDLES:
        raise ValueError(f"Unknown bundle {name!r}; choose from {sorted(STALE_BUNDLES)}")
    path = STALE_BUNDLES[key]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_stale_module(bundle: str):
    bundle_path = resolve_stale_bundle(bundle)
    spec = importlib.util.spec_from_file_location(bundle_path.stem, bundle_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, bundle


def load_stale_engine(
    mod,
    bundle_key: str,
    mode: str,
    extraction: str,
    llm: Optional[Callable[[str, str], str]],
    *,
    pipeline: str = "",
):
    kwargs: Dict[str, Any] = {"mode": mode, "extraction_mode": extraction, "llm": llm}
    if bundle_key == "promptonly":
        kwargs["pipeline"] = pipeline or "prompt_only"
    elif bundle_key == "v5":
        kwargs["pipeline"] = pipeline or "v5"
    elif bundle_key == "v6":
        kwargs["pipeline"] = pipeline or "v6"
    elif pipeline:
        kwargs["pipeline"] = pipeline
    try:
        return mod.StaleEngine(**kwargs)
    except TypeError:
        kwargs.pop("pipeline", None)
        return mod.StaleEngine(**kwargs)


def run_stale_graphmem(
    data_path: Path,
    out_path: Path,
    *,
    bundle: str,
    mode: str,
    extraction: str,
    filter_type: str,
    max_samples: int,
    use_llm: bool,
    answer_mode: str,
    session_mode: str,
    fresh: bool,
    pipeline: str = "",
) -> None:
    mod, bundle_key = load_stale_module(bundle)
    runtime = build_llm_runtime() if use_llm else None
    llm_extract: Optional[Callable[[str, str], str]] = None
    llm_answer: Optional[Callable[[str, str], str]] = None
    native_answer = bundle_key in NATIVE_ANSWER_BUNDLES
    model_name = f"stale_graphmem_{bundle_key}[{mode}/{extraction}]"
    native_tag = {
        "promptonly": "prompt_only",
        "v5": "v5",
        "v6": "v6",
    }.get(bundle_key, "native")

    if use_llm:
        llm_extract = mod.build_llm_callable(
            model=runtime["model"],
            base_url=runtime["base_url"],
        )
        if llm_extract is None:
            raise RuntimeError("Failed to initialize extraction LLM client.")
        if native_answer:
            model_name = (
                f"stale_graphmem_{bundle_key}[{mode}/{extraction}+LLM:{runtime['model']}/{native_tag}]"
            )
        else:
            llm_answer = build_target_llm(
                model=runtime["model"],
                api_key=runtime["api_key"],
                base_url=runtime["base_url"],
            )
            model_name = (
                f"stale_graphmem_{bundle_key}[{mode}/{extraction}+LLM:{runtime['model']}/{answer_mode}]"
            )

    engine = load_stale_engine(
        mod,
        bundle_key,
        mode,
        extraction,
        llm_extract if use_llm else None,
        pipeline=pipeline,
    )
    done = {} if fresh else load_answers(out_path)
    answers = list(done.values())
    processed = 0
    started = time.perf_counter()

    for rec in iter_records(data_path, filter_type=filter_type):
        if max_samples and processed >= max_samples:
            break
        uid = str(rec.get("uid"))
        if uid in done:
            continue

        work_rec = select_sessions(rec, session_mode)
        gm, atms, bb, ma, res = engine.adjudicate(work_rec)
        if native_answer and use_llm:
            responses = engine.answer(gm, atms, bb, ma, res, work_rec.get("probing_queries", {}))
        elif use_llm and answer_mode in {"llm", "hybrid"}:
            responses = answer_with_llm(
                work_rec,
                llm_answer,
                answer_mode=answer_mode,
                formal_res=res if answer_mode == "hybrid" else None,
            )
        else:
            responses = engine.answer(gm, atms, bb, ma, res, work_rec.get("probing_queries", {}))

        row = {
            "uid": uid,
            "target_model": model_name,
            "target_model_responses": responses,
            "type": rec.get("conflict_type") or rec.get("type"),
        }
        answers.append(row)
        done[uid] = row
        processed += 1
        save_answers(out_path, answers, model_name=model_name)
        print(f"[{len(answers)}] uid={uid} type={row['type']} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    print(json.dumps({"answers_path": str(out_path), "num_items": len(answers)}, ensure_ascii=False, indent=2))


def run_cup_mem(
    data_path: Path,
    out_path: Path,
    *,
    session_mode: str,
    filter_type: str,
    max_samples: int,
    embedding_model_path: str,
    fresh: bool,
) -> None:
    sys.path.insert(0, str(CUP_MEM_ROOT))
    from cup_mem import CupMemEngine, PipelineThresholds, TraceConfig
    from cup_mem.run_cup_mem import (
        build_llm_client,
        build_runtime_config,
        resolve_embedding_model_path,
        validate_inputs,
    )

    class Args:
        wire_api = "auto"
        chat_supported = "auto"
        enable_debug_trace = False

    runtime = build_runtime_config(Args())
    emb_path = resolve_embedding_model_path(embedding_model_path)
    validate_inputs(data_path, emb_path)

    run_dir = out_path.parent / ".run_cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = build_llm_client(
        model=str(runtime["model"]),
        api_key=str(runtime["api_key"]),
        base_url=str(runtime["base_url"]),
        cache_dir=run_dir / ".cache",
        wire_api=str(runtime["wire_api"]),
        chat_supported=runtime["chat_supported"],
    )
    engine = CupMemEngine(
        llm=llm,
        embedding_model_path=str(emb_path),
        embedding_device="cpu",
        thresholds=PipelineThresholds(),
        trace_config=TraceConfig(enable_debug_trace=False),
    )

    done = {} if fresh else load_answers(out_path)
    answers = list(done.values())
    model_name = f"CUPMem[{runtime['model']}]"
    processed = 0
    started = time.perf_counter()

    dim_map = {
        "dim1_query": "dim1_response",
        "dim2_query": "dim2_response",
        "dim3_query": "dim3_response",
    }

    for sample_index, rec in enumerate(iter_records(data_path, filter_type=filter_type)):
        if max_samples and processed >= max_samples:
            break
        uid = str(rec.get("uid"))
        if uid in done:
            continue
        result = engine.run_sample(rec, sample_index=sample_index, session_mode=session_mode)
        query_logs = result.get("query_logs", {}) or {}
        responses = {}
        for qk, rk in dim_map.items():
            log = query_logs.get(qk, {})
            responses[rk] = str(((log.get("answer") or {}).get("answer")) or "")
        row = {
            "uid": uid,
            "target_model": model_name,
            "target_model_responses": responses,
            "type": rec.get("conflict_type") or rec.get("type"),
        }
        answers.append(row)
        done[uid] = row
        processed += 1
        save_answers(out_path, answers, model_name=model_name)
        print(f"[{len(answers)}] uid={uid} type={row['type']} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    print(json.dumps({"answers_path": str(out_path), "num_items": len(answers)}, ensure_ascii=False, indent=2))


def split_answers_by_type(answers_path: Path, out_dir: Path) -> None:
    payload = json.loads(answers_path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    by_type: Dict[str, List[Dict[str, Any]]] = {"T1": [], "T2": []}
    for row in rows:
        t = str(row.get("type") or "").upper()
        if t in by_type:
            by_type[t].append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, items in by_type.items():
        save_answers(out_dir / f"answers_{t}.json", items, model_name=payload.get("summary", {}).get("target_model", ""))
    print(json.dumps({k: len(v) for k, v in by_type.items()}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream-run full STALE benchmark.")
    p.add_argument("--method", required=True, choices=["stale_graphmem", "cup_mem", "split"])
    p.add_argument("--bundle", default="v2", choices=sorted(STALE_BUNDLES))
    p.add_argument("--data-path", default=str(ROOT / "data" / "STALE" / "T1_T2_400_FULL.json"))
    p.add_argument("--output-path", required=True)
    p.add_argument("--mode", default="ours")
    p.add_argument("--extraction", default="llm", choices=["oracle", "llm", "hand_schema"])
    p.add_argument(
        "--answer-mode",
        default="hybrid",
        choices=["formal", "polish", "llm", "hybrid", "native"],
        help="v4/v5/v6 use native LLM answer pipelines; v2/v3 support hybrid haystack+formal",
    )
    p.add_argument("--use-llm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fresh", action="store_true", help="Ignore existing answers and rerun from scratch.")
    p.add_argument("--session-mode", default="relevant_only", choices=["relevant_only", "full"])
    p.add_argument("--filter-type", default="", choices=["", "T1", "T2"])
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--embedding-model-path", default="")
    p.add_argument("--split-dir", default="")
    p.add_argument(
        "--pipeline",
        default="",
        choices=["", "prompt_only", "verified", "v5", "v6"],
        help="promptonly: prompt_only|verified; v5/v6: default pipeline when --bundle v5|v6",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    out_path = Path(args.output_path).resolve()

    if args.method == "split":
        split_answers_by_type(out_path, Path(args.split_dir or out_path.parent))
        return
    if args.method == "stale_graphmem":
        answer_mode = args.answer_mode
        if args.bundle in NATIVE_ANSWER_BUNDLES and args.use_llm:
            answer_mode = "native"
        elif not args.use_llm:
            answer_mode = "formal"
        run_stale_graphmem(
            data_path,
            out_path,
            bundle=args.bundle,
            mode=args.mode,
            extraction=args.extraction,
            filter_type=args.filter_type,
            max_samples=args.max_samples,
            use_llm=args.use_llm,
            answer_mode=answer_mode,
            session_mode=args.session_mode,
            fresh=args.fresh,
            pipeline=args.pipeline,
        )
        return
    run_cup_mem(
        data_path,
        out_path,
        session_mode=args.session_mode,
        filter_type=args.filter_type,
        max_samples=args.max_samples,
        embedding_model_path=args.embedding_model_path,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
