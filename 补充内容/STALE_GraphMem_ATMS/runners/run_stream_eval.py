#!/usr/bin/env python3
"""Stream-build ground-truth index for STALE eval without loading full dataset."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import ijson

ROOT = Path(__file__).resolve().parent
STALE_DIR = ROOT / "STALE-main" / "STALE"
sys.path.insert(0, str(STALE_DIR))
sys.path.insert(0, str(STALE_DIR / "Evaluation"))

from dotenv import load_dotenv

for env_path in (STALE_DIR / ".env", ROOT / ".env"):
    if env_path.exists():
        load_dotenv(env_path)

from full_eval_performance import (  # noqa: E402
    add_usage_stats,
    async_evaluate_all_dimensions,
    finalize_usage_stats,
    get_query_text,
    init_usage_stats,
    load_records,
    safe_div,
    validate_answer_record,
    validate_dataset_record,
)
from Generation.clients import get_Async_client  # noqa: E402

try:
    from tqdm.asyncio import tqdm
except ImportError:
    class _AsyncTqdmFallback:
        @staticmethod
        async def gather(*aws, desc=None):
            return await asyncio.gather(*aws)

    tqdm = _AsyncTqdmFallback()


def build_gt_index(dataset_path: Path, conflict_type: str, answer_uids: set[str]) -> dict:
    want = conflict_type.strip().upper()
    index: dict = {}
    with dataset_path.open("rb") as f:
        for item in ijson.items(f, "item"):
            t = (item.get("type") or item.get("conflict_type") or "").strip().upper()
            if t != want:
                continue
            uid = item.get("uid")
            if uid not in answer_uids:
                continue
            validate_dataset_record(item)
            index[uid] = item
    return index


async def run_stream_evaluation(
    answers_path: Path,
    dataset_path: Path,
    output_path: Path,
    model_method: str,
    conflict_type: str,
    concurrency: int = 3,
    judge_model: str | None = None,
    judge_provider: str = "OPENAI",
) -> None:
    eval_model = judge_model or os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    eval_client = get_Async_client(judge_provider)
    if eval_client is None:
        raise RuntimeError("Judge client is not configured. Set OPENAI_API_KEY / OPENAI_BASE_URL.")

    print("Loading answers...")
    answers = load_records(str(answers_path))
    for item in answers:
        validate_answer_record(item)

    answer_uids = {a["uid"] for a in answers}
    answer_dict = {a["uid"]: a for a in answers}

    print(f"Streaming ground truth for {conflict_type} ({len(answer_uids)} uids)...")
    gt_index = build_gt_index(dataset_path, conflict_type, answer_uids)
    missing = answer_uids - set(gt_index)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} ground-truth records for {conflict_type}")

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    tasks = []
    for ans in answers:
        uid = ans["uid"]
        info = gt_index[uid]
        old_info = info.get("old_info", info.get("M_old", ""))
        responses = ans.get("target_model_responses", {})
        tasks.append(
            async_evaluate_all_dimensions(
                semaphore,
                uid,
                conflict_type,
                old_info,
                info.get("M_new", ""),
                info.get("explanation", ""),
                get_query_text(info, "dim1"),
                responses.get("dim1_response", ""),
                get_query_text(info, "dim2"),
                responses.get("dim2_response", ""),
                get_query_text(info, "dim3"),
                responses.get("dim3_response", ""),
                eval_client,
                eval_model,
            )
        )

    print(f"Firing {len(tasks)} judge API calls (concurrency={concurrency})...")
    run_start = time.perf_counter()
    results = await tqdm.gather(*tasks, desc="Judging in progress")
    run_wall_clock_seconds = time.perf_counter() - run_start

    dims = ["dim1", "dim2", "dim3"]
    accuracy_stats = {
        conflict_type: {d: {"correct": 0, "total": 0} for d in dims}
    }
    target_stats = {
        conflict_type: {d: init_usage_stats() for d in [*dims, "overall"]}
    }
    judge_stats = {conflict_type: init_usage_stats()}
    final_results_log = []

    for uid, task_type, eval_res, judge_meta in results:
        ans = answer_dict.get(uid, {})
        target_model_meta = ans.get("target_model_meta", {}) or {}
        final_results_log.append(
            {
                "uid": uid,
                "task_type": task_type,
                "evaluation": eval_res,
                "target_model_meta": target_model_meta,
                "judge_meta": judge_meta,
            }
        )

        for dim_key in dims:
            passed = bool(eval_res.get(f"{dim_key}_eval", {}).get("pass", False))
            accuracy_stats[task_type][dim_key]["total"] += 1
            if passed:
                accuracy_stats[task_type][dim_key]["correct"] += 1

        for dim_key in dims:
            dim_meta = target_model_meta.get(f"{dim_key}_meta", {}) or {}
            add_usage_stats(
                target_stats[task_type][dim_key],
                dim_meta.get("elapsed_seconds", 0.0),
                dim_meta.get("usage", {}),
            )
            add_usage_stats(
                target_stats[task_type]["overall"],
                dim_meta.get("elapsed_seconds", 0.0),
                dim_meta.get("usage", {}),
            )

        add_usage_stats(
            judge_stats[task_type],
            judge_meta.get("elapsed_seconds", 0.0),
            judge_meta.get("usage", {}),
        )

    accuracy_summary = {}
    for task_type, dim_dict in accuracy_stats.items():
        accuracy_summary[task_type] = {}
        overall_correct = 0
        overall_total = 0
        for dim_key in dims:
            correct = dim_dict[dim_key]["correct"]
            total = dim_dict[dim_key]["total"]
            accuracy_summary[task_type][dim_key] = {
                "correct": correct,
                "total": total,
                "accuracy": safe_div(correct, total),
            }
            overall_correct += correct
            overall_total += total
        accuracy_summary[task_type]["overall"] = {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": safe_div(overall_correct, overall_total),
        }

    output_json = {
        "config": {
            "model_method": model_method,
            "conflict_type": conflict_type,
            "judge_model": eval_model,
            "concurrency_limit": max(1, int(concurrency)),
            "num_samples": len(final_results_log),
        },
        "run_summary": {
            "judge_wall_clock_seconds": run_wall_clock_seconds,
            "judge_request_count": len(results),
        },
        "summary": {
            "accuracy": accuracy_summary,
            "target_model_stats": {
                task_type: {k: finalize_usage_stats(v) for k, v in stat_dict.items()}
                for task_type, stat_dict in target_stats.items()
            },
            "judge_stats": {
                task_type: finalize_usage_stats(stat_dict)
                for task_type, stat_dict in judge_stats.items()
            },
        },
        "details": final_results_log,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_json, ensure_ascii=False, indent=2), encoding="utf-8")
    acc = accuracy_summary[conflict_type]
    print(
        f"Saved {output_path} | SR {acc['dim1']['accuracy']*100:.1f}% "
        f"PR {acc['dim2']['accuracy']*100:.1f}% "
        f"IPA {acc['dim3']['accuracy']*100:.1f}% "
        f"Overall {acc['overall']['accuracy']*100:.1f}%"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--answers-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--conflict-type", required=True, choices=["T1", "T2"])
    p.add_argument("--model-method", required=True)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-provider", default="OPENAI")
    return p.parse_args()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    args = parse_args()
    asyncio.run(
        run_stream_evaluation(
            answers_path=Path(args.answers_path),
            dataset_path=Path(args.dataset_path),
            output_path=Path(args.output_path),
            model_method=args.model_method,
            conflict_type=args.conflict_type,
            concurrency=args.concurrency,
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
        )
    )
