#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STALE Benchmark - Evaluator

Orchestrates the full evaluation loop:
  1. Load (or generate) the dataset
  2. For each instance, run the three probe queries independently through the adapter
  3. Score each response via the judge
  4. Aggregate and report metrics
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests.stale.stale_types import (
    STALEAdapter,
    STALEInput,
    STALEInstance,
    load_dataset,
)
from tests.stale.stale_judge import JudgeResult, LLMJudge, RuleBasedJudge


# ---------------------------------------------------------------------------
# Per-instance evaluation helper
# ---------------------------------------------------------------------------

def evaluate_single_instance(
    adapter: STALEAdapter,
    instance: STALEInstance,
    judge: Any,
) -> Dict[str, Any]:
    """
    Run all three STALE probe queries for one instance and return scored results.

    Each dimension is called independently to prevent information leakage.
    """
    sessions = instance.haystack_sessions
    # 传入 session_n 的 id，让 NPCMemoryAdapter 精确标记 M_new 所在 session
    adapter.ingest_history(sessions, session_n_id=instance.session_n.session_id)
    responses: Dict[str, str] = {}
    retrieved: Dict[str, Optional[List[str]]] = {}
    latencies: Dict[str, float] = {}

    for dim in ("sr", "pr", "ipa"):
        query_text = instance.queries[dim].question
        stale_input = STALEInput(
            uid=instance.uid,
            sessions=sessions,
            query=query_text,
            query_type=dim,
        )

        t0 = time.time()
        output = adapter.process_query(stale_input)
        latencies[dim] = round((time.time() - t0) * 1000, 2)  # ms

        responses[dim] = output.response
        retrieved[dim] = output.retrieved_memories

    # Score via judge
    judge_result: JudgeResult = judge.judge_instance(instance, responses)

    return {
        "uid": instance.uid,
        "conflict_type": instance.conflict_type,
        "attribute": instance.attribute,
        "m_old": instance.m_old,
        "m_new": instance.m_new,
        "responses": responses,
        "retrieved": retrieved,
        "latencies_ms": latencies,
        "judge": judge_result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class STALEEvaluator:
    """
    Full STALE evaluation pipeline.

    Example usage:
        adapter = SimpleLLMAdapter()
        adapter.initialize({})

        evaluator = STALEEvaluator(adapter, dataset_path="data/stale_dataset.jsonl")
        results = evaluator.run_evaluation(limit=10)
        evaluator.print_summary(results)
        evaluator.save_results(results, "results/stale_results.json")
    """

    def __init__(
        self,
        adapter: STALEAdapter,
        dataset_path: Optional[str] = None,
        instances: Optional[List[STALEInstance]] = None,
        use_llm_judge: bool = True,
        judge_model: str = "gpt-4o-mini",
    ) -> None:
        if dataset_path is not None:
            self.instances = load_dataset(dataset_path)
        elif instances is not None:
            self.instances = instances
        else:
            raise ValueError("Provide either dataset_path or instances.")

        self.adapter = adapter
        self.judge = LLMJudge(model_name=judge_model) if use_llm_judge else RuleBasedJudge()

    # ------------------------------------------------------------------
    def run_evaluation(
        self,
        limit: Optional[int] = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation.

        Args:
            limit: Evaluate only the first `limit` instances (useful for quick tests).
            parallel: Use a thread pool (speeds up I/O-bound LLM calls).
            max_workers: Thread pool size when parallel=True.

        Returns:
            Dictionary with per-instance raw results and aggregated scores.
        """
        instances = self.instances[:limit] if limit else self.instances
        print(f"\n[STALEEvaluator] Evaluating {len(instances)} instances "
              f"(judge={'LLM' if isinstance(self.judge, LLMJudge) else 'rule-based'}, "
              f"parallel={parallel})")

        # Optionally pre-ingest history for memory-based adapters
        # (done once per instance; adapter is responsible for clearing state)
        raw_results: List[Dict[str, Any]] = []

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self._run_one, inst): inst
                    for inst in instances
                }
                for future in as_completed(futures):
                    try:
                        raw_results.append(future.result())
                    except Exception as exc:
                        inst = futures[future]
                        print(f"  [ERROR] uid={inst.uid[:8]}: {exc}")
        else:
            for i, inst in enumerate(instances, 1):
                print(f"  [{i}/{len(instances)}] uid={inst.uid[:8]}… attr={inst.attribute}")
                try:
                    raw_results.append(self._run_one(inst))
                except Exception as exc:
                    print(f"    ERROR: {exc}")

        scores = self._aggregate(raw_results)

        return {
            "summary": scores,
            "per_instance": raw_results,
        }

    def _run_one(self, inst: STALEInstance) -> Dict[str, Any]:
        """Ingest history (if applicable) and evaluate one instance."""
        self.adapter.ingest_history(inst.haystack_sessions)
        return evaluate_single_instance(self.adapter, inst, self.judge)

    # ------------------------------------------------------------------
    def _aggregate(self, raw_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute per-dimension and overall accuracy."""
        total = len(raw_results)
        if total == 0:
            return {"total": 0, "sr": 0.0, "pr": 0.0, "ipa": 0.0, "overall": 0.0}

        counts = {"sr": 0, "pr": 0, "ipa": 0}
        type_counts: Dict[str, Dict[str, int]] = {}

        for r in raw_results:
            judge = r.get("judge", {})
            for dim in ("sr", "pr", "ipa"):
                if judge.get(dim, {}).get("pass", False):
                    counts[dim] += 1

            # Break down by conflict type
            ctype = r.get("conflict_type", "unknown")
            if ctype not in type_counts:
                type_counts[ctype] = {"total": 0, "pass": 0}
            type_counts[ctype]["total"] += 1
            if judge.get("overall_pass", False):
                type_counts[ctype]["pass"] += 1

        overall_pass = sum(
            1 for r in raw_results if r.get("judge", {}).get("overall_pass", False)
        )

        summary = {
            "total": total,
            "sr": round(counts["sr"] / total, 4),
            "pr": round(counts["pr"] / total, 4),
            "ipa": round(counts["ipa"] / total, 4),
            "overall": round(
                (counts["sr"] + counts["pr"] + counts["ipa"]) / (total * 3), 4
            ),
            "overall_all_pass": round(overall_pass / total, 4),
            "by_type": {
                t: round(v["pass"] / v["total"], 4) if v["total"] > 0 else 0.0
                for t, v in type_counts.items()
            },
        }
        return summary

    # ------------------------------------------------------------------
    @staticmethod
    def print_summary(results: Dict[str, Any]) -> None:
        """Pretty-print the evaluation summary."""
        summary = results.get("summary", {})
        per_instance = results.get("per_instance", [])

        print("\n" + "=" * 60)
        print("STALE BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Instances evaluated : {summary.get('total', 0)}")
        print(f"  SR  accuracy        : {summary.get('sr', 0):.2%}")
        print(f"  PR  accuracy        : {summary.get('pr', 0):.2%}")
        print(f"  IPA accuracy        : {summary.get('ipa', 0):.2%}")
        print(f"  Overall (avg dims)  : {summary.get('overall', 0):.2%}")
        print(f"  Overall (all-pass)  : {summary.get('overall_all_pass', 0):.2%}")

        by_type = summary.get("by_type", {})
        if by_type:
            print("\n  By conflict type (all-dim pass rate):")
            for ctype, acc in sorted(by_type.items()):
                print(f"    Type {ctype}: {acc:.2%}")

        if per_instance:
            print("\n  Per-instance breakdown:")
            for r in per_instance:
                j = r.get("judge", {})
                sr  = "PASS" if j.get("sr",  {}).get("pass") else "FAIL"
                pr  = "PASS" if j.get("pr",  {}).get("pass") else "FAIL"
                ipa = "PASS" if j.get("ipa", {}).get("pass") else "FAIL"
                print(
                    f"    {r['uid'][:8]}... [{r.get('attribute','?'):20s}] "
                    f"SR:{sr}  PR:{pr}  IPA:{ipa}"
                )
        print("=" * 60)

    # ------------------------------------------------------------------
    @staticmethod
    def save_results(results: Dict[str, Any], path: str) -> None:
        """Save full results to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"[STALEEvaluator] Results saved → {path}")
