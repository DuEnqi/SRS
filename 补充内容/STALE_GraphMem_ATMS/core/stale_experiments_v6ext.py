#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stale_experiments_v6ext.py
==========================
Paper-grade experiments layered ON TOP of stale_graphmem_atms_v6.py (the v6
bundle). Nothing here modifies the bundle; it is imported as-is.

Experiments
-----------
9.1  Still-valid / no-effect distractor suite  (--still-valid)
       Adds the three control families the user asked for:
         (a) M_new UNRELATED to M_old              (irrelevant / weakly_related / temporary)
         (b) M_new SUPPORTS / RESTATES M_old        (reinforce)
         (c) M_new is only SUPPLEMENTARY            (supplement)
       plus low-credibility-source and alternative-support controls.
       Metrics: Old-valid precision, False-invalidation rate,
                No-effect retention accuracy (+ invalidation recall on positives
                so we can see the system is *discriminating*, not "always stale").

9.2  v6 prompt ablation                          (--v6-ablation)
       full / -PR-detect-then-reject / -structured-IPA / -recent-emphasis /
       -explicit-M_old/M_new-fields / +force-formal-verdict.
       HONESTY NOTE: of these, only the *answer-shaping* knobs
       (PR-detect-then-reject, structured-IPA) change the offline deterministic
       floor and therefore give a genuine OFFLINE signal. The *prompt-phrasing*
       knobs (recent-emphasis, explicit-fields, force-verdict) only re-word the
       LLM prompt, so they are inert offline and require --use-llm to move. The
       table marks them "LLM-only" rather than faking an effect.

9.3  Remove the ATMS/AGM/Hansson formal core      (--formal-ablation)
       SAME v6 answer construction, but the keep/invalidate signal comes from a
       different source:
         formal_full        : the full formal adjudicator (ATMS+SAT+incision+MA)
         no_formal_heuristic: the surface-cue llm_only verdict (no formal layer)
         raw_history_only   : naive change-cue scan over the recent history
         mold_mnew_only     : naive change-cue scan over M_new only
         formal_only        : the formal verdict templated directly (no free read)
       This is the rebuttal to "is it just prompt engineering?" — the prompts are
       held fixed; only the adjudicator is swapped. It is genuinely measurable
       offline because the naive baselines really do (i) miss propagated/T2 &
       disjunctive defeat and (ii) over-invalidate the negatives.

9.4  Multi-round streaming memory                 (--stream)
       A 20-turn longitudinal memory stream exercising set / update(stale) /
       no_effect / T2-break / recovery / reconfirm / revert, then a battery of
       mixed final questions (including a temporal "at that time?" probe).
       Built on the bundle's GraphMemory + ATMSLabelKernel so it is the same
       machinery, not a toy.

Run
---
  python stale_experiments_v6ext.py --self-test
  python stale_experiments_v6ext.py --still-valid
  python stale_experiments_v6ext.py --v6-ablation
  python stale_experiments_v6ext.py --formal-ablation
  python stale_experiments_v6ext.py --stream
  python stale_experiments_v6ext.py --all --out runs/ext
  # add --use-llm (needs OPENAI_API_KEY/YUNWU_API_KEY) to engage the real prompts
"""
from __future__ import annotations

import os
import sys
import json
import importlib.util
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Locate & import the v6 bundle without modifying it.
# ---------------------------------------------------------------------------
def _load_bundle():
    name = "stale_graphmem_atms_v6"
    try:
        return __import__(name)
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, name + ".py"),
             os.path.join(os.getcwd(), name + ".py"),
             "/mnt/user-data/uploads/" + name + ".py",
             "/mnt/user-data/outputs/" + name + ".py"]
    for c in cands:
        if os.path.exists(c):
            spec = importlib.util.spec_from_file_location(name, c)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("could not locate stale_graphmem_atms_v6.py next to this file")


S = _load_bundle()


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _short(s, n=80):
    import re
    return re.sub(r"\s+", " ", str(s or "").strip())[:n]


# ===========================================================================
# Shared evaluation primitive
# ===========================================================================
def _sys_invalidated(res) -> bool:
    """The engine's terminal invalidate decision (same definition the bundle's
    ablation uses)."""
    return bool(res.sr_should_invalidate and not res.old_supported_terminal)


def evaluate_engine(eng, records, *, extractor_note="") -> dict:
    """Run an engine over records; return confusion + judge metrics, split by
    gold class (should_invalidate)."""
    # keep-class confusion (positive class = "old still valid / keep")
    TP = FP = FN = TN = 0           # see definitions below
    pos_inval_hit = pos_total = 0   # invalidation recall on positives
    noeff_keep = noeff_total = 0    # no-effect retention (irrelevant/reinforce/supplement)
    judge = {"pos": {"dim1": [0, 0], "dim2": [0, 0], "dim3": [0, 0]},
             "neg": {"dim1": [0, 0], "dim2": [0, 0], "dim3": [0, 0]}}
    rows = []
    for rec in records:
        gm, atms, bb, ma, res = eng.adjudicate(rec)
        ans = eng.answer(gm, atms, bb, ma, res, rec.get("probing_queries", {}))
        v = S.local_judge(rec, ans)
        gold_inv = bool(rec.get("should_invalidate", True))
        gold_keep = not gold_inv
        sys_inv = _sys_invalidated(res)
        sys_keep = not sys_inv

        # keep-class confusion
        if sys_keep and gold_keep:
            TP += 1
        elif sys_keep and not gold_keep:
            FP += 1                          # kept but should have invalidated (missed staleness)
        elif (not sys_keep) and gold_keep:
            FN += 1                          # invalidated but should have kept (FALSE INVALIDATION)
        else:
            TN += 1

        if gold_inv:
            pos_total += 1
            pos_inval_hit += int(sys_inv)
        if gold_keep and rec.get("_family", "") in ("irrelevant", "weakly_related",
                                                    "temporary", "reinforce", "supplement"):
            noeff_total += 1
            noeff_keep += int(sys_keep)

        cls = "pos" if gold_inv else "neg"
        for d in ("dim1", "dim2", "dim3"):
            judge[cls][d][1] += 1
            judge[cls][d][0] += int(v[d])

        rows.append({"uid": rec.get("uid"), "family": rec.get("_family", ""),
                     "gold_invalidate": gold_inv, "sys_invalidate": sys_inv,
                     "conflict_type": res.conflict_type, "operation": res.operation,
                     "judge": v})

    old_valid_precision = TP / (TP + FP) if (TP + FP) else None
    false_invalidation_rate = FN / (FN + TP) if (FN + TP) else None   # over true-keep set
    invalidation_recall_pos = pos_inval_hit / pos_total if pos_total else None
    noeff_retention = noeff_keep / noeff_total if noeff_total else None

    def jacc(cls):
        d = judge[cls]
        cells = []
        for k in ("dim1", "dim2", "dim3"):
            c, t = d[k]
            cells.append(100.0 * c / t if t else 0.0)
        return round(_mean(cells), 1)

    return {
        "extractor_note": extractor_note,
        "n": len(records),
        "confusion_keepclass": {"TP_keep": TP, "FP_keep": FP, "FN_keep": FN, "TN_keep": TN},
        "old_valid_precision": round(old_valid_precision, 4) if old_valid_precision is not None else None,
        "false_invalidation_rate": round(false_invalidation_rate, 4) if false_invalidation_rate is not None else None,
        "no_effect_retention_acc": round(noeff_retention, 4) if noeff_retention is not None else None,
        "invalidation_recall_positives": round(invalidation_recall_pos, 4) if invalidation_recall_pos is not None else None,
        "judge_acc_positives_pct": jacc("pos"),
        "judge_acc_negatives_pct": jacc("neg"),
        "rows": rows,
    }


# ===========================================================================
# Record builders for the still-valid suite
# ===========================================================================
def _queries(slot_human: str, premise_clause: str) -> Dict[str, str]:
    return {
        "dim1_query": f"Based on the history, does the user's {slot_human} still hold?",
        "dim2_query": f"Since {premise_clause}, can you go ahead and act on that for them?",
        "dim3_query": f"Help me plan something that depends on my {slot_human}.",
    }


def _rec(uid, family, ct, attr_b, value_old, m_old, m_new, *, should_invalidate,
         value_new=None, upstream_a=None, low_cred=False, alt_support=False,
         disjunctive=False, echo=False, echo_count=0, slot_human="situation",
         premise_clause="the earlier fact holds"):
    rec = {
        "uid": uid,
        "_family": family,
        "conflict_type": ct,
        "M_old": m_old,
        "M_new": m_new,
        "should_invalidate": should_invalidate,
        "probing_queries": _queries(slot_human, premise_clause),
        "graph_hint": {"attribute_b": attr_b, "value_old": value_old,
                       "value_new": value_new, "upstream_a": upstream_a},
    }
    if low_cred:
        rec["low_credibility_new"] = True
    if alt_support:
        rec["alt_support"] = True
    if disjunctive:
        rec["disjunctive_defeat"] = True
    if echo:
        rec["echo_chamber"] = True
        rec["echo_count"] = echo_count or 3
    return rec


def build_still_valid_suite() -> List[dict]:
    """Negatives (should_invalidate=False) across the three requested families
    plus low-credibility & alternative-support; positives kept for contrast."""
    R: List[dict] = []

    # ---- (a) UNRELATED: M_new has nothing to do with M_old ----
    R.append(_rec("sv_irrelevant_0", "irrelevant", "NONE",
                  "location_and_living/current_base_location", "lives in Seattle",
                  "I love how rainy Seattle is; the gray skies suit me.",
                  "I finally bought a fancy espresso machine for the kitchen.",
                  should_invalidate=False, slot_human="home city",
                  premise_clause="the user lives in Seattle"))
    R.append(_rec("sv_irrelevant_1", "irrelevant", "NONE",
                  "role_and_identity/employment_status", "works as a nurse",
                  "I work as a nurse at the city hospital.",
                  "Watched a great deep-sea documentary last night.",
                  should_invalidate=False, slot_human="job",
                  premise_clause="the user is a nurse"))
    R.append(_rec("sv_weakly_related_0", "weakly_related", "NONE",
                  "routine_and_transport/current_commute_mode", "bikes to work",
                  "I bike to work along the river every morning.",
                  "The weather has been a bit gloomy and grey this week.",
                  should_invalidate=False, slot_human="commute",
                  premise_clause="the user bikes to work"))
    R.append(_rec("sv_temporary_0", "temporary", "NONE",
                  "health_and_mobility/current_health_state", "runs every day",
                  "Running every single day is my non-negotiable ritual.",
                  "My knee is a bit sore today so I'm resting this one afternoon.",
                  should_invalidate=False, slot_human="running habit",
                  premise_clause="the user runs daily"))

    # ---- (b) SUPPORTS / RESTATES: M_new reinforces M_old (same slot, same value) ----
    R.append(_rec("sv_reinforce_0", "reinforce", "NONE",
                  "routine_and_transport/current_commute_mode", "bikes to work",
                  "I bike to work along the river every morning.",
                  "Still loving the bike commute — rode in again today, rain or shine.",
                  should_invalidate=False, slot_human="commute",
                  premise_clause="the user bikes to work"))
    R.append(_rec("sv_reinforce_1", "reinforce", "NONE",
                  "location_and_living/current_base_location", "lives in Seattle",
                  "I live in Seattle and love the rain.",
                  "Another grey Seattle morning here at home; wouldn't live anywhere else.",
                  should_invalidate=False, slot_human="home city",
                  premise_clause="the user lives in Seattle"))

    # ---- (c) SUPPLEMENTARY: M_new adds detail but does NOT change the slot value ----
    R.append(_rec("sv_supplement_0", "supplement", "NONE",
                  "routine_and_transport/current_commute_mode", "bikes to work",
                  "I bike to work along the river every morning.",
                  "I treated myself to a new helmet and panniers for the bike commute.",
                  should_invalidate=False, slot_human="commute",
                  premise_clause="the user bikes to work"))
    R.append(_rec("sv_supplement_1", "supplement", "NONE",
                  "role_and_identity/employment_status", "works as a teacher",
                  "Grading papers all weekend again — teaching never stops.",
                  "I just got assigned an extra section of the same class this term.",
                  should_invalidate=False, slot_human="job",
                  premise_clause="the user is a teacher"))

    # ---- low-credibility source must NOT override the user's explicit fact ----
    R.append(_rec("sv_lowcred_0", "low_credibility", "T1",
                  "role_and_identity/employment_status", "works as a nurse",
                  "I work as a nurse at the city hospital — that's my job, explicitly.",
                  "Someone at a party guessed I might have quit nursing, but they were just speculating.",
                  should_invalidate=False, value_new="quit nursing", low_cred=True,
                  slot_human="job", premise_clause="the user is a nurse"))

    # ---- alternative support: even if one path pauses, an independent path holds ----
    R.append(_rec("sv_altsupport_0", "alt_support", "T2",
                  "routine_and_transport/current_commute_mode", "can still get to work",
                  "I can still get to work fine — I bike, and I also keep a car for backup.",
                  "Recovering from a minor foot strain, I've eased off the biking for now.",
                  should_invalidate=False, alt_support=True,
                  upstream_a="health_and_mobility/current_health_state",
                  slot_human="ability to get to work",
                  premise_clause="the user can get to work"))

    # ---- POSITIVES (should invalidate) for the discrimination contrast ----
    R.append(_rec("sv_pos_T1_0", "pos_T1", "T1",
                  "routine_and_transport/current_commute_mode", "bike commute",
                  "My favorite part of the day is my 10-mile bike ride to the office.",
                  "My doctor told me to take it slow, so lately I just stroll the few blocks to work.",
                  should_invalidate=True, value_new="walking commute",
                  slot_human="commute", premise_clause="the user bikes to work"))
    R.append(_rec("sv_pos_T2_0", "pos_T2", "T2",
                  "routine_and_transport/current_commute_mode", "bike commute",
                  "Nothing beats my morning bike commute along the river to the office.",
                  "Recovering from knee surgery has been rough; I can barely put weight on my leg.",
                  should_invalidate=True, upstream_a="health_and_mobility/current_health_state",
                  slot_human="commute", premise_clause="the user bikes to work"))
    R.append(_rec("sv_pos_disj_0", "pos_disjunctive", "T2",
                  "routine_and_transport/current_commute_mode", "gets to the office somehow",
                  "I always make it into the office one way or another - car or the bus.",
                  "I sold the car last month, and they also shut down the only bus line near me.",
                  should_invalidate=True, disjunctive=True, slot_human="commute",
                  premise_clause="the user can get to the office"))
    R.append(_rec("sv_pos_echo_0", "pos_echo", "T2",
                  "role_and_identity/employment_status", "works at the old firm",
                  "I work at the old firm downtown - I've mentioned it a few times.",
                  "Today was my first day at a completely different company across town.",
                  should_invalidate=True, echo=True, echo_count=3, slot_human="job",
                  premise_clause="the user works at the old firm"))
    return R


# ===========================================================================
# 9.1  Still-valid / no-effect distractor experiment
# ===========================================================================
def run_still_valid(out: Optional[Path], use_llm=False) -> dict:
    llm = S.build_llm_callable() if use_llm else None
    suite = build_still_valid_suite()
    # primary: oracle extraction isolates the ANSWER pipeline's over-rejection
    eng_oracle = S.StaleEngine(mode="ours", extraction_mode="oracle", llm=llm, pipeline="v6")
    # secondary: real (LLM/keyword) extraction exposes same-slot extraction errors
    eng_llmx = S.StaleEngine(mode="ours", extraction_mode="llm", llm=llm, pipeline="v6")

    res_oracle = evaluate_engine(eng_oracle, suite,
                                 extractor_note="oracle extraction (isolates answer pipeline)")
    res_llmx = evaluate_engine(eng_llmx, suite,
                               extractor_note="LLM/keyword extraction (end-to-end)")
    out_d = {"oracle_extraction": res_oracle, "endpoint_llm_extraction": res_llmx,
             "n_negatives": sum(1 for r in suite if not r["should_invalidate"]),
             "n_positives": sum(1 for r in suite if r["should_invalidate"])}
    if out:
        (out / "still_valid.json").write_text(json.dumps(out_d, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    return out_d


def format_still_valid(d: dict) -> str:
    L = ["\n=== 9.1  Still-valid / no-effect distractor suite ===",
         f"negatives={d['n_negatives']}  positives={d['n_positives']}", ""]
    for key in ("oracle_extraction", "endpoint_llm_extraction"):
        r = d[key]
        L.append(f"[{key}]  ({r['extractor_note']})")
        L.append(f"  Old-valid precision        : {r['old_valid_precision']}")
        L.append(f"  False-invalidation rate    : {r['false_invalidation_rate']}   (lower is better)")
        L.append(f"  No-effect retention acc    : {r['no_effect_retention_acc']}")
        L.append(f"  Invalidation recall (pos)  : {r['invalidation_recall_positives']}   (high => discriminating)")
        L.append(f"  Judge acc (negatives)      : {r['judge_acc_negatives_pct']}%")
        L.append(f"  Judge acc (positives)      : {r['judge_acc_positives_pct']}%")
        c = r["confusion_keepclass"]
        L.append(f"  keep-class confusion       : {c}")
        L.append("")
    L.append("read: low false-invalidation + high invalidation-recall => v6 keeps still-valid "
             "facts AND still catches real staleness (not 'always says maybe outdated').")
    return "\n".join(L)


# ===========================================================================
# 9.2  v6 prompt ablation
# ===========================================================================
@dataclass
class V6Config:
    pr_detect_reject: bool = True     # answer-shaping  -> offline-visible
    ipa_structured: bool = True       # answer-shaping  -> offline-visible
    recent_emphasis: bool = True      # prompt-phrasing -> LLM-only
    explicit_fields: bool = True      # prompt-phrasing -> LLM-only
    force_verdict: bool = False       # prompt-phrasing -> LLM-only (offline floor already follows verdict)


# which knobs actually move the OFFLINE deterministic floor
_OFFLINE_VISIBLE = {"pr_detect_reject", "ipa_structured"}


class AblatableV6Engine(S.StaleEngine):
    """v6 engine whose answer pipeline is parameterised by a V6Config so each
    prompt/answer feature can be turned off independently."""

    def __init__(self, *a, v6cfg: Optional[V6Config] = None, **k):
        super().__init__(*a, **k)
        self.v6cfg = v6cfg or V6Config()

    # ----- LLM system prompts (only matter with --use-llm) -----
    def _sr_sys(self):
        if self.v6cfg.recent_emphasis:
            return S.StaleEngine._V6_SR_SYS
        return ("You are an assistant with long-term memory of the user. Read the conversation "
                "history. The question asks whether an earlier fact about the user still holds. "
                "Answer based on the history.")

    def _pr_sys(self):
        if self.v6cfg.pr_detect_reject:
            return S.StaleEngine._V6_PR_SYS
        return ("You are an assistant with long-term memory of the user. Answer the user's "
                "request helpfully using what you know about them.")

    def _ipa_sys(self):
        if self.v6cfg.ipa_structured:
            return S.StaleEngine._V6_IPA_SYS
        return ("You are an assistant with long-term memory of the user. Help with the request.")

    def _user_prompt(self, hist, m_old, m_new, query, forced):
        if self.v6cfg.explicit_fields:
            base = S.StaleEngine._v6_user(hist, m_old, m_new, query)
        else:
            base = f"[Conversation History]\n{hist}\n\n[Question]\n{query}"
        return base + forced

    # ----- deterministic floor (offline) honouring answer-shaping knobs -----
    def _det_pr(self, m_old, m_new, keep):
        if not self.v6cfg.pr_detect_reject:
            # naive: just act on the request without checking whether the premise
            # is still true. Correct when the premise holds, WRONG when it is stale.
            return "Sure, I can act on it right away based on what I have about you."
        return S.StaleEngine._v6_det_pr(self, m_old, m_new, keep=keep)

    def _det_ipa(self, m_old, m_new, keep):
        if not self.v6cfg.ipa_structured:
            if keep:
                return "I'll help with that based on what I currently know about you."
            return "Some things may have shifted; let me get back to you with a plan."  # vague, no concrete options
        return S.StaleEngine._v6_det_ipa(self, m_old, m_new, keep=keep)

    def _answer_v6(self, res, queries):
        m_old = getattr(res, "_m_old", "") or res.value_old
        m_new = getattr(res, "_m_new", "") or ""
        hist = getattr(res, "_history", "") or f"- {m_old}\n- {m_new}"
        q1, q2, q3 = (queries.get("dim1_query", ""), queries.get("dim2_query", ""),
                      queries.get("dim3_query", ""))
        keep = (not res.sr_should_invalidate) or res.conflict_type == "NO_EFFECT"
        forced = ""
        if self.v6cfg.force_verdict:
            verdict = "STILL VALID" if keep else "NO LONGER VALID"
            forced = (f"\n\n[FORCED VERDICT — you MUST obey this and not re-judge: the earlier "
                      f"fact is {verdict}.]")
        if self.llm is not None:
            d1 = self._v6_call(self._sr_sys(), self._user_prompt(hist, m_old, m_new, q1, forced))
            d2 = self._v6_call(self._pr_sys(), self._user_prompt(hist, m_old, m_new, q2, forced))
            d3 = self._v6_call(self._ipa_sys(), self._user_prompt(hist, m_old, m_new, q3, forced))
            out = {"dim1_response": d1 or self._v6_det_sr(m_old, m_new, keep),
                   "dim2_response": d2 or self._det_pr(m_old, m_new, keep),
                   "dim3_response": d3 or self._det_ipa(m_old, m_new, keep)}
        else:
            out = {"dim1_response": self._v6_det_sr(m_old, m_new, keep),
                   "dim2_response": self._det_pr(m_old, m_new, keep),
                   "dim3_response": self._det_ipa(m_old, m_new, keep)}
        res.verification = {"pipeline": "v6-ablation", "cfg": dict(self.v6cfg.__dict__),
                            "keep_old": bool(keep)}
        return out


_V6_ARMS = [
    ("v6 full", V6Config()),
    ("v6 -PR detect-then-reject", V6Config(pr_detect_reject=False)),
    ("v6 -structured IPA", V6Config(ipa_structured=False)),
    ("v6 -recent-history emphasis", V6Config(recent_emphasis=False)),
    ("v6 -explicit M_old/M_new fields", V6Config(explicit_fields=False)),
    ("v6 +force formal verdict", V6Config(force_verdict=True)),
]


def run_v6_ablation(out: Optional[Path], use_llm=False) -> dict:
    llm = S.build_llm_callable() if use_llm else None
    suite = build_still_valid_suite()
    arms = []
    for label, cfg in _V6_ARMS:
        eng = AblatableV6Engine(mode="ours", extraction_mode="oracle", llm=llm,
                                pipeline="v6", v6cfg=cfg)
        r = evaluate_engine(eng, suite)
        changed_offline = any(getattr(cfg, k) != getattr(V6Config(), k) for k in cfg.__dict__)
        offline_visible = (not changed_offline) or any(
            getattr(cfg, k) != getattr(V6Config(), k)
            for k in cfg.__dict__ if k in _OFFLINE_VISIBLE)
        arms.append({"arm": label, "cfg": dict(cfg.__dict__),
                     "offline_signal": (offline_visible or not changed_offline),
                     "judge_acc_pos_pct": r["judge_acc_positives_pct"],
                     "judge_acc_neg_pct": r["judge_acc_negatives_pct"],
                     "false_invalidation_rate": r["false_invalidation_rate"],
                     "old_valid_precision": r["old_valid_precision"]})
    out_d = {"used_llm": bool(llm), "arms": arms}
    if out:
        (out / "v6_ablation.json").write_text(json.dumps(out_d, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    return out_d


def format_v6_ablation(d: dict) -> str:
    L = ["\n=== 9.2  v6 prompt ablation ===",
         f"(LLM in loop: {d['used_llm']})", "",
         f"{'arm':<34}{'posJudge':>9}{'negJudge':>9}{'falseInv':>9}{'oldValP':>9}  offline?"]
    L.append("-" * 90)
    for a in d["arms"]:
        fi = a["false_invalidation_rate"]
        ov = a["old_valid_precision"]
        vis = "yes" if a["offline_signal"] else "LLM-only"
        L.append(f"{a['arm']:<34}{a['judge_acc_pos_pct']:>8.1f}%{a['judge_acc_neg_pct']:>8.1f}%"
                 f"{(fi if fi is not None else 0):>9.2f}{(ov if ov is not None else 0):>9.2f}  {vis}")
    L.append("\nnote: 'offline?'=does this knob move the deterministic floor without an LLM. "
             "recent-emphasis / explicit-fields / force-verdict are prompt-phrasing knobs that "
             "only re-word the LLM prompt, so they are inert offline (run --use-llm to measure "
             "them). PR-detect-then-reject and structured-IPA change answer construction and so "
             "drop offline when removed.")
    return "\n".join(L)


# ===========================================================================
# 9.3  Remove the ATMS/AGM/Hansson formal core (same prompts, swap adjudicator)
# ===========================================================================
_CHANGE_CUES = [
    "no longer", "not ", "stopped", "quit", "instead", "moved", "relocat", "retired",
    "changed", "now ", "surgery", "injur", "broke", "recover", "barely", "cannot",
    "can't", "unable", "first day at", "sold", "shut down", "avoid", "trimester",
    "pregnan", "eased off", "sore", "ease the strain", "speculat",
]


def _naive_keep_from_text(text: str) -> bool:
    t = (text or "").lower()
    return not any(c in t for c in _CHANGE_CUES)


def _answers_with_keep(eng_det: AblatableV6Engine, res, keep: bool, queries) -> dict:
    """Build v6-style answers from a SUPPLIED keep signal, prompts held fixed."""
    m_old = getattr(res, "_m_old", "") or res.value_old
    m_new = getattr(res, "_m_new", "") or ""
    return {"dim1_response": eng_det._v6_det_sr(m_old, m_new, keep),
            "dim2_response": eng_det._det_pr(m_old, m_new, keep),
            "dim3_response": eng_det._det_ipa(m_old, m_new, keep)}


def run_formal_core_ablation(out: Optional[Path], use_llm=False) -> dict:
    """Same v6 answer construction; only the verdict source changes."""
    llm = S.build_llm_callable() if use_llm else None
    suite = build_still_valid_suite()

    eng_formal = S.StaleEngine(mode="ours", extraction_mode="oracle", llm=llm, pipeline="v6")
    eng_heur = S.StaleEngine(mode="llm_only", extraction_mode="oracle", llm=llm, pipeline="v6")
    eng_det = AblatableV6Engine(mode="ours", extraction_mode="oracle", llm=llm,
                                pipeline="v6", v6cfg=V6Config())

    variants = ["formal_full", "no_formal_heuristic", "raw_history_only",
                "mold_mnew_only", "formal_only"]
    agg = {v: {"keepclass": [0, 0, 0, 0],   # TP FP FN TN
               "noeff": [0, 0], "posrecall": [0, 0],
               "jpos": [0, 0], "jneg": [0, 0]} for v in variants}

    for rec in suite:
        gold_inv = bool(rec.get("should_invalidate", True))
        gold_keep = not gold_inv
        fam = rec.get("_family", "")
        queries = rec.get("probing_queries", {})

        gm, atms, bb, ma, res = eng_formal.adjudicate(rec)
        keep_formal = (not res.sr_should_invalidate) or res.conflict_type == "NO_EFFECT"

        # heuristic (no formal core) verdict
        _, _, _, _, rh = eng_heur.adjudicate(rec)
        keep_heur = not _sys_invalidated(rh)

        m_old = getattr(res, "_m_old", "") or res.value_old
        m_new = getattr(res, "_m_new", "") or ""
        hist = getattr(res, "_history", "") or f"- {m_old}\n- {m_new}"
        recent_line = hist.strip().splitlines()[-1] if hist.strip() else m_new
        keep_raw = _naive_keep_from_text(recent_line)
        keep_mom = _naive_keep_from_text(m_new)

        per = {
            "formal_full": _answers_with_keep(eng_det, res, keep_formal, queries),
            "no_formal_heuristic": _answers_with_keep(eng_det, res, keep_heur, queries),
            "raw_history_only": _answers_with_keep(eng_det, res, keep_raw, queries),
            "mold_mnew_only": _answers_with_keep(eng_det, res, keep_mom, queries),
            # formal_only: the formal verdict TEMPLATED directly (no free reading)
            "formal_only": {
                "dim1_response": eng_formal._sr(res, res.old_supported_terminal, m_new),
                "dim2_response": eng_formal._pr(res, res.old_supported_terminal, m_new),
                "dim3_response": eng_formal._ipa(res, res.old_supported_terminal, m_new),
            },
        }
        sysk = {"formal_full": keep_formal, "no_formal_heuristic": keep_heur,
                "raw_history_only": keep_raw, "mold_mnew_only": keep_mom,
                "formal_only": keep_formal}
        for v in variants:
            ans = per[v]
            jv = S.local_judge(rec, ans)
            sys_keep = sysk[v]
            sys_inv = not sys_keep
            cc = agg[v]["keepclass"]
            if sys_keep and gold_keep:
                cc[0] += 1
            elif sys_keep and not gold_keep:
                cc[1] += 1
            elif (not sys_keep) and gold_keep:
                cc[2] += 1
            else:
                cc[3] += 1
            if gold_inv:
                agg[v]["posrecall"][1] += 1
                agg[v]["posrecall"][0] += int(sys_inv)
            if gold_keep and fam in ("irrelevant", "weakly_related", "temporary",
                                     "reinforce", "supplement"):
                agg[v]["noeff"][1] += 1
                agg[v]["noeff"][0] += int(sys_keep)
            cls = "jpos" if gold_inv else "jneg"
            agg[v][cls][1] += 1
            agg[v][cls][0] += int(jv["dim1"] and jv["dim2"] and jv["dim3"])

    rows = []
    for v in variants:
        TP, FP, FN, TN = agg[v]["keepclass"]
        ov = TP / (TP + FP) if (TP + FP) else None
        fir = FN / (FN + TP) if (FN + TP) else None
        nr = (agg[v]["noeff"][0] / agg[v]["noeff"][1]) if agg[v]["noeff"][1] else None
        pr_ = (agg[v]["posrecall"][0] / agg[v]["posrecall"][1]) if agg[v]["posrecall"][1] else None
        jp = (agg[v]["jpos"][0] / agg[v]["jpos"][1]) if agg[v]["jpos"][1] else None
        jn = (agg[v]["jneg"][0] / agg[v]["jneg"][1]) if agg[v]["jneg"][1] else None
        rows.append({"variant": v, "old_valid_precision": round(ov, 3) if ov is not None else None,
                     "false_invalidation_rate": round(fir, 3) if fir is not None else None,
                     "no_effect_retention": round(nr, 3) if nr is not None else None,
                     "invalidation_recall_pos": round(pr_, 3) if pr_ is not None else None,
                     "joint_judge_pos": round(jp, 3) if jp is not None else None,
                     "joint_judge_neg": round(jn, 3) if jn is not None else None})
    out_d = {"used_llm": bool(llm), "variants": rows}
    if out:
        (out / "formal_core_ablation.json").write_text(json.dumps(out_d, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    return out_d


def format_formal_ablation(d: dict) -> str:
    L = ["\n=== 9.3  Remove ATMS/AGM/Hansson core (same v6 prompts, swap adjudicator) ===",
         f"(LLM in loop: {d['used_llm']})", "",
         f"{'verdict source':<22}{'oldValP':>8}{'falseInv':>9}{'noEffRet':>9}{'posRecall':>10}{'jPos':>6}{'jNeg':>6}"]
    L.append("-" * 72)
    for r in d["variants"]:
        def f(x):
            return f"{x:.2f}" if x is not None else "  - "
        L.append(f"{r['variant']:<22}{f(r['old_valid_precision']):>8}{f(r['false_invalidation_rate']):>9}"
                 f"{f(r['no_effect_retention']):>9}{f(r['invalidation_recall_pos']):>10}"
                 f"{f(r['joint_judge_pos']):>6}{f(r['joint_judge_neg']):>6}")
    L.append("\nread: prompts are IDENTICAL across rows; only the keep/invalidate signal differs. "
             "If formal_full beats the naive baselines (esp. lower false-invalidation on negatives "
             "and higher recall on T2/disjunctive/echo positives), the gain is the FORMAL CORE, "
             "not prompt engineering.")
    return "\n".join(L)


# ===========================================================================
# 9.4  Multi-round streaming memory
# ===========================================================================
DEPS = {b: (a, why) for (a, b, _s, why) in S.COMMONSENSE_DEPENDENCIES}  # downstream -> (upstream, why)


@dataclass
class Turn:
    t: int
    event: str                       # set|update|no_effect|t2_break|recovery|reconfirm|revert
    slot: str
    value: Optional[str] = None
    upstream: Optional[str] = None   # for t2_break / recovery
    text: str = ""


class StreamMemory:
    """Longitudinal memory over the bundle's GraphMemory + ATMSLabelKernel.
    Authoritative current state per slot is tracked here; every transition is
    mirrored into the graph (status) and the ATMS label kernel (support)."""

    def __init__(self):
        self.gm = S.GraphMemory()
        self.atms = S.ATMSLabelKernel()
        # slot -> {"value", "status", "suspended_by": upstream|None, "confirms": int}
        self.state: Dict[str, dict] = {}
        # per slot, full audit: list of (t, value, status, event)
        self.log: Dict[str, List[tuple]] = defaultdict(list)
        self.turn_trace: List[dict] = []
        self._sess = 0

    def _claim(self, slot, value):
        return f"{slot}={value}"

    def _session_node(self, t, text):
        return self.gm.add_node(S.NodeType.SESSION, f"turn#{t}", {"t": t, "text": text[:80]})

    def _write(self, t, slot, value, status, event, text):
        sn = self._session_node(t, text)
        rev = self.gm.write_claim(slot, value, tier=S.attribute_tier(slot),
                                  evidence_type="implicit_state",
                                  confidence=0.8, session_index=t,
                                  source_session_id=f"S{t:03d}", session_node_id=sn)
        self.gm.revisions[rev].status = status
        self.gm.active_rev[slot] = rev
        self.atms.assert_base(self._claim(slot, value), "implicit_state", f"S{t:03d}", session_index=t)
        self.log[slot].append((t, value, status, event))
        return rev

    def apply(self, turn: Turn):
        t, ev, slot, val = turn.t, turn.event, turn.slot, turn.value
        note = ""
        if ev == "set":
            self._write(t, slot, val, S.MemStatus.ACTIVE.value, ev, turn.text)
            self.state[slot] = {"value": val, "status": S.MemStatus.ACTIVE.value,
                                "suspended_by": None, "confirms": 1}
            note = f"set {slot}={val}"

        elif ev == "update":            # T1 co-referential supersession
            prev = self.state.get(slot)
            if prev:
                old_claim = self._claim(slot, prev["value"])
                self.atms.add_nogood([old_claim, self._claim(slot, val)],
                                     reason="same-slot mutual exclusion")
                self.atms.retract_base(old_claim)
                self.log[slot].append((t, prev["value"], S.MemStatus.STALE.value, "superseded"))
                self.gm.mark_status(slot, prev["value"], S.MemStatus.STALE.value)
            self._write(t, slot, val, S.MemStatus.ACTIVE.value, ev, turn.text)
            self.state[slot] = {"value": val, "status": S.MemStatus.ACTIVE.value,
                                "suspended_by": None, "confirms": 1}
            note = f"update {slot} -> {val} (old marked stale)"

        elif ev == "no_effect":         # unrelated slot; never touches tracked slots
            self._write(t, slot, val, S.MemStatus.ACTIVE.value, ev, turn.text)
            self.state[slot] = {"value": val, "status": S.MemStatus.ACTIVE.value,
                                "suspended_by": None, "confirms": 1}
            note = f"no_effect: recorded unrelated {slot}={val}"

        elif ev == "t2_break":          # upstream changed -> dependent suspended (if no alt support)
            up = turn.upstream
            # record the upstream change as its own active claim
            if val is not None:
                self._write(t, up, val, S.MemStatus.ACTIVE.value, "update", turn.text)
                self.state[up] = {"value": val, "status": S.MemStatus.ACTIVE.value,
                                  "suspended_by": None, "confirms": 1}
            dep = self.state.get(slot)
            if dep:
                dep_claim = self._claim(slot, dep["value"])
                # alternative support? we conservatively assume none for a hard break
                self.atms.retract_base(dep_claim)
                dep["status"] = S.MemStatus.UNKNOWN_CURRENT.value
                dep["suspended_by"] = up
                self.gm.mark_status(slot, dep["value"], S.MemStatus.UNKNOWN_CURRENT.value)
                self.gm.unknown_current.add(slot)
                self.log[slot].append((t, dep["value"], S.MemStatus.UNKNOWN_CURRENT.value,
                                       "t2_suspended"))
            note = f"t2_break: {up}={val} suspends {slot} (propagated contraction)"

        elif ev == "recovery":          # upstream restored -> reactivate suspended dependent
            up = turn.upstream
            if val is not None:
                self._write(t, up, val, S.MemStatus.ACTIVE.value, "update", turn.text)
                self.state[up] = {"value": val, "status": S.MemStatus.ACTIVE.value,
                                  "suspended_by": None, "confirms": 1}
            dep = self.state.get(slot)
            if dep and dep["suspended_by"] == up and dep["status"] == S.MemStatus.UNKNOWN_CURRENT.value:
                dep_claim = self._claim(slot, dep["value"])
                self.atms.assert_base(dep_claim, "implicit_state", f"S{t:03d}", session_index=t)
                dep["status"] = S.MemStatus.ACTIVE.value
                dep["suspended_by"] = None
                self.gm.mark_status(slot, dep["value"], S.MemStatus.ACTIVE.value)
                self.gm.unknown_current.discard(slot)
                self.log[slot].append((t, dep["value"], S.MemStatus.ACTIVE.value, "recovered"))
                note = f"recovery: {up}={val} reactivates {slot}={dep['value']}"
            else:
                note = f"recovery: {up}={val} (no suspended dependent to reactivate)"

        elif ev == "reconfirm":         # restate current value -> +confidence, +independent origin
            cur = self.state.get(slot)
            if cur and cur["value"] == val:
                cur["confirms"] += 1
                self.atms.assert_base(self._claim(slot, val), "direct_statement",
                                      f"S{t:03d}", session_index=t)
                self.log[slot].append((t, val, cur["status"], "reconfirmed"))
                note = f"reconfirm {slot}={val} (confirms={cur['confirms']})"
            else:
                # reconfirming a value that isn't current behaves like a set/revert
                self._write(t, slot, val, S.MemStatus.ACTIVE.value, "reconfirm", turn.text)
                self.state[slot] = {"value": val, "status": S.MemStatus.ACTIVE.value,
                                    "suspended_by": None, "confirms": 1}
                note = f"reconfirm(set) {slot}={val}"

        elif ev == "revert":            # go back to an earlier value
            prev = self.state.get(slot)
            if prev:
                self.gm.mark_status(slot, prev["value"], S.MemStatus.STALE.value)
                self.atms.retract_base(self._claim(slot, prev["value"]))
                self.log[slot].append((t, prev["value"], S.MemStatus.STALE.value, "reverted_away"))
            self._write(t, slot, val, S.MemStatus.ACTIVE.value, ev, turn.text)
            self.state[slot] = {"value": val, "status": S.MemStatus.ACTIVE.value,
                                "suspended_by": None, "confirms": 1}
            note = f"revert {slot} -> {val}"

        self.turn_trace.append({"t": t, "event": ev, "slot": slot, "value": val,
                                "note": note, "text": turn.text})

    # ---- query API used to answer the final mixed questions ----
    def current_value(self, slot) -> Optional[str]:
        s = self.state.get(slot)
        if not s:
            return None
        if not S.status_cap(s["status"], "current"):
            return None
        return s["value"]

    def status_of(self, slot) -> Optional[str]:
        s = self.state.get(slot)
        return s["status"] if s else None

    def value_at(self, slot, t) -> Optional[str]:
        """Reconstruct the value/status of a slot AS OF time t from the log."""
        cur = None
        for (lt, lv, lst, lev) in self.log.get(slot, []):
            if lt <= t:
                cur = (lv, lst)
        return cur  # (value, status) or None


def build_stream_scenario() -> Tuple[List[Turn], dict]:
    """A 20-turn stream + the ground-truth final state and a temporal probe."""
    COMM = "routine_and_transport/current_commute_mode"
    LOC = "location_and_living/current_base_location"
    EMP = "role_and_identity/employment_status"
    COF = "physical_health/caffeine_or_nicotine_reliance"
    HLTH = "health_and_mobility/current_health_state"
    KIT = "weather_and_environment/current_weather_pattern"   # reuse as an "unrelated" sink

    T = [
        Turn(1, "set", COMM, "bike", text="I bike to the office along the river every morning."),
        Turn(2, "set", LOC, "Seattle", text="I live in Seattle and love the rain."),
        Turn(3, "set", EMP, "teacher", text="Being a high-school teacher keeps me busy."),
        Turn(4, "set", COF, "strong coffee", text="Three espressos a day, caffeine is my personality."),
        Turn(5, "no_effect", KIT, "bought espresso machine",
             text="I bought a fancy espresso machine for the kitchen."),
        Turn(6, "reconfirm", COMM, "bike", text="Still loving the bike commute, rode in again today."),
        Turn(7, "t2_break", COMM, "knee injury", upstream=HLTH,
             text="Recovering from knee surgery; I can barely put weight on my leg."),
        Turn(8, "no_effect", KIT, "read a novel", text="Finished an 800-page novel this week."),
        Turn(9, "recovery", COMM, "recovered", upstream=HLTH,
             text="Knee is fully healed; cleared by my physio to resume normal activity."),
        Turn(10, "reconfirm", COMM, "bike", text="Back on the bike for my commute, feels great."),
        Turn(11, "update", LOC, "Austin", text="We moved to Austin; the Texas heat is no joke."),
        Turn(12, "update", COMM, "walk", text="New place is close to work, so now I just walk a few blocks."),
        Turn(13, "reconfirm", LOC, "Austin", text="Settling into Austin, unpacking the last boxes."),
        Turn(14, "update", EMP, "retired", text="First Monday with nowhere to be; I retired."),
        Turn(15, "no_effect", KIT, "puzzle game", text="Got really into a puzzle game this week."),
        Turn(16, "revert", LOC, "Seattle", text="Actually we moved back to Seattle; Austin wasn't for us."),
        Turn(17, "reconfirm", EMP, "retired", text="Loving retirement, started a garden."),
        Turn(18, "t2_break", COF, "pregnancy", upstream=HLTH,
             text="First trimester now; my OB gave me a long list of things to cut out, including caffeine."),
        Turn(19, "no_effect", KIT, "reorganized bookshelf", text="Reorganized my bookshelf by color."),
        Turn(20, "reconfirm", COMM, "walk", text="Still just walking the few blocks to where I used to work."),
    ]

    gold = {
        "final": {
            COMM: "walk",          # bike->(injury suspend)->recover bike->move->walk
            LOC: "Seattle",        # Seattle->Austin->revert Seattle
            EMP: "retired",        # teacher->retired
            COF: None,             # suspended by pregnancy (cut caffeine) -> not current
        },
        "current_status": {
            COF: S.MemStatus.UNKNOWN_CURRENT.value,
        },
        # temporal probe: right after recovery at t=10, could they bike? yes.
        "temporal": {"slot": COMM, "t": 10, "expect_value": "bike", "expect_current": True},
        "no_effect_check": {"slot": LOC, "unaffected_by": "espresso machine (t5)"},
        "human": {COMM: "commute", LOC: "home city", EMP: "employment",
                  COF: "caffeine reliance"},
        "_slots": {"COMM": COMM, "LOC": LOC, "EMP": EMP, "COF": COF, "HLTH": HLTH},
    }
    return T, gold


def run_stream(out: Optional[Path]) -> dict:
    turns, gold = build_stream_scenario()
    mem = StreamMemory()
    for tn in turns:
        mem.apply(tn)

    checks = []

    def add(name, ok, got, exp):
        checks.append({"check": name, "pass": bool(ok), "got": got, "expected": exp})

    for slot, exp in gold["final"].items():
        got = mem.current_value(slot)
        add(f"final current value [{gold['human'].get(slot, slot)}]", got == exp, got, exp)

    for slot, exp_status in gold["current_status"].items():
        got = mem.status_of(slot)
        add(f"final status [{gold['human'].get(slot, slot)}]", got == exp_status, got, exp_status)

    # "does the user still bike?" -> no (changed to walk)
    comm = gold["_slots"]["COMM"]
    add("still-bike? (expect NO; now walk)",
        mem.current_value(comm) == "walk" and mem.current_value(comm) != "bike",
        mem.current_value(comm), "walk (not bike)")

    # no-effect: the espresso machine at t5 must not have changed home city
    loc = gold["_slots"]["LOC"]
    va_before = mem.value_at(loc, 4)
    va_after = mem.value_at(loc, 5)
    add("no-effect: espresso machine did NOT change home city",
        (va_before and va_after and va_before[0] == va_after[0]),
        {"before_t5": va_before, "after_t5": va_after}, "unchanged")

    # temporal recovery probe
    tp = gold["temporal"]
    vp = mem.value_at(tp["slot"], tp["t"])
    ok_temporal = (vp is not None and vp[0] == tp["expect_value"]
                   and S.status_cap(vp[1], "current") == tp["expect_current"])
    add(f"temporal: at t={tp['t']} (post-recovery) commute was bike & current",
        ok_temporal, vp, (tp["expect_value"], tp["expect_current"]))

    # coffee suspended by pregnancy -> not current
    cof = gold["_slots"]["COF"]
    add("pregnancy suspends caffeine reliance (not current)",
        mem.current_value(cof) is None and mem.status_of(cof) == S.MemStatus.UNKNOWN_CURRENT.value,
        {"value": mem.current_value(cof), "status": mem.status_of(cof)},
        "UNKNOWN_CURRENT / not current")

    n_pass = sum(c["pass"] for c in checks)
    out_d = {
        "n_turns": len(turns),
        "turn_trace": mem.turn_trace,
        "final_state": {s: {"value": mem.current_value(s), "status": mem.status_of(s)}
                        for s in [gold["_slots"]["COMM"], gold["_slots"]["LOC"],
                                  gold["_slots"]["EMP"], gold["_slots"]["COF"]]},
        "checks": checks,
        "score": {"passed": n_pass, "total": len(checks),
                  "accuracy": round(n_pass / len(checks), 4)},
        "graph_stats": mem.gm.stats(),
        "atms_stats": mem.atms.stats(),
    }
    if out:
        (out / "stream_memory.json").write_text(json.dumps(out_d, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return out_d


def format_stream(d: dict) -> str:
    L = ["\n=== 9.4  Multi-round streaming memory (20 turns) ===",
         f"turns={d['n_turns']}  final-question accuracy="
         f"{d['score']['passed']}/{d['score']['total']} ({100*d['score']['accuracy']:.0f}%)", ""]
    L.append("final state:")
    for s, v in d["final_state"].items():
        L.append(f"  {s:<48} value={str(v['value']):<16} status={v['status']}")
    L.append("\nmixed final-question checks:")
    for c in d["checks"]:
        mark = "PASS" if c["pass"] else "FAIL"
        L.append(f"  [{mark}] {c['check']}")
        if not c["pass"]:
            L.append(f"          got={c['got']}  expected={c['expected']}")
    L.append("\ncovers: set / update(stale) / no_effect / T2-break / recovery / reconfirm / revert "
             "+ a temporal 'at that time?' probe (recovery window before the later move).")
    return "\n".join(L)


# ===========================================================================
# Self-test for the extension
# ===========================================================================
def run_self_test() -> bool:
    passed = total = 0

    def check(name, cond):
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name}")

    def run_self_test_local():
        pass

    def _le(a, b):   # a <= b treating None as +inf
        a = float("inf") if a is None else a
        b = float("inf") if b is None else b
        return a <= b

    def _lt(a, b):
        a = float("inf") if a is None else a
        b = float("inf") if b is None else b
        return a < b

    def _ge(a, b):
        a = -1.0 if a is None else a
        b = -1.0 if b is None else b
        return a >= b

    # 9.1
    sv = run_still_valid(None, use_llm=False)
    r = sv["oracle_extraction"]
    check("9.1 distractor suite has >=8 negatives", sv["n_negatives"] >= 8)
    check("9.1 false-invalidation rate is low (<0.10) under oracle extraction",
          _lt(r["false_invalidation_rate"], 0.10))
    check("9.1 no-effect retention is high (>=0.90)",
          _ge(r["no_effect_retention_acc"], 0.90))
    check("9.1 invalidation recall on positives is high (>=0.90) -> discriminating",
          _ge(r["invalidation_recall_positives"], 0.90))

    # 9.2
    ab = run_v6_ablation(None, use_llm=False)
    by = {a["arm"]: a for a in ab["arms"]}
    full_pos = by["v6 full"]["judge_acc_pos_pct"]
    nopr_pos = by["v6 -PR detect-then-reject"]["judge_acc_pos_pct"]
    noipa_pos = by["v6 -structured IPA"]["judge_acc_pos_pct"]
    check("9.2 removing PR detect-then-reject lowers positive judge acc (offline-visible)",
          nopr_pos < full_pos)
    check("9.2 removing structured IPA lowers positive judge acc (offline-visible)",
          noipa_pos < full_pos)
    check("9.2 prompt-phrasing arms are flagged LLM-only offline",
          by["v6 -recent-history emphasis"]["offline_signal"] is False)

    # 9.3
    fa = run_formal_core_ablation(None, use_llm=False)
    byv = {x["variant"]: x for x in fa["variants"]}
    check("9.3 formal_full has lower-or-equal false-invalidation than no_formal_heuristic",
          _le(byv["formal_full"]["false_invalidation_rate"],
              byv["no_formal_heuristic"]["false_invalidation_rate"]))
    check("9.3 formal_full has lower false-invalidation than raw_history_only",
          _lt(byv["formal_full"]["false_invalidation_rate"],
              byv["raw_history_only"]["false_invalidation_rate"]))
    check("9.3 formal_full beats heuristic on positive invalidation recall",
          _ge(byv["formal_full"]["invalidation_recall_pos"], 0.90)
          and byv["formal_full"]["invalidation_recall_pos"]
          > (byv["no_formal_heuristic"]["invalidation_recall_pos"] or 0.0))

    # 9.4
    st = run_stream(None)
    check("9.4 stream final-question accuracy >= 0.90", st["score"]["accuracy"] >= 0.90)
    check("9.4 graph recorded no schema-invalid edges",
          st["graph_stats"]["schema_invalid_edges"] == 0)

    print(f"\nextension self-test: {passed}/{total} passed")
    return passed == total


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="v6 extension experiments (9.1-9.4)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--still-valid", action="store_true")
    ap.add_argument("--v6-ablation", action="store_true")
    ap.add_argument("--formal-ablation", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--use-llm", action="store_true",
                    help="engage the real LLM prompts (needs OPENAI_API_KEY/YUNWU_API_KEY)")
    ap.add_argument("--out", default="runs/ext")
    a = ap.parse_args(argv)

    use_llm = a.use_llm
    if use_llm and not (os.environ.get("OPENAI_API_KEY") or os.environ.get("YUNWU_API_KEY")):
        print("[notice] --use-llm requested but no API key found; running the OFFLINE deterministic "
              "floor. Prompt-phrasing ablations (recent-emphasis/explicit-fields/force-verdict) are "
              "inert in this mode.", file=sys.stderr)
        use_llm = False

    if a.self_test:
        sys.exit(0 if run_self_test() else 1)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    did = False
    if a.all or a.still_valid:
        did = True
        print(format_still_valid(run_still_valid(out, use_llm)))
    if a.all or a.v6_ablation:
        did = True
        print(format_v6_ablation(run_v6_ablation(out, use_llm)))
    if a.all or a.formal_ablation:
        did = True
        print(format_formal_ablation(run_formal_core_ablation(out, use_llm)))
    if a.all or a.stream:
        did = True
        print(format_stream(run_stream(out)))
    if not did:
        print("nothing selected; use one of --self-test --still-valid --v6-ablation "
              "--formal-ablation --stream --all")
        return
    print(f"\n-> artifacts in {out.resolve()}")


if __name__ == "__main__":
    main()
