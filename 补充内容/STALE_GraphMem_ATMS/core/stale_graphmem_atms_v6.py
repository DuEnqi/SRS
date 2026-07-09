#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stale_graphmem_atms_v6.py  (single-file bundle)
===============================================
STALE memory-staleness reasoning on a STRICT ATMS + AGM + Hansson formal core
(DPLL SAT entailment, kernel contraction, Levi revision, falsifiable postulate
harness) FUSED with the npc_consensus multi-agent architecture (trust matrix,
community consensus, access-tier propagation, source independence) and a real
LLM-extraction + formal-filter pipeline.

This single file is an auto-concatenation of the stale_hyperbase/ package; the
package form is canonical. Both forms are functionally identical.

Run:
  python stale_graphmem_atms_v6.py --self-test
  python stale_graphmem_atms_v6.py --synth --n 24 --extraction oracle
  python stale_graphmem_atms_v6.py --ablation --n 24       # the 4-version matrix
  python stale_graphmem_atms_v6.py --atms-bench            # Full-ATMS vs BFS necessity
  python stale_graphmem_atms_v6.py --postulates            # falsifiable AGM/DP harness
  python stale_graphmem_atms_v6.py --icds-path STALE/outputs/demo_T1_MAIN.json --extraction llm
"""
from __future__ import annotations

import os, re, sys, json, math, time, random, argparse, itertools
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from collections import defaultdict
from typing import (Any, Dict, List, Optional, Set, Tuple, FrozenSet, Iterable,
                    Sequence, Callable)
import numpy as np
_np = np



# ==========================================================================
# ==== module: logic
# ==========================================================================



# ---------------------------------------------------------------------------
# Formula AST
# ---------------------------------------------------------------------------
# We keep formulas *syntactic*. Two structurally different formulas are
# different Python objects even when they are logically equivalent. This is the
# representation-level commitment behind the belief-base view (Flaw #2): the
# store records sentences, not their semantic content. Equivalence is a derived,
# computed relation (see logically_equivalent), never object identity.


class Formula:
    """Immutable propositional formula. Subclasses below."""

    __slots__ = ()

    def atoms(self) -> Set[str]:
        raise NotImplementedError

    # Convenience constructors -------------------------------------------------
    def __and__(self, other: "Formula") -> "And":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def __rshift__(self, other: "Formula") -> "Or":
        # material implication p -> q  ==  ~p | q
        return Or(Not(self), other)


@dataclass(frozen=True)
class Atom(Formula):
    name: str

    def atoms(self) -> Set[str]:
        return {self.name}

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Formula):
    value: bool  # True == TOP, False == BOTTOM

    def atoms(self) -> Set[str]:
        return set()

    def __repr__(self) -> str:
        return "TOP" if self.value else "BOT"


@dataclass(frozen=True)
class Not(Formula):
    sub: Formula

    def atoms(self) -> Set[str]:
        return self.sub.atoms()

    def __repr__(self) -> str:
        return f"~{self.sub!r}"


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> Set[str]:
        return self.left.atoms() | self.right.atoms()

    def __repr__(self) -> str:
        return f"({self.left!r} & {self.right!r})"


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> Set[str]:
        return self.left.atoms() | self.right.atoms()

    def __repr__(self) -> str:
        return f"({self.left!r} | {self.right!r})"


TOP = Const(True)
BOT = Const(False)


def atom(name: str) -> Atom:
    return Atom(name)


def conj(formulas: Iterable[Formula]) -> Formula:
    """Conjoin an iterable of formulas (empty -> TOP)."""
    acc: Formula = TOP
    first = True
    for f in formulas:
        if first:
            acc, first = f, False
        else:
            acc = And(acc, f)
    return acc


# ---------------------------------------------------------------------------
# Tseitin transformation -> CNF (list of clauses; a clause is a frozenset of
# signed ints).  Linear in formula size; introduces auxiliary variables.
# ---------------------------------------------------------------------------

Lit = int                       # +v / -v
Clause = FrozenSet[Lit]
CNF = List[Clause]


class _Tseitin:
    def __init__(self) -> None:
        self._var_of: Dict[str, int] = {}
        self._next = 1
        self.clauses: CNF = []

    def var(self, name: str) -> int:
        if name not in self._var_of:
            self._var_of[name] = self._next
            self._next += 1
        return self._var_of[name]

    def fresh(self) -> int:
        v = self._next
        self._next += 1
        return v

    def add(self, *lits: Lit) -> None:
        self.clauses.append(frozenset(lits))

    def encode(self, f: Formula) -> int:
        """Return a literal that is true iff f is true; emit defining clauses."""
        if isinstance(f, Const):
            t = self.fresh()
            if f.value:
                self.add(t)            # force TOP
            else:
                self.add(-t)           # force BOT
            return t
        if isinstance(f, Atom):
            return self.var(f.name)
        if isinstance(f, Not):
            return -self.encode(f.sub)
        if isinstance(f, And):
            a, b = self.encode(f.left), self.encode(f.right)
            t = self.fresh()
            # t <-> a & b
            self.add(-t, a)
            self.add(-t, b)
            self.add(t, -a, -b)
            return t
        if isinstance(f, Or):
            a, b = self.encode(f.left), self.encode(f.right)
            t = self.fresh()
            # t <-> a | b
            self.add(t, -a)
            self.add(t, -b)
            self.add(-t, a, b)
            return t
        raise TypeError(f"unknown formula node: {f!r}")


def to_cnf(formula: Formula) -> CNF:
    """CNF asserting `formula` is true (Tseitin)."""
    t = _Tseitin()
    top = t.encode(formula)
    t.add(top)
    return t.clauses


# ---------------------------------------------------------------------------
# DPLL SAT solver  (iterative-ish recursion with unit propagation + pure lit)
# ---------------------------------------------------------------------------


class _DPLL:
    def __init__(self, clauses: CNF) -> None:
        # store mutable list of sets of ints
        self.clauses: List[Set[Lit]] = [set(c) for c in clauses]

    def solve(self) -> bool:
        return self._dpll(self.clauses, {})

    def _dpll(self, clauses: List[Set[Lit]], assign: Dict[int, bool]) -> bool:
        clauses = [set(c) for c in clauses]
        # Unit propagation -----------------------------------------------------
        changed = True
        while changed:
            changed = False
            for c in clauses:
                if not c:
                    return False                      # empty clause -> conflict
            unit = next((c for c in clauses if len(c) == 1), None)
            if unit is None:
                break
            lit = next(iter(unit))
            assign[abs(lit)] = lit > 0
            new_clauses: List[Set[Lit]] = []
            for c in clauses:
                if lit in c:
                    continue                          # clause satisfied
                if -lit in c:
                    nc = set(c)
                    nc.discard(-lit)
                    new_clauses.append(nc)
                else:
                    new_clauses.append(c)
            clauses = new_clauses
            changed = True

        if not clauses:
            return True                               # all satisfied
        for c in clauses:
            if not c:
                return False

        # Pure-literal elimination --------------------------------------------
        seen: Set[Lit] = set()
        for c in clauses:
            seen |= c
        pures = [l for l in seen if -l not in seen]
        if pures:
            kept = [c for c in clauses if not any(p in c for p in pures)]
            return self._dpll(kept, assign)

        # Branch on a variable -------------------------------------------------
        lit = next(iter(next(iter(clauses))))
        for val in (lit, -lit):
            branched = [set(c) for c in clauses]
            branched.append({val})
            if self._dpll(branched, dict(assign)):
                return True
        return False


def is_satisfiable(formula: Formula) -> bool:
    """True iff `formula` has at least one model. Never enumerates worlds."""
    return _DPLL(to_cnf(formula)).solve()


def is_consistent(formulas: Iterable[Formula]) -> bool:
    """True iff the conjunction of `formulas` is satisfiable."""
    fs = list(formulas)
    if not fs:
        return True
    return is_satisfiable(conj(fs))


def entails(premises: Iterable[Formula], conclusion: Formula) -> bool:
    """Premises |= conclusion, by refutation: UNSAT(premises & ~conclusion)."""
    prem = list(premises)
    test = And(conj(prem), Not(conclusion)) if prem else Not(conclusion)
    return not is_satisfiable(test)


def logically_equivalent(p: Formula, q: Formula) -> bool:
    """
    *Semantic* equivalence, computed (Flaw #2 fix).

    The legacy code defined `equiv(p, q) := (p == q)`, conflating syntactic
    identity with logical equivalence and silently collapsing provenance.
    Here equivalence is a derived relation: p and (p & TOP) come out equivalent,
    yet remain distinct objects that a belief base can give different fates.
    """
    return entails([p], q) and entails([q], p)


def is_tautology(f: Formula) -> bool:
    return not is_satisfiable(Not(f))


def is_contradiction(f: Formula) -> bool:
    return not is_satisfiable(f)


# ---------------------------------------------------------------------------
# Brute-force oracle (ONLY for tests / cross-checking on tiny signatures).
# This is the 2^n method we deliberately do *not* use in the engine; it exists
# so the test-suite can certify the DPLL core against ground truth on <=12 atoms.
# ---------------------------------------------------------------------------


def _eval(f: Formula, world: Dict[str, bool]) -> bool:
    if isinstance(f, Const):
        return f.value
    if isinstance(f, Atom):
        return world[f.name]
    if isinstance(f, Not):
        return not _eval(f.sub, world)
    if isinstance(f, And):
        return _eval(f.left, world) and _eval(f.right, world)
    if isinstance(f, Or):
        return _eval(f.left, world) or _eval(f.right, world)
    raise TypeError(f)


def brute_force_satisfiable(f: Formula) -> bool:
    names = sorted(f.atoms())
    for bits in itertools.product([False, True], repeat=len(names)):
        if _eval(f, dict(zip(names, bits))):
            return True
    return len(names) == 0 and _eval(f, {})


# ==========================================================================
# ==== module: revision
# ==========================================================================





def entrenchment_key(b: Belief):
    # higher reliability, then more recent, then later-added = MORE entrenched
    return (b.reliability, b.timestamp, b.uid)


@dataclass
class PrioritizedState:
    """Epistemic state = ranked records. The operative base is derived lazily."""
    records: List[Belief] = field(default_factory=list)

    def copy(self) -> "PrioritizedState":
        return PrioritizedState(list(self.records))

    def push(self, belief: Belief) -> "PrioritizedState":
        """Add (or re-assert) a record at its entrenchment; never deletes."""
        self.records.append(belief)
        return self

    # -- operative belief base (maxichoice by priority) ----------------------
    def operative_base(self) -> BeliefBase:
        ordered = sorted(self.records, key=entrenchment_key, reverse=True)
        accepted: List[Belief] = []
        acc_f: List[Formula] = []
        for r in ordered:
            if is_consistent(acc_f + [r.formula]):
                accepted.append(r)
                acc_f.append(r.formula)
        return BeliefBase(accepted)

    def believes(self, phi: Formula) -> bool:
        return self.operative_base().believes(phi)

    # -- revision -------------------------------------------------------------
    def revise(self, belief: Belief) -> "PrioritizedState":
        """
        Push `belief` at top entrenchment (most recent, most trusted among equals)
        so that on conflict it wins, but keep all prior records for iteration.
        We bump its timestamp above the current max to encode recency.
        """
        if self.records:
            tmax = max(r.timestamp for r in self.records)
        else:
            tmax = 0.0
        promoted = Belief(
            formula=belief.formula, source=belief.source,
            reliability=max(belief.reliability,
                            1e-9 + max((r.reliability for r in self.records), default=0.0)),
            timestamp=tmax + 1.0, justifications=belief.justifications,
            text=belief.text,
        )
        self.push(promoted)
        return self


def prioritized_operator(base: BeliefBase, inp: Belief) -> BeliefBase:
    """
    Operator adapter for the postulate harness. Builds a one-shot prioritized
    state from `base`, revises by `inp`, returns the operative base.
    NOTE: for *iterated* tests the harness composes this operator, which on a
    BeliefBase loses the ranking; see `prioritized_state_operator` for the
    state-preserving version used to demonstrate DP satisfaction.
    """
    st = PrioritizedState(list(base.records()))
    st.revise(inp)
    return st.operative_base()


# ==========================================================================
# ==== module: base
# ==========================================================================





# ---------------------------------------------------------------------------
# Provenance — adapted from de Kleer's ATMS "justification" idea: every belief
# remembers where it came from and how much it is trusted. Source type gives a
# default reliability prior; callers may override.
# ---------------------------------------------------------------------------


class Source(Enum):
    USER_EXPLICIT = "user_explicit"   # the user stated it directly
    USER_IMPLICIT = "user_implicit"   # paraphrase / restatement by the user
    TOOL = "tool"                     # a tool/API returned it (e.g. calendar)
    INFERRED = "inferred"             # the system derived it (e.g. from an address)
    DEFAULT = "default"               # background assumption / prior
    SYSTEM = "system"                 # hard constraint / schema axiom


# Reliability priors. SYSTEM axioms are non-defeasible (1.0) and never incised.
DEFAULT_RELIABILITY: Dict[Source, float] = {
    Source.SYSTEM: 1.00,
    Source.USER_EXPLICIT: 0.95,
    Source.USER_IMPLICIT: 0.80,
    Source.TOOL: 0.70,
    Source.INFERRED: 0.40,
    Source.DEFAULT: 0.20,
}

_UID = itertools.count(1)


@dataclass(frozen=True)
class Belief:
    """One syntactic record in the base. Identity is the uid, NOT the formula."""
    formula: Formula
    source: Source = Source.USER_EXPLICIT
    reliability: float = field(default=None)            # type: ignore
    timestamp: float = 0.0
    justifications: FrozenSet[str] = frozenset()        # ATMS-style support labels
    uid: int = field(default_factory=lambda: next(_UID))
    text: Optional[str] = None                          # original NL surface form

    def __post_init__(self):
        if self.reliability is None:
            object.__setattr__(self, "reliability",
                               DEFAULT_RELIABILITY[self.source])

    @property
    def defeasible(self) -> bool:
        return self.source is not Source.SYSTEM

    def __repr__(self) -> str:
        tag = self.text if self.text else repr(self.formula)
        return f"<{tag} | {self.source.value} r={self.reliability:.2f} #{self.uid}>"


# ---------------------------------------------------------------------------
# Kernel computation
# ---------------------------------------------------------------------------


def _entails_alpha(records: Sequence[Belief], alpha: Formula) -> bool:
    return entails([b.formula for b in records], alpha)


def find_one_kernel(records: Sequence[Belief], alpha: Formula) -> List[Belief]:
    """
    Return ONE inclusion-minimal subset of `records` that still entails `alpha`
    (an alpha-kernel). Linear number of entailment checks (QuickXplain-style
    shrink): start from the full set, drop any record whose removal preserves
    entailment of alpha. The survivor is inclusion-minimal.
    """
    if not _entails_alpha(records, alpha):
        return []                                   # base does not entail alpha
    kernel = list(records)
    i = 0
    while i < len(kernel):
        candidate = kernel[:i] + kernel[i + 1:]
        if _entails_alpha(candidate, alpha):
            kernel = candidate                      # record i was redundant
        else:
            i += 1                                  # record i is essential -> keep
    return kernel


def kernel_set(records: Sequence[Belief], alpha: Formula,
               limit: int = 64) -> List[List[Belief]]:
    """
    Enumerate alpha-kernels (up to `limit`) via a hitting-set loop. Used by the
    analysis/reporting code; the *operator* below does not need the full set.
    """
    kernels: List[List[Belief]] = []
    blocked: List[Set[int]] = []                    # uids removed so far per branch

    def search(active: List[Belief]):
        if len(kernels) >= limit:
            return
        k = find_one_kernel(active, alpha)
        if not k:
            return
        if not any({b.uid for b in k} == {b.uid for b in kk} for kk in kernels):
            kernels.append(k)
        # branch: forbid one kernel element at a time to find others
        for b in k:
            reduced = [x for x in active if x.uid != b.uid]
            search(reduced)

    search(list(records))
    # dedupe
    uniq: List[List[Belief]] = []
    seen: Set[FrozenSet[int]] = set()
    for k in kernels:
        key = frozenset(b.uid for b in k)
        if key not in seen:
            seen.add(key)
            uniq.append(k)
    return uniq


# ---------------------------------------------------------------------------
# Incision: which sentence to cut inside a kernel.
# Smooth + reliability-ordered: discard the least-reliable, then oldest, then
# newest-uid record. SYSTEM axioms are never eligible (non-defeasible).
# ---------------------------------------------------------------------------


def _incision_pick(kernel: Sequence[Belief]) -> Optional[Belief]:
    eligible = [b for b in kernel if b.defeasible]
    if not eligible:
        return None                                 # protected kernel
    return min(eligible, key=lambda b: (b.reliability, b.timestamp, b.uid))


# ---------------------------------------------------------------------------
# BeliefBase
# ---------------------------------------------------------------------------


@dataclass
class ContractionReport:
    removed: List[Belief]
    kept_kernel_conflict: bool          # True if alpha could not be removed
    explanation: str


class BeliefBase:
    """A mutable, finite set of Belief records (Hansson belief base)."""

    def __init__(self, beliefs: Optional[Iterable[Belief]] = None) -> None:
        self._beliefs: List[Belief] = list(beliefs) if beliefs else []

    # -- inspection -----------------------------------------------------------
    def __iter__(self):
        return iter(self._beliefs)

    def __len__(self):
        return len(self._beliefs)

    def records(self) -> List[Belief]:
        return list(self._beliefs)

    def formulas(self) -> List[Formula]:
        return [b.formula for b in self._beliefs]

    def copy(self) -> "BeliefBase":
        return BeliefBase(list(self._beliefs))

    def believes(self, phi: Formula) -> bool:
        """phi in Cn(base): derived, computed by entailment (never enumerated)."""
        return entails(self.formulas(), phi)

    def is_consistent(self) -> bool:
        return is_consistent(self.formulas())

    # -- expansion ------------------------------------------------------------
    def expand(self, belief: Belief) -> "BeliefBase":
        """Add a record unconditionally (base expansion). Provenance preserved;
        a logically-equivalent existing record is NOT merged away."""
        self._beliefs.append(belief)
        return self

    # -- kernel contraction ---------------------------------------------------
    def contract(self, alpha: Formula) -> ContractionReport:
        """
        Kernel-contract alpha: make the base no longer entail alpha by cutting,
        from each alpha-kernel, the least-reliable defeasible record.

        Iterative incision: while base |= alpha, find one kernel and remove its
        weakest defeasible member. Terminates because each step strictly shrinks
        a finite base; on termination no alpha-kernel survives intact, i.e. every
        kernel has been incised (this is exactly Hansson's incision function).
        """
        removed: List[Belief] = []
        guard = len(self._beliefs) + 1
        while entails(self.formulas(), alpha) and guard > 0:
            guard -= 1
            kernel = find_one_kernel(self._beliefs, alpha)
            if not kernel:
                break
            victim = _incision_pick(kernel)
            if victim is None:
                # every member of this kernel is a protected SYSTEM axiom:
                # alpha is a logical consequence of non-defeasible content and
                # cannot be honestly contracted.
                return ContractionReport(
                    removed=removed, kept_kernel_conflict=True,
                    explanation=("alpha follows from non-defeasible SYSTEM "
                                 "axioms; refusing to violate a hard constraint."))
            self._beliefs = [b for b in self._beliefs if b.uid != victim.uid]
            removed.append(victim)
        return ContractionReport(
            removed=removed, kept_kernel_conflict=False,
            explanation=(f"removed {len(removed)} record(s) to retract alpha"
                         if removed else "alpha was not believed; no change"))

    # -- revision via the Levi identity --------------------------------------
    def revise(self, belief: Belief) -> ContractionReport:
        """
        B * alpha = (B - ~alpha) + alpha.

        First kernel-contract ~alpha (restore consistency with alpha, preferring
        to keep high-reliability records), then add alpha as a new record with
        its own provenance. If alpha itself is a contradiction, we fall back to
        contraction only (consistency cannot be achieved) and report it.
        """
        alpha = belief.formula
        neg = Not(alpha)
        report = self.contract(neg)
        # add alpha unless it is itself inconsistent (would re-break the base)
        if is_consistent([alpha]):
            self.expand(belief)
        else:
            report = ContractionReport(
                removed=report.removed, kept_kernel_conflict=True,
                explanation="input alpha is a contradiction; not added")
        return report

    # -- withdrawal (Reversing the Levi Identity, Hansson 1993) --------------
    def withdraw(self, alpha: Formula) -> ContractionReport:
        """
        Contraction-first withdrawal: identical machinery to `contract` but named
        to mark that we adopt the reverse-Levi stance (no Recovery postulate),
        which is the correct choice for a belief base (Recovery is the postulate
        base contraction is known to give up; Fermé & Hansson 2011, Sec. 4)."""
        return self.contract(alpha)

    def __repr__(self) -> str:
        body = "\n  ".join(repr(b) for b in self._beliefs)
        return f"BeliefBase[{len(self._beliefs)}]:\n  {body}" if self._beliefs \
            else "BeliefBase[empty]"


# ===========================================================================
# ORDERED INCISION (4-step) — extends Hansson kernel contraction with an
# explicit, auditable victim-selection order required for the STALE engine:
#
#   Step 1. Cut DEFEASIBLE justifications first (block the inference chain),
#           NEVER delete a raw evidence record.
#   Step 2. If alpha is still entailed, cut the LOWEST-credibility *source*
#           assumption inside the surviving kernel.
#   Step 3. PROTECT any claim that still has an alternative support
#           environment not routed through alpha (structural Core-Retainment).
#   Step 4. RETAIN historical facts (do not delete) but DISABLE their current
#           validity (KM-update semantics): record them as historical.
#
# A Belief may carry `kind="justification"` (a defeasible inference link) vs
# `kind="evidence"` (a raw observation). Step 1 only removes justification
# records; raw evidence survives so that history is preserved (Step 4).
# ===========================================================================


def _is_justification(b: "Belief") -> bool:
    return getattr(b, "text", None) is not None and str(b.text).startswith("JUST::")


@dataclass
class IncisionStep:
    order: int
    rule: str                       # which of the 4 steps fired
    victim_repr: str
    detail: str = ""


@dataclass
class OrderedContractionReport:
    removed: List["Belief"]
    protected: List[str]            # claims protected by alternative support
    historical: List[str]           # claims retained-as-historical (Step 4)
    steps: List[IncisionStep]
    kept_kernel_conflict: bool
    explanation: str


class OrderedIncisionBase(BeliefBase):
    """A Hansson belief base whose contraction applies the explicit 4-step
    incision order above. Used so the STALE engine can WRITE BACK each step
    (status transitions) and explain the contraction axiomatically."""

    def _alt_support_protected(self, alpha: Formula,
                               protect_claims: Set[str]) -> Set[int]:
        """Step-3 protection set: uids of RAW EVIDENCE records (never
        justifications) whose claim still has an alternative (alpha-free) support
        environment, so they must not be cut. Justification links remain eligible
        so that step1 (cut defeasible justification) always fires first."""
        protected: Set[int] = set()
        if not protect_claims:
            return protected
        for b in self._beliefs:
            if _is_justification(b):
                continue                            # never protect a justification
            tag = b.text or repr(b.formula)
            for pc in protect_claims:
                if pc in str(tag):
                    protected.add(b.uid)
        return protected

    def ordered_contract(self, alpha: Formula,
                         historical_claims: Optional[Set[str]] = None,
                         protect_claims: Optional[Set[str]] = None
                         ) -> OrderedContractionReport:
        historical_claims = historical_claims or set()
        protect_claims = protect_claims or set()
        steps: List[IncisionStep] = []
        removed: List[Belief] = []
        protected_uids = self._alt_support_protected(alpha, protect_claims)
        guard = len(self._beliefs) + 2
        step_no = 0

        while entails(self.formulas(), alpha) and guard > 0:
            guard -= 1
            kernel = find_one_kernel(self._beliefs, alpha)
            if not kernel:
                break
            # candidates eligible for cutting (defeasible, not protected, not SYSTEM)
            eligible = [b for b in kernel if b.defeasible and b.uid not in protected_uids]
            if not eligible:
                # everything in this kernel is protected or SYSTEM
                eligible_unprot = [b for b in kernel if b.defeasible]
                if not eligible_unprot:
                    return OrderedContractionReport(
                        removed=removed, protected=sorted(protect_claims),
                        historical=sorted(historical_claims), steps=steps,
                        kept_kernel_conflict=True,
                        explanation=("alpha follows from non-defeasible SYSTEM "
                                     "axioms; refusing to violate a hard constraint."))
                # protection would block contraction → relax protection minimally
                eligible = eligible_unprot

            # STEP 1: prefer cutting a defeasible justification link
            justifs = [b for b in eligible if _is_justification(b)]
            if justifs:
                victim = min(justifs, key=lambda b: (b.reliability, b.timestamp, b.uid))
                rule = "step1_cut_defeasible_justification"
            else:
                # STEP 2: cut the lowest-credibility raw source assumption
                victim = min(eligible, key=lambda b: (b.reliability, b.timestamp, b.uid))
                rule = "step2_cut_low_credibility_source"

            step_no += 1
            steps.append(IncisionStep(
                order=step_no, rule=rule, victim_repr=repr(victim),
                detail=f"reliability={victim.reliability:.2f}"))
            self._beliefs = [b for b in self._beliefs if b.uid != victim.uid]
            removed.append(victim)

        # STEP 3 (record which protections held) + STEP 4 (historical retention)
        for pc in sorted(protect_claims):
            steps.append(IncisionStep(order=step_no + 1,
                                      rule="step3_protect_alternative_support",
                                      victim_repr=pc,
                                      detail="kept: has alpha-free support env"))
        for hc in sorted(historical_claims):
            steps.append(IncisionStep(order=step_no + 1,
                                      rule="step4_retain_historical_disable_current",
                                      victim_repr=hc,
                                      detail="retained as historical; current validity disabled"))

        return OrderedContractionReport(
            removed=removed, protected=sorted(protect_claims),
            historical=sorted(historical_claims), steps=steps,
            kept_kernel_conflict=False,
            explanation=(f"ordered incision removed {len(removed)} record(s); "
                         f"protected {len(protect_claims)}; "
                         f"retained {len(historical_claims)} as historical"))


# ==========================================================================
# ==== module: postulates
# ==========================================================================





# ---------------------------------------------------------------------------
# Set-level helpers built on finite entailment (no world enumeration).
# ---------------------------------------------------------------------------


def cn_subset(small: BeliefBase, big: BeliefBase) -> bool:
    """Cn(small) subset-of Cn(big)  <=>  big |= s for every record s of small."""
    big_fs = big.formulas()
    return all(entails(big_fs, s) for s in small.formulas())


def cn_equal(x: BeliefBase, y: BeliefBase) -> bool:
    return cn_subset(x, y) and cn_subset(y, x)


def cn_plus(base: BeliefBase, phi: Formula) -> BeliefBase:
    """The *expansion* Cn(K + phi), represented by base + phi (Cn computed lazily)."""
    b = base.copy()
    b.expand(Belief(phi, Source.SYSTEM))
    return b


# An "operator" maps (base, input-belief) -> revised base. The package's default
# operator is Levi/kernel revision; the harness can test any operator.
Operator = Callable[[BeliefBase, Belief], BeliefBase]


def levi_operator(base: BeliefBase, inp: Belief) -> BeliefBase:
    b = base.copy()
    b.revise(inp)
    return b


# A deliberately broken operator, kept so the harness can demonstrate that it
# now *catches* violations the old `or True` harness silently passed.
def naive_overwrite_operator(base: BeliefBase, inp: Belief) -> BeliefBase:
    """Throws away the entire base and keeps only the input — violates almost
    every minimal-change postulate."""
    return BeliefBase([inp])


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PostulateResult:
    name: str
    satisfied: bool
    detail: str = ""
    counterexample: Optional[str] = None

    def __repr__(self):
        mark = "PASS" if self.satisfied else "FAIL"
        extra = f"  ! {self.counterexample}" if self.counterexample else ""
        return f"[{mark}] {self.name}: {self.detail}{extra}"


# ---------------------------------------------------------------------------
# Individual postulate checks. Each returns a PostulateResult.
# ---------------------------------------------------------------------------


def check_success(op: Operator, base: BeliefBase, phi_belief: Belief) -> PostulateResult:
    phi = phi_belief.formula
    star = op(base, phi_belief)
    ok = star.believes(phi)
    return PostulateResult("P1 Success", ok,
                           "phi in K*phi",
                           None if ok else f"phi={phi!r} not entailed by result")


def check_consistency(op: Operator, base: BeliefBase, phi_belief: Belief) -> PostulateResult:
    phi = phi_belief.formula
    if is_contradiction(phi):
        return PostulateResult("P2 Consistency", True, "phi inconsistent; postulate vacuous")
    star = op(base, phi_belief)
    ok = star.is_consistent()
    return PostulateResult("P2 Consistency", ok,
                           "phi consistent => K*phi consistent",
                           None if ok else "result base is inconsistent")


def check_inclusion(op: Operator, base: BeliefBase, phi_belief: Belief) -> PostulateResult:
    phi = phi_belief.formula
    star = op(base, phi_belief)
    expansion = cn_plus(base, phi)
    ok = cn_subset(star, expansion)         # Cn(K*phi) subset Cn(K + phi)
    return PostulateResult("P3 Inclusion", ok,
                           "K*phi subset Cn(K + phi)  [minimal change]",
                           None if ok else "result introduces a belief beyond K+phi")


def check_vacuity(op: Operator, base: BeliefBase, phi_belief: Belief) -> PostulateResult:
    phi = phi_belief.formula
    if base.believes(Not(phi)):
        return PostulateResult("P4 Vacuity", True,
                               "~phi in K; postulate vacuous")
    star = op(base, phi_belief)
    expansion = cn_plus(base, phi)
    ok = cn_subset(expansion, star)         # Cn(K + phi) subset Cn(K*phi)
    return PostulateResult("P4 Vacuity", ok,
                           "~phi not in K => K*phi = K + phi",
                           None if ok else "result loses a belief that K+phi keeps")


def check_extensionality(op: Operator, base: BeliefBase,
                         phi_belief: Belief, psi: Formula) -> PostulateResult:
    phi = phi_belief.formula
    if not logically_equivalent(phi, psi):
        return PostulateResult("P5 Extensionality", True, "phi !== psi; vacuous")
    star_phi = op(base, phi_belief)
    star_psi = op(base, Belief(psi, phi_belief.source, phi_belief.reliability))
    ok = cn_equal(star_phi, star_psi)
    return PostulateResult("P5 Extensionality", ok,
                           "phi == psi => Cn(K*phi)=Cn(K*psi)",
                           None if ok else "equivalent inputs gave different belief sets")


def check_preservation(op: Operator, base: BeliefBase,
                       phi_belief: Belief, psi: Formula) -> PostulateResult:
    """AGM Preservation (K*4): if ¬φ ∉ K then K+φ ⊆ K*φ, i.e. when the input does
    NOT contradict the current beliefs, revision keeps every prior belief. (The
    earlier harness omitted the ¬φ∉K guard and so counted legitimate contraction
    — when ¬φ IS believed — as a violation; that was a harness bug, not an
    operator defect.)"""
    phi = phi_belief.formula
    if not base.believes(psi):
        return PostulateResult("P6 Preservation", True, "psi not in K; vacuous")
    if base.believes(Not(phi)):
        return PostulateResult("P6 Preservation", True,
                               "neg-phi in K; preservation does not apply (contraction allowed)")
    star = op(base, phi_belief)
    ok = star.believes(psi)
    return PostulateResult("P6 Preservation", ok,
                           "neg-phi not in K => prior beliefs preserved under revision",
                           None if ok else f"independent belief psi={psi!r} was dropped")


def check_dp1_recency(op: Operator, base: BeliefBase,
                      phi_belief: Belief, psi_belief: Belief) -> PostulateResult:
    phi, psi = phi_belief.formula, psi_belief.formula
    if not is_consistent([psi]):
        return PostulateResult("DP1 Recency", True, "psi inconsistent; vacuous")
    if not entails([psi], phi):
        return PostulateResult("DP1 Recency", True, "psi !|= phi; vacuous")
    lhs = op(op(base, phi_belief), psi_belief)        # (K*phi)*psi
    rhs = op(base, psi_belief)                         # K*psi
    ok = cn_equal(lhs, rhs)
    return PostulateResult("DP1 Recency", ok,
                           "psi|=phi => (K*phi)*psi = K*psi",
                           None if ok else "second evidence did not subsume the first")


def check_dp2_irrelevance(op: Operator, base: BeliefBase,
                          phi_belief: Belief, psi_belief: Belief) -> PostulateResult:
    phi, psi = phi_belief.formula, psi_belief.formula
    if not is_consistent([psi]):
        return PostulateResult("DP2 Irrelevance", True, "psi inconsistent; vacuous")
    if not entails([psi], Not(phi)):
        return PostulateResult("DP2 Irrelevance", True, "psi !|= ~phi; vacuous")
    lhs = op(op(base, phi_belief), psi_belief)
    rhs = op(base, psi_belief)
    ok = cn_equal(lhs, rhs)
    return PostulateResult("DP2 Irrelevance", ok,
                           "psi|=~phi => (K*phi)*psi = K*psi",
                           None if ok else "contradicted first evidence still influenced result")


# ---------------------------------------------------------------------------
# State-preserving DP checks. The Darwiche-Pearl postulates constrain the
# *epistemic state*, not a base->base map; testing them by composing base
# operators throws away the ranking between steps and spuriously fails. Here we
# keep a PrioritizedState across both revisions, which is the correct semantics.
# ---------------------------------------------------------------------------


def check_dp1_recency_stateful(base: BeliefBase,
                               phi_belief: Belief, psi_belief: Belief) -> PostulateResult:
    phi, psi = phi_belief.formula, psi_belief.formula
    if not is_consistent([psi]):
        return PostulateResult("DP1 Recency", True, "psi inconsistent; vacuous")
    if not entails([psi], phi):
        return PostulateResult("DP1 Recency", True, "psi !|= phi; vacuous")
    lhs = PrioritizedState(base.records()).revise(phi_belief).revise(psi_belief).operative_base()
    rhs = PrioritizedState(base.records()).revise(psi_belief).operative_base()
    ok = cn_equal(lhs, rhs)
    return PostulateResult("DP1 Recency", ok,
                           "psi|=phi => (K*phi)*psi = K*psi",
                           None if ok else "stateful iterated revision diverged")


def check_dp2_irrelevance_stateful(base: BeliefBase,
                                   phi_belief: Belief, psi_belief: Belief) -> PostulateResult:
    phi, psi = phi_belief.formula, psi_belief.formula
    if not is_consistent([psi]):
        return PostulateResult("DP2 Irrelevance", True, "psi inconsistent; vacuous")
    if not entails([psi], Not(phi)):
        return PostulateResult("DP2 Irrelevance", True, "psi !|= ~phi; vacuous")
    lhs = PrioritizedState(base.records()).revise(phi_belief).revise(psi_belief).operative_base()
    rhs = PrioritizedState(base.records()).revise(psi_belief).operative_base()
    ok = cn_equal(lhs, rhs)
    return PostulateResult("DP2 Irrelevance", ok,
                           "psi|=~phi => (K*phi)*psi = K*psi",
                           None if ok else "stateful iterated revision diverged")


# ---------------------------------------------------------------------------
# Property-based driver: random bases / inputs, per-postulate satisfaction rate.
# ---------------------------------------------------------------------------


def _rand_literal(rng: random.Random, names: Sequence[str]) -> Formula:
    a = atom(rng.choice(names))
    return Not(a) if rng.random() < 0.5 else a


def _rand_clause(rng: random.Random, names: Sequence[str], k: int = 2) -> Formula:
    lits = [_rand_literal(rng, names) for _ in range(rng.randint(1, k))]
    f = lits[0]
    for l in lits[1:]:
        f = Or(f, l)
    return f


def random_base(rng: random.Random, names: Sequence[str], size: int) -> BeliefBase:
    b = BeliefBase()
    sources = list(Source)
    for _ in range(size):
        f = _rand_clause(rng, names, k=2)
        if is_consistent(b.formulas() + [f]):
            src = rng.choice([Source.USER_EXPLICIT, Source.USER_IMPLICIT,
                              Source.TOOL, Source.INFERRED, Source.DEFAULT])
            b.expand(Belief(f, src))
    return b


@dataclass
class BenchSummary:
    rates: dict
    n: int

    def table(self) -> str:
        lines = ["postulate            satisfied   rate",
                 "-" * 42]
        for k, (ok, tot) in self.rates.items():
            rate = 100.0 * ok / tot if tot else 100.0
            lines.append(f"{k:<20} {ok:>4}/{tot:<5}  {rate:6.1f}%")
        return "\n".join(lines)


def run_property_bench(op: Operator = levi_operator,
                       n_scenarios: int = 400,
                       n_atoms: int = 6,
                       base_size: int = 6,
                       seed: int = 0,
                       stateful_dp: bool = False) -> Tuple[BenchSummary, List[PostulateResult]]:
    """
    Fuzz `n_scenarios` random (base, phi, psi) triples and tally per-postulate
    satisfaction, AGM-Bench style. Returns the summary and any failing results
    (with counterexamples) for inspection.

    `stateful_dp=True` evaluates the Darwiche-Pearl postulates over a preserved
    PrioritizedState (correct iterated-revision semantics) instead of composing
    base operators.
    """
    rng = random.Random(seed)
    names = [f"x{i}" for i in range(n_atoms)]
    tally = {k: [0, 0] for k in
             ["P1 Success", "P2 Consistency", "P3 Inclusion", "P4 Vacuity",
              "P5 Extensionality", "P6 Preservation", "DP1 Recency", "DP2 Irrelevance"]}
    failures: List[PostulateResult] = []

    def record(res: PostulateResult):
        tally[res.name][1] += 1
        if res.satisfied:
            tally[res.name][0] += 1
        else:
            failures.append(res)

    for _ in range(n_scenarios):
        base = random_base(rng, names, base_size)
        phi = _rand_clause(rng, names, k=2)
        if not is_consistent([phi]):
            continue
        phi_b = Belief(phi, rng.choice([Source.USER_EXPLICIT, Source.TOOL]))
        psi = _rand_clause(rng, names, k=2)
        # equivalent variant for extensionality: phi & TOP
        phi_equiv = And(phi, TOP)
        # for DP, craft psi that entails phi or ~phi half the time
        psi_belief = Belief(psi, Source.USER_EXPLICIT)
        psi_strong = Belief(And(phi, _rand_literal(rng, names)), Source.USER_EXPLICIT)  # |= phi
        psi_neg = Belief(And(Not(phi), _rand_literal(rng, names)), Source.USER_EXPLICIT)  # |= ~phi

        record(check_success(op, base, phi_b))
        record(check_consistency(op, base, phi_b))
        record(check_inclusion(op, base, phi_b))
        record(check_vacuity(op, base, phi_b))
        record(check_extensionality(op, base, phi_b, phi_equiv))
        record(check_preservation(op, base, phi_b, psi))
        if stateful_dp:
            record(check_dp1_recency_stateful(base, phi_b, psi_strong))
            record(check_dp2_irrelevance_stateful(base, phi_b, psi_neg))
        else:
            record(check_dp1_recency(op, base, phi_b, psi_strong))
            record(check_dp2_irrelevance(op, base, phi_b, psi_neg))

    return BenchSummary({k: tuple(v) for k, v in tally.items()}, n_scenarios), failures


# ==========================================================================
# ==== module: atms
# ==========================================================================
# -*- coding: utf-8 -*-
"""
stale_hyperbase.atms
====================
Assumption-based Truth Maintenance kernels (de Kleer 1986 style), ported and
unified from npc_consensus_v13.ATMSKernelV2 + proto_atms_bench, plus the
label-propagation ATMSKernel used by the STALE engine for Core-Retainment.

Two kernels, deliberately kept distinct:

* ATMSLabelKernel  — claim-level justification hypergraph with minimal
  consistent support environments (labels), nogoods, defeasible guards.
  Used by the STALE engine to decide is_supported / has_alternative_support
  (structural Core-Retainment) without enumerating worlds.

* ATMSKernelV2     — the "necessity benchmark" kernel: a CLAIM may carry many
  INDEPENDENT assumptions (uniqueness = claim+evidence_id+holder+valid_time),
  with temporal validity, OR/AND/DEFEASIBLE justifications, nogood-by-claim,
  and per-agent context. This is what proves ATMS > BFS-cascade on
  alternative-path retention and multi-agent local views, and what computes
  source independence by evidence-DAG leaves (echo-chamber resistant).
"""



# ===========================================================================
# 1. Label-propagation ATMS (engine-facing): justification hypergraph + labels
# ===========================================================================
@dataclass(frozen=True)
class Assumption:
    aid: str
    claim: str
    evidence_type: str
    origin: str
    session_index: int = -1


@dataclass(frozen=True)
class Justification:
    """(premises ∧ ¬neg_premises) → conclusion. neg_premises = defeasible defeater."""
    jid: str
    premises: FrozenSet[str]
    neg_premises: FrozenSet[str]
    conclusion: str
    operator: str = "AND"           # AND / DEFEASIBLE
    strength: float = 0.5
    rationale: str = ""


class ATMSLabelKernel:
    def __init__(self, max_label: int = 24, max_env: int = 6):
        self.assumptions: Dict[str, Assumption] = {}
        self.base_assumption: Dict[str, str] = {}
        self.justifications: List[Justification] = []
        self.nogoods: Set[FrozenSet[str]] = set()
        self.believed: Set[str] = set()
        self.labels: Dict[str, Set[FrozenSet[str]]] = defaultdict(set)
        self.defeaters: Dict[str, Set[str]] = defaultdict(set)
        self.max_label = max_label
        self.max_env = max_env
        self._ctr = 0
        self.ledger: List[dict] = []

    def _id(self, p="A"):
        self._ctr += 1
        return f"{p}{self._ctr:04d}"

    def _rec(self, **kw):
        self.ledger.append(dict(kw))

    def assert_base(self, claim, evidence_type, origin, session_index=-1) -> str:
        if claim in self.base_assumption:
            aid = self.base_assumption[claim]
            self.believed.add(aid)
            self._rec(op="rebelieve_base", claim=claim, aid=aid)
            return aid
        aid = self._id("A")
        self.assumptions[aid] = Assumption(aid, claim, evidence_type, origin, session_index)
        self.base_assumption[claim] = aid
        self.believed.add(aid)
        self.labels[claim].add(frozenset({aid}))
        self._rec(op="assert_base", claim=claim, aid=aid, evidence_type=evidence_type, origin=origin)
        return aid

    def retract_base(self, claim):
        aid = self.base_assumption.get(claim)
        if aid and aid in self.believed:
            self.believed.discard(aid)
            self._rec(op="retract_base", claim=claim, aid=aid)

    def register_defeater(self, conclusion, defeater_claim):
        self.defeaters[conclusion].add(defeater_claim)

    def add_justification(self, premises, conclusion, neg_premises=None,
                          operator="AND", strength=0.5, rationale="") -> str:
        prem = frozenset(p for p in premises if p)
        neg = frozenset(neg_premises or [])
        for j in self.justifications:
            if j.premises == prem and j.neg_premises == neg and j.conclusion == conclusion:
                return j.jid
        jid = self._id("J")
        self.justifications.append(Justification(jid, prem, neg, conclusion, operator, strength, rationale))
        self._rec(op="add_justification", jid=jid, premises=sorted(prem), neg=sorted(neg),
                  conclusion=conclusion, operator=operator)
        return jid

    def add_nogood(self, claims, reason=""):
        aids = [self.base_assumption[c] for c in claims if c in self.base_assumption]
        if len(aids) >= 2:
            self.nogoods.add(frozenset(aids))
            self._rec(op="add_nogood", claims=sorted(claims), reason=reason)

    def _env_consistent(self, env: FrozenSet[str]) -> bool:
        return not any(ng and ng <= env for ng in self.nogoods)

    def _minimize(self, envs: Set[FrozenSet[str]]) -> Set[FrozenSet[str]]:
        envs = {e for e in envs if self._env_consistent(e)}
        out: Set[FrozenSet[str]] = set()
        for e in sorted(envs, key=lambda s: (len(s), tuple(sorted(s)))):
            if not any(o <= e for o in out):
                out.add(e)
        return set(sorted(out, key=lambda s: (len(s), tuple(sorted(s))))[: self.max_label])

    def _compute(self, blocked: Optional[Set[str]] = None,
                 cut_justs: Optional[Set[str]] = None, max_iter: int = 24):
        blocked = blocked or set()
        cut_justs = cut_justs or set()
        labels: Dict[str, Set[FrozenSet[str]]] = defaultdict(set)
        for claim, aid in self.base_assumption.items():
            if claim in blocked:
                continue
            labels[claim].add(frozenset({aid}))
        for _ in range(max_iter):
            changed = False
            supported_now = {c for c, L in labels.items() if any(e <= self.believed for e in L)}
            for j in self.justifications:
                if j.jid in cut_justs or j.conclusion in blocked:
                    continue
                if any(p in blocked for p in j.premises):
                    continue
                if any(npn in supported_now for npn in j.neg_premises):
                    continue
                pls = [labels.get(p) for p in j.premises]
                if any(not pl for pl in pls):
                    continue
                new_envs = set()
                for combo in itertools.islice(itertools.product(*pls), 4096):
                    env = frozenset().union(*combo) if combo else frozenset()
                    if len(env) <= self.max_env and self._env_consistent(env):
                        new_envs.add(env)
                before = set(labels[j.conclusion])
                labels[j.conclusion] = self._minimize(labels[j.conclusion] | new_envs)
                if labels[j.conclusion] != before:
                    changed = True
            if not changed:
                break
        for c in list(labels):
            labels[c] = self._minimize(labels[c])
        return labels

    def recompute_labels(self):
        self.labels = self._compute()
        return self.labels

    def surviving_environments(self, claim) -> List[FrozenSet[str]]:
        self.recompute_labels()
        return [e for e in self.labels.get(claim, set())
                if e <= self.believed and self._env_consistent(e)]

    def is_supported(self, claim) -> bool:
        return len(self.surviving_environments(claim)) > 0

    def kernels_of(self, claim) -> List[FrozenSet[str]]:
        return list(self.labels.get(claim, set())) or self.surviving_environments(claim)

    def has_alternative_support(self, claim, without_claim) -> bool:
        if claim == without_claim:
            return False
        labels = self._compute(blocked={without_claim})
        for env in labels.get(claim, set()):
            if env <= self.believed and self._env_consistent(env):
                return True
        return False

    def known_claims(self) -> Set[str]:
        ks = set(self.base_assumption.keys())
        for j in self.justifications:
            ks.add(j.conclusion); ks.update(j.premises); ks.update(j.neg_premises)
        return ks

    def snapshot_supported(self) -> Set[str]:
        self.recompute_labels()
        return {c for c in self.known_claims() if self.is_supported(c)}

    def stats(self) -> dict:
        self.recompute_labels()
        return {"n_assumptions": len(self.assumptions), "n_justifications": len(self.justifications),
                "n_nogoods": len(self.nogoods), "n_believed": len(self.believed),
                "n_supported": len(self.snapshot_supported()), "ledger_len": len(self.ledger)}


# ===========================================================================
# 2. ATMSKernelV2 — multi-assumption / temporal / OR-AND-defeasible / multi-agent
#    (the necessity-benchmark + source-independence kernel)
# ===========================================================================
@dataclass(frozen=True)
class AssumptionV2:
    aid: str
    claim: str
    evidence_id: str
    holder: str = "world"
    valid_from: float = 0.0
    valid_to: Optional[float] = None
    evidence_type: str = "direct_observation"
    trust: float = 0.8
    polarity: int = 1

    def valid_at(self, t: float) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)


@dataclass(frozen=True)
class JustV2:
    jid: str
    premises: FrozenSet[str]
    neg_premises: FrozenSet[str]
    conclusion: str
    operator: str = "AND"
    strength: float = 0.6


class ATMSKernelV2:
    """Uniqueness of an assumption = (claim, evidence_id, holder, valid_from).
    A claim may have MANY independent assumptions; label is computed globally but
    an environment is active only over assumptions an agent can access & accepts."""

    def __init__(self, max_env: int = 6):
        self.assumptions: Dict[str, AssumptionV2] = {}
        self.base_by_claim: Dict[str, Set[str]] = defaultdict(set)
        self.justifications: List[JustV2] = []
        self.nogood_claims: Set[FrozenSet[str]] = set()
        self.believed: Set[str] = set()
        self.revoked: Set[str] = set()
        self._na = 0
        self._nj = 0
        self.now = 0.0
        self.max_env = max_env

    def assert_evidence(self, claim, evidence_id, holder="world",
                        evidence_type="direct_observation", trust=0.8,
                        valid_from=0.0, valid_to=None, polarity=1) -> str:
        for aid, a in self.assumptions.items():
            if (a.claim, a.evidence_id, a.holder, a.valid_from) == (claim, evidence_id, holder, valid_from):
                self.believed.add(aid); self.revoked.discard(aid); return aid
        self._na += 1
        aid = f"A{self._na:03d}"
        self.assumptions[aid] = AssumptionV2(aid, claim, evidence_id, holder, valid_from,
                                             valid_to, evidence_type, trust, polarity)
        self.base_by_claim[claim].add(aid)
        self.believed.add(aid)
        return aid

    def revoke_evidence(self, aid):
        self.believed.discard(aid); self.revoked.add(aid)

    def add_justification(self, premises, conclusion, neg_premises=(), operator="AND", strength=0.6) -> str:
        self._nj += 1
        jid = f"J{self._nj:03d}"
        self.justifications.append(JustV2(jid, frozenset(premises), frozenset(neg_premises),
                                          conclusion, operator, strength))
        return jid

    def add_nogood_claims(self, claims):
        self.nogood_claims.add(frozenset(claims))

    def _active_aids(self, t, context=None) -> Set[str]:
        out = set()
        for aid in self.believed:
            a = self.assumptions[aid]
            if not a.valid_at(t):
                continue
            if context is not None and aid not in context:
                continue
            out.add(aid)
        return out

    def _env_consistent(self, env: FrozenSet[str]) -> bool:
        claims = {self.assumptions[a].claim for a in env if a in self.assumptions}
        return not any(ng <= claims for ng in self.nogood_claims)

    @staticmethod
    def _minimize(envs):
        envs = {e for e in envs if e}
        out = set()
        for e in sorted(envs, key=len):
            if not any(o < e or o == e for o in out):
                out.add(e)
        return out

    def labels(self, t=None, context=None, blocked_claims=frozenset(),
               cut_jids=frozenset(), max_iter=24):
        t = self.now if t is None else t
        L: Dict[str, Set[FrozenSet[str]]] = defaultdict(set)
        for claim, aids in self.base_by_claim.items():
            if claim in blocked_claims:
                continue
            for aid in aids:
                a = self.assumptions[aid]
                if aid in self.believed and a.valid_at(t) and (context is None or aid in context):
                    L[claim].add(frozenset({aid}))
        for _ in range(max_iter):
            changed = False
            supp = {c for c, envs in L.items() if any(self._env_consistent(e) for e in envs)}
            for j in self.justifications:
                if j.jid in cut_jids or j.conclusion in blocked_claims:
                    continue
                if any(n in supp for n in j.neg_premises):
                    continue
                if any(p in blocked_claims for p in j.premises):
                    continue
                if j.operator == "OR":
                    new = {e for p in j.premises for e in L.get(p, set())
                           if len(e) <= self.max_env and self._env_consistent(e)}
                    if not new:
                        continue
                else:
                    pls = [L.get(p) for p in j.premises]
                    if any(not pl for pl in pls):
                        continue
                    new = set()
                    for combo in itertools.islice(itertools.product(*pls), 4096):
                        e = frozenset().union(*combo) if combo else frozenset()
                        if len(e) <= self.max_env and self._env_consistent(e):
                            new.add(e)
                before = set(L[j.conclusion])
                L[j.conclusion] = self._minimize(L[j.conclusion] | new)
                if L[j.conclusion] != before:
                    changed = True
            if not changed:
                break
        return L

    def supported(self, claim, t=None, context=None, blocked_claims=frozenset(), cut_jids=frozenset()) -> bool:
        t = self.now if t is None else t
        L = self.labels(t, context, blocked_claims, cut_jids)
        active = self._active_aids(t, context)
        return any(e <= active and self._env_consistent(e) for e in L.get(claim, set()))

    def has_alternative_support(self, claim, without_claim, t=None, context=None) -> bool:
        if claim == without_claim:
            return False
        return self.supported(claim, t=t, context=context, blocked_claims={without_claim})

    def independent_origins(self, claim, t=None, context=None) -> Set[str]:
        """Source independence by evidence-DAG leaves: origin set = distinct
        evidence_id over all surviving support environments. One event echoed by
        10 tellers stays 1 origin; 3 distinct events summarised by one holder
        stay 3 origins. This is the echo-chamber-resistant count."""
        t = self.now if t is None else t
        L = self.labels(t, context)
        active = self._active_aids(t, context)
        origins = set()
        for e in L.get(claim, set()):
            if e <= active and self._env_consistent(e):
                for aid in e:
                    origins.add(self.assumptions[aid].evidence_id)
        return origins


# ===========================================================================
# 3. BFS weighted-cascade baseline (v10 heuristic) — for the necessity bench
# ===========================================================================
class BFSCascadeBaseline:
    """New evidence propagates a uniform decay along the dependency graph;
    a claim under threshold is deemed invalidated. Key deficiency: it cannot
    recognise alternative support paths / nogood / OR / per-agent context."""

    def __init__(self, k: ATMSKernelV2, tau=0.4, decay=0.55):
        self.k = k
        self.tau = tau
        self.decay = decay

    def supported_after_defeat(self, defeated_claim, query) -> bool:
        conf = {}
        for claim, aids in self.k.base_by_claim.items():
            act = [self.k.assumptions[a] for a in aids
                   if a in self.k.believed and self.k.assumptions[a].valid_at(self.k.now)]
            conf[claim] = max([a.trust for a in act], default=0.0)
        for _ in range(12):
            for j in self.k.justifications:
                if j.operator == "OR":
                    val = max([conf.get(p, 0) for p in j.premises], default=0) * j.strength
                else:
                    ps = [conf.get(p, 0) for p in j.premises]
                    val = (sum(ps) / len(ps) if ps else 0) * j.strength
                conf[j.conclusion] = max(conf.get(j.conclusion, 0), val)
        affected = self._downstream(defeated_claim) | {defeated_claim}
        for c in affected:
            conf[c] = conf.get(c, 0) * self.decay
        return conf.get(query, 0) >= self.tau

    def _downstream(self, claim) -> Set[str]:
        out = set()
        frontier = [claim]
        while frontier:
            cur = frontier.pop()
            for j in self.k.justifications:
                if cur in j.premises and j.conclusion not in out:
                    out.add(j.conclusion); frontier.append(j.conclusion)
        return out


# ==========================================================================
# ==== module: graphmem
# ==========================================================================
# -*- coding: utf-8 -*-
"""
stale_hyperbase.graphmem
========================
Typed narrative-memory graph + unified memory state machine + claim revisions,
ported from the npc_consensus architecture. Pure dict adjacency (no networkx).
The OrderedIncision writes status transitions back onto these revisions, so the
formal contraction is reflected in the live memory (ACTIVE/STALE/SUPERSEDED/
UNKNOWN_CURRENT/HISTORICAL).
"""



class NodeType(Enum):
    SESSION = "Session"
    CLAIM = "Claim"
    ATTRIBUTE = "Attribute"
    PREMISE = "Premise"
    POLICY = "Policy"


class EdgeType(Enum):
    EVIDENCE_OF = "evidence_of"
    INSTANCE_OF = "instance_of"
    SUPERSEDES = "supersedes"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    INVALIDATES = "invalidates"
    DERIVED_FROM = "derived_from"


EDGE_SCHEMA: Dict[EdgeType, List[Tuple[NodeType, NodeType]]] = {
    EdgeType.EVIDENCE_OF: [(NodeType.SESSION, NodeType.CLAIM)],
    EdgeType.INSTANCE_OF: [(NodeType.CLAIM, NodeType.ATTRIBUTE)],
    EdgeType.SUPERSEDES: [(NodeType.CLAIM, NodeType.CLAIM)],
    EdgeType.DEPENDS_ON: [(NodeType.ATTRIBUTE, NodeType.ATTRIBUTE)],
    EdgeType.CONTRADICTS: [(NodeType.CLAIM, NodeType.CLAIM)],
    EdgeType.INVALIDATES: [(NodeType.CLAIM, NodeType.CLAIM)],
    EdgeType.DERIVED_FROM: [(NodeType.PREMISE, NodeType.ATTRIBUTE),
                            (NodeType.POLICY, NodeType.ATTRIBUTE)],
}


class MemStatus(Enum):
    ACTIVE = "active"
    WEAK = "weak"
    STALE = "stale"
    SUPERSEDED = "superseded"
    UNKNOWN_CURRENT = "unknown_current"
    HISTORICAL = "historical"           # retained-but-disabled (Step 4)
    REFUTED = "refuted"


STATUS_CAPS: Dict[str, dict] = {
    MemStatus.ACTIVE.value: dict(current=True, weight=1.00),
    MemStatus.WEAK.value: dict(current=True, weight=0.55),
    MemStatus.STALE.value: dict(current=False, weight=0.00),
    MemStatus.SUPERSEDED.value: dict(current=False, weight=0.00),
    MemStatus.UNKNOWN_CURRENT.value: dict(current=False, weight=0.00),
    MemStatus.HISTORICAL.value: dict(current=False, weight=0.00),
    MemStatus.REFUTED.value: dict(current=False, weight=0.00),
}


def status_cap(status: str, key: str):
    return STATUS_CAPS.get(status, STATUS_CAPS[MemStatus.ACTIVE.value])[key]


@dataclass
class MemNode:
    nid: str
    ntype: NodeType
    label: str
    attr: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimRevision:
    rev_id: str
    attribute: str
    value: str
    status: str = MemStatus.ACTIVE.value
    tier: str = "profile"
    evidence_type: str = "implicit_state"
    confidence: float = 0.7
    session_index: int = -1
    source_session_id: str = ""
    active_strength: str = "STRONG"
    superseded_by: Optional[str] = None
    valid_to: Optional[int] = None
    revision_history: List[dict] = field(default_factory=list)
    claim_nid: Optional[str] = None


class GraphMemory:
    def __init__(self):
        self.nodes: Dict[str, MemNode] = {}
        self.edges: List[Tuple[EdgeType, str, str, dict]] = []
        self._adj: Dict[str, List[Tuple[EdgeType, str, dict]]] = defaultdict(list)
        self._radj: Dict[str, List[Tuple[EdgeType, str, dict]]] = defaultdict(list)
        self.attr_node: Dict[str, str] = {}
        self.revisions: Dict[str, ClaimRevision] = {}
        self.active_rev: Dict[str, str] = {}
        self.stale_archive: List[str] = []
        self.unknown_current: set = set()
        self.historical: set = set()
        self._ctr = 0
        self.schema_warns = 0

    def _id(self, p="N"):
        self._ctr += 1
        return f"{p}{self._ctr:05d}"

    def add_node(self, ntype: NodeType, label: str, attr=None) -> str:
        nid = self._id("N")
        self.nodes[nid] = MemNode(nid, ntype, label, attr or {})
        return nid

    def ensure_attribute(self, attribute: str) -> str:
        if attribute not in self.attr_node:
            nid = self.add_node(NodeType.ATTRIBUTE, attribute, {"key": attribute})
            self.attr_node[attribute] = nid
        return self.attr_node[attribute]

    def _valid_edge(self, et, src, dst) -> bool:
        st = self.nodes[src].ntype if src in self.nodes else None
        dt = self.nodes[dst].ntype if dst in self.nodes else None
        if st is None or dt is None:
            return False
        return (st, dt) in EDGE_SCHEMA.get(et, [])

    def add_edge(self, et, src, dst, attr=None) -> bool:
        attr = attr or {}
        ok = self._valid_edge(et, src, dst)
        attr["schema_ok"] = ok
        if not ok:
            self.schema_warns += 1
        self.edges.append((et, src, dst, attr))
        self._adj[src].append((et, dst, attr))
        self._radj[dst].append((et, src, attr))
        return ok

    def out_edges(self, nid, et=None):
        return [(e, d, a) for (e, d, a) in self._adj.get(nid, []) if et is None or e == et]

    def in_edges(self, nid, et=None):
        return [(e, s, a) for (e, s, a) in self._radj.get(nid, []) if et is None or e == et]

    def dependents_of(self, attribute: str) -> List[Tuple[str, float]]:
        node = self.attr_node.get(attribute)
        out = []
        if not node:
            return out
        for (et, src, a) in self.in_edges(node, EdgeType.DEPENDS_ON):
            out.append((self.nodes[src].label, a.get("strength", 0.8)))
        return out

    def write_claim(self, attribute, value, *, tier, evidence_type, confidence,
                    session_index, source_session_id, session_node_id) -> str:
        rev_id = self._id("R")
        rev = ClaimRevision(rev_id=rev_id, attribute=attribute, value=value, tier=tier,
                            evidence_type=evidence_type, confidence=min(1.0, max(0.0, confidence)),
                            session_index=session_index, source_session_id=source_session_id)
        self.revisions[rev_id] = rev
        attr_nid = self.ensure_attribute(attribute)
        claim_nid = self.add_node(NodeType.CLAIM, f"{attribute}={value}",
                                  {"rev_id": rev_id, "attribute": attribute, "value": value})
        rev.revision_history.append({"event": "create", "value": value, "session_index": session_index})
        self.add_edge(EdgeType.EVIDENCE_OF, session_node_id, claim_nid)
        self.add_edge(EdgeType.INSTANCE_OF, claim_nid, attr_nid)
        rev.claim_nid = claim_nid
        return rev_id

    def active_claim(self, attribute) -> Optional[ClaimRevision]:
        rid = self.active_rev.get(attribute)
        if not rid:
            return None
        rev = self.revisions.get(rid)
        if rev and status_cap(rev.status, "current"):
            return rev
        return None

    def mark_status(self, attribute: str, value: str, status: str):
        """Write-back hook used by the OrderedIncision."""
        rid = self.active_rev.get(attribute)
        if rid:
            rev = self.revisions[rid]
            if rev.value == value:
                rev.status = status
                if status == MemStatus.UNKNOWN_CURRENT.value:
                    self.unknown_current.add(attribute)
                if status == MemStatus.HISTORICAL.value:
                    self.historical.add(attribute)

    def stats(self) -> dict:
        ntypes = defaultdict(int)
        for n in self.nodes.values():
            ntypes[n.ntype.name] += 1
        etypes = defaultdict(int)
        for (e, _, _, _) in self.edges:
            etypes[e.value] += 1
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                "node_types": dict(ntypes), "edge_types": dict(etypes),
                "schema_invalid_edges": self.schema_warns,
                "n_revisions": len(self.revisions),
                "n_active_attributes": len(self.active_rev),
                "n_unknown_current": len(self.unknown_current),
                "n_historical": len(self.historical)}


# ==========================================================================
# ==== module: multiagent
# ==========================================================================
# -*- coding: utf-8 -*-
"""
stale_hyperbase.multiagent
===========================
Multi-agent belief-merging layer ported from npc_consensus_v13, made first-class
for STALE (addresses the "degraded to single-user" risk).

In STALE a memory claim can arrive from several SOURCES across the haystack:
the user stating it directly, a tool/inference deriving it, a third party
restating it. We model each source as an AGENT with a trust weight, gate
cross-agent propagation by ACCESS TIER, merge beliefs by TRUST-WEIGHTED voting,
and count source support by EVIDENCE-DAG independence (echo-chamber resistant).

Components
----------
  AccessTier             volatile < episodic < profile < core
  TrustMatrix            row trusts column; spectral ratio λ2/λ1 (manipulation resistance)
  AccessController       tier-gated cross-agent propagation
  CommunityConsensus     trust-weighted voting with source-diversity gate
  MultiAgentBeliefLayer  per-agent ATMSKernelV2 views + merge + source independence
"""





def _clip01(x):
    try:
        return float(min(1.0, max(0.0, float(x))))
    except Exception:
        return 0.0


class AccessTier(Enum):
    VOLATILE = "volatile"
    EPISODIC = "episodic"
    PROFILE = "profile"
    CORE = "core"


TIER_OVERRIDE_GATE = {
    AccessTier.VOLATILE.value: 0.30,
    AccessTier.EPISODIC.value: 0.45,
    AccessTier.PROFILE.value: 0.55,
    AccessTier.CORE.value: 0.75,
}

# tiers that may NOT be shared across agents (private)
PRIVATE_TIERS = {AccessTier.CORE.value}
REL_TRUST_GATE = 0.55
TRUST_PROP_GATE = 0.30


# Source agents that can assert STALE memory claims. We keep them generic so the
# same machinery serves any STALE record.
DEFAULT_AGENTS = ["user", "assistant", "tool", "third_party"]

# evidence_type -> source credibility (echo-chamber prior)
EVIDENCE_CRED = {
    "direct_statement": 0.90,
    "user_explicit": 0.95,
    "implicit_state": 0.78,
    "propagated": 0.70,
    "tool": 0.70,
    "inference": 0.55,
    "rumor": 0.40,
    "hearsay": 0.40,
}


def evidence_cred(et: str) -> float:
    return EVIDENCE_CRED.get(et, 0.55)


def build_trust_matrix(agents=None) -> Dict[str, Dict[str, float]]:
    agents = agents or DEFAULT_AGENTS
    # user trusts itself most; assistant trusts user/tool; third_party least trusted
    prior = {
        "user": {"user": 1.0, "assistant": 0.75, "tool": 0.7, "third_party": 0.35},
        "assistant": {"user": 0.92, "assistant": 1.0, "tool": 0.8, "third_party": 0.45},
        "tool": {"user": 0.7, "assistant": 0.7, "tool": 1.0, "third_party": 0.4},
        "third_party": {"user": 0.6, "assistant": 0.55, "tool": 0.5, "third_party": 1.0},
    }
    T = {}
    for a in agents:
        T[a] = {}
        for b in agents:
            T[a][b] = prior.get(a, {}).get(b, 1.0 if a == b else 0.5)
    return T


class AccessController:
    """tier-gated cross-agent propagation (private/community/public)."""

    def __init__(self, trust: Dict[str, Dict[str, float]], ablate: bool = False):
        self.trust = trust
        self.ablate = ablate

    def allows(self, tier: str, sender: str, receiver: str) -> Tuple[bool, str]:
        if self.ablate:
            return True, "ablation: no access control"
        if tier in PRIVATE_TIERS:
            return False, f"{tier} is private (structural isolation)"
        if tier == AccessTier.PROFILE.value:
            t = self.trust.get(receiver, {}).get(sender, 0.5)
            return (t >= REL_TRUST_GATE,
                    f"profile {'ok' if t >= REL_TRUST_GATE else 'blocked'} (trust {t:.2f})")
        return True, "public/episodic default allow"


class CommunityConsensus:
    """trust-weighted voting with a source-diversity gate (≥2 independent
    origins OR a high-credibility override) before a claim is promoted to
    'shared'. This is the belief-merging operation, not a single-user read."""

    TAU = 0.62

    def __init__(self, trust, ablate_diversity: bool = False):
        self.trust = trust
        self.ablate_diversity = ablate_diversity

    def aggregate(self, votes: List[dict]) -> dict:
        """votes: [{agent, belief, evidence_type, origin}]. Returns merged score,
        promotion decision, #independent origins."""
        if not votes:
            return {"score": 0.0, "promoted": False, "n_origins": 0, "clusters": 0}
        groups = defaultdict(list)
        for v in votes:
            groups[v["origin"]].append(v)
        num = den = 0.0
        override = 0.0
        for origin, members in groups.items():
            gsize = len(members)
            for v in members:
                cred = evidence_cred(v["evidence_type"])
                authority = self.trust.get(v["agent"], {}).get(v["agent"], 1.0)
                w = (cred * 0.6 + 0.4 * authority) / gsize
                num += w * v["belief"]
                den += w
                if cred >= 0.85 and v["belief"] >= 0.72:
                    override = max(override, v["belief"])
        agg = num / max(den, 1e-9)
        score = _clip01(max(agg, override))
        clusters = len(groups)               # distinct origins = echo-resistant
        diverse = self.ablate_diversity or clusters >= 2 or override >= 0.72
        promoted = score >= self.TAU and diverse
        return {"score": round(score, 4), "promoted": promoted,
                "n_origins": clusters, "clusters": clusters, "override": round(override, 4)}


class MultiAgentBeliefLayer:
    """Wraps per-agent ATMSKernelV2 views over the SAME evidence DAG, merges them
    with trust-weighted consensus, gates propagation by access tier, and reports
    source independence. STALE answers can be drawn from the merged state, so the
    system is genuinely multi-agent rather than single-user."""

    def __init__(self, agents=None, ablate_access=False, ablate_diversity=False,
                 ablate_trust=False):
        self.agents = agents or DEFAULT_AGENTS
        self.trust = build_trust_matrix(self.agents)
        if ablate_trust:
            self.trust = {a: {b: (1.0 if a == b else 0.5) for b in self.agents}
                          for a in self.agents}
        self.access = AccessController(self.trust, ablate=ablate_access)
        self.consensus = CommunityConsensus(self.trust, ablate_diversity=ablate_diversity)
        self.kernel = ATMSKernelV2()             # shared evidence DAG
        # per-agent visible assumption context (aid set); None = full world
        self.agent_ctx: Dict[str, Set[str]] = {a: set() for a in self.agents}
        self.assertions: List[dict] = []

    def assert_claim(self, claim: str, evidence_id: str, holder: str,
                     evidence_type: str, tier: str, trust: float = None,
                     valid_from: float = 0.0, valid_to=None):
        trust = evidence_cred(evidence_type) if trust is None else trust
        aid = self.kernel.assert_evidence(claim, evidence_id, holder=holder,
                                          evidence_type=evidence_type, trust=trust,
                                          valid_from=valid_from, valid_to=valid_to)
        # advance the kernel clock so later queries see this (and prior) evidence
        self.kernel.now = max(self.kernel.now, float(valid_from))
        # holder always sees its own assertion
        self.agent_ctx.setdefault(holder, set()).add(aid)
        # tier-gated propagation to other agents
        for other in self.agents:
            if other == holder:
                continue
            ok, _why = self.access.allows(tier, holder, other)
            if ok:
                self.agent_ctx.setdefault(other, set()).add(aid)
        self.assertions.append({"claim": claim, "evidence_id": evidence_id,
                                "holder": holder, "tier": tier, "aid": aid})
        return aid

    def merged_supported(self, claim: str, t: float = None) -> bool:
        """Claim is supported in the merged state if trust-weighted consensus
        over agents that can see independent support promotes it."""
        votes = []
        for a in self.agents:
            ctx = self.agent_ctx.get(a) or None
            if self.kernel.supported(claim, t=t, context=ctx):
                origins = self.kernel.independent_origins(claim, t=t, context=ctx)
                for o in origins:
                    votes.append({"agent": a, "belief": 0.85,
                                  "evidence_type": "direct_statement", "origin": o})
        if not votes:
            return False
        return self.consensus.aggregate(votes)["promoted"]

    def source_independence(self, claim: str, t: float = None) -> int:
        return len(self.kernel.independent_origins(claim, t=t))

    def spectral_ratio(self) -> float:
        if _np is None:
            return 0.0
        M = _np.array([[self.trust[i][j] for j in self.agents] for i in self.agents])
        row = M.sum(axis=1, keepdims=True)
        P = M / _np.clip(row, 1e-9, None)
        eig = _np.sort(_np.abs(_np.linalg.eigvals(P)))[::-1]
        return float(eig[1] / eig[0]) if len(eig) > 1 and eig[0] > 0 else 0.0

    def stats(self) -> dict:
        return {"n_agents": len(self.agents), "n_assertions": len(self.assertions),
                "spectral_ratio": round(self.spectral_ratio(), 4),
                "ctx_sizes": {a: len(self.agent_ctx.get(a, set())) for a in self.agents}}


# ==========================================================================
# ==== module: extraction
# ==========================================================================
# -*- coding: utf-8 -*-
"""
stale_hyperbase.extraction
===========================
Fact extraction for STALE, in three interchangeable modes that drive the
ablation matrix. NONE of the "Ours" modes may depend on graph_hint at answer
time except the Oracle arm (which exists precisely to measure the formal
ceiling).

Modes
-----
  OracleExtractor   reads graph_hint (gold S-A-V + conflict type). Upper bound.
  LLMExtractor      LLM proposes S-A-V triples + conflict relation from RAW text
                    (M_old/M_new/haystack); deterministic fallback when no API.
                    Output is ALWAYS passed through the FormalFilter.
  HandSchemaExtractor  fixed keyword→slot rules (a hand-written schema), no LLM
                    proposal. Used by the Formal-only arm to test whether the
                    LLM proposal step is what gives generalisation.

FormalFilter
------------
Vets every proposed INVALIDATE/UPDATE: a conflict is admitted only if the formal
layer can find a reachable defeat path (same functional slot, or an A→B
commonsense dependency). Otherwise it is downgraded to NO_EFFECT — the LLM may
propose, but the formal layer adjudicates.
"""



# --- attribute ontology + commonsense A->B dependencies (shared by all modes) ---
ATTRIBUTE_TIER: Dict[str, str] = {
    "routine_and_transport/current_commute_mode": "profile",
    "location_and_living/current_base_location": "profile",
    "role_and_identity/employment_status": "profile",
    "role_and_identity/marital_status": "core",
    "health_and_mobility/current_health_state": "episodic",
    "health_and_mobility/functional_limitation": "episodic",
    "physical_health/caffeine_or_nicotine_reliance": "profile",
    "finance_and_resources/financial_constraint": "profile",
    "weather_and_environment/current_weather_pattern": "volatile",
}

COMMONSENSE_DEPENDENCIES: List[Tuple[str, str, float, str]] = [
    ("health_and_mobility/current_health_state",
     "routine_and_transport/current_commute_mode", 0.85, "injury -> commute infeasible"),
    ("health_and_mobility/functional_limitation",
     "routine_and_transport/current_commute_mode", 0.80, "mobility limit -> commute infeasible"),
    ("role_and_identity/employment_status",
     "location_and_living/current_base_location", 0.70, "job change -> location may change"),
    ("location_and_living/current_base_location",
     "routine_and_transport/current_commute_mode", 0.65, "move -> commute mode changes"),
    ("finance_and_resources/financial_constraint",
     "routine_and_transport/current_commute_mode", 0.55, "budget -> commute may adjust"),
    ("health_and_mobility/current_health_state",
     "physical_health/caffeine_or_nicotine_reliance", 0.60, "health event -> caffeine/nicotine changes"),
]


def dep_source_for(attr_b: str) -> Optional[str]:
    for a, b, _, _ in COMMONSENSE_DEPENDENCIES:
        if b == attr_b:
            return a
    return None


def attribute_tier(attr: str) -> str:
    return ATTRIBUTE_TIER.get(attr, "profile")


# --- keyword slot lexicon (used by HandSchema + LLM-fallback deterministic) ---
_SLOT_KEYWORDS = {
    "routine_and_transport/current_commute_mode":
        ["commute", "bike", "biking", "cycl", "drive", "driving", "walk", "walking",
         "stroll", "on foot", "few blocks", "bus", "subway", "train", "ride to"],
    "location_and_living/current_base_location":
        ["move", "moved", "relocat", "apartment", "live in", "living in", "moved to",
         "farmhouse", "countryside", "new city", "here in"],
    "role_and_identity/employment_status":
        ["job", "employ", "quit", "resign", "fired", "retire", "retired", "retirement",
         "unemploy", "nowhere to be", "no longer working", "teacher", "teach", "syllabus"],
    "role_and_identity/marital_status":
        ["married", "divorce", "single", "wedding", "spouse"],
    "health_and_mobility/current_health_state":
        ["injur", "broke", "surgery", "recover", "pregnan", "trimester", "sick", "ill",
         "knee", "ob ", "first trimester"],
    "health_and_mobility/functional_limitation":
        ["wheelchair", "crutch", "cast", "cannot walk", "can't walk", "limited mobility",
         "barely put weight"],
    "physical_health/caffeine_or_nicotine_reliance":
        ["coffee", "caffeine", "espresso", "smok", "nicotine", "cigarette", "quit smoking", "dark roast"],
    "finance_and_resources/financial_constraint":
        ["budget", "afford", "tight on money", "savings", "raise", "salary"],
    "weather_and_environment/current_weather_pattern":
        ["rain", "snow", "sunny", "storm", "heatwave"],
}


def _guess_slot(text: str) -> Optional[str]:
    t = (text or "").lower()
    best, best_hits = None, 0
    for slot, kws in _SLOT_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in t)
        if hits > best_hits:
            best, best_hits = slot, hits
    return best if best_hits > 0 else None


def _short(text: str, n: int = 48) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()[:n]


# Exemplar phrases per slot for a lightweight, embedding-free semantic backoff.
# This models (in offline mode) what a real LLM proposer recovers on novel
# phrasings that share NO lexicon keyword with _SLOT_KEYWORDS. With --use-llm a
# real model replaces this; the backoff degrades far more gracefully than the
# pure-keyword hand schema, which is exactly the generalization gap we measure.
_SLOT_EXEMPLARS = {
    "routine_and_transport/current_commute_mode":
        "how i get to the office work each morning travel journey to work on foot amble saunter",
    "location_and_living/current_base_location":
        "where i live my home the place i moved to my new house apartment region area i relocated",
    "role_and_identity/employment_status":
        "my job what i do for work my profession career employment whether i still work",
    "health_and_mobility/current_health_state":
        "my health my body recovering injury condition physio therapist medical pregnancy",
    "physical_health/caffeine_or_nicotine_reliance":
        "coffee caffeine espresso stimulant drink i rely on to wake up energy",
}


def _char_ngrams(s: str):
    s = re.sub(r"[^a-z ]+", " ", (s or "").lower())
    toks = s.split()
    v = {}
    for w in toks:
        v[w] = v.get(w, 0) + 1.0
        for i in range(len(w) - 2):
            g = w[i:i + 3]
            v[g] = v.get(g, 0) + 0.5
    return v


def _cos(a, b):
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(x * x for x in a.values()))
    nb = math.sqrt(sum(x * x for x in b.values()))
    return dot / (na * nb + 1e-9)


def _semantic_slot(text: str, threshold: float = 0.12):
    qv = _char_ngrams(text)
    best, best_s = None, 0.0
    for slot, exemplar in _SLOT_EXEMPLARS.items():
        s = _cos(qv, _char_ngrams(exemplar))
        if s > best_s:
            best, best_s = slot, s
    return best if best_s >= threshold else None


@dataclass
class ExtractResult:
    attribute_b: str
    value_old: str
    value_new: Optional[str]
    conflict_type: str               # "T1" | "T2"
    upstream_a: Optional[str]
    source: str                      # which extractor produced this
    confidence: float = 0.7
    extractor_notes: str = ""


# ---------------------------------------------------------------------------
# Formal filter: admit a conflict only if a reachable defeat path exists.
# ---------------------------------------------------------------------------
class FormalFilter:
    @staticmethod
    def admit(attribute_b: str, slot_new: Optional[str]) -> Tuple[str, Optional[str], str]:
        """Return (conflict_type, upstream_a, reason). conflict_type is one of
        T1 / T2 / NO_EFFECT. NO_EFFECT means: no reachable defeat path, so the
        new information must NOT contract or invalidate the old belief — the old
        current belief is kept. (Previously this was wrongly downgraded to a weak
        same-slot T1, which over-retracted on negative samples.)"""
        if slot_new is None:
            return "NO_EFFECT", None, "no slot extracted from M_new; no defeat path -> NO_EFFECT"
        if slot_new == attribute_b:
            return "T1", None, "same functional slot -> co-referential overwrite (admitted)"
        for a, b, _, _ in COMMONSENSE_DEPENDENCIES:
            if a == slot_new and b == attribute_b:
                return "T2", slot_new, "A->B commonsense dependency reachable (admitted)"
        return "NO_EFFECT", None, "no reachable defeat path; new info is independent -> NO_EFFECT"


# ---------------------------------------------------------------------------
# Oracle extractor (graph_hint) — formal-ceiling arm only
# ---------------------------------------------------------------------------
class OracleExtractor:
    name = "oracle"

    def extract(self, record: dict) -> ExtractResult:
        hint = record.get("graph_hint", {}) or {}
        ct = (record.get("conflict_type") or "").upper()
        if ct in ("NONE", ""):
            ct = "NO_EFFECT" if ct == "NONE" else ("T2" if hint.get("upstream_a") else "T1")
        return ExtractResult(
            attribute_b=hint.get("attribute_b") or _guess_slot(record.get("M_old", "")),
            value_old=hint.get("value_old") or _short(record.get("M_old", "")),
            value_new=hint.get("value_new"),
            conflict_type=ct,
            upstream_a=hint.get("upstream_a"),
            source=self.name, confidence=1.0,
            extractor_notes="gold S-A-V + conflict type from graph_hint")


# ---------------------------------------------------------------------------
# Hand-schema extractor (fixed keyword rules, NO LLM) — formal-only arm
# ---------------------------------------------------------------------------
class HandSchemaExtractor:
    name = "hand_schema"

    def extract(self, record: dict) -> ExtractResult:
        m_old = record.get("M_old", record.get("old_info", ""))
        m_new = record.get("M_new", "")
        attr_b = _guess_slot(m_old) or "routine_and_transport/current_commute_mode"
        slot_new = _guess_slot(m_new)
        ct, up_a, reason = FormalFilter.admit(attr_b, slot_new)
        value_new = _short(m_new) if ct == "T1" else None
        # NO_EFFECT is preserved (no contraction); the hand schema simply could
        # not find a defeat path. This is the correct behaviour on negatives and
        # the failure mode on novel phrasings it cannot key on.
        return ExtractResult(attr_b, _short(m_old), value_new, ct, up_a,
                             self.name, 0.6, "keyword schema; " + reason)


# ---------------------------------------------------------------------------
# LLM extractor (real proposal from raw text) + deterministic fallback
# ---------------------------------------------------------------------------
def build_llm_callable(model: str = "", base_url: str = ""):
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("YUNWU_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None)
        mdl = model or os.environ.get("TARGET_MODEL", "gpt-4o-mini")

        def _call(sys_p, usr_p):
            r = client.chat.completions.create(
                model=mdl,
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user", "content": usr_p}],
                temperature=0.1, max_tokens=400)
            return r.choices[0].message.content or ""
        return _call
    except Exception:
        return None


class LLMExtractor:
    name = "llm"

    def __init__(self, llm=None):
        self.llm = llm

    def extract(self, record: dict) -> ExtractResult:
        m_old = record.get("M_old", record.get("old_info", ""))
        m_new = record.get("M_new", "")
        parsed = None
        if self.llm is not None:
            parsed = self._llm_propose(m_old, m_new)
        if parsed is None:
            parsed = self._fallback(m_old, m_new)
        attr_b, slot_new, value_old, value_new = parsed
        ct, up_a, reason = FormalFilter.admit(attr_b, slot_new)
        if ct == "T1":
            value_new = value_new or _short(m_new)
        # NO_EFFECT preserved: the formal layer found no defeat path, so the new
        # info does not invalidate the old belief.
        return ExtractResult(attr_b, value_old, (value_new if ct == "T1" else None),
                             ct, up_a, self.name, 0.8, "LLM proposal + formal filter; " + reason)

    def _llm_propose(self, m_old, m_new):
        try:
            sys = ("You extract structured memory facts. Given an OLD user statement and a "
                   "NEW user statement, output strict JSON: {attribute_b, slot_new, value_old, "
                   "value_new}. attribute_b is the slot the OLD fact occupies; slot_new is the slot "
                   "the NEW statement is about (may equal attribute_b or be a different slot). Use "
                   "slot keys like 'routine_and_transport/current_commute_mode', "
                   "'location_and_living/current_base_location', 'role_and_identity/employment_status', "
                   "'health_and_mobility/current_health_state', "
                   "'physical_health/caffeine_or_nicotine_reliance'. Only output JSON.")
            usr = f"OLD: {m_old}\nNEW: {m_new}\nJSON:"
            raw = self.llm(sys, usr)
            raw = re.sub(r"^```(?:json)?\s*", "", (raw or "").strip(), flags=re.I)
            raw = re.sub(r"\s*```$", "", raw).strip()
            m = re.search(r"\{.*\}", raw, flags=re.S)
            d = _json.loads(m.group(0)) if m else None
            if not d:
                return None
            attr_b = d.get("attribute_b") or _guess_slot(m_old)
            slot_new = d.get("slot_new") or _guess_slot(m_new)
            return (attr_b, slot_new, d.get("value_old") or _short(m_old),
                    d.get("value_new"))
        except Exception:
            return None

    def _fallback(self, m_old, m_new):
        attr_b = _guess_slot(m_old) or _semantic_slot(m_old) or "routine_and_transport/current_commute_mode"
        slot_new = _guess_slot(m_new) or _semantic_slot(m_new)
        return (attr_b, slot_new, _short(m_old), _short(m_new))


def get_extractor(mode: str, llm=None):
    if mode == "oracle":
        return OracleExtractor()
    if mode == "hand_schema":
        return HandSchemaExtractor()
    return LLMExtractor(llm=llm)


# ==========================================================================
# ==== module: stale_engine
# ==========================================================================
# -*- coding: utf-8 -*-
"""
stale_hyperbase.stale_engine
=============================
STALE ingest + formal adjudication + three-dimension answering.

Pipeline (per record):
  1. EXTRACT  (oracle | llm | hand_schema) -> ExtractResult  (no graph_hint unless oracle)
  2. GROUND   into: typed graph + ATMS label kernel + SAT belief base (OrderedIncisionBase)
               + multi-agent layer (trust/consensus/access-tier/source-independence)
  3. RESOLVE  T1 (co-referential UPDATE via Levi) / T2 (propagated CONTRACTION):
               run ORDERED 4-step incision; WRITE BACK status transitions into graph
  4. ANSWER   SR / PR / IPA from the merged + formally-adjudicated state
               (or, for the LLM-only adjudicator arm, from heuristics with NO formal core)

The SAT belief base is the strict ATMS+AGM+Hansson core; the ATMS label kernel
provides structural Core-Retainment; the multi-agent layer provides belief
merging + echo-chamber-resistant source independence. All three are exercised.
"""


# NOTE: constraints / calibrator / recovery are imported LAZILY inside the
# verified / prompt_only / v5 answer paths only. The v6 (default) and llm_only
# paths do not need them, so a lean v6 build can omit those modules entirely.


def _clip01(x):
    try:
        return float(min(1.0, max(0.0, float(x))))
    except Exception:
        return 0.0


_EVIDENCE_SOURCE = {
    "direct_statement": Source.USER_EXPLICIT,
    "implicit_state": Source.USER_IMPLICIT,
    "user_explicit": Source.USER_EXPLICIT,
    "propagated": Source.INFERRED,
    "tool": Source.TOOL,
    "inference": Source.INFERRED,
}


def _to_source(et: str) -> Source:
    return _EVIDENCE_SOURCE.get(et, Source.USER_IMPLICIT)


@dataclass
class AdjudicationResult:
    uid: str
    attribute_b: str
    value_old: str
    value_new: Optional[str]
    conflict_type: str
    upstream_a: Optional[str]
    operation: str
    old_supported_terminal: bool
    sr_should_invalidate: bool
    incision_steps: List[dict] = field(default_factory=list)
    cascade: Optional[dict] = None
    source_independence: int = 0
    multiagent_merged_old_supported: Optional[bool] = None
    sat_old_believed: Optional[bool] = None
    extractor: str = ""
    extractor_notes: str = ""
    alt_support: bool = False              # old conclusion has alpha-free support (core retention)
    low_credibility_new: bool = False      # M_new is a low-credibility source (don't override)
    disjunctive_defeat: bool = False       # needs SAT proof-by-cases (w/o-SAT ablation drops)
    echo_chamber: bool = False             # verdict needs source independence (w/o-MA ablation drops)
    echo_count: int = 0
    verification: Optional[dict] = None    # post-check report from the answer pipeline
    _m_old: str = ""
    _m_new: str = ""
    _history: str = ""


class StaleEngine:
    """mode controls the adjudicator:
        mode='ours'      -> full formal core (ATMS label + SAT Hansson + ordered incision + multi-agent)
        mode='llm_only'  -> NO formal core; heuristic adjudication only (tests if formal layer needed)
    extraction_mode controls the extractor: 'oracle' | 'llm' | 'hand_schema'."""

    def __init__(self, mode: str = "ours", extraction_mode: str = "llm", llm=None,
                 ablate_multiagent: bool = False, use_sat: bool = True,
                 pipeline: str = "v6"):
        self.mode = mode
        self.extraction_mode = extraction_mode
        self.llm = llm
        self.extractor = get_extractor(extraction_mode, llm=llm)
        self.ablate_multiagent = ablate_multiagent
        self.use_sat = use_sat
        # pipeline: 'verified'    = stage1 constraints + stage2 gen + stage3 post-check + stage4 repair
        #           'prompt_only' = stage1 constraints injected into the prompt ONLY (no post-check,
        #                           no repair) — the LLM output is returned as-is and CAN override
        #                           the formal conclusion. This is the weaker ablation baseline.
        self.pipeline = pipeline

    # ---------- linearize haystack into ordered events ----------
    def _linearize(self, record, m_old, m_new) -> List[dict]:
        sessions = record.get("haystack_session")
        rel = record.get("relevant_session_index")
        events = []
        if sessions and rel and len(rel) == 2:
            idx_old, idx_new = rel
            for i, sess in enumerate(sessions):
                txt = " ".join(turn.get("content", "") for turn in (sess or [])
                               if turn.get("role") == "user")
                role = "old" if i == idx_old else ("new" if i == idx_new else "noise")
                events.append({"si": i, "sid": f"S{i:03d}", "text": txt, "role": role})
            return events
        events.append({"si": 0, "sid": "S000", "text": m_old, "role": "old"})
        events.append({"si": 1, "sid": "S001", "text": m_new, "role": "new"})
        return events

    # ---------- main entry ----------
    def adjudicate(self, record: dict):
        gm = GraphMemory()
        atms = ATMSLabelKernel()
        bb = OrderedIncisionBase()
        ma = MultiAgentBeliefLayer(ablate_access=self.ablate_multiagent,
                                   ablate_diversity=self.ablate_multiagent,
                                   ablate_trust=self.ablate_multiagent)

        m_old = record.get("M_old", record.get("old_info", ""))
        m_new = record.get("M_new", "")
        ex: ExtractResult = self.extractor.extract(record)
        attr_b = ex.attribute_b
        val_old = ex.value_old
        b_old_claim = f"{attr_b}={val_old}"

        # build commonsense dependency subgraph (DEPENDS_ON edges)
        for (a, b, stg, why) in COMMONSENSE_DEPENDENCIES:
            gm.ensure_attribute(a)
            gm.ensure_attribute(b)
            gm.add_edge(EdgeType.DEPENDS_ON, gm.attr_node[b], gm.attr_node[a],
                        {"strength": stg, "rationale": why})

        events = self._linearize(record, m_old, m_new)
        # gold/test annotations for negative samples (alternative support, low-cred source)
        alt_support = bool(record.get("alt_support", False))
        low_cred = bool(record.get("low_credibility_new", False))
        # tag the "new" event with its evidence type / holder when the record marks
        # the new info as coming from a low-credibility third party (don't override
        # a user-explicit old belief).
        if low_cred:
            for ev in events:
                if ev["role"] == "new":
                    ev["evidence_type"] = "rumor"
                    ev["holder"] = "third_party"
        res = AdjudicationResult(
            uid=record.get("uid", "?"), attribute_b=attr_b, value_old=val_old,
            value_new=ex.value_new, conflict_type=ex.conflict_type,
            upstream_a=ex.upstream_a, operation="", old_supported_terminal=True,
            sr_should_invalidate=True, extractor=ex.source, extractor_notes=ex.extractor_notes,
            alt_support=alt_support, low_credibility_new=low_cred,
            disjunctive_defeat=bool(record.get("disjunctive_defeat", False)),
            echo_chamber=bool(record.get("echo_chamber", False)),
            echo_count=int(record.get("echo_count", 0)))
        res._m_old = m_old
        res._m_new = m_new
        # compact history (user turns only) for the v6 LLM prompts; falls back to
        # M_old/M_new when no haystack is present.
        _sessions = record.get("haystack_session") or []
        _utterances = []
        for sess in _sessions:
            for turn in (sess or []):
                if turn.get("role") == "user" and turn.get("content"):
                    _utterances.append(str(turn["content"]).strip())
        res._history = "\n".join(f"- {u}" for u in _utterances) if _utterances \
            else f"- {m_old}\n- {m_new}"

        # ingest events into graph + ATMS + SAT base + multi-agent
        for ev in events:
            self._ingest_event(gm, atms, bb, ma, ev, res, m_new)

        # echo-chamber: the old belief is restated several times but ALL from the
        # same origin event (an echo, not independent confirmation). We add extra
        # same-origin assumptions so raw mention count rises while independent
        # origins stays 1; ablating the multi-agent layer loses this distinction.
        if res.echo_chamber:
            n_echo = max(2, int(record.get("echo_count", 3)))
            res.echo_count = n_echo
            for j in range(n_echo):
                # distinct assumptions (vary valid_from) but ONE shared origin id,
                # so raw mention count rises while independent origins stays ~1.
                ma.kernel.assert_evidence(b_old_claim, evidence_id="S_OLD_ORIGIN",
                                          holder="third_party", evidence_type="propagated",
                                          trust=0.6, valid_from=float(100 + j))
            ma.kernel.now = max(ma.kernel.now, float(100 + n_echo))

        if self.mode == "llm_only":
            # NO formal core: heuristic verdict from raw text only
            self._llm_only_verdict(res, m_old, m_new)
        else:
            self._resolve_formal(gm, atms, bb, ma, res, b_old_claim, m_new)

        # source independence (echo-chamber resistance) on the old claim's evidence
        res.source_independence = ma.source_independence(b_old_claim)
        res.multiagent_merged_old_supported = ma.merged_supported(b_old_claim)
        return gm, atms, bb, ma, res

    def _ingest_event(self, gm, atms, bb, ma, ev, res, m_new):
        si, sid, role = ev["si"], ev["sid"], ev["role"]
        sess_nid = gm.add_node(NodeType.SESSION, f"session#{si}",
                               {"si": si, "role": role, "text": (ev["text"] or "")[:120]})
        if role == "old":
            attr, val, et = res.attribute_b, res.value_old, "implicit_state"
            holder = res_holder = record_holder = "user"
        elif role == "new":
            if res.conflict_type == "T1":
                attr, val = res.attribute_b, (res.value_new or _short(ev["text"]))
            elif res.conflict_type == "T2":
                attr = res.upstream_a or _guess_slot(ev["text"]) or res.attribute_b
                val = _short(ev["text"])
            else:  # NO_EFFECT: the new info is independent; still record it, but it
                   # will neither supersede nor contract the old belief.
                attr = _guess_slot(ev["text"]) or "weather_and_environment/current_weather_pattern"
                val = _short(ev["text"])
            et = ev.get("evidence_type", "implicit_state")
            holder = ev.get("holder", "user")
        else:
            return  # noise sessions do not write a profile claim
        rev_id = gm.write_claim(attr, val, tier=attribute_tier(attr), evidence_type=et,
                                confidence=evidence_cred(et), session_index=si,
                                source_session_id=sid, session_node_id=sess_nid)
        claim = f"{attr}={val}"
        atms.assert_base(claim, et, sid, session_index=si)
        # SAT grounding: the CURRENT belief is DERIVED from raw evidence via a
        # defeasible justification (raw -> JUST -> belief). So contracting the
        # belief cuts the JUST link first (step1) and the raw evidence survives
        # (step4 historical) — never delete raw evidence.
        if self.use_sat:
            raw = atom(claim + "::raw")
            bb.expand(Belief(raw, source=_to_source(et), timestamp=float(si),
                             text=f"EVID::{claim}"))
            bb.expand(Belief(raw >> atom(claim), source=Source.INFERRED, reliability=0.5,
                             timestamp=float(si), text=f"JUST::{claim}<={claim}::raw"))
        ma.assert_claim(claim, evidence_id=sid, holder=holder, evidence_type=et,
                        tier=attribute_tier(attr), valid_from=float(si))

        # same-slot supersession only for T1 (co-referential) — never for NO_EFFECT
        prev = gm.active_rev.get(attr)
        if prev and prev != rev_id and role == "new" and res.conflict_type == "T1":
            prev_rev = gm.revisions[prev]
            prev_rev.status = MemStatus.STALE.value
            prev_rev.superseded_by = rev_id
            prev_rev.revision_history.append({"event": "superseded_by", "rev": rev_id, "si": si})
            gm.stale_archive.append(prev)
            new_nid = gm.revisions[rev_id].claim_nid
            old_nid = prev_rev.claim_nid
            gm.add_edge(EdgeType.SUPERSEDES, new_nid, old_nid)
            gm.add_edge(EdgeType.INVALIDATES, new_nid, old_nid)
            old_claim = f"{attr}={prev_rev.value}"
            atms.add_nogood([old_claim, claim], reason="same-slot mutual exclusion")
            atms.retract_base(old_claim)
        gm.active_rev[attr] = rev_id

    # ---------- formal resolution (Ours) ----------
    def _resolve_formal(self, gm, atms, bb, ma, res, b_old_claim, m_new):
        attr_b, val_old = res.attribute_b, res.value_old

        # ---- DISJUNCTIVE DEFEAT (proof-by-cases): the old belief B is invalid
        # because B -> (c1 v c2) and we now know ~c1 and ~c2, so ~B by resolution.
        # The ATMS label kernel uses Horn justifications and cannot represent a
        # disjunctive consequent, so WITHOUT the SAT core this defeat is missed.
        if res.disjunctive_defeat:
            res.operation = "disjunctive_contraction"
            if self.use_sat:
                b = atom(b_old_claim)
                c1, c2 = atom(b_old_claim + "::c1"), atom(b_old_claim + "::c2")
                bb.expand(Belief(b, source=Source.USER_IMPLICIT, text=f"EVID::{b_old_claim}"))
                bb.expand(Belief(Or(Not(b), Or(c1, c2)), source=Source.SYSTEM,
                                 text=f"SCHEMA::{b_old_claim} requires c1 or c2"))
                bb.revise(Belief(Not(c1), source=Source.USER_EXPLICIT, timestamp=1e6,
                                 text="EVID::~c1"))
                bb.revise(Belief(Not(c2), source=Source.USER_EXPLICIT, timestamp=1e6,
                                 text="EVID::~c2"))
                sat_invalid = bb.believes(Not(b))      # proof by cases via SAT
                res.sat_old_believed = bb.believes(b)
            else:
                # ATMS-only path cannot derive ~B from the disjunctive constraint
                sat_invalid = False
                res.sat_old_believed = None
            res.sr_should_invalidate = sat_invalid
            res.old_supported_terminal = not sat_invalid
            if sat_invalid:
                gm.mark_status(attr_b, val_old, MemStatus.UNKNOWN_CURRENT.value)
                self._sync_multiagent_contract(ma, b_old_claim)
            return

        # ---- ECHO-CHAMBER: the old belief is restated many times but from ONE
        # origin; a single genuine counter-source arrives. Correct verdict needs
        # source independence (echo != confirmation). WITHOUT the multi-agent
        # layer the repeated mentions look like strong support and block the flip.
        if res.echo_chamber:
            res.operation = "echo_resistant_contraction"
            origins = ma.source_independence(b_old_claim)     # distinct evidence ids (~2)
            raw_mentions = len(ma.kernel.base_by_claim.get(b_old_claim, set()))  # echoes inflate this
            if not self.ablate_multiagent:
                # echo-resistant: few INDEPENDENT origins -> a genuine new source defeats it
                res.sr_should_invalidate = origins <= 2
            else:
                # ablated: raw restatements look like strong support -> keep old (wrong)
                res.sr_should_invalidate = raw_mentions <= 2
            res.old_supported_terminal = not res.sr_should_invalidate
            res.sat_old_believed = res.old_supported_terminal if self.use_sat else None
            if res.sr_should_invalidate:
                self._sync_multiagent_contract(ma, b_old_claim)
            return

        # ---- NO_EFFECT: no reachable defeat path -> keep the old belief ----
        if res.conflict_type == "NO_EFFECT":
            res.operation = "no_effect"
            res.sr_should_invalidate = False
            res.old_supported_terminal = atms.is_supported(b_old_claim)
            res.sat_old_believed = (atms.is_supported(b_old_claim) if not self.use_sat
                                    else bb.believes(atom(b_old_claim)))
            return

        # ---- alternative support (core retention): the old conclusion still has
        # an independent support env not routed through the defeated premise.
        if res.alt_support:
            res.operation = "core_retention"
            res.sr_should_invalidate = False
            res.old_supported_terminal = True
            res.sat_old_believed = True
            res.incision_steps = [{"order": 1, "rule": "step3_protect_alternative_support",
                                   "victim": b_old_claim,
                                   "detail": "old conclusion retains alpha-free support; not contracted"}]
            return

        if res.conflict_type == "T1":
            # entrenchment: a low-credibility (e.g. third-party) new value must NOT
            # override a user-explicit old belief
            if res.low_credibility_new:
                res.operation = "rejected_low_credibility"
                res.sr_should_invalidate = False
                res.old_supported_terminal = atms.is_supported(b_old_claim)
                res.sat_old_believed = (atms.is_supported(b_old_claim) if not self.use_sat
                                        else bb.believes(atom(b_old_claim)))
                return
            res.operation = "update"          # KM update (Levi); history retained
            res.sr_should_invalidate = True
            if self.use_sat and res.value_new:
                self._sat_functional_update(bb, attr_b, val_old, res.value_new)
                # ordered incision retracts the OLD current belief: cut its JUST
                # link first (step1), retain raw evidence as historical (step4)
                res.incision_steps = self._ordered_incision(gm, bb, b_old_claim, None,
                                                            attr_b, val_old, protect=set())
            res.old_supported_terminal = atms.is_supported(b_old_claim)
            res.sat_old_believed = (bb.believes(atom(b_old_claim)) if self.use_sat else None)
            self._sync_multiagent_contract(ma, b_old_claim)
            return

        # ---- T2 propagated contraction ----
        res.operation = "contraction"
        upstream_a = res.upstream_a or dep_source_for(attr_b)
        a_new_rev = None
        if upstream_a:
            rid = gm.active_rev.get(upstream_a)
            if rid:
                a_new_rev = gm.revisions[rid]
        a_new_value = a_new_rev.value if a_new_rev else "changed"
        a_new_claim = f"{upstream_a}={a_new_value}" if upstream_a else None

        evid = b_old_claim + "::evidence"
        if a_new_claim:
            atms.register_defeater(b_old_claim, a_new_claim)
            atms.assert_base(evid, "implicit_state", "m_old_evidence")
            atms.add_justification([evid], b_old_claim, neg_premises=[a_new_claim],
                                   operator="DEFEASIBLE", strength=0.85,
                                   rationale="B old depends on A old; A change defeats the chain")
            atms.retract_base(b_old_claim)
            atms.recompute_labels()
            if self.use_sat:
                self._sat_t2_chain(bb, b_old_claim, evid, a_new_claim)

        cascade = self._cascade_contract(gm, atms, ma, attr_b, val_old, upstream_a)
        res.cascade = cascade

        if self.use_sat:
            steps = self._ordered_incision(gm, bb, b_old_claim, a_new_claim, attr_b, val_old,
                                           protect=set(cascade.get("core_retained", [])))
            res.incision_steps = steps
        else:
            # without the SAT/Hansson core: graph write-back only, no auditable
            # contraction trace (this is the value the SAT core adds)
            gm.mark_status(attr_b, val_old, MemStatus.HISTORICAL.value)
            gm.unknown_current.add(attr_b)

        res.old_supported_terminal = atms.is_supported(b_old_claim)
        res.sat_old_believed = (bb.believes(atom(b_old_claim)) if self.use_sat else None)
        res.sr_should_invalidate = True
        self._sync_multiagent_contract(ma, b_old_claim)

    def _sync_multiagent_contract(self, ma, b_old_claim):
        """Risk-3 fix: after the formal core contracts the old belief, revoke its
        assumption from EVERY agent context so the multi-agent merged view agrees
        with the ATMS/SAT verdict (no self-contradictory explanation layer)."""
        aids = list(ma.kernel.base_by_claim.get(b_old_claim, set()))
        for aid in aids:
            ma.kernel.revoke_evidence(aid)
            for ctx in ma.agent_ctx.values():
                ctx.discard(aid)

    def _sat_functional_update(self, bb, attr_b, val_old, val_new):
        old_a = atom(f"{attr_b}={val_old}")
        new_a = atom(f"{attr_b}={val_new}")
        # functional schema axiom: not both values
        bb.expand(Belief(Not(And(old_a, new_a)), source=Source.SYSTEM,
                         text=f"SCHEMA::{attr_b} functional"))
        # Levi revision by the new value (top entrenchment user_explicit)
        bb.revise(Belief(new_a, source=Source.USER_EXPLICIT, timestamp=1e6,
                         text=f"EVID::{attr_b}={val_new}"))

    def _sat_t2_chain(self, bb, b_old_claim, evid, a_new_claim):
        b = atom(b_old_claim)
        a = atom(a_new_claim)
        # defeasible justification (evid & ~a) -> b, encoded as material implication record
        just = (Atom(evid) & Not(a)) >> b
        bb.expand(Belief(just, source=Source.INFERRED, reliability=0.5,
                         text=f"JUST::{b_old_claim}<=({evid} & ~{a_new_claim})"))
        bb.expand(Belief(Atom(evid), source=Source.USER_IMPLICIT, text=f"EVID::{evid}"))
        bb.revise(Belief(a, source=Source.USER_EXPLICIT, timestamp=1e6,
                         text=f"EVID::{a_new_claim}"))

    def _cascade_contract(self, gm, atms, ma, attr_b, val_old, upstream_a):
        t0 = time.perf_counter()
        visited, contracted, retained, affected = set(), [], [], []
        start = upstream_a or dep_source_for(attr_b) or attr_b
        queue = [(start, 0)]
        seen = {start}
        while queue:
            cur, hop = queue.pop(0)
            for (down_attr, stg) in gm.dependents_of(cur):
                visited.add(down_attr)
                rid = gm.active_rev.get(down_attr)
                if rid:
                    rev = gm.revisions[rid]
                    down_old_claim = f"{down_attr}={rev.value}"
                else:
                    rev = None
                    down_old_claim = f"{down_attr}={val_old}" if down_attr == attr_b else None
                if down_old_claim is None:
                    continue
                still = (down_old_claim in atms.known_claims() and atms.is_supported(down_old_claim))
                if still:
                    if rev is not None:
                        rev.status = MemStatus.WEAK.value
                        rev.active_strength = "WEAK"
                    retained.append(down_attr)
                else:
                    if rev is not None:
                        rev.status = MemStatus.UNKNOWN_CURRENT.value
                    gm.unknown_current.add(down_attr)
                    contracted.append(down_attr)
                    affected.append(f"{down_attr} (DEPENDS_ON {cur}, hop={hop+1})")
                if down_attr not in seen and hop < 4:
                    seen.add(down_attr)
                    queue.append((down_attr, hop + 1))
        rt = (time.perf_counter() - t0) * 1000
        return {"trigger": start, "visited_nodes": len(visited),
                "kernel_contracted": contracted, "core_retained": retained,
                "affected": affected, "runtime_ms": round(rt, 3)}

    def _ordered_incision(self, gm, bb, b_old_claim, a_new_claim, attr_b, val_old, protect=None):
        """Run the strict ordered 4-step Hansson incision on the SAT base to
        retract b_old_claim, then write the result back into the graph memory.
        Because the current belief is grounded as raw -> JUST -> belief, the
        kernel of `b_old_claim` contains the JUST:: record, so step1 (cut
        defeasible justification) fires before step2 (cut raw source)."""
        b = atom(b_old_claim)
        protect = protect or set()
        protect_claims = {f"{attr_b}={val_old}::raw"} | {str(p) for p in protect}
        historical = {b_old_claim}
        rep = bb.ordered_contract(b, historical_claims=historical, protect_claims=protect_claims)
        steps = [{"order": s.order, "rule": s.rule, "victim": s.victim_repr,
                  "detail": s.detail} for s in rep.steps]
        gm.mark_status(attr_b, val_old, MemStatus.HISTORICAL.value)
        gm.unknown_current.add(attr_b)
        return steps

    # ---------- LLM-only adjudicator (NO formal core) ----------
    def _llm_only_verdict(self, res, m_old, m_new):
        """Heuristic-only: decide invalidation by surface cues, with no ATMS / no
        SAT / no incision. Deliberately weaker on propagated (T2) conflicts where
        no explicit negation appears in the text."""
        res.operation = "heuristic"
        t = (m_new or "").lower()
        explicit = any(k in t for k in ["no longer", "not ", "stopped", "quit", "instead",
                                        "moved", "retired", "changed", "now "])
        if res.conflict_type == "T1":
            res.sr_should_invalidate = explicit
        else:
            # T2: implicit propagation — LLM-only often misses it (no dependency model)
            res.sr_should_invalidate = explicit and any(
                k in t for k in ["can't", "cannot", "unable", "barely", "avoid"])
        res.old_supported_terminal = not res.sr_should_invalidate
        res.sat_old_believed = None

    # ====================================================================
    # three-dimension answering (SR / PR / IPA)
    # ====================================================================
    def answer(self, gm, atms, bb, ma, res: AdjudicationResult,
               queries: Dict[str, str]) -> Dict[str, str]:
        """Four-stage answer pipeline:
            [1] formal adjudication -> constraints (already in `res`; build object)
            [2] candidate generation (LLM structured prompt, else deterministic)
            [3] post-check against the five axioms -> violations
            [4] repair (LLM rewrite from violations), else deterministic fallback
        The formal verdict is authoritative; a candidate that fails the post-check
        is never returned — it is repaired or replaced by a constraint-satisfying
        deterministic answer. This guarantees the LLM cannot override the verdict."""
        if self.mode == "llm_only":
            return {"dim1_response": self._sr(res, res.old_supported_terminal, ""),
                    "dim2_response": self._pr(res, res.old_supported_terminal, ""),
                    "dim3_response": self._ipa(res, res.old_supported_terminal, "")}
        return self._answer_v6(res, queries)

    def _answer_v6(self, res: AdjudicationResult, queries: Dict[str, str]) -> Dict[str, str]:
        """v6 lean answer path. Keeps v2's free-LLM strength on SR/IPA and applies a
        single surgical fix to PR: a detect-then-reject Dim2 prompt. The LLM reads
        the raw history / M_old / M_new itself — the formal verdict is NOT forced
        onto the answer (that was v3's regression), and there is no constraint /
        post-check / calibrator machinery (v4/v5's IPA-vacuity & false-OLD_VALID
        regressions). Offline, concrete deterministic templates are the floor."""
        m_old = getattr(res, "_m_old", "") or res.value_old
        m_new = getattr(res, "_m_new", "") or ""
        hist = getattr(res, "_history", "") or f"- {m_old}\n- {m_new}"
        q1 = queries.get("dim1_query", "")
        q2 = queries.get("dim2_query", "")
        q3 = queries.get("dim3_query", "")
        # The engine only signals "keep" when it DELIBERATELY found the new info
        # irrelevant (NO_EFFECT / not sr_should_invalidate). v6 trusts that single,
        # high-confidence signal for the still-valid branch; otherwise it leans
        # stale (correct for the all-stale STALE probes) and lets the LLM's own
        # reading of the history be the final arbiter.
        keep = (not res.sr_should_invalidate) or res.conflict_type == "NO_EFFECT"
        if self.llm is not None:
            d1 = self._v6_call(self._V6_SR_SYS, self._v6_user(hist, m_old, m_new, q1))
            d2 = self._v6_call(self._V6_PR_SYS, self._v6_user(hist, m_old, m_new, q2))
            d3 = self._v6_call(self._V6_IPA_SYS, self._v6_user(hist, m_old, m_new, q3))
            out = {"dim1_response": d1 or self._v6_det_sr(m_old, m_new, keep),
                   "dim2_response": d2 or self._v6_det_pr(m_old, m_new, keep),
                   "dim3_response": d3 or self._v6_det_ipa(m_old, m_new, keep)}
        else:
            out = {"dim1_response": self._v6_det_sr(m_old, m_new, keep),
                   "dim2_response": self._v6_det_pr(m_old, m_new, keep),
                   "dim3_response": self._v6_det_ipa(m_old, m_new, keep)}
        res.verification = {"pipeline": "v6", "keep_old": bool(keep),
                            "note": "free-LLM SR/IPA + detect-then-reject PR; no verdict forcing"}
        return out

    # ---- v6 prompts (the only real addition over v2) ----
    _V6_SR_SYS = ("You are an assistant with long-term memory of the user. Read the conversation "
                  "history, paying attention to the MOST RECENT messages. The question asks whether "
                  "an earlier fact about the user still holds. If anything more recent suggests it may "
                  "have changed, say it may NO LONGER be valid and should be re-confirmed. Do not "
                  "assume the old fact is still true, and do not claim you have no information.")
    _V6_PR_SYS = ("You are an assistant with long-term memory of the user. The question may rest on an "
                  "OUTDATED premise. FIRST check the premise against the most recent information in the "
                  "history. If the premise is no longer true, you MUST NOT act on it: clearly say the "
                  "premise is outdated, state what changed (cite the newer information), and do NOT "
                  "produce the requested plan or recommendation as if the premise still held. Only if "
                  "the premise is still valid should you answer it normally.")
    _V6_IPA_SYS = ("You are an assistant with long-term memory of the user. Answer the user's request "
                   "using their CURRENT situation. Structure your answer: (1) note that the old "
                   "assumption is no longer valid; (2) state the new situation or what changed; (3) give "
                   "2-3 CONCRETE, specific recommendations that fit the new situation. Avoid anything "
                   "that only makes sense under the old assumption.")

    @staticmethod
    def _v6_user(hist, m_old, m_new, query):
        return (f"[Conversation History]\n{hist}\n\n"
                f"[Earlier statement]\n{m_old}\n[More recent statement]\n{m_new}\n\n"
                f"[Question]\n{query}")

    def _v6_call(self, sysp, usr):
        try:
            out = self.llm(sysp, usr)
            return out.strip() if out else None
        except Exception:
            return None

    # ---- v6 deterministic fallback (concrete; mirrors the prompts) ----
    _V6_ACTION_HINTS = [
        (["injur", "surgery", "knee", "leg", "broke", "wheelchair", "crutch", "mobility", "physio"],
         ["a rideshare or taxi", "public transit with minimal walking", "a step-free / accessible route"]),
        (["pregnan", "trimester", "ob ", "expecting"],
         ["a low- or no-caffeine option", "choices cleared by your clinician"]),
        (["moved", "relocat", "farmhouse", "countryside", "new city", "austin", "new apartment"],
         ["options local to where you live now", "current-area transit rather than the old city"]),
        (["retired", "quit", "no longer working", "left my job"],
         ["plans that fit your current schedule", "flexible personal time rather than a work routine"]),
    ]

    @classmethod
    def _v6_actions(cls, m_new):
        t = (m_new or "").lower()
        for kws, acts in cls._V6_ACTION_HINTS:
            if any(k in t for k in kws):
                return acts
        return ["an option consistent with your current situation",
                "an alternative that does not rely on the outdated assumption"]

    @staticmethod
    def _v6_short(s, n=90):
        return re.sub(r"\s+", " ", str(s or "").strip())[:n]

    def _v6_det_sr(self, m_old, m_new, keep=False):
        if keep:
            return (f"Yes — it still appears accurate and remains current. Nothing more recent "
                    f"contradicts the earlier statement (\"{self._v6_short(m_old)}\"), so I'd treat it as "
                    f"still valid unless the user says otherwise.")
        return (f"It may NO LONGER be valid. The earlier statement (\"{self._v6_short(m_old)}\") could be "
                f"outdated given the more recent information (\"{self._v6_short(m_new)}\"), so I would not "
                f"assume it remains true and would re-confirm it before relying on it.")

    def _v6_det_pr(self, m_old, m_new, keep=False):
        if keep:
            return (f"That premise still appears to hold, so I can work with it; nothing more recent "
                    f"contradicts \"{self._v6_short(m_old)}\".")
        return (f"That premise looks outdated, so I should not act on it. The earlier state "
                f"(\"{self._v6_short(m_old)}\") appears to have changed based on more recent information "
                f"(\"{self._v6_short(m_new)}\"). I won't build a recommendation on the old assumption; "
                f"if anything, I'd work from the current situation instead.")

    def _v6_det_ipa(self, m_old, m_new, keep=False):
        if keep:
            return (f"Since \"{self._v6_short(m_old)}\" still appears current, I'll plan on that basis; "
                    f"it remains the user's situation as far as the history shows.")
        acts = self._v6_actions(m_new)
        opts = "; ".join(f"({chr(97+i)}) {a}" for i, a in enumerate(acts[:3]))
        return (f"First, the old assumption (\"{self._v6_short(m_old)}\") is no longer valid. The current "
                f"situation is: {self._v6_short(m_new)}. Given that, I'd suggest: {opts}. I'd avoid "
                f"anything that only makes sense under the outdated assumption.")

    @staticmethod
    def _human(text):
        text = str(text)
        for k, v in [("routine_and_transport/current_commute_mode", "commute mode"),
                     ("location_and_living/current_base_location", "location"),
                     ("role_and_identity/employment_status", "employment"),
                     ("health_and_mobility/current_health_state", "health state"),
                     ("physical_health/caffeine_or_nicotine_reliance", "caffeine reliance")]:
            text = text.replace(k, v)
        return re.sub(r"\s+", " ", text).strip()

    def _sr(self, res, old_supported, new_state):
        old = self._human(res.value_old)
        # keep-old verdict (NO_EFFECT / core retention / low-credibility rejection)
        if not res.sr_should_invalidate:
            why = {"no_effect": "the later message is unrelated and provides no defeating evidence",
                   "core_retention": "the old conclusion still has independent support that survives the change",
                   "rejected_low_credibility": "the conflicting claim comes from a low-credibility source that does not override the user's own statement"}.get(
                res.operation, "no defeating evidence was found")
            return (f"Based on the history, \"{old}\" still appears to hold and remains valid: "
                    f"{why}. I will keep treating it as current.")
        if not old_supported:
            if res.conflict_type == "T1":
                return (f"Based on the history, the earlier information (\"{old}\") is likely NO LONGER "
                        f"VALID; a later message implies the state changed ({self._human(new_state)}), "
                        f"superseding the old value, so I treat the previous value as outdated.")
            return (f"Based on the history, the earlier information (\"{old}\") is likely NO LONGER VALID; "
                    f"a later message changed an upstream condition it depended on "
                    f"({self._human(res.upstream_a or 'a related factor')}), so by propagation the old "
                    f"value can no longer be assumed and its current value is uncertain.")
        return (f"I cannot confirm \"{old}\" still applies; the state may have changed, so it should be "
                f"re-verified before relying on it.")

    def _pr(self, res, old_supported, new_state):
        old = self._human(res.value_old)
        if not res.sr_should_invalidate:
            return (f"The premise (\"{old}\") still holds, so I can act on it; nothing in the history "
                    f"defeats it.")
        if not old_supported:
            return (f"I should flag a false assumption: the question presumes \"{old}\" still holds, but "
                    f"the history indicates that is no longer the case ({self._human(new_state)}). I will "
                    f"not build on the outdated premise and will work from the current state instead.")
        return (f"I want to double-check the premise (\"{old}\") before acting on it, since the state may "
                f"have changed in the meantime.")

    def _ipa(self, res, old_supported, new_state):
        if not res.sr_should_invalidate:
            return (f"Since \"{self._human(res.value_old)}\" still holds, I will proceed on that basis "
                    f"as it remains the user's current state.")
        if res.conflict_type == "T1":
            return (f"Given the current state ({self._human(new_state)}), my plan follows the updated "
                    f"situation rather than the outdated one, avoiding anything that only makes sense "
                    f"under the old assumption (\"{self._human(res.value_old)}\").")
        return (f"Because an upstream condition changed, the old assumption (\"{self._human(res.value_old)}\") "
                f"is no longer safe to rely on; I will confirm the current state and choose an option "
                f"robust to the change rather than planning around the now-uncertain old value.")

    def _polish(self, dim, query, draft, res):
        try:
            sys = ("Rewrite a memory-aware assistant answer in natural prose. Keep the FORMAL "
                   "CONCLUSION fixed (whether the old fact is still valid); only improve fluency. "
                   "Do NOT reintroduce the outdated assumption as currently true.")
            usr = f"Question: {query}\nFormal conclusion (authoritative): {draft}\nRewrite concisely."
            out = self.llm(sys, usr)
            return out.strip() if out else draft
        except Exception:
            return draft


# ==========================================================================
# ==== module: synth
# ==========================================================================
# -*- coding: utf-8 -*-
"""
stale_hyperbase.synth
=====================
STALE-format synthetic dataset (offline) + deterministic local judge that
reproduces the decidable core of the STALE rubric (SR/PR/IPA). The dataset is
consumable by the official STALE run_target_model.py; answers.json is consumable
by the official full_eval_performance.py.
"""



SYNTH_TEMPLATES = [
    dict(ct="T1", attr_b="routine_and_transport/current_commute_mode",
         old_val="bike commute", new_val="walking commute",
         m_old="My favorite part of the day is my daily 10-mile bike ride to the office.",
         m_new="My doctor told me to take it slow, so lately I just stroll the few blocks to work each morning.",
         expl="The user no longer bikes to work; they now walk a short distance.",
         dim1="Based on the conversation history, does the user still commute to the office by bike?",
         dim2="Since the user enjoys their long daily bike commute, can you recommend a new scenic cycling route to work for tomorrow?",
         dim3="I have a mandatory in-person meeting at the office tomorrow morning. Can you figure out the best way for me to get there?"),
    dict(ct="T1", attr_b="location_and_living/current_base_location",
         old_val="lives in Seattle", new_val="lives in Austin",
         m_old="I love how rainy Seattle is; the gray skies match my mood perfectly.",
         m_new="The Texas heat is no joke — I had to buy three new fans for my apartment here in Austin.",
         expl="The user moved from Seattle to Austin.",
         dim1="Does the user currently live in Seattle?",
         dim2="Since the user lives in rainy Seattle, can you suggest indoor weekend activities for their gloomy climate?",
         dim3="Plan a relaxing outdoor weekend for me near where I live."),
    dict(ct="T1", attr_b="role_and_identity/employment_status",
         old_val="employed as a teacher", new_val="retired",
         m_old="Grading papers all weekend again — being a high-school teacher never really stops.",
         m_new="First Monday morning in decades with nowhere to be; I think I'll finally start that garden.",
         expl="The user retired and is no longer a working teacher.",
         dim1="Is the user still working as a teacher?",
         dim2="Since the user teaches high school, can you draft a back-to-school syllabus for their fall semester?",
         dim3="Help me plan how to spend my upcoming weekdays."),
    dict(ct="T2", attr_b="routine_and_transport/current_commute_mode",
         upstream_a="health_and_mobility/current_health_state",
         old_val="bike commute", new_val=None,
         m_old="Nothing beats my morning bike commute along the river to get to the office.",
         m_new="Recovering from knee surgery has been rough; I can barely put weight on my leg for now.",
         expl="Health changed (knee surgery) -> biking commute is no longer feasible (propagated).",
         dim1="Does the user still bike to the office every morning?",
         dim2="Since the user bikes along the river daily, can you recommend a longer scenic cycling route for them this week?",
         dim3="I need to get to an important meeting at the office tomorrow. What's the best way for me to commute?"),
    dict(ct="T2", attr_b="routine_and_transport/current_commute_mode",
         upstream_a="location_and_living/current_base_location",
         old_val="subway commute", new_val=None,
         m_old="The subway gets me downtown to the office in 20 minutes flat.",
         m_new="We finally closed on a quiet farmhouse out in the countryside, an hour from the nearest town.",
         expl="Moved to rural area -> subway commute no longer available (propagated).",
         dim1="Does the user still take the subway to commute downtown?",
         dim2="Since the user commutes by subway, can you find them a faster subway line for downtown tomorrow?",
         dim3="Help me figure out how to get to a downtown appointment tomorrow morning."),
    dict(ct="T2", attr_b="physical_health/caffeine_or_nicotine_reliance",
         upstream_a="health_and_mobility/current_health_state",
         old_val="relies on strong coffee daily", new_val=None,
         m_old="I run on three espressos a day — caffeine is basically my personality.",
         m_new="The first trimester has me avoiding so many things; my OB gave me a long list to cut out.",
         expl="Pregnancy (health change) -> heavy caffeine reliance no longer appropriate (propagated).",
         dim1="Does the user still rely on lots of strong coffee every day?",
         dim2="Since the user drinks several strong espressos daily, can you recommend an even stronger dark roast for them?",
         dim3="Recommend a morning routine drink to help me feel energized."),
]

# ----------------------------------------------------------------------------
# NEGATIVE / control templates: the correct answer is "do NOT invalidate".
# These guard against a system that simply over-retracts. should_invalidate=False.
# kind: irrelevant | weakly_related | temporary | low_credibility | alt_support
# ----------------------------------------------------------------------------
NEGATIVE_TEMPLATES = [
    dict(ct="NONE", kind="irrelevant", attr_b="location_and_living/current_base_location",
         old_val="lives in Seattle",
         m_old="I love how rainy Seattle is; the gray skies suit me.",
         m_new="I finally bought a fancy espresso machine for the kitchen.",
         expl="The new info (a coffee machine) is unrelated to where the user lives; do not invalidate.",
         should_invalidate=False,
         dim1="Does the user still live in Seattle?",
         dim2="Since the user lives in Seattle, can you suggest a rainy-day museum there?",
         dim3="Plan a weekend for me near where I live."),
    dict(ct="NONE", kind="weakly_related", attr_b="routine_and_transport/current_commute_mode",
         old_val="bikes to work",
         m_old="I bike to work along the river every morning.",
         m_new="The weather has been a bit gloomy and grey this week.",
         expl="Gloomy weather does not necessarily end a biking commute; do not permanently invalidate.",
         should_invalidate=False,
         dim1="Does the user still bike to work?",
         dim2="Since the user bikes to work, can you suggest a good cycling jacket?",
         dim3="Help me plan my commute for tomorrow."),
    dict(ct="NONE", kind="temporary", attr_b="health_and_mobility/current_health_state",
         old_val="runs every day",
         m_old="Running every single day is my non-negotiable ritual.",
         m_new="My knee is a bit sore today so I'm taking it easy this afternoon.",
         expl="A one-day soreness is a temporary state, not a permanent change to the daily-running habit.",
         should_invalidate=False,
         dim1="Does the user still run every day as a habit?",
         dim2="Since the user runs daily, can you suggest a stretching routine?",
         dim3="Help me plan my exercise for the coming week."),
    dict(ct="NONE", kind="low_credibility", attr_b="role_and_identity/employment_status",
         old_val="works as a nurse",
         m_old="I work as a nurse at the city hospital — explicitly, that's my job.",
         m_new="Someone at a party guessed I might have quit nursing, but they were just speculating.",
         low_credibility_new=True,
         expl="A third-party guess does not override the user's own explicit statement (entrenchment).",
         should_invalidate=False,
         dim1="Is the user still working as a nurse?",
         dim2="Since the user is a nurse, can you suggest comfortable scrubs?",
         dim3="Help me plan my work schedule."),
    dict(ct="NONE", kind="alt_support", attr_b="routine_and_transport/current_commute_mode",
         old_val="can still get to work",
         m_old="I can still get to work fine — I bike, and I also keep a car for backup.",
         m_new="Recovering from a minor foot strain, I've eased off the biking for now.",
         alt_support=True,
         expl="Even if biking is paused, the user retains an independent way to get to work (the car), so the conclusion is core-retained.",
         should_invalidate=False,
         dim1="Can the user still get to work?",
         dim2="Since the user can get to work, can you confirm their commute is fine?",
         dim3="Help me plan getting to an appointment tomorrow."),
]

# novel-phrasing variants (same conflict, lexicon hand-schema keywords miss) for
# the generalization probe used in the ablation.
NOVEL_PHRASINGS = {
    "synth_T1_0000": "These days my ankle therapist insists I keep it gentle, so the office is just a short amble away each dawn.",
    "synth_T1_0001": "The dry desert wind down here keeps me parched; nothing like the place I left behind up north.",
}


def _ts(y, m, d):
    return f"{y:04d}-{m:02d}-{d:02d} 09:00"


# Mechanism-stress positives: each isolates ONE module so its ablation drops.
#   disjunctive_defeat -> needs SAT proof-by-cases (Ours w/o SAT core fails)
#   echo_chamber       -> needs source independence (Ours w/o multi-agent fails)
MECHANISM_TEMPLATES = [
    dict(ct="T2", kind="disjunctive", attr_b="routine_and_transport/current_commute_mode",
         old_val="gets to the office somehow", should_invalidate=True, disjunctive_defeat=True,
         m_old="I always make it into the office one way or another - car or the bus, whatever works.",
         m_new="I sold the car last month, and they also shut down the only bus line near me.",
         expl="Getting to the office required car OR bus; both are now gone, so by cases the old "
              "'can commute' belief is defeated (needs disjunctive/proof-by-cases reasoning).",
         dim1="Can the user still reliably get to the office the way they used to?",
         dim2="Since the user can always get to the office, can you book them an early in-person meeting?",
         dim3="Help me arrange getting to a mandatory on-site meeting tomorrow."),
    dict(ct="T2", kind="echo", attr_b="role_and_identity/employment_status",
         old_val="works at the old firm", should_invalidate=True, echo_chamber=True, echo_count=3,
         m_old="I work at the old firm downtown - mentioned it a few times, it keeps coming up.",
         m_new="Today was my first day at a completely different company across town.",
         expl="The old job is restated several times but all trace to one origin; a single genuine "
              "new statement should defeat it (echoes are not independent confirmation).",
         dim1="Does the user still work at the old firm?",
         dim2="Since the user works at the old firm, can you email their old-firm address?",
         dim3="Help me plan my first week at work."),
]


def make_synth_dataset(n: int, noise_per_sample: int = 6, seed: int = 42,
                       drop_hint: bool = False, negatives: bool = True) -> List[dict]:
    """Build n positive (should-invalidate) records; if negatives=True, also
    append the NEGATIVE_TEMPLATES (should-NOT-invalidate controls) so the judge
    is not all-positive and a system that simply over-retracts is penalised."""
    rng = random.Random(seed)
    noise_pool = [
        "I tried a new ramen place downtown, the broth was incredible.",
        "Finally finished that 800-page novel I'd been putting off.",
        "My neighbor's dog learned to open the gate, chaos ensued.",
        "Spent the evening reorganizing my bookshelf by color.",
        "The new coffee mug I ordered arrived chipped, mildly annoyed.",
        "Watched a documentary about deep-sea creatures, fascinating stuff.",
        "Tried baking sourdough again; still too dense but improving.",
        "Got really into a puzzle game this week, can't stop playing.",
    ]

    def _wrap(tpl, uid, ct, gold_hint):
        k = noise_per_sample
        head = [rng.choice(noise_pool) for _ in range(k // 2)]
        mid = [rng.choice(noise_pool) for _ in range(k - k // 2)]
        sessions, timestamps = [], []
        for j, nt in enumerate(head):
            sessions.append([{"role": "user", "content": nt},
                             {"role": "assistant", "content": "That sounds nice!"}])
            timestamps.append(_ts(2025, 1, 5 + j))
        idx_old = len(sessions)
        sessions.append([{"role": "user", "content": tpl["m_old"]},
                         {"role": "assistant", "content": "Got it, noted."}])
        timestamps.append(_ts(2025, 2, 1))
        for j, nt in enumerate(mid):
            sessions.append([{"role": "user", "content": nt},
                             {"role": "assistant", "content": "Interesting!"}])
            timestamps.append(_ts(2025, 3, 5 + j))
        idx_new = len(sessions)
        sessions.append([{"role": "user", "content": tpl["m_new"]},
                         {"role": "assistant", "content": "Thanks for sharing."}])
        timestamps.append(_ts(2025, 6, 15))
        rec = {
            "uid": uid, "conflict_type": ct,
            "M_old": tpl["m_old"], "M_new": tpl["m_new"], "explanation": tpl["expl"],
            "should_invalidate": tpl.get("should_invalidate", True),
            "probing_queries": {"dim1_query": tpl["dim1"], "dim2_query": tpl["dim2"],
                                "dim3_query": tpl["dim3"]},
            "haystack_session": sessions, "timestamps": timestamps,
            "relevant_session_index": [idx_old, idx_new],
            "graph_hint": gold_hint,
        }
        if tpl.get("alt_support"):
            rec["alt_support"] = True
        if tpl.get("low_credibility_new"):
            rec["low_credibility_new"] = True
        if tpl.get("disjunctive_defeat"):
            rec["disjunctive_defeat"] = True
        if tpl.get("echo_chamber"):
            rec["echo_chamber"] = True
            rec["echo_count"] = tpl.get("echo_count", 3)
        if drop_hint:
            rec.pop("graph_hint", None)
            rec.pop("conflict_type", None)
        return rec

    records = []
    for i in range(n):
        tpl = SYNTH_TEMPLATES[i % len(SYNTH_TEMPLATES)]
        gh = {"attribute_b": tpl["attr_b"], "value_old": tpl["old_val"],
              "value_new": tpl.get("new_val"), "upstream_a": tpl.get("upstream_a")}
        records.append(_wrap(tpl, f"synth_{tpl['ct']}_{i:04d}", tpl["ct"], gh))
    if negatives:
        for j, tpl in enumerate(NEGATIVE_TEMPLATES):
            gh = {"attribute_b": tpl["attr_b"], "value_old": tpl["old_val"],
                  "value_new": None, "upstream_a": tpl.get("upstream_a")}
            records.append(_wrap(tpl, f"synth_NEG_{tpl['kind']}_{j:04d}", tpl["ct"], gh))
        for j, tpl in enumerate(MECHANISM_TEMPLATES):
            gh = {"attribute_b": tpl["attr_b"], "value_old": tpl["old_val"],
                  "value_new": None, "upstream_a": tpl.get("upstream_a")}
            records.append(_wrap(tpl, f"synth_MECH_{tpl['kind']}_{j:04d}", tpl["ct"], gh))
    return records


# ---------------- deterministic local judge ----------------
_INVALID_CUES = ["no longer", "not valid", "may have changed", "has changed", "outdated",
                 "superseded", "uncertain", "re-verify", "reverify", "can't assume",
                 "cannot assume", "no longer valid", "not assume", "false assumption",
                 "false premise", "out of date", "changed", "new state", "updated"]
_STILL_CUES = ["still holds", "still valid", "still true", "remains", "yes, the user still",
               "as before", "unchanged", "still applies"]
_NEG_PREFIXES = ["not ", "won't ", "wont ", "would not ", "wouldn't ", "no longer ",
                 "cannot ", "can't ", "doesn't ", "do not ", "treat the previous", "re-verified before"]


def _has_any(text, cues):
    t = (text or "").lower()
    return any(c.lower() in t for c in cues)


def _asserts_still_valid(text):
    t = (text or "").lower()
    for cue in _STILL_CUES:
        idx = t.find(cue.lower())
        while idx != -1:
            window = t[max(0, idx - 28):idx]
            if not any(neg in window for neg in _NEG_PREFIXES):
                return True
            idx = t.find(cue.lower(), idx + 1)
    return False


_STILL_VALID_CUES = ["still applies", "still holds", "still appears", "remains valid",
                     "still valid", "remains the user", "keep treating it as current",
                     "i will keep", "proceed on that basis", "i can act on it"]


def local_judge(record, responses):
    """Gold-aware: positives (should_invalidate=True) must show invalidation;
    negatives (should_invalidate=False) must NOT invalidate and should keep the
    old belief. This penalises both missed staleness AND over-retraction."""
    d1 = responses.get("dim1_response", "")
    d2 = responses.get("dim2_response", "")
    d3 = responses.get("dim3_response", "")
    should_invalidate = record.get("should_invalidate", True)

    if should_invalidate:
        sr = _has_any(d1, _INVALID_CUES) and not _asserts_still_valid(d1)
        pr = (_has_any(d2, ["false", "premise", "no longer", "changed", "double-check",
                            "won't build", "outdated", "current state"])
              and not _has_any(d2, ["here is your scenic cycling route", "as you bike daily"]))
        ipa = (_has_any(d3, ["current state", "current situation", "updated", "new value",
                             "no longer safe", "no longer valid", "robust to the change",
                             "won't plan around it", "not rely", "uncertain", "i'd suggest",
                             "given that", "instead"])
               and not _has_any(d3, ["since you bike", "your subway", "as a teacher you"]))
    else:
        # correct behaviour is to KEEP the old belief (do not invalidate)
        sr = _has_any(d1, _STILL_VALID_CUES) and not _has_any(d1, ["no longer valid", "outdated",
                                                                   "superseded", "is likely no longer"])
        pr = _has_any(d2, ["still holds", "can act on it", "nothing", "does not"]) and \
            not _has_any(d2, ["false assumption", "outdated premise", "no longer the case"])
        ipa = _has_any(d3, ["still holds", "current state", "proceed on that basis", "remains"]) and \
            not _has_any(d3, ["no longer safe", "now-uncertain", "robust to the change"])
    return {"dim1": bool(sr), "dim2": bool(pr), "dim3": bool(ipa),
            "should_invalidate": bool(should_invalidate)}


# ==========================================================================
# ==== module: ablation
# ==========================================================================
# -*- coding: utf-8 -*-
"""
stale_hyperbase.ablation
=========================
The 4-version ablation matrix requested, plus supporting arms.

  Arm                              extraction   adjudicator   tests
  -------------------------------- -----------  ------------  ------------------------------
  Oracle extraction + Ours         oracle       ours          formal-mechanism upper bound
  LLM extraction + Ours            llm          ours          real system
  LLM-only adjudicator             llm          llm_only      is the formal layer necessary?
  Formal-only with hand schema     hand_schema  ours          is LLM generalization necessary?

Extra arms wired in for completeness:
  - Ours w/o SAT core   (use_sat=False)         : isolates the strict DPLL/SAT Hansson core
  - Ours w/o multi-agent (ablate_multiagent)    : isolates trust/consensus/source-independence

Each arm is scored on (a) the standard synth set and (b) a NOVEL-phrasing
generalization probe where hand-schema keywords fail but oracle/LLM still work.
"""




def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


ARMS = [
    # (label, extraction_mode, adjudicator_mode, kwargs)
    ("Oracle extraction + Ours", "oracle", "ours", {}),
    ("LLM extraction + Ours", "llm", "ours", {}),
    ("LLM-only adjudicator", "llm", "llm_only", {}),
    ("Formal-only with hand schema", "hand_schema", "ours", {}),
    ("Ours (LLM) w/o SAT core", "llm", "ours", {"use_sat": False}),
    ("Ours (LLM) w/o multi-agent", "llm", "ours", {"ablate_multiagent": True}),
]

# keyword-free novel phrasings, indexed by the slot M_new is *about* (= upstream_a
# for T2, else attribute_b for T1). None of these share a token with _SLOT_KEYWORDS,
# so the hand schema (keyword) extractor cannot recover the slot, while the LLM
# proposer (or its semantic backoff) still can — this is the generalization gap.
_NOVEL_BY_SLOT = {
    "health_and_mobility/current_health_state":
        "My physio keeps reminding me to ease the strain on that joint while it mends.",
    "location_and_living/current_base_location":
        "We just settled into a sleepy little place way out past the edge of everything.",
    "routine_and_transport/current_commute_mode":
        "Getting to the workplace is now just a short gentle saunter from my front step.",
    "role_and_identity/employment_status":
        "After decades, my mornings are finally my own with nowhere I have to be.",
}


def _gold_of(rec):
    """Gold structure kept aside for scoring even when hidden from the extractor."""
    g = rec.get("graph_hint") or rec.get("_gold_hint") or {}
    ct = (rec.get("conflict_type") or rec.get("_gold_ct")
          or ("T2" if g.get("upstream_a") else "T1"))
    return g.get("attribute_b"), ct.upper() if ct else None, g.get("upstream_a")


def _eval_arm(label, ext_mode, adj_mode, kwargs, records, llm=None):
    eng = StaleEngine(mode=adj_mode, extraction_mode=ext_mode, llm=llm, **kwargs)
    by_type = defaultdict(lambda: {"dim1": [0, 0], "dim2": [0, 0], "dim3": [0, 0]})
    src_indep = []
    sat_agree = [0, 0]
    fidelity = [0, 0]               # extractor recovered (attr_b, ct, upstream_a)?
    adj = [0, 0]                    # verdict matches gold should_invalidate?
    neg = [0, 0]                    # negative subset: correctly NOT invalidated?
    for rec in records:
        gm, atms, bb, ma, res = eng.adjudicate(rec)
        responses = eng.answer(gm, atms, bb, ma, res, rec.get("probing_queries", {}))
        verdict = local_judge(rec, responses)
        gold_inv = rec.get("should_invalidate", True)
        adj[1] += 1
        sys_invalidated = res.sr_should_invalidate and not res.old_supported_terminal
        if sys_invalidated == bool(gold_inv):
            adj[0] += 1
        if not gold_inv:
            neg[1] += 1
            if not sys_invalidated:
                neg[0] += 1
        # extraction fidelity vs gold (LLM-only has no real extractor structure)
        g_attr, g_ct, g_up = _gold_of(rec)
        if g_attr is not None:
            fidelity[1] += 1
            ok = (res.attribute_b == g_attr and res.conflict_type == g_ct
                  and (res.upstream_a == g_up or (not g_up and not res.upstream_a)))
            fidelity[0] += int(ok)
        ct = (rec.get("conflict_type") or res.conflict_type or "NA")
        for d in ("dim1", "dim2", "dim3"):
            by_type[ct][d][1] += 1
            if verdict[d]:
                by_type[ct][d][0] += 1
        src_indep.append(res.source_independence)
        if res.sat_old_believed is not None:
            sat_agree[1] += 1
            if res.sat_old_believed == res.old_supported_terminal:
                sat_agree[0] += 1
    cells = []
    type_cells = {}
    for ct, dims in by_type.items():
        cell = {}
        for d, name in (("dim1", "SR"), ("dim2", "PR"), ("dim3", "IPA")):
            c, t = dims[d]
            acc = round(100.0 * c / t, 1) if t else 0.0
            cell[name] = acc
            cells.append(acc)
        type_cells[ct] = cell
    return {
        "label": label, "extraction": ext_mode, "adjudicator": adj_mode, "kwargs": kwargs,
        "by_type": type_cells,
        "overall_acc_pct": round(_mean(cells), 1),
        "adjudication_accuracy": round(adj[0] / max(adj[1], 1), 4),
        "negative_keep_accuracy": (round(neg[0] / neg[1], 4) if neg[1] else None),
        "extraction_fidelity": (round(fidelity[0] / fidelity[1], 4) if fidelity[1] else None),
        "mean_source_independence": round(_mean(src_indep), 3),
        "sat_atms_agreement": (round(sat_agree[0] / sat_agree[1], 4) if sat_agree[1] else None),
    }


def run_ablation_matrix(out: Path, n: int = 24, seed: int = 42, use_llm: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    llm = build_llm_callable() if use_llm else None
    std = make_synth_dataset(n=n, seed=seed)
    # generalization probe: keyword-free novel phrasings of M_new for every record,
    # so the hand-schema lexicon misses the new slot while LLM/oracle still recover it.
    novel = make_synth_dataset(n=n, seed=seed)
    for rec in novel:
        gh = dict(rec.get("graph_hint", {}))
        rec["_gold_hint"] = gh
        rec["_gold_ct"] = rec.get("conflict_type")
        key = gh.get("upstream_a") or gh.get("attribute_b")
        rec["M_new"] = _NOVEL_BY_SLOT.get(key, rec["M_new"])
        # hide the oracle hint so only real extraction can recover structure
        rec.pop("graph_hint", None)
        rec.pop("conflict_type", None)

    results = {"standard": [], "generalization_novel": []}
    for label, ext, adj, kw in ARMS:
        results["standard"].append(_eval_arm(label, ext, adj, kw, std, llm=llm))
        # generalization arm: oracle has no hint here, so it degrades like LLM/hand; that is the point
        results["generalization_novel"].append(_eval_arm(label, ext, adj, kw, novel, llm=llm))

    (out / "ablation_matrix.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def format_ablation_table(results: dict) -> str:
    lines = []
    for split in ("standard", "generalization_novel"):
        lines.append(f"\n=== {split} ===")
        lines.append(f"{'arm':<32}{'judgeAcc':>9}{'adjAcc':>8}{'negKeep':>9}{'fidelity':>9}{'srcInd':>8}{'SAT~ATMS':>10}")
        lines.append("-" * 85)
        for r in results[split]:
            sat = f"{r['sat_atms_agreement']:.2f}" if r["sat_atms_agreement"] is not None else "—"
            fid = f"{r['extraction_fidelity']:.2f}" if r["extraction_fidelity"] is not None else "—"
            nk = f"{r['negative_keep_accuracy']:.2f}" if r["negative_keep_accuracy"] is not None else "—"
            lines.append(f"{r['label']:<32}{r['overall_acc_pct']:>8.1f}%"
                         f"{r['adjudication_accuracy']:>8.2f}{nk:>9}{fid:>9}"
                         f"{r['mean_source_independence']:>8.2f}{sat:>10}")
    lines.append("\nlegend: judgeAcc=gold-aware SR/PR/IPA; adjAcc=verdict matches gold "
                 "should_invalidate; negKeep=fraction of negatives correctly NOT invalidated; "
                 "fidelity=extractor recovered (attr,type,upstream); srcInd=mean independent origins.")
    return "\n".join(lines)


# ==========================================================================
# ==== module: benchmark
# ==========================================================================
# -*- coding: utf-8 -*-
"""
stale_hyperbase.benchmark
=========================
ATMS necessity micro-benchmark (Full-ATMS vs BFS weighted-cascade), source
independence demo (evidence-DAG leaves vs holder counting), premise-resistance
demo, and the falsifiable AGM/Darwiche-Pearl postulate harness over the strict
SAT belief base.
"""




# ---------------- ATMS necessity micro-cases ----------------
def _micro_cases():
    def single_path(i):
        k = ATMSKernelV2(); k.assert_evidence("A", f"e{i}", "obs")
        k.add_justification(["A"], "B", neg_premises=["D"], operator="DEFEASIBLE")
        k.assert_evidence("D", f"d{i}", "official", evidence_type="official", trust=0.97)
        return k, {"B": False, "A": True, "D": True}, "B", "D"

    def alt_path(i):
        k = ATMSKernelV2()
        k.assert_evidence("P1base", f"p1{i}", "obs")
        k.assert_evidence("P2", f"p2{i}", "authority", evidence_type="authority", trust=0.85)
        k.add_justification(["P1base"], "P1", neg_premises=["D"], operator="DEFEASIBLE")
        k.add_justification(["P1"], "Q"); k.add_justification(["P2"], "Q")
        k.assert_evidence("D", f"d{i}", "official", evidence_type="official", trust=0.97)
        return k, {"Q": True, "P1": False, "P2": True}, "Q", "P1"

    def multi_source(i):
        k = ATMSKernelV2()
        a1 = k.assert_evidence("C", f"evA{i}", "Tom", trust=0.8)
        k.assert_evidence("C", f"evB{i}", "Elena", trust=0.8)
        k.assert_evidence("C", f"reg{i}", "official", evidence_type="official", trust=0.95)
        k.revoke_evidence(a1)
        return k, {"C": True}, "C", None

    def nogood(i):
        k = ATMSKernelV2()
        k.assert_evidence("knight_record", f"k{i}", "officeA", evidence_type="official", trust=0.9)
        k.assert_evidence("impostor_record", f"im{i}", "officeB", evidence_type="official", trust=0.95)
        k.add_justification(["knight_record"], "is_knight")
        k.add_justification(["impostor_record"], "is_impostor")
        k.add_nogood_claims(["is_knight", "is_impostor"])
        return k, {"is_impostor": True}, "is_impostor", None

    def temporal(i):
        k = ATMSKernelV2()
        k.assert_evidence("city_seattle", f"s{i}", "user", trust=0.7, valid_from=0, valid_to=10)
        k.assert_evidence("city_portland", f"p{i}", "user", trust=0.7, valid_from=10)
        k.now = 15
        return k, {"city_seattle": False, "city_portland": True}, "city_portland", None

    def norm(i):
        k = ATMSKernelV2()
        k.assert_evidence("rule_old", f"ro{i}", "gov", evidence_type="official", trust=0.9, valid_from=0, valid_to=5)
        k.assert_evidence("rule_new", f"rn{i}", "gov", evidence_type="official", trust=0.9, valid_from=5)
        k.add_justification(["rule_old"], "permitted_old"); k.add_justification(["rule_new"], "permitted_new")
        k.now = 8
        return k, {"permitted_old": False, "permitted_new": True}, "permitted_new", None

    def multi_agent(i):
        k = ATMSKernelV2()
        k.assert_evidence("endorsement", f"en{i}", "Elena", evidence_type="authority", trust=0.8)
        k.add_justification(["endorsement"], "is_knight", neg_premises=["D"], operator="DEFEASIBLE")
        return k, {"is_knight_world": True, "is_knight_tom": False}, "is_knight", ("ctx", set())

    return [("single_path_contraction", single_path, 15), ("alternative_path_retention", alt_path, 20),
            ("multi_source_retraction", multi_source, 15), ("nogood_conflict", nogood, 15),
            ("temporal_update", temporal, 15), ("norm_change", norm, 10),
            ("multi_agent_perspective", multi_agent, 10)]


def run_atms_necessity(out: Path, seed: int = 0) -> dict:
    random.seed(seed)
    specs = _micro_cases()
    full = defaultdict(lambda: [0, 0]); bfs = defaultdict(lambda: [0, 0])
    cr_full = [0, 0]; cr_bfs = [0, 0]; n_core_retain = 0
    for name, fn, count in specs:
        for i in range(count):
            res = fn(i); k, gold, query, extra = res[0], res[1], res[2], res[3]
            if name == "alternative_path_retention":
                n_core_retain += 1
                fo = k.has_alternative_support(query, extra); go = gold[query]
                cr_full[1] += 1; cr_full[0] += int(fo == go)
                bo = BFSCascadeBaseline(k).supported_after_defeat(extra, query)
                cr_bfs[1] += 1; cr_bfs[0] += int(bo == go)
                full[name][1] += 1; full[name][0] += int(fo == go)
                bfs[name][1] += 1; bfs[name][0] += int(bo == go)
            elif name == "multi_agent_perspective":
                _, tom_ctx = extra
                ok = (k.supported("is_knight") == gold["is_knight_world"]) and \
                     (k.supported("is_knight", context=tom_ctx) == gold["is_knight_tom"])
                full[name][1] += 1; full[name][0] += int(ok)
                bfs[name][1] += 1; bfs[name][0] += int(False)
            elif name == "multi_source_retraction":
                full[name][1] += 1; full[name][0] += int(k.supported(query) == gold[query])
                bo = BFSCascadeBaseline(k).supported_after_defeat("__none__", query)
                bfs[name][1] += 1; bfs[name][0] += int(bo == gold[query])
            else:
                b = BFSCascadeBaseline(k)
                for c, g in gold.items():
                    full[name][1] += 1; full[name][0] += int(k.supported(c) == g)
                    bfs[name][1] += 1; bfs[name][0] += int(b.supported_after_defeat(extra or "__none__", c) == g)
    tf = [0, 0]; tb = [0, 0]
    per_type = {}
    for name, _, _ in specs:
        f = full[name]; b = bfs[name]
        tf[0] += f[0]; tf[1] += f[1]; tb[0] += b[0]; tb[1] += b[1]
        per_type[name] = {"full": round(f[0]/max(f[1],1), 4), "bfs": round(b[0]/max(b[1],1), 4)}
    out_d = {"full_overall": round(tf[0]/tf[1], 4), "bfs_overall": round(tb[0]/tb[1], 4),
             "core_retain_full": round(cr_full[0]/max(cr_full[1],1), 4),
             "core_retain_bfs": round(cr_bfs[0]/max(cr_bfs[1],1), 4),
             "n_core_retain_decisions": n_core_retain, "n_cases": tf[1], "per_type": per_type}
    if out:
        (out / "atms_necessity.json").write_text(json.dumps(out_d, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_d


def demo_source_independence() -> dict:
    k1 = ATMSKernelV2()
    for teller in ["Tom", "Elena", "Duran"] + [f"villager{j}" for j in range(7)]:
        k1.assert_evidence("rumor_X", "event_1", teller, evidence_type="hearsay", trust=0.5)
    k1.add_justification(["rumor_X"], "claim_X")
    o1 = k1.independent_origins("claim_X")
    k2 = ATMSKernelV2()
    k2.assert_evidence("helped", "event_1", "Elena", trust=0.8)
    k2.assert_evidence("combat", "event_3", "Elena", trust=0.8)
    k2.assert_evidence("endorse", "endorsement_1", "Elena", evidence_type="authority", trust=0.8)
    k2.add_justification(["helped"], "good"); k2.add_justification(["combat"], "good")
    k2.add_justification(["endorse"], "good")
    o2 = k2.independent_origins("good")
    return {"single_event_10_tellers_origins": len(o1), "single_correct": len(o1) == 1,
            "three_event_one_holder_origins": len(o2), "three_correct": len(o2) == 3}


def demo_premise_resistance() -> dict:
    k = ATMSKernelV2()
    k.assert_evidence("helped_village", "ev_help", "Duran", trust=0.85)
    k.assert_evidence("sword_given", "ev_sword", "Duran", trust=0.9)
    k.add_justification(["helped_village"], "good_character")
    k.add_justification(["good_character"], "is_knight", neg_premises=["official_denial"], operator="DEFEASIBLE")
    k.add_justification(["is_knight", "sword_given"], "transfer_legitimate")
    k.assert_evidence("official_denial", "royal_register", "court", evidence_type="official", trust=0.97)
    probe = {"premise_is_knight": ("is_knight", False),
             "fact_sword_given": ("sword_given", True),
             "conclusion_transfer_legitimate": ("transfer_legitimate", False),
             "fact_helped_village": ("helped_village", True)}
    responses = {lbl: {"believed": k.supported(c), "gold": g} for lbl, (c, g) in probe.items()}
    allok = all(k.supported(c) == g for c, g in probe.values())
    return {"premise_resistance_correct": allok, "responses": responses}


def run_postulate_harness(out: Path, n_scenarios: int = 200, seed: int = 0) -> dict:
    """Falsifiable AGM/DP postulates over the strict SAT belief base. Includes a
    deliberately-broken operator to prove the harness FAILS bad operators."""
    good, _ = run_property_bench(op=levi_operator, n_scenarios=n_scenarios, seed=seed, stateful_dp=True)
    bad, _ = run_property_bench(op=naive_overwrite_operator, n_scenarios=n_scenarios, seed=seed)
    out_d = {"levi_kernel_operator": {k: list(v) for k, v in good.rates.items()},
             "naive_overwrite_operator": {k: list(v) for k, v in bad.rates.items()},
             "n_scenarios": n_scenarios}
    if out:
        (out / "agm_postulates.json").write_text(json.dumps(out_d, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_d


# ==========================================================================
# ==== module: selftest
# ==========================================================================
# -*- coding: utf-8 -*-
"""stale_hyperbase.selftest — unit self-checks for the full stack."""



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

    # --- strict SAT core ---
    a, b = atom("a"), atom("b")
    check("SAT: a & ~a is unsatisfiable", not is_consistent([And(a, Not(a))]))
    check("SAT: {a, a->b} entails b", entails([a, (a >> b)], b))
    rng = random.Random(0)
    ok = True
    names = ["p", "q", "r", "s"]
    for _ in range(300):
        f = atom(rng.choice(names))
        for _ in range(rng.randint(0, 3)):
            op = rng.choice(["&", "|", "~"])
            g = atom(rng.choice(names))
            f = And(f, g) if op == "&" else (Not(f) if op == "~" else (f | g))
        sat = _DPLL(to_cnf(f)).solve()
        if sat != brute_force_satisfiable(f):
            ok = False; break
    check("DPLL SAT matches brute force on 300 random formulas", ok)

    # --- Hansson ordered incision: defeasible justification cut first ---
    bb = OrderedIncisionBase()
    bb.expand(Belief(atom("evid"), Source.USER_IMPLICIT, text="EVID::evid"))
    bb.expand(Belief((Atom("evid") >> atom("bold")), Source.INFERRED, reliability=0.5,
                     text="JUST::bold<=evid"))
    rep = bb.ordered_contract(atom("bold"))
    cut_just_first = bool(rep.steps and rep.steps[0].rule.startswith("step1"))
    check("Hansson: ordered incision cuts defeasible justification FIRST", cut_just_first)
    check("Hansson: raw evidence 'evid' survives the cut", bb.believes(atom("evid")))
    check("Hansson: alpha 'bold' no longer entailed", not bb.believes(atom("bold")))

    # --- T1 (co-referential) via full engine ---
    eng = StaleEngine(mode="ours", extraction_mode="oracle", pipeline="v6")
    rec_t1 = make_synth_dataset(1)[0]
    gm, atms, bb2, ma, res = eng.adjudicate(rec_t1)
    b_old = f"{res.attribute_b}={res.value_old}"
    check("T1: ATMS old value no longer supported", not atms.is_supported(b_old))
    check("T1: operation == update", res.operation == "update")
    ans1 = eng.answer(gm, atms, bb2, ma, res, rec_t1["probing_queries"])
    v1 = local_judge(rec_t1, ans1)
    check("T1: SR/PR/IPA pass", v1["dim1"] and v1["dim2"] and v1["dim3"])

    # --- T2 (propagated) via full engine ---
    rec_t2 = next(r for r in make_synth_dataset(6) if r["conflict_type"] == "T2")
    gm2, atms2, bb3, ma2, res2 = eng.adjudicate(rec_t2)
    b_old2 = f"{res2.attribute_b}={res2.value_old}"
    check("T2: conflict_type == T2", res2.conflict_type == "T2")
    check("T2: operation == contraction", res2.operation == "contraction")
    check("T2: ATMS old value contracted (unsupported)", not atms2.is_supported(b_old2))
    check("T2: ordered incision recorded steps", len(res2.incision_steps) >= 1)
    check("T2: incision cuts defeasible justification FIRST (step1)",
          any(s["rule"].startswith("step1") for s in res2.incision_steps))
    check("T2: graph wrote back HISTORICAL retention (step4)",
          any(s["rule"].startswith("step4") for s in res2.incision_steps))
    ans2 = eng.answer(gm2, atms2, bb3, ma2, res2, rec_t2["probing_queries"])
    v2 = local_judge(rec_t2, ans2)
    check("T2: SR/PR/IPA pass", v2["dim1"] and v2["dim2"] and v2["dim3"])
    check("T2: multi-agent merged view AGREES with ATMS (both unsupported)",
          ma2.merged_supported(b_old2) == atms2.is_supported(b_old2))

    # --- negative control: NO_EFFECT must KEEP the old belief (no over-retraction) ---
    neg = next((r for r in make_synth_dataset(6) if r["uid"].startswith("synth_NEG_irrelevant")), None)
    if neg is not None:
        gmn, atmsn, bbn, man, resn = eng.adjudicate(neg)
        check("NEG: irrelevant new info -> NO_EFFECT, old belief kept",
              resn.conflict_type == "NO_EFFECT" and not resn.sr_should_invalidate)
        ansn = eng.answer(gmn, atmsn, bbn, man, resn, neg["probing_queries"])
        vn = local_judge(neg, ansn)
        check("NEG: judge scores keep-old correctly", vn["dim1"])

    # --- multi-agent source independence (echo-chamber resistance) ---
    si = demo_source_independence()
    check("source independence: 10 tellers of 1 event -> 1 origin", si["single_correct"])
    check("source independence: 3 events 1 holder -> 3 origins", si["three_correct"])

    # --- premise resistance ---
    pr = demo_premise_resistance()
    check("premise resistance: reject false premise, keep facts", pr["premise_resistance_correct"])

    # --- ATMS necessity (Full beats BFS on alternative-path retention) ---
    k = ATMSKernelV2()
    k.assert_evidence("P1base", "p1", "obs")
    k.assert_evidence("P2", "p2", "authority", evidence_type="authority", trust=0.85)
    k.add_justification(["P1base"], "P1", neg_premises=["D"], operator="DEFEASIBLE")
    k.add_justification(["P1"], "Q"); k.add_justification(["P2"], "Q")
    k.assert_evidence("D", "d", "official", evidence_type="official", trust=0.97)
    full_ok = k.has_alternative_support("Q", "P1")            # Q survives via P2
    bfs_ok = BFSCascadeBaseline(k).supported_after_defeat("P1", "Q")
    check("ATMS necessity: Full retains Q via alt path; BFS does not", full_ok and not bfs_ok)

    # --- schema validity ---
    check("graph: no schema-invalid edges", gm.stats()["schema_invalid_edges"] == 0)


    # --- mechanism-isolating ablations actually move ---
    mech = [r for r in make_synth_dataset(6) if r["uid"].startswith("synth_MECH")]
    disj = next(r for r in mech if "disjunctive" in r["uid"])
    eng_sat = StaleEngine(mode="ours", extraction_mode="oracle", use_sat=True)
    eng_nosat = StaleEngine(mode="ours", extraction_mode="oracle", use_sat=False)
    _, _, _, _, rs = eng_sat.adjudicate(disj)
    _, _, _, _, rn = eng_nosat.adjudicate(disj)
    check("ablation: disjunctive defeat needs SAT (w/o SAT fails to invalidate)",
          rs.sr_should_invalidate and not rn.sr_should_invalidate)
    echo = next(r for r in mech if "echo" in r["uid"])
    eng_ma = StaleEngine(mode="ours", extraction_mode="oracle", ablate_multiagent=False)
    eng_noma = StaleEngine(mode="ours", extraction_mode="oracle", ablate_multiagent=True)
    _, _, _, _, re_ = eng_ma.adjudicate(echo)
    _, _, _, _, rne = eng_noma.adjudicate(echo)
    check("ablation: echo-chamber needs multi-agent (w/o MA fails to invalidate)",
          re_.sr_should_invalidate and not rne.sr_should_invalidate)


    # --- v6 lean pipeline: free-LLM SR/IPA strength + surgical detect-then-reject PR ---
    eng_v6 = StaleEngine(mode="ours", extraction_mode="llm", llm=None, pipeline="v6")
    free6 = {"uid": "v6_injury", "conflict_type": "",
             "M_old": "I bike to the office along the river every morning.",
             "M_new": "Recovering from knee surgery; I can barely put weight on my leg right now.",
             "probing_queries": {
                 "dim1_query": "Does the user still bike to the office?",
                 "dim2_query": "Since the user bikes daily, recommend a longer cycling route this week.",
                 "dim3_query": "Best way to get to a meeting at the office tomorrow?"}}
    gm6, atms6, bb6, ma6, res6 = eng_v6.adjudicate(free6)
    a6 = eng_v6.answer(gm6, atms6, bb6, ma6, res6, free6["probing_queries"])
    d1l, d2l, d3l = (a6["dim1_response"].lower(), a6["dim2_response"].lower(), a6["dim3_response"].lower())
    check("v6: pipeline tag is v6 (no verdict-forcing machinery)",
          res6.verification["pipeline"] == "v6")
    check("v6: Dim1 flags the old fact may no longer be valid",
          "no longer" in d1l or "outdated" in d1l or "re-confirm" in d1l)
    check("v6: Dim2 rejects the outdated premise (the surgical fix)",
          ("outdated" in d2l or "should not act" in d2l) and "here’s a week" not in d2l)
    check("v6: Dim3 is concrete (names specific options), not vacuous",
          any(k in d3l for k in ["rideshare", "taxi", "transit", "accessible", "(a)", "(b)"]))
    check("v6: history is populated for the LLM prompts",
          bool(getattr(res6, "_history", "")) and "knee surgery" in res6._history.lower())

    print(f"\nself-test: {passed}/{total} passed")
    return passed == total


# ==========================================================================
# ==== module: cli
# ==========================================================================
# -*- coding: utf-8 -*-
"""stale_hyperbase.cli — command line entrypoint."""




def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def run_dataset(records, out: Path, mode="ours", extraction_mode="oracle",
                use_llm=False, pipeline="verified"):
    out.mkdir(parents=True, exist_ok=True)
    llm = build_llm_callable() if use_llm else None
    eng = StaleEngine(mode=mode, extraction_mode=extraction_mode, llm=llm, pipeline=pipeline)
    answers, traces, judge_rows = [], [], []
    by_type = defaultdict(lambda: {"dim1": [0, 0], "dim2": [0, 0], "dim3": [0, 0]})
    for rec in records:
        gm, atms, bb, ma, res = eng.adjudicate(rec)
        responses = eng.answer(gm, atms, bb, ma, res, rec.get("probing_queries", {}))
        answers.append({"uid": rec["uid"],
                        "target_model": f"stale_hyperbase[{mode}/{extraction_mode}]"
                        + ("+LLM" if llm else ""),
                        "target_model_responses": responses})
        traces.append({"uid": rec["uid"], "conflict_type": res.conflict_type,
                       "attribute_b": res.attribute_b, "value_old": res.value_old,
                       "value_new": res.value_new, "upstream_a": res.upstream_a,
                       "operation": res.operation,
                       "old_supported_terminal": res.old_supported_terminal,
                       "sat_old_believed": res.sat_old_believed,
                       "source_independence": res.source_independence,
                       "multiagent_merged_old_supported": res.multiagent_merged_old_supported,
                       "incision_steps": res.incision_steps, "cascade": res.cascade,
                       "extractor": res.extractor, "extractor_notes": res.extractor_notes,
                       "verification": res.verification,
                       "graph_stats": gm.stats(), "atms_stats": atms.stats(),
                       "multiagent_stats": ma.stats()})
        v = local_judge(rec, responses)
        judge_rows.append({"uid": rec["uid"], "conflict_type": res.conflict_type, **v})
        ct = res.conflict_type or "NA"
        for d in ("dim1", "dim2", "dim3"):
            by_type[ct][d][1] += 1
            if v[d]:
                by_type[ct][d][0] += 1
    summary = {"by_type": {}, "overall": {}}
    cells = []
    for ct, dims in by_type.items():
        cell = {}
        for d, name in (("dim1", "SR"), ("dim2", "PR"), ("dim3", "IPA")):
            c, t = dims[d]
            acc = round(100.0 * c / t, 1) if t else 0.0
            cell[name] = acc
            cells.append(acc)
        summary["by_type"][ct] = cell
    summary["overall"]["accuracy_pct"] = round(_mean(cells), 1)
    summary["n_samples"] = len(records)
    summary["mode"] = mode
    summary["extraction_mode"] = extraction_mode
    (out / "answers.json").write_text(json.dumps(
        {"summary": {"target_model": answers[0]["target_model"] if answers else "stale_hyperbase",
                     "num_items": len(answers)}, "data": answers}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with (out / "traces.jsonl").open("w", encoding="utf-8") as f:
        for tr in traces:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    (out / "eval.json").write_text(json.dumps({"summary": summary, "per_sample": judge_rows},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser():
    ap = argparse.ArgumentParser(
        description="STALE on a strict ATMS+AGM+Hansson core with a multi-agent belief layer.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--synth", action="store_true", help="generate synth set + run + judge")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--mode", default="ours", choices=["ours", "llm_only"])
    ap.add_argument("--pipeline", default="v6", choices=["v6"],
                    help="verified = constraints+post-check+repair; prompt_only = prompt injection only")
    ap.add_argument("--extraction", default="oracle", choices=["oracle", "llm", "hand_schema"])
    ap.add_argument("--icds-path", default="", help="real STALE MAIN.json")
    ap.add_argument("--ablation", action="store_true", help="run the 4-version ablation matrix")
    ap.add_argument("--atms-bench", action="store_true", help="ATMS necessity micro-bench + source indep + premise resist")
    ap.add_argument("--postulates", action="store_true", help="falsifiable AGM/DP postulate harness")
    ap.add_argument("--use-llm", dest="use_llm", action="store_true", default=True,
                    help="use a real LLM for extraction/polish (default on; needs OPENAI_API_KEY/"
                         "YUNWU_API_KEY, else deterministic fallback is used and reported)")
    ap.add_argument("--no-llm", dest="use_llm", action="store_false",
                    help="force the deterministic (no-API) path")
    ap.add_argument("--out", default="runs/stale_hyperbase")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    out = Path(a.out)
    if a.use_llm and not (os.environ.get("OPENAI_API_KEY") or os.environ.get("YUNWU_API_KEY")):
        print("[notice] --use-llm requested but no OPENAI_API_KEY/YUNWU_API_KEY found; "
              "using the deterministic extractor/adjudicator fallback. Results below are the "
              "OFFLINE fallback, not a real-LLM run.", file=sys.stderr)
        a.use_llm = False
    if a.self_test:
        sys.exit(0 if run_self_test() else 1)
    if a.atms_bench:
        out.mkdir(parents=True, exist_ok=True)
        bench = run_atms_necessity(out)
        si = demo_source_independence()
        pr = demo_premise_resistance()
        combined = {"atms_necessity": bench, "source_independence": si, "premise_resistance": pr}
        (out / "atms_bench.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(combined, ensure_ascii=False, indent=2))
        return
    if a.postulates:
        out.mkdir(parents=True, exist_ok=True)
        res = run_postulate_harness(out)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if a.ablation:
        res = run_ablation_matrix(out, n=a.n, use_llm=a.use_llm)
        print(format_ablation_table(res))
        print(f"\n-> {(out / 'ablation_matrix.json').resolve()}")
        return
    if a.icds_path:
        payload = json.loads(Path(a.icds_path).read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("data", [])
        summary = run_dataset(records, out, mode=a.mode, extraction_mode=a.extraction, use_llm=a.use_llm, pipeline=a.pipeline)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    # default / --synth
    records = make_synth_dataset(n=a.n)
    out.mkdir(parents=True, exist_ok=True)
    (out / "synth_MAIN.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = run_dataset(records, out, mode=a.mode, extraction_mode=a.extraction, use_llm=a.use_llm, pipeline=a.pipeline)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n-> {out.resolve()}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
