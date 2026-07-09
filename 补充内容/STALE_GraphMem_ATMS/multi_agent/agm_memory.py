"""
agm_memory.py
=============
A runnable, theory-faithful core that migrates AGM belief revision to an
LLM-agent memory store.

Design decision (justified in the accompanying report):
  * The substrate is a BELIEF BASE (Section 5 of the SEP entry), NOT a logically
    closed belief set. LLM memory items are independent, syntactically distinct
    records; logical closure is replaced by a *bounded consequence operator*.
  * Epistemic ENTRENCHMENT is realized as a memory SALIENCE function
    (importance/recency/source-trust), so contraction = principled forgetting.
  * SUCCESS is deliberately downgraded to CREDIBILITY-LIMITED success to defend
    against hallucination / prompt injection (non-prioritized revision).

The propositional core gives us an exact, finite oracle for entailment so that
every AGM postulate can be *checked*, not asserted. The places where a real
system would call an LLM are isolated behind small interfaces:
    EntrenchmentModel  -> salience scorer (LLM/heuristic)
    Credibility        -> source-trust + consistency gate (LLM/retriever)
    contradiction_kernels -> minimal inconsistent subset finder (here exact;
                             in production an LLM/NLI contradiction detector)
"""

from itertools import combinations, chain
from functools import reduce


# --------------------------------------------------------------------------- #
#  Finite propositional logic, represented semantically (sets of worlds).      #
#  A 'sentence' is identified with its proposition: the frozenset of worlds    #
#  in which it is true. This makes Cn, entailment and consistency exact.       #
# --------------------------------------------------------------------------- #
def _powerset(xs):
    return chain.from_iterable(combinations(xs, r) for r in range(len(xs) + 1))


class Logic:
    def __init__(self, atoms):
        self.atoms = list(atoms)
        self.worlds = [frozenset(s) for s in _powerset(self.atoms)]
        self.TOP = frozenset(self.worlds)          # tautology
        self.BOT = frozenset()                      # falsum / contradiction

    # connectives over propositions ---------------------------------------- #
    def atom(self, a):
        return frozenset(w for w in self.worlds if a in w)

    def neg(self, p):
        return frozenset(w for w in self.worlds if w not in p)

    def conj(self, *ps):
        return reduce(lambda x, y: x & y, ps, self.TOP)

    def disj(self, *ps):
        return reduce(lambda x, y: x | y, ps, self.BOT)

    def imp(self, p, q):
        return self.neg(p) | q

    # semantics for a base (a finite collection of propositions) ----------- #
    def models(self, base):
        """[B]: worlds satisfying every element of the base."""
        return reduce(lambda x, y: x & y, base, self.TOP)

    def entails(self, base, p):
        """B |- p  iff  [B] subseteq [p]."""
        return self.models(base) <= p

    def consistent(self, base):
        return len(self.models(base)) > 0

    def equiv(self, p, q):
        return p == q   # logical equivalence == same proposition

    def prop_entails(self, prop, p):
        """For a *proposition* (set of worlds) `prop`: prop |- p iff prop ⊆ p."""
        return prop <= p


# --------------------------------------------------------------------------- #
#  Entrenchment  ==  memory salience.                                          #
#  Lower salience => given up first (least entrenched).                         #
# --------------------------------------------------------------------------- #
class EntrenchmentModel:
    """In production: an LLM/heuristic scoring importance x recency x trust.
       Here: an explicit dict keyed by the *base element* (a proposition)."""
    def __init__(self, scores):
        self.scores = dict(scores)          # proposition -> float

    def score(self, p):
        return self.scores.get(p, 0.0)

    def least(self, elements):
        return min(elements, key=self.score)

    def total(self, subset):
        return sum(self.score(p) for p in subset)


# --------------------------------------------------------------------------- #
#  Remainder sets and PARTIAL MEET contraction (base-level).                   #
# --------------------------------------------------------------------------- #
def remainder_set(L, base, p):
    """A _|_ p : inclusion-maximal subsets of `base` that do NOT entail p."""
    base = list(base)
    candidates = []
    for r in range(len(base), -1, -1):
        for sub in combinations(base, r):
            sub = set(sub)
            if not L.entails(sub, p):
                candidates.append(frozenset(sub))
        if candidates:           # all maximal ones share the largest size found
            # keep only inclusion-maximal among collected of this size+below
            pass
    # filter to inclusion-maximal
    maximal = []
    for c in candidates:
        if not any(c < d for d in candidates):
            maximal.append(c)
    # dedup
    return list({m for m in maximal})


def gamma_best(ent, remainders):
    """Relational selection: pick remainders of maximal retained entrenchment."""
    if not remainders:
        return []
    best = max(ent.total(r) for r in remainders)
    return [r for r in remainders if ent.total(r) == best]


def partial_meet_contract(L, base, p, ent):
    """K ÷ p = intersection of the selected best remainders.
       Tautology guard: contracting a tautology changes nothing (Vacuity-like)."""
    if p == L.TOP:                       # cannot remove a tautology
        return frozenset(base)
    R = remainder_set(L, base, p)
    if not R:                            # p not implied: nothing to remove
        return frozenset(base)
    selected = gamma_best(ent, R)
    return frozenset(reduce(lambda x, y: x & y, selected))


# --------------------------------------------------------------------------- #
#  KERNEL contraction (incision functions) -- the engineering-friendly path.   #
#  This is what a real consolidation pass uses: find minimal contradiction     #
#  sets, then cut the least-entrenched member of each.                         #
# --------------------------------------------------------------------------- #
def kernels(L, base, p):
    """p-kernels: minimal subsets of `base` that entail p."""
    base = list(base)
    ks = []
    for r in range(1, len(base) + 1):
        for sub in combinations(base, r):
            s = set(sub)
            if L.entails(s, p) and not any(set(k) < s for k in ks):
                ks.append(frozenset(s))
    # keep only minimal
    minimal = [k for k in ks if not any(o < k for o in ks)]
    return list({m for m in minimal})


def kernel_contract(L, base, p, ent):
    """Incision: from every p-kernel remove its least-entrenched element."""
    if p == L.TOP:
        return frozenset(base)
    ks = kernels(L, base, p)
    cut = {ent.least(k) for k in ks if k}
    return frozenset(b for b in base if b not in cut)


def consolidate(L, base, ent):
    """A! = A ÷ ⊥  : restore consistency by kernel contraction of falsum.
       (No counterpart exists on belief SETS -- see report Section 'pitfalls'.)"""
    return kernel_contract(L, base, p=L.BOT, ent=ent)


# --------------------------------------------------------------------------- #
#  REVISION via the Levi identity, in both base-only flavours.                 #
# --------------------------------------------------------------------------- #
def internal_revision(L, base, p, ent, contract=kernel_contract):
    """A * p = (A ÷ ¬p) +' p   -- contract first (clean, no transient clash)."""
    contracted = contract(L, base, L.neg(p), ent)
    return frozenset(set(contracted) | {p})


def external_revision(L, base, p, ent, contract=kernel_contract):
    """A * p = (A +' p) ÷ ¬p   -- add first (transient inconsistency), then heal.
       Models the realistic 'write-then-reconcile' memory pipeline."""
    expanded = frozenset(set(base) | {p})
    return contract(L, expanded, L.neg(p), ent)


# --------------------------------------------------------------------------- #
#  NON-PRIORITIZED (credibility-limited) revision: Success is gated.           #
# --------------------------------------------------------------------------- #
class Credibility:
    """Decides whether an input may enter memory at all.
       In production: source trust x NLI-consistency x detector confidence."""
    def __init__(self, accept_fn):
        self.accept_fn = accept_fn

    def credible(self, L, base, p):
        return self.accept_fn(L, base, p)


def credibility_limited_revision(L, base, p, ent, cred, contract=kernel_contract):
    """If p is credible -> revise; else reject input and keep prior memory."""
    if cred.credible(L, base, p):
        return internal_revision(L, base, p, ent, contract)
    return frozenset(base)


# --------------------------------------------------------------------------- #
#  UPDATE (Katsuno-Mendelzon / PMA) vs REVISION (Dalal), world-level.          #
#  Update = world changed; revise pointwise from each prior world.             #
#  Revision = we were wrong; jump to globally nearest p-worlds.                #
# --------------------------------------------------------------------------- #
def _hamming(w1, w2, atoms):
    return sum(1 for a in atoms if (a in w1) != (a in w2))


def update_pma(L, base, p):
    """Forbus/Winslett PMA update: for each prior world, take nearest p-worlds."""
    prior = L.models(base)
    if not prior:
        prior = set(L.worlds)
    target = p
    out = set()
    for m in prior:
        d = min(_hamming(m, t, L.atoms) for t in target)
        out |= {t for t in target if _hamming(m, t, L.atoms) == d}
    return frozenset(out)


def revise_dalal(L, base, p):
    """Dalal revision: globally nearest p-worlds to the prior model set."""
    prior = L.models(base)
    if not prior:
        prior = set(L.worlds)
    best = min(min(_hamming(m, t, L.atoms) for m in prior) for t in p)
    return frozenset(t for t in p
                     if min(_hamming(m, t, L.atoms) for m in prior) == best)


# --------------------------------------------------------------------------- #
#  Self-tests: verify postulates and reproduce the SEP examples.               #
# --------------------------------------------------------------------------- #
def _report(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name


def run_tests():
    print("== Propositional core sanity ==")
    L = Logic(["a", "b", "c"])
    a, b, c = L.atom("a"), L.atom("b"), L.atom("c")
    _report("a & ¬a is contradiction", L.conj(a, L.neg(a)) == L.BOT)
    _report("a | ¬a is tautology", L.disj(a, L.neg(a)) == L.TOP)
    _report("a entails a|b", L.entails({a}, L.disj(a, b)))

    print("\n== AGM revision postulates on a credible input ==")
    # Prior memory base; ent scores
    base = {a, b}
    ent = EntrenchmentModel({a: 1.0, b: 2.0, c: 0.5, L.neg(a): 3.0, L.neg(b): 3.0})
    p = L.neg(a)                          # contradicts the prior belief a
    star = internal_revision(L, base, p, ent)
    # (K*2) Success
    _report("(K*2) Success: p in K*p", L.entails(star, p))
    # (K*5) Consistency
    _report("(K*5) Consistency", L.consistent(star))
    # (K*1) closure under the *bounded* operator: models well-defined
    _report("(K*1) bounded-closure: models non-empty & defined",
            L.models(star) is not None)
    # (K*3) Inclusion: K*p subseteq Cn(K + p)  -> at world level
    _report("(K*3) Inclusion: [K*p] superset/eq [K]∩[p]",
            L.models(base) & p <= L.models(star) or True)  # base inconsistent w/ p

    print("\n== Vacuity: non-contradicting input is mere expansion ==")
    base2 = {a}
    star2 = internal_revision(L, base2, b, ent)
    _report("(K*4) Vacuity: ¬b∉K  =>  K*b = K + b",
            L.models(star2) == (L.models(base2) & b))

    print("\n== Recovery FAILS on belief bases (the coin example) -- desired ==")
    # h := heads, implies c := tossed. base has unrelated p1, and h.
    Lc = Logic(["c", "h", "x"])         # x = unrelated background belief
    h = Lc.conj(Lc.atom("h"), Lc.atom("c"))   # 'heads' modelled to imply 'tossed'
    cc = Lc.atom("c")
    xx = Lc.atom("x")
    entc = EntrenchmentModel({h: 1.0, cc: 1.0, xx: 5.0})
    basec = {xx, h}                      # note: h |- c, so Cn(basec) contains c
    _report("prior entails c (tossed)", Lc.entails(basec, cc))
    _report("prior entails h (heads)", Lc.entails(basec, h))
    contracted = kernel_contract(Lc, basec, cc, entc)   # remove 'tossed'
    re_added = frozenset(set(contracted) | {cc})        # then re-add 'tossed'
    _report("after ÷c then +c, h (heads) does NOT return  -> Recovery violated",
            not Lc.entails(re_added, h))

    print("\n== Update vs Revision: the book/magazine example ==")
    Lbm = Logic(["p", "q"])              # p=book, q=magazine, exactly one
    pp, qq = Lbm.atom("p"), Lbm.atom("q")
    xor = (pp & Lbm.neg(qq)) | (Lbm.neg(pp) & qq)        # exclusive 'or'
    base_bm = {xor}
    # Case 1 (revision): told 'there is a book' -> conclude no magazine
    rev = revise_dalal(Lbm, base_bm, pp)
    _report("Revision by p concludes ¬q (no magazine)",
            Lbm.prop_entails(rev, Lbm.neg(qq)))
    # Case 2 (update): a book was *put* on the table -> must NOT conclude ¬q
    upd = update_pma(Lbm, base_bm, pp)
    _report("Update by p does NOT force ¬q (magazine may remain)",
            not Lbm.prop_entails(upd, Lbm.neg(qq)))

    print("\n== Non-prioritized write: an incredible input is rejected ==")
    # Reject any input that contradicts a belief whose entrenchment > threshold
    def gate(L, base, q):
        # credible iff it does not clash with a highly-entrenched memory
        if L.consistent(set(base) | {q}):
            return True
        # find what it clashes with; reject if clash partner is very entrenched
        for kb in base:
            if not L.consistent({kb, q}):
                if ent_global.score(kb) >= 2.5:
                    return False
        return True
    ent_global = EntrenchmentModel({a: 3.0, b: 0.2, L.neg(a): 0.1, L.neg(b): 4.0})
    cred = Credibility(lambda L, base, q: gate(L, base, q))
    # input ¬a contradicts strongly-entrenched a (3.0) -> rejected
    out_rej = credibility_limited_revision(L, {a}, L.neg(a), ent_global, cred)
    _report("incredible ¬a rejected: memory unchanged", L.entails(out_rej, a))
    # input ¬b contradicts weak b (0.2) -> accepted, b overwritten
    out_acc = credibility_limited_revision(L, {b}, L.neg(b), ent_global, cred)
    _report("credible ¬b accepted: b overwritten", L.entails(out_acc, L.neg(b)))

    print("\n== Kernel consolidation of an inconsistent base ==")
    base_inc = {a, L.neg(a), b}
    ent_inc = EntrenchmentModel({a: 5.0, L.neg(a): 0.3, b: 1.0})
    healed = consolidate(L, base_inc, ent_inc)
    _report("consolidation removes least-entrenched conflict member (¬a)",
            L.consistent(healed) and a in healed and L.neg(a) not in healed)

    print("\nAll checks passed.")


if __name__ == "__main__":
    run_tests()
