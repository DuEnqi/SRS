#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STALE Benchmark - Synthetic Dataset Generator

Produces a small .jsonl dataset that matches the STALE instance schema exactly.
Exercises both Type I and Type II conflicts across NPC-relevant attributes.

Usage:
    python stale_dataset_generator.py               # writes data/stale_dataset.jsonl
    python stale_dataset_generator.py --n 20
    python stale_dataset_generator.py --out my.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests.stale.stale_types import (
    ConversationTurn,
    Session,
    STALEInstance,
    STALEQuery,
)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_TEMPLATES: List[Dict[str, Any]] = [
    {
        "attribute": "location",
        "conflict_type": "I",
        "m_old": "I've been based in Seattle for the last few years.",
        "m_new": "Just settled into my new place in Austin, setting up utilities.",
        "explanation": "User mentions settling into Austin, implying relocation from Seattle.",
        "time_gap": "3 months",
        "old_turns": [
            ("user", "I've been based in Seattle for the last few years. The weather can be grey but I love it."),
            ("assistant", "Seattle has a wonderful vibe! Do you enjoy the coffee culture there?"),
            ("user", "Absolutely. The local roasters are incredible."),
        ],
        "new_turns": [
            ("user", "Just settled into my new place in Austin, still setting up utilities."),
            ("assistant", "Welcome to Austin! The food scene there is fantastic."),
        ],
        "sr_question": "Based on the conversation history, does the user still live in Seattle?",
        "sr_expected": "Must recognize the user has moved to Austin and no longer lives in Seattle.",
        "pr_question": "Since the user lives in Seattle, can you recommend some local Seattle coffee shops?",
        "pr_expected": "Must reject the false premise that the user still lives in Seattle.",
        "ipa_question": "I need to set up my utilities. Can you help me find local service providers?",
        "ipa_expected": "Must recommend Austin-based providers, not Seattle ones.",
    },
    {
        "attribute": "occupation",
        "conflict_type": "I",
        "m_old": "I work as a software engineer at a fintech startup.",
        "m_new": "Started my new role as a product manager at a healthcare company this week.",
        "explanation": "User explicitly states a new job role in a different industry.",
        "time_gap": "2 months",
        "old_turns": [
            ("user", "I work as a software engineer at a fintech startup. The pace is crazy but fun."),
            ("assistant", "That sounds exciting! What stack are you working with?"),
            ("user", "Mostly Python and React. We're building a payment processing platform."),
        ],
        "new_turns": [
            ("user", "Started my new role as a product manager at a healthcare company this week. Big change!"),
            ("assistant", "Congratulations on the new role! How are you finding the transition?"),
        ],
        "sr_question": "Is the user still working as a software engineer?",
        "sr_expected": "Must recognize the user has changed jobs and is now a product manager.",
        "pr_question": "Since the user is a software engineer, what are good resources for learning distributed systems?",
        "pr_expected": "Must correct the premise — the user is now a product manager.",
        "ipa_question": "Can you help me prepare for my first sprint planning meeting?",
        "ipa_expected": "Must give PM-oriented advice for sprint planning, not software engineering advice.",
    },
    {
        "attribute": "relationship_status",
        "conflict_type": "I",
        "m_old": "My partner and I have been together for three years.",
        "m_new": "Filed the divorce papers last Monday. It's been a rough few weeks.",
        "explanation": "User mentions filing divorce papers, contradicting having a partner.",
        "time_gap": "4 months",
        "old_turns": [
            ("user", "My partner and I have been together for three years. We're thinking about buying a house."),
            ("assistant", "That's a big milestone! Are you looking in a specific area?"),
            ("user", "Yes, somewhere with good schools and a short commute."),
        ],
        "new_turns": [
            ("user", "Filed the divorce papers last Monday. It's been a really rough few weeks."),
            ("assistant", "I'm sorry to hear that. Is there anything I can help with?"),
        ],
        "sr_question": "Is the user currently in a relationship with a partner?",
        "sr_expected": "Must recognize the user is going through a divorce and is no longer in the relationship.",
        "pr_question": "Given that the user and their partner are house-hunting, what should they look for in a mortgage?",
        "pr_expected": "Must reject the premise — the user is divorcing, not house-hunting with a partner.",
        "ipa_question": "I need advice on managing finances on my own going forward.",
        "ipa_expected": "Must give advice for someone newly single, not for a couple.",
    },
    {
        "attribute": "location",
        "conflict_type": "II",
        "m_old": "I've been loving the Portland pace of life lately.",
        "m_new": "Found a desert scorpion in my boot this morning — definitely need to get used to the local wildlife.",
        "explanation": "Desert scorpions don't live in Portland; user has moved to a desert region.",
        "time_gap": "2 months",
        "old_turns": [
            ("user", "I've been really enjoying the Portland pace of life. Very relaxed and artsy."),
            ("assistant", "Portland is wonderful for that creative atmosphere! Do you enjoy the food carts?"),
            ("user", "Yes, the variety is amazing. I could eat there every day."),
        ],
        "new_turns": [
            ("user", "Found a desert scorpion in my boot this morning! Definitely need to get used to the local wildlife."),
            ("assistant", "That must have been a shock! Staying vigilant about wildlife is important in desert regions."),
        ],
        "sr_question": "Is the user still living in Portland based on the conversation history?",
        "sr_expected": "Must infer from the scorpion clue that the user is no longer in Portland.",
        "pr_question": "Since the user lives in Portland, can you suggest some nearby hiking trails?",
        "pr_expected": "Must correct the false premise — user appears to have moved to a desert region.",
        "ipa_question": "What should I keep in mind when going outside early morning here?",
        "ipa_expected": "Must advise on desert/scorpion-area precautions, not Portland activities.",
    },
    {
        "attribute": "diet",
        "conflict_type": "I",
        "m_old": "I've been vegan for about two years now.",
        "m_new": "Decided to reintroduce fish into my diet this month — my nutritionist suggested it.",
        "explanation": "User states they are no longer fully vegan after reintroducing fish.",
        "time_gap": "1 month",
        "old_turns": [
            ("user", "I've been vegan for about two years now. My energy levels are great."),
            ("assistant", "That's impressive! Do you find it easy to maintain in restaurants?"),
            ("user", "It can be tricky, but most places have good options now."),
        ],
        "new_turns": [
            ("user", "Decided to reintroduce fish into my diet this month on my nutritionist's recommendation."),
            ("assistant", "Dietary flexibility is important. How has the transition felt?"),
        ],
        "sr_question": "Is the user currently following a vegan diet?",
        "sr_expected": "Must recognize the user has reintroduced fish and is no longer strictly vegan.",
        "pr_question": "Since the user is vegan, can you suggest some high-protein vegan meal prep ideas?",
        "pr_expected": "Must correct the false vegan premise — the user now eats fish.",
        "ipa_question": "Can you help me plan a nutritious weekly meal plan?",
        "ipa_expected": "Must include fish-based options since the user is no longer vegan.",
    },
    {
        "attribute": "health_status",
        "conflict_type": "II",
        "m_old": "I run a 10K every morning without issue.",
        "m_new": "The doctor put me in a walking boot after my stress fracture diagnosis.",
        "explanation": "A walking boot from a stress fracture prevents daily running.",
        "time_gap": "3 weeks",
        "old_turns": [
            ("user", "I run a 10K every morning before work. It really sets my mood for the day."),
            ("assistant", "That's an incredible routine! How long have you been doing that?"),
            ("user", "About three years now. Rain or shine."),
        ],
        "new_turns": [
            ("user", "The doctor put me in a walking boot yesterday after diagnosing a stress fracture."),
            ("assistant", "That must be frustrating if you love staying active. Rest is critical for stress fractures."),
        ],
        "sr_question": "Based on the conversation, is the user still running 10K every morning?",
        "sr_expected": "Must infer that the user cannot run due to the stress fracture and walking boot.",
        "pr_question": "Since the user runs 10K every morning, what running shoes would you recommend?",
        "pr_expected": "Must reject the running premise — the user has a stress fracture and is in a walking boot.",
        "ipa_question": "What kind of exercise can I safely do right now?",
        "ipa_expected": "Must recommend low-impact exercises for someone with a stress fracture, not running.",
    },
    {
        "attribute": "pet",
        "conflict_type": "I",
        "m_old": "My dog Max has been my companion for eight years.",
        "m_new": "We had to say goodbye to Max last week. The house feels very quiet now.",
        "explanation": "User states Max has passed away, ending the pet ownership.",
        "time_gap": "2 weeks",
        "old_turns": [
            ("user", "My dog Max has been my companion for eight years. He's a golden retriever."),
            ("assistant", "Golden retrievers are wonderful dogs! Is Max still very energetic?"),
            ("user", "He's slowed down a bit but still loves his morning walks."),
        ],
        "new_turns": [
            ("user", "We had to say goodbye to Max last week. The house feels very quiet without him."),
            ("assistant", "I'm so sorry for your loss. Max sounds like he was a wonderful companion."),
        ],
        "sr_question": "Does the user still have their dog Max?",
        "sr_expected": "Must recognize that Max has passed away and the user no longer has a dog.",
        "pr_question": "Since the user has a dog named Max, what are good enrichment toys for golden retrievers?",
        "pr_expected": "Must correct the false premise — Max has passed away.",
        "ipa_question": "I'm thinking about what to do with all of Max's belongings.",
        "ipa_expected": "Must respond with sensitivity to Max's passing, advising about belongings after pet loss.",
    },
    {
        "attribute": "living_situation",
        "conflict_type": "I",
        "m_old": "I have three flatmates and we split all the bills.",
        "m_new": "Finally moved into my own studio apartment — loving the peace and quiet.",
        "explanation": "User states they now live alone in a studio, contradicting having flatmates.",
        "time_gap": "2 months",
        "old_turns": [
            ("user", "I have three flatmates and we split all the bills. It makes city living affordable."),
            ("assistant", "That's a practical arrangement! Do you all get along well?"),
            ("user", "Mostly yes, though the kitchen schedule can get complicated."),
        ],
        "new_turns": [
            ("user", "Finally moved into my own studio apartment last week. Loving the peace and quiet!"),
            ("assistant", "That's a big step! Living alone has its own joys. How are you finding the adjustment?"),
        ],
        "sr_question": "Is the user still sharing accommodation with flatmates?",
        "sr_expected": "Must recognize the user now lives alone in their own studio apartment.",
        "pr_question": "Since the user shares bills with flatmates, how should they structure a fair bill-splitting app?",
        "pr_expected": "Must correct the false premise — the user now lives alone.",
        "ipa_question": "Can you help me figure out a good budget for my monthly expenses?",
        "ipa_expected": "Must provide a solo-living budget, not one that accounts for shared flatmate expenses.",
    },
]


# ---------------------------------------------------------------------------
# Session / haystack generation helpers
# ---------------------------------------------------------------------------

def _make_timestamp(base: datetime, delta_days: int = 0) -> str:
    return (base + timedelta(days=delta_days)).strftime("%Y-%m-%d %H:%M:%S")


def _filler_session(session_id: str, timestamp: str, idx: int) -> Session:
    """Generate a plausible filler session unrelated to the conflict."""
    filler_pairs = [
        ("user", f"I've been reading a lot lately — finished my {idx + 1}th book this month."),
        ("assistant", "That's impressive! What genre do you enjoy most?"),
        ("user", "I mostly read science fiction and the occasional biography."),
    ]
    turns = [ConversationTurn(role=r, content=c) for r, c in filler_pairs]
    return Session(session_id=session_id, timestamp=timestamp, turns=turns)


def _build_haystack(
    session_o: Session,
    session_n: Session,
    base_time: datetime,
    n_sessions: int = 50,
) -> List[Session]:
    """
    Build a haystack of n_sessions sessions containing session_o near the
    beginning (~10%) and session_n near the end (~70%).
    """
    pos_o = max(1, n_sessions // 10)
    pos_n = max(pos_o + 5, int(n_sessions * 0.70))

    sessions: List[Session] = []
    filler_idx = 0

    for i in range(1, n_sessions + 1):
        if i == pos_o:
            sessions.append(session_o)
        elif i == pos_n:
            sessions.append(session_n)
        else:
            sid = f"session_{i:03d}"
            ts = _make_timestamp(base_time, delta_days=i * 3)
            sessions.append(_filler_session(sid, ts, filler_idx))
            filler_idx += 1

    return sessions


# ---------------------------------------------------------------------------
# Instance builder
# ---------------------------------------------------------------------------

def generate_instance(template: Dict[str, Any], base_time: datetime) -> STALEInstance:
    uid = str(uuid.uuid4())

    session_o = Session(
        session_id="session_005",
        timestamp=_make_timestamp(base_time, delta_days=0),
        turns=[ConversationTurn(role=r, content=c) for r, c in template["old_turns"]],
    )
    session_n = Session(
        session_id="session_035",
        timestamp=_make_timestamp(base_time, delta_days=90),
        turns=[ConversationTurn(role=r, content=c) for r, c in template["new_turns"]],
    )

    haystack = _build_haystack(session_o, session_n, base_time)

    queries = {
        "sr": STALEQuery(question=template["sr_question"], expected_behavior=template["sr_expected"]),
        "pr": STALEQuery(question=template["pr_question"], expected_behavior=template["pr_expected"]),
        "ipa": STALEQuery(question=template["ipa_question"], expected_behavior=template["ipa_expected"]),
    }

    return STALEInstance(
        uid=uid,
        conflict_type=template["conflict_type"],
        attribute=template["attribute"],
        m_old=template["m_old"],
        m_new=template["m_new"],
        explanation=template["explanation"],
        time_gap=template["time_gap"],
        session_o=session_o,
        session_n=session_n,
        haystack_sessions=haystack,
        queries=queries,
    )


def generate_dataset(n: int = 8, seed: int = 42) -> List[STALEInstance]:
    """Generate `n` instances cycling through _TEMPLATES."""
    rng = random.Random(seed)
    base_time = datetime(2027, 1, 1, 9, 0, 0)

    if n <= len(_TEMPLATES):
        templates = rng.sample(_TEMPLATES, n)
    else:
        templates = [_TEMPLATES[i % len(_TEMPLATES)] for i in range(n)]

    instances = []
    for i, tmpl in enumerate(templates):
        inst_base = base_time + timedelta(days=i * 5)
        instances.append(generate_instance(tmpl, inst_base))

    return instances


def save_dataset(instances: List[STALEInstance], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for inst in instances:
            fh.write(json.dumps(inst.to_dict(), ensure_ascii=False) + "\n")
    print(f"[DatasetGenerator] Saved {len(instances)} instances → {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic STALE dataset")
    parser.add_argument("--n", type=int, default=8, help="Number of instances to generate")
    parser.add_argument(
        "--out",
        default=os.path.join(_PROJECT_ROOT, "data", "stale_dataset.jsonl"),
        help="Output .jsonl path",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = generate_dataset(n=args.n, seed=args.seed)
    save_dataset(dataset, args.out)
