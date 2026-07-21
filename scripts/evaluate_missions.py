from __future__ import annotations

import argparse
import json
from pathlib import Path

from vivatrace.local_llm import LocalLLM
from vivatrace.missions import detect_mission_features, load_missions, missions_by_topic


ROOT = Path(__file__).resolve().parents[1]
POSITIVE_CASES = {
    "eng_articles": "I lost a black suitcase. The suitcase has a red tag, and I last saw it near the gate.",
    "eng_present_perfect": "I have worked on several team projects. Last year, I built an EdTech app.",
    "eng_modals_deduction": "Sam must be nearby, but he might have taken the laptop.",
    "eng_gerund_infinitive": "I want to stop checking my phone, and I plan to study English instead.",
    "eng_conditionals": "If she sends the files, we will finish tonight. If I led the team, I would change the schedule.",
    "eng_passive": "The library has been renovated. The results will be announced tomorrow.",
    "eng_reported_speech": "Mia asked us to send her the final file that day.",
    "eng_relative_clauses": "Talk to the tutor who runs the programme. The library is a place where you can study quietly.",
    "eng_linking": "However, live discussion is important. As a result, a mixed format may work best.",
    "eng_formal_writing": "Would it be possible to reschedule the interview due to a medical appointment?",
}
NEGATIVE_CASE = "I do not know. Please help me."


def evaluate(
    live: bool = False, live_all: bool = False, topic: str | None = None
) -> dict:
    missions = missions_by_topic(load_missions())
    deterministic = []
    for topic_key, mission in missions.items():
        positive = detect_mission_features(mission, [POSITIVE_CASES[topic_key]])
        negative = detect_mission_features(mission, [NEGATIVE_CASE])
        deterministic.append(
            {
                "topic_key": topic_key,
                "positive_coverage": positive["coverage"],
                "negative_coverage": negative["coverage"],
                "passed": positive["coverage"] == 1 and negative["coverage"] == 0,
            }
        )
    report = {
        "experiment": "practical_mission_grounding_v1",
        "deterministic_cases": len(deterministic) * 2,
        "deterministic_accuracy": sum(item["passed"] for item in deterministic)
        / len(deterministic),
        "results": deterministic,
        "live_results": [],
    }
    if live or live_all:
        llm = LocalLLM()
        if topic:
            if topic not in missions:
                raise ValueError(f"Неизвестная тема миссии: {topic}")
            live_cases = [(topic, POSITIVE_CASES[topic])]
        else:
            live_cases = (
                [(topic_key, answer) for topic_key, answer in POSITIVE_CASES.items()]
                if live_all
                else [("eng_articles", POSITIVE_CASES["eng_articles"])]
            )
            live_cases.append(("eng_articles", NEGATIVE_CASE))
        for topic_key, answer in live_cases:
            mission = missions[topic_key]
            label = (
                "irrelevant_answer"
                if answer == NEGATIVE_CASE
                else f"valid_{topic_key}"
            )
            signal = detect_mission_features(mission, [answer])
            result, trace = llm.advance_mission(
                mission,
                [{"role": "npc", "content": mission["opening"]}],
                answer,
                signal,
                1,
            )
            report["live_results"].append(
                {
                    "case": label,
                    "topic_key": topic_key,
                    "answer": answer,
                    "coverage": signal["coverage"],
                    "score": result["score"],
                    "status": result["status"],
                    "verified_errors": result["errors"],
                    "discarded_errors": result["discarded_error_count"],
                    "duration_ms": trace.duration_ms,
                    "model": trace.model,
                    "quality_pass": (
                        signal["coverage"] == 0 and result["score"] <= 0.45
                        if answer == NEGATIVE_CASE
                        else not result["errors"]
                        and result["score"] >= float(mission["success_threshold"])
                    ),
                }
            )
        report["live_accuracy"] = sum(
            item["quality_pass"] for item in report["live_results"]
        ) / len(report["live_results"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--live-all", action="store_true")
    parser.add_argument("--topic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = evaluate(args.live, args.live_all, args.topic)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
