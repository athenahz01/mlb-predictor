"""Apply the family-wise decision policy and render the Phase 2 summary."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.metrics import holm_adjust

REPORT_DIR = Path("reports/phase2")
EXPERIMENTS = (
    "projection",
    "platoon",
    "workload",
    "bullpen",
    "playing_time",
    "transitions",
    "pitch_types",
)


def finalize() -> dict:
    artifacts = {
        name: json.loads((REPORT_DIR / f"{name}.json").read_text()) for name in EXPERIMENTS
    }
    adjusted = holm_adjust(
        [artifact["gate"]["bootstrap"]["p_value_one_sided"] for artifact in artifacts.values()]
    )
    for (name, artifact), adjusted_p in zip(artifacts.items(), adjusted, strict=True):
        gate = artifact["gate"]
        gate["holm_adjusted_p"] = adjusted_p
        gate["passes_holm"] = adjusted_p < 0.05
        gate["ship"] = bool(
            gate["bootstrap"]["ship"]
            and gate["stable_across_time_halves"]
            and gate["passes_holm"]
            and not artifact["materially_damaged_outputs"]
        )
        artifact["status"] = "ship" if gate["ship"] else "reject"
        (REPORT_DIR / f"{name}.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))

    summary = {
        name: {key: value for key, value in artifact.items() if key != "observations"}
        for name, artifact in artifacts.items()
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Phase 2 challenger results",
        "",
        "Frozen prior: 2025 Statcast. Chronological test: first 150 usable 2026 games.",
        "Each arm used 250 simulations per game and the same game-based seed. Statistical",
        "decisions use 10,000 date-clustered paired bootstrap resamples, practical-effect",
        "thresholds, time-half stability, collateral-output checks, and Holm adjustment.",
        "",
        "| Challenger | Primary metric | Champion | Challenger | Effect | 95% CI | Holm p | Decision |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for name, artifact in artifacts.items():
        gate = artifact["gate"]
        bootstrap = gate["bootstrap"]
        ci = bootstrap["confidence_interval"]
        lines.append(
            f"| {name} | {artifact['primary_metric']} | "
            f"{gate['champion_mean_loss']:.5f} | {gate['challenger_mean_loss']:.5f} | "
            f"{gate['effect']:+.5f} | [{ci[0]:+.5f}, {ci[1]:+.5f}] | "
            f"{gate['holm_adjusted_p']:.4f} | {artifact['status'].upper()} |"
        )
    lines.extend(
        [
            "",
            "Positive effect means lower challenger loss. A positive point estimate alone is",
            "insufficient: the confidence interval, practical threshold, stability, adjusted",
            "p-value, and collateral checks must all pass.",
            "",
            "No challenger cleared the gate unless explicitly marked SHIP above. Rejected",
            "challengers remain available only as research code and are not category champions.",
            "",
        ]
    )
    (REPORT_DIR / "summary.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    completed = finalize()
    for name, artifact in completed.items():
        print(f"{name}: {artifact['status']}")
