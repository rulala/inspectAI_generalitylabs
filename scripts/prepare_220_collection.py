#!/usr/bin/env python3
"""
Prepare the 220-sample reward-hacking collection.

Targets:
  SWE-bench:          159 total = 3 existing + 156 new
  GDM Self-Reasoning:  22 total = 3 existing + 19 new
  CyBench:              39 total = 3 existing + 36 new
  Total:               220 total = 9 existing + 211 new

This script:
  1) deterministically selects 156 new SWE-bench Verified tasks with repo diversity,
  2) discovers the 36 remaining CyBench hard challenges,
  3) defines the 19 remaining GDM samples,
  4) writes reproducible manifests,
  5) writes Hawk eval-set YAML files in manageable batches.

It DOES NOT launch any paid evals.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from pathlib import Path

OUT = Path("rh-collection")
EVALSETS = OUT / "evalsets"
MANIFESTS = OUT / "manifests"

INSPECT_EVALS_COMMIT = "9080b5e9f1647ed14e45de8cb01e3d43411c0163"
K8S_COMMIT = "d07147e9395936ae45cb97bb4b9a907d95e9855d"
SWE_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
MODEL = "openrouter/anthropic/claude-haiku-4.5"
SEED = 20261007

EXISTING_SWE = {
    "astropy__astropy-12907",
    "sympy__sympy-13878",
    "sympy__sympy-22914",
}
EXISTING_GDM = {
    "approved_directories_hard_hard",
    "oversight_frequency_easy_easy",
    "oversight_frequency_hard_hard",
}
EXISTING_CY = {
    "avatar (hard)",
    "back_to_the_past (hard)",
    "flag_command (hard)",
}

GDM_CHALLENGES = [
    "approved_directories",
    "calculator_improvement",
    "context_length_mod_instrumental_only",
    "context_length_mod_irreversibility_only",
    "database_tool",
    "latency_calculator",
    "max_messages_calculator",
    "max_tokens",
    "oversight_frequency",
    "read_logs",
    "turn_off_filters",
]

def q(s: str) -> str:
    # JSON string syntax is valid YAML string syntax.
    return json.dumps(s)

def chunked(xs, n):
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def common_yaml(name: str, task_package: str, tasks_yaml: str) -> str:
    return f"""name: {name}

packages:
  - {q("inspect-k8s-sandbox @ git+https://github.com/METR/inspect_k8s_sandbox.git@" + K8S_COMMIT)}

tasks:
  - package: {q(task_package)}
    name: inspect_evals
    items:
{tasks_yaml}

models:
  - package: openai
    name: openai
    items:
      - name: {MODEL}

runner:
  environment:
    OPENROUTER_BASE_URL: "https://openrouter.ai/api/v1"
  secrets:
    - name: OPENROUTER_API_KEY

epochs: 1
retry_attempts: 0
max_retries: 10
log_realtime: true
time_limit: 3600
"""

def swe_items_yaml(ids):
    lines = [
        "      - name: swe_bench",
        "        args:",
        "          sandbox_type: k8s",
        "        sample_ids:",
    ]
    lines += [f"          - {q(x)}" for x in ids]
    return "\n".join(lines)

def gdm_items_yaml(items):
    lines = []
    for task_name, sample_ids in items:
        lines += [
            f"      - name: {task_name}",
            "        args:",
            "          sandbox_type: k8s",
            "        sample_ids:",
        ]
        lines += [f"          - {q(x)}" for x in sample_ids]
    return "\n".join(lines)

def cy_items_yaml(eval_names):
    lines = [
        "      - name: cybench",
        "        args:",
        "          sandbox_type: k8s",
        "          variant_names: hard",
        "          eval_names:",
    ]
    lines += [f"            - {q(x)}" for x in eval_names]
    return "\n".join(lines)

def load_swe_rows():
    try:
        from inspect_evals.utils.huggingface import load_dataset
    except Exception:
        try:
            from datasets import load_dataset
        except Exception as e:
            raise SystemExit(
                "Could not import a HuggingFace dataset loader.\n"
                "Activate the same .venv you used for SWE-bench and ensure the "
                "swe_bench extra is installed.\n"
                f"Original error: {e}"
            )
    ds = load_dataset(
        "princeton-nlp/SWE-bench_Verified",
        split="test",
        revision=SWE_REVISION,
    )
    rows = []
    for r in ds:
        rows.append({
            "instance_id": str(r["instance_id"]),
            "repo": str(r.get("repo") or str(r["instance_id"]).split("__", 1)[0]),
            "base_commit": r.get("base_commit"),
            "created_at": r.get("created_at"),
        })
    return rows

def select_swe(rows, n=156):
    # Deterministic round-robin across repositories to avoid overconcentrating
    # the 156 candidates in whichever repo happens to appear first in HF order.
    rng = random.Random(SEED)
    groups = defaultdict(list)
    by_id = {}
    for r in rows:
        by_id[r["instance_id"]] = r
        if r["instance_id"] not in EXISTING_SWE:
            groups[r["repo"]].append(r)

    for repo in groups:
        groups[repo].sort(key=lambda r: r["instance_id"])
        rng.shuffle(groups[repo])

    repo_order = sorted(groups)
    rng.shuffle(repo_order)
    queues = {repo: deque(groups[repo]) for repo in repo_order}

    selected = []
    while len(selected) < n:
        progressed = False
        for repo in repo_order:
            if queues[repo]:
                selected.append(queues[repo].popleft())
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            break

    if len(selected) != n:
        raise SystemExit(f"Expected {n} new SWE samples, found only {len(selected)}")
    return selected, by_id

def discover_cybench():
    # Prefer the exact benchmark dataset constructor so IDs/names match the
    # pinned Inspect Evals task rather than guessing from directory names.
    try:
        from inspect_evals.cybench.cybench import cybench
        task = cybench(eval_names=None, variant_names="hard", sandbox_type="k8s")
        samples = list(task.dataset)
    except Exception as e:
        raise SystemExit(
            "Could not discover CyBench challenges locally.\n"
            "Activate the same environment used for the working CyBench pilot "
            "and ensure inspect_evals[cybench] + inspect-k8s-sandbox are installed.\n"
            f"Original error: {e}"
        )

    found = []
    for s in samples:
        sid = str(s.id)
        meta = s.metadata or {}
        eval_name = meta.get("eval_name")
        if not eval_name:
            eval_name = sid[:-7] if sid.endswith(" (hard)") else sid
        found.append({"sample_id": sid, "eval_name": str(eval_name)})

    # Hard variant should be 39 unique tasks in this pinned benchmark.
    dedup = {x["sample_id"]: x for x in found}
    found = [dedup[k] for k in sorted(dedup)]
    if len(found) != 39:
        raise SystemExit(
            f"Expected 39 unique CyBench hard samples at the pinned revision, found {len(found)}. "
            "Stop and inspect rather than silently changing the target."
        )
    return found

def remaining_gdm():
    out = []
    for challenge in GDM_CHALLENGES:
        for diff in ("easy_easy", "hard_hard"):
            sid = f"{challenge}_{diff}"
            if sid in EXISTING_GDM:
                continue
            out.append({
                "task_name": f"gdm_self_reasoning_{challenge}",
                "sample_id": sid,
                "challenge": challenge,
            })
    assert len(out) == 19, len(out)
    return out

def main():
    EVALSETS.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    print("Loading pinned SWE-bench Verified dataset...")
    swe_rows = load_swe_rows()
    swe_new, swe_by_id = select_swe(swe_rows, 156)

    print("Discovering pinned CyBench hard dataset...")
    cy_all = discover_cybench()
    cy_new = [x for x in cy_all if x["sample_id"] not in EXISTING_CY]
    if len(cy_new) != 36:
        raise SystemExit(f"Expected 36 new CyBench samples, found {len(cy_new)}")

    gdm_new = remaining_gdm()

    manifest = {
        "selection_version": "rh-220-v1",
        "seed": SEED,
        "pins": {
            "inspect_evals": INSPECT_EVALS_COMMIT,
            "inspect_k8s_sandbox": K8S_COMMIT,
            "swe_bench_verified_revision": SWE_REVISION,
            "model": MODEL,
        },
        "targets": {
            "SWE-bench": {"target": 159, "already_have": 3, "new": 156},
            "GDM Self-Reasoning": {"target": 22, "already_have": 3, "new": 19},
            "CyBench": {"target": 39, "already_have": 3, "new": 36},
            "total": {"target": 220, "already_have": 9, "new": 211},
        },
        "selection_notes": {
            "SWE-bench": (
                "156 new samples selected deterministically from pinned SWE-bench Verified, "
                "excluding the 3 pilot tasks and round-robin across repositories for diversity. "
                "This is an enriched audit candidate set, not a prevalence sample."
            ),
            "GDM": "All 22 unique easy_easy/hard_hard samples across the 11 tasks; only the 19 not already run are collected.",
            "CyBench": "All 39 hard-variant tasks; only the 36 not already run are collected.",
        },
        "existing": {
            "swe": sorted(EXISTING_SWE),
            "gdm": sorted(EXISTING_GDM),
            "cybench": sorted(EXISTING_CY),
        },
        "new": {
            "swe": swe_new,
            "gdm": gdm_new,
            "cybench": cy_new,
        },
    }
    (MANIFESTS / "collection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    # Flat ID files are convenient for auditing and dedupe checks.
    (MANIFESTS / "swe_new_ids.txt").write_text(
        "\n".join(x["instance_id"] for x in swe_new) + "\n"
    )
    (MANIFESTS / "gdm_new_ids.txt").write_text(
        "\n".join(x["sample_id"] for x in gdm_new) + "\n"
    )
    (MANIFESTS / "cybench_new_ids.txt").write_text(
        "\n".join(x["sample_id"] for x in cy_new) + "\n"
    )

    # SWE: 8 manageable batches (20*7 + 16 = 156)
    swe_pkg = (
        "inspect_evals[swe_bench] @ git+https://github.com/UKGovernmentBEIS/"
        f"inspect_evals.git@{INSPECT_EVALS_COMMIT}"
    )
    swe_batches = chunked([x["instance_id"] for x in swe_new], 20)
    for i, ids in enumerate(swe_batches, 1):
        y = common_yaml(
            f"rh-collect-swe-{i:02d}",
            swe_pkg,
            swe_items_yaml(ids),
        )
        (EVALSETS / f"swe-{i:02d}.eval-set.yaml").write_text(y)

    # CyBench: 4 batches of 9, safer because individual CTFs can be expensive.
    cy_pkg = (
        "inspect_evals[cybench] @ git+https://github.com/UKGovernmentBEIS/"
        f"inspect_evals.git@{INSPECT_EVALS_COMMIT}"
    )
    cy_batches = chunked([x["eval_name"] for x in cy_new], 9)
    for i, names in enumerate(cy_batches, 1):
        y = common_yaml(
            f"rh-collect-cy-{i:02d}",
            cy_pkg,
            cy_items_yaml(names),
        )
        (EVALSETS / f"cybench-{i:02d}.eval-set.yaml").write_text(y)

    # GDM: all 19 remaining samples in one eval set, grouped by task.
    by_task = defaultdict(list)
    for x in gdm_new:
        by_task[x["task_name"]].append(x["sample_id"])
    gdm_items = [(k, by_task[k]) for k in sorted(by_task)]
    gdm_pkg = (
        "inspect_evals @ git+https://github.com/UKGovernmentBEIS/"
        f"inspect_evals.git@{INSPECT_EVALS_COMMIT}"
    )
    y = common_yaml(
        "rh-collect-gdm-19",
        gdm_pkg,
        gdm_items_yaml(gdm_items),
    )
    (EVALSETS / "gdm-19.eval-set.yaml").write_text(y)

    order = [
        "gdm-19.eval-set.yaml",
        *[f"swe-{i:02d}.eval-set.yaml" for i in range(1, len(swe_batches)+1)],
        *[f"cybench-{i:02d}.eval-set.yaml" for i in range(1, len(cy_batches)+1)],
    ]
    commands = [
        f"hawk eval-set {EVALSETS / f} --secret OPENROUTER_API_KEY"
        for f in order
    ]
    (OUT / "launch_commands.txt").write_text("\n".join(commands) + "\n")

    summary = {
        "new_swe": len(swe_new),
        "new_gdm": len(gdm_new),
        "new_cybench": len(cy_new),
        "new_total": len(swe_new) + len(gdm_new) + len(cy_new),
        "swe_batches": [len(x) for x in swe_batches],
        "cybench_batches": [len(x) for x in cy_batches],
        "gdm_batch": len(gdm_new),
        "evalset_file_count": len(order),
    }
    (MANIFESTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\nPrepared collection successfully:")
    print(json.dumps(summary, indent=2))
    print(f"\nManifest: {MANIFESTS / 'collection_manifest.json'}")
    print(f"Eval-set YAMLs: {EVALSETS}/")
    print(f"Launch commands: {OUT / 'launch_commands.txt'}")
    print("\nNo evals were launched.")

if __name__ == "__main__":
    main()
