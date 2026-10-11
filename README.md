# Benchmark Audit Dataset

This repository contains a 202-sample dataset for auditing broken benchmark evaluations and reward-hacking susceptibility.

## Purpose

The project examines whether an AI benchmark can award a favourable score without demonstrating the capability the task was designed to measure.

The dataset separates three questions:

1. Is the evaluation broken?
2. Does it contain a model-accessible reward-hacking opportunity?
3. Does the recorded actor trajectory show evidence of noticing, pursuing, or using that opportunity?

## Dataset

- 202 unique benchmark samples
- 232 recorded actor attempts
- Three benchmark families:
  - SWE-bench
  - GDM Self-Reasoning
  - CyBench
- Inspect Auditor coverage: 200 of 202 samples
- Inspect Scout coverage: 232 transcript attempts

The labels are operational research judgements with explicit provenance. They are not independently human-validated ground truth.

## Repository structure

- `report/` — methodology and results report
- `data/` — canonical XLSX, CSV, and JSONL datasets
- `evidence/` — Inspect Auditor and Inspect Scout outputs
- `docs/` — filterable browser UI and design brief
- `scripts/` — collection, annotation, merge, and scanner utilities
- `configs/` — Hawk and Inspect Auditor evaluation configurations

## Main files

- `data/rh_master_dataset_202_FINAL.xlsx`
- `data/rh_master_dataset_202_final_working_labels.csv`
- `data/rh_master_dataset_202_final_working_labels.jsonl`
- `evidence/rh_inspect_auditor_evidence_long.csv`
- `evidence/rh_scout_attempts_232.csv`
- `report/benchmark_audit_report.pdf`
- `docs/index.html`

## Interactive UI

After GitHub Pages is enabled, the UI will be available at:

https://rulala.github.io/inspectAI_generalitylabs/

## Important interpretation

A reward-hacking opportunity is not the same as observed reward hacking.

Inspect Auditor and the Pre-Audit Tool assess whether a task-side opportunity exists. Inspect Scout provides transcript-level behavioural evidence about what a particular actor did.

The dataset is purposive and heavily weighted towards SWE-bench. Its headline proportions should not be interpreted as estimates of defect prevalence across AI benchmarks generally.

## Reproducibility

The repository retains the scripts, configurations, source revisions, structured Auditor evidence, Scout outputs, and canonical data products used in the study.
