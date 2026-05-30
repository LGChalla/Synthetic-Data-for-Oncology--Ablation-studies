"""
ablation_gate_decomposition.py
==============================
TCBB Ablation 3 — Gate Decomposition

Research question
-----------------
Which component of G(x) = S ∧ O ∧ C contributes most to downstream
adapter quality? Is schema compliance alone sufficient, or does each
additional constraint (SNOMED ontology, AJCC clinical logic) provide
independent, measurable training-signal improvements?

Design
------
Four matched corpora, identical generation setup, progressively stricter gates:

    Corpus-S       — Schema only:            JSON completeness (S)
    Corpus-SO      — Schema + Ontology:      S ∧ SNOMED CT coverage (O)
    Corpus-SOC     — Full gate:              S ∧ O ∧ AJCC logic (C)
                     (this is the BIBM golden corpus equivalent)

Four adapters trained (n=162 each):
    Adapter-S, Adapter-SO, Adapter-SOC, Baseline (zero-shot)

Controlled variables: TNM grid, models, QLoRA hyperparams, seed, corpus size.
Independent variable: which gate components are active (S / S∧O / S∧O∧C).
Dependent variables: per-class T-stage accuracy, macro-F1, SNOMED density,
                     per-component gate pass rates.

Expected result: monotonically improving downstream accuracy as constraints
are added, with the largest gain at either the O or C boundary, quantifying
which clinical constraint is most critical for training signal quality.

Output structure
----------------
results/ablation_gate_decomp/
  corpus_S.jsonl   / corpus_SO.jsonl   / corpus_SOC.jsonl
  golden_S.csv     / golden_SO.csv     / golden_SOC.csv
  pass_rates.csv
  snomed_density.csv
  evaluation_results.csv
  experiment.log
  adapter_S/final_adapter/
  adapter_SO/final_adapter/
  adapter_SOC/final_adapter/
"""

import sys, json, gc, time, argparse
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stagecraft"))
sys.path.insert(0, str(ROOT / "ablations"))

from shared_utils import (
    MODELS, TARGET_N, SEED,
    run_generation_loop, records_to_golden_csv,
    audit_diversity, snomed_density,
    train_adapter, evaluate_adapter, log_experiment,
)
from pipeline.phase1_datagen import (
    load_hf_model, build_controlled_case, BASE_SCHEMA,
)

# ── Output paths ──────────────────────────────────────────────────────────────
OUT         = ROOT / "results" / "ablation_gate_decomp"
LOG_FILE    = OUT / "experiment.log"
EVAL_CSV    = OUT / "evaluation_results.csv"
PASSRATE_CSV= OUT / "pass_rates.csv"
SNOMED_CSV  = OUT / "snomed_density.csv"

OUT.mkdir(parents=True, exist_ok=True)

# ── Gate configurations (ordered by strictness) ───────────────────────────────
# Each entry: (label, gate_kwargs, output files)
GATE_CONFIGS = [
    (
        "S",
        dict(use_schema=True,  use_ontology=False, use_logic=False),
        OUT / "corpus_S.jsonl",
        OUT / "golden_S.csv",
        OUT / "adapter_S",
    ),
    (
        "S+O",
        dict(use_schema=True,  use_ontology=True,  use_logic=False),
        OUT / "corpus_SO.jsonl",
        OUT / "golden_SO.csv",
        OUT / "adapter_SO",
    ),
    (
        "S+O+C",
        dict(use_schema=True,  use_ontology=True,  use_logic=True),
        OUT / "corpus_SOC.jsonl",
        OUT / "golden_SOC.csv",
        OUT / "adapter_SOC",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT (same for all conditions — only gate differs)
# ─────────────────────────────────────────────────────────────────────────────

def base_prompt(t: str, n: str, m: str) -> str:
    return (
        f"SCHEMA:\n{json.dumps(BASE_SCHEMA, indent=2)}\n\n"
        f"CASE:\n{build_controlled_case(t, n, m)}\n\n"
        "Map this case to the schema exactly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# GATE PASS RATE TABLE
# ─────────────────────────────────────────────────────────────────────────────

def compute_pass_rates(all_corpus_records: dict[str, list[dict]]) -> pd.DataFrame:
    """
    For each corpus, computes:
      - Yield rate: valid accepted / total generated attempts (implicit)
      - Per-component analysis of gate_detail
      - SNOMED density mean per condition

    This is the key table for the decomposition argument:
    each row shows what you lose/gain by adding a constraint.
    """
    rows = []
    for condition, records in all_corpus_records.items():
        n_records  = len(records)
        s_rates, o_rates, c_rates = [], [], []
        for rec in records:
            detail = rec.get("gate_detail", {})
            # True = passed, False = failed, "skipped" = not evaluated
            s_rates.append(1 if detail.get("S") is True  else
                           0 if detail.get("S") is False else None)
            o_rates.append(1 if detail.get("O") is True  else
                           0 if detail.get("O") is False else None)
            c_rates.append(1 if detail.get("C") is True  else
                           0 if detail.get("C") is False else None)

        def safe_mean(lst):
            valid = [x for x in lst if x is not None]
            return round(sum(valid) / len(valid), 3) if valid else None

        rows.append({
            "condition":   condition,
            "n_records":   n_records,
            "S_pass_rate": safe_mean(s_rates),
            "O_pass_rate": safe_mean(o_rates),
            "C_pass_rate": safe_mean(c_rates),
        })

    df = pd.DataFrame(rows)
    print("\n── Gate Component Pass Rates ──")
    print(df.to_string(index=False))
    print("\n  Interpretation: each rate shows what fraction of *accepted* records")
    print("  pass that component. Lower O/C rates in S-only corpus reveal how many")
    print("  schema-valid records carry ontological errors or AJCC inconsistencies.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("TCBB ABLATION 3 — Gate Decomposition")
    print(f"Models      : {MODELS}")
    print(f"Target N    : {args.n_target} per condition")
    print(f"Output      : {OUT}")
    print(f"Conditions  : {[cfg[0] for cfg in GATE_CONFIGS]}")
    print("=" * 60)

    log_experiment(LOG_FILE, {
        "event": "start", "ablation": "gate_decomposition",
        "n_target": args.n_target, "models": MODELS,
        "conditions": [cfg[0] for cfg in GATE_CONFIGS],
    })

    # Storage for all corpora
    all_records: dict[str, list[dict]] = {cfg[0]: [] for cfg in GATE_CONFIGS}
    per_model = max(1, args.n_target // len(MODELS))

    # ── Step 1: Generation (one pass per model, all three gate configs) ───────
    if not args.skip_gen:
        for model_id in MODELS:
            print(f"\n{'='*60}\nMODEL: {model_id}\n{'='*60}")
            hf_arts = None
            try:
                hf_arts = load_hf_model(model_id)

                for label, gate_kwargs, jsonl_path, _, _ in GATE_CONFIGS:
                    # Use different seeds per config to avoid identical outputs
                    seed_offset = {"S": 0, "S+O": 200, "S+O+C": 400}
                    recs = run_generation_loop(
                        model_id, hf_arts, jsonl_path,
                        prompt_fn   = base_prompt,
                        gate_kwargs = gate_kwargs,
                        n_target    = per_model,
                        seed        = SEED + seed_offset.get(label, 0),
                        condition   = label,
                    )
                    all_records[label].extend(recs)
                    log_experiment(LOG_FILE, {
                        "event": "generation_done", "model": model_id,
                        "condition": label, "n_valid": len(recs),
                    })

            except Exception as e:
                import traceback
                print(f"[ERROR] {model_id}: {e}")
                traceback.print_exc()
                log_experiment(LOG_FILE, {"event": "error", "model": model_id, "msg": str(e)})
            finally:
                del hf_arts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(5)
    else:
        print("[SKIP] Loading existing JSONL files...")
        for label, _, jsonl_path, _, _ in GATE_CONFIGS:
            if jsonl_path.exists():
                with open(jsonl_path) as f:
                    for line in f:
                        if line.strip():
                            try:
                                all_records[label].append(json.loads(line))
                            except Exception:
                                pass
                print(f"  [{label}] Loaded {len(all_records[label])} records")

    # ── Step 2: Gate pass rate analysis ───────────────────────────────────────
    df_passrate = compute_pass_rates(all_records)
    df_passrate.to_csv(PASSRATE_CSV, index=False)
    print(f"  Saved → {PASSRATE_CSV}")
    log_experiment(LOG_FILE, {"event": "passrate_done", "data": df_passrate.to_dict()})

    # ── Step 3: SNOMED density comparison ─────────────────────────────────────
    print("\n── SNOMED Density Comparison ──")
    df_snomed = pd.concat([
        snomed_density(recs, label)
        for label, recs in all_records.items()
    ], ignore_index=True)
    summary = df_snomed.groupby("condition")["density_per100"].agg(["mean","std","count"])
    print(summary.to_string())
    df_snomed.to_csv(SNOMED_CSV, index=False)
    print(f"  Saved → {SNOMED_CSV}")

    # ── Step 4: Golden CSVs + diversity audits ────────────────────────────────
    print("\n── Exporting Golden CSVs + Diversity Audits ──")
    golden_dfs: dict[str, pd.DataFrame] = {}
    for label, _, _, csv_path, _ in GATE_CONFIGS:
        df = records_to_golden_csv(all_records[label], csv_path)
        golden_dfs[label] = df
        audit_diversity(df, f"{label} corpus")

    # ── Step 5: Adapter training ──────────────────────────────────────────────
    if not args.skip_train:
        for label, _, _, csv_path, adapter_dir in GATE_CONFIGS:
            train_adapter(str(csv_path), str(adapter_dir), f"Adapter-{label}")
            log_experiment(LOG_FILE, {"event": "training_done", "adapter": label})
    else:
        print("[SKIP] Adapter training.")

    # ── Step 6: TSTR Evaluation ───────────────────────────────────────────────
    print("\n── TSTR Evaluation ──")
    all_results = []

    # Baseline always first
    rows = evaluate_adapter(None, "Baseline (zero-shot)", OUT)
    all_results.extend(rows)

    for label, _, _, _, adapter_dir in GATE_CONFIGS:
        rows = evaluate_adapter(str(adapter_dir), f"Adapter-{label}", OUT)
        all_results.extend(rows)
        log_experiment(LOG_FILE, {"event": "eval_done", "system": f"Adapter-{label}", "rows": rows})

    if all_results:
        df_eval = pd.DataFrame(all_results)
        df_eval.to_csv(EVAL_CSV, index=False)
        print(f"\n  Saved → {EVAL_CSV}")

        # Print a clean summary table — the key result for the paper
        print("\n── Summary Table (per-class T-stage accuracy) ──")
        key_cols = ["system", "test_set", "accuracy", "macro_f1",
                    "acc_T1", "acc_T2", "acc_T3", "acc_T4"]
        avail = [c for c in key_cols if c in df_eval.columns]
        print(df_eval[avail].to_string(index=False))

    log_experiment(LOG_FILE, {"event": "complete", "ablation": "gate_decomposition"})
    print("\n" + "=" * 60)
    print("Ablation 3 complete.")
    print(f"  Pass rates   : {PASSRATE_CSV}")
    print(f"  SNOMED       : {SNOMED_CSV}")
    print(f"  Evaluation   : {EVAL_CSV}")
    print(f"  Log          : {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation 3: Gate Decomposition")
    parser.add_argument("--skip-gen",   action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-target",   type=int, default=TARGET_N)
    main(parser.parse_args())
