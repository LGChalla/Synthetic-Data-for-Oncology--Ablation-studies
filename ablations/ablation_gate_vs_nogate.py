"""
ablation_gate_vs_nogate.py
==========================
TCBB Ablation 2 — Gate vs. No-Gate

Research question
-----------------
What is the downstream training-signal cost of skipping the symbolic gate G(x)?
I.e., does accepting raw (unvalidated) LLM output as training data produce a
measurably worse adapter than one trained on gate-filtered records?

BIBM established: 0% valid JSON under unconstrained generation (Table II),
proving schema enforcement is a categorical prerequisite for structured output.
This ablation goes one step further: given that controlled generation already
produces some valid JSON without a gate, does *applying* the gate still matter
for downstream adapter quality?

Design
------
Three matched corpora, same TNM grid, same models, same generation params:

    Corpus-NoGate  — raw LLM output accepted if JSON parses; no S/O/C checks
    Corpus-Gate    — full G(x) = S ∧ O ∧ C; matches BIBM golden corpus

Both corpuses are size-matched (n=162). Three adapters trained:
    Adapter-NoGate  — trained on ungated corpus
    Adapter-Gate    — trained on gated corpus  (BIBM Adapter B analogue)
    Baseline        — zero-shot Llama-3-8B (no adapter)

Controlled variables: TNM grid, models, QLoRA hyperparams, seed, corpus size.
Independent variable: gate applied / bypassed.
Dependent variable: per-class T-stage accuracy, macro-F1, SNOMED density.

Expected result: Adapter-NoGate collapses on minority T-stages because
ungated records contain hallucinated staging labels, inconsistent AJCC logic,
and low SNOMED term density — all of which degrade the training signal.

Output structure
----------------
results/ablation_gate/
  nogate_corpus.jsonl
  gate_corpus.jsonl
  nogate_golden.csv
  gate_golden.csv
  gate_pass_rates.csv
  evaluation_results.csv
  experiment.log
  adapter_nogate/final_adapter/
  adapter_gate/final_adapter/
"""

import sys, json, gc, time, argparse, itertools
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
    symbolic_gate,
)
from pipeline.phase1_datagen import (
    load_hf_model, build_controlled_case, BASE_SCHEMA,
    generate_structured, set_global_seed,
    TNM_GRID,
)

# ── Output paths ──────────────────────────────────────────────────────────────
OUT           = ROOT / "results" / "ablation_gate"
NOGATE_JSONL  = OUT / "nogate_corpus.jsonl"
GATE_JSONL    = OUT / "gate_corpus.jsonl"
NOGATE_CSV    = OUT / "nogate_golden.csv"
GATE_CSV      = OUT / "gate_golden.csv"
PASSRATE_CSV  = OUT / "gate_pass_rates.csv"
EVAL_CSV      = OUT / "evaluation_results.csv"
LOG_FILE      = OUT / "experiment.log"
ADAP_NOGATE   = OUT / "adapter_nogate"
ADAP_GATE     = OUT / "adapter_gate"

OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT (identical for both conditions — isolation of gate effect)
# ─────────────────────────────────────────────────────────────────────────────

def base_prompt(t: str, n: str, m: str) -> str:
    return (
        f"SCHEMA:\n{json.dumps(BASE_SCHEMA, indent=2)}\n\n"
        f"CASE:\n{build_controlled_case(t, n, m)}\n\n"
        "Map this case to the schema exactly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# UNGATED GENERATION LOOP
# Accepts any record where JSON is parseable — no S/O/C validation.
# Mirrors run_generation_loop but with gate bypassed.
# ─────────────────────────────────────────────────────────────────────────────

def run_nogate_loop(
    model_id: str,
    hf_arts,
    out_jsonl: Path,
    n_target:  int  = TARGET_N,
    seed:      int  = SEED,
) -> list[dict]:
    """
    Generates records without any symbolic gate.
    Records are accepted as long as JSON parses (parsed_json_valid == True).
    All gate detail fields are set to 'bypassed' for auditability.
    """
    set_global_seed(seed)
    accepted, attempt = [], 0
    grid_cycle = itertools.cycle(enumerate(TNM_GRID))

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GENERATION (NO GATE)  |  model={model_id}")
    print(f"Target: {n_target} records  →  {out_jsonl}")
    print(f"{'='*60}")

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        while len(accepted) < n_target:
            cell_idx, (t, n, m) = next(grid_cycle)
            attempt += 1

            params = {
                "max_new_tokens": 1000,
                "temperature":    0.7,
                "top_p":          0.9,
                "top_k":          50,
                "do_sample":      True,
                "use_chat_template": True,
                "strict_json":    True,
                "seed":           seed + attempt,
                "intended_T":     t,
                "intended_N":     n,
                "intended_M":     m,
                "condition":      "No-Gate",
                "cell_index":     cell_idx,
            }

            result = generate_structured(
                model_id, hf_arts, base_prompt(t, n, m), params,
                layer="controlled", prompt_type="nogate-ablation",
            )

            # Only require valid JSON — no further checks
            if not result.get("parsed_json_valid"):
                print(f"  [{len(accepted):>3}/{n_target}] attempt {attempt:>4} "
                      f"{t}/{n}/{m} — INVALID JSON, skip")
                continue

            # Record what the gate *would have* decided (for analysis)
            _, gate_detail = symbolic_gate(
                parsed_json  = result.get("parsed_json", {}),
                snomed_hits  = result.get("snomed_codes", []),
                val_errors   = result.get("validation_errors", []),
                use_schema   = True,
                use_ontology = True,
                use_logic    = True,
            )
            result["gate_bypassed"] = True
            result["gate_detail"]   = gate_detail   # what WOULD have happened
            result["condition"]     = "No-Gate"

            accepted.append(result)
            fout.write(json.dumps(result) + "\n")
            fout.flush()

            would_pass = all(
                v is True for v in gate_detail.values() if v != "skipped"
            )
            print(f"  [{len(accepted):>3}/{n_target}] attempt {attempt:>4} "
                  f"{t}/{n}/{m} — accepted  "
                  f"[gate would {'PASS' if would_pass else 'FAIL'}: {gate_detail}]")

    print(f"\n  Done: {len(accepted)} accepted / {attempt} attempts")
    return accepted


# ─────────────────────────────────────────────────────────────────────────────
# GATE PASS RATE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_gate_pass_rates(nogate_records: list[dict]) -> pd.DataFrame:
    """
    Analyses the gate_detail of ungated records to compute:
    - Overall would-pass rate
    - Per-component (S, O, C) pass rate
    - Per-model breakdown

    This surfaces how many ungated records would have been rejected
    and why — quantifying the gate's filtering effect.
    """
    rows = []
    for rec in nogate_records:
        detail = rec.get("gate_detail", {})
        rows.append({
            "model":     rec.get("model", ""),
            "condition": rec.get("condition", ""),
            "S_pass":    detail.get("S") is True,
            "O_pass":    detail.get("O") is True,
            "C_pass":    detail.get("C") is True,
            "full_pass": all(detail.get(k) is True for k in ["S", "O", "C"]),
        })
    df = pd.DataFrame(rows)

    summary = df.groupby("model")[["S_pass", "O_pass", "C_pass", "full_pass"]].mean()
    print("\n── Gate Pass Rate Analysis (on No-Gate corpus) ──")
    print("  (What % of accepted records would have passed each gate component?)")
    print(summary.to_string())
    return summary.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("TCBB ABLATION 2 — Gate vs. No-Gate")
    print(f"Models   : {MODELS}")
    print(f"Target N : {args.n_target} per condition")
    print(f"Output   : {OUT}")
    print("=" * 60)

    log_experiment(LOG_FILE, {
        "event": "start", "ablation": "gate_vs_nogate",
        "n_target": args.n_target, "models": MODELS,
    })

    nogate_records, gate_records = [], []
    per_model = max(1, args.n_target // len(MODELS))

    # ── Step 1: Generation ────────────────────────────────────────────────────
    if not args.skip_gen:
        for model_id in MODELS:
            print(f"\n{'='*60}\nMODEL: {model_id}\n{'='*60}")
            hf_arts = None
            try:
                hf_arts = load_hf_model(model_id)

                # No-Gate condition
                recs = run_nogate_loop(
                    model_id, hf_arts, NOGATE_JSONL,
                    n_target=per_model, seed=SEED,
                )
                nogate_records.extend(recs)
                log_experiment(LOG_FILE, {
                    "event": "generation_done", "model": model_id,
                    "condition": "No-Gate", "n_valid": len(recs),
                })

                # Gated condition (full G(x))
                recs = run_generation_loop(
                    model_id, hf_arts, GATE_JSONL,
                    prompt_fn   = base_prompt,
                    gate_kwargs = dict(use_schema=True, use_ontology=True, use_logic=True),
                    n_target    = per_model, seed=SEED + 500, condition="Gate",
                )
                gate_records.extend(recs)
                log_experiment(LOG_FILE, {
                    "event": "generation_done", "model": model_id,
                    "condition": "Gate", "n_valid": len(recs),
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
        for path, store in [(NOGATE_JSONL, nogate_records), (GATE_JSONL, gate_records)]:
            if path.exists():
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            try:
                                store.append(json.loads(line))
                            except Exception:
                                pass
                print(f"  Loaded {len(store)} from {path}")

    # ── Step 2: Gate pass rate analysis ───────────────────────────────────────
    if nogate_records:
        df_passrate = compute_gate_pass_rates(nogate_records)
        df_passrate.to_csv(PASSRATE_CSV, index=False)
        print(f"  Saved → {PASSRATE_CSV}")
        log_experiment(LOG_FILE, {
            "event": "gate_passrate", "data": df_passrate.to_dict()
        })

    # ── Step 3: SNOMED density comparison ─────────────────────────────────────
    print("\n── SNOMED Density Comparison ──")
    df_snomed = pd.concat([
        snomed_density(nogate_records, "No-Gate"),
        snomed_density(gate_records,   "Gate"),
    ], ignore_index=True)
    summary = df_snomed.groupby("condition")["density_per100"].agg(["mean","std","count"])
    print(summary.to_string())

    # ── Step 4: Golden CSVs + diversity audit ─────────────────────────────────
    print("\n── Exporting Golden CSVs ──")
    df_nogate = records_to_golden_csv(nogate_records, NOGATE_CSV)
    df_gate   = records_to_golden_csv(gate_records,   GATE_CSV)

    print("\n── Diversity Audits ──")
    audit_diversity(df_nogate, "No-Gate corpus")
    audit_diversity(df_gate,   "Gate corpus")

    # ── Step 5: Adapter training ──────────────────────────────────────────────
    if not args.skip_train:
        train_adapter(str(NOGATE_CSV), str(ADAP_NOGATE), "Adapter-NoGate")
        log_experiment(LOG_FILE, {"event": "training_done", "adapter": "NoGate"})

        train_adapter(str(GATE_CSV),   str(ADAP_GATE),   "Adapter-Gate")
        log_experiment(LOG_FILE, {"event": "training_done", "adapter": "Gate"})
    else:
        print("[SKIP] Adapter training.")

    # ── Step 6: TSTR Evaluation ───────────────────────────────────────────────
    print("\n── TSTR Evaluation ──")
    all_results = []
    for label, adapter_dir in [
        ("Baseline (zero-shot)", None),
        ("Adapter-NoGate",       str(ADAP_NOGATE)),
        ("Adapter-Gate",         str(ADAP_GATE)),
    ]:
        rows = evaluate_adapter(adapter_dir, label, OUT)
        all_results.extend(rows)
        log_experiment(LOG_FILE, {"event": "eval_done", "system": label, "rows": rows})

    if all_results:
        df_eval = pd.DataFrame(all_results)
        df_eval.to_csv(EVAL_CSV, index=False)
        print(f"\n  Saved → {EVAL_CSV}")
        print(df_eval.to_string(index=False))

    log_experiment(LOG_FILE, {"event": "complete", "ablation": "gate_vs_nogate"})
    print("\n" + "=" * 60)
    print("Ablation 2 complete.")
    print(f"  Pass rate analysis : {PASSRATE_CSV}")
    print(f"  Evaluation         : {EVAL_CSV}")
    print(f"  Log                : {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation 2: Gate vs. No-Gate")
    parser.add_argument("--skip-gen",   action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-target",   type=int, default=TARGET_N)
    main(parser.parse_args())
