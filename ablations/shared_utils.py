"""
shared_utils.py
===============
Utilities shared across all three TCBB ablation scripts.

Provides:
  - Symbolic gate G(x) with configurable component toggles
  - Golden CSV builder (Phase 4 format)
  - TSTR evaluation runner (wraps phase3_benchmark logic)
  - Shannon entropy diversity audit
  - Structured JSON logger
  - SNOMED density calculator
"""

import os, sys, json, gc, re, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import requests
from scipy.stats import entropy as shannon_entropy

# ── Make StageCraft pipeline importable ───────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
STAGECRAFT = ROOT / "stagecraft"
sys.path.insert(0, str(STAGECRAFT))

from pipeline.phase1_datagen import (
    generate_structured,
    load_hf_model,
    build_controlled_case,
    TNM_GRID, T_VALUES, N_VALUES, M_VALUES,
    BASE_SCHEMA, SYSTEM_PROMPT,
    bioportal_annotate_snomed,
    extract_last_json,
    set_global_seed,
    DIVERSITY_ENTROPY_FLOOR,
)
from pipeline.phase4_finetuning import (
    train_qlora_adapter,
    check_label_diversity,
    prepare_dataset,
    MODEL_ID as FINETUNE_MODEL_ID,
)
from transformers import AutoTokenizer

# ── Models used across all ablations (no GPT — cost) ─────────────────────────
MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "wanglab/ClinicalCamel-70B",
]

TARGET_N = 162   # match BIBM golden corpus size
SEED     = 42

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOLIC GATE  G(x) = S ∧ O ∧ C  with toggleable components
# ─────────────────────────────────────────────────────────────────────────────

def symbolic_gate(
    parsed_json:  dict,
    snomed_hits:  list,
    val_errors:   list,
    use_schema:   bool = True,   # S — JSON schema completeness
    use_ontology: bool = True,   # O — SNOMED CT coverage
    use_logic:    bool = True,   # C — AJCC clinical logic
) -> tuple[bool, dict]:
    """
    Configurable symbolic gate.

    Parameters
    ----------
    parsed_json  : validated JSON object from the LLM
    snomed_hits  : list of SNOMED concept dicts from BioPortal
    val_errors   : list of AJCC logic error strings from phase1
    use_schema   : toggle Schema check  (S)
    use_ontology : toggle Ontology check (O)
    use_logic    : toggle AJCC logic check (C)

    Returns
    -------
    (passed: bool, detail: dict)  — detail records which components passed/failed
    """
    detail = {"S": None, "O": None, "C": None}
    notes  = parsed_json.get("notes", [{}]) if parsed_json else [{}]
    note   = notes[0] if notes else {}
    stg    = note.get("staging", {})

    # ── S: Schema completeness ─────────────────────────────────────────────
    if use_schema:
        required = ["staging", "histology", "molecular", "demographics",
                    "imaging", "treatment", "equity", "free_text"]
        s_pass = all(
            note.get(f) not in ("", None, [], {})
            for f in required
        )
        detail["S"] = s_pass
        if not s_pass:
            return False, detail
    else:
        detail["S"] = "skipped"

    # ── O: SNOMED ontology coverage ────────────────────────────────────────
    if use_ontology:
        o_pass = len(snomed_hits) >= 1
        detail["O"] = o_pass
        if not o_pass:
            return False, detail
    else:
        detail["O"] = "skipped"

    # ── C: AJCC clinical logic (M1 → Stage IV) ────────────────────────────
    if use_logic:
        m_val   = str(stg.get("M", "")).upper()
        s_group = str(stg.get("stage_group", "")).upper()
        c_pass  = not ("M1" in m_val and "IV" not in s_group)
        # Also fail if phase1 already flagged errors
        if val_errors:
            c_pass = False
        detail["C"] = c_pass
        if not c_pass:
            return False, detail
    else:
        detail["C"] = "skipped"

    return True, detail


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION LOOP  (shared by all three ablations)
# ─────────────────────────────────────────────────────────────────────────────

def run_generation_loop(
    model_id:     str,
    hf_arts,
    out_jsonl:    Path,
    prompt_fn,               # callable(t, n, m) -> str
    gate_kwargs:  dict,      # passed to symbolic_gate()
    n_target:     int  = TARGET_N,
    seed:         int  = SEED,
    condition:    str  = "unknown",
) -> list[dict]:
    """
    Core generation loop used by all ablations.

    Iterates over the TNM grid round-robin until n_target records pass
    the gate configured by gate_kwargs. Streams results to out_jsonl.

    Parameters
    ----------
    model_id    : HuggingFace model string
    hf_arts     : (model, tokenizer) tuple from load_hf_model()
    out_jsonl   : path to append valid records to
    prompt_fn   : function(t, n, m) -> prompt string
    gate_kwargs : dict of use_schema / use_ontology / use_logic booleans
    n_target    : how many valid records to collect
    seed        : random seed
    condition   : label written into each record for downstream analysis
    """
    import itertools
    set_global_seed(seed)
    golden, attempt = [], 0
    grid_cycle = itertools.cycle(enumerate(TNM_GRID))

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GENERATION  |  model={model_id}  |  condition={condition}")
    print(f"Gate: S={gate_kwargs.get('use_schema',True)}  "
          f"O={gate_kwargs.get('use_ontology',True)}  "
          f"C={gate_kwargs.get('use_logic',True)}")
    print(f"Target: {n_target} valid records → {out_jsonl}")
    print(f"{'='*60}")

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        while len(golden) < n_target:
            cell_idx, (t, n, m) = next(grid_cycle)
            attempt += 1

            prompt = prompt_fn(t, n, m)
            params = {
                "max_new_tokens":    1000,
                "temperature":       0.7,
                "top_p":             0.9,
                "top_k":             50,
                "do_sample":         True,
                "use_chat_template": True,
                "strict_json":       True,
                "seed":              seed + attempt,
                "intended_T":        t,
                "intended_N":        n,
                "intended_M":        m,
                "condition":         condition,
                "cell_index":        cell_idx,
                **{k: v for k, v in gate_kwargs.items()},
            }

            result = generate_structured(
                model_id, hf_arts, prompt, params,
                layer="controlled", prompt_type="ablation",
            )

            if not result.get("parsed_json_valid"):
                print(f"  [{len(golden):>3}/{n_target}] attempt {attempt:>4} "
                      f"{t}/{n}/{m} — INVALID JSON")
                continue

            passed, gate_detail = symbolic_gate(
                parsed_json  = result.get("parsed_json", {}),
                snomed_hits  = result.get("snomed_codes", []),
                val_errors   = result.get("validation_errors", []),
                **gate_kwargs,
            )

            result["gate_passed"]  = passed
            result["gate_detail"]  = gate_detail
            result["condition"]    = condition

            if not passed:
                print(f"  [{len(golden):>3}/{n_target}] attempt {attempt:>4} "
                      f"{t}/{n}/{m} — GATE FAIL {gate_detail}")
                continue

            golden.append(result)
            fout.write(json.dumps(result) + "\n")
            fout.flush()

            snomed_n = len(result.get("snomed_codes", []))
            print(f"  [{len(golden):>3}/{n_target}] attempt {attempt:>4} "
                  f"{t}/{n}/{m} — PASS ✓  SNOMED={snomed_n}")

    print(f"\n  Done: {len(golden)} valid / {attempt} attempts "
          f"({attempt - len(golden)} rejected)")
    return golden


# ─────────────────────────────────────────────────────────────────────────────
# GOLDEN CSV EXPORT  (Phase 4 format)
# ─────────────────────────────────────────────────────────────────────────────

def records_to_golden_csv(records: list[dict], csv_path: Path) -> pd.DataFrame:
    """
    Converts a list of gate-passed generation records into the CSV format
    expected by phase4_finetuning.prepare_dataset():
        T_target, N_target, M_target, free_text, model, condition, run_id
    """
    rows = []
    for rec in records:
        pj    = rec.get("parsed_json") or {}
        notes = pj.get("notes", [{}])
        note  = notes[0] if notes else {}
        stg   = note.get("staging", {})
        p     = rec.get("params", {})
        rows.append({
            "T_target":  stg.get("T",  p.get("intended_T", "Unknown")),
            "N_target":  stg.get("N",  p.get("intended_N", "Unknown")),
            "M_target":  stg.get("M",  p.get("intended_M", "Unknown")),
            "free_text": note.get("free_text", rec.get("raw_output", "")),
            "model":     rec.get("model", ""),
            "condition": rec.get("condition", ""),
            "run_id":    rec.get("run_id", ""),
        })
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  Exported {len(df)} records → {csv_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DIVERSITY AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def audit_diversity(df: pd.DataFrame, label: str) -> bool:
    """Shannon entropy audit. Returns True if all TNM dimensions pass floors."""
    print(f"\n── Diversity audit: {label} (n={len(df)}) ──")
    all_pass = True
    for col, key in [("T_target", "T"), ("N_target", "N"), ("M_target", "M")]:
        if col not in df.columns:
            print(f"  [{col}] MISSING"); continue
        counts = df[col].value_counts()
        ent    = shannon_entropy(counts) if len(counts) > 1 else 0.0
        floor  = DIVERSITY_ENTROPY_FLOOR[key]
        ok     = ent >= floor
        if not ok:
            all_pass = False
        print(f"  [{col}]  H={ent:.3f}  floor={floor:.3f}  "
              f"{'✓ PASS' if ok else '✗ FAIL'}")
        print(f"          {counts.to_dict()}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# SNOMED DENSITY
# ─────────────────────────────────────────────────────────────────────────────

def snomed_density(records: list[dict], condition: str) -> pd.DataFrame:
    """SNOMED CT terms per 100 words — matches BIBM Section IV metric."""
    rows = []
    for rec in records:
        pj        = rec.get("parsed_json") or {}
        notes     = pj.get("notes", [{}])
        free_text = (notes[0].get("free_text", "") if notes else "") \
                    or rec.get("raw_output", "")
        words     = max(len(free_text.split()), 1)
        n_snomed  = len(rec.get("snomed_codes", []))
        rows.append({
            "condition":      condition,
            "model":          rec.get("model", ""),
            "snomed_n":       n_snomed,
            "word_count":     words,
            "density_per100": round((n_snomed / words) * 100, 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER TRAINING WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def train_adapter(csv_path: str, adapter_dir: str, run_label: str):
    """
    Thin wrapper around phase4's train_qlora_adapter.
    Loads tokenizer once, runs diversity check, trains, saves.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {run_label}")
    print(f"  corpus : {csv_path}")
    print(f"  output : {adapter_dir}")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path} — skipping training.")
        return

    df = pd.read_csv(csv_path)
    ok = check_label_diversity(df, csv_path, abort_on_fail=False)
    if not ok:
        print("[WARN] Diversity check failed — training will proceed but "
              "results may be degenerate. Increase corpus size or re-generate.")

    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset(csv_path, tokenizer)
    if dataset is None:
        print(f"[ERROR] Dataset preparation failed for {csv_path}")
        return

    os.makedirs(adapter_dir, exist_ok=True)
    train_qlora_adapter(dataset, adapter_dir, run_label, tokenizer)

    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# TSTR EVALUATION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_adapter(
    adapter_dir:  str | None,
    system_label: str,
    results_dir:  Path,
) -> list[dict]:
    """
    Runs TSTR evaluation for one system (baseline or adapter) across
    all three test sets: synthetic held-out, MTSamples Lung, MTSamples All-Cancer.

    Returns a list of result dicts (one per test set).

    This wraps the inference + per-class accuracy logic from phase3_benchmark
    directly so the ablation scripts don't need to call phase3 as a subprocess.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from sklearn.metrics import accuracy_score, f1_score
    from pipeline.phase3_benchmark import (
        normalize_tnm_label,
        per_class_accuracy,
        bootstrap_ci_accuracy,
        constant_classifier_baseline,
        filter_for_valid_gt,
    )

    TEST_SETS = {
        "Synthetic held-out": str(STAGECRAFT / "data_splits" / "test_synthetic.csv"),
        "MTSamples Lung":     str(STAGECRAFT / "data_splits" / "test_mtsamples_lung.csv"),
        "MTSamples All-Cancer": str(STAGECRAFT / "data_splits" / "test_mtsamples_all.csv"),
    }

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model_id = FINETUNE_MODEL_ID

    print(f"\n── Evaluating: {system_label} ──")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb, device_map="auto"
    )
    if adapter_dir and os.path.exists(os.path.join(adapter_dir, "final_adapter")):
        model = PeftModel.from_pretrained(
            model, os.path.join(adapter_dir, "final_adapter")
        )
        print(f"  Loaded adapter from {adapter_dir}/final_adapter")
    else:
        print(f"  No adapter — zero-shot baseline")

    model.eval()

    EXTRACT_PROMPT = (
        "You are a clinical data extractor. Read the clinical note and extract "
        "the TNM staging. Return ONLY a JSON object with keys 'T', 'N', 'M'. "
        "Example: {{\"T\": \"T2\", \"N\": \"N1\", \"M\": \"M0\"}}.\n\n"
        "NOTE: {text}"
    )

    def predict_tstage(text: str) -> str:
        prompt = EXTRACT_PROMPT.format(text=text[:800])
        msgs   = [{"role": "user", "content": prompt}]
        inp    = tokenizer.apply_chat_template(
            msgs, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to(next(model.parameters()).device)
        with torch.inference_mode():
            out = model.generate(
                **inp, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(
            out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
        )
        try:
            obj = json.loads(extract_last_json.__wrapped__(decoded)
                             if hasattr(extract_last_json, "__wrapped__")
                             else decoded)
            return normalize_tnm_label(str(obj.get("T", "Unknown")), "T")
        except Exception:
            # Try regex fallback
            m = re.search(r'"T"\s*:\s*"([^"]+)"', decoded)
            return normalize_tnm_label(m.group(1), "T") if m else "Unknown"

    all_results = []
    for set_name, csv_path in TEST_SETS.items():
        if not os.path.exists(csv_path):
            print(f"  [SKIP] Test set not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        valid = filter_for_valid_gt(df)["T_valid"]
        if len(valid) == 0:
            print(f"  [SKIP] No valid T ground truth in {set_name}")
            continue

        preds = [predict_tstage(str(r)) for r in valid["free_text"]]
        gts   = [normalize_tnm_label(str(v), "T") for v in valid["T_target"]]

        acc      = accuracy_score(gts, preds)
        lo, hi   = bootstrap_ci_accuracy(gts, preds)
        macro_f1 = f1_score(gts, preds, average="macro", zero_division=0)
        per_cls  = per_class_accuracy(gts, preds, "T")

        const = constant_classifier_baseline(gts, "T")

        row = {
            "system":       system_label,
            "test_set":     set_name,
            "n":            len(gts),
            "accuracy":     round(acc, 3),
            "ci_lower":     lo,
            "ci_upper":     hi,
            "macro_f1":     round(macro_f1, 3),
            "const_acc":    const["constant_acc"],
        }
        # Flatten per-class accuracy into row
        for _, r in per_cls.iterrows():
            row[f"acc_{r['class']}"] = r["per_class_accuracy"]

        all_results.append(row)
        print(f"  {set_name}: acc={acc:.3f} [{lo},{hi}]  "
              f"macro-F1={macro_f1:.3f}  const={const['constant_acc']:.3f}")
        print(per_cls.to_string(index=False))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED JSON LOGGER
# ─────────────────────────────────────────────────────────────────────────────

def log_experiment(log_path: Path, payload: dict):
    """Appends a timestamped JSON event to the experiment log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(), **payload}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
