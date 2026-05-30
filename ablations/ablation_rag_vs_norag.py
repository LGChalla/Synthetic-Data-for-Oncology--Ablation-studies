"""
ablation_rag_vs_norag.py
========================
TCBB Ablation 1 — RAG vs. No-RAG

Research question
-----------------
Does MedCPT retrieval grounding improve downstream T-stage extraction
accuracy beyond the SNOMED vocabulary enrichment reported in BIBM?

BIBM established: RAG → +18.4% SNOMED density (22.62 → 26.79 per 100 words).
This ablation completes the story: does that vocabulary enrichment
translate into better adapter training signal and higher TSTR accuracy?

Design
------
Two matched corpora, identical everything except retrieval:

    Corpus-RAG    — MedCPT retrieved PubMed abstracts prepended to prompt
    Corpus-NoRAG  — schema + case only, no retrieval context

Both pass G(x) = S ∧ O ∧ C (full gate), same entropy floors.
Two adapters trained (n=162 each), evaluated TSTR on three test sets.

Controlled variables: TNM grid, models, hyperparameters, seed, gate.
Independent variable: presence of MedCPT retrieval context.
Dependent variables: per-class T-stage accuracy, macro-F1, SNOMED density.

Output structure
----------------
results/ablation_rag/
  rag_corpus.jsonl
  norag_corpus.jsonl
  rag_golden.csv
  norag_golden.csv
  snomed_density.csv
  evaluation_results.csv
  experiment.log
  adapter_rag/final_adapter/
  adapter_norag/final_adapter/
"""

import sys, json, gc, time, argparse, requests, numpy as np
from pathlib import Path

import torch

# ── Shared utilities ──────────────────────────────────────────────────────────
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
OUT         = ROOT / "results" / "ablation_rag"
RAG_JSONL   = OUT / "rag_corpus.jsonl"
NORAG_JSONL = OUT / "norag_corpus.jsonl"
RAG_CSV     = OUT / "rag_golden.csv"
NORAG_CSV   = OUT / "norag_golden.csv"
SNOMED_CSV  = OUT / "snomed_density.csv"
EVAL_CSV    = OUT / "evaluation_results.csv"
LOG_FILE    = OUT / "experiment.log"
ADAP_RAG    = OUT / "adapter_rag"
ADAP_NORAG  = OUT / "adapter_norag"

OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MedCPT RETRIEVER  (FAISS dense or PubMed eSearch fallback)
# ─────────────────────────────────────────────────────────────────────────────

class MedCPTRetriever:
    """
    Dense retrieval via FAISS + MedCPT-Query-Encoder when index is available.
    Falls back to PubMed eSearch API when index is absent.
    Mirrors the retriever used in the BIBM pipeline (phase1_datagen).
    """

    def __init__(self, index_path: str = None, k: int = 3):
        self.k     = k
        self.index = None
        self.abstracts: list[str] = []
        self._try_load_faiss(index_path)

    def _try_load_faiss(self, index_path):
        if not index_path or not Path(index_path).exists():
            print("[MedCPT] No FAISS index supplied — using PubMed eSearch fallback.")
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            print(f"[MedCPT] Loading FAISS index: {index_path}")
            self.encoder = SentenceTransformer("ncats/MedCPT-Query-Encoder")
            self.index   = faiss.read_index(index_path)
            abs_path = index_path.replace(".faiss", ".abstracts.json")
            if Path(abs_path).exists():
                with open(abs_path) as f:
                    self.abstracts = json.load(f)
            print(f"[MedCPT] Loaded {self.index.ntotal} vectors, "
                  f"{len(self.abstracts)} abstracts.")
        except Exception as e:
            print(f"[MedCPT] FAISS load failed ({e}) — falling back to eSearch.")
            self.index = None

    def retrieve(self, query: str) -> str:
        if self.index is not None:
            return self._dense(query)
        return self._efetch(query)

    def _dense(self, query: str) -> str:
        vec = self.encoder.encode([query], normalize_embeddings=True)
        _, idxs = self.index.search(np.array(vec, dtype=np.float32), self.k)
        hits = [self.abstracts[i] for i in idxs[0] if i < len(self.abstracts)]
        return "\n".join(f"[REF {i+1}] {h[:500]}" for i, h in enumerate(hits))

    def _efetch(self, query: str) -> str:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        try:
            r = requests.get(
                f"{base}/esearch.fcgi",
                params={"db": "pubmed", "term": query,
                        "retmax": self.k, "retmode": "json"},
                timeout=10,
            )
            ids = r.json()["esearchresult"]["idlist"]
            if not ids:
                return ""
            r2 = requests.get(
                f"{base}/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(ids),
                        "rettype": "abstract", "retmode": "text"},
                timeout=15,
            )
            return r2.text[:2000]
        except Exception as e:
            print(f"[MedCPT eSearch] {e}")
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def prompt_with_rag(retriever: MedCPTRetriever):
    """Returns a prompt_fn that prepends retrieved PubMed context."""
    def _fn(t: str, n: str, m: str) -> str:
        query   = f"lung cancer {t} {n} {m} AJCC staging treatment SNOMED CT"
        context = retriever.retrieve(query)
        parts   = [f"SCHEMA:\n{json.dumps(BASE_SCHEMA, indent=2)}\n"]
        if context:
            parts.append(
                f"CLINICAL EVIDENCE (retrieved from PubMed):\n{context}\n"
                "Ground realistic biomarkers, treatment modalities, and "
                "SNOMED-coded findings using the evidence above.\n"
            )
        parts.append(
            f"CASE:\n{build_controlled_case(t, n, m)}\n\n"
            "Map this case to the schema exactly."
        )
        return "\n".join(parts)
    return _fn


def prompt_no_rag(t: str, n: str, m: str) -> str:
    """Plain prompt — schema + case, no retrieval context."""
    return (
        f"SCHEMA:\n{json.dumps(BASE_SCHEMA, indent=2)}\n\n"
        f"CASE:\n{build_controlled_case(t, n, m)}\n\n"
        "Map this case to the schema exactly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("TCBB ABLATION 1 — RAG vs. No-RAG")
    print(f"Models   : {MODELS}")
    print(f"Target N : {args.n_target} per condition")
    print(f"Output   : {OUT}")
    print("=" * 60)

    log_experiment(LOG_FILE, {
        "event": "start", "ablation": "rag_vs_norag",
        "n_target": args.n_target, "models": MODELS,
    })

    rag_records, norag_records = [], []

    # ── Step 1: Generation ────────────────────────────────────────────────────
    if not args.skip_gen:
        retriever = MedCPTRetriever(index_path=args.faiss_index, k=3)

        per_model = max(1, args.n_target // len(MODELS))

        for model_id in MODELS:
            print(f"\n{'='*60}\nMODEL: {model_id}\n{'='*60}")
            hf_arts = None
            try:
                hf_arts = load_hf_model(model_id)

                # RAG
                recs = run_generation_loop(
                    model_id, hf_arts, RAG_JSONL,
                    prompt_fn   = prompt_with_rag(retriever),
                    gate_kwargs = dict(use_schema=True, use_ontology=True, use_logic=True),
                    n_target    = per_model, seed=SEED, condition="RAG",
                )
                rag_records.extend(recs)
                log_experiment(LOG_FILE, {
                    "event": "generation_done", "model": model_id,
                    "condition": "RAG", "n_valid": len(recs),
                })

                # No-RAG
                recs = run_generation_loop(
                    model_id, hf_arts, NORAG_JSONL,
                    prompt_fn   = prompt_no_rag,
                    gate_kwargs = dict(use_schema=True, use_ontology=True, use_logic=True),
                    n_target    = per_model, seed=SEED + 1000, condition="No-RAG",
                )
                norag_records.extend(recs)
                log_experiment(LOG_FILE, {
                    "event": "generation_done", "model": model_id,
                    "condition": "No-RAG", "n_valid": len(recs),
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
        for path, store in [(RAG_JSONL, rag_records), (NORAG_JSONL, norag_records)]:
            if path.exists():
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            try:
                                store.append(json.loads(line))
                            except Exception:
                                pass
                print(f"  Loaded {len(store)} records from {path}")

    # ── Step 2: SNOMED density comparison ─────────────────────────────────────
    import pandas as pd
    print("\n── SNOMED Density Comparison ──")
    df_snomed = pd.concat([
        snomed_density(rag_records,   "RAG"),
        snomed_density(norag_records, "No-RAG"),
    ], ignore_index=True)
    summary = df_snomed.groupby("condition")["density_per100"].agg(["mean","std","count"])
    print(summary.to_string())
    df_snomed.to_csv(SNOMED_CSV, index=False)
    print(f"  Saved → {SNOMED_CSV}")
    log_experiment(LOG_FILE, {"event": "snomed_density", "summary": summary.to_dict()})

    # ── Step 3: Golden CSVs + diversity audit ─────────────────────────────────
    print("\n── Exporting Golden CSVs ──")
    df_rag   = records_to_golden_csv(rag_records,   RAG_CSV)
    df_norag = records_to_golden_csv(norag_records, NORAG_CSV)

    rag_ok   = audit_diversity(df_rag,   "RAG corpus")
    norag_ok = audit_diversity(df_norag, "No-RAG corpus")
    if not rag_ok or not norag_ok:
        print("[WARN] Diversity check failed — supplement generation before training.")

    # ── Step 4: Adapter training ──────────────────────────────────────────────
    if not args.skip_train:
        train_adapter(str(RAG_CSV),   str(ADAP_RAG),   "Adapter-RAG")
        log_experiment(LOG_FILE, {"event": "training_done", "adapter": "RAG"})

        train_adapter(str(NORAG_CSV), str(ADAP_NORAG), "Adapter-NoRAG")
        log_experiment(LOG_FILE, {"event": "training_done", "adapter": "NoRAG"})
    else:
        print("[SKIP] Adapter training.")

    # ── Step 5: TSTR Evaluation ───────────────────────────────────────────────
    print("\n── TSTR Evaluation ──")
    all_results = []
    for label, adapter_dir in [
        ("Baseline (zero-shot)", None),
        ("Adapter-RAG",          str(ADAP_RAG)),
        ("Adapter-NoRAG",        str(ADAP_NORAG)),
    ]:
        rows = evaluate_adapter(adapter_dir, label, OUT)
        all_results.extend(rows)
        log_experiment(LOG_FILE, {"event": "eval_done", "system": label, "rows": rows})

    if all_results:
        import pandas as pd
        df_eval = pd.DataFrame(all_results)
        df_eval.to_csv(EVAL_CSV, index=False)
        print(f"\n  Saved → {EVAL_CSV}")
        print(df_eval.to_string(index=False))

    log_experiment(LOG_FILE, {"event": "complete", "ablation": "rag_vs_norag"})
    print("\n" + "=" * 60)
    print("Ablation 1 complete.")
    print(f"  SNOMED density : {SNOMED_CSV}")
    print(f"  Evaluation     : {EVAL_CSV}")
    print(f"  Log            : {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation 1: RAG vs. No-RAG")
    parser.add_argument("--faiss-index", default=None,
                        help="Path to MedCPT FAISS .faiss file (optional).")
    parser.add_argument("--skip-gen",   action="store_true",
                        help="Skip generation; use existing JSONL files.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip adapter training; run evaluation only.")
    parser.add_argument("--n-target",   type=int, default=TARGET_N,
                        help=f"Golden records per condition (default {TARGET_N}).")
    main(parser.parse_args())
