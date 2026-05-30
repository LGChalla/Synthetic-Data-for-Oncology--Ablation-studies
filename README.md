# StageCraft — TCBB Ablation Studies

Extension of [StageCraft (BIBM 2026)](https://github.com/LGChalla/Synthetic-Data-Generation-for-Lung-Cancer-Staging-Using-LLMs) with three controlled ablation experiments for the IEEE/ACM TCBB journal submission.

## Ablations

| # | Name | Research Question |
|---|------|-------------------|
| 1 | **RAG vs. No-RAG** | Does MedCPT retrieval improve downstream T-stage accuracy beyond SNOMED vocabulary enrichment? |
| 2 | **Gate vs. No-Gate** | What is the training-signal cost of skipping symbolic gate G(x) entirely? |
| 3 | **Gate Decomposition** | Which gate component (S / O / C) contributes most — individually and in combination? |

## Setup

```bash
# 1. Clone with submodule
git clone --recurse-submodules https://github.com/LGChalla/Synthetic-Data-for-Oncology--Ablation-studies.git
cd Synthetic-Data-for-Oncology--Ablation-studies

# 2. Install Node dependencies
npm install

# 3. Install Python dependencies (from StageCraft)
pip install -r stagecraft/requirements.txt --break-system-packages

# 4. Set environment variables
export HF_TOKEN=your_hf_token
export BIOPORTAL_API_KEY=your_bioportal_key
```

## Running

```bash
npm run run:rag       # Ablation 1 — RAG vs No-RAG
npm run run:gate      # Ablation 2 — Gate vs No-Gate
npm run run:decomp    # Ablation 3 — Gate Decomposition
npm run run:all       # All three in sequence

# CLI for fine-grained control
node scripts/cli.js run rag --n-target 162
node scripts/cli.js run rag --skip-gen       # skip generation, train only
node scripts/cli.js run rag --skip-train     # skip training, evaluate only
node scripts/cli.js run all --push           # run all + auto push to GitHub
```

## Monitoring

```bash
node scripts/cli.js status           # what has run / what is pending
node scripts/cli.js results          # print all evaluation tables
node scripts/cli.js results rag      # one ablation
node scripts/cli.js logs rag         # tail live log
node scripts/cli.js push "message"   # commit + push to GitHub
```

## Structure

```
.
├── ablations/
│   ├── shared_utils.py                 # Gate, eval, logging utilities
│   ├── ablation_rag_vs_norag.py        # Ablation 1
│   ├── ablation_gate_vs_nogate.py      # Ablation 2
│   └── ablation_gate_decomposition.py  # Ablation 3
├── scripts/
│   └── cli.js                          # Node.js orchestration CLI
├── stagecraft/                          # Git submodule — StageCraft pipeline
├── results/                             # All experiment outputs
└── logs/                                # CLI run logs
```

## Compute

- GPU: H200 recommended
- Models: Llama-3.3-70B-Instruct, ClinicalCamel-70B (4-bit NF4 quantisation)
- Fine-tuning: Llama-3-8B-Instruct via QLoRA (identical config to BIBM)
- Estimated runtime per ablation: ~4–6 hours
