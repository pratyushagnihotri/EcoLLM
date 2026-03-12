<h1 align="center">
  <img src="reference_images/ecoLLM.png" alt="EcoLLM Logo" width="300"/>
  <br>Welcome to EcoLLM - Code and Documentation
</h1>

# EcoLLM: Energy-aware LLM Bench for Sustainable AI Data Systems

This repo benchmarks offline + online LLMs on energy/data-system workloads with metrics:

- Quality: pass rate, JSON-valid rate, numeric MAE/RMSE, anomaly F1, code/sql checks
- Efficiency: latency (avg/p95), tokens
- Cost: API cost (online), energy cost (kWh→€), etc.
- Sustainability: energy (kWh) + emissions (kg CO₂e)
- Resources: CPU/memory deltas

We start with EIA SEDS (industrial energy consumption) and progressively add task families and datasets.

## 0) Repo structure (what matters)

Key folders:

- runs/ → all experiment outputs (parquet + csv) per run directory
- tasks/ → generated YAML tasks (family/complexity/table size sweeps)
- scripts/ → dataset prep, task generation, exporters, plotting tools
- dashboard/ or Streamlit scripts → UI for inspection & comparison

Important files:

- Offline/Local benchmark: <code>bench.py</code>
- Online/OpenAI+EcoLogits benchmark: <code>bench_ecologits_online_merged.py</code>
- Export parquet→CSV: <code>scripts/export_parquet_to_csv.py</code>
- Plotting (insight plots): <code>scripts/plot_insight_bench.py</code>
- Streamlit leaderboard: <code>ecoLLM_benchmark.py</code>

## 1) Setup (one-time)

### 1.1 Create environment

```bash 
python -m venv .venv
source .venv/bin/activate
pip install -U pip
1.2 Install dependencies
```

Use your repo’s `requirements.txt`, or minimally:

```bash
pip install pandas pyarrow pyyaml numpy matplotlib plotly streamlit psutil requests scienceplots openpyxl
pip install "ecologits[openai]" openai
```

## 2) Phase 1 — Offline models (Ollama)

### 2.1 Start Ollama and pull models

Make sure Ollama runs:

```bash 
ollama serve
```

Pull models you want (examples):

```bash 
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b-instruct
ollama pull mistral:7b-instruct
ollama pull llama3:latest
ollama pull gemma:latest
ollama pull deepseek-coder:latest
ollama pull codellama:latest
```

Confirm:

```bash
ollama list
```

## 3) Phase 2 — Get dataset (EIA SEDS)

### 3.1 Fetch and prepare SEDS

Run:

```bash
python scripts/seds_fetch_prepare.py --repo-root . --overwrite --industrial-only
```

Expected outputs (examples):

- `data/raw/Complete_SEDS.csv`
- `data/processed/seds_industrial_consumption.parquet`

If you don’t see `data/processed/...`, fix that first before generating tasks.

## 4) Phase 3 — Generate tasks (Families + Complexity + Table sizes)

We use Families and Complexities:

<b>Family definitions (high level)</b>

- <b>Family 1 (F1):</b> Data Q&A grounded in a table (aggregation, max/min, deltas, averages)
- <b>Family 2 (F2):</b> Anomaly triage (spike, missing, step change, drift) + checks
- <b>Family 3 (F3):</b> Root cause / explanation style tasks (more reasoning, evidence)
- <b>Family 4 (F4):</b> Code/Query generation (SQL/Pandas operators, pipelines)

<b>Complexity levels (C1–C4)</b>

- <b>C1 (easy):</b> single operator (max / filter / simple groupby)
- <b>C2 (medium):</b> top-k + aggregation + rename / constraints
- <b>C3 (hard):</b> YoY delta + join/merge/pivot logic
- <b>C4 (hard+):</b> rolling/window/ranking per state, multi-year stats

<b>Table sizes</b>

We benchmark scaling using:
`[20, 100, 250, 500, 1000]` rows.

### 4.1 Generate tasks (example)

```
python scripts/gen_seds_family1_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
python scripts/gen_seds_family2_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
python scripts/gen_seds_family3_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
python scripts/gen_seds_family4_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
```

This should populate:

- `tasks/family1_qa/seds_f1_C1_sweep.yaml … C4`
- `tasks/family2_anomaly/seds_f2_C1_sweep.yaml … C4`

etc.

## 5) Run offline benchmarks (Ollama)

You run per family + complexity (recommended to debug small first).

Example (F1-C1):

```
python bench.py all --config configs/run_seds_f1_C1.yaml
```

It produces:

- `runs/<run_name>/runs.parquet`
- `runs/<run_name>/results.parquet`
- `runs/<run_name>/leaderboard_by_family.parquet`
- `runs/<run_name>/leaderboard_by_family_rows.parquet`
- CSV versions if enabled

### 5.1 Watch progress

Every run writes:

- `runs/<run_name>/progress.log`
- `runs/<run_name>/progress.txt`

So you can tail:

```bash
tail -f runs/seds_f1_C1_sweep/progress.log
```

## 6) Run online benchmarks (OpenAI + EcoLogits)

### 6.1 Set API key

```
export OPENAI_API_KEY="..."
```

### 6.2 Run online benchmark script

Use the merged EcoLogits benchmark:

```
python bench_ecologits_online_merged.py all --config configs/run_seds_f2_C1_online.yaml
```

This script:
- Uses <b>EcoLogits energy/emissions</b> for OpenAI calls when available
- Generates `leaderboard_by_family_rows.*` so plotting works

## 7) Export all runs to CSV (batch)

You can export a run:

```
python scripts/export_parquet_to_csv.py --run-dir runs/seds_f1_C2_sweep
```

Or export <b>all offline + online sweeps</b>:

```
bash scripts/export_all_seds_csv.sh
```

This creates `runs/<run>/csv/*.csv`.

## 8) Plotting (paper-ready figures)

### 8.1 Insight plots (offline / online / combined)

Use the plot suite script:

offline only:

```
python scripts/plot_insight_bench_v4.py --runs-root runs --out-root plots_insight_v4 --mode offline
```

online only:

```
python scripts/plot_insight_bench_v4.py --runs-root runs --out-root plots_insight_v4 --mode online
```

combined:

```
python scripts/plot_insight_bench_v4.py --runs-root runs --out-root plots_insight_v4 --mode combined
```

all three:

```
python scripts/plot_insight_bench_v4.py --runs-root runs --out-root plots_insight_v4 --mode all
```

Outputs in:

- `plots_insight_v4/offline/...`
- `plots_insight_v4/online/...`
- `plots_insight_v4/combined/...`

## 9) Frontend (Streamlit)

We provide a lightweight UI to:

- browse leaderboards
- inspect per-task outputs
- visualize tradeoffs and table-size scaling

### 9.1 Streamlit leaderboard

Run:

```bash 
streamlit run ecoLLM_benchmark.py
```

You can point it to any run directory under `runs/`.

Common pages:

- <b>Leaderboard</b> (model rankings)
- <b>Tradeoffs</b> (quality vs latency vs energy vs cost)
- <b>Table-size</b> impact (rows vs metrics)
- <b>Task inspector</b> (prompt/output/fail reason)

### 9.2 Where the UI reads from

The UI expects these files per run:
- `leaderboard_by_family.parquet`
- `leaderboard_by_family_rows.parquet` (for scaling plots)
- `results.parquet` (for per-task inspection)

If a run lacks `leaderboard_by_family_rows.parquet`, run:

```
python bench.py score --run-dir <run>
python bench.py aggregate --run-dir <run>
```

Or online:

```
python bench_ecologits_online_merged.py score --run-dir <run>
python bench_ecologits_online_merged.py aggregate --run-dir <run>
```

## 10) Recommended workflow (fast + safe)

<b>Step A: Validate pipeline (small)</b>

Run 1 model, 1 family, 1 complexity, 1 table size first.

<b>Step B: Sweep scaling</b>

Run table sizes `[20,100,250,500,1000]` for that family/complexity.

<b>Step C: Add models</b>

Add more Ollama models, rerun same configs.

<b>Step D: Add online models</b>

Run online configs with EcoLogits (energy/emissions).

<b>Step E: Export + plot</b>

Export all CSV, generate plots (offline / online / combined).

<b>Step F: Paper evaluation claims</b>

Use:

- scaling plots (rows vs latency/energy/cost)
- tradeoff plots (quality vs energy/cost)
- cross-family comparisons (C1–C4)

## 11) Typical issues and fixes

- <b>“No online runs found” in plotting</b>

Your online run folder exists but is missing:

`leaderboard_by_family_rows.parquet` or `csv/leaderboard_by_family_rows.csv`

Fix by rerunning score/aggregate with the merged online script.

- <b>o3-mini errors: “max_tokens not supported”</b>

Use the merged script; it sends `max_completion_tokens` for `o*` models.

## 12) What you will have at the end

A reproducible benchmark suite with:

- <b>Offline vs online model comparison</b>
- <b>Energy / emissions / cost vs accuracy tradeoffs</b>
- <b>Scaling study:</b> how table size increases latency/energy/cost
- <b>Leaderboards by family and complexity</b>
- <b>A Streamlit UI</b> to demo and debug results live