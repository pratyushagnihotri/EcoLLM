<h1 align="center">
  <img src="reference_images/ecoLLM.png" alt="EcoLLM Logo" width="300"/>
  <br>Welcome to EcoLLM - Code and Documentation
</h1>

# EcoLLM: Energy-aware LLM Bench for Sustainable AI Data Systems

EcoLLM benchmarks offline and online LLMs on energy/data-system workloads using a unified evaluation pipeline.

The benchmark tracks:

- **Quality**: task pass rate, anomaly classification accuracy/F1, code compilation checks
- **Efficiency**: latency (avg / p95)
- **Sustainability**: energy (kWh)

We start with EIA SEDS industrial energy consumption data and generate benchmark task families from it. The current benchmark design defines difficulty in terms of **data-management operator semantics** rather than prompt length. The four workload families now align with semantic filtering, anomaly/validation reasoning, join-based analytics, and pipeline code generation. 

---

## Repository structure

Important folders:

- `runs/` → experiment outputs per run directory
- `tasks/` → generated YAML benchmark tasks
- `ecoLLM/` → dataset prep, task generation, export, plotting
- `reference_images/` → project assets
- `Streamlit/UI` → leaderboard and inspection frontend

Important files:

- **Offline / local benchmark:** `bench.py` 
- **Online / OpenAI + EcoLogits benchmark:** `bench_ecologits_online.py`
- **Dataset preparation:** `ecoLLM/seds_fetch_prepare.py`
- **SEDS task generators:** `ecoLLM/gen_seds_family*_sweep.py`
- **Export parquet → CSV:** `ecoLLM/export_parquet_to_csv.py`
- **Plotting:** `ecoLLM/plots_ecoLLM.py`
- **Streamlit leaderboard:** `ecoLLM_benchmark.py`

## 1) Setup

### 1.1 Create a virtual environment

```bash 
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 1.2 Install dependencies
Use our project repo’s `requirements.txt`, or minimally install:

```bash
pip install pandas pyarrow pyyaml numpy matplotlib plotly streamlit psutil requests scienceplots openpyxl
pip install "ecologits[openai]" openai
```

If you also use local Ollama models:

```bash
pip install requests
```

## 2) Offline models (Ollama)

### 2.1 Start Ollama

Make sure Ollama runs:

```bash 
ollama serve
```

### 2.2 Pull models
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

Check installed models:

```bash
ollama list
```

## 3) Dataset preparation (EIA SEDS)

### 3.1 Fetch and prepare SEDS

Run:

```bash
python ecoLLM/seds_fetch_prepare.py --repo-root . --overwrite --industrial-only --add-report-text
```
This script:

- downloads EIA SEDS CSV + codebook
- enriches rows with MSN descriptions and units
- writes Parquet output
- adds an LLM-friendly Report_Text column used by the new task generators

Expected outputs (examples):

- `data/raw/Complete_SEDS.csv`
- `data/raw/Codes_and_Descriptions.xlsx`
- `data/processed/seds_enriched.parquet`
- `data/processed/seds_industrial_consumption.parquet`

If you don’t see `data/processed/...`, fix that first before generating tasks.


## 4) Benchmark families and complexity

### 4.1 Workload families

The current benchmark families are:

- <b>Family 1 (F1):</b> Semantic tabular reasoning
  - text-aware filtering over Report_Text
  - aggregation, grouping, ranking over a single table
- <b>Family 2 (F2):</b> Data-quality reasoning / anomaly classification
  - classify anomaly type over structured state-year histories
  - combines numeric trends and report narratives
- <b>Family 3 (F3):</b> Join-based multi-step analytics
  - joins an energy table with a reports/events table
  - performs semantic filtering, subset comparison, aggregation, ranking
- <b>Family 4 (F4):</b> Pipeline code generation
  - generate Python/pandas pipelines
  - includes semantic filtering and multi-table joins

### 4.2 Complexity levels
- <b>C1 (easy):</b>  single-step operator
- <b>C2 (medium):</b> multi-operator query
- <b>C3 (hard):</b> derived statistic / subset comparison / multi-step reasoning
- <b>C4 (hard+):</b> composed analytical pipeline


### 4.3 Table sizes

We benchmark scaling using:
`[20, 100, 250, 500, 1000]` rows.


## 5) Generate tasks (example)

```
python ecoLLM/gen_seds_family1_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
python ecoLLM/gen_seds_family2_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
python ecoLLM/gen_seds_family3_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
python ecoLLM/gen_seds_family4_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
```

This should generate:

- `tasks/family1..4_qa/seds_f1_C1_sweep.yaml`
- `tasks/family1..4_qa/seds_f1_C2_sweep.yaml`
- `tasks/family1..4_qa/seds_f1_C3_sweep.yaml`
- `tasks/family1..4_qa/seds_f1_C4_sweep.yaml`

## 6) Run offline benchmarks (Ollama)

You can run per family + complexity (recommended to debug small first).

Example (F1-C1):

```
python bench.py all --config configs/run_seds_f1_C1.yaml
```

The offline benchmark script:

- loads YAML tasks
- renders prompts with strict JSON output contract
- runs local models (dummy / Ollama)
- records latency, energy emissions, and pass rate
- scores outputs
- writes leaderboards and results files

Typical outputs:

- `runs/<run_name>/runs.jsonl`
- `runs/<run_name>/runs.parquet`
- `runs/<run_name>/results.parquet`
- `runs/<run_name>/leaderboard_by_family.parquet`
- `runs/<run_name>/leaderboard_by_family_rows.parquet`
- `runs/<run_name>/progress.log`
- `runs/<run_name>/progress.txt`

### 6.1 Watch progress

Every run writes:

- `runs/<run_name>/progress.log`
- `runs/<run_name>/progress.txt`

So you can tail:

```bash
tail -f runs/seds_f1_C1_sweep/progress.log
```

## 7) Run online benchmarks (OpenAI + EcoLogits)

### 7.1 Set API key

```
export OPENAI_API_KEY="..."
```

### 7.2 Run online benchmark script

Use the merged EcoLogits benchmark:

```
python bench_ecologits_online_merged.py all --config configs/run_seds_f2_C1_online.yaml
```

The online benchmark script:

- uses OpenAI models through the OpenAI SDK
- uses EcoLogits for energy and emissions when available
- falls back to local heuristic energy for non-online adapters
- supports GPT-family and reasoning-family models
- writes the same benchmark output structure as the offline runner

### 7.3 Important OpenAI note

Reasoning models such as `o3-mini` require `max_completion_tokens` internally, while `GPT-4.1` models use the normal token-limit path. The patched online benchmark file already handles this distinction.

## 8) Export all runs/results to CSV (batch)

You can export a run:

```
python ecoLLM/export_parquet_to_csv.py --run-dir runs/seds_f1_C2_sweep
```

Or export <b>all offline + online sweeps</b>:

```
bash ecoLLM/export_all_seds_csv.sh
```

This creates `runs/<run>/csv/*.csv`.

## 9) Plotting

### 9.1 Insight plots (offline / online / combined)

Use the plot suite script:

offline only:

```
python ecoLLM/plots_ecoLLM.py --runs-root runs --out-root plots_insight_v4 --mode offline
```

online only:

```
python ecoLLM/plots_ecoLLM.py --runs-root runs --out-root plots_insight --mode online
```

combined:

```
python ecoLLM/plots_ecoLLM.py --runs-root runs --out-root plots_insight --mode combined
```

all three:

```
python ecoLLM/plots_ecoLLM.py --runs-root runs --out-root plots_insight --mode all
```

Outputs in:

- `plots_insight/offline/...`
- `plots_insight/online/...`
- `plots_insight/combined/...`

## 10) Frontend (Streamlit)

We provide a lightweight UI to:

- browse leaderboards
- inspect per-task outputs
- visualize tradeoffs and table-size scaling

### 10.1 Streamlit leaderboard

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

### 10.2 Where the UI reads from

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

## 11) Recommended workflow (fast + safe)

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

## 12) Typical issues and fixes

- <b>“No online runs found” in plotting</b>

Your online run folder exists but is missing:

`leaderboard_by_family_rows.parquet` or `csv/leaderboard_by_family_rows.csv`

Fix by rerunning score/aggregate with the merged online script.

- <b>o3-mini errors: “max_tokens not supported”</b>

Use the merged script; it sends `max_completion_tokens` for `o*` models.

- <b>No online runs found in plotting</b>

Make sure the run directory contains:

- `leaderboard_by_family.parquet`
- `leaderboard_by_family_rows.parquet`

If not, rerun score and aggregate.


## 13) What you will have at the end

At the end, EcoLLM gives you a reproducible benchmark suite for comparing LLMs across:

- <b>Offline vs online model comparison</b>
- <b>quality vs latency vs energy tradeoffs</b>
- <b>Scaling study:</b> how table size increases latency/energy/accuracy
- <b>Leaderboards by family and complexity</b>
- <b>A Streamlit UI</b> to demo and debug results live
