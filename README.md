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
- Streamlit leaderboard: <code>streamlit_leaderboard_mvp.py</code>

## 1) Setup (one-time)

### 1.1 Create environment

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
1.2 Install dependencies
```

Use your repo’s `requirements.txt`, or minimally:

pip install pandas pyarrow pyyaml numpy matplotlib plotly streamlit psutil requests scienceplots openpyxl
pip install "ecologits[openai]" openai