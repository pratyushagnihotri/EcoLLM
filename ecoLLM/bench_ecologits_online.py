#!/usr/bin/env python3
from __future__ import annotations

"""
Energy LLM Benchmark

Requirements for EcoLogits online tracking:
    pip install "ecologits[openai]" openai

Env:
    OPENAI_API_KEY=...
"""

import argparse
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
import logging

try:
    import psutil
except Exception:
    psutil = None

try:
    import requests
except Exception:
    requests = None

# EcoLogits + OpenAI SDK are OPTIONAL (only needed if you run adapter=openai)
try:
    from ecologits import EcoLogits  # type: ignore
except Exception:
    EcoLogits = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


# ----------------------------
# Pricing defaults (USD per 1M tokens, Standard tier)
# ----------------------------
OPENAI_PRICE_USD_PER_1M = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5": (1.25, 10.00),
    "gpt-5-codex": (1.25, 10.00),
    "gpt-5.2-codex": (1.75, 14.00),
    "o3-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_run_logger(out_dir: str, level: str = "INFO") -> logging.Logger:
    """Logs progress to console and to <out_dir>/progress.log."""
    ensure_dir(out_dir)
    logger = logging.getLogger("bench")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_path = os.path.join(out_dir, "progress.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_jsonl(path: str, rows: List[Dict[str, Any]], mode: str = "a") -> None:
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jsonl_to_parquet(jsonl_path: str, parquet_path: str) -> None:
    data: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    if not data:
        return
    df = pd.json_normalize(data, sep=".")
    df.to_parquet(parquet_path, index=False)



def maybe_write_csv(df: pd.DataFrame, run_dir: str, filename: str) -> None:
    """Best-effort CSV export for easier inspection/sharing."""
    out_path = os.path.join(run_dir, filename)
    df.to_csv(out_path, index=False)


def compute_api_cost_usd(tokens_in: Optional[int], tokens_out: Optional[int], model_id: str, model_params: dict) -> float:
    ti = float(tokens_in or 0)
    to = float(tokens_out or 0)

    pin_1m = model_params.get("price_in_usd_per_1m")
    pout_1m = model_params.get("price_out_usd_per_1m")

    if pin_1m is None or pout_1m is None:
        pin_1k = model_params.get("price_in_usd_per_1k")
        pout_1k = model_params.get("price_out_usd_per_1k")
        if pin_1k is not None and pout_1k is not None:
            return (ti / 1000.0) * float(pin_1k) + (to / 1000.0) * float(pout_1k)

        adapter = (model_params.get("adapter") or "").lower()
        mid = (model_params.get("model_name") or model_id or "").lower()
        if adapter == "openai" or mid.startswith("gpt-") or mid.startswith("o"):
            if mid in OPENAI_PRICE_USD_PER_1M:
                pin_1m, pout_1m = OPENAI_PRICE_USD_PER_1M[mid]

    if pin_1m is None or pout_1m is None:
        return 0.0

    return (ti / 1_000_000.0) * float(pin_1m) + (to / 1_000_000.0) * float(pout_1m)


def _safe_isnan(x: Any) -> bool:
    return isinstance(x, float) and pd.isna(x)


def normalize_listlike(x: Any) -> Optional[List[Any]]:
    if x is None or _safe_isnan(x):
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, (str, bytes, dict)):
        return None
    try:
        return list(x)
    except Exception:
        return None


def infer_table_rows(task: Dict[str, Any]) -> Optional[int]:
    meta = task.get("meta") or {}
    tr = meta.get("table_rows")
    if tr is not None and not _safe_isnan(tr):
        try:
            return int(tr)
        except Exception:
            pass
    tid = str(task.get("task_id") or "")
    m = re.search(r"(?:^|_)r(\d+)(?:_|$)", tid)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def extract_json_block_strict(text: str) -> Optional[dict]:
    match = re.fullmatch(r"\s*```json\s*(\{.*?\})\s*```\s*", text, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        return None
    raw = match.group(1).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def get_by_path(d: dict, path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def get_row_value(row: Dict[str, Any], path: str) -> Any:
    """Read either a flattened parquet column or a nested in-memory dict path."""
    if path in row:
        return row[path]
    return get_by_path(row, path)


def estimate_energy_kwh(wall_s: float, cpu_s: float, cpu_power_w: float, idle_power_w: float, cpu_count: int) -> float:
    if wall_s <= 0 or cpu_count <= 0:
        return 0.0
    util = cpu_s / (wall_s * cpu_count)
    util = max(0.0, min(1.0, util))
    avg_power_w = float(idle_power_w) + util * (float(cpu_power_w) - float(idle_power_w))
    return (avg_power_w * (wall_s / 3600.0)) / 1000.0


def estimate_emissions_kg(energy_kwh: float, grid_kgco2_per_kwh: float) -> float:
    return float(energy_kwh) * float(grid_kgco2_per_kwh)


def load_tasks(task_paths: List[str]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for p in task_paths:
        with open(p, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        tasks.extend(doc.get("tasks", []))
    return tasks


def render_prompt(task: Dict[str, Any]) -> str:
    prompt = task["input"]
    ctx = task.get("context")
    if ctx:
        ctx_yaml = yaml.safe_dump(ctx, sort_keys=False, allow_unicode=True)
        prompt = prompt + "\n\nCONTEXT:\n" + ctx_yaml

    contract = (
        "OUTPUT FORMAT (STRICT, REQUIRED)\n"
        "You MUST output exactly ONE fenced JSON block and NOTHING else.\n\n"
        "```json\n"
        "{\n"
        "  \"answer\": \"short human-readable answer\",\n"
        "  \"numbers\": { \"some_metric\": 123.0 },\n"
        "  \"label\": { \"anomaly_type\": \"spike|drop|drift|inconsistent|other\" },\n"
        "  \"code\": { \"language\": \"python|sql\", \"content\": \"...\" },\n"
        "  \"checks\": [\"...\", \"...\"],\n"
        "  \"evidence\": [\"...\"]\n"
        "}\n"
        "```\n\n"
        "Rules:\n"
        "- Always include: \"answer\"\n"
        "- Include \"numbers\" only if numeric results are required.\n"
        "- Include \"label.anomaly_type\" only for anomaly tasks.\n"
        "- Include \"code\" only for code/query generation tasks.\n"
        "- Do NOT output any text outside the JSON block.\n"
    )
    return prompt + "\n\n" + contract


@dataclass
class GenerationResult:
    text: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    latency_ms: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None
    impacts_energy_kwh: Optional[float] = None
    impacts_gwp_kgco2eq: Optional[float] = None


class ModelAdapter:
    def __init__(self, model_id: str, **params: Any):
        self.model_id = model_id
        self.params = params

    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> GenerationResult:
        raise NotImplementedError


class DummyAdapter(ModelAdapter):
    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> GenerationResult:
        start = time.perf_counter()
        lower = prompt.lower()

        if "highest" in lower:
            payload = {"answer": "Region C has the highest industrial energy use in 2019.", "numbers": {"industrial_energy_tbtu": 143}}
        elif "year-over-year" in lower or "yoy" in lower:
            payload = {"answer": "The YoY change is +10 TBtu.", "numbers": {"delta_tbtu": 10}}
        elif "anomaly" in lower or "spike" in lower:
            payload = {"answer": "Likely anomaly type: spike.", "label": {"anomaly_type": "spike"}, "checks": ["verify sensor validity", "correlate with operations events"]}
        elif "python" in lower and "pandas" in lower:
            code = (
                "import pandas as pd\n"
                "out = df.groupby('region', as_index=False)['industrial_energy_tbtu'].sum()\n"
                "out = out.rename(columns={'industrial_energy_tbtu':'total_tbtu'})\n"
                "print(out)\n"
            )
            payload = {"answer": "Pandas code provided.", "code": {"language": "python", "content": code}}
        elif "sql" in lower:
            sql = (
                "SELECT region, industrial_energy_tbtu\n"
                "FROM energy_facts\n"
                "WHERE year = 2019\n"
                "ORDER BY industrial_energy_tbtu DESC\n"
                "LIMIT 3;"
            )
            payload = {"answer": "SQL provided.", "code": {"language": "sql", "content": sql}}
        else:
            payload = {"answer": "No-op."}

        text = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
        time.sleep(0.02)
        return GenerationResult(
            text=text,
            tokens_in=len(prompt) // 4,
            tokens_out=len(text) // 4,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


class OllamaAdapter(ModelAdapter):
    def __init__(self, model_id: str, **params: Any):
        super().__init__(model_id, **params)
        if requests is None:
            raise RuntimeError("Install requests: pip install requests")
        self.model_name = params.get("model_name")
        if not self.model_name:
            raise ValueError("OllamaAdapter requires params.model_name")
        self.host = params.get("host", "http://localhost:11434").rstrip("/")
        self.keep_alive = params.get("keep_alive", "5m")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 500,
        timeout_s: float = 120.0,
        **kwargs: Any,
    ) -> GenerationResult:
        url = f"{self.host}/api/chat"
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
        }

        start = time.perf_counter()
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = (data.get("message") or {}).get("content", "") or ""
        tokens_in = data.get("prompt_eval_count")
        tokens_out = data.get("eval_count")

        return GenerationResult(text=text, tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms, raw=data)


_ECOLOGITS_INITIALIZED = False


class OpenAIEcoLogitsAdapter(ModelAdapter):
    def __init__(self, model_id: str, **params: Any):
        super().__init__(model_id, **params)

        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed. Install: pip install openai")
        if EcoLogits is None:
            raise RuntimeError("EcoLogits not installed. Install: pip install 'ecologits[openai]'")

        global _ECOLOGITS_INITIALIZED
        if not _ECOLOGITS_INITIALIZED:
            EcoLogits.init(providers=["openai"])
            _ECOLOGITS_INITIALIZED = True

        self.model_name = params.get("model_name") or params.get("model") or model_id
        self.api_key = params.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OpenAIEcoLogitsAdapter requires OPENAI_API_KEY env var or params.api_key")
        self.timeout_s = float(params.get("timeout_s", 120.0))
        self.client = OpenAI(api_key=self.api_key)

    def _extract_text(self, resp: Any) -> str:
        try:
            if hasattr(resp, "choices") and resp.choices:
                return (resp.choices[0].message.content or "").strip()
        except Exception:
            pass
        try:
            return str(resp)
        except Exception:
            return ""

    def _extract_usage(self, resp: Any) -> tuple[Optional[int], Optional[int]]:
        try:
            u = getattr(resp, "usage", None)
            if u is None:
                return None, None
            ti = getattr(u, "prompt_tokens", None)
            to = getattr(u, "completion_tokens", None)
            if ti is None and hasattr(u, "input_tokens"):
                ti = getattr(u, "input_tokens", None)
            if to is None and hasattr(u, "output_tokens"):
                to = getattr(u, "output_tokens", None)
            return (int(ti) if ti is not None else None, int(to) if to is not None else None)
        except Exception:
            return None, None

    def _extract_impacts(self, resp: Any) -> tuple[Optional[float], Optional[float]]:
        energy_kwh = None
        gwp_kg = None
        try:
            impacts = getattr(resp, "impacts", None)
            if impacts is None:
                return None, None
            energy = getattr(impacts, "energy", None)
            if energy is not None:
                v = getattr(energy, "value", None)
                if v is not None:
                    energy_kwh = float(v.mean) if hasattr(v, "mean") else float(v)
            gwp = getattr(impacts, "gwp", None)
            if gwp is not None:
                v = getattr(gwp, "value", None)
                if v is not None:
                    gwp_kg = float(v.mean) if hasattr(v, "mean") else float(v)
        except Exception:
            return energy_kwh, gwp_kg
        return energy_kwh, gwp_kg

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 500,
        timeout_s: float = 120.0,
        **kwargs: Any,
    ) -> GenerationResult:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        #start = time.perf_counter()
        #resp = self.client.chat.completions.create(
        #    model=self.model_name,
        #    messages=messages,
        #    temperature=float(temperature),
        #    max_tokens=int(max_tokens),
        #    timeout=float(timeout_s or self.timeout_s),
        #)
        #latency_ms = (time.perf_counter() - start) * 1000.0
        # Reasoning models like o3/o4 expect max_completion_tokens
        model_name_l = str(self.model_name).lower()
        is_reasoning_model = model_name_l.startswith("o3") or model_name_l.startswith("o4")

        req: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "timeout": float(timeout_s or self.timeout_s),
        }

        if is_reasoning_model:
            req["max_completion_tokens"] = int(max_tokens)
            # Many reasoning models do not use temperature the same way; omit it unless you know it is supported.
        else:
            req["max_tokens"] = int(max_tokens)
            req["temperature"] = float(temperature)

        start = time.perf_counter()
        resp = self.client.chat.completions.create(**req)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = self._extract_text(resp)
        tokens_in, tokens_out = self._extract_usage(resp)
        impacts_energy_kwh, impacts_gwp_kg = self._extract_impacts(resp)

        return GenerationResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            raw=None,
            impacts_energy_kwh=impacts_energy_kwh,
            impacts_gwp_kgco2eq=impacts_gwp_kg,
        )


def make_adapter(model_cfg: Dict[str, Any]) -> ModelAdapter:
    adapter = model_cfg["adapter"]
    model_id = model_cfg["model_id"]
    params = model_cfg.get("params") or {}
    if adapter == "dummy":
        return DummyAdapter(model_id, **params)
    if adapter == "ollama":
        return OllamaAdapter(model_id, **params)
    if adapter == "openai":
        return OpenAIEcoLogitsAdapter(model_id, **params)
    raise ValueError(f"Unknown adapter: {adapter}")


def score_one(run_row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    text = run_row.get("output_text") or ""
    err = run_row.get("error")

    if err is not None:
        out["metric.json_valid"] = 0
        out["metric.task_pass"] = 0
        out["metric.fail_reason"] = "runtime_error"
        return out

    j = extract_json_block_strict(text)
    out["metric.json_valid"] = 1 if j is not None else 0
    if j is None:
        out["metric.task_pass"] = 0
        out["metric.fail_reason"] = "missing_or_invalid_json"
        return out

    def _first_number(s: Any) -> Optional[float]:
        if s is None:
            return None
        try:
            s = str(s)
        except Exception:
            return None
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    passes: List[int] = []

    numeric_targets = normalize_listlike(get_row_value(run_row, "expected.numeric_targets"))
    if isinstance(numeric_targets, list):
        single_numeric = len([x for x in numeric_targets if x is not None]) == 1
        for tgt in numeric_targets:
            if not isinstance(tgt, dict):
                tgt_obj = tgt
                tgt = None
                if hasattr(tgt_obj, "as_py"):
                    try:
                        tgt = tgt_obj.as_py()
                    except Exception:
                        tgt = None
                if tgt is None and hasattr(tgt_obj, "items"):
                    try:
                        tgt = dict(tgt_obj.items())
                    except Exception:
                        tgt = None
                if tgt is None:
                    try:
                        tgt = dict(tgt_obj)
                    except Exception:
                        tgt = None
            if not isinstance(tgt, dict):
                continue

            name = tgt.get("name")
            if not name:
                continue
            try:
                target_val = float(tgt.get("value"))
            except Exception:
                continue
            tol = float(tgt.get("tolerance_abs", 0))

            pred_val = get_by_path(j, f"numbers.{name}")
            if pred_val is None:
                pred_val = _first_number(j.get("answer")) if isinstance(j, dict) else None
            if pred_val is None and single_numeric:
                try:
                    pred_val = _first_number(json.dumps(j, ensure_ascii=False))
                except Exception:
                    pred_val = None

            try:
                pred = float(pred_val) if pred_val is not None else None
            except Exception:
                pred = None

            abs_err = None if pred is None else abs(pred - target_val)
            out[f"metric.numeric.{name}.pred"] = pred
            out[f"metric.numeric.{name}.abs_error"] = abs_err
            out[f"metric.numeric.{name}.pass"] = 1 if (abs_err is not None and abs_err <= tol) else 0
            passes.append(out[f"metric.numeric.{name}.pass"])

    exp_anom = get_row_value(run_row, "expected.labels.anomaly_type")
    if exp_anom is not None and not _safe_isnan(exp_anom):
        exp = str(exp_anom).strip().lower()
        pred = get_by_path(j, "label.anomaly_type")
        if pred is None and isinstance(j, dict):
            if isinstance(j.get("label"), dict):
                pred = j["label"].get("anomaly_type")
            if pred is None:
                pred = j.get("anomaly_type")
        pred_norm = str(pred).strip().lower() if pred is not None else ""
        out["metric.anomaly.pred"] = pred_norm
        out["metric.anomaly.pass"] = 1 if pred_norm == exp else 0
        passes.append(out["metric.anomaly.pass"])

    det_specs = normalize_listlike(get_row_value(run_row, "scoring_spec.deterministic"))
    if isinstance(det_specs, list):
        for spec in det_specs:
            if not isinstance(spec, dict):
                continue
            t = spec.get("type")

            if t == "code_exec_smoke":
                lang = str(get_by_path(j, "code.language") or "").strip().lower()
                code = str(get_by_path(j, "code.content") or "")
                out["metric.code.language"] = lang
                out["metric.code.present"] = 1 if code.strip() else 0
                if lang != "python" or not code.strip():
                    out["metric.code.python_compiles"] = 0
                    passes.append(0)
                else:
                    try:
                        compile(code, "<gen>", "exec")
                        out["metric.code.python_compiles"] = 1
                        passes.append(1)
                    except Exception:
                        out["metric.code.python_compiles"] = 0
                        passes.append(0)

            elif t == "sql_parse_smoke":
                lang = str(get_by_path(j, "code.language") or "").strip().lower()
                sql = str(get_by_path(j, "code.content") or "")
                out["metric.sql.language"] = lang
                out["metric.sql.present"] = 1 if sql.strip() else 0
                out["metric.sql_has_select"] = 1 if ("select" in sql.lower()) else 0
                ok = (lang in ["sql", "postgres", "sqlite"]) and out["metric.sql_has_select"] == 1
                passes.append(1 if ok else 0)

            elif t == "label_match":
                label_name = str(spec.get("label") or "").strip()
                if not label_name:
                    passes.append(0)
                else:
                    exp_val = get_row_value(run_row, f"expected.labels.{label_name}")
                    pred_val = get_by_path(j, f"label.{label_name}")

                    if pred_val is None and isinstance(j, dict):
                        if isinstance(j.get("label"), dict):
                            pred_val = j["label"].get(label_name)
                        if pred_val is None:
                            pred_val = j.get(label_name)

                    exp_norm = str(exp_val).strip().lower() if exp_val is not None and not _safe_isnan(exp_val) else ""
                    pred_norm = str(pred_val).strip().lower() if pred_val is not None else ""

                    out[f"metric.label.{label_name}.pred"] = pred_norm
                    out[f"metric.label.{label_name}.pass"] = 1 if pred_norm == exp_norm else 0
                    passes.append(out[f"metric.label.{label_name}.pass"])

    if len(passes) == 0:
        out["metric.task_pass"] = 1
        out["metric.fail_reason"] = None
        return out

    out["metric.task_pass"] = 1 if all(p == 1 for p in passes) else 0
    out["metric.fail_reason"] = None if out["metric.task_pass"] == 1 else "metric_failed"
    return out


def cmd_run(config_path: str) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_cfg = cfg["run"]
    out_dir = run_cfg["output_dir"]
    ensure_dir(out_dir)

    logger = setup_run_logger(out_dir, level=str(run_cfg.get("log_level", "INFO")))

    electricity_price_eur_per_kwh = float(run_cfg.get("electricity_price_eur_per_kwh", 0.30))
    grid_kgco2_per_kwh = float(run_cfg.get("grid_kgco2_per_kwh", 0.35))
    cpu_power_w = float(run_cfg.get("cpu_power_w", 45.0))
    idle_power_w = float(run_cfg.get("idle_power_w", 8.0))

    tasks = load_tasks(cfg["tasks"]["include"])
    models = cfg["models"]
    decoding = run_cfg.get("decoding", {}) or {}
    repeats = int(run_cfg.get("repeats", 1))
    timeout_s = float(run_cfg.get("constraints", {}).get("timeout_s", 120))
    progress_every = int(run_cfg.get("progress_every", 10))

    total = len(models) * len(tasks) * repeats
    done = 0
    t0 = time.perf_counter()
    last_log = 0

    run_id = run_cfg.get("run_id", "run")
    run_uuid = str(uuid.uuid4())

    jsonl_path = os.path.join(out_dir, "runs.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    proc = psutil.Process(os.getpid()) if psutil else None
    cpu_count = psutil.cpu_count(logical=True) if psutil else os.cpu_count() or 1

    buffer: List[Dict[str, Any]] = []

    for m in models:
        adapter = make_adapter(m)
        model_params = m.get("params", {}) or {}

        for t in tasks:
            for rep in range(repeats):
                prompt_full = render_prompt(t)

                start = time.perf_counter()
                mem_before = proc.memory_info().rss if proc else None
                cpu_before = proc.cpu_times() if proc else None

                err = None
                result_text = ""
                tokens_in = None
                tokens_out = None
                latency_ms_model = None
                impacts_energy_kwh = None
                impacts_gwp_kg = None

                try:
                    res = adapter.generate(prompt_full, system_prompt="", timeout_s=timeout_s, **decoding)
                    result_text = res.text
                    tokens_in = res.tokens_in
                    tokens_out = res.tokens_out
                    latency_ms_model = res.latency_ms
                    impacts_energy_kwh = res.impacts_energy_kwh
                    impacts_gwp_kg = res.impacts_gwp_kgco2eq
                except Exception as e:
                    err = repr(e)

                end = time.perf_counter()
                mem_after = proc.memory_info().rss if proc else None
                cpu_after = proc.cpu_times() if proc else None

                latency_ms_total = (end - start) * 1000.0

                cpu_user_s_delta = None
                cpu_system_s_delta = None
                mem_rss_delta_mb = None

                if proc and mem_before is not None and mem_after is not None:
                    mem_rss_delta_mb = (mem_after - mem_before) / (1024 * 1024)
                if proc and cpu_before and cpu_after:
                    cpu_user_s_delta = (cpu_after.user - cpu_before.user)
                    cpu_system_s_delta = (cpu_after.system - cpu_before.system)

                cpu_s = float((cpu_user_s_delta or 0.0) + (cpu_system_s_delta or 0.0))
                wall_s = max(0.000001, end - start)

                adapter_name = str(m.get("adapter") or "").lower()
                energy_source = "local_heuristic"
                energy_kwh = estimate_energy_kwh(wall_s, cpu_s, cpu_power_w, idle_power_w, int(cpu_count))
                emissions_kg = estimate_emissions_kg(energy_kwh, grid_kgco2_per_kwh)

                if adapter_name == "openai" and impacts_energy_kwh is not None:
                    energy_source = "ecologits"
                    energy_kwh = float(impacts_energy_kwh)
                    if impacts_gwp_kg is not None:
                        emissions_kg = float(impacts_gwp_kg)

                row: Dict[str, Any] = {
                    "run_id": run_id,
                    "run_uuid": run_uuid,
                    "timestamp_utc": now_utc_iso(),
                    "model_id": m["model_id"],
                    "model_adapter": m["adapter"],
                    "model_params": model_params,
                    "task_id": t["task_id"],
                    "meta.table_rows": infer_table_rows(t),
                    "family": t.get("family"),
                    "difficulty": t.get("difficulty"),
                    "repeat_idx": rep,
                    "prompt": prompt_full,
                    "output_text": result_text,
                    "expected": t.get("expected"),
                    "scoring_spec": t.get("scoring"),
                    "decoding": decoding,
                    "latency_ms_total": latency_ms_total,
                    "latency_ms_model": latency_ms_model,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "error": err,
                    "cpu_user_s_delta": cpu_user_s_delta,
                    "cpu_system_s_delta": cpu_system_s_delta,
                    "mem_rss_delta_mb": mem_rss_delta_mb,
                    "energy_kwh": energy_kwh,
                    "emissions_kg": emissions_kg,
                    "energy_source": energy_source,
                    "energy_cost_eur": energy_kwh * electricity_price_eur_per_kwh,
                    "api_cost_usd": compute_api_cost_usd(
                        tokens_in,
                        tokens_out,
                        m["model_id"],
                        {**model_params, "adapter": m.get("adapter"), "model_name": model_params.get("model_name")},
                    ),
                }

                buffer.append(row)
                done += 1
                if total > 0 and (done == 1 or (done - last_log) >= progress_every):
                    elapsed = time.perf_counter() - t0
                    ips = (done / elapsed) if elapsed > 0 else 0.0
                    pct = (done / total) * 100.0
                    logger.info(
                        f"[{done}/{total} | {pct:5.1f}% | {ips:.2f} it/s] "
                        f"model={m['model_id']} task={t['task_id']} family={t.get('family')} rep={rep} "
                        f"lat_total_ms={latency_ms_total:.1f} energy_kwh={energy_kwh:.6g} src={energy_source} err={err}"
                    )
                    try:
                        with open(os.path.join(out_dir, "progress.txt"), "w", encoding="utf-8") as pf:
                            pf.write(
                                f"{datetime.now(UTC).isoformat().replace('+00:00', 'Z')}\n"
                                f"{done}/{total}\n"
                                f"{pct:.2f}\n"
                                f"{m['model_id']}\n"
                                f"{t['task_id']}\n"
                                f"{t.get('family')}\n"
                            )
                    except Exception:
                        pass
                    last_log = done

                if len(buffer) >= 50:
                    write_jsonl(jsonl_path, buffer, mode="a")
                    buffer = []

    if buffer:
        write_jsonl(jsonl_path, buffer, mode="a")

    parquet_path = os.path.join(out_dir, "runs.parquet")
    jsonl_to_parquet(jsonl_path, parquet_path)

    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {parquet_path}")
    return out_dir


def cmd_score(run_dir: str) -> None:
    runs_path = os.path.join(run_dir, "runs.parquet")
    if not os.path.exists(runs_path):
        raise FileNotFoundError(runs_path)

    df = pd.read_parquet(runs_path)
    metrics = df.apply(lambda r: score_one(r.to_dict()), axis=1, result_type="expand")
    out_df = pd.concat([df, metrics], axis=1)

    out_path = os.path.join(run_dir, "results.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path}")

    try:
        maybe_write_csv(out_df, run_dir, "results.csv")
    except Exception:
        pass

    # Row-level leaderboard for sweep/table-size experiments
    if "meta.table_rows" in out_df.columns:
        d2 = out_df.dropna(subset=["meta.table_rows"]).copy()
        if not d2.empty:
            numeric_abs_cols = [c for c in d2.columns if c.startswith("metric.numeric.") and c.endswith(".abs_error")]

            def row_numeric_abs_errors(row):
                vals: List[float] = []
                for c in numeric_abs_cols:
                    v = row.get(c)
                    if v is not None and pd.notna(v):
                        vals.append(float(v))
                return vals

            d2["_numeric_abs_errors"] = d2.apply(row_numeric_abs_errors, axis=1)
            d2["_numeric_sq_errors"] = d2["_numeric_abs_errors"].apply(lambda xs: [x * x for x in xs])

            if "expected.labels.anomaly_type" in d2.columns and "metric.anomaly.pred" in d2.columns:
                d2["_anomaly_true"] = d2["expected.labels.anomaly_type"].apply(
                    lambda x: None if (x is None or _safe_isnan(x)) else str(x).strip().lower()
                )
                d2["_anomaly_pred"] = d2["metric.anomaly.pred"].astype(str).str.strip().str.lower()
            else:
                d2["_anomaly_true"] = None
                d2["_anomaly_pred"] = None

            def agg_numeric_mae(series_of_lists: pd.Series) -> float:
                all_err: List[float] = []
                for lst in series_of_lists:
                    if isinstance(lst, list):
                        all_err.extend(lst)
                return float(pd.Series(all_err).mean()) if all_err else float("nan")

            def agg_numeric_rmse(series_of_lists: pd.Series) -> float:
                all_sq: List[float] = []
                for lst in series_of_lists:
                    if isinstance(lst, list):
                        all_sq.extend(lst)
                return float((pd.Series(all_sq).mean()) ** 0.5) if all_sq else float("nan")

            def agg_f1_macro(subdf: pd.DataFrame) -> float:
                d = subdf.dropna(subset=["_anomaly_true"])
                if d.empty:
                    return float("nan")
                labs = sorted(set(d["_anomaly_true"].tolist()) | set(d["_anomaly_pred"].tolist()))
                f1s: List[float] = []
                for lab in labs:
                    tp = int(((d["_anomaly_true"] == lab) & (d["_anomaly_pred"] == lab)).sum())
                    fp = int(((d["_anomaly_true"] != lab) & (d["_anomaly_pred"] == lab)).sum())
                    fn = int(((d["_anomaly_true"] == lab) & (d["_anomaly_pred"] != lab)).sum())
                    prec = tp / (tp + fp) if (tp + fp) else 0.0
                    rec = tp / (tp + fn) if (tp + fn) else 0.0
                    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
                    f1s.append(f1)
                return float(sum(f1s) / len(f1s)) if f1s else float("nan")

            rows2 = []
            for (model_id, family, table_rows), g in d2.groupby(["model_id", "family", "meta.table_rows"], dropna=False):
                rows2.append({
                    "model_id": model_id,
                    "family": family,
                    "table_rows": int(table_rows) if pd.notna(table_rows) else None,
                    "n": int(len(g)),
                    "pass_rate": float(g["metric.task_pass"].mean()) if "metric.task_pass" in g.columns else float("nan"),
                    "json_valid_rate": float(g["metric.json_valid"].mean()) if "metric.json_valid" in g.columns else float("nan"),
                    "avg_latency_ms": float(g["latency_ms_total"].mean()) if "latency_ms_total" in g.columns else float("nan"),
                    "p95_latency_ms": float(g["latency_ms_total"].quantile(0.95)) if "latency_ms_total" in g.columns else float("nan"),
                    "error_rate": float(pd.notna(g["error"]).mean()) if "error" in g.columns else float("nan"),
                    "avg_tokens_out": float(g["tokens_out"].mean()) if "tokens_out" in g.columns else float("nan"),
                    "avg_energy_kwh": float(g["energy_kwh"].mean()) if "energy_kwh" in g.columns else float("nan"),
                    "avg_energy_cost_eur": float(g["energy_cost_eur"].mean()) if "energy_cost_eur" in g.columns else float("nan"),
                    "avg_api_cost_usd": float(g["api_cost_usd"].mean()) if "api_cost_usd" in g.columns else float("nan"),
                    "avg_mem_rss_delta_mb": float(g["mem_rss_delta_mb"].mean()) if "mem_rss_delta_mb" in g.columns else float("nan"),
                    "avg_cpu_user_s_delta": float(g["cpu_user_s_delta"].mean()) if "cpu_user_s_delta" in g.columns else float("nan"),
                    "avg_cpu_system_s_delta": float(g["cpu_system_s_delta"].mean()) if "cpu_system_s_delta" in g.columns else float("nan"),
                    "numeric_mae": agg_numeric_mae(g["_numeric_abs_errors"]),
                    "numeric_rmse": agg_numeric_rmse(g["_numeric_sq_errors"]),
                    "anomaly_f1_macro": agg_f1_macro(g),
                })

            lb2 = pd.DataFrame(rows2).sort_values(
                ["family", "table_rows", "pass_rate", "avg_latency_ms"],
                ascending=[True, True, False, True],
            )
            out_path2 = os.path.join(run_dir, "leaderboard_by_family_rows.parquet")
            lb2.to_parquet(out_path2, index=False)
            print(f"Wrote: {out_path2}")

            try:
                maybe_write_csv(lb2, run_dir, "leaderboard_by_family_rows.csv")
            except Exception:
                pass

def cmd_aggregate(run_dir: str) -> None:
    results_path = os.path.join(run_dir, "results.parquet")
    if not os.path.exists(results_path):
        raise FileNotFoundError(results_path)

    df = pd.read_parquet(results_path)
    numeric_abs_cols = [c for c in df.columns if c.startswith("metric.numeric.") and c.endswith(".abs_error")]

    def row_numeric_abs_errors(row):
        vals: List[float] = []
        for c in numeric_abs_cols:
            v = row.get(c)
            if v is not None and pd.notna(v):
                vals.append(float(v))
        return vals

    df["_numeric_abs_errors"] = df.apply(row_numeric_abs_errors, axis=1)
    df["_numeric_sq_errors"] = df["_numeric_abs_errors"].apply(lambda xs: [x * x for x in xs])

    def extract_true_anomaly(row):
        exp = row.get("expected.labels.anomaly_type")
        if exp is None or _safe_isnan(exp):
            return None
        return str(exp).strip().lower()

    if "metric.anomaly.pred" in df.columns:
        df["_anomaly_true"] = df.apply(extract_true_anomaly, axis=1)
        df["_anomaly_pred"] = df["metric.anomaly.pred"].astype(str).str.strip().str.lower()
    else:
        df["_anomaly_true"] = None
        df["_anomaly_pred"] = None

    def agg_numeric_mae(series_of_lists: pd.Series) -> float:
        all_err: List[float] = []
        for lst in series_of_lists:
            if isinstance(lst, list):
                all_err.extend(lst)
        return float(pd.Series(all_err).mean()) if all_err else float("nan")

    def agg_numeric_rmse(series_of_lists: pd.Series) -> float:
        all_sq: List[float] = []
        for lst in series_of_lists:
            if isinstance(lst, list):
                all_sq.extend(lst)
        return float((pd.Series(all_sq).mean()) ** 0.5) if all_sq else float("nan")

    def agg_f1_macro(subdf: pd.DataFrame) -> float:
        d = subdf.dropna(subset=["_anomaly_true"])
        if d.empty:
            return float("nan")
        labs = sorted(set(d["_anomaly_true"].tolist()) | set(d["_anomaly_pred"].tolist()))
        f1s: List[float] = []
        for lab in labs:
            tp = int(((d["_anomaly_true"] == lab) & (d["_anomaly_pred"] == lab)).sum())
            fp = int(((d["_anomaly_true"] != lab) & (d["_anomaly_pred"] == lab)).sum())
            fn = int(((d["_anomaly_true"] == lab) & (d["_anomaly_pred"] != lab)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
            f1s.append(f1)
        return float(sum(f1s) / len(f1s)) if f1s else float("nan")

    rows = []
    for (model_id, family), g in df.groupby(["model_id", "family"], dropna=False):
        rows.append({
            "model_id": model_id,
            "family": family,
            "n": int(len(g)),
            "pass_rate": float(g["metric.task_pass"].mean()) if "metric.task_pass" in g.columns else float("nan"),
            "json_valid_rate": float(g["metric.json_valid"].mean()) if "metric.json_valid" in g.columns else float("nan"),
            "avg_latency_ms": float(g["latency_ms_total"].mean()) if "latency_ms_total" in g.columns else float("nan"),
            "p95_latency_ms": float(g["latency_ms_total"].quantile(0.95)) if "latency_ms_total" in g.columns else float("nan"),
            "error_rate": float(pd.notna(g["error"]).mean()) if "error" in g.columns else float("nan"),
            "avg_tokens_out": float(g["tokens_out"].mean()) if "tokens_out" in g.columns else float("nan"),
            "avg_energy_kwh": float(g["energy_kwh"].mean()) if "energy_kwh" in g.columns else float("nan"),
            "avg_energy_cost_eur": float(g["energy_cost_eur"].mean()) if "energy_cost_eur" in g.columns else float("nan"),
            "avg_api_cost_usd": float(g["api_cost_usd"].mean()) if "api_cost_usd" in g.columns else float("nan"),
            "avg_mem_rss_delta_mb": float(g["mem_rss_delta_mb"].mean()) if "mem_rss_delta_mb" in g.columns else float("nan"),
            "avg_cpu_user_s_delta": float(g["cpu_user_s_delta"].mean()) if "cpu_user_s_delta" in g.columns else float("nan"),
            "avg_cpu_system_s_delta": float(g["cpu_system_s_delta"].mean()) if "cpu_system_s_delta" in g.columns else float("nan"),
            "numeric_mae": agg_numeric_mae(g["_numeric_abs_errors"]),
            "numeric_rmse": agg_numeric_rmse(g["_numeric_sq_errors"]),
            "anomaly_f1_macro": agg_f1_macro(g),
        })

    lb = pd.DataFrame(rows).sort_values(["family", "pass_rate", "avg_latency_ms"], ascending=[True, False, True])
    out_path = os.path.join(run_dir, "leaderboard_by_family.parquet")
    lb.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path}")

    try:
        maybe_write_csv(lb, run_dir, "leaderboard_by_family.csv")
    except Exception:
        pass

    if "meta.table_rows" in df.columns:
        d2 = df.dropna(subset=["meta.table_rows"]).copy()
        if not d2.empty:
            rows2 = []
            for (model_id, family, table_rows), g in d2.groupby(["model_id", "family", "meta.table_rows"], dropna=False):
                rows2.append({
                    "model_id": model_id,
                    "family": family,
                    "table_rows": int(table_rows) if pd.notna(table_rows) else None,
                    "n": int(len(g)),
                    "pass_rate": float(g["metric.task_pass"].mean()) if "metric.task_pass" in g.columns else float("nan"),
                    "json_valid_rate": float(g["metric.json_valid"].mean()) if "metric.json_valid" in g.columns else float("nan"),
                    "avg_latency_ms": float(g["latency_ms_total"].mean()) if "latency_ms_total" in g.columns else float("nan"),
                    "p95_latency_ms": float(g["latency_ms_total"].quantile(0.95)) if "latency_ms_total" in g.columns else float("nan"),
                    "error_rate": float(pd.notna(g["error"]).mean()) if "error" in g.columns else float("nan"),
                    "avg_tokens_out": float(g["tokens_out"].mean()) if "tokens_out" in g.columns else float("nan"),
                    "avg_energy_kwh": float(g["energy_kwh"].mean()) if "energy_kwh" in g.columns else float("nan"),
                    "avg_energy_cost_eur": float(g["energy_cost_eur"].mean()) if "energy_cost_eur" in g.columns else float("nan"),
                    "avg_api_cost_usd": float(g["api_cost_usd"].mean()) if "api_cost_usd" in g.columns else float("nan"),
                    "avg_mem_rss_delta_mb": float(g["mem_rss_delta_mb"].mean()) if "mem_rss_delta_mb" in g.columns else float("nan"),
                    "avg_cpu_user_s_delta": float(g["cpu_user_s_delta"].mean()) if "cpu_user_s_delta" in g.columns else float("nan"),
                    "avg_cpu_system_s_delta": float(g["cpu_system_s_delta"].mean()) if "cpu_system_s_delta" in g.columns else float("nan"),
                    "numeric_mae": agg_numeric_mae(g["_numeric_abs_errors"]),
                    "numeric_rmse": agg_numeric_rmse(g["_numeric_sq_errors"]),
                    "anomaly_f1_macro": agg_f1_macro(g),
                })

            lb2 = pd.DataFrame(rows2).sort_values(
                ["family", "table_rows", "pass_rate", "avg_latency_ms"],
                ascending=[True, True, False, True],
            )
            out_path2 = os.path.join(run_dir, "leaderboard_by_family_rows.parquet")
            lb2.to_parquet(out_path2, index=False)
            print(f"Wrote: {out_path2}")

            try:
                maybe_write_csv(lb2, run_dir, "leaderboard_by_family_rows.csv")
            except Exception:
                pass

def cmd_all(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["run"]["output_dir"]
    cmd_run(config_path)
    cmd_score(out_dir)
    cmd_aggregate(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Energy LLM Benchmark (EcoLogits for online OpenAI)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("all", help="run + score + aggregate")
    p_all.add_argument("--config", required=True)

    p_run = sub.add_parser("run", help="run only")
    p_run.add_argument("--config", required=True)

    p_score = sub.add_parser("score", help="score only")
    p_score.add_argument("--run-dir", required=True)

    p_agg = sub.add_parser("aggregate", help="aggregate only")
    p_agg.add_argument("--run-dir", required=True)

    args = ap.parse_args()

    if args.cmd == "all":
        cmd_all(args.config)
    elif args.cmd == "run":
        cmd_run(args.config)
    elif args.cmd == "score":
        cmd_score(args.run_dir)
    elif args.cmd == "aggregate":
        cmd_aggregate(args.run_dir)
    else:
        raise ValueError("unknown command")


if __name__ == "__main__":
    main()
