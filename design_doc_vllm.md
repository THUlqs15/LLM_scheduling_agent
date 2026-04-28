# LARRYSmith: LLM-Guided Hyperparameter Optimization for LARRY v6

## Instructions for Claude Code

You are an LLM scheduling researcher. Your mission is to find the optimal hyperparameter configuration for the LARRY v6 scheduling algorithm by running iterative experiments on a vLLM server.


**Read this entire document carefully before doing anything.** Then follow the execution plan step by step.

---

## 0. Environment Configuration

```
CONDA_ENV         = myvllm           # editable vLLM lives here
VLLM_PROJECT      = /workspace/lqs3/LLM_scheduling/vllm
AGENT_DIR         = /workspace/lqs3/LLM_scheduling/LLM_agent   # this project's working dir
RESULTS_DIR       = ${AGENT_DIR}/larry_results
LARRY_CFG_DIR     = ${AGENT_DIR}/larry_configs
SERVER_LOG_DIR    = ${AGENT_DIR}/server_logs
DATASET           = ~/.etc/ShareGPT_V3_unfiltered_cleaned_split.json
MODEL             = Butter_L3_8B_RPMaster_v2     
SERVER_HOST       = 127.0.0.1
SERVER_PORT       = 8000
SERVER_BASE       = http://${SERVER_HOST}:${SERVER_PORT}
```

**IMPORTANT — environment hygiene rules:**

1. All Python / vLLM commands MUST run inside `conda run -n myvllm --no-banner ...` (or after `conda activate myvllm`). Do not use the base interpreter.
2. **All code, configs, logs, and results this project produces go under `${AGENT_DIR}` (i.e. `/workspace/lqs3/LLM_scheduling/LLM_agent`).** Do not pollute the vLLM source tree with experiment artifacts.
3. You MAY edit any file under `${VLLM_PROJECT}` to install the LARRY hook, but every edit must be guarded by an env flag (default OFF) so that an unsuspecting user invoking `vllm` normally sees zero behavior change. The flag is `VLLM_USE_LARRY` (described in §2). The repo on disk must remain a working stock vLLM when the flag is unset.
4. Any extra Python packages installed for this project must be uninstalled at the end (`pip uninstall ... -y`).
5. Confirm the conda env exists before doing anything: `conda run -n myvllm --no-banner python -c "import vllm; print(vllm.__version__, vllm.__file__)"`. The path printed must be inside `${VLLM_PROJECT}` (proving the editable install is the one being used).

---

## 1. Background: LARRY v6 Algorithm

LARRY v6 is a priority-based scheduler for LLM inference that replaces the default FCFS policy. It scores each waiting request and schedules the highest-scoring one first.

### 1.1 Scoring Formula

```
v6_score = alpha × wait_time
         - q_len × effective_remaining × (1 + AMPLIFIER × decode_pressure)
         + CACHE_WEIGHT × cached
         + PROGRESS_WEIGHT × turns_completed
         + continuity_bonus(adaptive)
         + SHORT_BOOST × I(eff_remaining ≤ SHORT_THRESHOLD)

where:
  decode_pressure     = min(1.0, num_running / DECODE_PRESSURE_THRESHOLD)
  effective_remaining = max(0, prompt_tokens - num_computed_tokens - cached_tokens)
  alpha               = ALPHA_BASE × max(1, q_len / 16)
  continuity_bonus    = adaptive_bonus × exp(-elapsed × ln2 / DECAY)
  adaptive_bonus      = clamp(BASE_BONUS × avg_prompt / REFERENCE_LEN,
                              MIN_BONUS, MAX_BONUS)
```



### 1.2 Hyperparameters and Search Space

Formula symbol names are used below; the corresponding `LarryConfig` field names (impl) follow in parentheses where they differ.

| Parameter (impl name) | Default | Range | Type | Description |
|-----------|---------|-------|------|-------------|
| ALPHA_BASE | 10240 | [1000, 200000] | int | Aging weight base. alpha = ALPHA_BASE × max(1, q_len/16) |
| MIN_QUEUE | 4 | [4, 32] | int | Skip reordering when queue ≤ this; see §1.3 for load-adaptive semantics |
| CACHE_WEIGHT | 2048 | [500, 20000] | int | Score bonus per cached prefix token |
| CACHE_PROBE_INTERVAL | 4 | [1, 16] | int | Probe prefix cache every N scheduling rounds |
| PROGRESS_WEIGHT (SESSION_PROGRESS_WEIGHT) | 4096 | [0, 50000] | int | Bonus per completed session turn |
| CONTINUITY_BONUS | 100000 | [10000, 1000000] | int | Peak continuity bonus (static fallback when avg_prompt_len = 0) |
| DECAY (CONTINUITY_DECAY) | 60.0 | [10, 300] | float | Half-life seconds for continuity decay |
| BASE_BONUS (ADAPTIVE_BASE_BONUS) | 50000 | [5000, 500000] | int | Reference bonus for adaptive continuity |
| REFERENCE_LEN (ADAPTIVE_REFERENCE_LEN) | 30000 | [5000, 100000] | int | Reference prompt length for scaling |
| MIN_BONUS (ADAPTIVE_MIN_BONUS) | 10000 | [0, 100000] | int | Lower clamp for adaptive bonus |
| MAX_BONUS (ADAPTIVE_MAX_BONUS) | 200000 | [50000, 2000000] | int | Upper clamp for adaptive bonus |
| AMPLIFIER (PRESSURE_AMPLIFIER) | 4.0 | [0.5, 10.0] | float | Decode pressure multiplier |
| DECODE_PRESSURE_THRESHOLD | 3 | [1, 10] | int | Running requests for full pressure |
| SHORT_BOOST (SHORT_PREFILL_BOOST) | 0 | [0, 500000] | int | Flat bonus for short-prefill requests |
| SHORT_THRESHOLD (SHORT_PREFILL_THRESHOLD) | 8192 | [512, 32768] | int | Token threshold for "short prefill" |

**Constraints**: MIN_BONUS < BASE_BONUS ≤ MAX_BONUS, DECAY > 0, all integer params ≥ 0.

### 1.3 Design Principles

- **Higher score = higher priority** (scheduled first).
- The formula balances aging fairness, work cost, cache affinity, session progress, continuity for hot cache, and decode-pressure awareness.
- Under decode pressure, large uncached prefills are penalized → favoring short / cached requests.
- When `effective_remaining = 0` (fully cached), the pressure penalty vanishes → cache-warm requests are immune to pressure.
- **MIN_QUEUE provides load-adaptive behavior.** When the waiting queue is shallow (≤ MIN_QUEUE), LARRY skips reordering entirely and falls back to FCFS. This means a single MIN_QUEUE value can behave very differently across load regimes: at high load (burst arrival, deep queue) LARRY is always active; at medium load (steady-state, shallow queue) LARRY stays dormant and preserves FCFS fairness. The effective value of MIN_QUEUE must be calibrated against the typical peak queue depth at each target load level — it is not a simple "minimum size" knob but the primary mechanism controlling when SRPT scheduling is safe to apply.

---

## 2. vLLM Integration: Patching the Real Scheduler

### 2.1 Locate the scheduler

vLLM's runtime layout has changed across releases. **Discover before you patch:**

```bash
conda run -n myvllm --no-banner python -c "
import vllm, os, inspect
print('vllm version:', vllm.__version__)
print('vllm path   :', os.path.dirname(vllm.__file__))
# Try v1 first (current default since 0.7+)
try:
    from vllm.v1.core.sched.scheduler import Scheduler as S1
    print('v1 scheduler:', inspect.getsourcefile(S1))
except Exception as e:
    print('no v1 scheduler:', e)
# Then v0
try:
    from vllm.core.scheduler import Scheduler as S0
    print('v0 scheduler:', inspect.getsourcefile(S0))
except Exception as e:
    print('no v0 scheduler:', e)
"
```

Prefer the **v1** scheduler (`vllm/v1/core/sched/scheduler.py`) — it is the default in vLLM 0.11.x and is what the ShareGPT benchmark exercises. If only v0 is present, fall back to `vllm/core/scheduler.py`. If both exist, patch v1; vLLM 0.11.x runs v1 by default unless `VLLM_USE_V1=0`.

Identify in the source the place where the waiting queue is consumed in priority order. In v1 this is at the top of `Scheduler.schedule()` before the prefill / decode loop iterates `self.waiting`. The patch entry point is "right after we have the up-to-date `self.waiting` and `self.running` snapshots, before the dispatch loop begins."

### 2.2 The LARRY config + scoring module

Create **`${VLLM_PROJECT}/vllm/v1/core/sched/larry_hook.py`** (or `vllm/core/larry_hook.py` for v0). The file lives inside the vLLM tree on purpose so it is importable from the patched scheduler with no PYTHONPATH gymnastics.

```python
# vllm/.../larry_hook.py
"""LARRY v6 hyperparameter hook for the real vLLM scheduler.

The hook is OFF by default. It only activates when the env var
VLLM_USE_LARRY=1 is set at process start. When OFF, this module
imposes zero overhead: the patched scheduler short-circuits before
ever calling into here.
"""
from __future__ import annotations
import json, math, os, time, logging, threading
from dataclasses import dataclass, field, fields
from typing import Optional

logger = logging.getLogger("vllm.larry")

ENV_FLAG          = "VLLM_USE_LARRY"
ENV_CONFIG_PATH   = "LARRY_CONFIG_PATH"
ENV_LOG_PATH      = "LARRY_LOG_PATH"          # optional jsonl trace
ENV_RELOAD_EVERY  = "LARRY_RELOAD_EVERY_STEPS"  # default 50

@dataclass
class LarryConfig:
    ALPHA_BASE: int = 10240
    MIN_QUEUE: int = 4
    CACHE_WEIGHT: int = 2048
    CACHE_PROBE_INTERVAL: int = 4
    SESSION_PROGRESS_WEIGHT: int = 4096
    CONTINUITY_BONUS: int = 100000
    CONTINUITY_DECAY: float = 60.0
    ADAPTIVE_BASE_BONUS: int = 50000
    ADAPTIVE_REFERENCE_LEN: int = 30000
    ADAPTIVE_MIN_BONUS: int = 10000
    ADAPTIVE_MAX_BONUS: int = 200000
    PRESSURE_AMPLIFIER: float = 4.0
    DECODE_PRESSURE_THRESHOLD: int = 3
    SHORT_PREFILL_BOOST: int = 0
    SHORT_PREFILL_THRESHOLD: int = 8192

    @classmethod
    def from_dict(cls, d: dict) -> "LarryConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class LarryRuntime:
    """Singleton holding live config + hot-reload state."""
    _instance: Optional["LarryRuntime"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.enabled: bool = os.environ.get(ENV_FLAG, "0") == "1"
        self.config_path: Optional[str] = os.environ.get(ENV_CONFIG_PATH)
        self.log_path: Optional[str] = os.environ.get(ENV_LOG_PATH)
        self.reload_every: int = int(os.environ.get(ENV_RELOAD_EVERY, "50"))
        self._config: LarryConfig = LarryConfig()
        self._cfg_mtime: float = 0.0
        self._cfg_version: int = 0
        self._step: int = 0
        self._cache_probe_round: int = 0
        if self.enabled:
            self._load_now(force=True)
            logger.warning(
                "[LARRY] enabled. config_path=%s version=%d", self.config_path, self._cfg_version
            )
        else:
            logger.info("[LARRY] disabled (set VLLM_USE_LARRY=1 to enable)")

    @classmethod
    def get(cls) -> "LarryRuntime":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_now(self, force: bool = False) -> None:
        if not self.config_path or not os.path.exists(self.config_path):
            return
        try:
            mtime = os.path.getmtime(self.config_path)
            if not force and mtime == self._cfg_mtime:
                return
            with open(self.config_path) as f:
                raw = json.load(f)
            self._config = LarryConfig.from_dict(raw)
            self._cfg_mtime = mtime
            self._cfg_version += 1
            logger.warning(
                "[LARRY] reload v%d from %s: %s",
                self._cfg_version, self.config_path, raw,
            )
        except Exception as e:
            logger.exception("[LARRY] reload failed: %s", e)

    @property
    def config(self) -> LarryConfig:
        return self._config

    @property
    def version(self) -> int:
        return self._cfg_version

    def maybe_reload(self) -> None:
        """Cheap mtime check, called from schedule(); reloads at most once per
        reload_every scheduling steps to bound stat() overhead."""
        self._step += 1
        if self._step % max(1, self.reload_every) != 0:
            return
        self._load_now(force=False)

    def should_probe_cache(self) -> bool:
        self._cache_probe_round = (self._cache_probe_round + 1) % max(
            1, self._config.CACHE_PROBE_INTERVAL
        )
        return self._cache_probe_round == 0


def compute_score(
    cfg: LarryConfig,
    *,
    wait_time: float,
    prompt_tokens: int,
    num_computed_tokens: int,
    cached_tokens: int,
    queue_length: int,
    num_running: int,
    turns_completed: int = 0,
    last_seen_elapsed: float = math.inf,
    avg_prompt_len: float = 0.0,
) -> float:
    q_len = max(1, queue_length)
    alpha = cfg.ALPHA_BASE * max(1.0, q_len / 16.0)
    aging = alpha * wait_time

    eff_remaining = max(0, prompt_tokens - num_computed_tokens - cached_tokens)

    if cfg.DECODE_PRESSURE_THRESHOLD > 0:
        pressure = min(1.0, num_running / cfg.DECODE_PRESSURE_THRESHOLD)
    else:
        pressure = 0.0
    work_penalty = q_len * eff_remaining * (1.0 + cfg.PRESSURE_AMPLIFIER * pressure)

    cache_bonus = cfg.CACHE_WEIGHT * cached_tokens

    progress_bonus = cfg.SESSION_PROGRESS_WEIGHT * turns_completed

    if avg_prompt_len > 0 and cfg.ADAPTIVE_REFERENCE_LEN > 0:
        adaptive_bonus = cfg.ADAPTIVE_BASE_BONUS * avg_prompt_len / cfg.ADAPTIVE_REFERENCE_LEN
        adaptive_bonus = max(cfg.ADAPTIVE_MIN_BONUS, min(cfg.ADAPTIVE_MAX_BONUS, adaptive_bonus))
    else:
        adaptive_bonus = cfg.CONTINUITY_BONUS
    if cfg.CONTINUITY_DECAY > 0 and math.isfinite(last_seen_elapsed):
        decay = math.exp(-last_seen_elapsed * math.log(2) / cfg.CONTINUITY_DECAY)
    else:
        decay = 0.0
    continuity = adaptive_bonus * decay

    short_boost = (
        cfg.SHORT_PREFILL_BOOST
        if (cfg.SHORT_PREFILL_BOOST > 0 and eff_remaining <= cfg.SHORT_PREFILL_THRESHOLD)
        else 0.0
    )

    return aging - work_penalty + cache_bonus + progress_bonus + continuity + short_boost
```

### 2.3 Patching the v1 scheduler

In `${VLLM_PROJECT}/vllm/v1/core/sched/scheduler.py`:

1. Add the import near the other vllm-internal imports:
   ```python
   from vllm.v1.core.sched.larry_hook import LarryRuntime, compute_score
   ```

2. Add a method on `Scheduler` (file-local, not exported):
   ```python
   def _larry_reorder_waiting(self) -> None:
       """LARRY v6 priority reorder. No-op when VLLM_USE_LARRY != '1'."""
       rt = LarryRuntime.get()
       if not rt.enabled:
           return
       rt.maybe_reload()
       cfg = rt.config

       waiting = self.waiting
       try:
           q_len = len(waiting)
       except TypeError:
           # Some vLLM versions wrap waiting in a custom queue class.
           waiting = list(waiting)
           q_len = len(waiting)
       if q_len <= cfg.MIN_QUEUE:
           return

       num_running = len(self.running)
       now = time.monotonic()
       probe_cache = rt.should_probe_cache()

       # Optional: average prompt length for adaptive continuity bonus.
       try:
           avg_prompt_len = sum(r.num_prompt_tokens for r in waiting) / q_len
       except Exception:
           avg_prompt_len = 0.0

       reqs_with_score = []
       for req in waiting:
           prompt_tokens = getattr(req, "num_prompt_tokens",
                                   getattr(req, "prompt_token_ids", []) and len(req.prompt_token_ids) or 0)
           num_computed = getattr(req, "num_computed_tokens", 0)
           # arrival timestamp; vLLM v1 stores arrival_time on Request
           arrival = getattr(req, "arrival_time", None) or getattr(req, "arrival_ts", None) or now
           wait_time = max(0.0, now - arrival)

           cached_tokens = 0
           if probe_cache and hasattr(self, "kv_cache_manager"):
               try:
                   # v1 returns (computed_blocks, num_computed_tokens) for an unscheduled request
                   _, ct = self.kv_cache_manager.get_computed_blocks(req)
                   cached_tokens = int(ct)
                   # cache the result on the request to avoid re-probing each step
                   req._larry_cached_tokens = cached_tokens
               except Exception:
                   cached_tokens = getattr(req, "_larry_cached_tokens", 0)
           else:
               cached_tokens = getattr(req, "_larry_cached_tokens", 0)

           score = compute_score(
               cfg,
               wait_time=wait_time,
               prompt_tokens=prompt_tokens,
               num_computed_tokens=num_computed,
               cached_tokens=cached_tokens,
               queue_length=q_len,
               num_running=num_running,
               avg_prompt_len=avg_prompt_len,
           )
           reqs_with_score.append((score, req))

       reqs_with_score.sort(key=lambda x: x[0], reverse=True)

       # Replace the waiting queue contents in priority order.
       # Use whatever public mutation the queue type supports.
       sorted_reqs = [r for _, r in reqs_with_score]
       try:
           # If `self.waiting` is a list / deque
           waiting.clear()
           for r in sorted_reqs:
               waiting.append(r)
       except AttributeError:
           # If it's an opaque RequestQueue — rebuild it
           cls = type(self.waiting)
           self.waiting = cls(sorted_reqs) if sorted_reqs else cls()
   ```

3. At the **very top of `Scheduler.schedule()`**, before any iteration of `self.waiting`, add a single line:
   ```python
   self._larry_reorder_waiting()
   ```

4. Add the standard library import at the top of the file if it isn't there:
   ```python
   import time
   ```

The hook is **inert** when `VLLM_USE_LARRY` is not `"1"`. `LarryRuntime.get()` initializes with `enabled=False` and `_larry_reorder_waiting()` returns immediately, so vanilla `vllm serve ...` is byte-identical to upstream behavior.

### 2.4 Sanity check after patching

```bash
# Patched module imports cleanly
conda run -n myvllm --no-banner python -c "from vllm.v1.core.sched.larry_hook import LarryRuntime; print(LarryRuntime.get().enabled)"
# Should print: False

# Stock vLLM still imports & launches
conda run -n myvllm --no-banner python -c "import vllm; print(vllm.__version__)"
```

Then run a 30-request smoke benchmark with the flag OFF (§4.2) and confirm metrics match a pre-patch baseline within noise. Only then proceed to §3.

---

## 3. Experiment Harness

All harness scripts live in `${AGENT_DIR}/agent/`. Create them as you go.

### 3.1 Directory layout

```
${AGENT_DIR}/
  design_doc_vllm.md          (this file — you may copy it here for reference)
  result.md                   (running log of every round)
  larry_configs/              (one config_<id>.json per candidate)
  larry_results/              (one result_<id>.json + metrics row per candidate)
  server_logs/                (vllm server stdout/stderr per server-lifetime)
  agent/
    start_server.sh           (launches the long-lived vllm serve process)
    stop_server.sh
    run_one.py                (writes config, runs benchmark, saves metrics)
    parse_metrics.py          (extracts the canonical metric set from bench JSON)
    score.py                  (composite score vs FCFS baseline)
    summarize.py              (rebuilds result.md tables from larry_results/)
```

### 3.2 Long-lived server

`agent/start_server.sh` template:

```bash
#!/usr/bin/env bash
set -euo pipefail
AGENT_DIR=/workspace/lqs3/LLM_scheduling/LLM_agent
LARRY_CFG=${AGENT_DIR}/larry_configs/active.json     # the file the agent rewrites between candidates
LOG=${AGENT_DIR}/server_logs/server_$(date +%Y%m%d_%H%M%S).log

mkdir -p "${AGENT_DIR}/server_logs" "${AGENT_DIR}/larry_configs" "${AGENT_DIR}/larry_results"

# Make sure an initial config file exists so the hook doesn't ignore the path
if [ ! -f "${LARRY_CFG}" ]; then
  echo '{}' > "${LARRY_CFG}"
fi

export VLLM_USE_LARRY=1
export LARRY_CONFIG_PATH=${LARRY_CFG}
export LARRY_RELOAD_EVERY_STEPS=20
# Echo to make the env unambiguous in the log
env | grep -E '^(VLLM_USE_LARRY|LARRY_)' >> "${LOG}"

conda run -n myvllm --no-banner --no-capture-output \
    vllm serve Butter_L3_8B_RPMaster_v2 \
        --host 127.0.0.1 --port 8000 \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --enable-prefix-caching \
        --disable-log-requests \
    >> "${LOG}" 2>&1 &
echo $! > "${AGENT_DIR}/server_logs/server.pid"
echo "Server PID $(cat ${AGENT_DIR}/server_logs/server.pid), logging to ${LOG}"
```

`agent/stop_server.sh`:
```bash
#!/usr/bin/env bash
PIDFILE=/workspace/lqs3/LLM_scheduling/LLM_agent/server_logs/server.pid
[ -f "$PIDFILE" ] && kill $(cat "$PIDFILE") 2>/dev/null || true
rm -f "$PIDFILE"
```

**Wait for readiness** before any benchmark:
```bash
for i in $(seq 1 120); do
  if curl -fs http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "server up"; exit 0
  fi
  sleep 2
done
echo "server failed to start"; exit 1
```

The same server stays up for the entire experiment session. `${LARRY_CFG}` (`active.json`) is the one file the harness rewrites between candidates.

### 3.3 One candidate run

`agent/run_one.py` (sketch — implement fully in code):

```
Inputs:
  --config-id   e.g. r3_c2 or fcfs_baseline
  --config-json path-to-candidate-config-json (or "fcfs" → write {} and disable hook for this run only)
  --rates       comma-sep, e.g. "inf,4,2"
  --num-prompts 512
  --dataset     ${DATASET}
  --model       Butter_L3_8B_RPMaster_v2
  --base-url    http://127.0.0.1:8000

For each rate:
  1. Copy candidate JSON over ${AGENT_DIR}/larry_configs/active.json
     (atomic: write to .tmp, os.replace).
     For "fcfs" runs, write a config with MIN_QUEUE=10**9 (so reorder always no-ops)
     instead of restarting the server.
  2. Sleep ~ (LARRY_RELOAD_EVERY_STEPS * avg_step_time) seconds, or send a single
     short warm-up request, to ensure the new config is picked up.
  3. Invoke the bench client (see §3.4). Capture its --save-result JSON.
  4. parse_metrics.py → row in larry_results/<config_id>_rate<R>.json plus
     append a flat row to larry_results/all_runs.csv.
```

Always log the LARRY hook's reload version that was active when the run completed (grep `[LARRY] reload v` from the server log) so we can audit "did this config actually take effect?"

### 3.4 Bench client

vLLM ships `benchmarks/benchmark_serving.py`. From `${VLLM_PROJECT}`:

```bash
conda run -n myvllm --no-banner python ${VLLM_PROJECT}/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model Butter_L3_8B_RPMaster_v2 \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path ~/.etc/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 512 \
    --request-rate <RATE>            # use "inf" for max-throughput
    --save-result \
    --result-dir ${AGENT_DIR}/larry_results \
    --result-filename <config_id>_rate<RATE>.json
```

The output JSON contains, among others: `duration`, `total_input_tokens`, `total_output_tokens`, `request_throughput`, `output_throughput`, `mean_ttft_ms`, `p99_ttft_ms`, `mean_tpot_ms`, `p99_tpot_ms`. These are the canonical metrics. **Use the same set of metric names result.md already records.**

If the installed vLLM version's `benchmark_serving.py` has slightly different flags, adapt — but keep the metric names normalized in `parse_metrics.py`.

### 3.5 Composite score (vs FCFS baseline)

Same formula as the simulator doc (so cross-run comparison works):

```
score(cur, base) =
    0.35 * (base_duration      - cur_duration)      / base_duration
  + 0.20 * (cur_throughput     - base_throughput)   / base_throughput
  + 0.15 * (base_mean_ttft     - cur_mean_ttft)     / base_mean_ttft
  + 0.10 * (base_p99_ttft      - cur_p99_ttft)      / base_p99_ttft
  + 0.10 * (base_mean_tpot     - cur_mean_tpot)     / base_mean_tpot
  + 0.10 * (base_p99_tpot      - cur_p99_tpot)      / base_p99_tpot
```

Compute per-rate, and report a per-rate score plus a simple mean over rates as the "headline" score.

### 3.6 Variance handling

Real GPU runs have variance — the simulator's "deterministic" assumption does **not** apply. Default protocol:

- During exploration / narrowing rounds, run each candidate **once** per rate — the cost of false confidence is offset by trying many configs.
- Before declaring a winner, run the **top-3** candidates **3× each** at every rate. Use the **median** for the final score.
- If two configs are within 1% composite score after the 3-run median, treat them as tied and prefer the simpler one (fewer non-default knobs).

---

## 4. Execution Plan

Run these steps **IN ORDER**. After each experiment, append to `${AGENT_DIR}/result.md`.

### Step 1: Setup

1. Verify `${VLLM_PROJECT}` is the editable install (`vllm.__file__` test from §0).
2. `mkdir -p ${AGENT_DIR}/{larry_configs,larry_results,server_logs,agent}`
3. Apply the patch (§2.2 + §2.3). Run the §2.4 sanity check.
4. Implement `agent/start_server.sh`, `stop_server.sh`, `run_one.py`, `parse_metrics.py`, `score.py`, `summarize.py`.

### Step 2: FCFS baseline

Start the server with `VLLM_USE_LARRY=0` (override the env in `start_server.sh` for this one run, OR keep it at `1` but write an `active.json` whose `MIN_QUEUE` is huge — both are equivalent; the env-flag-off path is cleaner for the baseline because it also exercises the "no hook overhead" guarantee).

```bash
bash ${AGENT_DIR}/agent/start_server.sh   # with VLLM_USE_LARRY temporarily unset
# wait for /v1/models
conda run -n myvllm --no-banner python ${AGENT_DIR}/agent/run_one.py \
    --config-id fcfs_baseline --config-json fcfs \
    --rates inf,4,2 --num-prompts 512
bash ${AGENT_DIR}/agent/stop_server.sh
```

Record per-rate metrics in `result.md` under "FCFS Baseline". From here on, the baseline numbers are frozen — every subsequent score is computed against these.

### Step 3: Bring the hook online + default LARRY config

Restart the server with `VLLM_USE_LARRY=1`. Write the default config to `${AGENT_DIR}/larry_configs/config_default.json`:

```json
{
    "ALPHA_BASE": 10240, "MIN_QUEUE": 4, "CACHE_WEIGHT": 2048,
    "CACHE_PROBE_INTERVAL": 4, "SESSION_PROGRESS_WEIGHT": 4096,
    "CONTINUITY_BONUS": 100000, "CONTINUITY_DECAY": 60.0,
    "ADAPTIVE_BASE_BONUS": 50000, "ADAPTIVE_REFERENCE_LEN": 30000,
    "ADAPTIVE_MIN_BONUS": 10000, "ADAPTIVE_MAX_BONUS": 200000,
    "PRESSURE_AMPLIFIER": 4.0, "DECODE_PRESSURE_THRESHOLD": 3,
    "SHORT_PREFILL_BOOST": 0, "SHORT_PREFILL_THRESHOLD": 8192
}
```

```bash
conda run -n myvllm --no-banner python ${AGENT_DIR}/agent/run_one.py \
    --config-id default --config-json ${AGENT_DIR}/larry_configs/config_default.json \
    --rates inf,4,2 --num-prompts 512
```

After this run, **grep the server log** for `[LARRY] reload v1` AND `[LARRY] enabled.` — both must be present. If neither shows up the hook didn't activate; debug before proceeding.

### Step 4: Iterative search loop (10–15 rounds, 5–8 candidates each)

#### Per round:

1. **Analyze** previous results: which configs performed best per rate? Which parameters seem most influential? Sanity-check that the LARRY version logged in the server log changed for each candidate — if it did not, `run_one.py` is not actually triggering hot-reload, fix the harness before running more configs.

2. **Propose** 5–8 candidates:
   - 3–4 *exploitation* configs: small perturbations from the running best.
   - 1–2 *exploration* configs: dramatically different parameter combinations.
   - 1–2 *interpolation* configs: blend two good configs.

3. **Run** each candidate via `run_one.py`. Configs go in `${AGENT_DIR}/larry_configs/config_r<N>_c<M>.json`, results in `${AGENT_DIR}/larry_results/`.

4. **Record** in `result.md` with a per-rate metric table.

5. **Reason** about WHY certain configs performed better. The reasoning is as important as the numbers — write it down before the next round.

#### Search strategy

**Parameter priority order (most to least impactful in practice):**

1. **MIN_QUEUE** — the dominant knob. Must exceed the typical peak waiting-queue depth at medium load to suppress SRPT tail starvation. Its effective range depends on `max_num_seqs` and arrival rate; values in [16, 32] are usually needed. This should be the first axis explored.
2. **ALPHA_BASE** — controls aging aggressiveness. Too low → long requests starve; too high → aging dominates and SRPT loses effect. Typically [50k, 150k] is the productive range.
3. **AMPLIFIER** — decode-pressure sensitivity. Moderate values (0.5–2.0) tend to work; extremes hurt one rate or another.
4. **CACHE_WEIGHT, SHORT_BOOST / SHORT_THRESHOLD** — secondary. SHORT_BOOST is a no-op if SHORT_THRESHOLD covers most requests (e.g. 8192 tokens for ShareGPT). CACHE_WEIGHT helps only when prefix-cache hit rate is significant.
5. **Continuity / session bonus group** (PROGRESS_WEIGHT, BASE_BONUS, REFERENCE_LEN, MIN_BONUS, MAX_BONUS, DECAY, CONTINUITY_BONUS) — near-zero impact for single-turn workloads. Zero these out early and focus budget elsewhere.

**Round 1–2 (exploration).** Establish that ALPHA_BASE is large enough to avoid starvation (≥ 50k). Sweep MIN_QUEUE across [4, 8, 16, 20, 25] to locate the load-adaptive threshold. Try a few AMPLIFIER values. Zero out continuity/session bonus terms from the start.

**Round 3–5 (narrowing).** Grid search MIN_QUEUE × ALPHA_BASE around the best found pair. Fix AMPLIFIER. Only revisit CACHE_WEIGHT or SHORT_BOOST if rate=inf results plateau.

**Round 6–8 (fine-tuning).** Tiny adjustments to MIN_QUEUE (±2–3) and ALPHA_BASE (±20%). Re-run the FCFS baseline at the end of the session to confirm the box hasn't drifted thermally; if it has, recompute composite scores.

#### Per-rate scoring priorities

Same weights as §3.5 above. Headline = mean over rates {inf, 4, 2} req/s. Per-rate ranks are kept in result.md so we can spot regimes where LARRY helps high-load throughput at one rate but hurts another.

### Step 5: Final verification & output

When you've converged:

1. Run the **top-3** configs **3× per rate** (variance protocol from §3.6). Take the median.
2. Save the winner as `${AGENT_DIR}/larry_configs/best_config.json`.
3. Do one **clean restart** verification run: `stop_server.sh`, `start_server.sh`, run the winner with the same rates. Confirm the medians hold within ~2%.
4. Make sure `${AGENT_DIR}/result.md` has the full log: baseline, every round, the 3-replicate verification table, and the final best config. The doc-template in §5 of this file is the contract.

---

## 5. result.md Template

```markdown
# LARRYSmith Real-vLLM Optimization Results

## Environment
- Dataset: ~/.etc/ShareGPT_V3_unfiltered_cleaned_split.json
- vLLM: <version> editable @ /workspace/lqs3/LLM_scheduling/vllm
- Engine: v1 (or v0 — note which)
- Model: Butter_L3_8B_RPMaster_v2 on <GPU/host>
- Server flags: TP=1, PP=1, max_num_seqs=32, max_num_batched_tokens=8192, prefix-caching=on
- Workload: 512 ShareGPT requests; rates {inf, 4, 2, 1} req/s
- Date: <date>

## FCFS Baseline (frozen reference)
| Rate | duration (s) | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | throughput (t/s) |
|------|--------------|--------------|-------------|--------------|-------------|------------------|
| inf  | ...          | ...          | ...         | ...          | ...         | ...              |
| 4    | ...          | ...          | ...         | ...          | ...         | ...              |
| 2    | ...          | ...          | ...         | ...          | ...         | ...              |

## LARRY default
| Rate | duration | mean_ttft | p99_ttft | mean_tpot | p99_tpot | throughput | score vs FCFS |
| ...  | ...      | ...       | ...      | ...       | ...      | ...        | ...           |

## Iteration Log

### Round 1: Exploration
**Strategy**: ...

| Config ID | non-default knobs | rate | duration | mean_ttft | p99_ttft | mean_tpot | p99_tpot | thr | score |
|-----------|-------------------|------|----------|-----------|----------|-----------|----------|-----|-------|
| r1_c1     | ALPHA_BASE=100000 | inf  | ...      | ...       | ...      | ...       | ...      | ... | ...   |

**Analysis**: ...
**Best this round**: r1_cX (composite ...)

### Round 2: ...

## Verification (top-3 × 3 replicates)
| Config | rate | run | duration | mean_ttft | p99_ttft | ... | score |

## Convergence Summary
- Total experiments: N (M unique configs, K replicate runs)
- Best config found in round: ...
- Improvement over FCFS (median, mean over rates): X% / Y%
- Per-rate improvements: inf=...%, 4=...%, 2=...%

## Best Configuration
{JSON}

## Key Findings
1. ...
2. ...
3. ...

## Patch Audit
- vLLM file modified: vllm/v1/core/sched/scheduler.py (+ larry_hook.py added)
- Behavior with VLLM_USE_LARRY unset: verified identical to upstream (smoke test result: ...)
- Cleanup status: server stopped, no lingering processes, no extra pip installs left behind.
```

---

## 6. Important Notes

- **Always confirm a candidate actually took effect.** After every `run_one.py` invocation, grep the server log for the new `[LARRY] reload v<n>` line whose timestamp falls inside the run window. If you don't see it, the bench ran on the previous config — the result is invalid.
- **Always confirm `VLLM_USE_LARRY=0` reproduces stock vLLM** (§2.4). This is the contract that lets the user keep using vLLM normally outside this project.
- **SRPT tail starvation at medium load is the dominant failure mode.** Even with moderate parameters, if MIN_QUEUE is too small, LARRY activates at medium load (e.g. rate=4) and continuously re-prioritizes new short arrivals over long requests that have been waiting. This shows up as p99_TTFT regressing 100–300% versus FCFS while mean_TTFT looks fine. The fix is always MIN_QUEUE — not ALPHA_BASE or AMPLIFIER. Do not mistake this for "aggressive starvation" caused by extreme parameters; it happens with perfectly reasonable ALPHA_BASE values.
- **If a config destabilizes the server** (OOM, request stuck, NaNs in metrics): record it as `FAILED`, kill the server, restart, continue. True aggressive starvation (e.g. ALPHA_BASE=0 + MIN_QUEUE=1 + huge SHORT_BOOST) is also possible.
- **Real runs are noisy.** Don't chase 0.5% differences in the exploration phase. Save the variance-aware comparisons for the verification step.
- **Don't mix server lifetimes between rates without saying so.** If you must restart the server mid-round (e.g. to apply a max_num_seqs change), call out which rows came from which lifetime in result.md.
- **Record EVERYTHING** in result.md — failed experiments, surprising observations, harness bugs you found and fixed, and any deviations from this document. If you change this doc, note it in result.md with a diff summary.
- **Cleanup.** At the end of the session: stop the server, remove `${AGENT_DIR}/server_logs/server.pid`, and `pip uninstall` anything you added to `myvllm` for the harness. The vLLM patch itself stays — but it must remain inert when `VLLM_USE_LARRY` is unset.
