# LARRYSmith: LLM-Guided Hyperparameter Optimization for LARRY v6

## Instructions for Claude Code

You are an LLM scheduling researcher. Your mission is to find the optimal hyperparameter configuration for the LARRY v6 scheduling algorithm by running iterative experiments on the ROSS simulator.

**Read this entire document carefully before doing anything.** Then follow the execution plan step by step.

---

## 0. Environment Configuration

```
CONDA_ENV=simulator
PROJECT_DIR=/workspace/lqs3/ross
ROSS_CMD="cd ${PROJECT_DIR}/ross && python ross_predict.py"
ROSS_CONFIG=${PROJECT_DIR}/ross/config/test_vllm.json
RESULTS_DIR=${PROJECT_DIR}/larry_results
DATASET=~/.etc/ShareGPT_V3_unfiltered_cleaned_split.json
```

**IMPORTANT**: All commands MUST be run inside `conda activate simulator`. Always prefix commands with:
```bash
cd /workspace/lqs3/ross/ross && conda run -n simulator --no-banner python ross_predict.py ...
```

---

## 1. Background: LARRY v6 Algorithm

LARRY v6 is a priority-based scheduler for LLM inference that replaces the default FIFO policy. It scores each waiting request and schedules the highest-scoring one first.

### 1.1 Scoring Formula

```
v6_score = alpha × wait_time
         - q_len × effective_remaining × (1 + AMPLIFIER × decode_pressure)
         + CACHE_WEIGHT × cached
         + PROGRESS_WEIGHT × turns_completed
         + continuity_bonus(adaptive)
         + SHORT_BOOST × I(eff_remaining ≤ SHORT_THRESHOLD)

where:
  decode_pressure   = min(1.0, num_running / DECODE_PRESSURE_THRESHOLD)
  effective_remaining = max(0, prompt_tokens - num_computed_tokens - cached_tokens)
  alpha             = ALPHA_BASE × max(1, q_len / 16)
  continuity_bonus  = adaptive_bonus × exp(-elapsed × ln2 / DECAY)
  adaptive_bonus    = clamp(BASE_BONUS × avg_prompt / REFERENCE_LEN, MIN_BONUS, MAX_BONUS)
```

### 1.2 Hyperparameters and Search Space

| Parameter | Default | Range | Type | Description |
|-----------|---------|-------|------|-------------|
| ALPHA_BASE | 10240 | [1000, 100000] | int | Aging weight base. alpha = ALPHA_BASE * max(1, q_len/16) |
| MIN_QUEUE | 4 | [1, 16] | int | Skip reordering when queue ≤ this (use FCFS) |
| CACHE_WEIGHT | 2048 | [500, 20000] | int | Score bonus per cached token |
| CACHE_PROBE_INTERVAL | 4 | [1, 16] | int | Probe prefix cache every N rounds |
| SESSION_PROGRESS_WEIGHT | 4096 | [0, 50000] | int | Bonus per completed session turn |
| CONTINUITY_BONUS | 100000 | [10000, 1000000] | int | Peak continuity bonus (static fallback) |
| CONTINUITY_DECAY | 60.0 | [10, 300] | float | Half-life seconds for continuity decay |
| ADAPTIVE_BASE_BONUS | 50000 | [5000, 500000] | int | Reference bonus for adaptive continuity |
| ADAPTIVE_REFERENCE_LEN | 30000 | [5000, 100000] | int | Reference prompt length for scaling |
| ADAPTIVE_MIN_BONUS | 10000 | [0, 100000] | int | Lower clamp for adaptive bonus |
| ADAPTIVE_MAX_BONUS | 200000 | [50000, 2000000] | int | Upper clamp for adaptive bonus |
| PRESSURE_AMPLIFIER | 4.0 | [0.5, 20.0] | float | Decode pressure multiplier |
| DECODE_PRESSURE_THRESHOLD | 3 | [1, 10] | int | Running requests for full pressure |
| SHORT_PREFILL_BOOST | 0 | [0, 500000] | int | Flat bonus for short-prefill requests |
| SHORT_PREFILL_THRESHOLD | 8192 | [1024, 32768] | int | Token threshold for "short prefill" |

**Constraints**: ADAPTIVE_MIN_BONUS < ADAPTIVE_BASE_BONUS ≤ ADAPTIVE_MAX_BONUS, CONTINUITY_DECAY > 0

### 1.3 Design Principles

- **Higher score = higher priority** (scheduled first)
- The formula balances: aging fairness, work cost, cache affinity, session progress, continuity for hot cache, and decode pressure awareness
- Under decode pressure, large uncached prefills are penalized → favoring short/cached requests
- When `effective_remaining = 0` (fully cached), the pressure penalty vanishes → cache-warm requests are immune to pressure

---

## 2. ROSS Integration: Patching the vLLM Scheduler

The ROSS simulator's vLLM scheduler uses FIFO ordering. We need to inject LARRY v6 scoring.

### 2.1 Integration Point

In `ross/vllm_sim/scheduler/scheduler.py`, the method `_schedule_waiting_requests()` always takes `self.waiting[0]` (FIFO). Our patch adds a `_reorder_waiting_queue()` call inside `schedule()` to sort `self.waiting` by LARRY v6 score before the FIFO logic runs.

### 2.2 Patch Code

**Step 1**: Create `ross/vllm_sim/scheduler/larry_config.py`:

```python
"""LARRY v6 hyperparameter configuration for ROSS integration."""
import json
import math
import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def from_env(cls):
        path = os.environ.get("LARRY_CONFIG_PATH")
        if path and os.path.exists(path):
            logger.info(f"Loading LARRY config from {path}")
            return cls.from_json(path)
        return cls()

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


def compute_larry_score(
    cfg,
    wait_time,
    prompt_tokens,
    num_computed_tokens,
    cached_tokens,
    queue_length,
    num_running,
    avg_prompt_len=0,
):
    """Compute LARRY v6 priority score. Higher = higher priority."""
    q_len = queue_length
    alpha = cfg.ALPHA_BASE * max(1, q_len / 16)
    aging_score = alpha * wait_time

    effective_remaining = max(0, prompt_tokens - num_computed_tokens - cached_tokens)
    decode_pressure = min(1.0, num_running / cfg.DECODE_PRESSURE_THRESHOLD) if cfg.DECODE_PRESSURE_THRESHOLD > 0 else 0.0
    pressure_multiplier = 1.0 + cfg.PRESSURE_AMPLIFIER * decode_pressure
    work_penalty = q_len * effective_remaining * pressure_multiplier

    cache_bonus = cfg.CACHE_WEIGHT * cached_tokens

    # Adaptive continuity bonus (simplified - no session tracking in basic mode)
    continuity = 0.0

    # Short prefill boost
    short_boost = 0.0
    if cfg.SHORT_PREFILL_BOOST > 0 and effective_remaining <= cfg.SHORT_PREFILL_THRESHOLD:
        short_boost = cfg.SHORT_PREFILL_BOOST

    return aging_score - work_penalty + cache_bonus + continuity + short_boost
```

**Step 2**: Patch `ross/vllm_sim/scheduler/scheduler.py`. Add these imports at top and a new method + modify `schedule()`:

Add after the existing imports:
```python
from scheduler.larry_config import LarryConfig, compute_larry_score
```

Add this method to the `Scheduler` class (before `schedule()`):
```python
    def _reorder_waiting_queue(self):
        """Sort waiting queue by LARRY v6 priority score (highest first)."""
        if not hasattr(self, '_larry_config'):
            self._larry_config = LarryConfig.from_env()
            self._larry_enabled = os.environ.get("LARRY_ENABLED", "0") == "1"

        if not self._larry_enabled:
            return  # FIFO mode

        cfg = self._larry_config
        if len(self.waiting) <= cfg.MIN_QUEUE:
            return  # Shallow queue, keep FIFO

        import time as _time
        now = max((r.arrive_time for r in self.waiting), default=0)
        # Use wall_time from the simulation context if available
        if self.running:
            now = max(now, max((r.arrive_time for r in self.running), default=0))

        num_running = len(self.running)
        q_len = len(self.waiting)

        for req in self.waiting:
            wait_time = max(0, now - req.arrive_time)
            cached_tokens = 0  # ROSS doesn't simulate prefix cache in basic mode
            req._larry_score = compute_larry_score(
                cfg,
                wait_time=wait_time,
                prompt_tokens=req.prompt_tokens,
                num_computed_tokens=req.num_computed_tokens,
                cached_tokens=cached_tokens,
                queue_length=q_len,
                num_running=num_running,
            )

        # Sort: highest score first
        self.waiting.sort(key=lambda r: getattr(r, '_larry_score', 0), reverse=True)
```

In the `schedule()` method, add the reorder call right after cleaning up completed requests:
```python
    def schedule(self) -> SchedulerOutput:
        """Execute one scheduling step"""
        self.running = [req for req in self.running if req.status != RequestStatus.FINISHED]
        # >>> LARRY v6: reorder waiting queue by priority <<<
        self._reorder_waiting_queue()
        if self.should_terminate():
            return None
        # ... rest unchanged ...
```

Also add `import os` to the imports at the top of scheduler.py.

---

## 3. Execution Plan

Follow these steps IN ORDER. After each experiment, record results in `result.md`.

### Step 1: Setup

1. Apply the LARRY v6 patch to ROSS (create `larry_config.py`, modify `scheduler.py`)
2. Create the results directory: `mkdir -p /workspace/lqs3/ross/larry_results`
3. Verify ROSS still works with LARRY disabled (FCFS baseline)

### Step 2: Run FCFS Baseline

```bash
cd /workspace/lqs3/ross/ross
conda run -n simulator --no-banner python ross_predict.py \
    --config config/test_vllm.json \
    --record-path ../larry_results/baseline_fcfs.csv
```

Record all metrics: duration, throughput, mean_ttft_ms, p99_ttft_ms, mean_tpot_ms, p99_tpot_ms.

### Step 3: Run LARRY v6 with Default Hyperparams

Write default config to `/workspace/lqs3/ross/larry_results/config_default.json`:
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
cd /workspace/lqs3/ross/ross
LARRY_ENABLED=1 LARRY_CONFIG_PATH=/workspace/lqs3/ross/larry_results/config_default.json \
conda run -n simulator --no-banner python ross_predict.py \
    --config config/test_vllm.json \
    --record-path ../larry_results/result_default.csv
```

Compare with baseline. If LARRY defaults are worse than FCFS, that's expected for some workloads — the optimizer will find better configs.

### Step 4: Iterative Search Loop

This is the core optimization loop. You will run **10-15 rounds**, each with **5-8 candidate configurations**.

#### For each round:

1. **Analyze** previous results: which configs performed best? Which parameters seem most influential?

2. **Propose** 5-8 new candidate configs based on your analysis:
   - 3-4 "exploitation" configs: small perturbations from the best known config
   - 1-2 "exploration" configs: dramatically different parameter combinations
   - 1-2 "interpolation" configs: blend two good configs

3. **Run** each candidate:
   ```bash
   # Write config JSON
   cat > /workspace/lqs3/ross/larry_results/config_roundN_cM.json << 'EOF'
   { "ALPHA_BASE": ..., ... }
   EOF

   # Run simulation
   cd /workspace/lqs3/ross/ross
   LARRY_ENABLED=1 LARRY_CONFIG_PATH=/workspace/lqs3/ross/larry_results/config_roundN_cM.json \
   conda run -n simulator --no-banner python ross_predict.py \
       --config config/test_vllm.json \
       --record-path ../larry_results/result_roundN_cM.csv
   ```

4. **Record** results in `result.md` with a table showing all metrics for each candidate.

5. **Analyze** and decide the search direction for the next round. Write your reasoning.

#### Search Strategy Guidelines

**Round 1-3 (Exploration)**:
- Try extreme values for key parameters: very high/low ALPHA_BASE, PRESSURE_AMPLIFIER, CACHE_WEIGHT
- Test whether SHORT_PREFILL_BOOST > 0 helps
- Try different DECODE_PRESSURE_THRESHOLD values
- Goal: understand which parameters matter most

**Round 4-7 (Narrowing)**:
- Focus on the 3-5 most influential parameters identified in exploration
- Use smaller perturbations around the best config
- Start fixing less-important parameters at their best values

**Round 8-10+ (Fine-tuning)**:
- Very small adjustments to the top 2-3 parameters
- Verify stability: run the best config 2-3 times to check for variance
- Compare final best against FCFS baseline

#### Scoring Priorities (for deciding which config is "best"):

1. **duration** — lower is better (PRIMARY metric, weight 0.35)
2. **throughput (tokens/s)** — higher is better (weight 0.20)
3. **mean_ttft_ms** — lower is better (weight 0.15)
4. **p99_ttft_ms** — lower is better (weight 0.10)
5. **mean_tpot_ms** — lower is better (weight 0.10)
6. **p99_tpot_ms** — lower is better (weight 0.10)

Calculate composite improvement vs FCFS baseline:
```
score = 0.35 * (base_duration - cur_duration) / base_duration
      + 0.20 * (cur_throughput - base_throughput) / base_throughput
      + 0.15 * (base_mean_ttft - cur_mean_ttft) / base_mean_ttft
      + 0.10 * (base_p99_ttft - cur_p99_ttft) / base_p99_ttft
      + 0.10 * (base_mean_tpot - cur_mean_tpot) / base_mean_tpot
      + 0.10 * (base_p99_tpot - cur_p99_tpot) / base_p99_tpot
```

### Step 5: Final Output

When you've converged on the best configuration:

1. Save the best config as `/workspace/lqs3/ross/larry_results/best_config.json`
2. Run it one final time to confirm results
3. Write the complete `result.md` to `/workspace/lqs3/ross/larry_results/result.md`

---

## 4. result.md Template

The `result.md` file should contain:

```markdown
# LARRYSmith Optimization Results

## Environment
- Dataset: {dataset path}
- ROSS config: {config file used}
- Date: {date}

## FCFS Baseline
| Metric | Value |
|--------|-------|
| duration | ... |
| throughput | ... |
| mean_ttft_ms | ... |
| ... | ... |

## Iteration Log

### Round 1: Exploration
**Strategy**: ...

| Config ID | ALPHA_BASE | PRESSURE_AMPLIFIER | ... | duration | throughput | mean_ttft | score |
|-----------|------------|-------------------|-----|----------|------------|-----------|-------|
| r1_c1 | ... | ... | ... | ... | ... | ... | ... |

**Analysis**: ...
**Best this round**: ...

### Round 2: ...
...

## Convergence Summary
- Total experiments: N
- Best config found in round: M
- Improvement over FCFS: X%

## Best Configuration
{JSON of best config}

## Key Findings
1. ...
2. ...
3. ...
```

---

## 5. Important Notes

- **Always check if the simulation actually ran successfully** by examining the output CSV. If it's empty or has errors, debug before continuing.
- **If a config causes OOM or crashes**, record it as "FAILED" and move on. Very aggressive parameters (e.g., extremely low ALPHA_BASE) may cause starvation.
- **The ROSS simulator is deterministic** — same config + same dataset = same results. No need to run the same config twice unless you changed something.
- **Be patient with analysis**. After each round, take time to reason about WHY certain configs performed better. The reasoning is as important as the numbers.
- **If you modify this document** during the process (e.g., to fix a bug in the patch, adjust search ranges), note the change in result.md.
- **Record EVERYTHING** in result.md — including failed experiments, unexpected observations, and bugs you found and fixed.
