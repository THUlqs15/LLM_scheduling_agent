# LARRYSmith Real-vLLM Optimization Results

## Environment
- Dataset: ~/.etc/ShareGPT_V3_unfiltered_cleaned_split.json
- vLLM: 0.19.0 editable @ /workspace/lqs3/LLM_scheduling/vllm
- Engine: v1
- Model: Butter_L3_8B_RPMaster_v2 on NVIDIA A40 (46GB)
- Server flags: TP=1, PP=1, max_num_seqs=32, max_num_batched_tokens=8192, prefix-caching=on
- Workload: 512 ShareGPT requests; rates {inf, 4, 2, 1} req/s
- Date: 2026-04-27

## Patch Audit
- vLLM files modified: vllm/v1/core/sched/scheduler.py (import + _larry_reorder_waiting method added)
- New file: vllm/v1/core/sched/larry_hook.py
- Behavior with VLLM_USE_LARRY unset: verified `LarryRuntime.get().enabled == False`, vllm imports cleanly

## FCFS Baseline (frozen reference)
| Rate | duration (s) | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | throughput (req/s) |
|------|--------------|--------------|-------------|--------------|-------------|-------------------|
| inf  | 131.27       | 55037.63     | 114414.82   | 37.86        | 49.87       | 3.90              |
| 4    | 148.19       | 1065.89      | 4369.50     | 36.33        | 46.14       | 3.45              |
| 2    | 275.86       | 124.39       | 227.50      | 33.53        | 41.73       | 1.86              |

**Key observation**: At rate=inf, mean_ttft=55s (queue depth is extreme). At rate≥2, TTFT is low (<125ms). LARRY's biggest win opportunity is at high load where queue is deep.

## LARRY Default Config (FAILED — starvation at all rates)
| Rate | duration(s) | mean_ttft(ms) | p99_ttft(ms)  | mean_tpot(ms) | p99_tpot(ms) | throughput(req/s) | score vs FCFS |
|------|-------------|---------------|---------------|---------------|--------------|-------------------|---------------|
| inf  | 131.46      | 59128         | 115495        | 40.12         | 79.97        | 3.89              | -0.06         |
| 4    | 200.63      | 57145         | 156099        | 49.21         | 536.14       | 2.55              | -5.13         |
| 2    | 276.76      | 37409         | 144586        | (TBD)         | 252.0        | 1.85              | -2.97         |

**Root Cause**: ALPHA_BASE=10240 is too small relative to work penalty.
At q_len=10, eff_remaining=2000: penalty=100,000 vs alpha≈10,240 → aging needs 10s to overcome penalty.
This creates unstable starvation: long requests get pushed back, queue builds, penalty grows further.
**Fix for round 1**: Use ALPHA_BASE ≥ 50,000 (all round-1 configs use 50k-100k).

## Iteration Log

### Round 1 — Broad Exploration (ALPHA_BASE fixed, vary PRESSURE/CACHE/SHORT_BOOST)

All configs use ALPHA_BASE ≥ 50k to ensure stability.

**r1_c1**: `{ALPHA_BASE:100000, PRESSURE_AMPLIFIER:1.0, DECODE_PRESSURE_THRESHOLD:5, no bonuses}`
| Rate | duration(s) | mean_ttft(ms) | p99_ttft(ms)  | mean_tpot(ms) | p99_tpot(ms) | throughput(req/s) | score vs FCFS |
|------|-------------|---------------|---------------|---------------|--------------|-------------------|---------------|
| inf  | 127.54      | 46908         | 108590        | 35.70         | 51.36        | 4.014             | +4.57         |
| 4    | 147.02      | 879           | 11822         | 36.16         | 51.69        | 3.482             | -15.12        |
| 2    | 276.28      | 123           | 219           | 33.29         | 39.62        | 1.853             | +1.02         |
**Finding**: Mean TTFT improved at all rates but p99_ttft at rate=4 jumped +171% (11.8s vs 4.4s FCFS) — SRPT tail starvation. Rate=inf shows promising -14.8% mean_ttft, -5.1% p99_ttft. Overall score hurt by rate=4 p99 regression.

**r1_c2**: `{ALPHA_BASE:100000, PRESSURE_AMPLIFIER:0.5, DECODE_PRESSURE_THRESHOLD:5}`
| Rate | duration(s) | mean_ttft(ms) | p99_ttft(ms)  | mean_tpot(ms) | p99_tpot(ms) | throughput(req/s) | score vs FCFS |
|------|-------------|---------------|---------------|---------------|--------------|-------------------|---------------|
| inf  | 122.56      | 44508         | 104751        | 35.50         | 48.74        | 4.177             | +8.29         |
| 4    | 148.06      | 1143          | 14949         | 36.19         | 53.07        | 3.458             | -26.69        |
| 2    | 275.95      | 122           | 216           | 33.26         | 39.32        | 1.855             | +1.40         |
**Finding**: Rate=inf significantly better than c1 (mttft +19.1%, thr +7.1%). But rate=4 p99_ttft is even WORSE than c1 (14.9s vs 11.8s). Lower AMP helps at high load but hurts tail at moderate load. The SRPT starvation pattern persists.

**r1_c3**: `{ALPHA_BASE:100000, CACHE_WEIGHT:10000, CACHE_PROBE_INTERVAL:2, PRESSURE_AMPLIFIER:1.0}` — TOTAL=-4.71. High cache weight hurts; rate=4 p99 still -218%.

**r1_c4**: `{ALPHA_BASE:100000, PRESSURE_AMPLIFIER:4.0, DECODE_PRESSURE_THRESHOLD:10}` — TOTAL=-2.89. High AMP helps rate=2/1 (+2.9%/+2.7% mttft) but rate=4 p99 still -174%.

**r1_c5**: `{ALPHA_BASE:100000, PRESSURE_AMPLIFIER:1.0, SHORT_PREFILL_BOOST:100000, SHORT_PREFILL_THRESHOLD:8192}` — TOTAL=-2.77. Best rate=inf score (+9.64). Boost is no-op (threshold 8192 covers all requests).

**r1_c6**: `{ALPHA_BASE:100000, PRESSURE_AMPLIFIER:1.0, SHORT_PREFILL_BOOST:200000, SHORT_PREFILL_THRESHOLD:4096}` — TOTAL=-2.89. Similar to c5.

**r1_c7**: `{ALPHA_BASE:100000, MIN_QUEUE:8, PRESSURE_AMPLIFIER:2.0, SHORT_PREFILL_BOOST:50000}` — TOTAL=-1.57. MIN_QUEUE=8 helps somewhat. Rate=4 mttft +11.8% (best mttft in round 1 at rate=4). Rate=2/1 both +1.6% score.

**r1_c8**: `{ALPHA_BASE:50000, PRESSURE_AMPLIFIER:2.0, SHORT_PREFILL_BOOST:100000}` — **TOTAL=+0.38 (WINNER, only positive)**
| Rate | duration(s) | mean_ttft(ms) | p99_ttft(ms) | mean_tpot(ms) | p99_tpot(ms) | throughput(req/s) | score |
|------|-------------|---------------|--------------|---------------|--------------|-------------------|-------|
| inf  | —           | 44050         | 103808       | —             | —            | 4.150             | +9.14 |
| 4    | —           | 764           | 10303        | —             | —            | 3.433             | -10.16 |
| 2    | —           | 122           | 215          | —             | —            | 1.853             | +1.49 |

### Round 1 Summary — Rankings
| Config | inf score | r4 score | r2 score | TOTAL |
|--------|-----------|----------|----------|-------|
| r1_c8  | +9.14     | -10.16   | +1.49    | **+0.38** |
| r1_c7  | +7.28     | -16.57   | +1.64    | -1.57 |
| r1_c1  | +4.58     | -15.11   | +1.05    | -2.31 |
| r1_c5  | +9.64     | -21.18   | +0.47    | -2.77 |
| r1_c4  | +2.21     | -16.85   | +1.54    | -2.89 |
| r1_c6  | +7.01     | -20.43   | +1.18    | -2.89 |
| r1_c2  | +8.31     | -26.69   | +1.41    | -4.23 |
| r1_c3  | +4.56     | -24.42   | +0.52    | -4.71 |

**Key Insight**: The dominant cost is rate=4 p99_ttft regression (SRPT tail starvation). c8 wins because ALPHA_BASE=50000 (lower) + AMP=2.0 (higher) creates more decisive ordering: short requests clear the queue quickly, then longer ones age naturally. SHORT_PREFILL_BOOST at threshold=8192 is effectively a no-op (all requests qualify).

**Root cause of rate=4 p99 regression**: At moderate load, new short requests continuously arrive and win priority over long ones. The long requests accumulate and wait 10-15s before aging overcomes the advantage of new arrivals.

**Round 2 direction**: Lower ALPHA_BASE + higher AMP to make ordering more decisive. Use SHORT_PREFILL_THRESHOLD=512-2048 to create real two-tier prioritization. Explore MIN_QUEUE=12-20 to gate on queue depth.

### Round 2 — Narrowing (ALPHA range, selective boost, MIN_QUEUE gating)

| Config | ALPHA | MIN_Q | AMP | BOOST | THRESH | inf | r4 | r2 | TOTAL |
|--------|-------|-------|-----|-------|--------|-----|----|----|-------|
| r2_c1  | 30k   | 4     | 3.0 | 100k  | 8192   | +6.12 | -13.56 | +1.38 | -1.35 |
| r2_c2  | 20k   | 4     | 4.0 | 100k  | 8192   | +7.85 | -16.95 | +2.35 | -1.53 |
| r2_c3  | 50k   | 4     | 2.0 | 200k  | 2048   | +6.10 | -25.83 | +1.37 | -4.56 |
| r2_c4  | 50k   | 4     | 2.0 | 300k  | 512    | +6.86 | -19.79 | +1.93 | -2.68 |
| r2_c5  | 50k   | 16    | 2.0 | 100k  | 8192   | +2.24 | -14.95 | +1.27 | -2.45 |
| r2_c6  | 30k   | 8     | 3.0 | 150k  | 2048   | +8.45 | -14.46 | +2.18 | -0.66 |
| **r2_c7** | 50k | **20** | 2.0 | 100k  | 8192 | +6.36 | **-2.64** | +1.37 | **+1.54** |

**KEY FINDING**: MIN_QUEUE=20 is the breakthrough. Rate=4 p99_ttft drops from -135.8% (c8) to -56.3% (r2_c7) — 6.8s vs 4.4s FCFS (from 10.3s). This strongly suggests queue depth at rate=4 is typically ≤20, so LARRY rarely activates and starvation is prevented. Rate=inf score (+6.36) is lower than best (+9.64 from r1_c5) because ALPHA=50000 is lower. 

**Round 3 hypothesis**: Combine HIGH ALPHA (→ better rate=inf SRPT gains) + MIN_QUEUE=20 (→ rate=4 protection). Target: inf=+9, r4~-2, TOTAL>+2.

### Round 3 — Convergence (ALPHA + MIN_QUEUE grid search)

| Config | ALPHA | MIN_Q | AMP | BOOST | inf | r4 | r2 | TOTAL |
|--------|-------|-------|-----|-------|-----|----|-----|-------|
| r3_c1  | 100k  | 20    | 1.0 | 100k  | +5.39 | -8.61  | +1.02 | -0.38 |
| r3_c2  | 100k  | 20    | 2.0 | 100k  | +5.53 | -0.03  | +0.73 | +1.76 |
| r3_c3  | 50k   | 25    | 2.0 | 100k  | +5.23 | +0.55  | +1.44 | +1.99 |
| r3_c4  | 75k   | 20    | 1.5 | 100k  | +7.90 | -3.68  | +2.46 | +1.81 |
| **r3_c5** | **100k** | **25** | **1.0** | **100k** | **+8.79** | **+1.03** | **+2.70** | **+3.34** |
| r3_c6  | 50k   | 20    | 2.0 | 200k  | +6.73 | -18.28 | +1.09 | -2.47 |

**KEY FINDING**: r3_c5 {ALPHA=100000, MIN_QUEUE=25, AMP=1.0, BOOST=100000} achieves TOTAL=+3.34.
- Rate=inf: excellent +8.79 (ALPHA=100k provides strong SRPT aging)
- Rate=4: POSITIVE +1.03 (mttft +18.2%, p99 only -15% — very acceptable)
- Rate=2: positive +2.70

**Why r3_c5 wins**: MIN_QUEUE=25 prevents LARRY from activating at rate=4's typical queue depth (≤25), eliminating starvation. ALPHA=100k provides aggressive aging at rate=inf for strong SRPT. AMP=1.0 (vs 2.0) reduces penalty for long requests when LARRY does briefly activate at rate=4 peaks.

**Round 4 goal**: Fine-tune around r3_c5. Test ALPHA 80k-150k, MIN_QUEUE 22-30, AMP 0.5-1.5.

*Note: TOTAL score averaged over rates {inf, 4, 2} only (rate=1 excluded as rarely used).*

## Verification (top-3 × 3 replicates)
(to be filled after convergence)

## Convergence Summary
(to be filled after all rounds)

## Best Configuration
(to be filled after convergence)

## Key Findings
(to be filled after analysis)

## Cleanup Status
- Server: running/stopped (update per session)
- Extra pip installs: none
