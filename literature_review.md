# SLO-aware LLM Scheduling Literature Review

本文总结工作目录下 7 篇与 LLM serving、SLO/SLA-aware scheduling、multi-SLO serving 相关的论文。为了避免术语混淆，先约定两个分类维度：

- **PD 分离**：Prefill 和 Decode 被放到不同的 serving instances / GPU 组上执行，并伴随 KV cache 迁移或跨实例交接。仅仅在算法里分别建模 prefill/decode，不算 PD 分离。
- **Instance-scheduling**：在一个 serving instance / 一个逻辑推理引擎内部决定请求顺序、batch、chunk、token/layer 执行计划、admission 等。
- **Cluster-scheduling**：跨 replica / instance 做路由、负载均衡、实例组分配、全局 admission 或资源池化决策。

## 0. 总览表

| 论文 | 主要研究问题 | SLO/SLA 指标 | PD 分离 | Scheduling 层次 | 核心机制 |
|---|---|---|---|---|---|
| FastServe, *Fast Distributed Inference Serving for Large Language Models* | 降低 LLM 推理 head-of-line blocking 和平均/尾延迟 | 平均延迟、tail latency，不是显式 heterogeneous SLO | **否** | 主要是 **instance-scheduling**；支持 tensor/pipeline parallel 的分布式执行，但不是跨 replica 路由 | token 粒度 preemption；Skip-Join MLFQ；proactive KV cache swapping |
| SOLA, *Optimizing SLO Attainment for LLM Serving with State-Aware Scheduling* | 在 TTFT/TPOT 双 SLO 下动态平衡 prefill 与 decode | TTFT、TPOT | **否**，论文明确主要关注 prefill/decode 同 GPU | **Instance-scheduling** | 每 iteration 选择排序函数 $F_i$、请求数 $n_i$、token 数 $k_i$；状态感知约束优化 |
| SLOs-Serve, *Optimized Serving of Multi-SLO LLMs* | 多阶段、多应用、多 SLO 请求的 token allocation 与 admission | per-stage TTFT/TPOT，多 decode tier，可含 reasoning stage | **否**，每个 replica 内 co-located prefill/decode；论文批评静态 PD ratio | **Instance + cluster**：单 replica 内 DP 规划；多 replica 间 SLO-driven routing | multi-SLO DP；soft admission control；dynamic batch-size tuning；burst-resilient tier；multi-replica routing |
| SCORPIO, *Serving the Right Requests at the Right Time for Heterogeneous SLOs* | 异构 TTFT/TPOT SLO 下最大化 goodput 和 adherence | heterogeneous TTFT、TPOT | **否** | **Instance-scheduling** | TTFT Guard: LDF + rejection；TPOT Guard: VBS admission + credit-based batching |
| Niyama, *Breaking the Silos of LLM Inference Serving* | 打破 interactive/batch silo，在共享基础设施上 co-schedule 多 QoS 类 | interactive: TTFT/TBT；non-interactive: TTLT | **否**，co-located scheduling | 主要是 **instance-scheduling**，但目标是 cluster 级资源池化、去 silo；没有重点做跨 replica 路由算法 | dynamic chunking；EDF/SRPF hybrid priority；eager relegation；selective preemption |
| SABER, *Adaptive Request Scheduling for CodeLLM Serving with SLA Guarantees* | 自托管 CodeLLM 中静态 batch size 难以适应负载和任务混合 | request completion deadline / tail-latency SLA | **否** | **Instance-scheduling**，偏单 GPU / 资源受限 | offline speed profiling；USL speed predictor；two-tier queue；admission control |
| Laser, *Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving* | iteration-level scheduling 太粗，无法服务 multi-SLO workload | TTFT、TBT/TPOT，多 SLO | **是**，基于 prefill-decode disaggregation | **Instance + cluster**：instance 内 layer-level scheduling；cluster/controller 做 inter-instance dispatch | layer-level chunked prefill；layer-level decode batching；modular latency model；group-based decode assignment |

## 1. FastServe: Fast Distributed Inference Serving for Large Language Models

### 研究问题

FastServe 关注的不是“heterogeneous SLO”本身，而是 LLM online inference 中一个基础调度问题：已有系统常用 FCFS / run-to-completion 或较粗粒度 iteration-level scheduling，长请求会阻塞短请求，导致 head-of-line blocking。LLM decode 是自回归的，每生成一个 token 后都天然出现一个调度点，因此可以做 token 粒度抢占。

核心问题可以写成：在不知道最终 output length 的情况下，如何通过抢占式调度降低 job completion time，并在 GPU KV cache 内存有限时避免抢占带来的缓存爆炸。

### 核心算法

**1. Skip-Join MLFQ**

FastServe 使用多级反馈队列 $Q_1,\dots,Q_n$，优先级从高到低，每个队列有 quantum：

$$
q_1 < q_2 < \cdots < q_n.
$$

传统 MLFQ 会让新 job 从最高优先级 $Q_1$ 开始，如果超过 quantum 就逐级降级。但 LLM 有一个特殊问题：第一轮 prefill / initialization 时间和 input length 强相关，如果长 prompt 也从 $Q_1$ 开始，很可能第一轮就耗尽 quantum，造成无意义 demotion 和重复调度开销。

FastServe 的做法是：

- 对新请求 $j$，用 profiling 模型估计第一轮执行时间 $t_{\text{init}}(j)$。
- 直接把请求放入满足 $q_p \ge t_{\text{init}}(j)$ 的最高优先级队列 $Q_p$，跳过更高优先级队列，所以叫 skip-join。
- 每生成一个 token 后，scheduler 可以决定继续执行、降级、或被更高优先级请求抢占。
- 若请求耗尽当前 quantum，则降级到低优先级队列。
- 若请求饥饿时间超过阈值 $\alpha$，则提升到 $Q_1$，避免长请求永久饿死。
- 每个 iteration 从高优先级到低优先级选择 ready jobs，直到达到 MaxBatchSize。

伪代码逻辑可以概括为：

$$
p(j)=\min\{i\mid q_i\ge t_{\text{init}}(j)\}
$$

新请求进入 $Q_{p(j)}$，运行中请求按 quantum 消耗降级，按 starvation threshold 提升。

**2. Proactive KV Cache Swapping**

抢占式调度会让大量“已开始但暂不执行”的请求保留 KV cache，GPU memory 可能不够。FastServe 把 KV cache 空间扩展到 CPU host memory，并提前 swap：

- inactive job 的 KV cache 可 swap out 到 host。
- 即将被调度的 job 提前 swap in。
- swap 与 GPU kernel execution 重叠，避免 reactive swap 阻塞 critical path。

swap 顺序用 Estimated Next Scheduled Time:

$$
\text{ENST}(i)=\min(T_{\text{promote}}(i), T_{\text{execute}}(i)).
$$

其中 $T_{\text{promote}}(i)$ 表示因 starvation prevention 被提升所需时间，$T_{\text{execute}}(i)$ 表示高优先级 job 运行到 job $i$ 可执行之前的估计时间。swap out 时优先换出 ENST 大的请求，swap in 时优先换入 ENST 小的请求。

**3. Distributed execution support**

FastServe 的 “distributed” 主要是 model parallel execution：支持 tensor parallelism 和 pipeline parallelism。在 pipeline 下，不同 stage 同时处理不同 batch，scheduler 仍维护 MLFQ 语义，但选择 pending state 中最高优先级的 job 来启动下一 stage。KV cache 也按模型并行分区，在对应 GPU 上本地管理。

### Key findings 与定位

- 相比 vLLM，FastServe 在相同 average latency / tail latency 要求下分别最高提升 throughput **31.4x / 17.9x**。
- Skip-Join MLFQ 相比 FCFS、naive MLFQ、fixed priority，在不同初始化/解码比例下更稳健。
- Proactive swapping 能把 KV cache 迁移成本移出关键路径，解决 preemption 与 GPU memory 的冲突。
- **PD 分离：否。** FastServe 区分 initialization/prefill 和 decode，但没有把 prefill/decode 放到不同实例上。
- **Scheduling 类型：instance-scheduling。** 它是一个逻辑 inference engine 内的 job scheduler；虽然支持多 GPU model parallel，但不是 multi-replica cluster routing。
- 对 SLO-aware serving 的启发：token 粒度 preemption 和 starvation prevention 是后续 SLO scheduler 可以借用的基础机制，但 FastServe 本身不处理 heterogeneous TTFT/TPOT SLO。

## 2. SOLA: Optimizing SLO Attainment for LLM Serving with State-Aware Scheduling

### 研究问题

SOLA 直接面向 LLM serving 的 TTFT 和 TPOT SLO。作者观察到固定策略会造成两类问题：

- **TTFT/TPOT bias**：prefill-prioritized 会改善 TTFT 但伤害 TPOT；decode-prioritized 反过来。
- **request variance**：不同请求之间 SLO attainment 差异大，有的请求远低于 SLO，有的请求严重超标。

研究问题是：如何在每个 iteration 动态改变调度策略，使系统在 TTFT 与 TPOT 之间、请求与请求之间做 trade-off，从而最大化整体 SLO attainment。

### 核心算法

**1. 调度设计空间**

SOLA 把第 $i$ 个 iteration 的调度策略抽象成三个变量：

- $F_i$：等待队列 $Q_i^{wait}$ 的排序函数，即先执行谁。
- $n_i$：本 iteration 最多执行多少个请求。
- $k_i$：本 iteration 最多执行多少个 token。

调度过程是：

1. 用 $F_i$ 排序 $Q_i^{wait}$。
2. 逐个尝试把请求放入 $Q_i^{run}$。
3. 用 peak memory predictor 判断加入后是否可能超过显存上限。
4. 直到请求数达到 $n_i$ 或 token 数达到 $k_i$。

对 decode 请求，新增 token 数 $k_{i,r}^{new}=1$；对 chunked prefill，$k_{i,r}^{new}\le l_r^{in}$。

**2. State Monitor**

SOLA 每个 iteration 后更新 request-level 和 system-level 状态：

$$
p_i^{TTFT}=\frac{\max_r t_{i,r}^{TTFT}}{T^{TTFT}},
\quad
p_i^{TPOT}=\frac{\max_r t_{i,r}^{TPOT}}{T^{TPOT}}.
$$

若 $p_i^{TTFT}\le 1$，说明当前 TTFT 满足；若 $p_i^{TPOT}\le 1$，说明 TPOT 满足。还会维护：

- 每个请求当前 TTFT/TPOT。
- 已生成长度 $l_{i,r}^{out}$。
- 预测剩余输出长度 $l_{i,r}^{left}$。
- 系统输出长度分布 $D_i$。
- 当前 KV cache memory ratio $m_i^{ratio}$。

**3. Strategy Generator: 动态约束优化**

SOLA 在两个问题之间切换：

- 优化 TTFT，约束 TPOT。
- 优化 TPOT，约束 TTFT。

如果 TPOT 更危险，就优先 decode；如果 TTFT 更危险，就优先 prefill。若 TTFT 和 TPOT 都不满足，SOLA 会把 max constraint 放松成 percentile-level max，让问题仍可求解，避免系统在不可行约束下振荡。

**4. 分层优先级 $F_i$**

第一层决定 phase priority：

- 优化 TTFT subject to TPOT：prefill 优先。
- 优化 TPOT subject to TTFT：decode 优先。

第二层决定同 phase 内请求顺序：

- prefill 请求按预测 TTFT 排序：

$$
t_{i,r}^{TTFT}+C_i^p(r)
$$

- decode 请求按预测 TPOT 排序：

$$
\frac{t_{i,r}^{TPOT}\cdot l_{i,r}^{out}+C_i^d(Q_{i-1}^{run})\cdot l_{i,r}^{left}}
l_{i,r}^{out}+l_{i,r}^{left}}.
$$

这里 $C_i^p$ 和 $C_i^d$ 是 prefill/decode cost model。

**5. 约束 workload size**

优化 TTFT 时，prefill 优先，但 prefill token 数 $k_i$ 不能把 TPOT 推爆。SOLA 选择满足 TPOT 约束的最大 tile-size multiple：

$$
\max_{r:t_{i,r}^{TPOT}>0}
\left(
t_{i,r}^{TPOT}+\frac{C_i^p(Q_i^{run})}{l_{i,r}^{out}}
\right)
\le T^{TPOT}.
$$

优化 TPOT 时，decode 优先，但加入 prefill 的数量 $n_i$ 不能让等待请求 TTFT 超标：

$$
\max_{r\in Q_i^{wait}}
\left(
t_{i,r}^{TTFT}+C_i^p(Q_i^{run})+C_i^d(Q_i^{run})+C_i^p(r)
\right)
\le T^{TTFT}.
$$

**6. Cost model 与 memory prediction**

SOLA 采用峰值显存预测避免 preemption/swap 的额外开销。延迟模型分别建模 prefill 和 decode：

$$
C_p=a_0\sum_r l_r^{has}l_r^{in}
b_0\sum_r(l_r^{in})^2
c_0\sum_r l_r^{in}+d_0,
$$

$$
C_d=a_1\sum_r 1+b_1\sum_r l_r^{has}+c_1.
$$

$C_p$ 主要刻画 attention/linear FLOPs，$C_d$ 主要刻画 KV cache memory access。

### Key findings 与定位

- SOLA 将 SLO attainment 从 **45.5% 提升到 99.4%**。
- 在满足 90% 和 99% 请求 SLO 的前提下，平均可多服务 **1.04-1.27x** 请求。
- 调度开销很低，大约 **0.40%-0.45%**。
- **PD 分离：否。** 论文明确说主要关注 prefill 和 decode 在同一 GPU / instance 上的系统。
- **Scheduling 类型：instance-scheduling。** SOLA 替换 vLLM 内部 scheduler，核心是每个 iteration 的 batch/order/token-size 决策，不做 cluster routing。
- 贡献重点：把“prefill vs decode”和“请求之间谁更危险”统一成 state-aware constrained optimization，比固定 prefill-first / decode-first 更细。


**4. Optional SLO-adaptive speculative decoding**

若使用 speculative decoding，SLOs-Serve 为不同 decode SLO tier 选择 speculation length $sl_l$。目标是最大化 prefill throughput：

$$
\max_{sl_{1:L}}
\text{prefillTpt}
=
\frac{\text{PrefillBgtPerBatch}}{\text{BatchTime}},
$$

$$
\text{PrefillBgtPerBatch}
=
\text{Time2BS}(T(sl_{1:L}),sl_{1:L})-\sum_i n_i sl_i,
$$

$$
T(sl_{1:L})
=
\min_l \left(TPOT_l\cdot Acc(sl_l)\right).
$$

也就是说，decode 越宽松的请求可以一次 speculative 更多 token，把 batch time 放大，从而给 prefill 留出更多吞吐空间。

**5. Burst-resilient scheduling 与 multi-replica routing**

在 burst 中，SLOs-Serve 会把当前 replica 上不可满足 SLO 的请求 offload 到 best-effort tier，等低负载时处理。

在多副本部署中，SLOs-Serve 用 centralized scheduler / routing 逻辑“虚拟化”每个 replica：

1. 请求先被 one-shot load balancer 送到某 replica。
2. 该 replica 的 scheduler 判断是否可满足 SLO。
3. 若不可满足，则顺序路由到下一个 replica。
4. 超过 route limit 后，进入 backup policy：decline 或 lower-tier resources。

这里的路由依据不是传统的队列长度或 memory usage，而是 **SLO attainability**。

### Key findings 与定位

- 6 个场景中，SLOs-Serve 平均提升 per-GPU serving capacity **2.2x**。
- 在 4 replica serving 下，bursty arrivals 场景中相对 1 replica 可达到最高 **6.2x** capacity。
- 单项消融中，request routing、SLO-adaptive speculative decoding、burst-resilient scheduling 分别带来约 **1.19x / 1.66x / 1.34x** capacity gain。
- 调度开销多数低于 **2 ms**，每次 resource planning 低于 **10 ms**。
- **PD 分离：否。** SLOs-Serve 的核心是 co-located mixed prefill/decode token planning；它讨论并批评了静态 PD disaggregation 在多应用动态负载下设备比例难以适配。
- **Scheduling 类型：instance + cluster。** 单个 replica 内是 DP-based instance scheduler；多 replica 下有 SLO-driven request routing，因此也属于 cluster-scheduling。
- 贡献重点：把 SLO serving 从“调度下一个请求”推进到“规划未来 batch token budget + admission”，并用 soft admission 保证 admitted requests 的多阶段 SLO。

## 4. SCORPIO: Serving the Right Requests at the Right Time for Heterogeneous SLOs

### 研究问题

SCORPIO 关注异构 TTFT 和 TPOT SLO。已有 vLLM/SGLang 等吞吐优先系统通常无差别处理请求，在高负载时会出现：

- TTFT 紧的请求被排在松 SLO 请求后面，错过首 token deadline。
- TPOT 紧的请求和 TPOT 松的请求每轮都一起 decode，导致严格请求被拖慢。
- 系统过载时贪心 admission 造成 cascading violation。

研究问题是：如何利用 SLO heterogeneity，在 queue management、admission control 和 batch selection 三个阶段选择“right requests at the right time”。

### 核心算法

**1. 目标函数**

请求 $r_i$ 同时满足：

$$
TTFT(r_i)\le S_{TT}(r_i),\quad
TPOT(r_i)\le S_{TP}(r_i)
$$

才算 SLO-compliant。定义：

$$
Goodput(\pi)=\frac{|R_{good}(\pi)|}{T},
\quad
Adherence(\pi)=\frac{|R_{good}(\pi)|}{|R|}.
$$

目标是设计 online scheduling policy $\pi$ 最大化长期 goodput 和 adherence。

**2. Predictor**

SCORPIO 用轻量序列长度预测器预测 output length，并用分析模型估计 TTFT/TPOT。

TPOT estimator：

$$
ITL(|R|,L_{avg}(R))
=
\alpha |R|L_{avg}(R)+\beta |R|+\gamma L_{avg}(R)+\delta.
$$

若新请求 $r$ 被加入 running set $R'$，其预测输出长度为 $P(r)$，估计：

$$
EstimatedTPOT
=
\epsilon\left[
(\alpha |R'|+\gamma)\left(L_{avg}(R')+\frac{P(r)}{2}\right)
\beta |R'|+\delta
\right].
$$

TTFT estimator 用 prefill time 近似排队前缀和：

$$
EstimatedTTFT(w_i)\ge \sum_{k=1}^{i} PrefillTime(w_k).
$$

**3. TTFT Guard**

TTFT Guard 包含两步：

- **Least Deadline First reordering**：按离 TTFT deadline 最近的请求优先。
- **Unattainable rejection**：若估计 TTFT 已经不可能满足 $S_{TT}$，直接 reject，避免占用资源拖累其他请求。

**4. TPOT Guard: TRP + credit-based batching**

SCORPIO 定义 TPOT-relative Proportionality:

$$
TRP(r)=
\frac{\min_{r'\in R(t)} S_{TP}(r')}
{S_{TP}(r)}.
$$

若请求 $r$ 的 TPOT SLO 越松，则 $TRP(r)$ 越小，它每轮获得的 credit 越少。每个 iteration：

$$
C_r(t)\leftarrow C_r(t)+TRP(r).
$$

当 $C_r(t)\ge 1$，请求进入本轮 batch，并扣除：

$$
C_r(t)\leftarrow C_r(t)-1.
$$

长期来看，请求被 batch 的频率收敛到 $TRP(r)$。严格 SLO 请求几乎每轮都被执行，宽松 SLO 请求会跳过部分 iteration。

**5. VBS-based admission control**

因为宽松请求不是每轮都 decode，直接用 running request count 会高估负载。SCORPIO 定义 virtual batch size：

$$
VBS(R')=\sum_{r\in R'}TRP(r).
$$

新请求只有在下面条件满足时才被 admit：

$$
EstimatedTPOT(VBS(R'),L_{avg}(R'))
\le
\min_{r'\in R'}S_{TP}(r').
$$

这把 admission control 和 heterogeneous TPOT batching 统一起来。

### Key findings 与定位

- SCORPIO 在高 QPS 下最高提升 goodput **14.4x**，SLO adherence 最高提升 **46.5%**。
- TTFT Guard 和 TPOT Guard 单独使用都不够；消融显示二者互补，否则会把违约从 TTFT 转移到 TPOT 或反之。
- 调度开销小于 **1%**。
- 低负载时复杂控制和 predictor 可能带来额外开销，论文也提到可在低负载切回简单策略。
- **PD 分离：否。** SCORPIO 是基于 vLLM 式 co-located engine 的 SLO-oriented scheduler；prefill-decode disaggregation 被列为未来方向。
- **Scheduling 类型：instance-scheduling。** 它做 queue reorder、admission、batch selection，不做 cluster-level routing。
- 贡献重点：非常清晰地把 heterogeneous TPOT 变成“不同 decode 频率”的 credit 机制，比只按 deadline 或 length 排序更直接。

## 5. Niyama: Breaking the Silos of LLM Inference Serving

### 研究问题

Niyama 的出发点是生产部署形态：很多系统把 interactive 和 batch workload 放到不同 silo，interactive 用小 chunk 保证低延迟，batch 用大 chunk 保证吞吐。这会造成资源利用率低、过度 provisioning，并且无法支持更细粒度 QoS class。

研究问题是：如何在共享基础设施上 co-schedule 多 QoS class，同时在 overload 下优雅降级，而不是让所有请求一起违反 SLO。

### 核心算法

**1. QoS class 与 deadline**

Niyama 定义两类 QoS：

- interactive：关注 TTFT 和 TBT。
- non-interactive：关注 TTLT。

interactive 请求首 token deadline：

$$
D_{first}=t_{arrival}+SLO_{TTFT}.
$$

第 $n$ 个 token 的 deadline：

$$
D_n=t_{arrival}+SLO_{TTFT}+(n-1)\cdot SLO_{TBT}.
$$

non-interactive 请求整体完成 deadline：

$$
D_{total}=t_{arrival}+SLO_{TTLT}.
$$

**2. Dynamic chunking**

Niyama 每个 iteration 构造 batch：

- decode queue 中所有 decode requests。
- prefill queue 中某个请求的一个 prefill chunk。

chunk size 不是固定的。Niyama 根据当前 decode requests 的 slack 计算“本轮最大安全 prefill chunk size”：

$$
slack_r = D_{next}(r)-t_{now}.
$$

选择最大的 chunk size，使本轮 batch latency 不会让任何 decode request 的 next-token deadline 被违反。这样在宽松时用大 chunk 提吞吐，在紧张时用小 chunk 保延迟。

**3. Hybrid prioritization: EDF + SRPF**

Niyama 发现 EDF 低负载好，但高负载会 queue buildup；SRPF/SJF 高负载下 median latency 好，但会牺牲长请求公平性。因此用参数 $\alpha$ 平滑插值。

interactive 请求优先级：

$$
P_i=t_{arrival}^i+SLO_{TTFT}^i+\alpha\cdot Prefill_{rem}^i.
$$

non-interactive 请求优先级：

$$
P_i=t_{arrival}^i+SLO_{TTLT}^i+\alpha\cdot(Prefill_{rem}^i+Decode_{rem}^i).
$$

$\alpha$ 越小越像 EDF，越大越偏向 shortest-remaining-prefill / shortest-job。scheduler 选择优先级更高的 prefill request 进入当前 batch。

**4. Eager relegation**

在 overload 下，即使再好的 scheduler 也无法保证所有请求。Niyama 的策略是：如果某个请求已经违反 TTFT/TTLT，或本轮即将违反，就把它移入 relegated queue，低负载时再 best-effort 服务。

在 multi-tenant 场景中，relegation 可结合应用 hint，例如 free tier 先被降级，paid tier 后被降级。它不是永久拒绝，而是 graceful degradation。

**5. Selective preemption**

Niyama 允许抢占 prefill queue 中已经处理了一些 chunk 的请求，但有两个约束：

- 只抢占 prefill，不抢占 decode，因为 decode 的 TBT 通常非常紧。
- 被抢占请求不能因此违反自己的 deadline。

这样能避免小 interactive 请求被长 prefill 阻塞，又不引入过大的 KV cache memory pressure。

### Key findings 与定位

- 相比 siloed deployment，Niyama serving capacity 最高提升 **32%**。
- 相比 Sarathi-FCFS，goodput 提升 **1.5x-2.4x**；相比 Sarathi-EDF 提升 **20%-40%**。
- 在 overload 下，Niyama 可承受约 **40%** 更高负载，同时满足 tail latency SLO。
- transient overload 中，Niyama 对 important tasks 无 deadline miss，总体仅约 **8.75%** miss。
- 消融中 dynamic chunking 带来约 **20%** throughput boost，eager relegation 额外约 **9%**。
- **PD 分离：否。** 论文明确假设 co-located LLM inference scheduling，prefill 和 decode 在同一 replica 上执行，并使用 chunked prefill。
- **Scheduling 类型：主要是 instance-scheduling。** 它的算法是 instance 内 prefill selector、chunk size、relegation、preemption；但论文动机和部署收益是 cluster 级“打破 silo、共享资源池”。它没有像 SLOs-Serve/Laser 那样重点设计跨 replica request routing。
- 贡献重点：从“保证所有请求”转向 overload 下“主动牺牲少数已经不可救请求”，避免 cascading deadline violations。

## 6. SABER: Adaptive Request Scheduling for CodeLLM Serving with SLA Guarantees

### 研究问题

SABER 面向自托管 CodeLLM，尤其是小团队或资源受限场景。vLLM 等 continuous batching 引擎通常需要静态 batch size / concurrency 配置，但 CodeLLM workload 的任务类型和请求率变化很大。一个静态配置在某个 workload 最优，换到另一个 workload 或 RPS 就可能 goodput 急剧下降。

研究问题是：在不重启服务、不改动推理引擎内部的情况下，如何动态判断 incoming request 是否还有机会满足 SLA，并调整 admission，提升 SLA goodput。

### 核心算法

**1. Offline speed profiling**

SABER 不直接预测 end-to-end latency，而是预测 token generation speed。离线阶段在不同 concurrency / workload mix 下采样：

$$
v=\frac{\text{generated tokens}}{\text{execution time}},
$$

得到 load-speed 数据对 $(L,v)$，其中 $L$ 是当前 active requests 数。

然后拟合估计函数：

$$
\hat v=f(L).
$$

论文选择 Universal Scalability Law (USL) 作为效果最好的函数族。直觉是：并发增加初期可能提升吞吐，但 contention/coherency overhead 会让 per-request speed 下降。

**2. Two-tier queue**

Online 阶段，每个请求有两个属性：

- deadline，即 tail-latency SLA。
- max generated tokens。

请求所需速度：

$$
reqSpd(r)=
\frac{r.maxTokens}{r.deadline-t_{now}}.
$$

若 $reqSpd(r)$ 已经超过系统最快单请求速度，说明它不可能满足 SLA，则从 high-priority queue 降到 low-priority queue。low-priority queue 只在 high-priority queue 为空时 best-effort 执行。

**3. Admission control**

SABER 持续从 high-priority queue 的 head window 中随机采样请求，避免 head-of-line blocking。对候选请求 $r$：

1. 预测加入后速度：

$$
predSpd=f(curLoad+1).
$$

2. 检查候选请求自己是否满足：

$$
predSpd \ge reqSpd(r).
$$

3. 检查已在执行请求是否会被拖到违反 SLA：

$$
predSpd \ge reqSpd(a),\quad \forall a\in ActiveRequests.
$$

只有两个条件都满足才 admit 到 continuous batching engine，否则继续留在 high-priority queue 等下一轮。

### Key findings 与定位

- 相比 best static configuration，SABER goodput 最高提升 **26%**。
- latency variability 最高降低 **45%**。
- 在无 contention 时，SABER 接近最优静态配置；在中高负载下，通过推迟 hopeless requests 保持已可满足请求的 SLA。
- high-fidelity predictor 很关键：USL 拟合 $R^2$ 高，替换成较弱 predictor 后某些 workload 会因过度保守或错误 admission 而下降。
- **PD 分离：否。** SABER 不区分 PD 架构，也不做 prefill/decode 分离；它面向单 CodeLLM serving engine。
- **Scheduling 类型：instance-scheduling。** 核心是单实例 admission control 和 queue management，不做 cluster routing。
- 贡献重点：工程上轻量、可接入已有 continuous batching engine，适合 resource-constrained CodeLLM serving；但 SLO 模型比 TTFT/TPOT 系统粗，主要是 request completion deadline。

## 7. Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

### 研究问题

Laser 针对 multi-SLO LLM serving 中 iteration-level scheduling 粒度过粗的问题。已有系统每个 iteration 通常执行完整模型 forward pass：

- prefill 阶段：固定 chunk size 会在“低延迟响应”和“高吞吐大 chunk”之间冲突。
- decode 阶段：统一 batch 会让 strict TBT 请求和 relaxed TBT 请求以同一节奏执行，浪费 relaxed 请求的 slack。

研究问题是：能否把调度粒度从 iteration-level 进一步细化到 layer-level，让严格 SLO 请求更快响应，同时把宽松 SLO 请求更高效地合并执行。

### 核心算法

Laser 是本文献集中最明确的 **PD 分离 + 双层调度** 系统：

- prefill instances 处理 prefill。
- decode instances 处理 decode。
- Global Controller 做 inter-instance dispatch。
- instance 内部做 layer-level scheduling。

**1. Modular latency modeling**

Laser 把 transformer layer 拆成 stateless modules 和 stateful self-attention。

Stateless module 延迟用分段线性模型：

$$
\omega(n)=
\begin{cases}
a_0n+b_0, & n\in[1,n_0)\\
\cdots\\
a_mn+b_m, & n\in[n_{m-1},n_m)
\end{cases}
$$

Stateful attention 延迟受 token count $n$ 和总 context length $\sum_{r=1}^{n}c_r$ 影响：

$$
\tau\left(n,\sum_{r=1}^{n}c_r\right)
=
\alpha n+\beta\sum_{r=1}^{n}c_r+\gamma.
$$

单层延迟：

$$
T\left(n,\sum_{r=1}^{n}c_r\right)=
\omega(n)+\tau\left(n,\sum_{r=1}^{n}c_r\right).
$$

**2. Layer-level chunked prefill**

在 prefill instance 中，scheduler 可以在 layer boundary 抢占或合并：

- 新请求到达时，计算其 TTFT slack。
- 若当前正在执行的 chunk 的剩余 iteration time 会让新请求错过 SLO，则在当前 layer 后 checkpoint intermediate state。
- 如果把新请求推进到同一 layer 并合并不会违反新请求 SLO，则 merge 执行。
- 如果 merge 会违反，则直接优先执行新请求。
- 若无需抢占，且 pending queue 为空，也尝试把新请求切成合适 chunk 并 merge 到当前执行 chunk。
- queue 中请求按 EDF 排序。

这解决了 iteration-level chunked prefill 的两个问题：长 chunk 不能及时让路、小 chunk 又吞吐低。

**3. Layer-level decode batching**

每个 decode 请求 $r$ 的 execution plan 有两个参数：

- $L_r$：每个 iteration 执行多少层。
- $O_r$：offset，表示下一次执行延后几个 iteration。

若模型总层数为 $N$，则请求 $r$ 的 layer $j$ 第一次执行在：

$$
\left\lceil\frac{j}{L_r}\right\rceil+O_r,
$$

之后每隔：

$$
\left\lceil\frac{N}{L_r}\right\rceil
$$

个 iteration 执行一次。用二元变量 $x(r,i,j)$ 表示请求 $r$ 是否在 iteration $i$ 执行 layer $j$。layer $j$ 在 iteration $i$ 的延迟可估计为：

$$
T_d\left(
\sum_{r\in R_d}x(r,i,j),
\sum_{r\in R_d}x(r,i,j)c_r
\right).
$$

整轮 iteration latency 是所有 layer latency 聚合。planner 会模拟多个未来 iteration，取最大预测值作为当前 plan 的 per-iteration latency。

**4. Execution plan construction**

Laser 的 planner 不是暴力搜索所有 $(L,O)$，而是 greedy 调整：

- 新请求先设置为 $L_r=N,O_r=0$，即每轮执行完整模型，保证严格响应。
- 设当前目标 $T_g=\min_r SLO_{TBT}(r)$。
- 如果预测 iteration latency $Iter>T_g$，说明过载，则选择 SLO 最宽松的请求，把它的 $L_r$ 降到刚好满足自己 SLO 的最小层数：

$$
L_r=\left\lceil
\frac{N\cdot T_g}{SLO_{TBT}(r)}
\right\rceil.
$$

- 再选 offset $O_r$ 来平衡相同 $L$ group 的 layer workload。
- 如果 $Iter<T_g$，说明还有余量，则优先把最严格请求恢复为 $L=N,O=0$。
- 只在 critical events 触发 plan update，例如请求到达/离开或 latency 接近最紧 SLO。

直觉是：严格请求每轮执行更多层；宽松请求分多轮执行完整模型，从而在同一 decode instance 中容纳更多 relaxed requests。

**5. Inter-instance request dispatching**

Laser 的 Global Controller 对 prefill 和 decode 用不同 dispatch 策略。

Prefill instance selection：

- 混合不同 SLO 的 prefill 请求反而有益，因为可利用 relaxed 请求 slack。
- Controller 收集各 prefill instance 的调度状态。
- 判断哪些 instance 能在 EDF + layer-level chunked prefill 下接收新请求而不违反 TTFT。
- 从 feasible instances 中选择最小 slack 最大的 instance；若没有可行 instance，则选择 prefill tokens 最少的 instance best-effort。

Decode request assignment：

- decode 更适合同 SLO grouping，因为 TBT 差异太大时 batching 效率差。
- Global Scheduler 维护 SLO-homogeneous instance groups。
- 新请求 $r^*$ 被预分配给各 decode instance，由 instance 本地运行 ExecPlan 评估影响。
- 若加入会导致 SLO violation 或 memory shortage，返回 $\infty$。
- 否则返回 aggregated TBT increment：

$$
\sum_{r\in R^*} Iter^*\left\lceil\frac{N}{L_r^*}\right\rceil
-
\sum_{r\in R} Iter\left\lceil\frac{N}{L_r}\right\rceil.
$$

Global Scheduler 先按 group SLO 与请求 SLO 的距离排序，再在 group 内选择 increment 最小且可行的 instance。若都不可行，则 least-loaded best-effort。

### Key findings 与定位

- 相比 Sarathi-Serve 和 DistServe，Laser goodput 最高提升 **1.67x**；在 99% SLO attainment target 下可达 **1.85x**。
- 对 Qwen-14B、Qwen-32B、Llama-70B，分别可获得约 **43.4% / 68.9% / 56.6%** goodput 提升。
- layer-level prefill 在高负载下降低 prefill SLO violation；平均 TTFT 可降低超过 **10%**。
- decode 中 layer-level batching 降低 TBT violation 超过 **6.7%**，再加 group-based decode assignment 额外降低 **10.5%**；TPOT violation 最多降低 **11.8%**。
- latency model 精度约 **94.8%-98.6%**；layer-level switching 开销小于 **1.5%**，scheduling 开销最高约 **3.8%**，prefill dispatch 约 **2 ms**，decode dispatch 约 **10 ms**。
- **PD 分离：是。** Laser 明确建立在 prefill-decode disaggregation 上，并做 KV cache 跨 prefill/decode instance 迁移。
- **Scheduling 类型：instance + cluster。** Instance 内是 layer-level prefill/decode scheduling；cluster/controller 层是 prefill instance selection 和 decode group-based assignment。
- 贡献重点：把调度粒度推进到 layer，使 multi-SLO slack 能在 layer 级别被利用；它也说明在 PD 架构下，cluster dispatch 必须感知 instance 内 layer-level runtime state。

## 8. 横向比较与启示

### 8.1 PD 分离 vs 非 PD 分离

这 7 篇中，只有 **Laser** 明确采用 PD 分离。SLOs-Serve 讨论了 disaggregated scheduling 的局限，但自身采用 co-located mixed prefill/decode token planning。SOLA、SCORPIO、Niyama、SABER、FastServe 都是非 PD 分离。

一个趋势是：

- 非 PD 系统更容易做细粒度 batch/chunk/admission，避免 KV migration 复杂性，适合单实例或中小规模服务。
- PD 系统更适合大规模 production serving，因为 prefill compute-bound、decode memory-bound 的资源需求差异明显，但 multi-SLO 下必须引入更复杂的 inter-instance dispatch。
- Laser 的贡献在于指出：PD 分离本身不够，若每个 instance 内仍是 iteration-level scheduling，就无法充分利用 heterogeneous SLO slack。

### 8.2 Cluster-scheduling vs instance-scheduling

可分成三类：

1. **纯 instance-scheduling**：FastServe、SOLA、SCORPIO、SABER。它们主要改变单个 engine 的队列、batch、admission 或 iteration 计划。
2. **instance scheduling 为主，cluster deployment 为背景**：Niyama。它的目标是打破 cluster 中 interactive/batch silo，但核心算法仍是 instance 内 co-located scheduler。
3. **instance + cluster 联合**：SLOs-Serve、Laser。SLOs-Serve 用 replica-local DP 加 SLO-driven routing；Laser 用 layer-level instance scheduler 加 global prefill/decode assignment。

### 8.3 SLO-aware scheduling 的算法谱系

这些论文的核心思想可以按“调度对象”排序：

- **请求级排序**：FastServe 的 MLFQ，Niyama 的 EDF/SRPF hybrid，SCORPIO 的 LDF。
- **Admission control**：SABER 判断 completion SLA feasibility；SCORPIO 判断 TTFT/TPOT unattainable；SLOs-Serve 用 DP soft admission；Niyama 用 eager relegation。
- **Batch/chunk 级资源分配**：SOLA 控制 $n_i,k_i$；Niyama 控制 prefill chunk size；SLOs-Serve 控制 prefill/decode token budget；SCORPIO 控制 decode frequency。
- **Layer 级执行计划**：Laser 控制每个请求每轮执行的层数 $L_r$ 和 offset $O_r$。
- **Cluster 路由**：SLOs-Serve 基于 SLO attainability 跨 replica routing；Laser 基于 phase-specific feasibility 和 TBT increment 做 instance assignment。

### 8.4 对后续工作的可能启发

- 如果目标是单机或单 replica 的 SLO attainment，SOLA/SCORPIO/Niyama/SABER 更直接：它们的核心问题是 admission、排序、chunk/batch size。
- 如果目标是多副本 serving capacity，SLOs-Serve 的 DP + routing 是更接近 cluster scheduler 的路线。
- 如果目标是 PD 分离架构下的 multi-SLO serving，Laser 最相关：它同时处理 prefill instance selection、decode assignment 和 instance 内 layer-level execution。
- 如果目标是尾延迟和抢占机制，FastServe 是基础：它不解决 heterogeneous SLO，但 token-level preemption、KV swapping 和 starvation prevention 对 SLO scheduler 很有价值。

## 9. 一句话总结

整体来看，SLO-aware LLM scheduling 正在从“固定 batch / FCFS”向四个方向演化：**admission control 保证可满足请求、state-aware scheduling 平衡 TTFT/TPOT、token/layer allocation 利用 SLO slack、cluster-level routing 把请求放到最可能满足 SLO 的实例上**。非 PD 系统更强调单实例 batch/chunk/admission；PD 系统则必须联合考虑 phase-specific instance state 和跨实例 dispatch。Laser 是目前这组论文中唯一明确采用 PD 分离且把 instance-scheduling 与 cluster-scheduling 深度耦合的工作。
