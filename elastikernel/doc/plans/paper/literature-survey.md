# Literature Survey: Resource-Aware Adaptive Kernel Auto-Tuning for Spatially Shared GPUs

> Generated: 2026-03-17
> Research Topic: Partition-aware deep learning kernel compilation and auto-tuning for multi-tenant GPU environments (MPS/MIG/Green Context)
> Target Venues: MLSys, ASPLOS, CGO, PLDI, OSDI, SOSP, EuroSys, ATC

---

## 1. Search Summary

### 1.1 Search Queries Used

| Category | Query | Results Found |
|----------|-------|---------------|
| **Methodological** | GPU spatial partitioning kernel optimization MPS multi-tenant | 10 |
| **Methodological** | resource-aware GPU kernel compilation Triton TVM shared GPU interference | 10 |
| **Methodological** | wave quantization GPU thread block scheduling SM-aware tiling | 10 |
| **Domain** | NVIDIA MIG MPS GPU partitioning performance | 10 |
| **Domain** | L2 cache contention GPU shared memory interference deep learning | 10 |
| **Domain** | GPU multi-tenant deep learning compilation auto-tuning | 10 |
| **Baseline** | KRISP kernel-wise right-sizing spatial partitioned GPU inference | 10 |
| **Baseline** | Orion interference-aware GPU sharing EuroSys 2024 | 10 |
| **Baseline** | MuxServe LLM serving spatial multiplexing GPU MPS | 10 |
| **Baseline** | Missile/SGDRC fine-grained GPU resource isolation | 10 |
| **Baseline** | Ansor TVM auto-scheduler cost model GPU kernel | 10 |
| **Baseline** | FlashAttention tiling SRAM GPU kernel optimization | 10 |
| **Supplementary** | Gpulet GPU resource partitioning MPS inference co-location | 10 |
| **Supplementary** | Bubbleless spatial-temporal sharing GPU EuroSys 2025 | 10 |
| **Supplementary** | StreamBox lightweight GPU sandbox serverless | 10 |
| **Supplementary** | GSLICE GPU resource management sharing inference | 10 |
| **Supplementary** | TritonForge profiling-guided Triton optimization | 5 |
| **Supplementary** | Hierarchical resource partitioning modern GPUs RL | 5 |

### 1.2 Candidate Paper Statistics

- **Total candidates found:** 28
- **Classified as reference (deep-read):** 5
- **Classified as related work (abstract-level):** 18
- **Skipped (not directly relevant):** 5

---

## 2. Reference Papers (Deep-Read for Writing Style)

### 2.1 KRISP: Enabling Kernel-wise RIght-sizing for Spatial Partitioned GPU Inference Servers
- **Venue:** HPCA 2023
- **Authors:** Marcus Chow, Ali Jahanshahi, Daniel Wong (UC Riverside)
- **PDF:** http://www.cs.ucr.edu/~ajaha004/files/KRISP.pdf
- **Platform:** AMD MI50 GPU (60 CUs, 4 Shader Engines), ROCm 4.5

#### Paper Summary
- **Problem:** GPU inference workloads are short-running and under-utilize GPU resources. Existing spatial partitioning techniques (MPS, MIG, AMD CU Masking) apply partitions at the **process or stream scope**, meaning partition resizing requires relaunching processes and reloading ML models (~10s overhead). As a result, servers perform only *model-wise* right-sizing (one partition size per model), leaving significant fine-grained under-utilization because individual kernels within an inference pass have vastly different resource requirements.
- **Key technique:** KRISP performs **kernel-wise right-sizing** — each individual GPU kernel within an inference pass gets its own partition size. Two parts: (1) **Profiling stage** (done once) determines the minimum CUs each kernel needs for peak latency — kernel size and input data size do NOT reliably predict this, so empirical profiling is necessary. (2) At runtime, the GPU runtime intercepts each kernel launch, looks up its minimum CU requirement, and a hardware-level **Kernel-scoped Partition Instance** generates a per-kernel resource mask using a "Conserved" CU distribution policy (Algorithm 1). The key enabler is extending the AMD AQL packet with a partition-size field and modifying Command Processor firmware.
- **System architecture:**
  1. **ML Framework Layer** (PyTorch/TF) — unmodified
  2. **GPU Runtime (ROCm/ROCR)** — intercepted layer with Required CUs Table + Kernel Right-Sizing Module
  3. **GPU Hardware (Command Processor)** — modified packet processor + resource monitor (per-CU kernel counters, 5 bits/CU) + Algorithm 1 resource allocation
  4. **Workgroup Dispatcher** — enforces resource mask, schedules TBs to assigned CUs (existing HW, unmodified)
  5. **Inference Server** — gRPC multi-threaded request handler with independent worker processes
- **Evaluation highlights:**
  - **Throughput:** KRISP-I (isolated) improves total throughput by ~2x over single-worker isolated inference; 1.22x over Static Equal at 4 concurrent workers; up to ~3.5x over MPS Default
  - **Tail Latency:** KRISP-I meets 2x SLO target for most models at 4 concurrent workers (other policies violate SLO)
  - **Energy:** Reduces energy per inference by 29% (2 workers) and 33% (4 workers)
  - **Concurrency:** Supports 4 concurrent instances for 5/8 models (vs. only alexnet for others)
  - **9 models tested:** albert, alexnet, densenet201, resnet152, resnext101, shufflenet, squeezenet, vgg19
  - **Key negative result:** Neither kernel size (thread count) nor input data size reliably predicts minimum CU requirements (Figures 6a-6b), justifying profiling-based approach

#### Writing Style Analysis (Deep-Read)
- **Introduction pattern:** Disciplined narrowing funnel: (1) broad context (ML inference + GPUs) → (2) existing solution (spatial partitioning) → (3) limitation (model-wise only) → (4) root cause (process-scoped partitioning, ~10s reload) → (5) insight (kernel-scoped would avoid this) → (6) contributions (bulleted list)
- **Three-panel progression figures:** Figure 1 (no partitioning → model-wise → kernel-wise) and Figure 2 (naive resize → shadow instance → kernel-scoped) visually walk reader through problem escalation — **extremely effective pattern for granularity-refinement papers**
- **"Case for X" section (Section III):** Dedicated motivation section BEFORE design with model-level sensitivity curves (Fig 3) and kernel-level phase traces (Fig 4) — makes design feel inevitable
- **Negative result justifies design:** Shows kernel size/input size don't predict CU needs, strengthening the case for profiling
- **Table-based positioning:** Tables I & II compare spatial partitioning techniques along clear dimensions (scope, granularity, overhead, transparency), with "This work" as last row
- **Hardware overhead analysis:** Precisely quantified (300 bits for CU counters, 1μs for mask generation) — expected at architecture venues
- **Generalizability section (IV-D4):** Explicitly addresses Nvidia applicability despite being AMD-focused

#### Relevance to Our Work
- **Closest prior work** — KRISP adjusts SM/CU allocation per-kernel but does NOT modify the kernel code itself. It only changes how many CUs run the kernel. Our work goes deeper: we propose re-compiling/re-tuning the kernel's tiling, grid size, and memory access strategy to be optimal FOR a given SM allocation.
- **Key difference:** KRISP = resource allocator (decides partition size per kernel); **Ours = kernel optimizer (makes kernel run optimally within whatever partition it gets)**. The two are **complementary** — KRISP decides the right partition, our work generates optimal code for that partition.
- **KRISP requires hardware modifications** (AQL packet extension, CP firmware changes); **our work is pure software**
- **Writing style lessons for our paper:**
  1. Adopt the three-panel progression figure: (a) default kernel tuning ignoring resources, (b) static tuning for fixed partition, (c) adaptive per-kernel tuning aware of current availability
  2. Create dedicated "Case for Resource-Aware Kernel Tuning" section with empirical data
  3. Use comparison table contrasting auto-tuning approaches along dimensions: resource-awareness, granularity, spatial-sharing awareness
  4. Include negative results (e.g., tuning parameters optimal for full GPU are suboptimal under partitioning)
- **Citation context:** "While KRISP [Chow et al., HPCA'23] demonstrates that per-kernel CU allocation improves efficiency, it does not modify the kernel code itself. The kernel's tiling strategy, grid dimensions, and memory access patterns remain optimized for full GPU occupancy, leading to sub-optimal execution when run on a reduced partition. Our work addresses this complementary dimension: generating kernels that are intrinsically optimized for a given resource partition."

---

### 2.2 Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications
- **Venue:** EuroSys 2024
- **Authors:** Foteini Strati, Xianzhe Ma, Ana Klimovic (ETH Zurich)
- **PDF:** https://fotstrt.github.io/files/2024-orion.pdf
- **Code:** https://github.com/eth-easl/orion
- **Implementation:** ~3000 lines C++/CUDA, integrated into PyTorch
- **Hardware:** NVIDIA V100-16GB (primary), A100-40GB (generalization)

#### Paper Summary
- **Problem:** GPU resources are chronically underutilized by individual DNN workloads because each workload consists of many short-lived operators (10s-1000s of microseconds) with different compute and memory requirements — when one resource type is saturated, the other is left idle. Existing sharing techniques are either too coarse-grained (MIG, Tick-Tock) or not sufficiently interference-aware (MPS, Streams, REEF).
- **Key technique:** Orion intercepts CUDA kernel launches from multiple clients and buffers them in per-client software queues. A centralized scheduler decides whether to launch a best-effort kernel alongside the high-priority kernel based on three criteria: (a) **resource profile complementarity** — the BE kernel must have opposite compute/memory intensity to the HP kernel; (b) **kernel size** — SM requirements below configurable `SM_THRESHOLD`; (c) **duration throttling** — cumulative outstanding BE kernel duration below `DUR_THRESHOLD` (default 2.5% of HP latency). Kernel profiles are collected offline using Nsight tools.
- **System architecture:**
  1. **Client Applications** — multiple DNN workloads (PyTorch) submitting GPU operations
  2. **Orion Interception Layer** — dynamically linked library overriding CUDA runtime API calls
  3. **Per-Client Software Queues** — buffer intercepted GPU ops in shared memory
  4. **Offline Profiler** — Nsight Compute/Systems per-kernel resource profiles
  5. **Orion Scheduler** — central scheduling loop polling queues, applying three-criteria policy
  6. **GPU Hardware** — separate CUDA streams (HP stream for priority client, default for BE)
- **Evaluation highlights:**
  - **Inf-Train:** p99 inference latency within 14% of ideal (vs. REEF at 2.5-3.4x higher). Aggregate throughput up to 2.3x of dedicated GPU
  - **Train-Train:** HP training throughput within 16% of ideal, BE makes meaningful progress
  - **Inf-Inf:** p99 latency within 15-22% of ideal (vs. MPS/Streams at 1.89x). Aggregate throughput up to 7.3x of dedicated GPU. 2x cost savings
  - **Multi-client (A100):** 5 inference clients; p99 latency within 9% of ideal
  - **Utilization improvement:** SM utilization from 11% to 49%, compute from 7% to 36%, memory BW from 10% to 47%
  - **Overhead:** Kernel interception <1%. Offline profiling ~2-5s per kernel (one-time)

#### Writing Style Analysis (Deep-Read)
- **Introduction — "Elimination funnel":** Opens with GPU importance → immediately introduces tension (underutilization) → systematically eliminates alternative explanations (data stalls, communication, batch sizes all removed, problem persists = fundamental) → introduces taxonomy (temporal vs spatial sharing) → shows each is insufficient with Figure 2 → states contribution with quantified results
- **Section 3 — Motivation section BEFORE design:** Profiling data (Table 1), kernel classification (Fig 4), and a **toy experiment (Table 2)** validating core hypothesis: co-locating kernels with opposite resource profiles yields 1.41x speedup, same-profile yields none. Reader is convinced the approach works before seeing the system design.
- **Compact pseudocode:** Core algorithm as ~30-line Listing 1. Policy explained as three simple conditions.
- **Modular design decomposition:** Policy → mechanisms (stream priorities, CUDA events) → memory management → profiling → framework integration
- **Evaluation opens with explicit research questions** (6 bullet points) then answers each
- **Ablation study:** Progressively adds Orion components to show incremental contribution
- **Discussion section:** Openly addresses limitations (cluster co-design, cache interference, security, LLMs)
- **Quantified claims throughout:** "within 14%", "up to 7.3x" — every comparison has numbers

#### Relevance to Our Work
- **Complementary work** — Orion decides WHICH kernels to co-schedule; our work optimizes HOW each kernel is compiled for a shared environment. Orion treats kernels as black boxes.
- **Key insight for us:** Orion's compute-vs-memory classification can inform our compilation strategy — a memory-bound kernel under sharing should be compiled with more aggressive cache bypass.
- **Writing style lessons:**
  1. **Elimination funnel in Introduction:** Show that even with MPS/MIG, kernel performance degrades because kernels are *tuned for full-GPU execution*
  2. **Early motivating experiment:** Like Table 2, include "a kernel auto-tuned for full GPU gets X% slower under 50% SM partition; our resource-aware tuning recovers Y%"
  3. **Dedicated motivation section before design** with profiling data
  4. **Explicit research questions before evaluation**
  5. **Progressive ablation study**
- **Citation context:** "Orion [Strati et al., EuroSys'24] demonstrates that interference-aware kernel co-scheduling significantly improves GPU sharing efficiency by pairing compute-bound and memory-bound kernels. However, it treats kernels as black boxes — selecting which kernels to run concurrently without modifying kernel code to be inherently resilient to interference."

---

### 2.3 Usher: Holistic Interference Avoidance for Resource Optimized ML Inference
- **Venue:** OSDI 2024
- **Authors:** Sudipta Saha Shubha et al.
- **PDF:** https://www.usenix.org/system/files/osdi24-shubha.pdf
- **Hardware:** 6 AWS EC2 p3.8xlarge (24 V100 GPUs total); Simulation: up to 6000 GPUs

#### Paper Summary
- **Problem:** Spatially multiplexing multiple ML inference models on a single GPU is desirable for cost efficiency, but existing systems suffer from inter-model interference (compute, memory, cache) causing SLO violations. Current approaches either avoid multiplexing entirely (leaving GPUs at 25-50% utilization) or multiplex but only optimize compute utilization while ignoring memory utilization and cache interference, reducing goodput by up to half.
- **Key technique:** Three-stage pipeline: (1) **GK-Estimator** — GPU kernel-based resource requirement estimator that analyzes operator graphs at kernel level, using stacked regression models to estimate compute and memory requirements without running the model (99.98% accuracy vs. offline profiling at zero GPU cost, in 31.6ms vs. 4.8 hours/$42.70 for offline); (2) **IR-Scheduler** — groups models via k-means so C-heavy and M-heavy models are paired, then performs holistic workload division (batch size, replication degree, placement) via heuristic bin-packing; (3) **OG-Merger** — groups architecturally similar models via DBSCAN, finds weight-similar operators via Hungarian matching, creates "Super-Operators" with fused GEMMs sharing common weight submatrices to minimize cache interference.
- **System architecture:**
  1. **GK-Estimator** — Mreq-Regressor + Time-Regressor (stacked ensemble models)
  2. **IR-Scheduler** — Model Grouping (k-means: C-heavy paired with M-heavy) + Configuration & Placement (Algorithms 1+2)
  3. **OG-Merger** — DBSCAN grouping → Hungarian matching → Super-Operator creation with common submatrix extraction
  4. MPS-based execution with allocated thread percentages; re-runs pipeline on workload changes (>0.5k req/s, every 45-300s)
- **Evaluation highlights:**
  - **20 DL models** (CNN + Transformer): LLaMA-2 13B, GPT-2, BERT, ResNets, Inceptions
  - **Baselines:** Shepherd (no multiplexing), GPUlet (MPS-based, compute-focused), AlpaServe (model parallelism + statistical multiplexing)
  - **Goodput (fixed cluster):** Up to 2.6x higher than baselines; 22-24.2% higher Cuti and 38.9-40.1% higher Muti vs. Shepherd
  - **Cost (non-fixed, heterogeneous):** 2.8x-3.5x lower cost
  - **GK-Estimator:** 99.98% accuracy, 31.6ms estimation time
  - **Ablation:** Each component contributes 24.3%-55.7% of total improvement
  - **SLO sensitivity:** Retains superiority even under ultra-strict SLOs (0.4x default)

#### Writing Style Analysis (Deep-Read)
- **Introduction opens with economic pain point:** "inference = 90% of production costs, $76B projected annual cost by 2028" — establishes urgency
- **"Observation-driven design" pattern:** Section 2 (before design) derives 6 numbered, boxed "Observations" (O1-O6) from real experiments. Each observation is directly referenced in design section ("Based on O2-O6"), creating tight logical chain from empirical findings to design decisions. **This is an extremely effective rhetorical device for systems papers.**
- **Running examples:** Small concrete scenarios (M1-M4 with specific percentages) build intuition before algorithms
- **Consistent notation:** (Cuti, Muti, Creq, Mreq, C-heavy, M-heavy, GPU#, model#) used uniformly
- **Formulation-first method section:** Opens with formal optimization problem, then decomposes into sub-problems
- **Evaluation:** Two major scenarios (fixed cluster / non-fixed cluster) with sub-cases, dedicated ablation + parameter sensitivity subsections

#### Relevance to Our Work
- **Motivational evidence** — Usher confirms our core thesis: interference is multi-dimensional (compute, memory bandwidth, cache)
- **Key difference:** Usher avoids interference through scheduling and graph merging; we mitigate it through code generation
- **Naturally complementary:** Usher allocates MPS thread percentages and places models, but assumes kernel performance is fixed. Our adaptive kernel tuning sits underneath: re-tuning kernels when Usher changes resource allocation, potentially improving Usher's resource estimates
- **Writing style lessons:**
  1. **"Observation-driven design"** — present empirical observations boxed/numbered, then reference them in design
  2. **Economic pain point in intro** establishes urgency
  3. **Running examples** with small concrete scenarios before general algorithms
  4. **Consistent compact notation** makes complex paper readable
- **Citation context:** "Usher [Shubha et al., OSDI'24] provides holistic interference modeling across multiple GPU resource dimensions, achieving 2.6x goodput improvement through kernel-level resource estimation and interference-aware scheduling. Our work leverages similar insights but applies them at the compiler level — generating kernels that are inherently resilient to interference rather than avoiding it through scheduling."

---

### 2.4 MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving
- **Venue:** ICML 2024
- **Authors:** Jiangfei Duan, Jiali Lu et al. (UCSD Hao AI Lab)
- **arXiv:** 2404.02015
- **Code:** https://github.com/hao-ai-lab/MuxServe
- **Hardware:** 4-node cluster, 8x NVIDIA A100 (80GB) per node (32 GPUs total), NVLink + 200Gbps IB

#### Paper Summary
- **Problem:** LLM endpoint providers must serve multiple LLMs concurrently. Spatial partitioning (dedicating GPUs to each LLM) leads to idle GPUs for unpopular models; temporal multiplexing ignores the distinct compute profiles of prefill vs decode phases, leaving GPUs underutilized during decode-dominated inference.
- **Key technique:** MuxServe introduces **flexible spatial-temporal multiplexing** built on two core insights: (1) **Disaggregates prefill and decode phases** into separate jobs with independently configurable SM allocations via CUDA MPS (decode is memory-bound, uses few SMs; prefill is compute-bound); (2) **Co-locates LLMs based on popularity** — popular models paired with unpopular ones. Three algorithmic components: (a) enumeration-based greedy placement algorithm; (b) **Adaptive Batch Scheduling (ADBS)** that round-robins prefill with priority, fills remaining capacity with decode, dynamically rebalances KV cache quotas; (c) unified resource manager with head-wise KV cache blocks for multi-LLM cache sharing.
- **System architecture:**
  1. **Global Scheduler** — maintains request queues, runs ADBS algorithm
  2. **Placement Optimizer** (offline) — enumeration-based greedy algorithm determining LLM-to-GPU-mesh assignments
  3. **Parallel Runtime** — manages SM partitioning via NVIDIA MPS, assigns different SM fractions to prefill vs decode
  4. **Memory Manager** (C++) — unified memory: head-wise KV cache pool + shared model weights + activation scratch; communicates via CUDA IPC
  5. **Runtime Engines** — separate vLLM processes for prefill and decode, each with different MPS SM limits
- **Evaluation highlights:**
  - **19 LLMs** (LLaMA family): 12 small (4-8B), 4 medium (8-21B), 2 large (21-41B), 1 XL (41-70B)
  - **Baselines:** Spatial partitioning with vLLM; Temporal multiplexing (AlpaServe-style)
  - **Up to 1.8x higher throughput** vs. both baselines on synthetic workloads
  - **Up to 2.9x more requests** served within 99% SLO attainment
  - **Real ChatLMSYS traces:** 1.38x over spatial, 1.46x over temporal
  - **Key empirical finding (Fig 3 / sm_relative):** Decode-phase latency is relatively insensitive to SM reduction (100% → 30% barely changes latency)
  - **Ablation:** Placement algorithm 1.3x over naive greedy; ADBS 1.43x over Round-Robin, 1.85x over FCFS

#### Writing Style Analysis (Deep-Read)
- **Introduction — "Straw man elimination":** Presents real-world scenario (multi-LLM endpoint), then systematically introduces and dismantles spatial partitioning (wastes GPUs on unpopular models) and temporal multiplexing (ignores prefill/decode asymmetry) using a **single running figure (Figure 1) with three panels (a, b, c)** contrasting GPU utilization
- **Early real traffic data (Figure 2)** as motivation — uses actual LLM popularity distributions
- **Formulation-first method section:** Opens with formal optimization problem (Eq. 1-2), then decomposes into three sub-problems (placement, scheduling, resource management) each with own subsection
- **Algorithms as pseudocode** (Algorithms 1-3) with inline explanation
- **Throughput estimator (Eq. 3):** Derived from intuitive observation (prefill runs sequentially, decode concurrently)
- **Implementation paragraph:** Concrete system details (built on vLLM, Python multiprocessing, C++ memory manager, CUDA IPC)
- **Standard systems evaluation format:** setup → end-to-end (synthetic) → end-to-end (real traces) → ablation

#### Relevance to Our Work
- **Directly motivates our work** — MuxServe uses MPS to partition SMs but runs **standard kernels compiled for full GPU**. This is exactly the scenario where our resource-aware compilation would help.
- **Key limitation we address:** SM allocation is **statically determined offline** and fixed per process. Kernels designed for 108 SMs (full A100) run on ~30 SMs with potentially suboptimal thread-block configurations, tile sizes, and occupancy. MuxServe acknowledges interference overhead (higher P99 TPOT) but does not mitigate at kernel level.
- **MPS limitations exposed:** MPS lacks fine-grained isolation — co-located kernels interfere via shared L2 cache, memory bandwidth, and other microarchitectural resources that MPS does not partition.
- **Our work is complementary:** MuxServe (or any spatial-sharing serving system) would benefit from kernels auto-tuned for their actual MPS partition, reducing interference overhead and enabling more aggressive SM sharing.
- **Citation context:** "MuxServe [Duan et al., ICML'24] demonstrates that MPS-based spatial partitioning is effective for multi-LLM serving, achieving 1.8x throughput improvement. However, the kernels executed within each partition remain compiled for full GPU occupancy. The SM allocation is fixed per process, and kernels designed for 108 SMs run on ~30 SMs with suboptimal configurations — creating the fundamental mismatch our work addresses."

---

### 2.5 Understanding GPU Resource Interference One Level Deeper
- **Venue:** SoCC 2025 (arxiv 2501.16909)
- **Authors:** Paul Elvinger et al. (ETH Zurich EASL Group)
- **arXiv:** 2501.16909
- **Hardware:** NVIDIA H100 NVL (132 SMs, 50MB L2) and RTX 3090 (82 SMs)

#### Paper Summary
- **Problem:** Most GPU sharing systems rely on oversimplified, single-dimensional utilization metrics (SM utilization, memory utilization) to model interference. This leads to poor co-location decisions. The paper asks: how to accurately measure GPU utilization and estimate interference across all multifaceted GPU resources?
- **Key technique:** Custom CUDA microbenchmarks stressing specific GPU resources (L2 cache, memory bandwidth, shared memory, warp scheduler IPC, compute pipelines). Co-locates real workloads (LLM decode via vLLM with Gemma3-1B and Llama3.1-8B) with stress benchmarks. Uses **CUDA Green Contexts** for inter-SM isolation (true mutual exclusion of SMs, unlike MPS). Demonstrates two pitfalls of state-of-the-art:
  - **Pitfall 1 (Usher):** Relying on achieved occupancy alone. A kernel with 6.25% occupancy still causes 1.73x slowdown because occupancy doesn't capture compute pipeline saturation.
  - **Pitfall 2 (Orion):** Classifying kernels as compute/memory-bound via arithmetic intensity while ignoring IPC. A kernel with IPC 3.99 (max 4) causes 2x slowdown to co-located memory-bound kernel.

#### Key Findings about L2 Cache Interference (Critical for Our Motivation)
- **L2 Cache Pollution (capacity contention):**
  - Two `copy` kernels on separate Green Contexts (64 SMs vs 68 SMs) on H100 (50MB L2)
  - Input sizes ≤ 8MB: both fit in L2, no slowdown
  - **At 16MB input per kernel (combined 64MB > 50MB L2): slowdown peaks at 2.15x** due to mutual cache eviction
  - Beyond 26MB: slowdown plateaus at ~1.12x (L2 locality already lost in isolation)
  - **This IS the "tragedy of the commons" for L2 cache**
- **L2/Memory Bandwidth Contention:**
  - Co-located Llama3.1-8B decode (64 SMs) with copy kernel (68 SMs) on separate Green Contexts
  - As copy kernel thread blocks increase 34→136: L2 BW utilization 37%→95%, memory BW 27%→81%
  - **P90 TBT increases from 16.9ms to 22ms — 1.3x slowdown**
  - **Even with full SM isolation via Green Contexts, L2 and memory BW remain shared and cause measurable interference**
  - All LLM decode kernels have poor L2 cache locality → inherently susceptible to BW contention

#### Key Findings about SM Interference
- **Block Scheduler / Wave Quantization:**
  - Co-locating Llama3-8B decode with lightweight `sleep` kernel (1 block/SM, 128 threads, 2048 registers/block) → remaining budget 63,288 registers/SM. Decode kernel needs 64,512 registers/block → exceeds capacity
  - **P90 TBT: 7.53ms → 16.56ms (2.2x slowdown)** due to head-of-line blocking
  - Takeaway: Today's kernels maximize per-SM resource usage, leaving almost no room for co-location
- **IPC / Warp Scheduler Contention:**
  - At IPC 1.18-2.90: TBT nearly unchanged (~5.75-6.24ms vs 5.59ms baseline)
  - **At IPC 3.45 (near architectural limit of 4): TBT jumps to 10.74ms (1.92x slowdown)**
- **Shared Memory Bank Conflicts:**
  - Co-located GEMM with strided copy kernel: **32-way bank conflict → 1.79x slowdown (dim 2048), 3.75x (dim 1024)**
- **Positive result:** Restricting LLM decode to <50% SMs causes **only 1.19x TBT slowdown** — trading marginal per-kernel performance for co-location is viable

#### Quantitative Evidence We Can Cite
1. **2.15x slowdown** from L2 cache pollution (Section 4.3, Figure 4)
2. **1.3x TBT slowdown** from memory BW contention even with complete SM isolation via Green Contexts (Section 4.3, Table 1)
3. **2.2x TBT slowdown** from block scheduler head-of-line blocking (Section 4.2)
4. **1.92x TBT slowdown** from warp scheduler IPC saturation (Section 4.4, Table 2)
5. **Up to 3.75x slowdown** from shared memory bank conflicts (Section 4.4.1, Figure 5)
6. **Only 1.19x slowdown** when restricting to <50% SMs — resource-aware scheduling with modest trade-offs is viable (Section 5.1)

#### Relevance to Our Work
- **Critical motivational evidence** — Provides the quantitative ammunition for both our key observations:
  - **Wave Quantization Trap:** Block scheduler interference (2.2x) and register pressure conflicts directly demonstrate our argument
  - **Tragedy of L2 Cache Commons:** 2.15x L2 pollution slowdown is textbook evidence
  - **Even with SM isolation, shared resources cause interference** — this motivates defensive compilation beyond just grid sizing
- **Supports our cost model design:** Bandwidth interference is non-linear → supports stochastic/noise-aware cost model proposal
- **Uses CUDA Green Contexts:** Demonstrates the exact technology we target (Green Context provides true SM isolation but NOT L2/BW isolation)
- **Citation context:** "Elvinger et al. [SoCC'25] reveal that GPU resource interference is fundamentally multi-dimensional and non-linear: L2 cache pollution causes up to 2.15x slowdown, memory bandwidth contention causes 1.3x slowdown even with complete SM isolation via Green Contexts, and block scheduler conflicts cause 2.2x slowdown from register pressure. These findings motivate our defensive compilation strategy that generates kernels intrinsically resilient to these interference vectors."

---

## 3. Related Work Papers (For Citations)

### 3.1 GPU Spatial Partitioning & Multi-Tenant Serving

| Paper | Venue | Core Idea | Relevance to Us | Key Difference |
|-------|-------|-----------|-----------------|----------------|
| **GSLICE** (Dhakal et al.) | SoCC 2020 | Dynamic GPU resource allocation framework using MPS for controlled spatial sharing of inference functions | Early work on MPS-based spatial sharing | Focuses on resource allocation, not kernel optimization |
| **Gpulet** (Choi et al.) | ATC 2022 | Multi-model ML inference serving with GPU spatial partitioning via MPS; predicts interference and adjusts partition size + batch size | Demonstrates performance prediction under sharing | Adjusts partition size but not kernel code |
| **ParvaGPU** (arxiv 2409.14447) | arxiv 2024 | Combines MIG + MPS for fine-grained spatial GPU sharing; minimizes underutilization within partitions | MIG+MPS combination for cloud inference | Does not modify kernel compilation |
| **SGDRC/Missile** (Zhang et al., arxiv 2407.13996) | arxiv 2024 (renamed) | Software-defined dynamic VRAM bandwidth and compute unit management; reverse-engineers NVIDIA VRAM channel hash mapping | Fine-grained resource isolation via software | Hardware-level focus vs. our compiler-level approach |
| **Hierarchical Resource Partitioning** (arxiv 2405.08754) | arxiv 2024 | RL-based co-optimization of MIG+MPS hierarchical partitioning setup and job co-scheduling | Automated partition configuration | Optimizes partition config, not kernel code |

**Suggested citation paragraph:**
> Spatial GPU sharing has been extensively studied at the resource management level. GSLICE [Dhakal et al., SoCC'20] pioneered dynamic MPS-based GPU resource allocation, while Gpulet [Choi et al., ATC'22] extended this with interference-aware partition sizing and batch adjustment. ParvaGPU [2024] further combines MIG and MPS for hierarchical isolation. SGDRC [Zhang et al., 2024] achieves fine-grained resource control by reverse-engineering NVIDIA's VRAM channel architecture. However, all these systems treat GPU kernels as opaque units — adjusting *how many* resources a kernel receives without modifying *how* the kernel utilizes those resources. Our work addresses this fundamental gap by making the kernel compiler aware of its resource constraints.

---

### 3.2 GPU Kernel Auto-Tuning & Compilation

| Paper | Venue | Core Idea | Relevance to Us | Key Difference |
|-------|-------|-----------|-----------------|----------------|
| **TVM** (Chen et al.) | OSDI 2018 | End-to-end optimizing compiler for deep learning with ML-based cost model | Foundational DL compiler; our baseline | Assumes full GPU occupancy |
| **Ansor** (Zheng et al.) | OSDI 2020 | Hierarchical search space for tensor program generation with learned cost model | State-of-the-art auto-scheduler; our baseline | Cost model assumes clean, unshared environment |
| **BOLT** (MLSys 2022) | MLSys 2022 | Bridges gap between auto-tuners and hardware-native performance; prototyped in TVM | Improved auto-tuning pipeline | Still targets exclusive GPU |
| **Triton** (Tillet et al.) | MAPL 2019 | Intermediate language and compiler for tiled neural network computations | Core language for our prototype | Block-level programming assumes full GPU |
| **TritonForge** (arxiv 2512.09196) | arxiv 2025 | Profiling-guided automated Triton kernel optimization via iterative code transformation | Related optimization approach | Single-tenant profiling only |
| **Two-Stage GPU Kernel Tuner** (arxiv 2601.12698) | arxiv 2026 | Combines semantic refactoring with search-based optimization for GPU kernels | Complementary tuning approach | Does not consider resource sharing |
| **The Anatomy of a Triton Attention Kernel** (arxiv 2511.11581) | arxiv 2025 | Cross-platform paged attention kernel in Triton with auto-tuning | Shows importance of Triton parameter tuning | Full-GPU assumption in tuning |

**Suggested citation paragraph:**
> Deep learning compilers have achieved remarkable performance through automated kernel optimization. TVM [Chen et al., OSDI'18] introduced ML-based cost models, and Ansor [Zheng et al., OSDI'20] extended this with hierarchical search spaces. Triton [Tillet et al., MAPL'19] provides a higher-level tiled programming model. Recent work including TritonForge [2025] and the Two-Stage Kernel Tuner [2026] further refine the optimization pipeline. However, all existing auto-tuning frameworks share a critical implicit assumption: the compiled kernel will exclusively occupy the entire GPU. Their cost models target peak hardware utilization (e.g., full memory bandwidth, all SMs active), leading to configurations that are optimal in isolation but degrade non-linearly under resource contention.

---

### 3.3 GPU Kernel Scheduling & Preemption

| Paper | Venue | Core Idea | Relevance to Us | Key Difference |
|-------|-------|-----------|-----------------|----------------|
| **REEF** (Han et al.) | OSDI 2022 | Microsecond-scale kernel preemption for concurrent GPU DNN inferences | Enables fine-grained GPU time-sharing | Temporal sharing, not spatial; kernel code unchanged |
| **Bubbleless Spatial-Temporal Sharing** (Zhang et al.) | EuroSys 2025 | Adaptive GPU resource allocation eliminating idle "bubbles" in spatial-temporal sharing | Fine-tuned resource allocation | Scheduling-level optimization |
| **StreamBox** (Wu et al.) | ATC 2024 | Lightweight GPU sandbox using CUDA streams for serverless inference | Isolation mechanism for multi-tenant GPU | Focus on isolation, not kernel optimization |

**Suggested citation paragraph:**
> Complementary to spatial partitioning, temporal scheduling approaches like REEF [Han et al., OSDI'22] enable microsecond-scale kernel preemption for priority-aware GPU sharing. Recent work on Bubbleless Spatial-Temporal Sharing [Zhang et al., EuroSys'25] adaptively eliminates idle resource bubbles. StreamBox [Wu et al., ATC'24] provides lightweight GPU sandboxing for isolation. These works focus on *when* and *where* kernels execute, while our work focuses on *how* kernels should be compiled for resource-constrained execution.

---

### 3.4 LLM Serving with GPU Sharing

| Paper | Venue | Core Idea | Relevance to Us | Key Difference |
|-------|-------|-----------|-----------------|----------------|
| **Bullet** (arxiv 2504.19516) | arxiv 2025 | Spatial-temporal orchestration for LLM serving; identifies wave quantization as root cause of prefill inefficiency | **Directly names wave quantization** as a problem | System-level orchestration, not kernel recompilation |
| **GPU Multitasking Vision** (arxiv 2508.08448) | arxiv 2025 | Vision paper arguing GPUs must embrace multitasking like CPUs did; proposes OS-like resource management layer | Motivational context for our work | High-level vision, no specific solutions |
| **LithOS** | SOSP 2025 | Operating system for efficient ML on GPUs with spatial multitenancy | OS-level GPU management | Manages resources, doesn't recompile kernels |
| **ConCo** | ICS 2025 | Optimizing compilation of concurrent tensor programs on shared GPU | Related compilation approach | Focuses on scheduling concurrent programs |

**Suggested citation paragraph:**
> In the LLM serving domain, Bullet [2025] explicitly identifies wave quantization as a root cause of GPU underutilization during prefill and proposes spatial-temporal orchestration. The GPU multitasking vision paper [2025] argues that GPUs must evolve toward CPU-like multitasking with proper resource management. LithOS [SOSP'25] builds an OS layer for GPU multitenancy. Our work provides the missing compiler-level component: generating kernels that perform optimally within the constrained resource partitions these systems create.

---

### 3.5 Hardware-Aware Kernel Design (Memory Hierarchy)

| Paper | Venue | Core Idea | Relevance to Us | Key Difference |
|-------|-------|-----------|-----------------|----------------|
| **FlashAttention** (Dao et al.) | NeurIPS 2022 | IO-aware attention algorithm using tiling to minimize HBM reads/writes by maximizing SRAM usage | Gold standard for memory-hierarchy-aware kernel design | Designed for full GPU; our work extends this philosophy to partial GPU |
| **FlashAttention-2** (Dao) | ICLR 2024 | Improved parallelism and work partitioning; notes "L2 cache is not directly controllable by programmer" | Acknowledges L2 unpredictability | Does not address multi-tenant L2 contention |
| **FlashAttention-3** (Dao et al.) | NeurIPS 2024 | Exploits GPU asynchrony and low-precision for further speedup | Latest in the FlashAttention line | Still assumes exclusive GPU |

**Suggested citation paragraph:**
> FlashAttention [Dao et al., NeurIPS'22] demonstrated the power of IO-aware kernel design by tiling attention to maximize SRAM usage and minimize HBM accesses. FlashAttention-2 explicitly notes that "the L2 cache is not directly controllable by the programmer," focusing optimization on SRAM and HBM. Our work takes this philosophy further: in multi-tenant environments where L2 cache is not only uncontrollable but actively contested, we propose "defensive" compilation strategies that treat L2 as unreliable and maximize on-chip (shared memory + register) data reuse.

---

## 4. Gap Analysis: Why This Paper is Needed

### 4.1 The Fundamental Gap

| Existing Work | What They Do | What They Don't Do |
|--------------|-------------|-------------------|
| KRISP, Gpulet, ParvaGPU | Adjust SM allocation per-kernel/model | Don't modify kernel code for the allocation |
| Orion, Usher | Schedule kernels to minimize interference | Don't make kernels resilient to interference |
| TVM/Ansor, Triton | Compile optimal kernels via auto-tuning | Assume exclusive GPU; cost models ignore sharing |
| MuxServe, Bullet | Partition GPU for LLM serving | Run standard kernels in partitions |
| FlashAttention | IO-aware kernel design | Assumes full GPU, exclusive L2 cache |

**The gap:** No existing work **re-compiles or re-tunes GPU kernels to be optimal for a specific resource partition and robust against multi-tenant interference.** This is the exact contribution space for our paper.

### 4.2 Specific Technical Gaps

1. **No partition-aware tiling/grid sizing in any compiler:**
   - TVM/Ansor grid sizes target physical SM count, not allocated SM count
   - Triton's `tl.num_programs()` returns the total grid dimension, not available SMs
   - No auto-tuner considers `Grid_Size mod SM_allocated ≈ 0` as an optimization objective

2. **No interference-aware cost model:**
   - All auto-tuners (Ansor, AutoTVM, Triton autotuner) profile on a clean, unshared GPU
   - No cost model accounts for bandwidth variance or L2 cache pollution from neighbors
   - Bullet [2025] identifies wave quantization but addresses it via scheduling, not compilation

3. **No cache-defensive code generation:**
   - No compiler explores cache bypass (`.cs` hints) as part of the search space
   - No auto-tuner trades register pressure for L2 independence
   - FlashAttention's SRAM maximization is a manual design, not an automated compiler strategy

4. **No JIT kernel family generation:**
   - Green Context / dynamic MPS can change SM allocation at runtime
   - No system generates a family of kernel variants pre-tuned for different partition sizes
   - KRISP does per-kernel SM sizing but still uses the same kernel binary

---

## 5. Writing Style Recommendations

### 5.1 Introduction Structure (Based on Reference Papers)

Recommended introduction flow based on KRISP, Orion, Usher, and MuxServe:

1. **Trend:** GPU spatial partitioning is becoming the norm (MPS, MIG, Green Context) for multi-tenant inference/serving
2. **Problem setup:** Current DL compilers generate kernels assuming exclusive GPU access
3. **Motivating observation 1 — Wave Quantization Trap:** Show a concrete example with numbers (e.g., 108-SM GPU, 30-SM partition, 4 waves with 40% tail waste). Cite Bullet [2025] which names this problem.
4. **Motivating observation 2 — L2 Cache Tragedy:** Show L2 hit rate collapse under co-location. Cite Elvinger et al. [SoCC'25] for quantitative evidence.
5. **Gap statement:** "Existing solutions (KRISP, Orion, Usher) operate at the scheduling level. No compiler or auto-tuner generates kernels that are intrinsically optimized for resource-constrained execution."
6. **Contribution list:** (1) Defensive search space with cache bypass + SM-aware grid sizing; (2) Noise-aware cost model; (3) JIT kernel family generation; (4) End-to-end system evaluation

### 5.2 Method/Design Section

Based on reference papers, recommended description pattern:
- **Top-down:** Start with system architecture overview figure → zoom into each component
- **KRISP style:** Use per-kernel profiling curves (SM count vs. performance) to motivate design decisions
- **Orion style:** Use a clear classification matrix (compute-bound vs. memory-bound × shared vs. exclusive) to explain when each optimization applies
- **FlashAttention style:** Use IO complexity analysis to formally justify defensive memory access choices

### 5.3 Evaluation Section

Recommended structure based on top-venue standards:
1. **Setup:** Hardware config, MPS/MIG settings, SM partition sizes tested
2. **Experiment 1 — Absolute Performance:** Throughput under different SM allocations (10%, 25%, 50%, 100%) vs. baselines (Vanilla Triton, Grid-only fix, Ours)
3. **Experiment 2 — Interference Resilience:** P99 latency under 0, 1, 2, 4 concurrent memory stressor processes
4. **Experiment 3 — Kernel-Level Analysis:** Breakdown showing which optimizations (cache bypass, SM-aware tiling, register reuse) contribute to gains
5. **Experiment 4 — Cost Model Accuracy:** Predicted vs. actual performance under noise
6. **Experiment 5 — JIT Overhead:** Compilation time for kernel family generation

---

## 6. Quick Reference — All Papers

| # | Paper | Type | Venue | Year | Read Depth | Key Takeaway |
|---|-------|------|-------|------|-----------|--------------|
| 1 | KRISP | Ref | HPCA | 2023 | Deep | Per-kernel SM right-sizing; closest prior work |
| 2 | Orion | Ref | EuroSys | 2024 | Deep | Interference-aware kernel co-scheduling |
| 3 | Usher | Ref | OSDI | 2024 | Deep | Multi-dimensional interference modeling |
| 4 | MuxServe | Ref | ICML | 2024 | Deep | MPS-based LLM serving; motivates our work |
| 5 | GPU Interference Deep | Ref | SoCC | 2025 | Deep | Quantitative L2 cache interference evidence |
| 6 | GSLICE | RW | SoCC | 2020 | Abstract | Early MPS-based spatial sharing |
| 7 | Gpulet | RW | ATC | 2022 | Abstract | Interference-aware partition sizing |
| 8 | REEF | RW | OSDI | 2022 | Abstract | Microsecond kernel preemption |
| 9 | TVM | RW | OSDI | 2018 | Abstract | Foundational DL compiler |
| 10 | Ansor | RW | OSDI | 2020 | Abstract | State-of-the-art auto-scheduler |
| 11 | BOLT | RW | MLSys | 2022 | Abstract | Auto-tuner improvements |
| 12 | FlashAttention | RW | NeurIPS | 2022 | Abstract | IO-aware kernel design paradigm |
| 13 | FlashAttention-2 | RW | ICLR | 2024 | Abstract | L2 uncontrollability noted |
| 14 | FlashAttention-3 | RW | NeurIPS | 2024 | Abstract | GPU asynchrony exploitation |
| 15 | Triton | RW | MAPL | 2019 | Abstract | Tiled GPU programming language |
| 16 | ParvaGPU | RW | arxiv | 2024 | Abstract | MIG+MPS combination |
| 17 | SGDRC/Missile | RW | arxiv | 2024 | Abstract | Software-defined GPU resource control |
| 18 | Hierarchical RP | RW | arxiv | 2024 | Abstract | RL-based partition optimization |
| 19 | Bubbleless S-T | RW | EuroSys | 2025 | Abstract | Adaptive spatial-temporal sharing |
| 20 | StreamBox | RW | ATC | 2024 | Abstract | Lightweight GPU sandbox |
| 21 | Bullet | RW | arxiv | 2025 | Abstract | Wave quantization in LLM serving |
| 22 | GPU Multitasking | RW | arxiv | 2025 | Abstract | Vision for GPU multitasking OS |
| 23 | LithOS | RW | SOSP | 2025 | Abstract | GPU OS for ML |
| 24 | TritonForge | RW | arxiv | 2025 | Abstract | Profiling-guided Triton optimization |
| 25 | Triton Attention Anatomy | RW | arxiv | 2025 | Abstract | Cross-platform Triton attention tuning |
| 26 | Two-Stage Tuner | RW | arxiv | 2026 | Abstract | Refactoring + search-based tuning |
| 27 | ConCo | RW | ICS | 2025 | Abstract | Concurrent tensor program compilation |
| 28 | Energy-efficient SM | RW | Workshop | 2025 | Abstract | SM allocation for energy efficiency |

---

## 7. Competitive Positioning Summary

```
                    Scheduling Level          Compiler/Code-Gen Level
                    ─────────────────         ──────────────────────
SM Allocation:      KRISP, Gpulet,            [OUR WORK: SM-aware
                    ParvaGPU, MuxServe         tiling & grid sizing]

Interference:       Orion, Usher              [OUR WORK: Defensive
Avoidance                                      code generation,
                                               cache bypass]

Cost Modeling:      (profiling-based)         [OUR WORK: Noise-aware
                                               stochastic cost model]

Runtime Adapt:      Green Context,            [OUR WORK: JIT kernel
                    Dynamic MPS                family generation]
```

**Our paper fills the right column — the compiler/code-generation level that no existing work addresses.**

---

## 8. Recommended BibTeX Keys

```
@inproceedings{krisp-hpca23,      % KRISP
@inproceedings{orion-eurosys24,   % Orion
@inproceedings{usher-osdi24,      % Usher
@inproceedings{muxserve-icml24,   % MuxServe
@article{gpuinterference-socc25,  % GPU Interference Deep Dive
@inproceedings{gslice-socc20,     % GSLICE
@inproceedings{gpulet-atc22,      % Gpulet
@inproceedings{reef-osdi22,       % REEF
@inproceedings{tvm-osdi18,        % TVM
@inproceedings{ansor-osdi20,      % Ansor
@inproceedings{bolt-mlsys22,      % BOLT
@inproceedings{flashattn-neurips22, % FlashAttention
@inproceedings{triton-mapl19,     % Triton
@article{parvagpu-arxiv24,        % ParvaGPU
@article{sgdrc-arxiv24,           % SGDRC/Missile
@inproceedings{bubbleless-eurosys25, % Bubbleless
@inproceedings{streambox-atc24,   % StreamBox
@article{bullet-arxiv25,          % Bullet
@inproceedings{lithos-sosp25,     % LithOS
```
