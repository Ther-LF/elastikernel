# ElastiKernel: Resource-Adaptive Kernel Auto-Tuning for Spatially Shared GPUs

> Design Document — 2026-03-17
> Implementation base: OpenAI Triton
> Hardware: Multi-GPU (A100/H100 + consumer GPUs)
> Strategy: Phase A (Autotuner layer) → Phase B (IR Compiler Pass)

---

## 1. Problem Statement

Current DL compilers (Triton, TVM, XLA) generate kernels assuming exclusive GPU access.
When these kernels run under MPS/MIG/Green Context with reduced SMs and shared L2 cache,
performance degrades **non-linearly** due to a 7-layer interference stack:

### 1.1 Complete Interference Taxonomy (7 Layers)

**Layer 1 — Wave Quantization Trap (SM-level, inter-wave)**
- Grid size optimized for 108 SMs → 4 waves on 30 SMs, last wave only 18 blocks → 40% SM idle
- Evidence: Bullet (arxiv'25) explicitly names this as root cause of prefill inefficiency
- **Severity: HIGH** — affects ALL compute-bound kernels; non-linear performance cliff at SM boundaries

**Layer 2 — L2 Cache Interference (chip-level, inter-tenant) — Three Sub-dimensions**
- **2a. Capacity Pollution:**
  Co-tenant processes evict your L2 cache lines → hit rate 80%→10%.
  Evidence: 2.15x slowdown when combined working set > L2 capacity (Elvinger et al., SoCC'25).
  Defense: .cs cache bypass can mitigate this (reduces L2 footprint, protects high-reuse tensor cache lines).
- **2b. Crossbar Contention:**
  L2 cache consists of 40+ slices; all SMs route to slices through an N×M crossbar interconnect.
  Even when data hits L2, co-tenant L2 requests can saturate the crossbar routing bandwidth,
  increasing L2 hit latency from ~30 cycles to ~50-80 cycles.
  Evidence: L2 bandwidth utilization rises from 37% to 95% (SoCC'25, co-located copy kernel).
  Defense: .cs bypass does NOT solve this (requests still traverse the crossbar). The only solution
  is to reduce total L2 request count (via register/SMEM on-chip reuse) and use 128-bit vectorized
  loads (reducing crossbar transaction count).
- **2c = Layer 3. DRAM Bandwidth Saturation:**
  L2 miss requests compete with co-tenant for physical DRAM bandwidth; see Layer 3.
- **Severity: HIGH** — all three sub-dimensions act simultaneously with compounding effects.

**Layer 3 — DRAM Bandwidth Saturation (chip-level, inter-tenant)**
- DRAM (HBM) bandwidth is a physically shared resource that **cannot be isolated by any software mechanism** (neither MPS nor Green Context).
  All L2 miss requests from all SMs share the same set of memory controllers and HBM channels.
- Evidence: 1.3x slowdown from memory BW contention even with **complete SM isolation** via
  Green Context (SoCC'25). Green Context isolates SMs but L2 cache and DRAM BW remain shared.
- **Severity: MEDIUM-HIGH** — affects ALL memory-bound kernels (Softmax, LayerNorm, LLM decode).
- **Key insight:** This is INDEPENDENT from L2 cache pollution — solving L2 pollution (Layer 2a)
  does NOT solve BW contention. A kernel that bypasses L2 (using .cs hint) actually puts
  MORE pressure on DRAM bandwidth because all requests go directly to HBM.
  There is a **fundamental tension**: cache bypass protects from L2 pollution but increases BW demand.
- **Defense hierarchy:**
  (1) Root solution: reduce total off-chip memory access volume (register/SMEM on-chip reuse) → §4.3, §4.4
  (2) Secondary: smart cache policy (keep high-reuse tensors in L2, bypass low-reuse tensors) → §4.2
  (3) Supplementary: when BW is already tight, accept L2 pollution rather than bypassing entirely (BW-aware decision) → §4.2

**Layer 4 — Occupancy Cliff / Register Pressure Inversion (SM-level, intra-SM)**
- Kernels tuned for full GPU maximize registers per thread (for best per-thread ILP).
  Full GPU: few blocks per SM → high register budget per block → fine.
  Reduced SMs: need MORE blocks per SM to maintain throughput → register file becomes bottleneck.
- Evidence: Block scheduler HoL blocking — 2.2x slowdown when decode kernel needs 64,512 registers
  but only 63,288 remain after co-located kernel (SoCC'25, Section 4.2)
- **Severity: HIGH** — creates a cliff effect: performance is fine until register budget is exceeded,
  then suddenly blocks cannot launch at all.
- **Key insight:** Traditional auto-tuners MAXIMIZE register usage per thread because it's "free"
  when you have the whole GPU. Under spatial sharing, this strategy backfires catastrophically.
  The compiler must learn to TRADE register usage for occupancy flexibility.

**Layer 5 — Warp Scheduler IPC Ceiling (SM-level, intra-SM)**
- Each SM's warp scheduler can issue ~4 instructions per cycle (architectural limit).
  Two sub-scenarios under spatial sharing:
- **5a. Cross-tenant intra-SM (MPS without Green Context):**
  MPS does not enforce SM mutual exclusion, so two tenants' kernels can share an SM.
  Combined IPC from both tenants' warps can exceed the ~4 instr/cycle ceiling.
  Evidence: At combined IPC 3.45 (near limit of 4): 1.92x TBT slowdown (SoCC'25, Section 4.4).
  NOTE: Does NOT apply under MIG or Green Context (separate SMs).
- **5b. Single-tenant higher occupancy (all isolation modes):**
  Even with SM isolation, fewer SMs → more blocks per SM from your own kernel → more warps
  competing for the warp scheduler. For compute-bound kernels (GEMM, MoE expert phase),
  the combined IPC from your own warps can approach the ceiling.
- **Severity: MEDIUM** — 5a is severe but MPS-specific; 5b is the universal concern.
- **Key insight:** This means we can't just blindly increase occupancy (more warps per SM) to
  compensate for fewer SMs — there's a ceiling. The compiler needs to find the sweet spot
  between occupancy (enough warps to hide latency) and IPC saturation (not too many warps).

**Layer 6 — Shared Memory Bank Conflict Amplification (SM-level, intra-SM)**
- Shared memory has 32 banks. Bank conflicts arise from two distinct sub-scenarios under spatial sharing:
- **6a. Cross-tenant intra-SM (MPS without Green Context):**
  MPS does NOT enforce mutual exclusion of SMs — it only caps the maximum SM count per client
  (SoCC'25, footnote 1: "MPS does not enforce mutual exclusion of SMs between MPS clients").
  This means two tenants' kernels CAN run on the same SM simultaneously under MPS.
  When a co-tenant kernel has non-optimal strided SMEM access patterns, it saturates the
  SM's shared memory pipeline, starving your kernel's SMEM accesses.
  Evidence: 32-way bank conflict from co-tenant → 3.75x slowdown for GEMM dim=1024 (SoCC'25, Section 4.4.1).
  NOTE: This sub-scenario does NOT apply under MIG or Green Context, which enforce true SM isolation.
- **6b. Single-tenant higher occupancy (all isolation modes):**
  Even with perfect SM isolation (Green Context/MIG), fewer SMs means each SM must run more
  thread blocks from YOUR OWN kernel to maintain throughput. More blocks per SM → more warps
  concurrently accessing SMEM → higher intra-kernel bank conflict probability.
  This is amplified when we increase SMEM usage to reduce L2 dependence (Layer 2 defense).
- **Severity: MEDIUM** — 6a is severe but only under MPS; 6b is moderate but universal.
  Both are amplified when we increase shared memory usage to avoid L2 (Layer 2 mitigation).
- **Key insight:** There is a second tension: increasing shared memory usage to avoid L2 pollution
  (Layer 2 defense) can increase bank conflicts (Layer 6). The compiler must balance this tradeoff.
  For 6a, the defense is XOR swizzle layout; for 6b, the defense is co-optimizing tile size
  and SMEM budget with the occupancy target from §4.3.

**Layer 7 — Static Cost Model Mismatch (system-level, cross-cutting)**
- Auto-tuners profile on clean, unshared GPU, choosing configs brittle to all above interferences.
- The "optimal" config under isolation is often the WORST config under sharing because it:
  (a) maximizes grid size → worst wave quantization
  (b) relies on L2 for data reuse → worst cache pollution sensitivity
  (c) maximizes register usage → worst occupancy cliff vulnerability
  (d) maximizes shared memory → worst bank conflict exposure
- **Severity: HIGH** — this is a meta-problem that amplifies all other layers.

### 1.2 Tension Graph — Why This Is Hard

The 7 layers create a web of conflicting optimization objectives:

```
  Layer 2 defense                Layer 6 defense
  "bypass L2, use SMEM"  ←─ CONFLICTS ─→  "reduce SMEM usage
   to avoid pollution"                      to avoid bank conflicts"
        │                                          │
        ▼                                          ▼
  Layer 3 worsened               Layer 4 defense
  "bypassing L2 puts             "reduce registers per thread
   MORE pressure on DRAM BW"      for occupancy headroom"
        │                                │
        ▼                                ▼
  Layer 1 defense                Layer 5 constraint
  "align grid to SM count"       "but don't over-increase
                                  occupancy → IPC ceiling"
```

This tension graph is what makes our problem a genuine RESEARCH problem, not just engineering.
No single heuristic can resolve all tensions simultaneously — we need a multi-objective
search that finds Pareto-optimal configurations across the interference dimensions.

### 1.3 The Defense Hierarchy — Root Cause Analysis

**Key insight: the L2 cache problem has THREE sub-dimensions, not one:**

```
Three Independent Contention Dimensions of L2 Cache:
┌─────────────────────────────────────────────────────────────┐
│ 2a. L2 Capacity Pollution                                   │
│     Co-tenant evicts your cache lines → hit rate drops      │
│     Measurable: L2 hit rate 80% → 10%                       │
│     Upper bound: 2.15x slowdown (SoCC'25)                   │
├─────────────────────────────────────────────────────────────┤
│ 2b. L2 Crossbar Contention                                  │
│     Even when data hits L2, crossbar saturated by co-tenant │
│     → L2 hit latency rises from ~30 cycles to ~50-80 cycles │
│     Measurable: L2 BW utilization 37% → 95% (SoCC'25)      │
│     NOTE: .cs bypass does NOT solve this — requests still   │
│           traverse the crossbar                              │
├─────────────────────────────────────────────────────────────┤
│ 2c = Layer 3. DRAM Bandwidth Saturation (HBM)               │
│     L2 miss requests compete with co-tenant for physical BW │
│     1.3x slowdown even with SM isolation (SoCC'25)          │
│     NOTE: .cs bypass WORSENS this — all requests go to DRAM │
└─────────────────────────────────────────────────────────────┘
```

**This means cache bypass (.cs) is a suboptimal palliative, not a root cure:**

```
Effect of .cs cache bypass:
  ✓ Solves 2a (Capacity Pollution) — no L2 footprint, protects other tensors' cache lines
  ✗ Does NOT solve 2b (Crossbar Contention) — requests still route through L2 crossbar
  ✗ Worsens 2c/Layer 3 (DRAM BW) — all requests go directly to DRAM bandwidth pipeline

Measured impact:
  Exclusive GPU:  L2 bypass may break even or slightly degrade (loses L2 caching benefit)
  Shared GPU:     L2 bypass HURTS high-reuse tensors (forces repeated DRAM loads)
                  L2 bypass HELPS low-reuse tensors (reduces L2 pollution)
```

**The true root defense is reducing total off-chip memory access volume, not choosing which path to take:**

```
Defense Strategy Priority (from root cause to surface):

Priority 1 (Root): Maximize on-chip data reuse
  ├── 4.3 Register Budgeting — keep intermediate results in registers
  │     → completely avoids L2 or DRAM access
  │     → but more registers reduce occupancy (Layer 4)
  │     → Triton already has maxnreg parameter for control
  │
  ├── 4.4 SMEM-enhanced reuse — explicitly stage intermediates in shared memory
  │     → avoids L2 or DRAM access
  │     → but more SMEM increases bank conflicts (Layer 6)
  │     → requires XOR swizzle to mitigate bank conflicts
  │
  └── Kernel Fusion — eliminate inter-kernel intermediate data round-trips
        → eliminates entire kernel's off-chip access
        → this is the core idea behind FlashAttention
        → we don't do fusion, but adopt its "on-chip first" philosophy

Priority 2 (Secondary): Smart off-chip access strategy (when off-chip access is unavoidable)
  ├── 4.2a Selective Cache Bypass — use .cs only for low-reuse tensors
  │     → protects high-reuse tensor L2 cache lines
  │     → reduces L2 capacity pollution from low-reuse tensors
  │     → Triton already has eviction_policy="evict_first" API
  │
  ├── 4.2b BW-Aware Tensor Classification — decide based on remaining BW budget
  │     → if DRAM BW is already tight, accept some L2 pollution instead
  │     → if DRAM BW has headroom, bypass L2 to reduce pollution
  │
  └── 4.2c Memory access coalescing & vectorization — reduce L2 crossbar request count
        → 128-bit vectorized load issues 3 fewer crossbar requests than 4x 32-bit loads
        → Triton compiler already does this partially, but may regress under certain configs

Priority 3 (Supplementary): Tolerate off-chip latency
  └── 4.1 SM-Aware Grid Sizing — ensure SM-cycles are not wasted on idle waiting
        → does not reduce off-chip access, but ensures SMs are not idle
```

**Significance for paper writing:**
> This three-level defense hierarchy (on-chip reuse > smart off-chip policy > latency tolerance)
> is itself a compelling paper framework. It explains why naive cache bypass is insufficient,
> and why the compiler must co-search across register/SMEM/cache-policy dimensions simultaneously.

### 1.4 Triton API Feasibility Verification

| Solution | Triton API Available? | Difficulty | Verification Source |
|----------|----------------------|-----------|-------------------|
| SM-Aware Grid Sizing | `grid=lambda meta: (...)` custom grid function | Low | Triton tutorial |
| Cache Bypass (.cs) | `tl.load(..., eviction_policy="evict_first")` | Low | triton-lang.org official docs |
| Cache Bypass (.cs) for store | `tl.store(..., cache_modifier=".cs")` | Medium (bug #1728) | GitHub issue |
| Register Budget | `triton.Config(maxnreg=N)` | **Zero** (already available) | triton.Config docs |
| SMEM Swizzle | Requires manual XOR index transform in kernel | Medium | CUTLASS CuTe, Lei Mao's blog |
| Noise-Aware Profiling | Requires custom autotuner; not natively supported | Medium-High | Needs triton.autotune extension |
| Kernel Family | Multiple autotune calls with cached results | Low | Standard Python wrapper |

### 1.3 Evidence Summary

| Layer | Interference Type | Evidence Source | Max Observed Slowdown |
|-------|------------------|----------------|----------------------|
| 1 | Wave Quantization | Bullet'25, SoCC'25 §4.2 | 2.2x (block scheduler HoL) |
| 2 | L2 Cache Pollution | SoCC'25 §4.3, Fig 4 | 2.15x |
| 3 | Memory BW Saturation | SoCC'25 §4.3, Table 1 | 1.3x (even with SM isolation) |
| 4 | Occupancy Cliff | SoCC'25 §4.2 | 2.2x (register budget exceeded) |
| 5 | Warp Scheduler IPC | SoCC'25 §4.4, Table 2 | 1.92x |
| 6 | SMEM Bank Conflict | SoCC'25 §4.4.1, Fig 5 | 3.75x |
| 7 | Static Cost Model | All papers (implicit) | Amplifies layers 1-6 |

**Combined worst-case:** These effects compound multiplicatively, not additively.
A kernel suffering wave quantization (2x) + L2 pollution (2x) + occupancy cliff (2x) can
see 4-8x total slowdown under spatial sharing.

## 2. System Architecture

```
User Triton Kernel (@triton.jit)
        │
        ▼
┌───────────────────────────────────────────┐
│         Resource Contract (Constraints)    │
│  {sm_count, noise_level, l2_share}         │
│  Sources: MPS config / Green Context /     │
│           runtime probe / user-specified    │
└─────────────────┬─────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌─────────┐ ┌──────────┐ ┌───────────┐
│SM-Aware │ │Defensive │ │Noise-Aware│
│Grid     │ │Cache     │ │Cost       │
│Sizer    │ │Policy    │ │Model      │
└────┬────┘ └────┬─────┘ └─────┬─────┘
     │           │             │
     ▼           ▼             ▼
┌───────────────────────────────────────────┐
│      Extended Search Space                 │
│  Standard: BLOCK_M/N/K, num_warps, stages  │
│  New: grid_strategy, cache_policy,         │
│       register_mode                        │
└─────────────────┬─────────────────────────┘
                  │
     ┌────────────┼────────────┐
     ▼            │            ▼
┌──────────┐     │     ┌────────────┐
│Phase A:  │     │     │Phase B:    │
│Autotuner │     │     │IR Compiler │
│Layer     │     │     │Pass        │
└────┬─────┘     │     └─────┬──────┘
     │           │           │
     ▼           ▼           ▼
┌───────────────────────────────────────────┐
│         Kernel Family + JIT Dispatcher     │
│  {sm=16: PTX_v1, sm=24: PTX_v2, ...}      │
└───────────────────────────────────────────┘
```

## 3. Target Kernels

Six kernels selected to cover the compute-memory spectrum and real-world importance:

| Kernel | Type | Why It Matters Under Spatial Sharing |
|--------|------|--------------------------------------|
| **GEMM** | Compute-bound | Grid size = f(M,N,K,BLOCK) typically tuned for full SM count. Wave quantization is the dominant effect. Classic auto-tuning target. |
| **Softmax** | Memory-bound | Near-zero compute reuse. Pure bandwidth kernel — directly measures L2/DRAM contention impact. Good for isolating cache interference. |
| **LayerNorm** | Memory-bound | Similar to Softmax but with reduction. Common in every Transformer layer. |
| **RoPE** | Memory-bound | Rotary positional encoding. Element-wise + read-heavy. Common in modern LLMs (LLaMA, etc.). |
| **FlashAttention** | Mixed (compute+memory) | Tiling strategy is hand-crafted for full SRAM budget. Under SM reduction, occupancy and shared memory per SM change, disrupting the carefully balanced tile pipeline. The most complex kernel to adapt. |
| **Fused MoE** | Mixed (compute+memory), irregular | **Critical addition.** MoE routing + expert GEMM in one kernel. Unique challenges under spatial sharing: (1) Grid size = num_tokens × top_k_experts, highly variable and rarely SM-aligned. (2) Token-to-expert assignment is dynamic and load-imbalanced — some experts get many tokens, others few — causing severe wave quantization when SMs are limited. (3) Multiple expert weights compete for L2 cache simultaneously. (4) Current implementations (vLLM fused_moe, Megablocks) assume full GPU for expert parallelism. (5) Represents the MoE architecture trend (Mixtral, DeepSeek-V3, Qwen-MoE). |

### 3.1 Fused MoE Kernel — Detailed Analysis

The Fused MoE kernel is particularly interesting because it exhibits ALL SEVEN
interference layers simultaneously:

**Layer 1 — Wave Quantization (most severe):**
```
Example: Mixtral 8x7B, batch=32 tokens, top_k=2
→ 64 token-expert pairs
→ With BLOCK_M=16: grid_size = 4 per expert × 8 experts = up to 32 blocks
→ On 30 SMs: 1 full wave + 2 stragglers → 93% SM idle in last wave
→ On 16 SMs: 2 full waves → perfect (lucky case)
→ On 24 SMs: 1 full wave + 8 stragglers → 67% SM idle

The problem: grid_size varies PER BATCH based on routing decisions.
Static tuning cannot anticipate this.
```

**Layer 2+3 — L2 Cache Pressure + BW Saturation (compounded):**
```
8 experts × weight_size_per_expert ≈ 8 × (hidden_dim × ffn_dim × 2bytes)
For Mixtral: 8 × (4096 × 14336 × 2) ≈ 878 MB of weights
Only a few experts are "hot" per batch, but under MoE load balancing,
all experts eventually get activated → entire weight set cycles through L2.
In multi-tenant MPS, co-located tasks evict these expert weights aggressively.
Bypassing L2 for cold experts helps Layer 2 but worsens Layer 3.
```

**Layer 4 — Occupancy Cliff (critical for MoE):**
```
Fused MoE kernels typically use high register counts because each thread
handles routing logic + GEMM accumulation. Under reduced SMs, the register
budget prevents enough concurrent blocks → throughput cliff.
Current vLLM fused_moe uses ~128 registers/thread → only 2 blocks/SM on A100.
At 30 SMs, this means 60 concurrent blocks max, often insufficient.
```

**Layer 5 — IPC Ceiling (during expert GEMM phase):**
```
The GEMM-heavy phase of each expert is compute-intensive.
When multiple expert blocks share an SM (forced by fewer total SMs),
combined IPC can approach the warp scheduler limit.
```

**Layer 6 — SMEM Bank Conflicts (during routing + gather phase):**
```
Token gathering for each expert uses shared memory to buffer selected tokens.
With more blocks per SM, concurrent SMEM accesses increase → bank conflicts.
```

**Dynamic Load Imbalance (unique to MoE, cross-cutting all layers):**
```
Expert popularity follows power-law distribution.
Expert 0 might get 40% of tokens, Expert 7 gets 2%.
When total SMs are few, the "hot expert" blocks dominate execution time
while "cold expert" blocks finish quickly → SM idle time from imbalance.
SM-aware tiling must account for per-expert token count distribution.
```

## 4. Technical Innovations — Mapping Solutions to 7-Layer Interference Stack

### 4.0 Solution-to-Problem Mapping

| Innovation | Addresses Layers | Core Mechanism |
|-----------|-----------------|----------------|
| **4.1 SM-Aware Grid Sizing** | Layer 1 (Wave Quantization) | Grid % SM ≈ 0 |
| **4.2 Adaptive Cache Policy** | Layer 2 (L2 Pollution) + Layer 3 (BW) | .cs hint + BW-aware tensor classification |
| **4.3 Occupancy-Aware Register Budgeting** | Layer 4 (Occupancy Cliff) + Layer 5 (IPC) | Register cap + occupancy floor constraint |
| **4.4 SMEM Layout Optimization** | Layer 6 (Bank Conflicts) | Padding + swizzle under higher SMEM pressure |
| **4.5 Multi-Objective Noise-Aware Cost Model** | Layer 7 (Static Mismatch) + ALL | Pareto search across interference dimensions |
| **4.6 Kernel Family + JIT Dispatch** | Runtime adaptation for all layers | Pre-compiled variants per SM level |

### 4.1 Innovation 1: SM-Aware Grid Sizing (→ Layer 1)

Extend Triton's autotuner config space — make grid size a function of `sm_allocated`:

**Dimension A — SM-Aware Grid Strategy:**
```python
def sm_aware_grid(M, N, BLOCK_M, BLOCK_N, sm_count):
    raw_grid = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)
    # Option 1: Wave-aligned — pad grid to fill all waves
    waves = ceil(raw_grid / sm_count)
    aligned_grid = waves * sm_count

    # Option 2: Wave-minimal — shrink grid to eliminate partial wave
    minimal_grid = floor(raw_grid / sm_count) * sm_count

    # Option 3: Adaptive — choose based on waste ratio
    waste_aligned = (aligned_grid - raw_grid) / aligned_grid
    waste_minimal = (raw_grid - minimal_grid) / raw_grid
    if waste_aligned < waste_minimal:
        return aligned_grid  # pad (slightly more work, but balanced)
    else:
        return minimal_grid  # shrink (slightly less work, but zero waste)
```

For Fused MoE: grid strategy must account for per-expert token counts.
Pre-sort experts by token count, then pack blocks to minimize wave waste.

### 4.2 Innovation 2: Adaptive Cache Policy (→ Layer 2a + 2b + Layer 3)

**Important nuance: L2 cache has THREE interference dimensions (see §1.3).**
Cache bypass (.cs) only solves 2a (capacity pollution), does NOT solve 2b (crossbar
contention), and WORSENS Layer 3 (DRAM BW saturation). Therefore, cache bypass is
a Priority 2 defense — useful only when off-chip access is unavoidable.

**Key principle: bypass is the LAST resort, not the first.**
Priority 1 is to avoid off-chip access entirely (via register/SMEM reuse — §4.3, §4.4).
Only for data that MUST come from off-chip, we apply selective cache policy.

**Triton API (verified — already available):**
```python
# Load with evict-first policy (= PTX .cs cache streaming)
data = tl.load(ptr + offsets, eviction_policy="evict_first")

# Load with default policy (= PTX .ca cache all)
data = tl.load(ptr + offsets)  # or eviction_policy=""

# Store with cache streaming
tl.store(ptr + offsets, data, cache_modifier=".cs")
# NOTE: store eviction policy has known bug (triton issue #1728)
# May need PTX inline asm fallback on some Triton versions
```

**Tensor-level cache policy classification (BW-aware):**

```python
def classify_tensor_cache_policy(tensor_info, sm_count, total_sms,
                                  co_tenant_bw_estimate):
    """
    Decide cache policy per tensor based on reuse pattern AND bandwidth budget.

    Returns: "evict_first" (bypass L2) or "" (normal caching)
    """
    reuse_count = tensor_info.reuse_count   # static analysis: how many times accessed
    tensor_bytes = tensor_info.size_bytes
    effective_l2 = (sm_count / total_sms) * L2_TOTAL_BYTES  # our approximate L2 share

    # Estimate available DRAM BW (shared resource, co-tenant takes a portion)
    available_bw = PEAK_DRAM_BW * (1.0 - co_tenant_bw_estimate)

    # Case 1: Read-once data (bias, routing scores, masks)
    # → ALWAYS bypass — no reuse benefit from caching, only pollutes
    if reuse_count <= 1:
        return "evict_first"

    # Case 2: Data larger than our L2 share
    # → Caching it would evict everything else (thrashing)
    # → But check BW budget first: can DRAM handle the extra load?
    if tensor_bytes > effective_l2 * 0.5:
        extra_bw_needed = tensor_bytes * reuse_count / kernel_time_estimate
        if extra_bw_needed < available_bw * 0.3:  # <30% of remaining BW
            return "evict_first"  # BW can handle it, bypass to protect L2
        else:
            return ""  # BW already tight, accept L2 pollution as lesser evil

    # Case 3: Small tensor with high reuse
    # → Worth protecting in L2, keep normal caching
    # → This is the GEMM inner loop tile, attention KV cache, etc.
    return ""

# Example classification for GEMM:
#   A matrix tiles (reuse=K/BLOCK_K times) → "" (keep in L2, high reuse)
#   B matrix tiles (reuse=M/BLOCK_M times) → "" (keep in L2, high reuse)
#   Bias vector (reuse=1)                  → "evict_first" (read-once)
#   Output C tiles (write-once)            → ".cs" store (write-once)

# Example classification for Fused MoE:
#   Expert weights (reuse depends on token count per expert):
#     Hot expert (40% tokens) → "" (high reuse, worth caching)
#     Cold expert (2% tokens) → "evict_first" (low reuse, don't pollute)
#   Routing scores (reuse=1) → "evict_first"
#   Token gather indices      → "evict_first"
```

**The Layer 2a ↔ Layer 3 tradeoff is resolved by BW-aware classification:**
> "Naive L2 bypass (.cs for all loads) trades cache pollution for bandwidth saturation.
> Our adaptive policy classifies tensors by reuse count AND available bandwidth budget:
> high-reuse small tensors stay in L2 (worth protecting), low-reuse or oversized tensors
> bypass L2 (not worth the cache space), BUT only if DRAM bandwidth can absorb the extra
> load. When bandwidth is already saturated, we accept L2 pollution as the lesser evil."

**What .cs does NOT solve (Layer 2b — crossbar contention):**
> Even with .cs, load requests still traverse the L2 crossbar for routing.
> The only way to reduce crossbar pressure is to reduce TOTAL off-chip request count,
> which brings us back to Priority 1: maximize on-chip reuse (§4.3, §4.4).
> Additionally, 128-bit vectorized loads (which Triton already generates in many cases)
> reduce crossbar transactions by 4x compared to scalar loads — we should verify
> that our configs don't regress vectorization.

### 4.3 Innovation 3: Occupancy-Aware Register Budgeting (→ Layer 4 + Layer 5)

**This is a Priority 1 defense — reducing off-chip access at the root.**

Problem: Triton/CUDA compilers maximize registers per thread for ILP (instruction-level
parallelism). This is optimal on a full GPU where each SM runs few concurrent blocks.
Under spatial sharing, fewer SMs must process the same total workload → each SM needs
MORE concurrent blocks → register file becomes the bottleneck.

```
Full GPU (108 SMs):           Shared GPU (30 SMs):
  Per SM: 2 blocks             Per SM: need ~7 blocks to maintain throughput
  Registers/block: 64,512     Registers available: 65,536 total per SM
  Total: 129,024 → fits ✓     7 × 64,512 = 451,584 → DOES NOT FIT ✗
                                → Only 1 block per SM → severe underutilization
```

**Triton API (verified — zero development cost):**
```python
# triton.Config already has maxnreg parameter!
# From triton.Config documentation:
# class triton.Config(kwargs, num_warps=4, num_stages=3, num_ctas=1,
#                     maxnreg=None, ...)
#
# maxnreg: maximum number of registers per thread.
# When set, compiler will try to limit register usage.
# If registers exceed this, compiler spills to local memory (slow)
# or shared memory (faster, see RegDem technique).

# Example: adding maxnreg to autotuner search space
@triton.autotune(
    configs=[
        # Full GPU config: generous registers, low occupancy OK
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},
                      num_warps=8, num_stages=3, maxnreg=None),  # no limit

        # 50% SM config: moderate registers, higher occupancy needed
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=4, num_stages=2, maxnreg=128),

        # 25% SM config: frugal registers, max occupancy
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=4, num_stages=2, maxnreg=64),
    ],
    key=['M', 'N', 'K'],
)
```

**Register-Occupancy sweet spot calculation:**

```python
def compute_register_budget(sm_count, total_sms, kernel_type):
    """
    Compute max registers per thread to ensure sufficient occupancy
    on a reduced SM partition.

    Key tradeoff: fewer registers → more blocks per SM (good for throughput)
                  but also → more register spills (bad for per-thread perf)
                  AND → more warps → potential IPC ceiling (Layer 5)
    """
    MAX_REGS_PER_SM = 65536     # A100/H100
    MAX_THREADS_PER_SM = 2048   # A100/H100
    WARP_SIZE = 32

    # Target: how many blocks per SM do we need?
    # On full GPU: typically 2-4 blocks/SM is enough
    # On reduced SM: need more blocks/SM to compensate
    scale = total_sms / sm_count   # e.g., 108/30 = 3.6x

    # But cap the scale — beyond ~4 blocks/SM, diminishing returns
    # and IPC ceiling (Layer 5) becomes a concern
    if kernel_type == 'compute_bound':
        # Compute-bound kernels hit IPC ceiling sooner
        # IPC limit ~4 instr/cycle, each warp issues ~1-2 instr/cycle
        # So ~2-4 active warps per SM is the practical ceiling
        target_blocks = min(ceil(2 * scale), 4)  # cap at 4
    elif kernel_type == 'memory_bound':
        # Memory-bound kernels benefit more from occupancy
        # (more warps to hide memory latency)
        # IPC is not the bottleneck
        target_blocks = min(ceil(2 * scale), 8)  # cap at 8
    else:  # mixed
        target_blocks = min(ceil(2 * scale), 6)

    threads_per_block = 256   # typical Triton block size (8 warps × 32)
    max_regs = MAX_REGS_PER_SM // (target_blocks * threads_per_block)

    # Clamp to reasonable range
    max_regs = max(32, min(max_regs, 255))
    return max_regs

# Example outputs:
# compute_register_budget(108, 108, 'compute_bound') → 128 (2 blocks/SM)
# compute_register_budget(30, 108, 'compute_bound')  → 64  (4 blocks/SM)
# compute_register_budget(16, 108, 'memory_bound')   → 32  (8 blocks/SM)
```

**Layer 4 ↔ Layer 5 tradeoff resolution:**
> "Reducing register usage increases occupancy (more blocks per SM), which helps hide
> memory latency. But for compute-bound kernels, too many warps hit the IPC ceiling
> (~4 instructions/cycle/SM). Our register budget is kernel-type-aware: memory-bound
> kernels get aggressive occupancy targets (up to 8 blocks/SM), while compute-bound
> kernels are capped at 4 blocks/SM to avoid IPC saturation."

**Related work: RegDem [Sakdhnagool et al., 2019]**
RegDem proposes binary translation (SASS-level) to spill registers to shared memory.
NVIDIA's official blog also validates this: "eliminating register spilling resulted in
7.76% kernel speedup." Our approach operates at a higher level: we use Triton's `maxnreg`
to tell the compiler our register budget upfront, letting it make better allocation
decisions rather than post-hoc binary patching.

**When maxnreg causes spilling — SMEM as spill target (connects to §4.4):**
If maxnreg forces the compiler to spill, the default spill target is local memory
(slow, goes through L2/DRAM — exactly what we want to avoid). An advanced optimization
is to redirect spills to shared memory (faster, on-chip). This connects to Innovation 4.4:
if we're already using SMEM for bank-conflict-free data staging, we need to budget SMEM
capacity for both explicit data + potential register spills.

### 4.4 Innovation 4: SMEM Layout Optimization Under Sharing (→ Layer 6)

**This is also Priority 1 — shared memory is on-chip, zero L2/DRAM traffic.**

When we increase SMEM usage to reduce L2 dependence (as part of Priority 1 defense),
we risk amplifying bank conflicts (Layer 6) because more concurrent blocks per SM
means more warps accessing SMEM simultaneously.

**Bank conflict basics:**
```
Shared memory = 32 banks × 4 bytes each (128 bytes per bank cycle)
If two threads in the same warp access different addresses in the SAME bank → conflict
Conflict penalty: serialized access (N-way conflict = Nx latency)

Under spatial sharing:
  Full GPU: 2 blocks/SM → 2 sets of warps accessing SMEM
  Reduced SM: 6 blocks/SM → 6 sets of warps → 3x more concurrent SMEM access
  → Higher probability of bank conflicts across blocks
```

**Solution: XOR-based swizzle (industry-proven technique):**

```python
# XOR swizzle in Triton kernel code
# Instead of: shared[row][col]
# Use:        shared[row][col ^ (row % NUM_BANKS)]
#
# This ensures that consecutive rows access different banks,
# even when accessed by different warps from different blocks.

# Triton implementation:
@triton.jit
def swizzled_smem_load(smem_ptr, row, col, SMEM_STRIDE: tl.constexpr):
    # XOR swizzle: col XOR (row & (NUM_BANKS-1))
    # For 32 banks with fp16 (2 bytes per element, 2 elements per bank):
    swizzled_col = col ^ ((row & 31) << 0)  # simplified XOR pattern
    offset = row * SMEM_STRIDE + swizzled_col
    return tl.load(smem_ptr + offset)

# Alternative: padding approach (simpler, wastes SMEM)
# shared[row][col + PAD]  where PAD avoids bank conflicts
# For fp16 on 32 banks: PAD = 8 (adds 16 bytes per row)
# Cost: ~6% SMEM capacity overhead for typical tile sizes
```

**Verified technique — used in production:**
- CUTLASS/CuTe uses `Swizzle<B, M, S>` template for bank-conflict-free SMEM layout
- Hopper TMA hardware natively supports swizzled SMEM writes
- AMD CK-Tile framework uses identical XOR swizzle for LDS bank conflict avoidance
- Lei Mao's blog provides detailed mathematical derivation of optimal swizzle params

**SMEM budget management (register spill + data staging):**
```
Per SM shared memory: 164KB (A100) / 228KB (H100)
Budget must cover:
  (a) Tile data staging for GEMM/attention (primary use)
  (b) Register spills from maxnreg constraint (§4.3 interaction)
  (c) Swizzle padding overhead (~6%)

Example budget for A100 with 4 blocks/SM:
  164KB / 4 = 41KB per block
  - 32KB for tile staging (BLOCK_M × BLOCK_K × 2 bytes × 2 buffers)
  - 4KB for register spill region
  - 5KB padding/swizzle overhead
  Total: 41KB → fits ✓

Example budget for A100 with 8 blocks/SM (aggressive occupancy):
  164KB / 8 = 20.5KB per block
  - Need to reduce tile sizes or use smaller BLOCK_K
  - Less room for register spills → may need higher maxnreg (tension with §4.3)
  This is the SMEM ↔ register tradeoff that the autotuner must navigate
```

**Layer 2 defense ↔ Layer 6 tradeoff resolution:**
> "Increasing shared memory usage to reduce L2 dependence (our Priority 1 defense)
> amplifies bank conflicts under higher per-SM occupancy. We resolve this by applying
> XOR-based swizzle patterns to SMEM layouts — a technique proven in CUTLASS and
> hardware-supported on Hopper — adding ~6% SMEM overhead but eliminating bank
> conflicts. The autotuner co-optimizes tile size, SMEM budget, register budget,
> and swizzle configuration as a joint search problem."

### 4.5 Innovation 5: Multi-Objective Noise-Aware Cost Model (→ Layer 7 + ALL)

The cost model must now optimize across ALL interference dimensions simultaneously,
not just pick the fastest config in isolation.

```python
class MultiObjectiveCostModel:
    """
    Multi-objective cost model that searches for Pareto-optimal configs
    across the 7-layer interference space.
    """

    def evaluate(self, config, sm_count, noise_profile):
        # Dimension 1: Clean performance (for comparison)
        latencies_clean = benchmark(config, sm_count, stressor=None, reps=50)

        # Dimension 2: Performance under L2/BW noise (Layer 2+3)
        latencies_mem_noise = benchmark(config, sm_count,
                                         stressor='memory_flood', reps=50)

        # Dimension 3: Performance under compute noise (Layer 5)
        latencies_compute_noise = benchmark(config, sm_count,
                                             stressor='compute_flood', reps=50)

        # Dimension 4: Performance under combined noise (worst case)
        latencies_combined = benchmark(config, sm_count,
                                        stressor='combined', reps=50)

        return {
            'p50_clean':     np.percentile(latencies_clean, 50),
            'p99_mem_noise': np.percentile(latencies_mem_noise, 99),
            'p99_cmp_noise': np.percentile(latencies_compute_noise, 99),
            'p99_combined':  np.percentile(latencies_combined, 99),
            'max_degradation': max(
                np.median(latencies_mem_noise),
                np.median(latencies_compute_noise),
                np.median(latencies_combined)
            ) / np.median(latencies_clean),
            # Wave efficiency: what fraction of SM-cycles are actually used?
            'wave_efficiency': compute_wave_efficiency(config, sm_count),
            # Occupancy achieved under this register config
            'achieved_occupancy': compute_occupancy(config, sm_count),
        }

    def select_best(self, candidates, objective='robust'):
        if objective == 'robust':
            # Minimize worst-case across all noise types
            return min(candidates, key=lambda c: c['p99_combined'])
        elif objective == 'pareto':
            # Multi-objective Pareto frontier
            return pareto_front(candidates, [
                'p50_clean',        # want low (fast when alone)
                'p99_combined',     # want low (stable under noise)
                'max_degradation',  # want low (graceful degradation)
            ])

    # Analytical pre-filter to prune obviously bad configs before profiling
    def analytical_prefilter(self, config, sm_count, total_sms):
        """Reject configs that will obviously fail, without profiling."""
        # Check 1: Wave quantization waste
        wave_waste = (config.grid_size % sm_count) / sm_count
        if wave_waste > 0.4:  # >40% SM idle in last wave
            return False

        # Check 2: Register budget → occupancy check
        regs_per_thread = estimate_registers(config)
        blocks_per_sm = 65536 // (regs_per_thread * config.threads_per_block)
        if blocks_per_sm < 2:  # Occupancy too low for reduced partition
            return False

        # Check 3: BW feasibility
        estimated_bw_demand = estimate_bandwidth(config)
        available_bw = PEAK_BW * (sm_count / total_sms) * 0.7  # 70% BW pessimistic
        if estimated_bw_demand > available_bw * 1.5:  # >1.5x oversubscribed
            return False

        return True  # Pass to expensive profiling
```

### 4.6 Innovation 6: Kernel Family + Late-Binding JIT Dispatcher (→ Runtime ALL)

Pre-compile kernel variants for discrete SM levels. At runtime, dispatch
the nearest variant in O(1).

```python
class KernelFamily:
    """Pre-compiled kernel variants for different SM allocations."""

    # SM levels chosen at "interesting" boundaries
    SM_LEVELS = [8, 16, 24, 32, 48, 64, 80, 108, 132]

    def __init__(self, kernel_fn, workload_shape):
        self.variants = {}
        for sm in self.SM_LEVELS:
            config = resource_aware_autotune(
                kernel_fn, workload_shape, sm_count=sm)
            self.variants[sm] = compile(kernel_fn, config)

    def dispatch(self, *args, sm_available):
        # Find nearest SM level (round down to be conservative)
        sm_key = max(s for s in self.SM_LEVELS if s <= sm_available)
        return self.variants[sm_key](*args)
```

For Fused MoE: kernel family also varies by num_experts and typical token distribution,
since routing patterns affect optimal grid strategy.

## 5. Implementation Plan

### Phase 1: Motivating Experiments + Autotuner PoC (Weeks 1-3)

#### Week 1: Baseline Performance Collapse Experiments

| Task | Description | Output |
|------|-------------|--------|
| 1.1 | Set up MPS environment on A100/H100 | Working MPS with configurable SM% |
| 1.2 | Implement 6 baseline Triton kernels (GEMM, Softmax, LayerNorm, RoPE, FlashAttention, Fused MoE) | Triton kernels with default autotuning |
| 1.3 | Benchmark each kernel at SM% = {10, 20, 30, 50, 75, 100} using default configs | Performance degradation curves |
| 1.4 | Measure wave quantization effects: grid size vs SM count alignment | Wave waste analysis data |
| 1.5 | Measure L2 interference: run with 0/1/2/4 memory stressor co-tenants | L2 hit rate collapse data |
| 1.6 | Profile Fused MoE specifically: per-expert token distribution → grid imbalance | MoE-specific degradation data |

**Expected output:** Figures for paper Motivation section showing non-linear collapse.

#### Week 2: SM-Aware Grid Sizing + Cache Policy PoC

| Task | Description | Output |
|------|-------------|--------|
| 2.1 | Implement `sm_aware_grid()` function with wave-aligned/minimal strategies | Grid sizing module |
| 2.2 | Implement cache bypass injection via Triton inline PTX (`.cs` hint) | Cache policy module |
| 2.3 | Benchmark GEMM + Softmax with SM-aware grid vs default grid | Grid optimization data |
| 2.4 | Benchmark with cache bypass vs default under co-tenant stressor | Cache defense data |
| 2.5 | Implement MoE-specific grid packing (expert-aware block scheduling) | MoE grid optimizer |

#### Week 3: Extended Autotuner + Noise-Aware Profiling

| Task | Description | Output |
|------|-------------|--------|
| 3.1 | Implement `@resource_aware_autotune` decorator extending Triton's autotuner | Extended autotuner |
| 3.2 | Implement noise-aware profiling (run with background stressor, take P50/P99) | Noise-aware cost model v1 |
| 3.3 | Implement background noise injector (configurable memory stressor process) | Stressor tool |
| 3.4 | Full sweep: 6 kernels × 6 SM levels × {clean, noisy} × {default, ours} | Core experimental data |
| 3.5 | Kernel family prototype: pre-compile 3 variants per kernel, test dispatch | Kernel family PoC |

### Phase 2: IR Compiler Pass + Complete Framework (Weeks 4-7)

#### Week 4-5: Triton IR ResourceAwarePass

| Task | Description | Output |
|------|-------------|--------|
| 4.1 | Study Triton MLIR pipeline internals (TritonGPU IR → LLVM IR → PTX) | Understanding doc |
| 4.2 | Implement ResourceAware grid rewrite pass (modify `tt.launch_grid` based on sm_count) | Grid rewrite pass |
| 4.3 | Implement cache policy injection pass (annotate `tt.load`/`tt.store` with eviction hints) | Cache policy pass |
| 5.1 | Implement register budget constraint pass | Register budget pass |
| 5.2 | Integration testing: all passes working together | Integrated pipeline |

#### Week 5-6: Noise-Aware Cost Model v2

| Task | Description | Output |
|------|-------------|--------|
| 5.3 | Implement analytical bandwidth penalty model (fast pruning) | Analytical model |
| 6.1 | Train lightweight ML model for perf prediction under noise | ML cost model |
| 6.2 | Evaluate cost model accuracy: predicted vs actual under various noise levels | Cost model validation |

#### Week 6-7: Kernel Family + Evaluation

| Task | Description | Output |
|------|-------------|--------|
| 6.3 | Full KernelFamily implementation with O(1) dispatch | Production kernel family |
| 7.1 | Complete evaluation across 6 kernels, multiple GPUs, all baselines | Paper-ready results |
| 7.2 | Fused MoE deep evaluation: vary num_experts, batch sizes, routing patterns | MoE-specific results |
| 7.3 | End-to-end integration test: plug into vLLM MPS serving pipeline | System-level validation |

## 6. Evaluation Plan

### 6.1 Baselines

| Baseline | Description |
|----------|-------------|
| **Vanilla Triton** | Default autotuned for full GPU, run under MPS |
| **Grid-Only Fix** | Only adjust grid size for SM count, no cache/register changes |
| **KRISP-style** | Oracle SM allocation per kernel (upper bound for scheduling-only) |
| **Orion-style** | Classify kernel as compute/mem-bound, schedule complementary pairs (no kernel retuning) |
| **ElastiKernel (Ours)** | Full resource-aware compilation |

### 6.2 Experiments

| # | Experiment | Metrics | Varies |
|---|-----------|---------|--------|
| E1 | Performance collapse baseline | Latency, throughput | SM% = {10,20,30,50,75,100} |
| E2 | Absolute performance | Throughput, speedup over vanilla | SM%, kernel type |
| E3 | Interference resilience | P99 latency, variance | Stressor type × intensity |
| E4 | Ablation study | Latency | Components: grid / cache / register / all |
| E5 | Cost model accuracy | Prediction error, ranking accuracy | Noise level |
| E6 | JIT dispatch overhead | Compilation time, dispatch latency | Num variants |
| E7 | Fused MoE deep dive | Throughput per expert config | num_experts, top_k, batch |
| E8 | End-to-end serving | Request throughput, P99 TTFT/TPOT | Multi-model MPS serving |

### 6.3 Hardware Matrix

| GPU | SMs | L2 Cache | MPS | MIG | Green Context |
|-----|-----|----------|-----|-----|---------------|
| A100 | 108 | 40MB | Yes | Yes | No |
| H100 | 132 | 50MB | Yes | Yes | Yes |
| RTX 4090 | 128 | 72MB | Yes | No | No |

### 6.4 SM Partitioning Setup

**Method 1: MPS with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (all GPUs)**

```bash
# Start MPS daemon
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d

# Launch client A at 50% SMs
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 python run_workload_a.py &

# Launch client B at 50% SMs
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 python run_workload_b.py &
```

> **Critical caveat (from Elvinger et al. SoCC'25):** MPS percentage mode does NOT
> enforce mutual exclusion. Two 50% clients CAN share the same SMs. This only limits
> the max number of thread blocks each client can place. Use for realistic evaluation
> of production deployments where this is the actual mechanism.

**Method 2: MPS Static SM Partitioning (Ampere+, mutually exclusive)**

```bash
nvidia-cuda-mps-control -d -S   # Start with static partitioning
echo "sm_partition add <GPU-UUID> 7" | nvidia-cuda-mps-control  # 7 chunks
# Returns partition ID; assign to client:
CUDA_MPS_SM_PARTITION=<GPU-UUID>/<part-id> python run_workload.py
echo "lspart" | nvidia-cuda-mps-control  # Verify
```

**Method 3: Green Contexts (H100 only, mutually exclusive)**

```python
# Using FlashInfer wrapper (CUDA 13.1+)
from flashinfer.green_ctx import split_device_green_ctx
streams, resources = split_device_green_ctx(torch.device("cuda:0"), 2, 66)
# 2 green contexts × 66 SMs each (remainder gets leftover SMs)
# Launch kernels on separate streams for true SM isolation
```

```c
// Or via CUDA Driver API:
CUdevResource smResource;
cuCtxGetDevResource(NULL, &smResource, CU_DEV_RESOURCE_TYPE_SM);
CUdevResource result[2], remainder;
cuDevSmResourceSplitByCount(&result[0], &minCount, &smResource, &remainder, 0, 64);
// result[0] = 64 SMs, remainder = 68 SMs — mutually exclusive
```

### 6.5 Stressor Design (Following Elvinger et al. SoCC'25 Methodology)

We need controlled stressor kernels to create reproducible interference for E3.
The Elvinger et al. paper provides the gold-standard stressor suite targeting each
shared resource layer independently.

**Stressor 1: Memory Bandwidth (`copy` kernel)**
- Copies a 4GB array using vectorized loads (128-bit)
- Control knob: number of thread blocks (34/68/102/136 on H100)
- More blocks → higher DRAM BW consumption
- L1 effects minimized by setting shared memory to maximum allocation

**Stressor 2: L2 Cache Pollution (`copy` kernel, sized)**
- Same copy kernel but with working set sized to L2 capacity
- Control knob: data size per instance (8MB → 32MB)
- Key finding: at 16MB × 4 instances = 64MB → exceeds H100's 50MB L2 → 2.15× slowdown
- Beyond 26MB per instance, slowdown plateaus at ~1.12×

**Stressor 3: Compute / IPC (`compute` kernel)**
- Independent fp32 multiplications (`__fmul_rn`)
- Control knob: ILP level (S1=1 op, S2=2, S3=3, S4=4 independent ops)
- S1→IPC=1.18, S4→IPC=3.45; degradation occurs when combined IPC→4.0 (architectural limit)
- FP64 variant (`__dmul_rn`) to test pipeline-specific contention

**Stressor 4: Shared Memory Bank Conflicts (`strided_copy` kernel)**
- Loads/stores 4KB array in shared memory with varying stride
- Control knob: stride = {1, 2, 4, 8, 16, 32} → controls bank conflict severity
- 32-way conflict → 1.79× slowdown on co-located torch.mm (dim=2048)

**Stressor 5: Block Scheduler (`sleep` kernel)**
- Calls `__nanosleep()` with ~10ms duration
- 132 thread blocks (1 per SM), 128 threads/block, 16 regs/thread
- Occupies SM slots without doing real work → head-of-line blocking
- Tests how well our register budgeting adapts to reduced available slots

**Experiment E3 matrix:**

| Target Kernel | Stressor | Co-location | Interference Level |
|--------------|----------|-------------|-------------------|
| GEMM | mem_bw | Inter-SM (MPS/GreenCtx) | {25%, 50%, 75%, 100%} BW |
| GEMM | L2_pollution | Inter-SM (MPS/GreenCtx) | {8, 16, 24, 32} MB working set |
| GEMM | compute_ipc | Intra-SM (streams) | {S1, S2, S3, S4} ILP |
| GEMM | smem_bank | Intra-SM (streams) | {1, 8, 16, 32}-way conflicts |
| Fused MoE | mem_bw | Inter-SM (MPS) | {25%, 50%, 75%, 100%} BW |
| Fused MoE | L2_pollution | Inter-SM (MPS) | {8, 16, 24, 32} MB working set |
| FlashAttention | mem_bw | Inter-SM (MPS) | {25%, 50%, 75%, 100%} BW |
| FlashAttention | L2_pollution | Inter-SM (MPS) | {8, 16, 24, 32} MB working set |

### 6.6 End-to-End Serving Setup (E8)

Following the Elvinger et al. and MuxServe methodology:

**Target models:** Gemma3-1B-IT, Llama3.1-8B-Instruct (via vLLM)
**Metric:** P90 Time Between Tokens (TBT) during decode phase
**Setup:** Two model instances co-located via MPS at {30/70, 50/50, 70/30} SM splits

Key insight from MuxServe: prefill is compute-bound, decode is memory-bound.
Co-locating prefill of model A with decode of model B is the sweet spot.
Our system should detect this automatically via the cost model and dispatch
appropriate kernel variants.

### 6.7 Runtime Resource Detection

ElastiKernel's JIT dispatcher needs to detect available resources at launch time.

**SM count detection:**
```python
import torch
# Under MPS, this returns the *restricted* SM count (verified in Elvinger et al.)
sm_count = torch.cuda.get_device_properties(0).multi_processor_count
# Under Green Context, same API returns the partition's SM count
```

```c
// CUDA Driver API — more reliable under MPS
int sm_count;
cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
// Returns reduced count under CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
```

**L2 cache size (hardware total, shared across tenants):**
```python
l2_size = torch.cuda.get_device_properties(0).l2_cache_size  # bytes
# A100: 40MB, H100: 50MB, RTX4090: 72MB
# Under MPS: returns FULL L2 size (L2 is NOT partitioned)
# Must assume effective L2 = total / num_tenants for cache policy decisions
```

**Detecting co-tenant count (heuristic):**
No direct API exists. Options:
1. Environment variable: `ELASTIKERNEL_NUM_TENANTS=2`
2. Infer from SM ratio: if `sm_count / total_sms < 1.0`, at least 2 tenants
3. Runtime probe: measure L2 hit rate with known working set, compare to solo baseline

## 7. Key Design Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Implementation language | Python + Triton DSL | Lowest friction; Triton's autotuner is Python-native |
| Phase A before Phase B | Autotuner first, then IR pass | Get motivating data fast; IR pass adds generality later |
| Fused MoE as target kernel | Yes, included | MoE is trending (DeepSeek-V3, Mixtral); uniquely exhibits all three problems; reviewer appeal |
| Noise injection method | Real memory stressor process under MPS | More realistic than analytical noise model alone |
| Cost model approach | Empirical (noise-aware profiling) + analytical (BW penalty) | Empirical for accuracy, analytical for search space pruning |
| Kernel family granularity | 9 SM levels | Covers common partition sizes; <5min compilation per kernel |
| Grid strategy for MoE | Expert-aware block packing | Standard wave alignment insufficient for MoE's irregular grid |

## 8. Risk Analysis

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Triton's `.cs` cache hint has no effect on some architectures | Medium | Fall back to register reuse strategy; test on multiple GPU generations |
| MPS SM partitioning is imprecise (only upper bound, not guarantee) | Medium | Use Green Context on H100 for precise experiments; note MPS limitation in paper |
| Fused MoE's dynamic routing makes offline tuning unreliable | Medium | Use kernel family with routing-pattern-aware dispatch |
| Search space explosion with 3 new dimensions | Low | Analytical model prunes 90%+ configs before profiling |
| Reviewer concern: "just engineering, not research" | Medium | Formalize noise-aware cost model; provide theoretical wave-waste analysis |
