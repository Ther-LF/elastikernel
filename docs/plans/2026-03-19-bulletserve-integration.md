# ElastiKernel + BulletServe Integration Design

> Date: 2026-03-19
> Goal: End-to-end SM-aware kernel tuning benchmark via BulletServe

---

## 1. Integration Positioning

ElastiKernel operates at the **kernel compilation layer**, BulletServe at the **scheduling layer**. They are orthogonal and complementary:

| Layer | Owner | Responsibility |
|-------|-------|----------------|
| Hardware SM constraint | libsmctrl (BulletServe) | TPC bitmask on CUDA stream |
| Kernel config selection | ElastiKernel (Triton autotune) | Optimal tile size for given SM count |
| Scheduling / SLO | BulletServe scheduler | When and how many SMs per engine |

**We do NOT modify BulletServe's scheduler or libsmctrl.** We only replace Triton kernels with SM-aware versions.

### Compatibility

libsmctrl sets TPC mask directly on the CUDA stream object's internal data structure. Triton launches kernels via standard `cuLaunchKernelEx` on the same stream. The mask applies transparently — confirmed by BulletServe shipping both FlashInfer (CUDA C++) and Triton kernels on the same masked stream.

---

## 2. Kernel Replacement Plan

| Kernel | Current Impl | Priority | Rationale |
|--------|-------------|:--------:|-----------|
| Fused MoE | Triton (`fused_moe_triton`) | **P0** | Dynamic grid from routing → worst wave quantization |
| GEMM (linear proj) | cuBLAS / Triton | **P1** | Prefill bottleneck, tile size sensitive to SM count |
| Attention | FlashInfer (CUDA C++) | P2 | Not Triton, larger effort |
| LayerNorm/RoPE/Softmax | Triton | P3 | Memory-bound, SM count less impactful |

### Code Change Pattern (identical for all Triton kernels)

```python
# Before (vanilla)
@triton.autotune(configs=..., key=['M', 'N', 'K'])
@triton.jit
def kernel(...):
    ...

# After (SM-aware)
@triton.autotune(configs=..., key=['M', 'N', 'K', 'NUM_SMS'])
@triton.jit
def kernel(..., NUM_SMS: tl.constexpr):
    ...  # kernel body unchanged
```

Callsite reads TPC count from BulletServe's SharedManager:

```python
num_sms = shared_manager.prefill_num_tpcs * 2  # TPC -> SM
kernel[grid](..., NUM_SMS=num_sms)
```

---

## 3. L2 / Bandwidth Pressure

**Phase 1 (now):** SM-aware only. No pressure parameter. Rationale:
- SM-aware is a complete contribution on its own
- L2 pressure is hard to quantify at runtime
- Prove SM-aware works first, then layer on cache policy

**Phase 2 (future):** Add `PRESSURE_LEVEL` to autotune key:
```python
@triton.autotune(configs=..., key=['M', 'N', 'K', 'NUM_SMS', 'PRESSURE_LEVEL'])
```
BulletServe scheduler estimates pressure from co-running batch sizes.

---

## 4. E2E Benchmark Pipeline

### Setup
- **Framework:** BulletServe (SGLang v0.3.0 fork)
- **Model:** Llama-3.1-8B (BulletServe's primary eval model, single GPU)
- **Hardware:** A100 (108 SMs / 54 TPCs)
- **Trace:** ShareGPT (standard real-world conversation distribution)

### A/B Comparison
```
A (baseline): BulletServe + vanilla SGLang Triton kernels
B (ours):     BulletServe + ElastiKernel SM-aware Triton kernels
```

### Metrics
- **TTFT** (Time To First Token) — primarily affected by prefill kernel efficiency
- **TBT/TPOT** (Time Between Tokens) — decode phase
- **Throughput** (tokens/sec)
- **SLO Attainment** (% requests meeting latency target)

### Experiment Matrix
Sweep prefill/decode TPC ratios: (40/14, 34/20, 27/27, 20/34, 14/40)
At each ratio, compare vanilla vs SM-aware kernel TTFT and throughput.

---

## 5. Work Breakdown

| Step | Description | Effort |
|------|-------------|--------|
| 1 | Fork BulletServe, build libsmctrl, verify baseline runs | 0.5 day |
| 2 | Replace fused_moe_kernel with SM-aware version (P0) | ~10 lines |
| 3 | Replace GEMM kernel if Triton-based (P1) | ~10 lines |
| 4 | Write benchmark script for A/B comparison | 1-2 days |
| 5 | Run experiments, collect data | GPU time dependent |
