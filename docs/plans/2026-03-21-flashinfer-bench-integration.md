# ElastiKernel + FlashInfer-Bench Integration Design

> Date: 2026-03-21
> Goal: Use FlashInfer-Bench framework to evaluate and deploy SM-aware kernels with AI agent workflow

---

## 1. Why FlashInfer-Bench

| 维度 | 自建 benchmark | FlashInfer-Bench |
|------|---------------|-----------------|
| Schema | 需要自己设计 | FlashInfer Trace 已定义 |
| Agent tools | 需要自己写 | `flashinfer_bench.agents` 已提供 |
| Deployment | 需要自己写 | `apply()` 零代码注入 SGLang/vLLM |
| Leaderboard | 需要自己搭 | 已有公开排行榜 |
| Green Context | 需要自己实现 | `flashinfer.green_ctx` 已有 |
| Community | 无 | MLSys'26 官方比赛框架 |

**结论：用 FlashInfer-Bench，专注扩展 SM-aware 维度，不重复造轮子。**

---

## 2. Extend FlashInfer Trace Schema

### 2.1 Definition Extension

FlashInfer Trace Definition 定义 kernel 签名，每个 axis 可以是 `const` 或 `var`。我们新增：

```json
{
  "name": "fused_moe_sm_aware",
  "axes": {
    "M": {"type": "var", "dtype": "int32"},
    "N": {"type": "var", "dtype": "int32"},
    "K": {"type": "var", "dtype": "int32"},
    "NUM_EXPERTS": {"type": "const", "value": 8},
    "TOP_K": {"type": "const", "value": 2},
    "NUM_SMS": {"type": "var", "dtype": "int32", "description": "SM partition size"}
  },
  "inputs": [...],
  "outputs": [...],
  "run": "def run(...): ..."  // PyTorch reference
}
```

**关键变化**：`NUM_SMS` 作为 `var` axis，由 Workload 提供具体值。

### 2.2 Workload Extension

```json
{
  "definition": "fused_moe_sm_aware",
  "axes_values": {
    "M": 4096,
    "N": 14336,
    "K": 4096,
    "NUM_SMS": 64
  },
  "inputs": {
    "hidden_states": {"type": "safetensors", "path": "..."},
    ...
  }
}
```

每种 (M, N, K) 组合生成多个 Workload，覆盖不同 SM count：
- NUM_SMS = 16, 32, 48, 64, 80, 96, 108 (A100 full)

### 2.3 Solution Extension

```json
{
  "name": "elastikernel_fused_moe_v1",
  "definition": "fused_moe_sm_aware",
  "author": "elastikernel",
  "spec": {
    "language": "triton",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.py::fused_moe_kernel",
    "dependencies": []
  },
  "sources": [
    {"path": "kernel.py", "content": "..."}
  ]
}
```

Solution 代码中：
```python
@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K', 'NUM_SMS']  # NUM_SMS in autotune key
)
@triton.jit
def fused_moe_kernel(..., NUM_SMS: tl.constexpr):
    # kernel body unchanged
    ...
```

---

## 3. AI Agent Workflow

### 3.1 Prompt Design for SM-Aware Kernel Generation

```
You are generating a Triton kernel for {definition_name}.

## Task Description
- Operation: {definition.run}  (PyTorch reference)
- Axes: {axes}
- NUM_SMS is the SM partition size. Your kernel must be optimized for this SM count.

## Key Optimization: SM-Aware Autotuning
1. Add NUM_SMS to triton.autotune key: key=['M', 'N', 'K', 'NUM_SMS']
2. NUM_SMS is passed as tl.constexpr but NOT used in kernel body
3. This forces Triton to re-tune for each SM partition size
4. Different tile sizes are optimal for different SM counts due to wave quantization

## Hardware Constraints
- Target GPU: {hardware}
- Total SMs: {total_sms}
- SM alignment: {sm_alignment} (min SMs per partition)

## Wave Efficiency (your config must maximize this)
- num_blocks = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)
- num_waves = ceil(num_blocks / NUM_SMS)
- last_wave_util = (num_blocks % NUM_SMS) / NUM_SMS
- Target: last_wave_util > 0.8 (minimize tail waste)

## Best Practices
1. Use GROUP_SIZE_M for better L2 cache locality
2. Prefer powers of 2 for BLOCK sizes
3. Consider register pressure: fewer SMs → more blocks/SM → need lower register usage

Generate the kernel code now.
```

### 3.2 Feedback Loop (Algorithm from FlashInfer-Bench paper)

```python
from flashinfer_bench.agents import (
    pack_solution_from_files,
    flashinfer_bench_run_sanitizer,
    flashinfer_bench_run_ncu,
)

def sm_aware_agent_loop(definition, sm_counts, max_iters=10):
    """
    Agent workflow for SM-aware kernel generation.

    Args:
        definition: FlashInfer Definition object
        sm_counts: List of SM partition sizes to test
        max_iters: Maximum refinement iterations

    Returns:
        Best solution for each SM count
    """
    best_solutions = {sm: None for sm in sm_counts}

    for iteration in range(max_iters):
        for num_sms in sm_counts:
            # 1. Generate kernel with NUM_SMS awareness
            kernel_code = agent.generate(
                definition=definition,
                num_sms=num_sms,
                prompt=SM_AWARE_PROMPT,
            )

            # 2. Pack into Solution
            solution = pack_solution_from_files(
                path="./generated",
                spec=BuildSpec(language="triton", ...),
                definition=definition.name,
            )

            # 3. Run sanitizer (memcheck, racecheck)
            sanitizer_output = flashinfer_bench_run_sanitizer(
                solution=solution,
                workload=workload_for_sm(num_sms),
                sanitizer_types=["memcheck", "racecheck"],
            )

            if sanitizer_output["status"] == "failed":
                # Feed error back to agent for fix
                agent.feedback(sanitizer_output["error"])
                continue

            # 4. Benchmark
            eval_result = flashinfer_bench.benchmark(
                definition=definition,
                solution=solution,
                workload=workload_for_sm(num_sms),
            )

            if eval_result["correct"]:
                speedup = eval_result["speedup_vs_baseline"]
                if best_solutions[num_sms] is None or \
                   speedup > best_solutions[num_sms]["speedup"]:
                    best_solutions[num_sms] = {
                        "solution": solution,
                        "speedup": speedup,
                    }

    return best_solutions
```

### 3.3 Multi-SM Optimization

Agent should generate configs that work well across SM counts:

```python
CONFIGS_FOR_SM = {
    16: [
        {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},  # Small tiles for few SMs
    ],
    32: [
        {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
    ],
    64: [
        {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  # Large tiles for many SMs
    ],
    108: [
        {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128},
    ],
}
```

---

## 4. Evaluation Matrix

### 4.1 Test Matrix

| Dimension | Values |
|-----------|--------|
| Kernel | fused_moe, gemm, attention |
| Model | DeepSeek-V3 MoE, Llama-3.1-8B |
| SM Count | 16, 32, 48, 64, 80, 96, 108 |
| Sequence Length | 512, 1024, 2048, 4096 |
| Batch Size | 1, 8, 32, 128 |

Total: 3 kernels × 2 models × 7 SM counts × 4 seq × 4 batch = **672 workloads**

### 4.2 Metrics

```python
# Primary metric: fast_p curve per SM count
fast_p[num_sms] = (1/N) * sum(correct_i and speedup_i > p)

# Secondary metrics:
# 1. Wave efficiency
wave_eff = num_blocks / (num_waves * num_sms)

# 2. SM-aware vs vanilla speedup
speedup[num_sms] = baseline_lat[num_sms] / sm_aware_lat[num_sms]

# 3. Cross-SM robustness: how consistent is speedup across SM counts?
cross_sm_variance = std([speedup[sm] for sm in sm_counts])
```

### 4.3 Leaderboard Integration

| Rank | Model | Kernel | SM=16 fast_0.95 | SM=32 fast_0.95 | ... | SM=108 fast_0.95 | Avg |
|------|-------|--------|-----------------|-----------------|-----|------------------|-----|
| 1 | ElastiKernel | fused_moe | 0.82 | 0.91 | ... | 0.95 | 0.89 |
| 2 | FlashInfer | fused_moe | 0.45 | 0.72 | ... | 0.95 | 0.71 |

---

## 5. Deployment via apply()

### 5.1 Runtime Dispatch

```python
import flashinfer_bench

# Enable SM-aware kernel selection
flashinfer_bench.apply.enable(
    definition="fused_moe_sm_aware",
    resolver=lambda args: f"fused_moe_sm_aware_{args['num_sms']}",
)

# In serving loop
@flashinfer_bench.apply(definition="fused_moe_sm_aware")
def fused_moe(hidden_states, weights, num_sms):
    # Fallback to baseline if no SM-aware kernel available
    return baseline_fused_moe(hidden_states, weights)
```

### 5.2 Green Context Integration

```python
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
import torch

# Create SM partitions
partitions = split_device_green_ctx_by_sm_count(
    torch.device("cuda:0"),
    sm_counts=[prefill_sms, decode_sms]
)

stream_prefill = partitions[0].stream
stream_decode = partitions[1].stream

# Dispatch to correct kernel
with torch.cuda.stream(stream_prefill):
    result = flashinfer_bench.apply(
        hidden_states,
        weights,
        num_sms=prefill_sms,
        stream=stream_prefill,
    )
```

---

## 6. Work Breakdown

| Step | Description | Effort |
|------|-------------|--------|
| 1 | Install FlashInfer-Bench, run baseline | 0.5 day |
| 2 | Extend Definition schema with NUM_SMS axis | 0.5 day |
| 3 | Generate Workloads covering SM counts | 1 day |
| 4 | Write agent prompt + refine loop | 1 day |
| 5 | Run experiments, collect data | GPU time |
| 6 | Integrate apply() with Green Context | 1 day |

---

## 7. References

- FlashInfer-Bench paper: arXiv 2601.00227
- FlashInfer-Bench repo: https://github.com/flashinfer-ai/flashinfer-bench
- MLSys'26 competition: https://mlsys26.flashinfer.ai/
- FlashInfer green_ctx: https://docs.flashinfer.ai/api/green_ctx.html
