# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ElastiKernel: Resource-adaptive kernel auto-tuning for spatially shared GPUs. The core insight is that Triton's autotune picks configs assuming exclusive GPU access — when kernels run under reduced SMs (MPS/MIG/Green Context), the optimal tile size changes due to wave quantization effects.

## Repository Structure

- `elastikernel/` — Main Python package (pushed to GitHub as `Ther-LF/elastikernel`)
  - `green_ctx.py` — CUDA Green Context wrapper using `cuda-python` driver API. Creates SM-partitioned streams without process restart (unlike MPS).
  - `sm_aware_matmul.py` — Two modes: `--analytical` (pure Python wave analysis, no GPU) and `--benchmark` (Triton GEMM with Green Context, needs GPU).
  - `verify_isolation.py` — Timing-based test proving disjoint Green Context partitions have real SM isolation.
- `docs/plans/` — Design documents
  - `2026-03-17-elastikernel-design.md` — Full design with 7-layer interference taxonomy
  - `2026-03-19-bulletserve-integration.md` — BulletServe e2e integration plan
  - `2026-03-21-flashinfer-bench-integration.md` — FlashInfer-Bench AI agent workflow integration
- `paper/` — LaTeX paper and literature survey
- `triton/` — Upstream Triton source (read-only reference)
- `DeepGEMM/` — DeepSeek's DeepGEMM (read-only reference)

## Commands

```bash
# Wave analysis (no GPU required, runs anywhere)
python -m elastikernel.sm_aware_matmul --analytical

# GPU benchmark (requires CUDA GPU CC 7.0+, cuda-python, triton)
python -m elastikernel.sm_aware_matmul --benchmark

# Test Green Context directly
python -c "from elastikernel.green_ctx import create_sm_partition; import torch; s, n = create_sm_partition(torch.device('cuda:0'), 16); print(f'Got {n} SMs')"
```

## Dependencies

- `torch` and `triton` — always needed for GPU paths
- `cuda-python` (`pip install cuda-python`) — required for Green Context (driver API bindings)
- Hardware: CUDA CC 7.0+ (SM alignment varies: CC7=2, CC8=2, CC9+=8)

## Architecture: The SM-Aware Autotune Trick

The key difference between `matmul_vanilla` and `matmul_sm_aware` is a single change:

```python
# Vanilla: same config reused regardless of SM count
@triton.autotune(configs=..., key=['M', 'N', 'K'])

# SM-aware: re-tunes when SM count changes
@triton.autotune(configs=..., key=['M', 'N', 'K', 'NUM_SMS'])
```

`NUM_SMS` is passed as a `tl.constexpr` argument but is **not used in the kernel body** — it only forces Triton to re-run autotuning for each distinct SM count.

## Development vs Runtime Environment

**本机 (macOS) 是纯开发环境，没有 GPU，代码不在本地运行。** 每次写完或修改代码后，必须给出一条可以直接在目标 GPU 服务器上执行的完整命令，格式示例：

```bash
# 在目标服务器上拉取最新代码并运行
cd /path/to/elastikernel && git pull && python -m elastikernel.sm_aware_matmul --benchmark
```

不要只说"运行 xxx"——要给出从 git pull 到执行的完整一行命令，让用户可以直接复制粘贴到服务器终端。

## Green Context: Disjoint Partitions

`create_sm_partition()` splits from full device each time — two calls get OVERLAPPING SMs.
Use `create_disjoint_partitions()` for guaranteed non-overlapping SM sets (chain-split from remaining resource).
Green Context isolates SMs only — L2 cache and DRAM bandwidth are still shared.

## E2E Testing: BulletServe Integration

ElastiKernel integrates into **BulletServe** (SGLang v0.3.0 fork, ASPLOS'26) for end-to-end LLM inference benchmarks. BulletServe handles scheduling + SM partitioning (libsmctrl), ElastiKernel replaces the kernel layer with SM-aware versions.

- libsmctrl sets TPC bitmask on CUDA stream → compatible with Triton (mask applies to any kernel on that stream)
- We read `SharedManager.prefill_num_tpcs * 2` → pass as `NUM_SMS` to Triton autotune key
- Kernel body unchanged, only autotune key gains `NUM_SMS` dimension
- Priority: P0 = fused_moe (Triton), P1 = GEMM (Triton), P2 = attention (FlashInfer CUDA, harder)

### BulletServe Setup (on GPU server)

```bash
# Clone and build
git clone https://github.com/zejia-lin/BulletServe.git && cd BulletServe
cd csrc && make config && make build && cd ..
pip install -e "python[all]"

# Start MPS (required for spatial sharing)
bash ./scripts/start_mps.sh

# Launch server (Llama-3.1-8B, single A100)
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B --enable-bullet-full

# Stop MPS when done
bash ./scripts/kill_mps.sh
```

Requirements: CUDA <= 12.6, Python >= 3.12.9, NVIDIA GPU with MPS support.

## E2E Testing: FlashInfer-Bench Integration

FlashInfer-Bench (MLSys'26) provides AI agent workflow for kernel generation, benchmarking, and deployment. We extend it with SM-aware dimension.

### Why FlashInfer-Bench

- **FlashInfer Trace Schema**: JSON format for kernel definitions, workloads, solutions
- **Agent Tools**: `flashinfer_bench.agents` for packing, sanitizer, NCU profiling
- **Deployment**: `apply()` injects kernels into SGLang/vLLM with zero code change
- **Leaderboard**: Track AI agent GPU programming capabilities
- **Green Context**: `flashinfer.green_ctx` already has SM partitioning API

### Integration Points

1. **Extend Definition Schema**:
   ```json
   "axes": {
     "NUM_SMS": {"type": "var", "dtype": "int32"}
   }
   ```

2. **Agent Prompt** includes NUM_SMS for SM-aware autotuning:
   ```
   Add NUM_SMS to triton.autotune key: key=['M', 'N', 'K', 'NUM_SMS']
   Different tile sizes are optimal for different SM counts.
   ```

3. **Evaluation Matrix**: SM counts × workloads × kernels
   - SM: 16, 32, 48, 64, 80, 96, 108 (A100)
   - Kernels: fused_moe, GEMM, attention
   - Metric: `fast_p` curve per SM count

4. **Deployment via apply()**:
   ```python
   flashinfer_bench.apply.enable(
       definition="fused_moe_sm_aware",
       resolver=lambda args: f"fused_moe_sm_aware_{args['num_sms']}",
   )
   ```

### FlashInfer-Bench Setup

```bash
# Install
pip install flashinfer-bench modal

# Download dataset
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
export FIB_DATASET_PATH=/path/to/flashinfer-trace

# Run local benchmark
python scripts/run_local.py

# Run on B200 via Modal
modal run scripts/run_modal.py
```

Full design: `docs/plans/2026-03-21-flashinfer-bench-integration.md`

## Design Conventions

- GPU imports (`torch`, `triton`, `cuda.bindings`) are lazy-loaded inside functions to keep `--analytical` mode runnable without GPU.
- `green_ctx.py` eagerly imports `torch` and `cuda-python` at module level (it's only imported by GPU code paths).
- The `CONFIGS` list in `sm_aware_matmul.py` is shared between analytical and benchmark modes — keep them in sync.
