# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ElastiKernel: Resource-adaptive kernel auto-tuning for spatially shared GPUs. The core insight is that Triton's autotune picks configs assuming exclusive GPU access — when kernels run under reduced SMs (MPS/MIG/Green Context), the optimal tile size changes due to wave quantization effects.

## Repository Structure

- `elastikernel/` — Main Python package (pushed to GitHub as `Ther-LF/elastikernel`)
  - `green_ctx.py` — CUDA Green Context wrapper using `cuda-python` driver API. Creates SM-partitioned streams without process restart (unlike MPS).
  - `sm_aware_matmul.py` — Two modes: `--analytical` (pure Python wave analysis, no GPU) and `--benchmark` (Triton GEMM with Green Context, needs GPU).
- `docs/plans/` — Design documents (the main one: `2026-03-17-elastikernel-design.md`)
- `paper/` — LaTeX paper and literature survey
- `triton/` — Upstream Triton source (git submodule/clone, read-only reference)
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

## Design Conventions

- GPU imports (`torch`, `triton`, `cuda.bindings`) are lazy-loaded inside functions to keep `--analytical` mode runnable without GPU.
- `green_ctx.py` eagerly imports `torch` and `cuda-python` at module level (it's only imported by GPU code paths).
- The `CONFIGS` list in `sm_aware_matmul.py` is shared between analytical and benchmark modes — keep them in sync.
