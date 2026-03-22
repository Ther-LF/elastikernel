"""
SM-Aware Kernel Benchmark Suite.

Tests two key operators for LLM inference:
1. GEMM - General Matrix Multiply
2. Fused MoE - Mixture of Experts with fused activation

Usage:
    python -m elastikernel.sm_aware_benchmark --gemm
    python -m elastikernel.sm_aware_benchmark --moe
    python -m elastikernel.sm_aware_benchmark --all
"""

import argparse


def run_gemm_benchmark():
    """Benchmark SM-aware GEMM."""
    import torch
    import triton
    import triton.language as tl
    from .green_ctx import create_sm_partition, get_total_sm_count, get_sm_alignment

    # --- Vanilla configs ---
    def make_configs():
        return [
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ]

    # --- SM-AWARE configs: tailored for different SM regimes ---
    def make_sm_aware_configs():
        return [
            # Ultra-small tiles for very few SMs (8-16 SMs)
            triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=2),
            # Small tiles for few SMs (16-32 SMs)
            triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 2}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 2}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 2}, num_stages=4, num_warps=2),
            # Medium tiles for moderate SMs (32-64 SMs)
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
            # Large tiles for many SMs (64+ SMs)
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        ]

    @triton.autotune(configs=make_configs(), key=['M', 'N', 'K'])
    @triton.jit
    def matmul_vanilla(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                       BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        gid = pid // (GROUP_SIZE_M * num_pid_n)
        first = gid * GROUP_SIZE_M
        gsz = min(num_pid_m - first, GROUP_SIZE_M)
        pid_m = first + ((pid % (GROUP_SIZE_M * num_pid_n)) % gsz)
        pid_n = (pid % (GROUP_SIZE_M * num_pid_n)) // gsz
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c = acc.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
                 c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

    @triton.autotune(configs=make_sm_aware_configs(), key=['M', 'N', 'K', 'NUM_SMS'])
    @triton.jit
    def matmul_sm_aware(a_ptr, b_ptr, c_ptr, M, N, K,
                        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                        NUM_SMS: tl.constexpr):
        # Same kernel body as vanilla
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        gid = pid // (GROUP_SIZE_M * num_pid_n)
        first = gid * GROUP_SIZE_M
        gsz = min(num_pid_m - first, GROUP_SIZE_M)
        pid_m = first + ((pid % (GROUP_SIZE_M * num_pid_n)) % gsz)
        pid_n = (pid % (GROUP_SIZE_M * num_pid_n)) // gsz
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c = acc.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
                 c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

    def call_vanilla(a, b):
        M, K = a.shape; K2, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        matmul_vanilla[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
        return c

    def call_sm_aware(a, b, num_sms):
        M, K = a.shape; K2, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        matmul_sm_aware[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), NUM_SMS=num_sms)
        return c

    # --- Test Matrix ---
    dev = torch.device("cuda:0")
    total_sms = get_total_sm_count(dev)
    min_sm, align = get_sm_alignment(dev)
    print(f"GEMM Benchmark on {torch.cuda.get_device_name(dev)}")
    print(f"Total SMs: {total_sms}, alignment: min={min_sm}, align={align}")

    sm_levels = [s for s in [16, 32, 48, 64] if s <= total_sms]

    # 1. Square matrices
    square_cases = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    # 2. M sweep (N=K=4096, typical LLM hidden dim)
    m_sweep_cases = [(M, 4096, 4096) for M in [1, 16, 32, 64, 128, 256, 512, 1024, 2048]]
    test_cases = square_cases + m_sweep_cases

    for M, N, K in test_cases:
        print(f"\n{'='*80}")
        print(f"GEMM: M={M}, N={N}, K={K}")
        print(f"{'='*80}")
        print(f"  {'SM':>4} {'Vanilla(us)':>11} {'Aware(us)':>11} {'Speedup':>8}")
        print(f"  {'-'*40}")

        torch.manual_seed(0)
        a = torch.randn((M, K), device=dev, dtype=torch.float16)
        b = torch.randn((K, N), device=dev, dtype=torch.float16)
        ref = torch.matmul(a, b)

        for sm_count in sm_levels:
            try:
                stream, actual_sm = create_sm_partition(dev, sm_count)
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        call_vanilla(a, b)
                        call_sm_aware(a, b, actual_sm)
                    torch.cuda.synchronize()

                    van_ms = triton.testing.do_bench(lambda: call_vanilla(a, b), rep=30)
                    aw_ms = triton.testing.do_bench(lambda: call_sm_aware(a, b, actual_sm), rep=30)

                sp = van_ms / aw_ms if aw_ms > 0 else 0
                print(f"  {sm_count:>4} {van_ms*1000:>11.1f} {aw_ms*1000:>11.1f} {sp:>7.2f}x")
            except Exception as e:
                print(f"  {sm_count:>4}  Error: {e}")


def run_moe_benchmark():
    """Benchmark SM-aware Fused MoE."""
    import torch
    import triton
    import triton.language as tl
    from .green_ctx import create_sm_partition, get_total_sm_count, get_sm_alignment

    # --- Fused MoE: grouped GEMM with activation ---
    # Key pattern: top_k tokens routed to different experts
    # Grid size = num_tokens * top_k, highly irregular

    def make_moe_vanilla_configs():
        """Standard MoE configs assuming full GPU."""
        return [
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ]

    def make_moe_sm_aware_configs():
        """SM-aware configs optimized for MoE's irregular grid pattern."""
        # MoE has num_tokens * top_k blocks, which can be very sparse
        # Need smaller tiles to increase block count and improve wave fill
        return [
            # Ultra-small for very low SMs or small token counts
            triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 2}, num_stages=4, num_warps=2),
            # Small tiles for few SMs (critical for MoE's irregular workload)
            triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 2}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5, num_warps=2),
            # Medium tiles
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            # Large tiles for many SMs
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        ]

    @triton.autotune(configs=make_moe_vanilla_configs(), key=['M', 'N', 'K', 'NUM_EXPERTS'])
    @triton.jit
    def fused_moe_vanilla(
        a_ptr, b_ptr, c_ptr,
        expert_offsets_ptr,  # [num_experts + 1], cumulative token counts per expert
        M, N, K, NUM_EXPERTS,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):
        """Fused MoE: Grouped GEMM with per-expert workloads."""
        # Get which expert this block handles
        expert_id = tl.program_id(1)
        num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)

        # Get token range for this expert
        start_token = tl.load(expert_offsets_ptr + expert_id)
        end_token = tl.load(expert_offsets_ptr + expert_id + 1)
        num_tokens_this_expert = end_token - start_token

        if num_tokens_this_expert == 0:
            return  # No work for this expert

        # Which block within this expert's workload
        block_idx = tl.program_id(0)
        num_blocks_m = tl.cdiv(num_tokens_this_expert, BLOCK_SIZE_M)

        if block_idx >= num_blocks_m * num_blocks_n:
            return

        # Compute block position
        pid_m = block_idx // num_blocks_n
        pid_n = block_idx % num_blocks_n

        # Pointers
        offs_m = start_token + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        # Bound check for tokens
        mask_m = offs_m < end_token

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + (expert_id * N * K) + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Store result directly (no SwiGLU for simplicity - grouped GEMM only)
        c = acc.to(tl.float16)
        offs_cm = start_token + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < end_token) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.autotune(configs=make_moe_sm_aware_configs(), key=['M', 'N', 'K', 'NUM_EXPERTS', 'NUM_SMS'])
    @triton.jit
    def fused_moe_sm_aware(
        a_ptr, b_ptr, c_ptr,
        expert_offsets_ptr,
        M, N, K, NUM_EXPERTS,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        """SM-aware Fused MoE - same logic, different autotune key."""
        expert_id = tl.program_id(1)
        num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)

        start_token = tl.load(expert_offsets_ptr + expert_id)
        end_token = tl.load(expert_offsets_ptr + expert_id + 1)
        num_tokens_this_expert = end_token - start_token

        if num_tokens_this_expert == 0:
            return

        block_idx = tl.program_id(0)
        num_blocks_m = tl.cdiv(num_tokens_this_expert, BLOCK_SIZE_M)

        if block_idx >= num_blocks_m * num_blocks_n:
            return

        pid_m = block_idx // num_blocks_n
        pid_n = block_idx % num_blocks_n

        offs_m = start_token + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        mask_m = offs_m < end_token

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + (expert_id * N * K) + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Store result directly (no SwiGLU for simplicity - grouped GEMM only)
        c = acc.to(tl.float16)
        offs_cm = start_token + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < end_token) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def call_moe_vanilla(hidden_states, weights, expert_offsets, num_experts, N, K):
        """Call vanilla MoE kernel."""
        M = hidden_states.shape[0]
        output = torch.empty((M, N), device=hidden_states.device, dtype=torch.float16)

        # Grid: (blocks per expert) x (num_experts)
        max_blocks_per_expert = triton.cdiv(M, 16) * triton.cdiv(N, 64)  # Conservative estimate
        grid = lambda META: (
            max_blocks_per_expert,
            num_experts,
        )

        fused_moe_vanilla[grid](
            hidden_states, weights, output, expert_offsets,
            M, N, K, num_experts,
            hidden_states.stride(0), hidden_states.stride(1),
            weights.stride(0), weights.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    def call_moe_sm_aware(hidden_states, weights, expert_offsets, num_experts, N, K, num_sms):
        """Call SM-aware MoE kernel."""
        M = hidden_states.shape[0]
        output = torch.empty((M, N), device=hidden_states.device, dtype=torch.float16)

        max_blocks_per_expert = triton.cdiv(M, 16) * triton.cdiv(N, 64)
        grid = lambda META: (
            max_blocks_per_expert,
            num_experts,
        )

        fused_moe_sm_aware[grid](
            hidden_states, weights, output, expert_offsets,
            M, N, K, num_experts,
            hidden_states.stride(0), hidden_states.stride(1),
            weights.stride(0), weights.stride(1),
            output.stride(0), output.stride(1),
            NUM_SMS=num_sms,
        )
        return output

    # --- Test Matrix ---
    dev = torch.device("cuda:0")
    total_sms = get_total_sm_count(dev)
    min_sm, align = get_sm_alignment(dev)
    print(f"\n{'='*80}")
    print(f"Fused MoE Benchmark on {torch.cuda.get_device_name(dev)}")
    print(f"Total SMs: {total_sms}, alignment: min={min_sm}, align={align}")
    print(f"{'='*80}")

    sm_levels = [s for s in [16, 32, 48, 64] if s <= total_sms]

    # MoE configurations (typical LLM settings)
    # num_tokens, hidden_dim, intermediate_dim (full), num_experts, top_k
    moe_configs = [
        # Small batch, moderate experts
        {'num_tokens': 128, 'hidden_dim': 4096, 'intermediate_dim': 4096, 'num_experts': 8, 'top_k': 2},
        {'num_tokens': 512, 'hidden_dim': 4096, 'intermediate_dim': 4096, 'num_experts': 8, 'top_k': 2},
        # Large batch
        {'num_tokens': 1024, 'hidden_dim': 4096, 'intermediate_dim': 4096, 'num_experts': 8, 'top_k': 2},
        # More experts
        {'num_tokens': 512, 'hidden_dim': 4096, 'intermediate_dim': 4096, 'num_experts': 16, 'top_k': 4},
    ]

    for cfg in moe_configs:
        num_tokens = cfg['num_tokens']
        hidden_dim = cfg['hidden_dim']  # K
        intermediate_dim = cfg['intermediate_dim']  # N (full intermediate size before SwiGLU)
        num_experts = cfg['num_experts']
        top_k = cfg['top_k']

        print(f"\n{'='*80}")
        print(f"MoE: tokens={num_tokens}, hidden={hidden_dim}, intermediate={intermediate_dim}, experts={num_experts}, top_k={top_k}")
        print(f"{'='*80}")
        print(f"  {'SM':>4} {'Vanilla(us)':>11} {'Aware(us)':>11} {'Speedup':>8}")
        print(f"  {'-'*40}")

        torch.manual_seed(42)

        # Input: [num_tokens, hidden_dim]
        hidden_states = torch.randn((num_tokens, hidden_dim), device=dev, dtype=torch.float16)

        # Weights: [num_experts * intermediate_dim, hidden_dim]
        # Each expert has (intermediate_dim, hidden_dim) weight
        weights = torch.randn((num_experts * intermediate_dim, hidden_dim), device=dev, dtype=torch.float16)

        # Simulate routing: assign tokens to experts
        # For simplicity, uniform distribution
        tokens_per_expert = (num_tokens * top_k) // num_experts
        expert_offsets = torch.zeros(num_experts + 1, device=dev, dtype=torch.int32)
        for i in range(num_experts):
            expert_offsets[i + 1] = min((i + 1) * tokens_per_expert, num_tokens)
        expert_offsets[-1] = num_tokens

        for sm_count in sm_levels:
            try:
                stream, actual_sm = create_sm_partition(dev, sm_count)
                with torch.cuda.stream(stream):
                    # Warmup
                    for _ in range(3):
                        call_moe_vanilla(hidden_states, weights, expert_offsets, num_experts, intermediate_dim, hidden_dim)
                        call_moe_sm_aware(hidden_states, weights, expert_offsets, num_experts, intermediate_dim, hidden_dim, actual_sm)
                    torch.cuda.synchronize()

                    van_ms = triton.testing.do_bench(
                        lambda: call_moe_vanilla(hidden_states, weights, expert_offsets, num_experts, intermediate_dim, hidden_dim),
                        rep=30
                    )
                    aw_ms = triton.testing.do_bench(
                        lambda: call_moe_sm_aware(hidden_states, weights, expert_offsets, num_experts, intermediate_dim, hidden_dim, actual_sm),
                        rep=30
                    )

                sp = van_ms / aw_ms if aw_ms > 0 else 0
                print(f"  {sm_count:>4} {van_ms*1000:>11.1f} {aw_ms*1000:>11.1f} {sp:>7.2f}x")
            except Exception as e:
                print(f"  {sm_count:>4}  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="SM-Aware Kernel Benchmark")
    parser.add_argument("--gemm", action="store_true", help="Benchmark GEMM")
    parser.add_argument("--moe", action="store_true", help="Benchmark Fused MoE")
    parser.add_argument("--all", action="store_true", help="Benchmark all operators")
    args = parser.parse_args()

    if args.all:
        run_gemm_benchmark()
        run_moe_benchmark()
    elif args.gemm:
        run_gemm_benchmark()
    elif args.moe:
        run_moe_benchmark()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
