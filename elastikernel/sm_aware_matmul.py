"""
SM-Aware Matrix Multiplication with Green Context Support.

Usage:
    python -m elastikernel.sm_aware_matmul --analytical
    python -m elastikernel.sm_aware_matmul --benchmark
"""

import argparse
from typing import List, Tuple, Dict


# =============================================================================
# Part 1: Wave Analysis (Pure Python, no GPU/triton needed)
# =============================================================================

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def wave_efficiency(M, N, block_m, block_n, num_sms):
    """Returns (num_waves, last_wave_util, overall_efficiency)."""
    num_blocks = ceil_div(M, block_m) * ceil_div(N, block_n)
    num_waves = ceil_div(num_blocks, num_sms)
    last_wave_blocks = num_blocks % num_sms
    if last_wave_blocks == 0:
        last_wave_blocks = num_sms
    return num_waves, last_wave_blocks / num_sms, num_blocks / (num_waves * num_sms)


CONFIGS = [
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
]


def run_analytical():
    sm_levels = [16, 32, 48, 64, 80, 96, 112, 132]
    test_cases = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]

    for M, N, K in test_cases:
        print(f"\n{'='*90}")
        print(f"Wave Analysis: M={M}, N={N}, K={K}")
        print(f"{'='*90}")

        best_per_sm = {}
        for num_sms in sm_levels:
            ranked = []
            for c in CONFIGS:
                nw, lwu, eff = wave_efficiency(M, N, c['BLOCK_SIZE_M'], c['BLOCK_SIZE_N'], num_sms)
                ranked.append((c, nw, lwu, eff))
            ranked.sort(key=lambda x: (x[1], -x[2]))
            best_per_sm[num_sms] = ranked[0]

            print(f"\n--- {num_sms} SMs ---")
            print(f"  {'Rank':<5} {'Config':<45} {'Waves':<7} {'LastUtil':<10} {'Eff':<10}")
            for i, (c, nw, lwu, eff) in enumerate(ranked[:5], 1):
                s = f"M={c['BLOCK_SIZE_M']},N={c['BLOCK_SIZE_N']},K={c['BLOCK_SIZE_K']}"
                print(f"  {i:<5} {s:<45} {nw:<7} {lwu*100:>5.1f}%    {eff*100:>5.1f}%")

        # Summary
        print(f"\n{'='*90}")
        print(f"Best Config per SM (M={M}, N={N}, K={K})")
        print(f"{'='*90}")
        print(f"  {'SM':<6} {'Best Config':<45} {'Waves':<7} {'Eff':<10} {'Changed?'}")
        prev = None
        for num_sms in sm_levels:
            c, nw, lwu, eff = best_per_sm[num_sms]
            s = f"M={c['BLOCK_SIZE_M']},N={c['BLOCK_SIZE_N']},K={c['BLOCK_SIZE_K']}"
            changed = " <-- CHANGED" if prev is not None and c != prev else ""
            print(f"  {num_sms:<6} {s:<45} {nw:<7} {eff*100:>5.1f}%{changed}")
            prev = c


# =============================================================================
# Part 2 & 3: GPU Kernels (only imported when --benchmark is used)
# =============================================================================

def run_benchmark():
    """Benchmark vanilla vs SM-aware matmul under Green Context."""
    import torch
    import triton
    import triton.language as tl
    from .green_ctx import create_sm_partition, get_total_sm_count, get_sm_alignment

    # --- Autotune configs ---
    def make_configs():
        return [
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ]

    # --- Vanilla kernel: key=['M','N','K'] ---
    @triton.autotune(configs=make_configs(), key=['M', 'N', 'K'])
    @triton.jit
    def matmul_vanilla(
        a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):
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

    # --- SM-aware kernel: key=['M','N','K','NUM_SMS'] ---
    @triton.autotune(configs=make_configs(), key=['M', 'N', 'K', 'NUM_SMS'])
    @triton.jit
    def matmul_sm_aware(
        a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        NUM_SMS: tl.constexpr,  # only for autotune key, not used in kernel body
    ):
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

    # --- Wrappers ---
    def call_vanilla(a, b):
        M, K = a.shape; K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        matmul_vanilla[grid](a, b, c, M, N, K,
                             a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
        return c

    def call_sm_aware(a, b, num_sms):
        M, K = a.shape; K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        matmul_sm_aware[grid](a, b, c, M, N, K,
                              a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                              NUM_SMS=num_sms)
        return c

    # --- Run benchmark ---
    dev = torch.device("cuda:0")
    total_sms = get_total_sm_count(dev)
    min_sm, align = get_sm_alignment(dev)
    print(f"Device: {torch.cuda.get_device_name(dev)}")
    print(f"Total SMs: {total_sms}, SM alignment: min={min_sm}, align={align}")

    sm_levels = [s for s in [16, 32, 48, 64, 80, 96, 112, 132] if s <= total_sms]
    test_cases = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]

    for M, N, K in test_cases:
        print(f"\n{'='*80}")
        print(f"M={M}, N={N}, K={K}")
        print(f"{'='*80}")
        print(f"  {'SM':>4} {'Actual':>6}  {'Vanilla(us)':>11} {'Aware(us)':>11} {'Speedup':>8}  Config")
        print(f"  {'-'*70}")

        torch.manual_seed(0)
        a = torch.randn((M, K), device=dev, dtype=torch.float16)
        b = torch.randn((K, N), device=dev, dtype=torch.float16)
        ref = torch.matmul(a, b)

        for sm_count in sm_levels:
            try:
                stream, actual_sm = create_sm_partition(dev, sm_count)
                with torch.cuda.stream(stream):
                    # Warmup
                    for _ in range(3):
                        call_vanilla(a, b)
                        call_sm_aware(a, b, actual_sm)
                    torch.cuda.synchronize()

                    van_ms = triton.testing.do_bench(lambda: call_vanilla(a, b), rep=50)
                    aw_ms = triton.testing.do_bench(lambda: call_sm_aware(a, b, actual_sm), rep=50)

                van_cfg = matmul_vanilla.best_config
                aw_cfg = matmul_sm_aware.best_config
                same = "SAME" if van_cfg.kwargs == aw_cfg.kwargs else "DIFF"
                sp = van_ms / aw_ms if aw_ms > 0 else 0
                aw_str = f"M={aw_cfg.kwargs['BLOCK_SIZE_M']},N={aw_cfg.kwargs['BLOCK_SIZE_N']}"

                print(f"  {sm_count:>4} {actual_sm:>6}  {van_ms*1000:>11.1f} {aw_ms*1000:>11.1f} {sp:>7.2f}x  {same} {aw_str}")

                # Correctness check
                with torch.cuda.stream(stream):
                    out = call_sm_aware(a, b, actual_sm)
                if not torch.allclose(out, ref, atol=1e-2, rtol=0):
                    print(f"         WARNING: result mismatch!")
            except Exception as e:
                print(f"  {sm_count:>4}  Error: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SM-Aware Matmul Benchmark")
    parser.add_argument("--analytical", action="store_true", help="Wave analysis (no GPU)")
    parser.add_argument("--benchmark", action="store_true", help="GPU benchmark with Green Context")
    args = parser.parse_args()

    if args.analytical:
        run_analytical()
    elif args.benchmark:
        run_benchmark()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
