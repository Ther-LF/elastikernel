"""
Verify that Green Context disjoint partitions provide true SM isolation.

Test strategy:
  1. Create two disjoint partitions (A and B), each with N SMs
  2. Run a compute-heavy kernel on A alone, B alone, and both simultaneously
  3. If SMs are truly disjoint:
       - parallel time ≈ single partition time (both use different SMs)
     If SMs overlap:
       - parallel time ≈ 2x single partition time (serialized on same SMs)
  4. As a control, create two OVERLAPPING partitions (both from full device)
     and show that the parallel speedup disappears

Usage:
    python -m elastikernel.verify_isolation
"""

import time
import torch
import torch.cuda

from .green_ctx import (
    create_sm_partition,
    create_disjoint_partitions,
    get_total_sm_count,
    get_sm_alignment,
)


def make_workload(dev, size=4096):
    """Create a compute-heavy matmul workload."""
    a = torch.randn(size, size, device=dev, dtype=torch.float16)
    b = torch.randn(size, size, device=dev, dtype=torch.float16)
    return a, b


def bench_single_stream(stream, a, b, iters=20):
    """Time repeated matmul on a single stream."""
    # Warmup
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.matmul(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iters):
            torch.matmul(a, b)
        end.record(stream)
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # ms


def bench_parallel_streams(stream_a, stream_b, a1, b1, a2, b2, iters=20):
    """Time repeated matmul on two streams simultaneously."""
    # Warmup
    with torch.cuda.stream(stream_a):
        for _ in range(3):
            torch.matmul(a1, b1)
    with torch.cuda.stream(stream_b):
        for _ in range(3):
            torch.matmul(a2, b2)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end_a = torch.cuda.Event(enable_timing=True)
    end_b = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.cuda.stream(stream_a):
        for _ in range(iters):
            torch.matmul(a1, b1)
        end_a.record(stream_a)
    with torch.cuda.stream(stream_b):
        for _ in range(iters):
            torch.matmul(a2, b2)
        end_b.record(stream_b)
    torch.cuda.synchronize()

    time_a = start.elapsed_time(end_a)
    time_b = start.elapsed_time(end_b)
    return max(time_a, time_b)  # wall time = max of both


def main():
    dev = torch.device("cuda:0")
    total_sms = get_total_sm_count(dev)
    min_sm, align = get_sm_alignment(dev)
    print(f"Device: {torch.cuda.get_device_name(dev)}")
    print(f"Total SMs: {total_sms}, alignment: min={min_sm}, align={align}")

    # Use half the SMs per partition (leave some for remaining)
    half_sms = (total_sms // 2 // align) * align
    print(f"\nPartition size: {half_sms} SMs each (total used: {half_sms * 2}/{total_sms})")

    size = 2048
    a1, b1 = make_workload(dev, size)
    a2, b2 = make_workload(dev, size)

    # =====================================================================
    # Test 1: Disjoint partitions (chain-split, guaranteed non-overlapping)
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 1: DISJOINT partitions (create_disjoint_partitions)")
    print(f"{'='*70}")

    parts = create_disjoint_partitions(dev, [half_sms, half_sms])
    stream_a, actual_a = parts[0]
    stream_b, actual_b = parts[1]
    print(f"  Partition A: {actual_a} SMs")
    print(f"  Partition B: {actual_b} SMs")

    time_a_solo = bench_single_stream(stream_a, a1, b1)
    time_b_solo = bench_single_stream(stream_b, a2, b2)
    time_parallel = bench_parallel_streams(stream_a, stream_b, a1, b1, a2, b2)
    avg_solo = (time_a_solo + time_b_solo) / 2
    speedup = (time_a_solo + time_b_solo) / time_parallel

    print(f"\n  A solo:     {time_a_solo:8.2f} ms")
    print(f"  B solo:     {time_b_solo:8.2f} ms")
    print(f"  Parallel:   {time_parallel:8.2f} ms")
    print(f"  Sequential: {time_a_solo + time_b_solo:8.2f} ms")
    print(f"  Speedup (seq/parallel): {speedup:.2f}x")
    print(f"  Parallel/AvgSolo ratio: {time_parallel/avg_solo:.2f}x")

    if speedup > 1.5:
        print(f"\n  >> PASS: Speedup {speedup:.2f}x indicates true SM isolation")
        print(f"  >> (If SMs overlapped, parallel ≈ sequential, speedup ≈ 1.0x)")
    else:
        print(f"\n  >> WARN: Speedup only {speedup:.2f}x — possible SM overlap or "
              f"memory bandwidth bottleneck")

    # =====================================================================
    # Test 2: Overlapping partitions (old API, both from full device)
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 2: OVERLAPPING partitions (create_sm_partition x2)")
    print(f"{'='*70}")

    stream_c, actual_c = create_sm_partition(dev, half_sms)
    stream_d, actual_d = create_sm_partition(dev, half_sms)
    print(f"  Partition C: {actual_c} SMs (from full device)")
    print(f"  Partition D: {actual_d} SMs (from full device, likely same SMs!)")

    time_c_solo = bench_single_stream(stream_c, a1, b1)
    time_d_solo = bench_single_stream(stream_d, a2, b2)
    time_overlap = bench_parallel_streams(stream_c, stream_d, a1, b1, a2, b2)
    avg_solo_2 = (time_c_solo + time_d_solo) / 2
    speedup_2 = (time_c_solo + time_d_solo) / time_overlap

    print(f"\n  C solo:     {time_c_solo:8.2f} ms")
    print(f"  D solo:     {time_d_solo:8.2f} ms")
    print(f"  Parallel:   {time_overlap:8.2f} ms")
    print(f"  Sequential: {time_c_solo + time_d_solo:8.2f} ms")
    print(f"  Speedup (seq/parallel): {speedup_2:.2f}x")
    print(f"  Parallel/AvgSolo ratio: {time_overlap/avg_solo_2:.2f}x")

    if speedup_2 < 1.3:
        print(f"\n  >> EXPECTED: Speedup only {speedup_2:.2f}x — SMs overlap, "
              f"kernels contend")
    else:
        print(f"\n  >> UNEXPECTED: Speedup {speedup_2:.2f}x — overlapping partitions "
              f"show parallelism (may have enough memory bandwidth)")

    # =====================================================================
    # Test 3: Correctness — both partitions produce correct results
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 3: Correctness check")
    print(f"{'='*70}")

    ref = torch.matmul(a1, b1)
    with torch.cuda.stream(stream_a):
        out_a = torch.matmul(a1, b1)
    with torch.cuda.stream(stream_b):
        out_b = torch.matmul(a1, b1)
    torch.cuda.synchronize()

    match_a = torch.allclose(out_a, ref, atol=1e-2, rtol=0)
    match_b = torch.allclose(out_b, ref, atol=1e-2, rtol=0)
    print(f"  Partition A matches reference: {match_a}")
    print(f"  Partition B matches reference: {match_b}")
    if match_a and match_b:
        print(f"  >> PASS: Both partitions produce correct results")
    else:
        print(f"  >> FAIL: Result mismatch!")

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Disjoint partitions parallel speedup:    {speedup:.2f}x (expect ~2.0x)")
    print(f"  Overlapping partitions parallel speedup:  {speedup_2:.2f}x (expect ~1.0x)")
    print(f"  Isolation ratio (disjoint/overlapping):   {speedup/speedup_2:.2f}x")

    if speedup > 1.5 * speedup_2:
        print(f"\n  VERDICT: Green Context disjoint partitions provide real SM isolation.")
    else:
        print(f"\n  VERDICT: Results inconclusive — may be memory-bandwidth bound.")
        print(f"  Try with smaller matrices (more compute-bound) or check Nsight.")


if __name__ == "__main__":
    main()
