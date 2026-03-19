"""
Green Context SM Partitioning Utility for ElastiKernel.

Creates CUDA Green Contexts to dynamically partition GPU SMs,
enabling profiling of Triton kernels under different SM counts
without restarting the process (unlike MPS).

IMPORTANT: Use create_disjoint_partitions() for multiple partitions.
The old create_sm_partition() splits from the full device each time,
so two calls can get OVERLAPPING SMs. create_disjoint_partitions()
uses chain-split from the remaining resource to guarantee non-overlap.

Requires: cuda-python (pip install cuda-python)
Hardware: CC 7.0+ (SM alignment: CC 7.x=2, CC 8.x=4, CC 9.0+=8)

Reference: FlashInfer's green_ctx.py implementation.
"""

from typing import Tuple, List, Optional
import torch

try:
    import cuda.bindings.driver as driver
    from cuda.bindings.driver import CUdevice, CUdevResource
except ImportError as e:
    raise ImportError(
        "cuda-python is required for Green Context support. "
        "Install with: pip install cuda-python"
    ) from e


def _check_cuda(result):
    """Check CUDA driver API result and raise on error."""
    if isinstance(result, tuple):
        err = result[0]
        ret = result[1] if len(result) == 2 else result[1:]
    else:
        err = result
        ret = None

    if isinstance(err, driver.CUresult) and err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err}")
    return ret


def get_sm_alignment(dev: torch.device) -> Tuple[int, int]:
    """Return (min_sm_count, sm_alignment) for the device's compute capability.

    CC 7.x: (2, 2)
    CC 8.x: (4, 2)  -- note: min=4 but alignment=2 on some 8.x
    CC 9.0+: (8, 8)
    """
    props = torch.cuda.get_device_properties(dev)
    major = props.major
    if major == 7:
        return (2, 2)
    elif major == 8:
        return (4, 2)
    elif major >= 9:
        return (8, 8)
    else:
        raise ValueError(f"Green Context requires CC 7.0+, got {major}.{props.minor}")


def get_total_sm_count(dev: torch.device) -> int:
    """Return total SM count for the device."""
    return torch.cuda.get_device_properties(dev).multi_processor_count


def _round_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _get_cudevice(dev: torch.device) -> CUdevice:
    """Get CUdevice handle, initializing if needed."""
    idx = dev.index if dev.index is not None else 0
    try:
        return _check_cuda(driver.cuDeviceGet(idx))
    except RuntimeError:
        _check_cuda(driver.cuInit(0))
        return _check_cuda(driver.cuDeviceGet(idx))


def _split_and_create_gc(cu_dev, resource, sm_count):
    """Split `sm_count` SMs from resource, create Green Context + stream.

    Returns: (green_ctx, cu_stream, remaining_resource, actual_sm_count)
    """
    results, _, remaining = _check_cuda(
        driver.cuDevSmResourceSplitByCount(1, resource, 0, sm_count)
    )

    desc = _check_cuda(driver.cuDevResourceGenerateDesc(results, 1))
    green_ctx = _check_cuda(
        driver.cuGreenCtxCreate(
            desc, cu_dev,
            driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
        )
    )
    cu_stream = _check_cuda(
        driver.cuGreenCtxStreamCreate(
            green_ctx,
            driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
            0,
        )
    )
    return green_ctx, cu_stream, remaining, sm_count


def _remaining_to_resource(cu_dev, remaining):
    """Convert a remaining CUdevResource into a splittable resource.

    The remaining from cuDevSmResourceSplitByCount cannot be directly
    re-split. We must: create descriptor -> create Green Context ->
    query its SM resource. This gives us a fresh resource handle
    that can be split again.

    Returns: (green_ctx_for_remaining, new_resource)
    """
    desc = _check_cuda(driver.cuDevResourceGenerateDesc([remaining], 1))
    gc = _check_cuda(
        driver.cuGreenCtxCreate(
            desc, cu_dev,
            driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
        )
    )
    new_resource = _check_cuda(
        driver.cuGreenCtxGetDevResource(
            gc, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )
    return gc, new_resource


# =========================================================================
# Public API
# =========================================================================

def create_sm_partition(dev: torch.device, sm_count: int) -> Tuple[torch.cuda.Stream, int]:
    """Create a single SM partition from the full device resource.

    WARNING: Multiple calls to this function produce OVERLAPPING partitions
    because each call splits from the full device. Use create_disjoint_partitions()
    if you need multiple non-overlapping partitions.

    Returns: (stream, actual_sm_count)
    """
    min_sm, alignment = get_sm_alignment(dev)
    rounded = _round_up(max(sm_count, min_sm), alignment)
    total = get_total_sm_count(dev)
    if rounded > total:
        raise ValueError(
            f"Requested {sm_count} SMs (rounded to {rounded}) exceeds "
            f"device total of {total} SMs"
        )

    cu_dev = _get_cudevice(dev)
    resource = _check_cuda(
        driver.cuDeviceGetDevResource(
            cu_dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )

    _, cu_stream, _, actual = _split_and_create_gc(cu_dev, resource, rounded)
    torch_stream = torch.cuda.ExternalStream(cu_stream, device=dev)
    return torch_stream, actual


def create_disjoint_partitions(
    dev: torch.device, sm_counts: List[int]
) -> List[Tuple[torch.cuda.Stream, int]]:
    """Create multiple SM partitions with GUARANTEED non-overlapping SMs.

    Uses chain-split: each partition is carved from the remaining resource
    of the previous split, ensuring disjoint SM sets.

    Args:
        dev: Target CUDA device.
        sm_counts: List of desired SM counts. Each will be rounded up
                   to meet alignment. Sum must not exceed total SMs.

    Returns:
        List of (stream, actual_sm_count) tuples. The SM sets backing
        these streams are guaranteed to be disjoint.

    Raises:
        ValueError: If total requested SMs exceed device capacity.

    Example:
        >>> parts = create_disjoint_partitions(torch.device("cuda:0"), [50, 50])
        >>> stream_a, sms_a = parts[0]  # 50 SMs
        >>> stream_b, sms_b = parts[1]  # 50 different SMs, no overlap
    """
    if not sm_counts:
        return []

    min_sm, alignment = get_sm_alignment(dev)
    total = get_total_sm_count(dev)

    # Pre-validate: round up all counts and check total
    rounded_counts = [_round_up(max(sc, min_sm), alignment) for sc in sm_counts]
    total_requested = sum(rounded_counts)
    if total_requested > total:
        raise ValueError(
            f"Total requested SMs {total_requested} (after rounding {sm_counts} -> "
            f"{rounded_counts}) exceeds device total of {total} SMs"
        )

    cu_dev = _get_cudevice(dev)

    # Start from full device resource
    current_resource = _check_cuda(
        driver.cuDeviceGetDevResource(
            cu_dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )

    partitions = []
    # We keep track of intermediate GCs created for remaining resources
    # so they don't get garbage collected while we still need the resource.
    _intermediate_gcs = []

    for i, rounded in enumerate(rounded_counts):
        gc, cu_stream, remaining, actual = _split_and_create_gc(
            cu_dev, current_resource, rounded
        )
        torch_stream = torch.cuda.ExternalStream(cu_stream, device=dev)
        partitions.append((torch_stream, actual))

        # If more partitions to create, convert remaining into a splittable resource
        if i < len(rounded_counts) - 1:
            intermediate_gc, current_resource = _remaining_to_resource(cu_dev, remaining)
            _intermediate_gcs.append(intermediate_gc)

    return partitions


if __name__ == "__main__":
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(dev)}")
    print(f"Total SMs: {get_total_sm_count(dev)}")
    min_sm, align = get_sm_alignment(dev)
    print(f"SM alignment: min={min_sm}, alignment={align}")

    try:
        stream, actual = create_sm_partition(dev, 16)
        print(f"\nSingle partition: requested=16, actual={actual}")
        with torch.cuda.stream(stream):
            a = torch.randn(512, 512, device=dev, dtype=torch.float16)
            c = a @ a
            torch.cuda.current_stream().synchronize()
        print(f"  Matmul OK, shape={c.shape}")
    except Exception as e:
        print(f"Single partition failed: {e}")

    try:
        parts = create_disjoint_partitions(dev, [32, 32])
        print(f"\nDisjoint partitions: requested=[32, 32]")
        for i, (s, n) in enumerate(parts):
            print(f"  Partition {i}: {n} SMs")
            with torch.cuda.stream(s):
                a = torch.randn(512, 512, device=dev, dtype=torch.float16)
                c = a @ a
                torch.cuda.current_stream().synchronize()
            print(f"    Matmul OK")
        print("Disjoint partitions created successfully!")
    except Exception as e:
        print(f"Disjoint partitions failed: {e}")
