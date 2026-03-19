"""
Green Context SM Partitioning Utility for ElastiKernel.

Creates CUDA Green Contexts to dynamically partition GPU SMs,
enabling profiling of Triton kernels under different SM counts
without restarting the process (unlike MPS).

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


def create_sm_partition(dev: torch.device, sm_count: int) -> Tuple[torch.cuda.Stream, int]:
    """Create a CUDA stream restricted to `sm_count` SMs via Green Context.

    Args:
        dev: Target CUDA device.
        sm_count: Desired number of SMs. Will be rounded up to meet
                  hardware alignment requirements.

    Returns:
        stream: A torch.cuda.Stream bound to the green context.
                Kernels launched on this stream use only the allocated SMs.
        actual_sm_count: The actual SM count after alignment rounding.

    Example:
        >>> stream, actual = create_sm_partition(torch.device("cuda:0"), 16)
        >>> with torch.cuda.stream(stream):
        ...     # All kernels here (including Triton) use only `actual` SMs
        ...     result = my_triton_kernel[grid](...)
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

    # Step 1: Get device SM resource
    resource = _check_cuda(
        driver.cuDeviceGetDevResource(
            cu_dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )

    # Step 2: Split — request 1 group with `rounded` SMs
    results, _, remaining = _check_cuda(
        driver.cuDevSmResourceSplitByCount(
            1,           # nbGroups
            resource,    # input
            0,           # flags
            rounded,     # minCount
        )
    )

    # Step 3: Generate resource descriptor
    desc = _check_cuda(driver.cuDevResourceGenerateDesc(results, 1))

    # Step 4: Create Green Context
    green_ctx = _check_cuda(
        driver.cuGreenCtxCreate(
            desc, cu_dev,
            driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
        )
    )

    # Step 5: Create stream on the Green Context
    cu_stream = _check_cuda(
        driver.cuGreenCtxStreamCreate(
            green_ctx,
            driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
            0,  # priority
        )
    )

    # Step 6: Wrap as torch.cuda.Stream
    torch_stream = torch.cuda.ExternalStream(
        cu_stream, device=dev
    )

    return torch_stream, rounded


def create_sm_partitions(
    dev: torch.device, sm_counts: List[int]
) -> List[Tuple[torch.cuda.Stream, int]]:
    """Create multiple independent SM partitions.

    Each partition gets its own Green Context with the specified SM count.
    Note: partitions may overlap in SM assignment (CUDA allows oversubscription).

    Args:
        dev: Target CUDA device.
        sm_counts: List of desired SM counts for each partition.

    Returns:
        List of (stream, actual_sm_count) tuples.
    """
    return [create_sm_partition(dev, sc) for sc in sm_counts]


if __name__ == "__main__":
    # Quick test
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(dev)}")
    print(f"Total SMs: {get_total_sm_count(dev)}")
    min_sm, align = get_sm_alignment(dev)
    print(f"SM alignment: min={min_sm}, alignment={align}")

    # Try creating a partition
    try:
        stream, actual = create_sm_partition(dev, 16)
        print(f"Created partition: requested=16, actual={actual}")
        # Quick matmul test on the partition
        with torch.cuda.stream(stream):
            a = torch.randn(512, 512, device=dev, dtype=torch.float16)
            b = torch.randn(512, 512, device=dev, dtype=torch.float16)
            c = a @ b
            torch.cuda.current_stream().synchronize()
        print(f"Matmul on green context stream OK, result shape={c.shape}")
    except Exception as e:
        print(f"Green Context not available: {e}")
        print("(Requires CC 7.0+ and cuda-python installed)")
