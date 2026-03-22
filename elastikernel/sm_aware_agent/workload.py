"""
SM-Aware Workload Generator.

Generates workloads with different SM counts for benchmarking.
"""

import json
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass

# Default SM counts based on common GPU configurations
SM_COUNTS = [8, 16, 24, 32, 48, 64, 80, 96, 108, 132]

# Matrix sizes for GEMM benchmarking
GEMM_SIZES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    # Non-square matrices
    (4096, 14336, 4096),  # LLM FFN up projection
    (14336, 4096, 4096),  # LLM FFN down projection
    (1, 4096, 4096),      # Decode phase (M=1)
    (128, 4096, 4096),    # Small batch
]


@dataclass
class Workload:
    """Workload specification."""
    uuid: str
    definition: str
    axes_values: Dict[str, int]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "definition": self.definition,
            "axes_values": self.axes_values,
            "description": self.description,
        }


def create_sm_aware_workloads(
    definition_name: str = "gemm_sm_aware",
    sm_counts: List[int] = None,
    matrix_sizes: List[tuple] = None,
) -> List[Workload]:
    """
    Create workloads covering SM count × matrix size matrix.

    Args:
        definition_name: Name of the definition
        sm_counts: List of SM counts to test
        matrix_sizes: List of (M, N, K) tuples

    Returns:
        List of Workload objects
    """
    if sm_counts is None:
        sm_counts = SM_COUNTS
    if matrix_sizes is None:
        matrix_sizes = GEMM_SIZES

    workloads = []
    for sm in sm_counts:
        for M, N, K in matrix_sizes:
            wl = Workload(
                uuid=str(uuid.uuid4()),
                definition=definition_name,
                axes_values={
                    "M": M,
                    "N": N,
                    "K": K,
                    "NUM_SMS": sm,
                },
                description=f"GEMM {M}x{N}x{K} with {sm} SMs",
            )
            workloads.append(wl)

    return workloads


def compute_wave_efficiency(M: int, N: int, block_m: int, block_n: int, num_sms: int) -> Dict[str, float]:
    """
    Compute wave efficiency metrics for a given configuration.

    Returns:
        Dict with num_blocks, num_waves, last_wave_util, overall_efficiency
    """
    num_blocks = ((M + block_m - 1) // block_m) * ((N + block_n - 1) // block_n)
    num_waves = (num_blocks + num_sms - 1) // num_sms
    last_wave_blocks = num_blocks % num_sms
    if last_wave_blocks == 0:
        last_wave_blocks = num_sms

    last_wave_util = last_wave_blocks / num_sms
    overall_efficiency = num_blocks / (num_waves * num_sms) if num_waves > 0 else 0

    return {
        "num_blocks": num_blocks,
        "num_waves": num_waves,
        "last_wave_util": last_wave_util,
        "overall_efficiency": overall_efficiency,
    }


def suggest_optimal_block_size(
    M: int, N: int, num_sms: int,
    block_sizes: List[tuple] = None,
) -> tuple:
    """
    Suggest optimal block size for given matrix dimensions and SM count.

    This is a heuristic based on wave efficiency maximization.

    Args:
        M, N: Matrix dimensions
        num_sms: SM count
        block_sizes: List of (BLOCK_M, BLOCK_N) candidates

    Returns:
        (block_m, block_n, wave_efficiency) tuple
    """
    if block_sizes is None:
        block_sizes = [
            (16, 64), (32, 64), (32, 128), (64, 64),
            (64, 128), (128, 64), (128, 128), (128, 256),
            (256, 128), (64, 256), (256, 64),
        ]

    best = None
    best_eff = 0

    for bm, bn in block_sizes:
        metrics = compute_wave_efficiency(M, N, bm, bn, num_sms)
        if metrics["overall_efficiency"] > best_eff:
            best_eff = metrics["overall_efficiency"]
            best = (bm, bn, metrics)

    return best


def generate_wave_analysis_report(
    matrix_sizes: List[tuple] = None,
    sm_counts: List[int] = None,
) -> str:
    """
    Generate a wave analysis report for SM-aware optimization guidance.

    Returns:
        Markdown formatted report string
    """
    if matrix_sizes is None:
        matrix_sizes = GEMM_SIZES
    if sm_counts is None:
        sm_counts = SM_COUNTS

    lines = ["# Wave Efficiency Analysis\n"]
    lines.append("This report analyzes wave quantization effects for different SM counts.\n")

    for M, N, K in matrix_sizes:
        lines.append(f"\n## Matrix {M}x{N}x{K}\n")
        lines.append("| SMs | Best Block | Waves | Last Wave Util | Efficiency |")
        lines.append("|-----|------------|-------|----------------|------------|")

        for sm in sm_counts:
            best = suggest_optimal_block_size(M, N, sm)
            if best:
                bm, bn, metrics = best
                lines.append(
                    f"| {sm} | {bm}x{bn} | {metrics['num_waves']} | "
                    f"{metrics['last_wave_util']*100:.1f}% | {metrics['overall_efficiency']*100:.1f}% |"
                )

    return "\n".join(lines)


if __name__ == "__main__":
    # Generate workloads
    workloads = create_sm_aware_workloads()
    print(f"Generated {len(workloads)} workloads")
    print(f"Example: {workloads[0].to_dict()}")

    # Generate wave analysis
    report = generate_wave_analysis_report(matrix_sizes=[(1024, 1024, 1024), (4096, 4096, 4096)])
    print("\n" + report)