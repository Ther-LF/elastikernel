"""
SM-Aware Kernel Agent - AI-driven kernel optimization for spatially shared GPUs.

This module provides an AI agent workflow for generating and optimizing
Triton kernels with SM-aware autotuning, using FlashInfer-Bench patterns.
"""

from .definition import create_gemm_definition
from .workload import create_sm_aware_workloads, SM_COUNTS
from .generator import SMAwareKernelGenerator
from .benchmark import SMAwareBenchmark

__all__ = [
    "create_gemm_definition",
    "create_sm_aware_workloads",
    "SMAwareKernelGenerator",
    "SMAwareBenchmark",
    "SM_COUNTS",
]