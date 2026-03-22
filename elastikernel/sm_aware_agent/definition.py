"""
SM-Aware Kernel Definitions.

Extends FlashInfer-Bench Definition schema with NUM_SMS axis.
"""

from typing import Dict, Any
import json


# Default SM counts for testing (based on H100=132, A100=108, H20=78)
SM_COUNTS = [8, 16, 24, 32, 48, 64, 80, 96, 108, 132]


def create_gemm_definition(
    name: str = "gemm_sm_aware",
    dtype: str = "float16",
) -> Dict[str, Any]:
    """
    Create a GEMM Definition with NUM_SMS axis.

    This Definition extends standard GEMM with an SM count dimension,
    enabling kernels to optimize for different SM partition sizes.

    Args:
        name: Definition name
        dtype: Data type for computation

    Returns:
        Definition dict compatible with FlashInfer-Bench
    """
    definition = {
        "name": name,
        "op_type": "gemm",
        "axes": {
            "M": {"type": "var", "description": "Number of rows in A and C"},
            "N": {"type": "var", "description": "Number of columns in B and C"},
            "K": {"type": "var", "description": "Number of columns in A / rows in B"},
            "NUM_SMS": {"type": "var", "description": "SM partition size for wave-aware optimization"},
        },
        "inputs": {
            "A": {
                "shape": ["M", "K"],
                "dtype": dtype,
                "description": "Left operand matrix",
            },
            "B": {
                "shape": ["K", "N"],
                "dtype": dtype,
                "description": "Right operand matrix",
            },
        },
        "outputs": {
            "C": {
                "shape": ["M", "N"],
                "dtype": dtype,
                "description": "Output matrix",
            },
        },
        "reference": '''
import torch

def run(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, NUM_SMS: int) -> None:
    """Reference implementation: C = A @ B"""
    C.copy_(torch.matmul(A, B))
''',
        "constraints": [
            "M > 0",
            "N > 0",
            "K > 0",
            "NUM_SMS > 0",
        ],
        "description": "General Matrix Multiplication with SM-aware optimization",
        "tags": ["gemm", "sm-aware", "triton"],
    }
    return definition


def create_fused_moe_definition(
    name: str = "fused_moe_sm_aware",
    num_experts: int = 8,
    top_k: int = 2,
    dtype: str = "float16",
) -> Dict[str, Any]:
    """
    Create a Fused MoE Definition with NUM_SMS axis.

    MoE kernels are particularly sensitive to wave quantization due to
    irregular grid sizes from dynamic token-to-expert routing.

    Args:
        name: Definition name
        num_experts: Number of experts
        top_k: Top-k routing
        dtype: Data type

    Returns:
        Definition dict
    """
    definition = {
        "name": name,
        "op_type": "moe",
        "axes": {
            "M": {"type": "var", "description": "Number of tokens"},
            "K": {"type": "var", "description": "Input hidden dimension"},
            "N": {"type": "var", "description": "Intermediate dimension (FFN)"},
            "NUM_EXPERTS": {"type": "const", "value": num_experts},
            "TOP_K": {"type": "const", "value": top_k},
            "NUM_SMS": {"type": "var", "description": "SM partition size"},
        },
        "inputs": {
            "hidden_states": {
                "shape": ["M", "K"],
                "dtype": dtype,
                "description": "Input token embeddings",
            },
            "gate_logits": {
                "shape": ["M", "NUM_EXPERTS"],
                "dtype": "float32",
                "description": "Router logits",
            },
            "w1": {
                "shape": ["NUM_EXPERTS", "N", "K"],
                "dtype": dtype,
                "description": "Up projection weights",
            },
            "w2": {
                "shape": ["NUM_EXPERTS", "K", "N"],
                "dtype": dtype,
                "description": "Down projection weights",
            },
            "w3": {
                "shape": ["NUM_EXPERTS", "N", "K"],
                "dtype": dtype,
                "description": "Gate projection weights",
            },
        },
        "outputs": {
            "out": {
                "shape": ["M", "K"],
                "dtype": dtype,
                "description": "Output after MoE",
            },
        },
        "reference": '''
import torch
import torch.nn.functional as F

def run(hidden_states, gate_logits, w1, w2, w3, out, M, K, N, NUM_EXPERTS, TOP_K, NUM_SMS):
    """Reference: MoE forward pass"""
    # Get top-k experts
    topk_weights, topk_ids = torch.topk(gate_logits.softmax(dim=-1), TOP_K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Simple loop over tokens (not optimized)
    result = torch.zeros_like(hidden_states)
    for i in range(M):
        for j in range(TOP_K):
            expert_id = topk_ids[i, j].item()
            weight = topk_weights[i, j].item()

            h = hidden_states[i]  # [K]
            up = F.silu(w1[expert_id] @ h)  # [N]
            gate = w3[expert_id] @ h  # [N]
            hidden = up * gate  # [N]
            out_i = w2[expert_id] @ hidden  # [K]

            result[i] += weight * out_i

    out.copy_(result)
''',
        "description": "Fused MoE with SM-aware optimization for irregular grid patterns",
        "tags": ["moe", "sm-aware", "triton"],
    }
    return definition


if __name__ == "__main__":
    import json
    print("GEMM Definition:")
    print(json.dumps(create_gemm_definition(), indent=2))