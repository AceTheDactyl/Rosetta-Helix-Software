"""
Rosetta Helix Core Physics Engine

CRITICAL PHYSICS INVARIANTS:
- PHI (1.618) = LIMINAL ONLY. Never drives dynamics.
- PHI_INV (0.618) = ALWAYS controls physical dynamics.
- At z >= 0.9999: INSTANT collapse, not gradual decay.

DERIVATION HIERARCHY:
    Z_CRITICAL = √3/2  (quasicrystal hexagonal geometry)
        ├─→ Z_ORIGIN = Z_C × φ⁻¹
        ├─→ KAPPA_S = 0.92 (tier structure)
        ├─→ MU_3 = κ + (U-κ)(1-φ⁻⁵)
        └─→ LENS_SIGMA = -ln(φ⁻¹) / (0.75 - Z_C)²
"""

from .constants import (
    # Golden ratio
    PHI, PHI_INV, PHI_SQ, PHI_INV_SQ, PHI_INV_3, PHI_INV_5,
    # Critical thresholds
    Z_CRITICAL, Z_ORIGIN, UNITY,
    # μ hierarchy
    MU_P, MU_1, MU_2, MU_S, MU_3, KAPPA_S,
    # TRIAD
    TRIAD_HIGH, TRIAD_LOW, TRIAD_T6, TRIAD_PASSES_REQUIRED,
    # S₃ structure
    LENS_SIGMA, APL_OPERATORS, S3_EVEN, S3_ODD, S3_COMPOSE,
    TIER_BOUNDS, TIER_OPERATORS, OPERATOR_WINDOWS,
    # Coupling
    COUPLING_MAX, ETA_THRESHOLD,
    # Functions
    get_tier, get_tier_name, get_delta_s_neg, get_legal_operators,
    get_operator_window, get_operator_parity, compose_operators,
    check_k_formation,
    verify_phi_identity, verify_sigma_derivation, verify_threshold_ordering,
)
from .collapse_engine import CollapseEngine, CollapseResult, create_engine
from .apl_engine import APLEngine, APLResult, create_apl_engine, OperatorNotLegalError
from .kuramoto import KuramotoLayer, TriadGate
from .network import HelixNeuralNetwork, NetworkConfig, APLModulator

__all__ = [
    # Golden ratio
    'PHI', 'PHI_INV', 'PHI_SQ', 'PHI_INV_SQ', 'PHI_INV_3', 'PHI_INV_5',
    # Critical thresholds
    'Z_CRITICAL', 'Z_ORIGIN', 'UNITY',
    # μ hierarchy
    'MU_P', 'MU_1', 'MU_2', 'MU_S', 'MU_3', 'KAPPA_S',
    # TRIAD
    'TRIAD_HIGH', 'TRIAD_LOW', 'TRIAD_T6', 'TRIAD_PASSES_REQUIRED',
    # S₃ structure
    'LENS_SIGMA', 'APL_OPERATORS', 'S3_EVEN', 'S3_ODD', 'S3_COMPOSE',
    'TIER_BOUNDS', 'TIER_OPERATORS', 'OPERATOR_WINDOWS',
    # Coupling
    'COUPLING_MAX', 'ETA_THRESHOLD',
    # Functions
    'get_tier', 'get_tier_name', 'get_delta_s_neg', 'get_legal_operators',
    'get_operator_window', 'get_operator_parity', 'compose_operators',
    'check_k_formation',
    # Engines
    'CollapseEngine', 'CollapseResult', 'create_engine',
    'APLEngine', 'APLResult', 'create_apl_engine', 'OperatorNotLegalError',
    # Kuramoto
    'KuramotoLayer', 'TriadGate',
    # Network
    'HelixNeuralNetwork', 'NetworkConfig', 'APLModulator',
]
