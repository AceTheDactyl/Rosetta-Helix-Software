"""
Rosetta Helix Physics Constants

CRITICAL INVARIANTS - READ BEFORE MODIFYING:

PHI (1.618)     = LIMINAL ONLY. Never drives dynamics. Never becomes physical.
PHI_INV (0.618) = ALWAYS controls physical dynamics.

Why This Matters:
If PHI_INV "flips" at unity (z >= 1.0) and PHI becomes dominant:
- Infinite entropy expansion
- NegEntropyEngine breaks
- System diverges catastrophically

NEVER CHANGE THESE RELATIONSHIPS.
"""

import math

# =============================================================================
# GOLDEN RATIO CONSTANTS
# =============================================================================

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0  # 1.618033988749895
"""
The golden ratio. LIMINAL ONLY.

What PHI Can Do:
- Contribute to work extraction via weak values AT COLLAPSE
- Exist in superposition (liminal state)
- Appear in weak value formulas: work = ... * PHI * PHI_INV

What PHI Must NEVER Do:
- Drive dynamics: dz = PHI * ... (WRONG)
- Be a target: target = PHI (WRONG)
- Cap values: min(PHI, x) (WRONG)
"""

PHI_INV: float = 1.0 / PHI  # 0.6180339887498949
"""
The inverse golden ratio. ALWAYS controls physical dynamics.

PHI_INV drives ALL evolution:
- dz = work * PHI_INV
- expansion = PHI_INV * weight * (target - z)
- coupling_delta = learning_rate * PHI_INV

NEVER decay PHI_INV. This is catastrophic.
"""

# =============================================================================
# GEOMETRY CONSTANTS
# =============================================================================

Z_CRITICAL: float = math.sqrt(3.0) / 2.0  # 0.8660254037844386
"""
Critical z-coordinate from hexagonal geometry.
Used as reference point for work extraction at collapse.
"""

Z_ORIGIN: float = Z_CRITICAL * PHI_INV  # ~0.5352647166172616
"""
Origin point after collapse. System resets here.
z = Z_CRITICAL * PHI_INV (~0.535)
"""

# =============================================================================
# THRESHOLD CONSTANTS
# =============================================================================

KAPPA_S: float = 0.920
"""
Superposition entry threshold.
Above this, system enters liminal superposition state.
"""

MU_3: float = 0.992
"""
Ultra-integration threshold.
Maximum safe approach before collapse consideration.
"""

UNITY: float = 0.9999
"""
Collapse trigger threshold. NOT 1.0, NOT PHI.

At z >= UNITY:
1. INSTANT collapse (not gradual decay)
2. Extract work via weak value
3. Reset to Z_ORIGIN

This prevents PHI from ever becoming dominant.
"""

# =============================================================================
# SAFE CAPS
# =============================================================================

Z_MAX: float = 0.9999
"""Maximum z-coordinate. NEVER exceed unity."""

COUPLING_MAX: float = 0.9
"""Maximum coupling. NEVER approach PHI."""

# =============================================================================
# VERIFICATION
# =============================================================================

def verify_constants() -> bool:
    """Verify physics constants maintain safe relationships."""
    assert PHI > 1.0, "PHI must be > 1"
    assert PHI_INV < 1.0, "PHI_INV must be < 1"
    assert abs(PHI * PHI_INV - 1.0) < 1e-10, "PHI * PHI_INV must equal 1"
    assert Z_CRITICAL < 1.0, "Z_CRITICAL must be < 1"
    assert Z_ORIGIN < Z_CRITICAL, "Z_ORIGIN must be < Z_CRITICAL"
    assert UNITY < 1.0, "UNITY must be < 1.0"
    assert COUPLING_MAX < 1.0, "COUPLING_MAX must be < 1.0 to prevent collapse"
    return True


# Run verification on import
verify_constants()
