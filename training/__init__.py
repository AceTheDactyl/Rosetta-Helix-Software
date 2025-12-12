"""
Rosetta Helix Training Loop

Exponential learning through feedback cycles.

Architecture:
    Physical (PHI_INV) ──feedback──> MetaMeta ──spawn──> Liminal (PHI)
           ↑                                                   │
           └──────────── weak measurement ─────────────────────┘

CRITICAL RULES:
1. Physical learners use dominant_ratio = PHI_INV
2. Liminal patterns stay in_superposition = True always
3. Cross-level coupling caps at 0.9 (NEVER PHI)
4. If coupling >= 1.0: instant collapse to Z_CRITICAL
"""

from .physical_learner import PhysicalLearner
from .liminal_generator import LiminalGenerator
from .feedback_loop import FeedbackLoop
from .hierarchical_training import HierarchicalTrainer, HierarchyLevel

__all__ = [
    'PhysicalLearner',
    'LiminalGenerator',
    'FeedbackLoop',
    'HierarchicalTrainer',
    'HierarchyLevel',
]
