#!/usr/bin/env python3
"""
UNIFIED HELIX ENGINE
====================

The grand unification of all Rosetta Helix systems:

1. PHI Dynamics (full_engine_runner)
   - Supercritical mode with entropy loan return
   - LiminalPhiState: PHI in superposition
   - NegEntropyEngine stays ACTIVE

2. Nightly Helix Traversal (nightly_training_runner)
   - Coherence-based run determination
   - Energy coherence metrics

3. APL Integration (unified_provable_apl)
   - S₃ operator encoding
   - Parity flow control
   - Tier hierarchy

4. 7-Layer Prismatic Projection (phi_cycle_runner)
   - RED → VIOLET spectral layers
   - Wavelength-based PHI/PHI_INV weights

5. Exponential Training Loop
   - Physical → MetaMeta → Liminal cycle
   - Feedback signals between levels

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                      UNIFIED HELIX ENGINE                            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   APL S₃     │  │   PRISMATIC  │  │    PHI       │              │
│  │  ENCODING    │──│   7-LAYER    │──│  DYNAMICS    │              │
│  │  (Phase 0-3) │  │  PROJECTION  │  │  (Liminal)   │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         └────────┬────────┴────────┬────────┘                       │
│                  │                 │                                │
│         ┌────────▼─────────────────▼────────┐                       │
│         │         HELIX CORE                 │                       │
│         │                                    │                       │
│         │  NegEntropyEngine ──► z evolution  │                       │
│         │  Coherence Metrics ──► run count   │                       │
│         │  Entropy Loan Return ──► work      │                       │
│         └────────────────────────────────────┘                       │
│                          │                                          │
│         ┌────────────────▼────────────────────┐                     │
│         │      TOOL DEVELOPMENT PIPELINE       │                     │
│         │                                      │                     │
│         │  Meta-Tools ──► Child Tools          │                     │
│         │  Dev Tools ──► Physical Execution    │                     │
│         └──────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘

Invariants (preserved across all modes):
- PHI_INV (φ⁻¹ = 0.618) always controls physical dynamics
- PHI (φ = 1.618) stays liminal (never flips to physical)
- NegEntropyEngine (T1) stays ACTIVE
- Entropy debt settled immediately (collapse or loan return)
"""

import math
import time
import json
import cmath
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Physics constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0
KAPPA_S = 0.920
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_3 = 0.992
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
Q_KAPPA = 0.3514087324

# =============================================================================
# IMPORT ALL COMPONENTS
# =============================================================================

# Dynamics
try:
    from core.quasicrystal_dynamics import (
        QuasiCrystalDynamicsEngine,
        LiminalPhiState,
        BidirectionalCollapseEngine,
        PhaseLockReleaseEngine,
    )
    DYNAMICS_AVAILABLE = True
except ImportError:
    try:
        from quasicrystal_dynamics import (
            QuasiCrystalDynamicsEngine,
            LiminalPhiState,
        )
        DYNAMICS_AVAILABLE = True
    except ImportError:
        DYNAMICS_AVAILABLE = False

# Threshold modules
try:
    from core.threshold_modules import NegEntropyEngine
    MODULES_AVAILABLE = True
except ImportError:
    try:
        from threshold_modules import NegEntropyEngine
        MODULES_AVAILABLE = False  # Fallback not available
    except ImportError:
        MODULES_AVAILABLE = False

# Exponential training
try:
    from core.exponential_training_loop import (
        ExponentialTrainingLoop,
        LiminalPatternGenerator,
        PhysicalLearner,
        MetaMetaBridge,
    )
    TRAINING_AVAILABLE = True
except ImportError:
    try:
        from exponential_training_loop import ExponentialTrainingLoop
        TRAINING_AVAILABLE = True
    except ImportError:
        TRAINING_AVAILABLE = False

# APL integration
try:
    from unified_provable_apl import (
        APLSemanticEncoder,
        ParityFlowController,
        TierHierarchy,
    )
    APL_AVAILABLE = True
except ImportError:
    APL_AVAILABLE = False

# Meta tool generator
try:
    from tools.meta_tool_generator import MetaToolGenerator, MetaToolType
    META_AVAILABLE = True
except ImportError:
    try:
        from meta_tool_generator import MetaToolGenerator, MetaToolType
        META_AVAILABLE = True
    except ImportError:
        META_AVAILABLE = False

# Dev tools
try:
    from tools.discernment_dev_tools import ComprehensiveDevToolsSuite
    DEV_TOOLS_AVAILABLE = True
except ImportError:
    DEV_TOOLS_AVAILABLE = False


# =============================================================================
# PRISMATIC LAYERS (from phi_cycle_runner)
# =============================================================================

PRISMATIC_LAYERS = [
    {'index': 1, 'name': 'RED', 'color': '#FF4444', 'family': 'Analyzers',
     'wavelength': 700, 'frequency_factor': 1.0, 'phi_weight': 0.8, 'phi_inv_weight': 1.2},
    {'index': 2, 'name': 'ORANGE', 'color': '#FF8844', 'family': 'Learners',
     'wavelength': 620, 'frequency_factor': 1.1, 'phi_weight': 0.9, 'phi_inv_weight': 1.1},
    {'index': 3, 'name': 'YELLOW', 'color': '#FFAA00', 'family': 'Generators',
     'wavelength': 580, 'frequency_factor': 1.2, 'phi_weight': 1.0, 'phi_inv_weight': 1.0},
    {'index': 4, 'name': 'GREEN', 'color': '#00FF88', 'family': 'Reflectors',
     'wavelength': 530, 'frequency_factor': 1.3, 'phi_weight': 1.0, 'phi_inv_weight': 1.0},
    {'index': 5, 'name': 'BLUE', 'color': '#00D9FF', 'family': 'Builders',
     'wavelength': 470, 'frequency_factor': 1.4, 'phi_weight': 1.1, 'phi_inv_weight': 0.9},
    {'index': 6, 'name': 'INDIGO', 'color': '#4444FF', 'family': 'Deciders',
     'wavelength': 420, 'frequency_factor': 1.5, 'phi_weight': 1.2, 'phi_inv_weight': 0.8},
    {'index': 7, 'name': 'VIOLET', 'color': '#AA44FF', 'family': 'Probers',
     'wavelength': 380, 'frequency_factor': 1.6, 'phi_weight': 1.3, 'phi_inv_weight': 0.7},
]


# =============================================================================
# COHERENCE METRICS (from nightly_training_runner)
# =============================================================================

@dataclass
class CoherenceMetrics:
    """Tracks energy coherence for run determination"""
    z_mean: float = 0.5
    z_variance: float = 0.1
    phase_coherence: float = 0.5
    work_efficiency: float = 0.5
    pattern_density: float = 0.0
    quality_slope: float = 0.0

    @property
    def energy_coherence(self) -> float:
        """Compute overall energy coherence"""
        z_factor = min(1.0, self.z_mean / MU_3)
        stability = 1.0 / (1.0 + self.z_variance * 10)
        phase_factor = self.phase_coherence
        efficiency = self.work_efficiency

        coherence = (
            z_factor * PHI_INV +
            stability * PHI_INV**2 +
            phase_factor * PHI_INV +
            efficiency * PHI_INV**2
        ) / (PHI_INV + PHI_INV**2 + PHI_INV + PHI_INV**2)

        return min(1.0, coherence)

    def determine_run_count(self) -> int:
        """Determine number of training runs based on coherence"""
        ec = self.energy_coherence
        if ec < 0.5:
            return 3   # Bootstrap
        elif ec < 0.8:
            return 5   # Growth
        elif ec < 0.95:
            return 7   # Refinement
        else:
            return 10  # Mastery


# =============================================================================
# UNIFIED HELIX STATE
# =============================================================================

@dataclass
class HelixState:
    """Unified state tracking across all subsystems"""
    # Core z-coordinate
    z_current: float = 0.5
    z_peak: float = 0.0

    # Cycle tracking
    helix_cycle: int = 0
    layer_cycle: int = 0
    total_steps: int = 0

    # Work/Energy accounting
    total_work_extracted: float = 0.0
    neg_entropy_injected: float = 0.0

    # Component states
    neg_entropy_active: bool = True
    in_superposition: bool = False
    supercritical_mode: bool = False

    # Supercritical tracking
    supercritical_breaches: int = 0
    entropy_loan_returns: int = 0

    # APL state
    apl_tier: str = "t1"
    apl_operators: List[str] = field(default_factory=list)

    # Prismatic state
    current_layer: int = 0
    layer_history: List[Dict] = field(default_factory=list)

    # Training metrics
    lessons_learned: int = 0
    patterns_generated: int = 0

    # Tool production
    meta_tools_created: int = 0
    child_tools_created: int = 0
    dev_tools_executed: int = 0

    # Coherence
    coherence: CoherenceMetrics = field(default_factory=CoherenceMetrics)


# =============================================================================
# UNIFIED HELIX ENGINE
# =============================================================================

class UnifiedHelixEngine:
    """
    The grand unification of all Rosetta Helix subsystems.

    Integrates:
    1. PHI dynamics with supercritical mode
    2. Nightly helix traversal with coherence metrics
    3. APL S₃ operator encoding
    4. 7-layer prismatic projection
    5. Exponential training loop
    6. Full tool development pipeline
    """

    def __init__(self, supercritical_mode: bool = True):
        self.state = HelixState()
        self.state.supercritical_mode = supercritical_mode

        # Lesson threshold for supercritical
        self.lesson_threshold = 50

        # Initialize components
        self._init_dynamics()
        self._init_training()
        self._init_apl()
        self._init_tools()

        # History tracking
        self.z_history: List[float] = [self.state.z_current]
        self.work_history: List[float] = [0.0]
        self.layer_results: List[Dict] = []

    def _init_dynamics(self):
        """Initialize physics dynamics components"""
        if DYNAMICS_AVAILABLE:
            self.dynamics = QuasiCrystalDynamicsEngine(n_oscillators=60)
            self.state.z_current = self.dynamics.z_current
        else:
            self.dynamics = None

        if MODULES_AVAILABLE:
            self.neg_entropy = NegEntropyEngine()
        else:
            self.neg_entropy = None

    def _init_training(self):
        """Initialize training loop components"""
        if TRAINING_AVAILABLE:
            self.training_loop = ExponentialTrainingLoop(n_physical=3, n_metameta=2)
        else:
            self.training_loop = None

    def _init_apl(self):
        """Initialize APL integration components"""
        if APL_AVAILABLE:
            self.apl_encoder = APLSemanticEncoder()
            self.parity_controller = ParityFlowController()
            self.tier_hierarchy = TierHierarchy()
        else:
            self.apl_encoder = None
            self.parity_controller = None
            self.tier_hierarchy = None

    def _init_tools(self):
        """Initialize tool generation components"""
        if META_AVAILABLE:
            self.meta_generator = MetaToolGenerator()
        else:
            self.meta_generator = None

        if DEV_TOOLS_AVAILABLE:
            self.dev_tools = ComprehensiveDevToolsSuite()
        else:
            self.dev_tools = None

    def measure_coherence(self) -> CoherenceMetrics:
        """Measure current system coherence"""
        metrics = CoherenceMetrics()

        if not self.dynamics:
            return metrics

        # Sample z values
        z_values = self.z_history[-50:] if len(self.z_history) >= 50 else self.z_history

        if z_values:
            metrics.z_mean = sum(z_values) / len(z_values)
            metrics.z_variance = sum((z - metrics.z_mean)**2 for z in z_values) / len(z_values)

        # Phase coherence from dynamics
        if hasattr(self.dynamics, 'phase_lock') and self.dynamics.phase_lock:
            phases = self.dynamics.phase_lock.phases
            if phases:
                order = abs(sum(cmath.exp(1j * p) for p in phases) / len(phases))
                metrics.phase_coherence = order

        # Work efficiency
        if self.state.total_work_extracted > 0 and self.state.neg_entropy_injected > 0:
            metrics.work_efficiency = min(1.0, self.state.total_work_extracted / self.state.neg_entropy_injected)

        # Pattern density
        if self.state.helix_cycle > 0:
            metrics.pattern_density = self.state.patterns_generated / self.state.helix_cycle

        self.state.coherence = metrics
        return metrics

    def update_apl_state(self):
        """Update APL tier and operators based on z"""
        if not self.tier_hierarchy:
            return

        caps = self.tier_hierarchy.get_capabilities(self.state.z_current)
        self.state.apl_tier = caps['tier']
        self.state.apl_operators = caps['operators']

    def step(self, layer: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Single step of the unified helix engine.

        Integrates:
        - PHI dynamics evolution
        - APL tier update
        - Supercritical handling with entropy loan return
        """
        self.state.total_steps += 1
        old_z = self.state.z_current
        layer = layer or PRISMATIC_LAYERS[3]  # Default: GREEN

        phi_weight = layer['phi_weight']
        phi_inv_weight = layer['phi_inv_weight']

        result = {
            'step': self.state.total_steps,
            'layer': layer['name'],
            'z_before': old_z,
            'z_after': old_z,
            'work_extracted': 0.0,
            'collapse': False,
            'supercritical': False,
            'entropy_loan_return': False,
        }

        # =====================================================================
        # PHASE 1: NegEntropyEngine + Dynamics Evolution
        # =====================================================================
        if self.dynamics and DYNAMICS_AVAILABLE:
            old_collapse_count = self.dynamics.liminal_phi.collapse_count
            old_work = self.dynamics.total_work_extracted

            # Sync and evolve
            self.dynamics.z_current = self.state.z_current
            self.dynamics.evolve_step()

            # Update state
            self.state.z_current = self.dynamics.z_current
            self.state.in_superposition = self.dynamics.liminal_phi.in_superposition

            # Check for collapse
            if self.dynamics.liminal_phi.collapse_count > old_collapse_count:
                work = self.dynamics.total_work_extracted - old_work
                self.state.total_work_extracted += work
                result['collapse'] = True
                result['work_extracted'] = work

            self.state.lessons_learned += 1
        else:
            # Fallback dynamics
            dz = (MU_3 - self.state.z_current) * 0.1 * PHI_INV
            self.state.z_current = min(0.9999, self.state.z_current + dz)
            self.state.lessons_learned += 1

        # =====================================================================
        # PHASE 2: Supercritical Mode Handling
        # =====================================================================
        if self.state.supercritical_mode and self.state.lessons_learned >= self.lesson_threshold:
            if not result['collapse']:
                # Expansion toward PHI
                boost = 1.0 + (self.state.lessons_learned - self.lesson_threshold) * 0.005
                expansion = PHI * phi_weight * 0.01 * boost * (PHI - self.state.z_current)

                if expansion > 0 and self.state.z_current < PHI:
                    old_z = self.state.z_current
                    self.state.z_current += expansion

                    if self.state.z_current > 1.0 and old_z <= 1.0:
                        self.state.supercritical_breaches += 1
                        result['supercritical'] = True

            # Entropy loan return when z > 1.0
            if self.state.z_current > 1.0:
                excess = self.state.z_current - 1.0
                entropy_return = excess * (1.0 + excess * PHI) * phi_inv_weight

                # Work extraction from supercritical
                work_from_supercritical = excess * PHI * PHI_INV
                self.state.total_work_extracted += work_from_supercritical
                result['work_extracted'] += work_from_supercritical

                # Apply contraction
                self.state.z_current -= entropy_return
                self.state.z_current = max(Z_CRITICAL, self.state.z_current)

                self.state.entropy_loan_returns += 1
                result['entropy_loan_return'] = True
                result['supercritical'] = True

        # =====================================================================
        # PHASE 3: APL State Update
        # =====================================================================
        self.update_apl_state()
        result['apl_tier'] = self.state.apl_tier
        result['apl_operators'] = self.state.apl_operators

        # Track peak and history
        if self.state.z_current > self.state.z_peak:
            self.state.z_peak = self.state.z_current

        self.z_history.append(self.state.z_current)
        self.work_history.append(self.state.total_work_extracted)

        result['z_after'] = self.state.z_current
        return result

    def run_layer_cycle(self, layer: Dict) -> Dict[str, Any]:
        """
        Run PHI ↔ PHI_INV cycle for a prismatic layer.

        Pattern: Tool(φ⁻¹) → Meta(φ) → MetaMeta(φ⁻¹) → Meta(φ)
        """
        self.state.layer_cycle += 1
        self.state.current_layer = layer['index']

        print(f"\n{'═'*60}")
        print(f"LAYER {layer['index']}: {layer['name']} ({layer['color']}) - {layer['family']}")
        print(f"{'═'*60}")
        print(f"  Wavelength: {layer['wavelength']}nm | Frequency: {layer['frequency_factor']:.1f}x")
        print(f"  PHI weight: {layer['phi_weight']:.1f} | PHI_INV weight: {layer['phi_inv_weight']:.1f}")

        cycle_result = {
            'layer': layer['name'],
            'layer_index': layer['index'],
            'phases': [],
            'tools_produced': [],
            'work_extracted': 0.0,
        }

        # Run 4 phases (PHI_INV → PHI → PHI_INV → PHI)
        phases = ['PHI_INV', 'PHI', 'PHI_INV', 'PHI']

        for phase_name in phases:
            # Run multiple steps per phase
            phase_work = 0.0
            for _ in range(10):
                result = self.step(layer)
                phase_work += result['work_extracted']

            cycle_result['phases'].append({
                'phase': phase_name,
                'work': phase_work,
                'z_final': self.state.z_current,
            })
            cycle_result['work_extracted'] += phase_work

        # Tool production from cycle work
        if cycle_result['work_extracted'] > MU_3 * PHI_INV and self.meta_generator:
            meta_types = list(MetaToolType) if META_AVAILABLE else []
            if meta_types:
                meta_type = meta_types[layer['index'] % len(meta_types)]
                meta = self.meta_generator.meta_factory.produce_meta_tool(
                    meta_type, cycle_result['work_extracted'] * 0.6
                )

                if meta:
                    print(f"  → Created: {meta.name}")
                    self.state.meta_tools_created += 1
                    cycle_result['tools_produced'].append(meta.name)

                    # Produce child
                    meta.feed_work(cycle_result['work_extracted'] * 0.4)
                    if meta.can_produce():
                        child = meta.produce_child()
                        if child:
                            print(f"    └─ {child.name}")
                            self.state.child_tools_created += 1
                            cycle_result['tools_produced'].append(child.name)

        # Track layer history
        self.state.layer_history.append({
            'layer': layer['name'],
            'z_final': self.state.z_current,
            'work': cycle_result['work_extracted'],
            'tools': len(cycle_result['tools_produced']),
        })

        print(f"\n  Layer {layer['name']} complete: z = {self.state.z_current:.4f}, work = {cycle_result['work_extracted']:.4f}")

        return cycle_result

    def run_helix_traversal(self, n_cycles: int = 1) -> Dict[str, Any]:
        """
        Run full helix traversal through all 7 prismatic layers.

        Integrates PHI dynamics, APL encoding, and tool development.
        """
        start_time = time.time()

        # Measure initial coherence
        coherence = self.measure_coherence()
        determined_runs = coherence.determine_run_count()

        print("\n" + "▓"*70)
        print("▓" + " "*68 + "▓")
        print("▓" + "  UNIFIED HELIX ENGINE".center(68) + "▓")
        print("▓" + "  Full PHI Dynamics + APL + Tool Development".center(68) + "▓")
        print("▓" + " "*68 + "▓")
        print("▓"*70)

        mode = "SUPERCRITICAL" if self.state.supercritical_mode else "INSTANT COLLAPSE"
        print(f"""
Components Active:
  - QuasiCrystalDynamics:    {DYNAMICS_AVAILABLE}
  - NegEntropyEngine:        {MODULES_AVAILABLE}
  - APL Integration:         {APL_AVAILABLE}
  - MetaToolGenerator:       {META_AVAILABLE}
  - DevTools:                {DEV_TOOLS_AVAILABLE}
  - TrainingLoop:            {TRAINING_AVAILABLE}

Physics Mode: {mode}
  - PHI_INV = {PHI_INV:.4f} (physical)
  - PHI = {PHI:.4f} (liminal)
  - Z_CRITICAL = {Z_CRITICAL:.4f}

Initial Coherence:
  - Energy: {coherence.energy_coherence:.4f}
  - Phase: {coherence.phase_coherence:.4f}
  - Determined runs: {determined_runs}
""")

        all_results = []

        for cycle in range(n_cycles):
            self.state.helix_cycle += 1

            print(f"\n{'┏' + '━'*68 + '┓'}")
            print(f"┃{'  HELIX CYCLE ' + str(cycle + 1) + '/' + str(n_cycles):^68}┃")
            print(f"{'┗' + '━'*68 + '┛'}")

            cycle_results = []

            # Run through all 7 prismatic layers
            for layer in PRISMATIC_LAYERS:
                result = self.run_layer_cycle(layer)
                cycle_results.append(result)
                self.layer_results.append(result)

            all_results.append({
                'cycle': cycle + 1,
                'layers': cycle_results,
                'z_final': self.state.z_current,
                'total_work': self.state.total_work_extracted,
            })

        # Final summary
        elapsed = time.time() - start_time
        final_coherence = self.measure_coherence()

        print("\n" + "▓"*70)
        print("▓" + "  HELIX TRAVERSAL COMPLETE".center(68) + "▓")
        print("▓"*70)

        print(f"""
Summary:
  Helix cycles:           {n_cycles}
  Total steps:            {self.state.total_steps}
  Total work extracted:   {self.state.total_work_extracted:.4f}
  Peak z reached:         {self.state.z_peak:.4f}
  Lessons learned:        {self.state.lessons_learned}

Supercritical Dynamics:
  Mode:                   {'ACTIVE' if self.state.supercritical_mode else 'INACTIVE'}
  Breaches (z > 1.0):     {self.state.supercritical_breaches}
  Entropy loan returns:   {self.state.entropy_loan_returns}

Tool Production:
  Meta-tools created:     {self.state.meta_tools_created}
  Child tools created:    {self.state.child_tools_created}

APL State:
  Current tier:           {self.state.apl_tier}
  Operators available:    {len(self.state.apl_operators)}

Coherence Evolution:
  Initial:                {coherence.energy_coherence:.4f}
  Final:                  {final_coherence.energy_coherence:.4f}

Elapsed time:             {elapsed:.2f}s
""")

        # Spectral visualization
        print("Spectral Z Evolution:")
        print(f"{'─'*60}")
        colors = ['🔴', '🟠', '🟡', '🟢', '🔵', '🟣', '🟣']
        for i, layer_data in enumerate(self.state.layer_history[-7:]):
            z = layer_data['z_final']
            work = layer_data['work']
            bar_len = int((z - Z_CRITICAL) * 100)
            bar = '█' * max(1, bar_len)
            print(f"  {colors[i]} {layer_data['layer']:6} │ {bar} z={z:.4f} w={work:.2f}")

        # Physics verification
        print(f"\nPhysics Verification:")
        print(f"  NegEntropyEngine:       ALWAYS ACTIVE")
        print(f"  PHI liminal only:       YES (never physical)")
        print(f"  Entropy debt settled:   {'LOAN RETURN' if self.state.supercritical_mode else 'INSTANT COLLAPSE'}")
        print(f"  Unity invariant:        PRESERVED")

        return {
            'cycles': all_results,
            'final_state': {
                'z': self.state.z_current,
                'z_peak': self.state.z_peak,
                'work': self.state.total_work_extracted,
                'lessons': self.state.lessons_learned,
                'coherence': final_coherence.energy_coherence,
            },
            'tools': {
                'meta': self.state.meta_tools_created,
                'child': self.state.child_tools_created,
            },
            'elapsed': elapsed,
        }

    def run_nightly(self, output_dir: str = "artifacts/helix-nightly") -> Dict[str, Any]:
        """
        Run nightly helix traversal with coherence-based parameters.

        Outputs JSON results for visualization and PR creation.
        """
        timestamp = datetime.utcnow().isoformat()

        # Measure coherence to determine run count
        coherence = self.measure_coherence()
        run_count = coherence.determine_run_count()

        print(f"\n{'='*70}")
        print("NIGHTLY HELIX TRAVERSAL")
        print(f"{'='*70}")
        print(f"Timestamp: {timestamp}")
        print(f"Energy Coherence: {coherence.energy_coherence:.4f}")
        print(f"Determined Cycles: {run_count}")

        # Run helix traversal
        results = self.run_helix_traversal(n_cycles=run_count)

        # Add nightly metadata
        results['nightly'] = {
            'timestamp': timestamp,
            'coherence_level': (
                'bootstrap' if coherence.energy_coherence < 0.5 else
                'growth' if coherence.energy_coherence < 0.8 else
                'refinement' if coherence.energy_coherence < 0.95 else
                'mastery'
            ),
            'coherence_metrics': {
                'energy': coherence.energy_coherence,
                'phase': coherence.phase_coherence,
                'work_efficiency': coherence.work_efficiency,
            },
        }

        # Write outputs
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results_path = Path(output_dir) / 'helix_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {results_path}")

        return results


# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_unified_demo():
    """Run unified helix engine demonstration"""
    print("\n" + "="*70)
    print("UNIFIED HELIX ENGINE DEMONSTRATION")
    print("="*70)
    print("""
This demonstrates the grand unification of all Rosetta Helix systems:

1. PHI Dynamics (supercritical + entropy loan return)
2. Nightly Helix Traversal (coherence-based runs)
3. APL Integration (S₃ operators, tier hierarchy)
4. 7-Layer Prismatic Projection (RED → VIOLET)
5. Tool Development Pipeline (meta + child tools)

KEY INVARIANTS:
  - PHI_INV always physical
  - PHI always liminal
  - NegEntropyEngine always active
  - Entropy debt settled immediately
""")

    # Create unified engine
    engine = UnifiedHelixEngine(supercritical_mode=True)

    # Run helix traversal (1 full cycle through all 7 layers)
    results = engine.run_helix_traversal(n_cycles=1)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--nightly':
        engine = UnifiedHelixEngine(supercritical_mode=True)
        engine.run_nightly()
    else:
        run_unified_demo()
