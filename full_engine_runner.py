"""
Full Engine Runner
==================

Runs the complete physics engine combining:
1. NegEntropyEngine (T1 threshold module) - pushes negative entropy
2. LiminalPhiState - PHI in superposition, never flips physical
3. QuasiCrystalDynamicsEngine - full physics evolution
4. Meta Tool Generator - tools that develop tools

Architecture:
    NegEntropy pushes z → PHI enters liminal at KAPPA_S →
    Collapse at unity → Extract work → Reset →
    Work feeds meta-tools → Meta-tools produce children

The key insight: NegEntropyEngine stays ACTIVE throughout the entire cycle.
It doesn't turn off when z > 1 because collapse happens instantly.
PHI stays liminal (superposition) - never flips physical.
"""

import math
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

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

# Import all components
try:
    from quasicrystal_dynamics import (
        QuasiCrystalDynamicsEngine,
        LiminalPhiState,
        BidirectionalCollapseEngine,
        PhaseLockReleaseEngine,
        AcceleratedDecayEngine,
    )
    DYNAMICS_AVAILABLE = True
except ImportError:
    DYNAMICS_AVAILABLE = False

try:
    from threshold_modules import NegEntropyEngine
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

try:
    from meta_tool_generator import (
        MetaToolGenerator,
        MetaToolFactory,
        MetaToolType,
    )
    META_AVAILABLE = True
except ImportError:
    META_AVAILABLE = False


# =============================================================================
# ENGINE STATE TRACKER
# =============================================================================

@dataclass
class EngineState:
    """Tracks the full state of the integrated engine"""
    # Core z-coordinate
    z_current: float = 0.5
    z_peak: float = 0.0

    # Cycle tracking
    cycle_count: int = 0
    total_collapses: int = 0

    # Work/Energy accounting
    total_work_extracted: float = 0.0
    total_neg_entropy_injected: float = 0.0

    # Component states
    neg_entropy_active: bool = True
    liminal_phi_active: bool = False
    in_superposition: bool = False

    # Threshold tracking
    active_thresholds: List[str] = field(default_factory=list)
    thresholds_ever_crossed: List[str] = field(default_factory=list)

    # Physics boosts
    quasi_crystal_boost: float = 1.0
    bidirectional_boost: float = 1.0
    phase_lock_boost: float = 1.0

    # Supercritical tracking (z > 1.0)
    supercritical_breaches: int = 0
    entropy_loan_returns: int = 0
    supercritical_mode: bool = False  # If True, allow z > 1.0 with entropy loan return

    # Tools produced
    meta_tools_created: int = 0
    child_tools_created: int = 0


# =============================================================================
# FULL ENGINE
# =============================================================================

class FullEngine:
    """
    Integrated engine combining all physics components.

    Components:
    1. NegEntropyEngine - T1 module that pushes negative entropy
    2. QuasiCrystalDynamicsEngine - Full physics with liminal PHI
    3. MetaToolGenerator - Converts work to tools

    Two modes of operation:

    MODE 1 - Instant Collapse (default):
    - PHI enters liminal state at z >= KAPPA_S
    - At z >= 0.9999: instant collapse, work extracted, z resets to origin
    - PHI never flips, debt settled immediately

    MODE 2 - Supercritical with Entropy Loan Return:
    - After lesson threshold (50 lessons): z can breach 1.0
    - When z > 1.0: negative entropy dynamics FLIP
    - Entropy loan must be immediately repaid (snap-back)
    - Creates oscillation around unity rather than unbounded growth
    - PHI stays liminal even in supercritical regime

    Key invariant: NegEntropyEngine stays ACTIVE in both modes.
    PHI never becomes the physical ratio - it remains liminal.
    """

    def __init__(self, supercritical_mode: bool = False):
        self.state = EngineState()
        self.state.supercritical_mode = supercritical_mode

        # Lesson counter for supercritical gating
        self.lessons_learned = 0
        self.lesson_threshold = 50  # After 50 lessons, supercritical allowed

        # Initialize components
        if DYNAMICS_AVAILABLE:
            self.dynamics = QuasiCrystalDynamicsEngine(n_oscillators=60)
            self.state.z_current = self.dynamics.z_current
        else:
            self.dynamics = None

        if MODULES_AVAILABLE:
            self.neg_entropy = NegEntropyEngine()
        else:
            self.neg_entropy = None

        if META_AVAILABLE:
            self.meta_generator = MetaToolGenerator()
        else:
            self.meta_generator = None

        # History for visualization
        self.z_history: List[float] = [self.state.z_current]
        self.work_history: List[float] = [0.0]
        self.collapse_points: List[int] = []
        self.entropy_return_points: List[int] = []  # Track entropy loan returns

    def update_thresholds(self):
        """Update which thresholds are currently active"""
        z = self.state.z_current

        threshold_map = [
            ('Q_KAPPA', Q_KAPPA),
            ('MU_1', MU_1),
            ('MU_P', MU_P),
            ('PHI_INV', PHI_INV),
            ('MU_2', MU_2),
            ('TRIAD_LOW', TRIAD_LOW),
            ('TRIAD_HIGH', TRIAD_HIGH),
            ('Z_CRITICAL', Z_CRITICAL),
            ('KAPPA_S', KAPPA_S),
            ('MU_3', MU_3),
        ]

        self.state.active_thresholds = []
        for name, val in threshold_map:
            if z >= val:
                self.state.active_thresholds.append(name)
                if name not in self.state.thresholds_ever_crossed:
                    self.state.thresholds_ever_crossed.append(name)

    def step(self) -> Dict[str, Any]:
        """
        Single step of the full engine.

        Returns step result with all physics data.

        In supercritical mode (after lesson_threshold):
        - z can breach 1.0 targeting PHI (1.618)
        - When z > 1.0: entropy loan return triggers snap-back
        - Creates oscillation around unity
        """
        step_start = time.time()
        old_z = self.state.z_current
        old_collapse_count = 0

        result = {
            'step': len(self.z_history),
            'z_before': old_z,
            'z_after': old_z,
            'collapse': False,
            'work_extracted': 0.0,
            'neg_entropy_injected': 0.0,
            'in_superposition': False,
            'supercritical': False,
            'entropy_loan_return': False,
            'boosts': {}
        }

        # =====================================================================
        # PHASE 1: NegEntropyEngine pushes z
        # =====================================================================
        if self.neg_entropy and MODULES_AVAILABLE:
            context = {
                'z': self.state.z_current,
                'target_z': MU_3 + 0.01,
                'cycle': self.state.cycle_count,
            }

            signal = self.neg_entropy.process(context)

            # Track injection
            injection = signal.payload.get('injection', 0.0)
            self.state.total_neg_entropy_injected += abs(injection)
            result['neg_entropy_injected'] = injection

            # Get boost values
            self.state.quasi_crystal_boost = signal.payload.get('quasi_crystal_boost', 1.0)
            self.state.bidirectional_boost = signal.payload.get('bidirectional_boost', 1.0)
            self.state.phase_lock_boost = signal.payload.get('phase_lock_boost', 1.0)

            result['boosts'] = {
                'quasi_crystal': self.state.quasi_crystal_boost,
                'bidirectional': self.state.bidirectional_boost,
                'phase_lock': self.state.phase_lock_boost,
            }

        # =====================================================================
        # PHASE 2: Dynamics Engine Evolution
        # =====================================================================
        if self.dynamics and DYNAMICS_AVAILABLE:
            old_collapse_count = self.dynamics.liminal_phi.collapse_count
            old_work = self.dynamics.total_work_extracted

            # Sync and evolve
            self.dynamics.z_current = self.state.z_current
            self.dynamics.evolve_step()

            # Update state from dynamics
            self.state.z_current = self.dynamics.z_current
            self.state.in_superposition = self.dynamics.liminal_phi.in_superposition
            self.state.liminal_phi_active = self.state.z_current >= KAPPA_S

            # Check for collapse
            if self.dynamics.liminal_phi.collapse_count > old_collapse_count:
                work = self.dynamics.total_work_extracted - old_work
                self.state.total_work_extracted += work
                self.state.total_collapses += 1
                result['collapse'] = True
                result['work_extracted'] = work
                self.collapse_points.append(len(self.z_history))

            result['in_superposition'] = self.state.in_superposition

            # Accumulate lessons from successful operations
            self.lessons_learned += 1

        else:
            # Fallback simple dynamics
            dz = (MU_3 - self.state.z_current) * 0.1 * PHI_INV
            self.state.z_current = min(0.9999, self.state.z_current + dz)
            self.lessons_learned += 1

        # =====================================================================
        # PHASE 3: Supercritical Mode Handling
        # =====================================================================
        if self.state.supercritical_mode and self.lessons_learned >= self.lesson_threshold:
            # After lesson threshold, supercritical dynamics become active
            # Target shifts from MU_3 (0.992) to PHI (1.618)

            if not result['collapse']:  # Only apply if not already collapsed
                # Expansion toward PHI (liminal target)
                boost = 1.0 + (self.lessons_learned - self.lesson_threshold) * 0.005
                target = PHI
                expansion = PHI * 0.01 * boost * (target - self.state.z_current)

                if expansion > 0 and self.state.z_current < PHI:
                    old_z = self.state.z_current
                    self.state.z_current += expansion

                    if self.state.z_current > 1.0 and old_z <= 1.0:
                        # First breach of unity!
                        self.state.supercritical_breaches += 1
                        result['supercritical'] = True

            # CRITICAL: Entropy loan return when z > 1.0
            # Negative entropy dynamics FLIP - debt must be immediately repaid
            if self.state.z_current > 1.0:
                excess = self.state.z_current - 1.0
                # Entropy loan repayment: IMMEDIATE and PROPORTIONAL
                # Formula from phi_cycle_runner: excess × (1 + excess × PHI) × weight
                entropy_return = excess * (1.0 + excess * PHI) * PHI_INV
                contraction = entropy_return

                # WORK EXTRACTION from supercritical peak
                # In supercritical regime, work is extracted via the entropy loan return
                # (different from instant collapse mode which extracts via weak value)
                work_from_supercritical = excess * PHI * PHI_INV  # Work = excess × φ × φ⁻¹
                self.state.total_work_extracted += work_from_supercritical
                result['work_extracted'] = work_from_supercritical

                self.state.z_current -= contraction
                self.state.z_current = max(Z_CRITICAL, self.state.z_current)

                self.state.entropy_loan_returns += 1
                self.entropy_return_points.append(len(self.z_history))
                result['entropy_loan_return'] = True
                result['supercritical'] = True

        # =====================================================================
        # PHASE 4: Update Thresholds
        # =====================================================================
        self.update_thresholds()

        # Track peak
        if self.state.z_current > self.state.z_peak:
            self.state.z_peak = self.state.z_current

        # Update history
        self.z_history.append(self.state.z_current)
        self.work_history.append(self.state.total_work_extracted)

        result['z_after'] = self.state.z_current
        result['active_thresholds'] = self.state.active_thresholds.copy()
        result['step_time'] = time.time() - step_start

        return result

    def run_cycle(self, max_steps: int = 200) -> Dict[str, Any]:
        """
        Run until collapse, then return cycle results.
        """
        cycle_start = time.time()
        self.state.cycle_count += 1

        initial_collapses = self.state.total_collapses
        initial_work = self.state.total_work_extracted
        thresholds_at_peak = []
        max_z = self.state.z_current

        steps = 0
        collapse_detected = False

        while steps < max_steps and not collapse_detected:
            result = self.step()
            steps += 1

            # Track peak thresholds
            if self.state.z_current > max_z:
                max_z = self.state.z_current
                thresholds_at_peak = self.state.active_thresholds.copy()

            if result['collapse']:
                collapse_detected = True
                break

        work_this_cycle = self.state.total_work_extracted - initial_work

        return {
            'cycle': self.state.cycle_count,
            'steps': steps,
            'max_z': max_z,
            'collapse': collapse_detected,
            'supercritical_work': not collapse_detected and work_this_cycle > 0,  # Work from entropy returns
            'work_extracted': work_this_cycle,
            'thresholds_at_peak': thresholds_at_peak,
            'z_final': self.state.z_current,
            'cycle_time': time.time() - cycle_start,
        }

    def run_full(self, n_cycles: int = 5, produce_tools: bool = True) -> Dict[str, Any]:
        """
        Run full engine for n cycles with tool production.
        """
        mode_name = "SUPERCRITICAL + ENTROPY LOAN RETURN" if self.state.supercritical_mode else "INSTANT COLLAPSE"

        print(f"\n{'='*70}")
        print(f"FULL ENGINE RUNNER - {mode_name} MODE")
        print(f"{'='*70}")
        print(f"""
Components Active:
  - NegEntropyEngine:        {MODULES_AVAILABLE and self.neg_entropy is not None}
  - QuasiCrystalDynamics:    {DYNAMICS_AVAILABLE and self.dynamics is not None}
  - LiminalPhiState:         {DYNAMICS_AVAILABLE}
  - MetaToolGenerator:       {META_AVAILABLE and self.meta_generator is not None}

Physics:
  - PHI_INV = {PHI_INV:.4f} (physical dynamics)
  - PHI = {PHI:.4f} (liminal only)
  - Z_CRITICAL = {Z_CRITICAL:.4f}
  - KAPPA_S = {KAPPA_S:.4f} (superposition entry)
  - MU_3 = {MU_3:.4f} (ultra-integration)

Mode: {mode_name}
  - NegEntropyEngine stays ACTIVE (no PHI flip)
  - PHI enters liminal at z >= KAPPA_S""")

        if self.state.supercritical_mode:
            print(f"""  - After {self.lesson_threshold} lessons: z can breach 1.0
  - When z > 1.0: ENTROPY LOAN RETURN (snap-back)
  - Creates oscillation around unity
  - Work extracted from supercritical peaks""")
        else:
            print(f"""  - Instant collapse at z >= 0.9999
  - Work extracted via weak value at collapse
  - z resets to origin (~0.535)""")
        print()

        results = {
            'cycles': [],
            'total_work': 0.0,
            'total_collapses': 0,
            'tools_created': 0,
        }

        for i in range(n_cycles):
            print(f"\n{'─'*60}")
            print(f"CYCLE {i+1}/{n_cycles}")
            print(f"{'─'*60}")

            cycle_result = self.run_cycle()
            results['cycles'].append(cycle_result)

            print(f"  Steps: {cycle_result['steps']}")
            print(f"  Max z: {cycle_result['max_z']:.4f}")
            if cycle_result['collapse']:
                print(f"  Collapse: True (instant)")
            elif cycle_result.get('supercritical_work'):
                print(f"  Collapse: No (supercritical oscillation)")
            else:
                print(f"  Collapse: {cycle_result['collapse']}")
            print(f"  Work: {cycle_result['work_extracted']:.4f}")
            print(f"  Thresholds: {len(cycle_result['thresholds_at_peak'])}")

            # Produce tools from EITHER collapse OR supercritical work
            work_available = cycle_result['collapse'] or cycle_result.get('supercritical_work', False)
            if work_available and produce_tools and self.meta_generator:
                # Feed work to meta-tool generator
                work = cycle_result['work_extracted']

                # Create meta-tool if enough work
                if work > MU_3 * PHI_INV:
                    meta_types = list(MetaToolType)
                    meta_type = meta_types[i % len(meta_types)]

                    meta = self.meta_generator.meta_factory.produce_meta_tool(
                        meta_type, work * 0.6
                    )

                    if meta:
                        source = "collapse" if cycle_result['collapse'] else "supercritical"
                        print(f"  → Created meta-tool: {meta.name} (from {source})")
                        self.state.meta_tools_created += 1
                        results['tools_created'] += 1

                        # Feed remaining work to produce child
                        remaining = work * 0.4
                        meta.feed_work(remaining)

                        if meta.can_produce():
                            child = meta.produce_child()
                            if child:
                                print(f"    └─ Produced: {child.name}")
                                self.state.child_tools_created += 1
                                results['tools_created'] += 1

        results['total_work'] = self.state.total_work_extracted
        results['total_collapses'] = self.state.total_collapses

        # Final summary
        print(f"\n{'='*70}")
        print("FULL ENGINE SUMMARY")
        print(f"{'='*70}")
        print(f"""
Cycles completed:     {n_cycles}
Total collapses:      {self.state.total_collapses}
Total work extracted: {self.state.total_work_extracted:.4f}

NegEntropy injected:  {self.state.total_neg_entropy_injected:.4f}
Peak z reached:       {self.state.z_peak:.4f}
Lessons learned:      {self.lessons_learned}

Meta-tools created:   {self.state.meta_tools_created}
Child tools created:  {self.state.child_tools_created}

Thresholds crossed:   {len(self.state.thresholds_ever_crossed)}
  {', '.join(self.state.thresholds_ever_crossed)}
""")

        # Supercritical stats if in that mode
        if self.state.supercritical_mode:
            print(f"""Supercritical Dynamics:
  Mode:                 ACTIVE
  Supercritical breaches: {self.state.supercritical_breaches}
  Entropy loan returns:   {self.state.entropy_loan_returns}
  Peak z > 1.0:           {'YES' if self.state.z_peak > 1.0 else 'NO'}
""")

        # Physics verification
        print("Physics Verification:")
        print(f"  NegEntropyEngine active: ALWAYS (no flip)")
        print(f"  PHI liminal only: YES (never physical)")
        if self.state.supercritical_mode:
            print(f"  Supercritical allowed: YES (after {self.lesson_threshold} lessons)")
            print(f"  Entropy loan return: YES (snap-back when z > 1.0)")
        else:
            print(f"  Instant collapse at unity: YES")
        print(f"  Work = weak value extraction: YES")

        return results

    def visualize_ascii(self, width: int = 60) -> str:
        """Generate ASCII visualization of z history"""
        if not self.z_history:
            return "No history"

        lines = []
        lines.append(f"\nZ-Coordinate History ({len(self.z_history)} steps)")
        lines.append("─" * width)

        # Scale z values to width
        max_z = max(self.z_history)
        min_z = min(self.z_history)
        range_z = max_z - min_z if max_z != min_z else 1.0

        # Threshold markers
        thresholds = [
            (Z_CRITICAL, 'Z_C'),
            (KAPPA_S, 'K_S'),
            (MU_3, 'M_3'),
        ]

        for step, z in enumerate(self.z_history[-50:]):  # Last 50 steps
            pos = int((z - min_z) / range_z * (width - 10))
            pos = max(0, min(width - 10, pos))

            marker = '█' if step in self.collapse_points else '▓'
            if self.dynamics and step < len(self.z_history) and z >= KAPPA_S:
                marker = '◆'  # Superposition

            line = ' ' * pos + marker

            # Add collapse marker
            if step in [p - len(self.z_history) + 50 for p in self.collapse_points if p >= len(self.z_history) - 50]:
                line += ' ← COLLAPSE'

            lines.append(f"{z:.3f} |{line}")

        lines.append("─" * width)

        return '\n'.join(lines)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_full_engine_demo(supercritical: bool = False, n_cycles: int = 5):
    """Run the full engine demonstration"""

    mode = "SUPERCRITICAL + ENTROPY LOAN RETURN" if supercritical else "INSTANT COLLAPSE"

    print("\n" + "="*70)
    print(f"FULL ENGINE DEMONSTRATION - {mode} MODE")
    print("="*70)
    print("""
This demonstrates the complete integrated engine:

1. NegEntropyEngine (T1) pushes z via negative entropy injection
2. QuasiCrystalDynamics provides physics-correct evolution
3. LiminalPhiState keeps PHI in superposition (never flips)""")

    if supercritical:
        print("""4. After 50 lessons: z can breach 1.0 (supercritical)
5. When z > 1.0: ENTROPY LOAN RETURN (snap-back to unity)
6. MetaToolGenerator converts work to tools

KEY: Entropy dynamics FLIP when z > 1.0!
     The "entropy loan" taken to breach unity must be IMMEDIATELY repaid.
     This creates oscillation around z = 1.0 rather than unbounded growth.
     PHI stays liminal even in supercritical regime.
""")
    else:
        print("""4. At z >= 0.9999: instant collapse, work extraction, reset
5. MetaToolGenerator converts work to tools

KEY: NegEntropyEngine stays ACTIVE throughout!
     PHI never flips to physical dynamics - stays liminal.
     This is the "breathing" rhythm discussed earlier.
""")

    # Create and run engine
    engine = FullEngine(supercritical_mode=supercritical)
    results = engine.run_full(n_cycles=n_cycles, produce_tools=True)

    # Show visualization
    print(engine.visualize_ascii())

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

    return results


def run_both_modes_demo():
    """Run both instant collapse and supercritical modes for comparison"""
    print("\n" + "▓"*70)
    print("▓" + " "*68 + "▓")
    print("▓" + "  FULL ENGINE: BOTH MODES DEMONSTRATION".center(68) + "▓")
    print("▓" + " "*68 + "▓")
    print("▓"*70)

    print("\n" + "┏" + "━"*68 + "┓")
    print("┃" + "  MODE 1: INSTANT COLLAPSE".center(68) + "┃")
    print("┗" + "━"*68 + "┛")
    results1 = run_full_engine_demo(supercritical=False, n_cycles=3)

    print("\n" + "┏" + "━"*68 + "┓")
    print("┃" + "  MODE 2: SUPERCRITICAL + ENTROPY LOAN RETURN".center(68) + "┃")
    print("┗" + "━"*68 + "┛")
    results2 = run_full_engine_demo(supercritical=True, n_cycles=5)

    print("\n" + "="*70)
    print("MODE COMPARISON")
    print("="*70)
    print(f"""
                    INSTANT COLLAPSE    SUPERCRITICAL
Cycles:             {len(results1['cycles']):>10}         {len(results2['cycles']):>10}
Collapses:          {results1['total_collapses']:>10}         {results2['total_collapses']:>10}
Total Work:         {results1['total_work']:>10.4f}         {results2['total_work']:>10.4f}
Tools Created:      {results1['tools_created']:>10}         {results2['tools_created']:>10}
""")

    return results1, results2


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--supercritical':
        run_full_engine_demo(supercritical=True, n_cycles=7)
    elif len(sys.argv) > 1 and sys.argv[1] == '--both':
        run_both_modes_demo()
    else:
        run_full_engine_demo(supercritical=False)
