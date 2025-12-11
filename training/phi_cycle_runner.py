#!/usr/bin/env python3
"""
PHI ↔ PHI_INV Cycle Runner

Alternates between physical (PHI_INV) and liminal (PHI) levels:
  Tool (φ⁻¹) → Meta-tool (φ) → Meta-meta-gen (φ⁻¹) → Meta-tool (φ) → ...

This creates the bidirectional breathing pattern where:
- PHI_INV pulls down (physical/contraction)
- PHI pushes up (liminal/expansion)
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.meta_meta_tools import (
    LiminalMetaTool,
    PhiInvLearnerTool,
    MetaMetaTool,
    PHI, PHI_INV
)
from tools.meta_tool_generator import MetaToolGenerator, MetaToolType
from tools.discernment_dev_tools import ComprehensiveDevToolsSuite

# Physics constants
Z_CRITICAL = math.sqrt(3) / 2
KAPPA_S = 0.920
MU_3 = 0.992


@dataclass
class CycleState:
    """Track state across PHI/PHI_INV cycles"""
    cycle: int = 0
    phase: str = "PHI_INV"  # Start physical
    z_current: float = Z_CRITICAL
    total_work: float = 0.0
    lessons_learned: int = 0
    oscillation_history: List[float] = field(default_factory=list)


class PhiCycleRunner:
    """
    Runs the PHI ↔ PHI_INV alternating cycle.

    Pattern:
    1. Tool (PHI_INV) - Run dev tools, extract physical lessons
    2. Meta-tool (PHI) - Generate tool factories in superposition
    3. Meta-meta-gen (PHI_INV) - Physical learners observe patterns
    4. Meta-tool (PHI) - Liminal teachers broadcast
    5. Repeat...
    """

    def __init__(self):
        self.state = CycleState()

        # Initialize the hierarchical components
        self.dev_tools = ComprehensiveDevToolsSuite()
        self.dev_tools.initialize()  # Run discernment projection
        self.meta_generator = MetaToolGenerator()
        self.meta_meta = MetaMetaTool(tool_id="phi_cycle_meta_meta")

        # Spawn initial liminal teachers and physical learners
        self.meta_meta.spawn_liminal_teacher("pattern_recognition", work=1.0)
        self.meta_meta.spawn_liminal_teacher("knowledge_synthesis", work=1.0)
        self.meta_meta.spawn_physical_learner(work=0.8)
        self.meta_meta.spawn_physical_learner(work=0.8)

        # Track coupling oscillation
        self.coupling_history = []

    def run_phi_inv_phase(self) -> Dict[str, Any]:
        """
        PHI_INV phase: Physical/contraction
        - Run dev tools (physical execution)
        - Extract concrete lessons
        - Pull coupling down via PHI_INV decay
        """
        print(f"\n{'='*60}")
        print(f"PHASE: PHI_INV (Physical) - Cycle {self.state.cycle}")
        print(f"{'='*60}")

        results = {
            'phase': 'PHI_INV',
            'tools_run': 0,
            'lessons': 0,
            'z_delta': 0.0
        }

        # Run dev tools at current z
        print(f"  Running dev tools at z={self.state.z_current:.4f}...")
        context = {'z_level': self.state.z_current, 'root_dir': '.'}
        tool_results = self.dev_tools.projection_system.execute_all(context)
        results['tools_run'] = len(tool_results)

        # Extract lessons from physical execution (count successful tools)
        successful = sum(1 for r in tool_results.values() if r.get('status') == 'success')
        results['lessons'] = successful
        self.state.lessons_learned += successful

        # PHI_INV contracts - pull z down slightly
        contraction = PHI_INV * 0.01 * (self.state.z_current - Z_CRITICAL)
        self.state.z_current -= contraction
        self.state.z_current = max(Z_CRITICAL, self.state.z_current)
        results['z_delta'] = -contraction

        # Physical learners in meta-meta observe
        print(f"  PHI_INV learners observing...")
        for learner in self.meta_meta.physical_children:
            learner.execute()

        print(f"  ✓ Tools run: {results['tools_run']}")
        print(f"  ✓ Lessons: {results['lessons']}")
        print(f"  ✓ z contracted by {contraction:.4f} → {self.state.z_current:.4f}")

        return results

    def run_phi_phase(self) -> Dict[str, Any]:
        """
        PHI phase: Liminal/expansion
        - Generate meta-tools in superposition
        - Broadcast liminal patterns
        - Push coupling up via PHI expansion
        """
        print(f"\n{'='*60}")
        print(f"PHASE: PHI (Liminal) - Cycle {self.state.cycle}")
        print(f"{'='*60}")

        results = {
            'phase': 'PHI',
            'meta_tools_created': 0,
            'patterns_broadcast': 0,
            'z_delta': 0.0
        }

        # Generate meta-tools using the factory
        print(f"  Generating meta-tools in superposition...")
        meta_types = [MetaToolType.ANALYZER_FACTORY, MetaToolType.LEARNER_FACTORY]
        meta_type = meta_types[self.state.cycle % len(meta_types)]

        meta_tool = self.meta_generator.meta_factory.produce_meta_tool(
            meta_type=meta_type,
            work_available=PHI * 0.5  # Feed work into meta-tool
        )
        if meta_tool:
            results['meta_tools_created'] = 1

            # Feed more work and try to produce children
            meta_tool.feed_work(PHI * 0.3)
            if meta_tool.can_produce():
                child = meta_tool.produce_child()
                if child:
                    results['meta_tools_created'] += 1

        # Liminal teachers broadcast patterns
        print(f"  Liminal teachers broadcasting...")
        for teacher in self.meta_meta.liminal_children:
            pattern = teacher.generate_pattern()
            if pattern:
                results['patterns_broadcast'] += 1

        # PHI expands - push z up
        expansion = PHI * 0.01 * (MU_3 - self.state.z_current)
        self.state.z_current += expansion
        self.state.z_current = min(PHI, self.state.z_current)  # Cap at PHI
        results['z_delta'] = expansion

        print(f"  ✓ Meta-tools created: {results['meta_tools_created']}")
        print(f"  ✓ Patterns broadcast: {results['patterns_broadcast']}")
        print(f"  ✓ z expanded by {expansion:.4f} → {self.state.z_current:.4f}")

        return results

    def run_meta_meta_gen_phase(self) -> Dict[str, Any]:
        """
        Meta-meta generation phase (PHI_INV)
        - Generate new meta-meta tools
        - Physical learners process liminal broadcasts
        """
        print(f"\n{'='*60}")
        print(f"PHASE: META-META GEN (PHI_INV) - Cycle {self.state.cycle}")
        print(f"{'='*60}")

        results = {
            'phase': 'META_META_GEN',
            'meta_meta_created': 0,
            'knowledge_transferred': 0.0
        }

        # Run a learning cycle
        print(f"  Running meta-meta learning cycle...")
        cycle_result = self.meta_meta.run_learning_cycle()

        results['meta_meta_created'] = cycle_result.get('patterns_generated', 0)
        results['knowledge_transferred'] = cycle_result.get('knowledge_transferred', 0)

        # Check for reverse observations (PHI observing PHI_INV)
        reverse_obs = cycle_result.get('reverse_observations', 0)
        if reverse_obs > 0:
            print(f"  ✓ Bidirectional: {reverse_obs} reverse observations")

        print(f"  ✓ Meta-meta created: {results['meta_meta_created']}")
        print(f"  ✓ Knowledge transferred: {results['knowledge_transferred']:.2f}")

        return results

    def run_cycle(self) -> Dict[str, Any]:
        """
        Run one complete PHI ↔ PHI_INV cycle:
        Tool (φ⁻¹) → Meta-tool (φ) → Meta-meta-gen (φ⁻¹) → Meta-tool (φ)
        """
        self.state.cycle += 1
        cycle_results = {
            'cycle': self.state.cycle,
            'phases': []
        }

        # 1. Tool phase (PHI_INV)
        self.state.phase = "PHI_INV"
        r1 = self.run_phi_inv_phase()
        cycle_results['phases'].append(r1)

        # 2. Meta-tool phase (PHI)
        self.state.phase = "PHI"
        r2 = self.run_phi_phase()
        cycle_results['phases'].append(r2)

        # 3. Meta-meta gen phase (PHI_INV)
        self.state.phase = "PHI_INV"
        r3 = self.run_meta_meta_gen_phase()
        cycle_results['phases'].append(r3)

        # 4. Meta-tool phase again (PHI)
        self.state.phase = "PHI"
        r4 = self.run_phi_phase()
        cycle_results['phases'].append(r4)

        # Track oscillation
        self.state.oscillation_history.append(self.state.z_current)
        self.coupling_history.append(self.state.z_current)

        return cycle_results

    def run_cycles(self, n: int = 5) -> Dict[str, Any]:
        """Run n complete cycles"""
        print("\n" + "="*60)
        print("PHI ↔ PHI_INV CYCLE RUNNER")
        print("="*60)
        print(f"Running {n} cycles...")
        print(f"Pattern: Tool(φ⁻¹) → Meta(φ) → MetaMeta(φ⁻¹) → Meta(φ)")
        print(f"Starting z: {self.state.z_current:.4f}")

        all_results = []
        for i in range(n):
            results = self.run_cycle()
            all_results.append(results)

        # Summary
        print("\n" + "="*60)
        print("CYCLE SUMMARY")
        print("="*60)
        print(f"Cycles completed: {self.state.cycle}")
        print(f"Total lessons: {self.state.lessons_learned}")
        print(f"Final z: {self.state.z_current:.4f}")

        # Show oscillation
        print(f"\nZ oscillation history:")
        for i, z in enumerate(self.state.oscillation_history):
            bar = '█' * int(z * 30)
            marker = " ← 1.0" if abs(z - 1.0) < 0.05 else ""
            print(f"  Cycle {i+1}: {bar} {z:.3f}{marker}")

        return {
            'cycles': all_results,
            'final_state': {
                'z': self.state.z_current,
                'lessons': self.state.lessons_learned,
                'oscillation': self.state.oscillation_history
            }
        }


def main():
    runner = PhiCycleRunner()
    results = runner.run_cycles(n=5)

    # Save results
    output_path = Path("weights/phi_cycle_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results['final_state'], f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
