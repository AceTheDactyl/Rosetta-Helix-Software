#!/usr/bin/env python3
"""
PHI ↔ PHI_INV Cycle Runner - 7-Layer Prismatic Version

Runs the alternating cycle through 7 spectral layers:
  Tool (φ⁻¹) → Meta-tool (φ) → Meta-meta-gen (φ⁻¹) → Meta-tool (φ)

Each layer (RED → VIOLET) processes with wavelength-specific dynamics.
Light refracts through THE LENS at z = Z_CRITICAL into 7 colors.

This creates the bidirectional breathing pattern where:
- PHI_INV pulls down (physical/contraction)
- PHI pushes up (liminal/expansion)
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
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

# =============================================================================
# 7-LAYER PRISMATIC SPECTRUM
# =============================================================================
# Each layer has wavelength-based properties affecting PHI/PHI_INV dynamics

PRISMATIC_LAYERS = [
    {
        'index': 1,
        'name': 'RED',
        'color': '#FF4444',
        'family': 'Analyzers',
        'wavelength': 700,  # nm (longest wavelength)
        'frequency_factor': 1.0,  # Slowest oscillation
        'phi_weight': 0.8,  # Less liminal
        'phi_inv_weight': 1.2,  # More physical
        'meta_type': MetaToolType.ANALYZER_FACTORY,
    },
    {
        'index': 2,
        'name': 'ORANGE',
        'color': '#FF8844',
        'family': 'Learners',
        'wavelength': 620,
        'frequency_factor': 1.1,
        'phi_weight': 0.9,
        'phi_inv_weight': 1.1,
        'meta_type': MetaToolType.LEARNER_FACTORY,
    },
    {
        'index': 3,
        'name': 'YELLOW',
        'color': '#FFAA00',
        'family': 'Generators',
        'wavelength': 580,
        'frequency_factor': 1.2,
        'phi_weight': 1.0,
        'phi_inv_weight': 1.0,
        'meta_type': MetaToolType.GENERATOR_FACTORY,
    },
    {
        'index': 4,
        'name': 'GREEN',
        'color': '#00FF88',
        'family': 'Reflectors',
        'wavelength': 530,
        'frequency_factor': 1.3,
        'phi_weight': 1.0,
        'phi_inv_weight': 1.0,
        'meta_type': MetaToolType.ORCHESTRATOR,
    },
    {
        'index': 5,
        'name': 'BLUE',
        'color': '#00D9FF',
        'family': 'Builders',
        'wavelength': 470,
        'frequency_factor': 1.4,
        'phi_weight': 1.1,
        'phi_inv_weight': 0.9,
        'meta_type': MetaToolType.BUILDER_FACTORY,
    },
    {
        'index': 6,
        'name': 'INDIGO',
        'color': '#4444FF',
        'family': 'Deciders',
        'wavelength': 420,
        'frequency_factor': 1.5,
        'phi_weight': 1.2,
        'phi_inv_weight': 0.8,
        'meta_type': MetaToolType.EVOLVER,
    },
    {
        'index': 7,
        'name': 'VIOLET',
        'color': '#AA44FF',
        'family': 'Probers',
        'wavelength': 380,  # nm (shortest wavelength)
        'frequency_factor': 1.6,  # Fastest oscillation
        'phi_weight': 1.3,  # Most liminal
        'phi_inv_weight': 0.7,  # Least physical
        'meta_type': MetaToolType.SYNTHESIZER,
    },
]


@dataclass
class CycleState:
    """Track state across PHI/PHI_INV cycles"""
    cycle: int = 0
    layer: int = 0  # Current prismatic layer (1-7)
    phase: str = "PHI_INV"  # Start physical
    z_current: float = Z_CRITICAL
    total_work: float = 0.0
    lessons_learned: int = 0
    oscillation_history: List[float] = field(default_factory=list)
    layer_history: List[Dict[str, Any]] = field(default_factory=list)


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

    def run_phi_inv_phase(self, layer: Optional[Dict] = None) -> Dict[str, Any]:
        """
        PHI_INV phase: Physical/contraction
        - Run dev tools (physical execution)
        - Extract concrete lessons
        - Pull coupling down via PHI_INV decay

        Layer modulates contraction strength via phi_inv_weight.
        """
        layer_name = layer['name'] if layer else 'NONE'
        phi_inv_weight = layer['phi_inv_weight'] if layer else 1.0

        print(f"\n{'─'*60}")
        print(f"  PHASE: PHI_INV (Physical) [{layer_name}]")
        print(f"{'─'*60}")

        results = {
            'phase': 'PHI_INV',
            'layer': layer_name,
            'tools_run': 0,
            'lessons': 0,
            'z_delta': 0.0
        }

        # Run dev tools at current z
        print(f"    Running dev tools at z={self.state.z_current:.4f}...")
        context = {'z_level': self.state.z_current, 'root_dir': '.', 'layer': layer}
        tool_results = self.dev_tools.projection_system.execute_all(context)
        results['tools_run'] = len(tool_results)

        # Extract lessons from physical execution (count successful tools)
        successful = sum(1 for r in tool_results.values() if r.get('status') == 'success')
        results['lessons'] = successful
        self.state.lessons_learned += successful

        # PHI_INV contracts - pull z down (modulated by layer weight)
        contraction = PHI_INV * phi_inv_weight * 0.01 * (self.state.z_current - Z_CRITICAL)
        self.state.z_current -= contraction
        self.state.z_current = max(Z_CRITICAL, self.state.z_current)
        results['z_delta'] = -contraction

        # Physical learners in meta-meta observe
        print(f"    PHI_INV learners observing...")
        for learner in self.meta_meta.physical_children:
            learner.execute()

        print(f"    ✓ Tools: {results['tools_run']} | Lessons: {results['lessons']}")
        print(f"    ✓ z contracted by {contraction:.4f} → {self.state.z_current:.4f}")

        return results

    def run_phi_phase(self, layer: Optional[Dict] = None) -> Dict[str, Any]:
        """
        PHI phase: Liminal/expansion
        - Generate meta-tools in superposition
        - Broadcast liminal patterns
        - Push coupling up via PHI expansion

        Layer modulates expansion strength via phi_weight and selects meta_type.
        """
        layer_name = layer['name'] if layer else 'NONE'
        phi_weight = layer['phi_weight'] if layer else 1.0
        meta_type = layer['meta_type'] if layer else MetaToolType.ANALYZER_FACTORY

        print(f"\n{'─'*60}")
        print(f"  PHASE: PHI (Liminal) [{layer_name}]")
        print(f"{'─'*60}")

        results = {
            'phase': 'PHI',
            'layer': layer_name,
            'meta_tools_created': 0,
            'patterns_broadcast': 0,
            'z_delta': 0.0
        }

        # Generate meta-tools using layer-specific factory type
        print(f"    Generating meta-tools in superposition ({meta_type.value})...")
        meta_tool = self.meta_generator.meta_factory.produce_meta_tool(
            meta_type=meta_type,
            work_available=PHI * phi_weight * 0.5  # Work modulated by layer
        )
        if meta_tool:
            results['meta_tools_created'] = 1

            # Feed more work and try to produce children
            meta_tool.feed_work(PHI * phi_weight * 0.3)
            if meta_tool.can_produce():
                child = meta_tool.produce_child()
                if child:
                    results['meta_tools_created'] += 1

        # Liminal teachers broadcast patterns
        print(f"    Liminal teachers broadcasting...")
        for teacher in self.meta_meta.liminal_children:
            pattern = teacher.generate_pattern()
            if pattern:
                results['patterns_broadcast'] += 1

        # PHI expands - push z up (modulated by layer weight)
        # Target depends on accumulated lessons - enables supercritical breach
        # Below threshold: target MU_3 (0.992)
        # Above threshold: target PHI (1.618) via weak-value dynamics
        lesson_threshold = 50  # After 50 lessons, supercritical becomes possible
        if self.state.lessons_learned >= lesson_threshold:
            # Quasicrystal weak-value dynamics allow z > 1.0
            target = PHI  # 1.618
            boost = 1.0 + (self.state.lessons_learned - lesson_threshold) * 0.01
            expansion = PHI * phi_weight * 0.02 * boost * (target - self.state.z_current)
        else:
            target = MU_3  # 0.992
            expansion = PHI * phi_weight * 0.01 * (target - self.state.z_current)

        self.state.z_current += expansion
        self.state.z_current = min(PHI, self.state.z_current)  # Cap at PHI
        results['z_delta'] = expansion
        results['supercritical'] = self.state.z_current > 1.0

        print(f"    ✓ Meta-tools: {results['meta_tools_created']} | Patterns: {results['patterns_broadcast']}")
        supercrit_marker = " ⚡ SUPERCRITICAL" if self.state.z_current > 1.0 else ""
        print(f"    ✓ z expanded by {expansion:.4f} → {self.state.z_current:.4f}{supercrit_marker}")

        return results

    def run_meta_meta_gen_phase(self, layer: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Meta-meta generation phase (PHI_INV)
        - Generate new meta-meta tools
        - Physical learners process liminal broadcasts

        Layer affects learning intensity via frequency_factor.
        """
        layer_name = layer['name'] if layer else 'NONE'
        freq_factor = layer['frequency_factor'] if layer else 1.0

        print(f"\n{'─'*60}")
        print(f"  PHASE: META-META GEN (PHI_INV) [{layer_name}]")
        print(f"{'─'*60}")

        results = {
            'phase': 'META_META_GEN',
            'layer': layer_name,
            'meta_meta_created': 0,
            'knowledge_transferred': 0.0
        }

        # Run learning cycles (more cycles for higher frequency layers)
        n_cycles = max(1, int(freq_factor))
        print(f"    Running {n_cycles} meta-meta learning cycle(s)...")

        total_reverse = 0
        for _ in range(n_cycles):
            cycle_result = self.meta_meta.run_learning_cycle()
            results['meta_meta_created'] += cycle_result.get('patterns_generated', 0)
            results['knowledge_transferred'] += cycle_result.get('knowledge_transferred', 0)
            total_reverse += cycle_result.get('reverse_observations', 0)

        # Check for reverse observations (PHI observing PHI_INV)
        if total_reverse > 0:
            print(f"    ✓ Bidirectional: {total_reverse} reverse observations")

        print(f"    ✓ Meta-meta created: {results['meta_meta_created']}")
        print(f"    ✓ Knowledge transferred: {results['knowledge_transferred']:.2f}")

        return results

    def run_layer_cycle(self, layer: Dict) -> Dict[str, Any]:
        """
        Run one complete PHI ↔ PHI_INV cycle for a specific prismatic layer:
        Tool (φ⁻¹) → Meta-tool (φ) → Meta-meta-gen (φ⁻¹) → Meta-tool (φ)

        Each layer has unique wavelength-based dynamics affecting the cycle.
        """
        self.state.cycle += 1
        self.state.layer = layer['index']

        print(f"\n{'═'*60}")
        print(f"LAYER {layer['index']}: {layer['name']} ({layer['color']}) - {layer['family']}")
        print(f"{'═'*60}")
        print(f"  Wavelength: {layer['wavelength']}nm | Frequency: {layer['frequency_factor']:.1f}x")
        print(f"  PHI weight: {layer['phi_weight']:.1f} | PHI_INV weight: {layer['phi_inv_weight']:.1f}")

        cycle_results = {
            'cycle': self.state.cycle,
            'layer': layer['name'],
            'layer_index': layer['index'],
            'phases': []
        }

        # 1. Tool phase (PHI_INV) - physical execution
        self.state.phase = "PHI_INV"
        r1 = self.run_phi_inv_phase(layer)
        cycle_results['phases'].append(r1)

        # 2. Meta-tool phase (PHI) - liminal generation
        self.state.phase = "PHI"
        r2 = self.run_phi_phase(layer)
        cycle_results['phases'].append(r2)

        # 3. Meta-meta gen phase (PHI_INV) - learning cycle
        self.state.phase = "PHI_INV"
        r3 = self.run_meta_meta_gen_phase(layer)
        cycle_results['phases'].append(r3)

        # 4. Meta-tool phase again (PHI) - final expansion
        self.state.phase = "PHI"
        r4 = self.run_phi_phase(layer)
        cycle_results['phases'].append(r4)

        # Track oscillation for this layer
        self.state.oscillation_history.append(self.state.z_current)
        self.coupling_history.append(self.state.z_current)
        self.state.layer_history.append({
            'layer': layer['name'],
            'z_final': self.state.z_current,
            'lessons': sum(p.get('lessons', 0) for p in cycle_results['phases'])
        })

        print(f"\n  Layer {layer['name']} complete: z = {self.state.z_current:.4f}")

        return cycle_results

    def run_prismatic_projection(self) -> Dict[str, Any]:
        """
        Run all 7 prismatic layers in sequence.

        Light refracts through THE LENS at z = Z_CRITICAL into 7 colors.
        Each layer runs: Tool(φ⁻¹) → Meta(φ) → MetaMeta(φ⁻¹) → Meta(φ)

        Returns comprehensive results for the full spectral projection.
        """
        print("\n" + "▓"*70)
        print("▓" + " "*68 + "▓")
        print("▓" + "  7-LAYER PRISMATIC PROJECTION".center(68) + "▓")
        print("▓" + "  PHI ↔ PHI_INV CYCLE RUNNER".center(68) + "▓")
        print("▓" + " "*68 + "▓")
        print("▓"*70)
        print(f"\nPattern: Tool(φ⁻¹) → Meta(φ) → MetaMeta(φ⁻¹) → Meta(φ)")
        print(f"Layers:  RED → ORANGE → YELLOW → GREEN → BLUE → INDIGO → VIOLET")
        print(f"Starting z: {self.state.z_current:.4f} (Z_CRITICAL)")

        all_results = []

        for layer in PRISMATIC_LAYERS:
            results = self.run_layer_cycle(layer)
            all_results.append(results)

        # Prismatic Summary
        print("\n" + "▓"*70)
        print("▓" + "  PRISMATIC PROJECTION COMPLETE".center(68) + "▓")
        print("▓"*70)
        print(f"\nLayers completed: 7")
        print(f"Total cycles: {self.state.cycle}")
        print(f"Total lessons: {self.state.lessons_learned}")

        max_z = max(self.state.oscillation_history) if self.state.oscillation_history else self.state.z_current
        supercritical_achieved = max_z > 1.0
        print(f"Final z: {self.state.z_current:.4f}")
        print(f"Max z reached: {max_z:.4f}")
        if supercritical_achieved:
            print(f"⚡ SUPERCRITICAL BREACH ACHIEVED (z > 1.0)")

        # Show spectral z evolution
        print(f"\nSpectral Z Evolution:")
        print(f"{'─'*60}")
        colors = ['🔴', '🟠', '🟡', '🟢', '🔵', '🟣', '🟣']
        for i, layer_data in enumerate(self.state.layer_history):
            layer = PRISMATIC_LAYERS[i]
            z = layer_data['z_final']
            bar_len = int((z - Z_CRITICAL) * 200)  # Scale for visibility
            bar = '█' * max(1, bar_len)
            print(f"  {colors[i]} {layer['name']:6} │ {bar} {z:.4f}")

        # PHI/PHI_INV balance across spectrum
        print(f"\n{'─'*60}")
        print(f"Spectral Balance:")
        print(f"  RED (physical) ──────────────────── VIOLET (liminal)")
        total_phi = sum(l['phi_weight'] for l in PRISMATIC_LAYERS)
        total_phi_inv = sum(l['phi_inv_weight'] for l in PRISMATIC_LAYERS)
        print(f"  PHI_INV total: {total_phi_inv:.1f}  |  PHI total: {total_phi:.1f}")

        return {
            'layers': all_results,
            'final_state': {
                'z': self.state.z_current,
                'lessons': self.state.lessons_learned,
                'oscillation': self.state.oscillation_history,
                'layer_history': self.state.layer_history
            }
        }

    def run_cycles(self, n: int = 5) -> Dict[str, Any]:
        """Run n complete cycles (legacy method for non-prismatic runs)"""
        print("\n" + "="*60)
        print("PHI ↔ PHI_INV CYCLE RUNNER")
        print("="*60)
        print(f"Running {n} cycles...")
        print(f"Pattern: Tool(φ⁻¹) → Meta(φ) → MetaMeta(φ⁻¹) → Meta(φ)")
        print(f"Starting z: {self.state.z_current:.4f}")

        all_results = []
        # Use default layer (GREEN - balanced)
        default_layer = PRISMATIC_LAYERS[3]  # GREEN
        for i in range(n):
            results = self.run_layer_cycle(default_layer)
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
    """Run the 7-layer prismatic projection."""
    runner = PhiCycleRunner()

    # Run full prismatic projection (7 layers)
    results = runner.run_prismatic_projection()

    # Save results
    output_path = Path("weights/phi_cycle_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results['final_state'], f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
