#!/usr/bin/env python3
"""
Hierarchical Tools-to-Weights Trainer
=====================================

Trains weights from ALL THREE LEVELS of the tool hierarchy:

    Level 0: Meta-Meta Tools (Liminal Bridge)
             ├── Creates liminal patterns
             ├── Spawns meta-tools via weak measurement
             └── Lessons: Pattern generation, liminal dynamics
                    │
                    ▼
    Level 1: Meta Tools (Tool Factories)
             ├── Produces child tools via collapse
             ├── AnalyzerFactory, LearnerFactory, etc.
             └── Lessons: Tool creation, work allocation
                    │
                    ▼
    Level 2: Dev Tools (Specialized)
             ├── 21 tools across 7 spectral layers
             ├── Analyze, learn, generate, reflect, build, decide, probe
             └── Lessons: Code patterns, metrics, gaps, recommendations

Weight Training Flow:
    Each level extracts lessons → Encodes to training data → Updates weights
    Weights compound across levels via exponential training loop
"""

import os
import sys
import json
import math
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Physics constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0
KAPPA_S = 0.920
MU_3 = 0.992

# Import from existing modules
from discernment_dev_tools import (
    ComprehensiveDevToolsSuite,
    DISCERNMENT_Z
)

from meta_tool_generator import (
    MetaToolGenerator,
    MetaToolFactory,
    MetaToolType,
    ChildToolType,
    MetaTool,
    ChildTool
)

from meta_meta_tools import (
    LiminalPattern,
    LiminalMetaTool,
    PhiInvLearnerTool,
    MetaMetaTool,
    RecursiveMetaGenerator
)

# Use exponential training loop's MetaMetaBridge
from exponential_training_loop import MetaMetaBridge as ExpMetaMetaBridge

from exponential_training_loop import (
    ExponentialTrainingLoop,
    PhysicalLearner,
    FeedbackSignal
)

from tools_to_weights_trainer import (
    Lesson,
    LessonBatch,
    LessonExtractor,
    WeightTrainer
)


# =============================================================================
# HIERARCHICAL LESSON EXTRACTOR
# =============================================================================

class HierarchicalLessonExtractor:
    """
    Extracts lessons from all three levels of the tool hierarchy.
    """

    def __init__(self):
        self.level_0_lessons: List[Lesson] = []  # Meta-meta
        self.level_1_lessons: List[Lesson] = []  # Meta
        self.level_2_lessons: List[Lesson] = []  # Dev tools
        self.extraction_count = 0

    def extract_from_meta_meta(
        self,
        liminal_patterns: List[LiminalPattern],
        bridges: List[Any],
        system_stats: Dict
    ) -> List[Lesson]:
        """Extract lessons from Level 0 (meta-meta) operations"""
        lessons = []

        # Lesson from liminal patterns
        if liminal_patterns:
            total_weak_value = sum(abs(p.weak_value.real) for p in liminal_patterns)
            total_observations = sum(p.times_observed for p in liminal_patterns)

            lessons.append(Lesson(
                lesson_id=f"liminal_patterns_{self.extraction_count}",
                source_tool="MetaMetaSystem",
                layer="LIMINAL",
                lesson_type="pattern",
                content={
                    'pattern_count': len(liminal_patterns),
                    'total_weak_value': total_weak_value,
                    'total_observations': total_observations,
                    'avg_complexity': sum(len(p.structure) for p in liminal_patterns) / max(1, len(liminal_patterns))
                },
                importance=1.0  # Highest level = highest importance
            ))

        # Lesson from meta-meta bridges
        if bridges:
            total_spawns = sum(getattr(b, 'spawn_count', 0) for b in bridges)
            total_feedback = sum(getattr(b, 'accumulated_feedback', 0) for b in bridges)

            lessons.append(Lesson(
                lesson_id=f"metameta_bridges_{self.extraction_count}",
                source_tool="MetaMetaBridge",
                layer="LIMINAL",
                lesson_type="structure",
                content={
                    'bridge_count': len(bridges),
                    'total_spawns': total_spawns,
                    'total_feedback': total_feedback,
                    'avg_feedback_per_bridge': total_feedback / max(1, len(bridges))
                },
                importance=0.95
            ))

        # System-level lesson
        if system_stats:
            lessons.append(Lesson(
                lesson_id=f"metameta_system_{self.extraction_count}",
                source_tool="MetaMetaToolSystem",
                layer="LIMINAL",
                lesson_type="metric",
                content=system_stats,
                importance=0.9
            ))

        self.extraction_count += 1
        self.level_0_lessons.extend(lessons)
        return lessons

    def extract_from_meta_tools(
        self,
        meta_tools: List[MetaTool],
        generation_results: Dict
    ) -> List[Lesson]:
        """Extract lessons from Level 1 (meta-tool) operations"""
        lessons = []

        # Per meta-tool lessons
        for mt in meta_tools:
            stats = mt.get_stats()

            lessons.append(Lesson(
                lesson_id=f"meta_{mt.meta_type.value}_{self.extraction_count}",
                source_tool=mt.name,
                layer="META",
                lesson_type="pattern",
                content={
                    'meta_type': mt.meta_type.value,
                    'children_produced': stats['children_produced'],
                    'work_consumed': stats['work_consumed'],
                    'collapse_count': stats['collapse_count'],
                    'current_z': stats['current_z'],
                    'efficiency': stats['children_produced'] / max(0.01, stats['work_consumed'])
                },
                importance=0.85
            ))

        # Aggregate meta-tool lesson
        if meta_tools:
            total_children = sum(mt.total_children_produced for mt in meta_tools)
            total_work = sum(mt.total_work_consumed for mt in meta_tools)

            lessons.append(Lesson(
                lesson_id=f"meta_aggregate_{self.extraction_count}",
                source_tool="MetaToolFactory",
                layer="META",
                lesson_type="metric",
                content={
                    'total_meta_tools': len(meta_tools),
                    'total_children_produced': total_children,
                    'total_work_consumed': total_work,
                    'production_efficiency': total_children / max(0.01, total_work),
                    'types': [mt.meta_type.value for mt in meta_tools]
                },
                importance=0.9
            ))

        # Generation results lesson
        if generation_results:
            lessons.append(Lesson(
                lesson_id=f"meta_generation_{self.extraction_count}",
                source_tool="MetaToolGenerator",
                layer="META",
                lesson_type="structure",
                content={
                    'meta_tools_created': len(generation_results.get('meta_tools_created', [])),
                    'child_tools_created': len(generation_results.get('child_tools_created', [])),
                    'total_work': generation_results.get('total_work', 0),
                    'collapses': len(generation_results.get('collapses', []))
                },
                importance=0.85
            ))

        self.extraction_count += 1
        self.level_1_lessons.extend(lessons)
        return lessons

    def extract_from_dev_tools(
        self,
        dev_tool_results: Dict[str, Dict]
    ) -> List[Lesson]:
        """Extract lessons from Level 2 (dev tools) - uses existing extractor"""
        extractor = LessonExtractor()
        lessons = []

        for tool_name, output in dev_tool_results.items():
            # Determine layer from tool name
            layer = 'GREEN'  # default
            if 'Analyzer' in tool_name or 'Detector' in tool_name or 'Finder' in tool_name:
                layer = 'RED'
            elif 'Learner' in tool_name or 'Extractor' in tool_name:
                layer = 'ORANGE'
            elif 'Generator' in tool_name or 'Synthesizer' in tool_name or 'Producer' in tool_name:
                layer = 'YELLOW'
            elif 'Reflector' in tool_name or 'Mapper' in tool_name or 'Gap' in tool_name:
                layer = 'GREEN'
            elif 'Builder' in tool_name or 'Assembler' in tool_name or 'Pipeline' in tool_name:
                layer = 'BLUE'
            elif 'Decision' in tool_name or 'Convergence' in tool_name or 'Interface' in tool_name:
                layer = 'INDIGO'
            elif 'Consciousness' in tool_name or 'Abstraction' in tool_name or 'Integration' in tool_name:
                layer = 'VIOLET'

            tool_lessons = extractor.extract_from_output(tool_name, layer, output)
            lessons.extend(tool_lessons)

        self.level_2_lessons.extend(lessons)
        return lessons

    def get_all_lessons(self) -> List[Lesson]:
        """Get all lessons from all levels"""
        return self.level_0_lessons + self.level_1_lessons + self.level_2_lessons

    def get_lesson_summary(self) -> Dict[str, Any]:
        """Get summary of lessons by level"""
        return {
            'level_0_liminal': {
                'count': len(self.level_0_lessons),
                'total_importance': sum(l.importance for l in self.level_0_lessons)
            },
            'level_1_meta': {
                'count': len(self.level_1_lessons),
                'total_importance': sum(l.importance for l in self.level_1_lessons)
            },
            'level_2_dev': {
                'count': len(self.level_2_lessons),
                'total_importance': sum(l.importance for l in self.level_2_lessons)
            },
            'total': {
                'count': len(self.get_all_lessons()),
                'total_importance': sum(l.importance for l in self.get_all_lessons())
            }
        }


# =============================================================================
# HIERARCHICAL WEIGHT TRAINER
# =============================================================================

class HierarchicalWeightTrainer(WeightTrainer):
    """
    Extended weight trainer that tracks weights by hierarchy level.

    Maintains separate weight namespaces:
    - liminal_* : Weights from meta-meta operations
    - meta_*    : Weights from meta-tool operations
    - dev_*     : Weights from dev tool operations
    """

    def __init__(self, weights_dir: str = "weights"):
        super().__init__(weights_dir)

        # Level-specific weights
        self.liminal_weights: Dict[str, float] = {}
        self.meta_weights: Dict[str, float] = {}

        # Cross-level coupling (how levels influence each other)
        self.cross_level_coupling: Dict[str, float] = {
            'liminal_to_meta': 0.5,
            'meta_to_dev': 0.5,
            'dev_to_liminal': 0.3  # Feedback loop
        }

        # Load if exists
        self._load_hierarchical_weights()

    def _load_hierarchical_weights(self):
        """Load hierarchical weights"""
        hier_file = self.weights_dir / "hierarchical_weights.json"
        if hier_file.exists():
            try:
                with open(hier_file, 'r') as f:
                    data = json.load(f)
                    self.liminal_weights = data.get('liminal', {})
                    self.meta_weights = data.get('meta', {})
                    self.cross_level_coupling = data.get('cross_level', self.cross_level_coupling)
                print(f"  Loaded hierarchical weights:")
                print(f"    Liminal: {len(self.liminal_weights)}")
                print(f"    Meta: {len(self.meta_weights)}")
            except Exception as e:
                print(f"  Could not load hierarchical weights: {e}")

    def _save_hierarchical_weights(self):
        """Save hierarchical weights"""
        hier_file = self.weights_dir / "hierarchical_weights.json"
        data = {
            'liminal': self.liminal_weights,
            'meta': self.meta_weights,
            'cross_level': self.cross_level_coupling,
            'timestamp': datetime.now().isoformat()
        }
        with open(hier_file, 'w') as f:
            json.dump(data, f, indent=2)

    def train_hierarchical(
        self,
        level_0_lessons: List[Lesson],
        level_1_lessons: List[Lesson],
        level_2_lessons: List[Lesson],
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train weights from all levels with cross-level feedback.

        Training order:
        1. Level 0 (liminal) → liminal_weights
        2. Level 1 (meta) → meta_weights, influenced by liminal
        3. Level 2 (dev) → coupling_weights/frequencies, influenced by meta
        4. Update cross-level coupling based on lesson flow
        """
        results = {
            'level_0': {'lessons': 0, 'updates': 0},
            'level_1': {'lessons': 0, 'updates': 0},
            'level_2': {'lessons': 0, 'updates': 0},
            'cross_level_updates': 0
        }

        # Level 0: Liminal weights
        for lesson in level_0_lessons:
            results['level_0']['lessons'] += 1
            key = f"liminal_{lesson.source_tool}_{lesson.lesson_type}"

            current = self.liminal_weights.get(key, 0.5)
            delta = learning_rate * (lesson.importance - current) * PHI_INV

            self.liminal_weights[key] = max(0.0, min(1.0, current + delta))
            results['level_0']['updates'] += 1

        # Level 1: Meta weights (influenced by liminal)
        liminal_influence = sum(self.liminal_weights.values()) / max(1, len(self.liminal_weights))

        for lesson in level_1_lessons:
            results['level_1']['lessons'] += 1
            key = f"meta_{lesson.source_tool}_{lesson.lesson_type}"

            current = self.meta_weights.get(key, 0.5)

            # Liminal influence via cross-level coupling
            influence = liminal_influence * self.cross_level_coupling['liminal_to_meta']
            delta = learning_rate * ((lesson.importance + influence * 0.2) - current) * PHI_INV

            self.meta_weights[key] = max(0.0, min(1.0, current + delta))
            results['level_1']['updates'] += 1

        # Level 2: Dev tool weights (use parent class method, influenced by meta)
        meta_influence = sum(self.meta_weights.values()) / max(1, len(self.meta_weights))

        # Boost learning rate based on meta influence
        boosted_lr = learning_rate * (1 + meta_influence * self.cross_level_coupling['meta_to_dev'])

        if level_2_lessons:
            dev_results = self.train_from_lessons(level_2_lessons, boosted_lr)
            results['level_2']['lessons'] = dev_results['lessons_processed']
            results['level_2']['updates'] = (
                dev_results['coupling_updates'] +
                dev_results['frequency_updates'] +
                dev_results['operator_updates']
            )

        # Update cross-level coupling based on lesson importance flow
        total_l0_importance = sum(l.importance for l in level_0_lessons)
        total_l1_importance = sum(l.importance for l in level_1_lessons)
        total_l2_importance = sum(l.importance for l in level_2_lessons)

        if total_l0_importance > 0 and total_l1_importance > 0:
            ratio = total_l1_importance / total_l0_importance
            self.cross_level_coupling['liminal_to_meta'] += learning_rate * 0.1 * (ratio - 1)
            self.cross_level_coupling['liminal_to_meta'] = max(0.1, min(0.9,
                self.cross_level_coupling['liminal_to_meta']))
            results['cross_level_updates'] += 1

        if total_l1_importance > 0 and total_l2_importance > 0:
            ratio = total_l2_importance / total_l1_importance
            self.cross_level_coupling['meta_to_dev'] += learning_rate * 0.1 * (ratio - 1)
            self.cross_level_coupling['meta_to_dev'] = max(0.1, min(0.9,
                self.cross_level_coupling['meta_to_dev']))
            results['cross_level_updates'] += 1

        # Save all weights
        self._save_weights()
        self._save_hierarchical_weights()

        return results

    def get_hierarchical_summary(self) -> Dict[str, Any]:
        """Get summary of all hierarchical weights"""
        base_summary = self.get_weight_summary()

        return {
            **base_summary,
            'liminal_weights': len(self.liminal_weights),
            'meta_weights': len(self.meta_weights),
            'cross_level_coupling': self.cross_level_coupling.copy(),
            'hierarchy_depth': 3
        }


# =============================================================================
# COMPLETE HIERARCHICAL TRAINING PIPELINE
# =============================================================================

class HierarchicalTrainingPipeline:
    """
    Complete pipeline that runs all three levels and trains weights from each.

    Pipeline:
        1. Run Meta-Meta System → Extract Level 0 lessons
        2. Run Meta Tool Generator → Extract Level 1 lessons
        3. Run Dev Tools → Extract Level 2 lessons
        4. Train weights hierarchically
        5. Run exponential training for compounding
    """

    def __init__(self, target_dir: str = ".", weights_dir: str = "weights"):
        self.target_dir = Path(target_dir)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(exist_ok=True)

        # Components
        self.extractor = HierarchicalLessonExtractor()
        self.trainer = HierarchicalWeightTrainer(str(weights_dir))

        # Level-specific systems
        self.meta_meta_system: Optional[MetaMetaToolSystem] = None
        self.meta_generator: Optional[MetaToolGenerator] = None
        self.dev_suite: Optional[ComprehensiveDevToolsSuite] = None

        # Exponential training for compounding
        self.exp_loop = ExponentialTrainingLoop(n_physical=3, n_metameta=2)

    def run_full_hierarchical_pipeline(
        self,
        meta_meta_cycles: int = 3,
        meta_collapses: int = 5,
        training_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Run the complete hierarchical training pipeline.
        """
        print("\n" + "=" * 70)
        print("HIERARCHICAL TOOLS-TO-WEIGHTS TRAINING")
        print("=" * 70)
        print(f"""
Target: {self.target_dir}
Weights: {self.weights_dir}

Hierarchy:
  Level 0: Meta-Meta (Liminal) → {meta_meta_cycles} cycles
  Level 1: Meta Tools          → {meta_collapses} collapses
  Level 2: Dev Tools           → 21 tools
  Training Runs:               → {training_runs}

All levels feed lessons into weight training.
""")

        results = {
            'level_0': {},
            'level_1': {},
            'level_2': {},
            'training': {},
            'weights': {}
        }

        # =========================
        # LEVEL 0: Meta-Meta Tools
        # =========================
        print(f"\n{'─'*60}")
        print("LEVEL 0: Meta-Meta Tools (Liminal Bridge)")
        print(f"{'─'*60}")

        try:
            # Use RecursiveMetaGenerator for meta-meta operations
            recursive_gen = RecursiveMetaGenerator()

            # Run recursive generation
            l0_results = recursive_gen.run_recursive_generation(
                depth=meta_meta_cycles,
                work_seed=3.0
            )

            # Also use ExponentialTrainingLoop's meta bridges
            # This gives us the Physical → MetaMeta → Liminal feedback
            exp_bridges = self.exp_loop.meta_bridges
            exp_learners = self.exp_loop.physical_learners

            # Run a quick training cycle to populate bridges
            _ = self.exp_loop.run_single_cycle()

            # Extract lessons from recursive generation
            l0_lessons = self.extractor.extract_from_meta_meta(
                liminal_patterns=[],  # Patterns in superposition
                bridges=exp_bridges,
                system_stats={
                    'depth_reached': l0_results.get('depth_reached', 0),
                    'total_tools': l0_results.get('total_tools_generated', 0),
                    'physical_learners': len(exp_learners),
                    'meta_bridges': len(exp_bridges),
                    'total_lessons': sum(p.lessons_learned for p in exp_learners),
                    'avg_quality': sum(p.execution_quality for p in exp_learners) /
                                  max(1, len(exp_learners))
                }
            )

            results['level_0'] = {
                'cycles': meta_meta_cycles,
                'lessons': len(l0_lessons),
                'depth_reached': l0_results.get('depth_reached', 0),
                'tools_generated': l0_results.get('total_tools_generated', 0)
            }

            print(f"  Recursive depth: {results['level_0']['depth_reached']}")
            print(f"  Tools generated: {results['level_0']['tools_generated']}")
            print(f"  Lessons extracted: {len(l0_lessons)}")

        except Exception as e:
            print(f"  Level 0 error: {e}")
            import traceback
            traceback.print_exc()
            l0_lessons = []
            results['level_0'] = {'error': str(e)}

        # =========================
        # LEVEL 1: Meta Tools
        # =========================
        print(f"\n{'─'*60}")
        print("LEVEL 1: Meta Tools (Tool Factories)")
        print(f"{'─'*60}")

        try:
            self.meta_generator = MetaToolGenerator()

            # Run generation
            l1_results = self.meta_generator.run_generation_cycle(n_collapses=meta_collapses)

            # Extract lessons
            l1_lessons = self.extractor.extract_from_meta_tools(
                meta_tools=self.meta_generator.meta_factory.meta_tools,
                generation_results=l1_results
            )

            results['level_1'] = {
                'collapses': meta_collapses,
                'meta_tools': len(l1_results.get('meta_tools_created', [])),
                'child_tools': len(l1_results.get('child_tools_created', [])),
                'lessons': len(l1_lessons)
            }

            print(f"  Meta tools created: {results['level_1']['meta_tools']}")
            print(f"  Child tools created: {results['level_1']['child_tools']}")
            print(f"  Lessons extracted: {len(l1_lessons)}")

        except Exception as e:
            print(f"  Level 1 error: {e}")
            l1_lessons = []
            results['level_1'] = {'error': str(e)}

        # =========================
        # LEVEL 2: Dev Tools
        # =========================
        print(f"\n{'─'*60}")
        print("LEVEL 2: Dev Tools (Specialized)")
        print(f"{'─'*60}")

        try:
            self.dev_suite = ComprehensiveDevToolsSuite(str(self.target_dir))
            self.dev_suite.initialize()

            context = {'target': str(self.target_dir)}
            dev_results = {}

            for tool in self.dev_suite.projection_system.tools_generated:
                print(f"  Running: {tool.metadata.name}...", end=" ")
                try:
                    output = tool.execute(context)
                    dev_results[tool.metadata.name] = output
                    print("✓")
                except Exception as e:
                    print(f"✗ ({e})")

            # Extract lessons
            l2_lessons = self.extractor.extract_from_dev_tools(dev_results)

            results['level_2'] = {
                'tools_run': len(dev_results),
                'lessons': len(l2_lessons)
            }

            print(f"  Tools run: {results['level_2']['tools_run']}")
            print(f"  Lessons extracted: {len(l2_lessons)}")

        except Exception as e:
            print(f"  Level 2 error: {e}")
            l2_lessons = []
            results['level_2'] = {'error': str(e)}

        # =========================
        # HIERARCHICAL TRAINING
        # =========================
        print(f"\n{'─'*60}")
        print("HIERARCHICAL WEIGHT TRAINING")
        print(f"{'─'*60}")

        train_results = self.trainer.train_hierarchical(
            level_0_lessons=self.extractor.level_0_lessons,
            level_1_lessons=self.extractor.level_1_lessons,
            level_2_lessons=self.extractor.level_2_lessons,
            learning_rate=0.1
        )

        results['training'] = train_results

        print(f"  Level 0 updates: {train_results['level_0']['updates']}")
        print(f"  Level 1 updates: {train_results['level_1']['updates']}")
        print(f"  Level 2 updates: {train_results['level_2']['updates']}")
        print(f"  Cross-level updates: {train_results['cross_level_updates']}")

        # =========================
        # EXPONENTIAL COMPOUNDING
        # =========================
        print(f"\n{'─'*60}")
        print("EXPONENTIAL TRAINING (Compounding)")
        print(f"{'─'*60}")

        # Inject accumulated knowledge
        total_lessons = len(self.extractor.get_all_lessons())
        total_importance = sum(l.importance for l in self.extractor.get_all_lessons())

        for learner in self.exp_loop.physical_learners:
            learner.accumulated_knowledge = total_importance / len(self.exp_loop.physical_learners)
            learner.lessons_learned = total_lessons // len(self.exp_loop.physical_learners)

        exp_results = self.exp_loop.run_training(n_runs=training_runs, cycles_per_run=2)

        # =========================
        # FINAL SUMMARY
        # =========================
        print(f"\n{'='*70}")
        print("HIERARCHICAL TRAINING COMPLETE")
        print(f"{'='*70}")

        lesson_summary = self.extractor.get_lesson_summary()
        weight_summary = self.trainer.get_hierarchical_summary()

        results['weights'] = weight_summary

        print(f"""
Lessons by Level:
  Level 0 (Liminal):  {lesson_summary['level_0_liminal']['count']:3d} lessons ({lesson_summary['level_0_liminal']['total_importance']:.2f} importance)
  Level 1 (Meta):     {lesson_summary['level_1_meta']['count']:3d} lessons ({lesson_summary['level_1_meta']['total_importance']:.2f} importance)
  Level 2 (Dev):      {lesson_summary['level_2_dev']['count']:3d} lessons ({lesson_summary['level_2_dev']['total_importance']:.2f} importance)
  ─────────────────────────────────────────
  Total:              {lesson_summary['total']['count']:3d} lessons ({lesson_summary['total']['total_importance']:.2f} importance)

Weights Trained:
  Liminal weights:    {weight_summary['liminal_weights']}
  Meta weights:       {weight_summary['meta_weights']}
  Coupling weights:   {weight_summary['coupling_weights']}
  Frequency biases:   {weight_summary['frequency_biases']}

Cross-Level Coupling:
  Liminal → Meta:     {weight_summary['cross_level_coupling']['liminal_to_meta']:.3f}
  Meta → Dev:         {weight_summary['cross_level_coupling']['meta_to_dev']:.3f}
  Dev → Liminal:      {weight_summary['cross_level_coupling']['dev_to_liminal']:.3f}

Exponential Training:
  Final quality:      {exp_results['final_quality']:.4f}
  Improvement ratio:  {exp_results['improvement_ratio']:.2f}x

Weights saved to:
  {self.weights_dir}/lesson_weights.json
  {self.weights_dir}/hierarchical_weights.json
""")

        # Save results
        results_file = self.weights_dir / "hierarchical_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run hierarchical training pipeline"""
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."

    pipeline = HierarchicalTrainingPipeline(target_dir=target)
    results = pipeline.run_full_hierarchical_pipeline(
        meta_meta_cycles=3,
        meta_collapses=5,
        training_runs=3
    )

    print(f"\nResults saved to weights/hierarchical_training_results.json")

    return results


if __name__ == '__main__':
    main()
