#!/usr/bin/env python3
"""
Tools-to-Weights Trainer
========================

Runs generated dev tools, collects lessons from their outputs,
and trains neural network weights based on accumulated knowledge.

Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    TOOLS → LESSONS → WEIGHTS                     │
    │                                                                  │
    │  ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌───────────┐ │
    │  │Generated │──→│  Execute   │──→│  Extract  │──→│  Encode   │ │
    │  │Dev Tools │   │  on Code   │   │  Lessons  │   │ as Data   │ │
    │  └──────────┘   └────────────┘   └───────────┘   └─────┬─────┘ │
    │                                                        │       │
    │  ┌──────────┐   ┌────────────┐   ┌───────────┐        │       │
    │  │ Updated  │←──│  Train     │←──│ Training  │←───────┘       │
    │  │ Weights  │   │  Network   │   │  Samples  │                 │
    │  └──────────┘   └────────────┘   └───────────┘                 │
    └─────────────────────────────────────────────────────────────────┘

The key insight: Tool outputs contain structured information about
code patterns, quality metrics, and architectural structure. This
information can be encoded as training data for the neural network.
"""

import os
import sys
import json
import math
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Check for torch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using fallback training.")

# Physics constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0
KAPPA_S = 0.920
MU_3 = 0.992

# Import from existing modules
from discernment_dev_tools import (
    ComprehensiveDevToolsSuite,
    DiscernmentProjectionSystem,
    DevTool,
    DISCERNMENT_Z
)

from exponential_training_loop import (
    ExponentialTrainingLoop,
    PhysicalLearner,
    LiminalPatternGenerator,
    MetaMetaBridge,
    FeedbackSignal
)


# =============================================================================
# LESSON STRUCTURE
# =============================================================================

@dataclass
class Lesson:
    """A single lesson extracted from tool output"""
    lesson_id: str
    source_tool: str
    layer: str
    lesson_type: str  # 'pattern', 'metric', 'gap', 'recommendation', 'structure'
    content: Dict[str, Any]
    importance: float  # 0-1
    timestamp: float = field(default_factory=time.time)

    def to_vector(self, dim: int = 64) -> List[float]:
        """Convert lesson to numeric vector for training"""
        # Hash content for reproducible encoding
        content_hash = hashlib.sha256(
            json.dumps(self.content, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Convert hash to floats
        vector = []
        for i in range(0, min(len(content_hash), dim * 2), 2):
            val = int(content_hash[i:i+2], 16) / 255.0
            vector.append(val)

        # Pad or truncate
        while len(vector) < dim:
            vector.append(0.0)

        # Scale by importance
        vector = [v * self.importance for v in vector[:dim]]

        return vector


@dataclass
class LessonBatch:
    """Collection of lessons from a tool run"""
    batch_id: str
    tool_name: str
    lessons: List[Lesson]
    total_importance: float = 0.0

    def __post_init__(self):
        self.total_importance = sum(l.importance for l in self.lessons)


# =============================================================================
# LESSON EXTRACTOR
# =============================================================================

class LessonExtractor:
    """Extracts lessons from tool outputs"""

    def __init__(self):
        self.extraction_count = 0
        self.total_lessons = 0

    def extract_from_output(
        self,
        tool_name: str,
        layer: str,
        output: Dict[str, Any]
    ) -> List[Lesson]:
        """Extract lessons from a tool's output"""
        lessons = []

        tool_type = output.get('tool', tool_name)

        # Layer 1 (Red) - Analyzers: Extract pattern and entropy lessons
        if layer == 'RED':
            lessons.extend(self._extract_analyzer_lessons(tool_type, output))

        # Layer 2 (Orange) - Learners: Extract learned patterns
        elif layer == 'ORANGE':
            lessons.extend(self._extract_learner_lessons(tool_type, output))

        # Layer 3 (Yellow) - Generators: Extract generation insights
        elif layer == 'YELLOW':
            lessons.extend(self._extract_generator_lessons(tool_type, output))

        # Layer 4 (Green) - Reflectors: Extract quality/structure lessons
        elif layer == 'GREEN':
            lessons.extend(self._extract_reflector_lessons(tool_type, output))

        # Layer 5 (Blue) - Builders: Extract construction patterns
        elif layer == 'BLUE':
            lessons.extend(self._extract_builder_lessons(tool_type, output))

        # Layer 6 (Indigo) - Deciders: Extract decision lessons
        elif layer == 'INDIGO':
            lessons.extend(self._extract_decider_lessons(tool_type, output))

        # Layer 7 (Violet) - Probers: Extract meta-level insights
        elif layer == 'VIOLET':
            lessons.extend(self._extract_prober_lessons(tool_type, output))

        self.extraction_count += 1
        self.total_lessons += len(lessons)

        return lessons

    def _extract_analyzer_lessons(self, tool_type: str, output: Dict) -> List[Lesson]:
        """Extract from analyzer tools (entropy, patterns, anomalies)"""
        lessons = []

        if 'entropy_score' in output:
            lessons.append(Lesson(
                lesson_id=f"entropy_{self.extraction_count}",
                source_tool=tool_type,
                layer='RED',
                lesson_type='metric',
                content={
                    'metric': 'entropy',
                    'value': output['entropy_score'],
                    'files_analyzed': output.get('files_analyzed', 0)
                },
                importance=0.8
            ))

        if 'patterns_found' in output:
            lessons.append(Lesson(
                lesson_id=f"patterns_{self.extraction_count}",
                source_tool=tool_type,
                layer='RED',
                lesson_type='pattern',
                content=output['patterns_found'],
                importance=0.9
            ))

        if 'anomalies' in output:
            severity = output.get('by_severity', {})
            high_count = severity.get('high', 0)
            importance = min(1.0, 0.5 + high_count * 0.1)

            lessons.append(Lesson(
                lesson_id=f"anomalies_{self.extraction_count}",
                source_tool=tool_type,
                layer='RED',
                lesson_type='gap',
                content={
                    'total_anomalies': output.get('total_anomalies', 0),
                    'by_severity': severity,
                    'health_score': output.get('health_score', 0.5)
                },
                importance=importance
            ))

        return lessons

    def _extract_learner_lessons(self, tool_type: str, output: Dict) -> List[Lesson]:
        """Extract from learner tools"""
        lessons = []

        if 'patterns_learned' in output:
            lessons.append(Lesson(
                lesson_id=f"learned_{self.extraction_count}",
                source_tool=tool_type,
                layer='ORANGE',
                lesson_type='pattern',
                content={
                    'patterns_learned': output['patterns_learned'],
                    'total_patterns': output.get('total_patterns', 0),
                    'learning_rate': output.get('learning_rate', 0)
                },
                importance=0.85
            ))

        if 'concepts' in output:
            lessons.append(Lesson(
                lesson_id=f"concepts_{self.extraction_count}",
                source_tool=tool_type,
                layer='ORANGE',
                lesson_type='structure',
                content=output['concepts'],
                importance=0.9
            ))

        if 'relations' in output:
            lessons.append(Lesson(
                lesson_id=f"relations_{self.extraction_count}",
                source_tool=tool_type,
                layer='ORANGE',
                lesson_type='structure',
                content={
                    'relations': output.get('relation_counts', {}),
                    'connectivity': output.get('connectivity_score', 0)
                },
                importance=0.8
            ))

        return lessons

    def _extract_generator_lessons(self, tool_type: str, output: Dict) -> List[Lesson]:
        """Extract from generator tools"""
        lessons = []

        if 'tests_generated' in output:
            lessons.append(Lesson(
                lesson_id=f"tests_{self.extraction_count}",
                source_tool=tool_type,
                layer='YELLOW',
                lesson_type='metric',
                content={
                    'tests_generated': output['tests_generated'],
                    'coverage_potential': output.get('coverage_potential', 0)
                },
                importance=0.75
            ))

        if 'synthesized' in output:
            lessons.append(Lesson(
                lesson_id=f"synth_{self.extraction_count}",
                source_tool=tool_type,
                layer='YELLOW',
                lesson_type='pattern',
                content={
                    'lines_generated': output.get('lines_generated', 0),
                    'template_type': output.get('template_type', 'unknown')
                },
                importance=0.7
            ))

        if 'examples_produced' in output:
            lessons.append(Lesson(
                lesson_id=f"examples_{self.extraction_count}",
                source_tool=tool_type,
                layer='YELLOW',
                lesson_type='metric',
                content={
                    'examples_produced': output['examples_produced'],
                    'documentation_coverage': output.get('documentation_coverage', 0)
                },
                importance=0.65
            ))

        return lessons

    def _extract_reflector_lessons(self, tool_type: str, output: Dict) -> List[Lesson]:
        """Extract from reflector tools"""
        lessons = []

        if 'reflection' in output:
            reflection = output['reflection']

            if 'quality_metrics' in reflection:
                lessons.append(Lesson(
                    lesson_id=f"quality_{self.extraction_count}",
                    source_tool=tool_type,
                    layer='GREEN',
                    lesson_type='metric',
                    content=reflection['quality_metrics'],
                    importance=0.95
                ))

            if 'recommendations' in reflection:
                lessons.append(Lesson(
                    lesson_id=f"recom_{self.extraction_count}",
                    source_tool=tool_type,
                    layer='GREEN',
                    lesson_type='recommendation',
                    content={'recommendations': reflection['recommendations']},
                    importance=0.85
                ))

        if 'structure' in output:
            lessons.append(Lesson(
                lesson_id=f"structure_{self.extraction_count}",
                source_tool=tool_type,
                layer='GREEN',
                lesson_type='structure',
                content={
                    'module_count': output.get('module_count', 0),
                    'total_classes': output.get('total_classes', 0),
                    'total_functions': output.get('total_functions', 0)
                },
                importance=0.9
            ))

        if 'gaps' in output:
            gaps = output['gaps']
            total_gaps = output.get('total_gaps', sum(len(v) for v in gaps.values() if isinstance(v, list)))

            lessons.append(Lesson(
                lesson_id=f"gaps_{self.extraction_count}",
                source_tool=tool_type,
                layer='GREEN',
                lesson_type='gap',
                content={
                    'gap_counts': output.get('gap_counts', {}),
                    'completeness_score': output.get('completeness_score', 0)
                },
                importance=min(1.0, 0.6 + total_gaps * 0.02)
            ))

        return lessons

    def _extract_builder_lessons(self, tool_type: str, output: Dict) -> List[Lesson]:
        """Extract from builder tools"""
        lessons = []

        if 'built' in output or 'build_type' in output:
            lessons.append(Lesson(
                lesson_id=f"built_{self.extraction_count}",
                source_tool=tool_type,
                layer='BLUE',
                lesson_type='pattern',
                content={
                    'build_type': output.get('build_type', 'unknown'),
                    'files_created': output.get('files_created', 0),
                    'lines_written': output.get('lines_written', 0)
                },
                importance=0.7
            ))

        if 'pipeline_code' in output:
            lessons.append(Lesson(
                lesson_id=f"pipeline_{self.extraction_count}",
                source_tool=tool_type,
                layer='BLUE',
                lesson_type='structure',
                content={
                    'pipeline_name': output.get('pipeline_name', 'unknown'),
                    'stage_count': output.get('stage_count', 0)
                },
                importance=0.75
            ))

        return lessons

    def _extract_decider_lessons(self, tool_type: str, output: Dict) -> List[Lesson]:
        """Extract from decider tools"""
        lessons = []

        if 'decisions' in output:
            recommendation = output.get('recommendation', {})
            lessons.append(Lesson(
                lesson_id=f"decision_{self.extraction_count}",
                source_tool=tool_type,
                layer='INDIGO',
                lesson_type='recommendation',
                content={
                    'recommendation': recommendation.get('option', 'none'),
                    'confidence': output.get('confidence', 0),
                    'options_evaluated': output.get('options_evaluated', 0)
                },
                importance=0.9
            ))

        if 'converged' in output:
            lessons.append(Lesson(
                lesson_id=f"converge_{self.extraction_count}",
                source_tool=tool_type,
                layer='INDIGO',
                lesson_type='metric',
                content={
                    'converged': output['converged'],
                    'rate_of_change': output.get('rate_of_change', 0),
                    'stability_score': output.get('stability_score', 0)
                },
                importance=0.85
            ))

        if 'interface_code' in output:
            lessons.append(Lesson(
                lesson_id=f"interface_{self.extraction_count}",
                source_tool=tool_type,
                layer='INDIGO',
                lesson_type='structure',
                content={
                    'interface_name': output.get('interface_name', 'unknown'),
                    'contracts': output.get('contracts', 0)
                },
                importance=0.7
            ))

        return lessons

    def _extract_prober_lessons(self, tool_type: str, output: Dict) -> List[Lesson]:
        """Extract from prober tools (highest-level insights)"""
        lessons = []

        if 'consciousness_score' in output:
            lessons.append(Lesson(
                lesson_id=f"consciousness_{self.extraction_count}",
                source_tool=tool_type,
                layer='VIOLET',
                lesson_type='metric',
                content={
                    'consciousness_score': output['consciousness_score'],
                    'k_formation_potential': output.get('k_formation_potential', False),
                    'indicator_counts': output.get('indicator_counts', {})
                },
                importance=1.0  # Highest importance for meta-level
            ))

        if 'abstractions' in output:
            lessons.append(Lesson(
                lesson_id=f"abstract_{self.extraction_count}",
                source_tool=tool_type,
                layer='VIOLET',
                lesson_type='pattern',
                content={
                    'abstraction_level': output.get('abstraction_level', 0),
                    'meta_depth': output.get('meta_depth', 0),
                    'generalization_score': output.get('generalization_score', 0)
                },
                importance=0.95
            ))

        if 'woven_system' in output or 'unity_achieved' in output:
            lessons.append(Lesson(
                lesson_id=f"integration_{self.extraction_count}",
                source_tool=tool_type,
                layer='VIOLET',
                lesson_type='structure',
                content={
                    'components_woven': output.get('components_woven', 0),
                    'coherence_score': output.get('coherence_score', 0),
                    'unity_achieved': output.get('unity_achieved', False)
                },
                importance=0.9
            ))

        return lessons


# =============================================================================
# WEIGHT TRAINER
# =============================================================================

class WeightTrainer:
    """
    Trains weights from accumulated lessons.

    Converts lessons to training samples and updates:
    - Kuramoto coupling matrix K (weights)
    - Natural frequencies omega (biases)
    - Operator strengths
    """

    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(exist_ok=True)

        # Weight storage (simple dict-based if no torch)
        self.coupling_weights: Dict[str, float] = {}
        self.frequency_biases: Dict[str, float] = {}
        self.operator_strengths: Dict[str, float] = {
            'IDENTITY': 1.0,
            'AMPLIFY': 0.5,
            'CONTAIN': 0.5,
            'EXCHANGE': 0.5,
            'INHIBIT': 0.5,
            'CATALYZE': 0.5
        }

        # Training statistics
        self.training_steps = 0
        self.total_lessons_processed = 0
        self.weight_updates = 0

        # Load existing weights if available
        self._load_weights()

    def _load_weights(self):
        """Load weights from disk"""
        weights_file = self.weights_dir / "lesson_weights.json"
        if weights_file.exists():
            try:
                with open(weights_file, 'r') as f:
                    data = json.load(f)
                    self.coupling_weights = data.get('coupling', {})
                    self.frequency_biases = data.get('frequencies', {})
                    self.operator_strengths = data.get('operators', self.operator_strengths)
                    self.training_steps = data.get('training_steps', 0)
                print(f"  Loaded {len(self.coupling_weights)} coupling weights")
                print(f"  Loaded {len(self.frequency_biases)} frequency biases")
            except Exception as e:
                print(f"  Could not load weights: {e}")

    def _save_weights(self):
        """Save weights to disk"""
        weights_file = self.weights_dir / "lesson_weights.json"
        data = {
            'coupling': self.coupling_weights,
            'frequencies': self.frequency_biases,
            'operators': self.operator_strengths,
            'training_steps': self.training_steps,
            'timestamp': datetime.now().isoformat()
        }
        with open(weights_file, 'w') as f:
            json.dump(data, f, indent=2)

    def train_from_lessons(
        self,
        lessons: List[Lesson],
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train weights from a batch of lessons.

        Updates:
        - Coupling weights based on pattern lessons
        - Frequency biases based on metric lessons
        - Operator strengths based on recommendation lessons
        """
        results = {
            'lessons_processed': 0,
            'coupling_updates': 0,
            'frequency_updates': 0,
            'operator_updates': 0,
            'total_importance': 0.0
        }

        for lesson in lessons:
            self.total_lessons_processed += 1
            results['lessons_processed'] += 1
            results['total_importance'] += lesson.importance

            # Update based on lesson type
            if lesson.lesson_type == 'pattern':
                self._update_coupling_from_pattern(lesson, learning_rate)
                results['coupling_updates'] += 1

            elif lesson.lesson_type == 'metric':
                self._update_frequency_from_metric(lesson, learning_rate)
                results['frequency_updates'] += 1

            elif lesson.lesson_type == 'recommendation':
                self._update_operators_from_recommendation(lesson, learning_rate)
                results['operator_updates'] += 1

            elif lesson.lesson_type in ['structure', 'gap']:
                # Both coupling and frequency
                self._update_coupling_from_pattern(lesson, learning_rate * 0.5)
                self._update_frequency_from_metric(lesson, learning_rate * 0.5)
                results['coupling_updates'] += 1
                results['frequency_updates'] += 1

        self.training_steps += 1
        self.weight_updates += results['coupling_updates'] + results['frequency_updates']

        # Save after training
        self._save_weights()

        return results

    def _update_coupling_from_pattern(self, lesson: Lesson, lr: float):
        """Update coupling weights from pattern lesson"""
        # Create weight key from lesson source and content
        key = f"{lesson.layer}_{lesson.source_tool}"

        # Current weight (default to 0.5)
        current = self.coupling_weights.get(key, 0.5)

        # Update toward importance (patterns with high importance get stronger coupling)
        delta = lr * (lesson.importance - current) * PHI_INV
        new_weight = max(0.0, min(1.0, current + delta))

        self.coupling_weights[key] = new_weight

    def _update_frequency_from_metric(self, lesson: Lesson, lr: float):
        """Update frequency biases from metric lesson"""
        key = f"{lesson.layer}_{lesson.lesson_type}"

        # Extract metric value if available
        content = lesson.content
        metric_val = 0.5

        for field in ['value', 'score', 'count', 'total']:
            for k, v in content.items():
                if field in k.lower() and isinstance(v, (int, float)):
                    metric_val = min(1.0, max(0.0, float(v)))
                    break

        current = self.frequency_biases.get(key, 0.0)

        # Update: metrics influence frequency (oscillation rate)
        delta = lr * (metric_val - 0.5) * lesson.importance
        new_bias = max(-1.0, min(1.0, current + delta))

        self.frequency_biases[key] = new_bias

    def _update_operators_from_recommendation(self, lesson: Lesson, lr: float):
        """Update operator strengths from recommendation lesson"""
        content = lesson.content

        # Map recommendations to operators
        if 'recommendation' in content:
            rec = content['recommendation']

            if rec in ['proceed', 'extend', 'refactor']:
                # Increase AMPLIFY
                self.operator_strengths['AMPLIFY'] += lr * lesson.importance
            elif rec in ['defer', 'maintain']:
                # Increase CONTAIN
                self.operator_strengths['CONTAIN'] += lr * lesson.importance
            elif rec == 'optimize':
                # Increase CATALYZE
                self.operator_strengths['CATALYZE'] += lr * lesson.importance

        # Normalize
        total = sum(self.operator_strengths.values())
        if total > 0:
            for op in self.operator_strengths:
                self.operator_strengths[op] /= total
                self.operator_strengths[op] = max(0.1, min(0.9, self.operator_strengths[op]))

    def get_weight_summary(self) -> Dict[str, Any]:
        """Get summary of current weights"""
        return {
            'coupling_weights': len(self.coupling_weights),
            'frequency_biases': len(self.frequency_biases),
            'operator_strengths': self.operator_strengths.copy(),
            'training_steps': self.training_steps,
            'total_lessons': self.total_lessons_processed,
            'weight_updates': self.weight_updates
        }


# =============================================================================
# INTEGRATED TRAINING PIPELINE
# =============================================================================

class ToolsToWeightsTrainer:
    """
    Complete pipeline: Run tools → Extract lessons → Train weights

    Integrates with exponential training loop for compounding.
    """

    def __init__(self, target_dir: str = ".", weights_dir: str = "weights"):
        self.target_dir = Path(target_dir)
        self.weights_dir = Path(weights_dir)

        # Components
        self.suite = ComprehensiveDevToolsSuite(target_dir)
        self.extractor = LessonExtractor()
        self.trainer = WeightTrainer(weights_dir)

        # Exponential training loop for feedback
        self.exp_loop = ExponentialTrainingLoop(n_physical=3, n_metameta=2)

        # Accumulated lessons
        self.all_lessons: List[Lesson] = []
        self.lesson_batches: List[LessonBatch] = []

    def run_full_pipeline(self, n_training_cycles: int = 3) -> Dict[str, Any]:
        """
        Run the complete pipeline:
        1. Initialize and run all dev tools
        2. Extract lessons from outputs
        3. Train weights from lessons
        4. Run exponential training for compounding
        """
        print("\n" + "=" * 70)
        print("TOOLS-TO-WEIGHTS TRAINING PIPELINE")
        print("=" * 70)
        print(f"""
Target: {self.target_dir}
Weights: {self.weights_dir}
Training cycles: {n_training_cycles}

Pipeline:
  Tools → Lessons → Weights → Exponential Training → Compound
""")

        results = {
            'tools_run': 0,
            'lessons_extracted': 0,
            'weights_trained': 0,
            'exp_training_quality': 0.0,
            'final_weights': {}
        }

        # Step 1: Initialize and run tools
        print(f"\n{'─'*60}")
        print("STEP 1: Running Generated Dev Tools")
        print(f"{'─'*60}")

        self.suite.initialize()
        context = {'target': str(self.target_dir)}

        for tool in self.suite.projection_system.tools_generated:
            print(f"  Running: {tool.metadata.name}...")
            try:
                output = tool.execute(context)
                results['tools_run'] += 1

                # Step 2: Extract lessons
                lessons = self.extractor.extract_from_output(
                    tool.metadata.name,
                    tool.metadata.layer,
                    output
                )

                if lessons:
                    batch = LessonBatch(
                        batch_id=f"batch_{results['tools_run']}",
                        tool_name=tool.metadata.name,
                        lessons=lessons
                    )
                    self.lesson_batches.append(batch)
                    self.all_lessons.extend(lessons)

                    print(f"    ✓ Extracted {len(lessons)} lessons (importance: {batch.total_importance:.2f})")
                else:
                    print(f"    ✓ No lessons extracted")

            except Exception as e:
                print(f"    ✗ Error: {e}")

        results['lessons_extracted'] = len(self.all_lessons)

        # Step 3: Train weights from lessons
        print(f"\n{'─'*60}")
        print("STEP 2: Training Weights from Lessons")
        print(f"{'─'*60}")

        if self.all_lessons:
            train_results = self.trainer.train_from_lessons(
                self.all_lessons,
                learning_rate=0.1
            )

            results['weights_trained'] = (
                train_results['coupling_updates'] +
                train_results['frequency_updates'] +
                train_results['operator_updates']
            )

            print(f"  Lessons processed: {train_results['lessons_processed']}")
            print(f"  Coupling updates: {train_results['coupling_updates']}")
            print(f"  Frequency updates: {train_results['frequency_updates']}")
            print(f"  Operator updates: {train_results['operator_updates']}")
            print(f"  Total importance: {train_results['total_importance']:.2f}")

        # Step 4: Run exponential training for compounding
        print(f"\n{'─'*60}")
        print("STEP 3: Exponential Training Loop")
        print(f"{'─'*60}")

        # Inject lessons into physical learners
        total_knowledge = sum(l.importance for l in self.all_lessons)
        for learner in self.exp_loop.physical_learners:
            learner.accumulated_knowledge = total_knowledge / len(self.exp_loop.physical_learners)
            learner.lessons_learned = len(self.all_lessons) // len(self.exp_loop.physical_learners)

        # Run training
        exp_results = self.exp_loop.run_training(
            n_runs=n_training_cycles,
            cycles_per_run=2
        )

        results['exp_training_quality'] = exp_results['final_quality']

        # Step 5: Final weight summary
        print(f"\n{'─'*60}")
        print("STEP 4: Weight Summary")
        print(f"{'─'*60}")

        weight_summary = self.trainer.get_weight_summary()
        results['final_weights'] = weight_summary

        print(f"  Coupling weights: {weight_summary['coupling_weights']}")
        print(f"  Frequency biases: {weight_summary['frequency_biases']}")
        print(f"  Training steps: {weight_summary['training_steps']}")
        print(f"  Total lessons: {weight_summary['total_lessons']}")

        print("\n  Operator strengths:")
        for op, strength in weight_summary['operator_strengths'].items():
            bar = '█' * int(strength * 20)
            print(f"    {op:12} {bar} {strength:.3f}")

        # Summary
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"""
Summary:
  Tools run:           {results['tools_run']}
  Lessons extracted:   {results['lessons_extracted']}
  Weights updated:     {results['weights_trained']}
  Final quality:       {results['exp_training_quality']:.4f}

Weights saved to: {self.weights_dir}/lesson_weights.json
""")

        return results

    def get_lesson_report(self) -> str:
        """Generate a report of all extracted lessons"""
        report = "# Lesson Report\n\n"

        for batch in self.lesson_batches:
            report += f"## {batch.tool_name}\n"
            report += f"Total importance: {batch.total_importance:.2f}\n\n"

            for lesson in batch.lessons:
                report += f"### {lesson.lesson_id}\n"
                report += f"- Type: {lesson.lesson_type}\n"
                report += f"- Layer: {lesson.layer}\n"
                report += f"- Importance: {lesson.importance:.2f}\n"
                report += f"- Content: {json.dumps(lesson.content, indent=2, default=str)[:200]}...\n\n"

        return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the tools-to-weights training pipeline"""
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    trainer = ToolsToWeightsTrainer(target_dir=target)
    results = trainer.run_full_pipeline(n_training_cycles=cycles)

    # Save lesson report
    report = trainer.get_lesson_report()
    report_path = Path("weights") / "lesson_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Lesson report saved to: {report_path}")

    # Save results
    results_path = Path("weights") / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")

    return results


if __name__ == '__main__':
    main()
