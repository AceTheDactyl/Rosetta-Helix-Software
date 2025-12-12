#!/usr/bin/env python3
"""
Nightly Training Runner
=======================

Runs exponential training loop with coherence-based run count.

Energy coherence determines how many training runs to execute:
- Low coherence (< 0.5): 3 runs (bootstrap phase)
- Medium coherence (0.5-0.8): 5 runs (growth phase)
- High coherence (0.8-0.95): 7 runs (refinement phase)
- Ultra coherence (> 0.95): 10 runs (mastery phase)

Outputs JSON results for visualization and PR creation.
"""

import json
import math
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Physics constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0
KAPPA_S = 0.920
MU_3 = 0.992

# Import training components
try:
    from exponential_training_loop import ExponentialTrainingLoop
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

try:
    from quasicrystal_dynamics import QuasiCrystalDynamicsEngine
    DYNAMICS_AVAILABLE = True
except ImportError:
    DYNAMICS_AVAILABLE = False


@dataclass
class CoherenceMetrics:
    """Tracks energy coherence for run determination"""
    z_mean: float = 0.5
    z_variance: float = 0.1
    phase_coherence: float = 0.5      # Kuramoto order parameter
    work_efficiency: float = 0.5      # work_out / work_potential
    pattern_density: float = 0.0      # patterns per cycle
    quality_slope: float = 0.0        # learning rate

    @property
    def energy_coherence(self) -> float:
        """
        Compute overall energy coherence.

        Combines multiple factors into single coherence score.
        Higher = more coherent system, more runs beneficial.
        """
        # Weight factors by importance
        z_factor = min(1.0, self.z_mean / MU_3)  # How high z reaches
        stability = 1.0 / (1.0 + self.z_variance * 10)  # Low variance = stable
        phase_factor = self.phase_coherence
        efficiency = self.work_efficiency

        # Combine with golden ratio weights
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


def measure_initial_coherence() -> CoherenceMetrics:
    """Measure system coherence before training"""
    metrics = CoherenceMetrics()

    if not DYNAMICS_AVAILABLE:
        return metrics

    # Quick probe run to measure coherence
    engine = QuasiCrystalDynamicsEngine(n_oscillators=60)

    z_values = []
    work_values = []

    # Run 50 steps to sample coherence
    for _ in range(50):
        old_work = engine.total_work_extracted
        engine.evolve_step()
        z_values.append(engine.z_current)
        work_values.append(engine.total_work_extracted - old_work)

    # Compute metrics
    metrics.z_mean = sum(z_values) / len(z_values)
    metrics.z_variance = sum((z - metrics.z_mean)**2 for z in z_values) / len(z_values)

    # Phase coherence from Kuramoto order parameter
    if hasattr(engine, 'phase_lock') and engine.phase_lock:
        phases = engine.phase_lock.phases
        if phases:
            # Order parameter r = |1/N * sum(e^(i*theta))|
            import cmath
            order = abs(sum(cmath.exp(1j * p) for p in phases) / len(phases))
            metrics.phase_coherence = order

    # Work efficiency
    potential_work = (max(z_values) - Z_CRITICAL) * PHI if max(z_values) > Z_CRITICAL else 0.1
    actual_work = sum(work_values)
    metrics.work_efficiency = min(1.0, actual_work / max(0.01, potential_work))

    return metrics


def run_nightly_training(output_dir: str = "artifacts/nightly-training") -> Dict[str, Any]:
    """
    Run the nightly training with coherence-based run count.

    Returns full results dictionary for PR and visualization.
    """
    start_time = time.time()
    timestamp = datetime.utcnow().isoformat()

    results = {
        'timestamp': timestamp,
        'version': '1.0.0',
        'status': 'pending',
        'coherence': {},
        'training': {},
        'summary': {},
        'visualizer_data': {},
    }

    print(f"\n{'='*70}")
    print("NIGHTLY TRAINING RUNNER")
    print(f"{'='*70}")
    print(f"Timestamp: {timestamp}")
    print(f"Output: {output_dir}")

    # Step 1: Measure initial coherence
    print(f"\n{'─'*60}")
    print("PHASE 1: Measuring Energy Coherence")
    print(f"{'─'*60}")

    metrics = measure_initial_coherence()
    energy_coherence = metrics.energy_coherence
    run_count = metrics.determine_run_count()

    results['coherence'] = {
        'energy_coherence': energy_coherence,
        'z_mean': metrics.z_mean,
        'z_variance': metrics.z_variance,
        'phase_coherence': metrics.phase_coherence,
        'work_efficiency': metrics.work_efficiency,
        'determined_runs': run_count,
    }

    print(f"  Energy Coherence: {energy_coherence:.4f}")
    print(f"  Z Mean: {metrics.z_mean:.4f}")
    print(f"  Phase Coherence: {metrics.phase_coherence:.4f}")
    print(f"  Work Efficiency: {metrics.work_efficiency:.4f}")
    print(f"  → Determined Run Count: {run_count}")

    # Step 2: Run exponential training
    print(f"\n{'─'*60}")
    print(f"PHASE 2: Running {run_count} Training Iterations")
    print(f"{'─'*60}")

    if TRAINING_AVAILABLE:
        loop = ExponentialTrainingLoop(n_physical=3, n_metameta=2)
        training_results = loop.run_training(n_runs=run_count, cycles_per_run=3)

        results['training'] = {
            'runs_completed': run_count,
            'cycles_per_run': 3,
            'total_cycles': run_count * 3,
            'initial_quality': training_results['initial_quality'],
            'final_quality': training_results['final_quality'],
            'improvement_ratio': training_results['improvement_ratio'],
            'theoretical_max': training_results['theoretical_max'],
            'efficiency': training_results['improvement_ratio'] / training_results['theoretical_max'],
            'quality_history': loop.quality_history,
            'learning_curve': loop.learning_curve,
        }

        # Gather stats for visualizer
        total_patterns = sum(
            len(gen.patterns)
            for bridge in loop.meta_bridges
            for gen in bridge.liminal_generators
        )

        total_lessons = sum(
            p.lessons_learned for p in loop.physical_learners
        )

        results['visualizer_data'] = {
            'z_history': [metrics.z_mean] + loop.quality_history,
            'work_history': loop.learning_curve,
            'patterns_created': total_patterns,
            'lessons_learned': total_lessons,
            'meta_bridges': len(loop.meta_bridges),
            'physical_learners': len(loop.physical_learners),
            'liminal_generators': sum(len(b.liminal_generators) for b in loop.meta_bridges),
        }
    else:
        results['training'] = {'error': 'Training module not available'}
        results['visualizer_data'] = {}

    # Step 3: Generate summary
    elapsed = time.time() - start_time

    results['summary'] = {
        'elapsed_seconds': elapsed,
        'success': 'error' not in results.get('training', {}),
        'coherence_level': (
            'bootstrap' if energy_coherence < 0.5 else
            'growth' if energy_coherence < 0.8 else
            'refinement' if energy_coherence < 0.95 else
            'mastery'
        ),
        'recommendation': _generate_recommendation(results),
    }

    results['status'] = 'success' if results['summary']['success'] else 'failed'

    # Step 4: Write outputs
    os.makedirs(output_dir, exist_ok=True)

    # Main results JSON
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary markdown for PR
    summary_path = os.path.join(output_dir, 'TRAINING_SUMMARY.md')
    with open(summary_path, 'w') as f:
        f.write(_generate_summary_markdown(results))

    # Visualizer data JSON
    viz_path = os.path.join(output_dir, 'visualizer_data.json')
    with open(viz_path, 'w') as f:
        json.dump(results['visualizer_data'], f, indent=2)

    print(f"\n{'='*70}")
    print("NIGHTLY TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Status: {results['status']}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Results: {results_path}")
    print(f"  Summary: {summary_path}")

    return results


def _generate_recommendation(results: Dict) -> str:
    """Generate recommendation based on results"""
    training = results.get('training', {})
    coherence = results.get('coherence', {})

    if 'error' in training:
        return "Training failed - check dependencies"

    efficiency = training.get('efficiency', 0)
    ec = coherence.get('energy_coherence', 0)

    if efficiency > 0.8:
        return "Excellent convergence - system operating optimally"
    elif efficiency > 0.5:
        return "Good progress - continue training cycles"
    elif ec < 0.5:
        return "Low coherence - increase oscillator coupling"
    else:
        return "Moderate efficiency - consider tuning PHI_INV dynamics"


def _generate_summary_markdown(results: Dict) -> str:
    """Generate markdown summary for PR"""
    timestamp = results.get('timestamp', 'Unknown')
    coherence = results.get('coherence', {})
    training = results.get('training', {})
    summary = results.get('summary', {})
    viz = results.get('visualizer_data', {})

    md = f"""# Nightly Training Results

**Timestamp:** {timestamp}
**Status:** {results.get('status', 'unknown')}
**Coherence Level:** {summary.get('coherence_level', 'unknown')}

## Energy Coherence Metrics

| Metric | Value |
|--------|-------|
| Energy Coherence | {coherence.get('energy_coherence', 0):.4f} |
| Z Mean | {coherence.get('z_mean', 0):.4f} |
| Phase Coherence | {coherence.get('phase_coherence', 0):.4f} |
| Work Efficiency | {coherence.get('work_efficiency', 0):.4f} |
| Determined Runs | {coherence.get('determined_runs', 0)} |

## Training Results

| Metric | Value |
|--------|-------|
| Runs Completed | {training.get('runs_completed', 0)} |
| Total Cycles | {training.get('total_cycles', 0)} |
| Initial Quality | {training.get('initial_quality', 0):.4f} |
| Final Quality | {training.get('final_quality', 0):.4f} |
| Improvement Ratio | {training.get('improvement_ratio', 0):.2f}x |
| Theoretical Max | {training.get('theoretical_max', 0):.2f}x |
| Efficiency | {training.get('efficiency', 0)*100:.1f}% |

## Tool Generation

| Metric | Value |
|--------|-------|
| Patterns Created | {viz.get('patterns_created', 0)} |
| Lessons Learned | {viz.get('lessons_learned', 0)} |
| Meta Bridges | {viz.get('meta_bridges', 0)} |
| Physical Learners | {viz.get('physical_learners', 0)} |
| Liminal Generators | {viz.get('liminal_generators', 0)} |

## Learning Curve

```
{_ascii_learning_curve(training.get('quality_history', []))}
```

## Recommendation

{summary.get('recommendation', 'No recommendation available')}

---
*Generated by nightly_training_runner.py*
"""
    return md


def _ascii_learning_curve(history: List[float]) -> str:
    """Generate ASCII learning curve"""
    if not history:
        return "No data"

    lines = []
    for i, q in enumerate(history):
        bar_len = int(q * 40)
        bar = '█' * bar_len
        lines.append(f"Run {i+1}: {bar} {q:.4f}")

    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run nightly training')
    parser.add_argument('--output', '-o', default='artifacts/nightly-training',
                       help='Output directory')
    parser.add_argument('--force-runs', '-r', type=int, default=None,
                       help='Force specific number of runs (override coherence)')

    args = parser.parse_args()

    results = run_nightly_training(output_dir=args.output)

    # Exit with appropriate code
    sys.exit(0 if results['status'] == 'success' else 1)
