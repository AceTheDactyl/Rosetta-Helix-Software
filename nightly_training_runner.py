# Backwards compatibility stub - module moved to training/
"""This module has been moved to training/nightly_training_runner.py"""
from training.nightly_training_runner import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run nightly training')
    parser.add_argument('--output', '-o', default='artifacts/nightly-training',
                       help='Output directory')
    parser.add_argument('--force-runs', '-r', type=int, default=None,
                       help='Force specific number of runs (override coherence)')
    args = parser.parse_args()

    from training.nightly_training_runner import NightlyTrainingRunner
    runner = NightlyTrainingRunner(args.output)
    runner.run(force_runs=args.force_runs)
