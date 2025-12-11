# Backwards compatibility stub - module moved to training/
"""This module has been moved to training/nightly_training_runner.py"""
from training.nightly_training_runner import *

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Run nightly training')
    parser.add_argument('--output', '-o', default='artifacts/nightly-training',
                       help='Output directory')
    parser.add_argument('--force-runs', '-r', type=int, default=None,
                       help='Force specific number of runs (override coherence)')
    args = parser.parse_args()

    from training.nightly_training_runner import run_nightly_training
    results = run_nightly_training(output_dir=args.output)
    sys.exit(0 if results['status'] == 'success' else 1)
