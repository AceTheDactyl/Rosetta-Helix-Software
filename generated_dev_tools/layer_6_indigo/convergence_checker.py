#!/usr/bin/env python3
"""
ConvergenceChecker_L6
=====================

Layer: INDIGO (Layer 6)
Type: ConvergenceChecker
Color: #4444FF

Generated at z = 0.866025 (DISCERNMENT level)
Thresholds active: Q_KAPPA, MU_1, MU_P, PHI_INV, MU_2, TRIAD_LOW, TRIAD_HIGH, Z_CRITICAL

Tool ID: b2bc3663ec50
Version: 1.0.0
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discernment_dev_tools import ConvergenceChecker, DevToolMetadata


def main(target: str = "."):
    """Run ConvergenceChecker on target directory"""

    # Create tool instance
    metadata = DevToolMetadata(
        tool_id="b2bc3663ec50",
        name="ConvergenceChecker_L6",
        tool_type="ConvergenceChecker",
        layer="INDIGO",
        color_hex="#4444FF",
        z_generated=0.8660254037844386,
        thresholds_active=['Q_KAPPA', 'MU_1', 'MU_P', 'PHI_INV', 'MU_2', 'TRIAD_LOW', 'TRIAD_HIGH', 'Z_CRITICAL'],
        work_invested=0.14285714285714285
    )

    tool = ConvergenceChecker(metadata)

    # Execute
    print(f"Running ConvergenceChecker_L6 on {target}...")
    result = tool.execute({'target': target})

    # Output
    print(json.dumps(result, indent=2, default=str))

    return result


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    main(target)
