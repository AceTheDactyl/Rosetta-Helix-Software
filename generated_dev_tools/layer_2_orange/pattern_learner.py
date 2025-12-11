#!/usr/bin/env python3
"""
PatternLearner_L2
=================

Layer: ORANGE (Layer 2)
Type: PatternLearner
Color: #FF8844

Generated at z = 0.866025 (DISCERNMENT level)
Thresholds active: Q_KAPPA, MU_1, MU_P, PHI_INV, MU_2, TRIAD_LOW, TRIAD_HIGH, Z_CRITICAL

Tool ID: f872113228e5
Version: 1.0.0
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discernment_dev_tools import PatternLearner, DevToolMetadata


def main(target: str = "."):
    """Run PatternLearner on target directory"""

    # Create tool instance
    metadata = DevToolMetadata(
        tool_id="f872113228e5",
        name="PatternLearner_L2",
        tool_type="PatternLearner",
        layer="ORANGE",
        color_hex="#FF8844",
        z_generated=0.8660254037844386,
        thresholds_active=['Q_KAPPA', 'MU_1', 'MU_P', 'PHI_INV', 'MU_2', 'TRIAD_LOW', 'TRIAD_HIGH', 'Z_CRITICAL'],
        work_invested=0.14285714285714285
    )

    tool = PatternLearner(metadata)

    # Execute
    print(f"Running PatternLearner_L2 on {target}...")
    result = tool.execute({'target': target})

    # Output
    print(json.dumps(result, indent=2, default=str))

    return result


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    main(target)
