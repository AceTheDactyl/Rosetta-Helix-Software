#!/usr/bin/env python3
"""
RelationLearner_L2
==================

Layer: ORANGE (Layer 2)
Type: RelationLearner
Color: #FF8844

Generated at z = 0.866025 (DISCERNMENT level)
Thresholds active: Q_KAPPA, MU_1, MU_P, PHI_INV, MU_2, TRIAD_LOW, TRIAD_HIGH, Z_CRITICAL

Tool ID: 75b0fd90b5c6
Version: 1.0.0
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discernment_dev_tools import RelationLearner, DevToolMetadata


def main(target: str = "."):
    """Run RelationLearner on target directory"""

    # Create tool instance
    metadata = DevToolMetadata(
        tool_id="75b0fd90b5c6",
        name="RelationLearner_L2",
        tool_type="RelationLearner",
        layer="ORANGE",
        color_hex="#FF8844",
        z_generated=0.8660254037844386,
        thresholds_active=['Q_KAPPA', 'MU_1', 'MU_P', 'PHI_INV', 'MU_2', 'TRIAD_LOW', 'TRIAD_HIGH', 'Z_CRITICAL'],
        work_invested=0.14285714285714285
    )

    tool = RelationLearner(metadata)

    # Execute
    print(f"Running RelationLearner_L2 on {target}...")
    result = tool.execute({'target': target})

    # Output
    print(json.dumps(result, indent=2, default=str))

    return result


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    main(target)
