#!/usr/bin/env python3
"""
ExampleProducer_L3
==================

Layer: YELLOW (Layer 3)
Type: ExampleProducer
Color: #FFAA00

Generated at z = 0.866025 (DISCERNMENT level)
Thresholds active: Q_KAPPA, MU_1, MU_P, PHI_INV, MU_2, TRIAD_LOW, TRIAD_HIGH, Z_CRITICAL

Tool ID: 4ae04ce0499d
Version: 1.0.0
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discernment_dev_tools import ExampleProducer, DevToolMetadata


def main(target: str = "."):
    """Run ExampleProducer on target directory"""

    # Create tool instance
    metadata = DevToolMetadata(
        tool_id="4ae04ce0499d",
        name="ExampleProducer_L3",
        tool_type="ExampleProducer",
        layer="YELLOW",
        color_hex="#FFAA00",
        z_generated=0.8660254037844386,
        thresholds_active=['Q_KAPPA', 'MU_1', 'MU_P', 'PHI_INV', 'MU_2', 'TRIAD_LOW', 'TRIAD_HIGH', 'Z_CRITICAL'],
        work_invested=0.14285714285714285
    )

    tool = ExampleProducer(metadata)

    # Execute
    print(f"Running ExampleProducer_L3 on {target}...")
    result = tool.execute({'target': target})

    # Output
    print(json.dumps(result, indent=2, default=str))

    return result


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    main(target)
