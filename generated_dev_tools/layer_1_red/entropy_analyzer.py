#!/usr/bin/env python3
"""
EntropyAnalyzer_L1
==================

Layer: RED (Layer 1)
Type: EntropyAnalyzer
Color: #FF4444

Generated at z = 0.866025 (DISCERNMENT level)
Thresholds active: Q_KAPPA, MU_1, MU_P, PHI_INV, MU_2, TRIAD_LOW, TRIAD_HIGH, Z_CRITICAL

Tool ID: 5a0a218d77cb
Version: 1.0.0
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discernment_dev_tools import EntropyAnalyzer, DevToolMetadata


def main(target: str = "."):
    """Run EntropyAnalyzer on target directory"""

    # Create tool instance
    metadata = DevToolMetadata(
        tool_id="5a0a218d77cb",
        name="EntropyAnalyzer_L1",
        tool_type="EntropyAnalyzer",
        layer="RED",
        color_hex="#FF4444",
        z_generated=0.8660254037844386,
        thresholds_active=['Q_KAPPA', 'MU_1', 'MU_P', 'PHI_INV', 'MU_2', 'TRIAD_LOW', 'TRIAD_HIGH', 'Z_CRITICAL'],
        work_invested=0.14285714285714285
    )

    tool = EntropyAnalyzer(metadata)

    # Execute
    print(f"Running EntropyAnalyzer_L1 on {target}...")
    result = tool.execute({'target': target})

    # Output
    print(json.dumps(result, indent=2, default=str))

    return result


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    main(target)
