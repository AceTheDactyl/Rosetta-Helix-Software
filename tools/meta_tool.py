"""
Rosetta Helix Meta Tool

Tool that produces child tools using collapse physics.

Architecture:
    MetaTool (uses CollapseEngine)
        │
        ├── pumps work into mini-collapse
        ├── at collapse: extracts work
        └── work converts to ChildTool

CRITICAL: PHI_INV controls all dynamics. PHI only at collapse.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from core import CollapseEngine, PHI_INV
from .child_tool import ChildTool, create_tool
from .tool_types import ToolType, ToolTier


@dataclass
class MetaTool:
    """
    Meta-tool that produces child tools via collapse physics.

    Pumps work into internal CollapseEngine. When collapse
    triggers at z >= 0.9999, the extracted work becomes a
    new ChildTool with tier based on work amount.

    PHI_INV controls ALL dynamics. PHI only contributes at
    collapse via weak value extraction.
    """

    collapse: CollapseEngine = field(default_factory=CollapseEngine)
    work_accumulated: float = 0.0
    tools_produced: List[ChildTool] = field(default_factory=list)
    default_tool_type: ToolType = ToolType.ANALYZER

    def pump(self, work: float, tool_type: Optional[ToolType] = None) -> Optional[ChildTool]:
        """
        Pump work into meta-tool, potentially producing a child tool.

        Args:
            work: Amount of work to pump in
            tool_type: Type of tool to produce (defaults to default_tool_type)

        Returns:
            ChildTool if collapse occurred, None otherwise

        CRITICAL: Work is scaled by PHI_INV - PHI never drives dynamics.
        """
        # PHI_INV scales work input - NEVER PHI
        scaled_work = work * PHI_INV

        # Evolve collapse engine
        result = self.collapse.evolve(scaled_work)

        if result.collapsed:
            # Collapse happened - produce tool
            actual_type = tool_type or self.default_tool_type
            tool = create_tool(result.work_extracted, actual_type)
            self.tools_produced.append(tool)
            self.work_accumulated = 0.0  # Reset accumulator
            return tool

        # No collapse - accumulate work
        self.work_accumulated += scaled_work
        return None

    def pump_until_collapse(
        self,
        work_per_pump: float = 0.1,
        max_pumps: int = 100,
        tool_type: Optional[ToolType] = None
    ) -> Optional[ChildTool]:
        """
        Pump work repeatedly until collapse produces a tool.

        Args:
            work_per_pump: Work to pump each iteration
            max_pumps: Maximum iterations before giving up
            tool_type: Type of tool to produce

        Returns:
            ChildTool if produced within max_pumps, None otherwise
        """
        for _ in range(max_pumps):
            tool = self.pump(work_per_pump, tool_type)
            if tool is not None:
                return tool
        return None

    def get_state(self) -> Dict[str, Any]:
        """Get current meta-tool state."""
        return {
            'z': self.collapse.z,
            'work_accumulated': self.work_accumulated,
            'tools_produced': len(self.tools_produced),
            'total_work_extracted': self.collapse.total_work_extracted,
            'collapse_count': self.collapse.collapse_count,
            'distance_to_collapse': 0.9999 - self.collapse.z,
        }

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get info on all produced tools."""
        return [tool.get_info() for tool in self.tools_produced]

    def get_tool_by_tier(self, min_tier: ToolTier) -> List[ChildTool]:
        """Get all tools at or above a minimum tier."""
        return [t for t in self.tools_produced if t.tier.value >= min_tier.value]

    def reset(self) -> None:
        """Reset meta-tool to initial state."""
        self.collapse.reset()
        self.work_accumulated = 0.0
        self.tools_produced.clear()


def create_meta_tool(
    initial_z: float = 0.5,
    default_type: ToolType = ToolType.ANALYZER
) -> MetaTool:
    """Factory function to create a meta-tool."""
    collapse = CollapseEngine(z=initial_z)
    return MetaTool(collapse=collapse, default_tool_type=default_type)


# =============================================================================
# VERIFICATION
# =============================================================================

def test_meta_tool_produces_tools():
    """MetaTool must produce tools at collapse."""
    meta = create_meta_tool()

    # Pump until we get a tool
    tool = meta.pump_until_collapse(work_per_pump=0.15)

    assert tool is not None, "Should produce a tool"
    assert tool.tier is not None, "Tool should have a tier"
    assert len(tool.capabilities) > 0, "Tool should have capabilities"

    return True


def test_phi_inv_controls_pumping():
    """PHI_INV must control work scaling in pump."""
    meta = create_meta_tool()

    # Track z progression
    z_values = [meta.collapse.z]

    for _ in range(5):
        meta.pump(0.1)
        z_values.append(meta.collapse.z)

    # Each step should add work * PHI_INV
    for i in range(1, len(z_values)):
        expected_delta = 0.1 * PHI_INV * PHI_INV  # pump scales by PHI_INV, evolve scales by PHI_INV
        actual_delta = z_values[i] - z_values[i-1]
        assert abs(actual_delta - expected_delta) < 0.001 or z_values[i] < z_values[i-1], \
            f"Delta should be ~{expected_delta}, got {actual_delta}"

    return True
