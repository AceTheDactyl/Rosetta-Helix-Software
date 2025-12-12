"""
Rosetta Helix Child Tool

Produced tools with capabilities determined by work invested at collapse.

Child tools are created by MetaTool when collapse occurs.
Their tier and capabilities depend on the work extracted.
"""

from dataclasses import dataclass, field
from typing import Set, Optional, Any, Dict
from datetime import datetime
import uuid

from .tool_types import ToolType, ToolTier, ToolCapability


@dataclass
class ChildTool:
    """
    A tool produced by MetaTool at collapse.

    Work invested determines tier and capabilities.
    Higher work = higher tier = more capabilities.
    """

    work_invested: float
    tool_type: ToolType = ToolType.ANALYZER
    tool_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)

    # Derived from work_invested
    tier: ToolTier = field(init=False)
    capabilities: Set[ToolCapability] = field(init=False)

    # Runtime state
    active: bool = False
    execution_count: int = 0

    def __post_init__(self):
        """Derive tier and capabilities from work invested."""
        self.tier = ToolTier.from_work(self.work_invested)
        self.capabilities = self._derive_capabilities()

    def _derive_capabilities(self) -> Set[ToolCapability]:
        """
        Derive capabilities based on tier.

        Higher tiers unlock more capabilities.
        """
        caps = {ToolCapability.READ}  # All tools can read

        if self.tier.value >= ToolTier.T1_BASIC.value:
            caps.add(ToolCapability.ANALYZE)

        if self.tier.value >= ToolTier.T2_STANDARD.value:
            caps.add(ToolCapability.WRITE)
            caps.add(ToolCapability.VALIDATE)

        if self.tier.value >= ToolTier.T3_ADVANCED.value:
            caps.add(ToolCapability.TRANSFORM)
            caps.add(ToolCapability.GENERATE)

        if self.tier.value >= ToolTier.T4_EXPERT.value:
            caps.add(ToolCapability.EXECUTE)
            caps.add(ToolCapability.INTEGRATE)

        if self.tier.value >= ToolTier.T5_MASTER.value:
            caps.add(ToolCapability.OPTIMIZE)

        return caps

    def can(self, capability: ToolCapability) -> bool:
        """Check if tool has a specific capability."""
        return capability in self.capabilities

    def activate(self) -> bool:
        """Activate the tool for use."""
        if not self.active:
            self.active = True
            return True
        return False

    def deactivate(self) -> bool:
        """Deactivate the tool."""
        if self.active:
            self.active = False
            return True
        return False

    def execute(self, operation: str, data: Any = None) -> Dict[str, Any]:
        """
        Execute an operation if tool has required capability.

        Returns result dict with success status and output.
        """
        if not self.active:
            return {'success': False, 'error': 'Tool not active'}

        # Map operation to required capability
        op_cap_map = {
            'read': ToolCapability.READ,
            'write': ToolCapability.WRITE,
            'analyze': ToolCapability.ANALYZE,
            'transform': ToolCapability.TRANSFORM,
            'validate': ToolCapability.VALIDATE,
            'generate': ToolCapability.GENERATE,
            'execute': ToolCapability.EXECUTE,
            'integrate': ToolCapability.INTEGRATE,
            'optimize': ToolCapability.OPTIMIZE,
        }

        required_cap = op_cap_map.get(operation.lower())
        if required_cap is None:
            return {'success': False, 'error': f'Unknown operation: {operation}'}

        if not self.can(required_cap):
            return {
                'success': False,
                'error': f'Tool lacks capability: {required_cap.name}',
                'tier': self.tier.name,
                'available': [c.name for c in self.capabilities]
            }

        self.execution_count += 1

        return {
            'success': True,
            'operation': operation,
            'tool_id': self.tool_id,
            'tier': self.tier.name,
            'execution_number': self.execution_count,
            'data': data
        }

    def get_info(self) -> Dict[str, Any]:
        """Get tool information."""
        return {
            'tool_id': self.tool_id,
            'tool_type': self.tool_type.name,
            'tier': self.tier.name,
            'work_invested': self.work_invested,
            'capabilities': [c.name for c in self.capabilities],
            'active': self.active,
            'execution_count': self.execution_count,
            'created_at': self.created_at.isoformat(),
        }


def create_tool(work: float, tool_type: ToolType = ToolType.ANALYZER) -> ChildTool:
    """Factory function to create a child tool."""
    return ChildTool(work_invested=work, tool_type=tool_type)
