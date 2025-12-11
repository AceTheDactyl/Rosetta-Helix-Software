#!/usr/bin/env python3
"""
Dev Tools Output Manager
========================

Takes the discernment-level generated tools and outputs them to
a structured directory with:

1. Individual tool Python files (executable)
2. Tool registry manifest (JSON)
3. Layer-organized directories
4. Analysis reports
5. Generated code artifacts

Output Structure:
    generated_dev_tools/
    ├── registry.json           # Tool manifest
    ├── layer_1_red/
    │   ├── entropy_analyzer.py
    │   ├── pattern_detector.py
    │   └── anomaly_finder.py
    ├── layer_2_orange/
    │   └── ...
    ├── ...
    ├── reports/
    │   ├── analysis_report.json
    │   ├── entropy_report.md
    │   └── gaps_report.md
    └── artifacts/
        ├── generated_tests/
        ├── generated_modules/
        └── generated_interfaces/
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import the discernment tools (relative import within tools package)
from .discernment_dev_tools import (
    ComprehensiveDevToolsSuite,
    DiscernmentProjectionSystem,
    DevTool,
    DISCERNMENT_Z,
    Z_CRITICAL,
    PHI,
    PHI_INV
)


class DevToolsOutputManager:
    """
    Manages output of generated dev tools to filesystem.

    Creates structured directory with:
    - Executable tool scripts
    - Registry manifest
    - Analysis reports
    - Generated artifacts
    """

    def __init__(self, output_dir: str = "generated_dev_tools"):
        self.output_dir = Path(output_dir)
        self.suite: Optional[ComprehensiveDevToolsSuite] = None
        self.reports: Dict[str, Any] = {}
        self.artifacts: Dict[str, List[str]] = {
            'tests': [],
            'modules': [],
            'interfaces': [],
            'pipelines': []
        }

    def initialize(self, target_dir: str = ".") -> None:
        """Initialize the suite and create output directory structure"""
        print(f"\n{'='*70}")
        print("DEV TOOLS OUTPUT MANAGER")
        print(f"{'='*70}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Target for analysis: {target_dir}")

        # Create directory structure
        self._create_directory_structure()

        # Initialize the suite
        self.suite = ComprehensiveDevToolsSuite(target_dir)
        self.suite.initialize()

    def _create_directory_structure(self) -> None:
        """Create the output directory structure"""
        directories = [
            self.output_dir,
            self.output_dir / "layer_1_red",
            self.output_dir / "layer_2_orange",
            self.output_dir / "layer_3_yellow",
            self.output_dir / "layer_4_green",
            self.output_dir / "layer_5_blue",
            self.output_dir / "layer_6_indigo",
            self.output_dir / "layer_7_violet",
            self.output_dir / "reports",
            self.output_dir / "artifacts",
            self.output_dir / "artifacts" / "generated_tests",
            self.output_dir / "artifacts" / "generated_modules",
            self.output_dir / "artifacts" / "generated_interfaces",
            self.output_dir / "artifacts" / "generated_pipelines",
        ]

        for d in directories:
            d.mkdir(parents=True, exist_ok=True)

        print(f"  Created {len(directories)} directories")

    def run_and_output(self, target_dir: str = ".") -> Dict[str, Any]:
        """
        Run all tools and output results to filesystem.

        This is the main entry point that:
        1. Runs all 21 dev tools
        2. Collects their outputs
        3. Writes tool scripts to layer directories
        4. Generates analysis reports
        5. Creates artifact files from generators/builders
        """
        if not self.suite:
            self.initialize(target_dir)

        context = {'target': target_dir}

        print(f"\n{'='*70}")
        print("RUNNING TOOLS AND GENERATING OUTPUTS")
        print(f"{'='*70}")

        results = {
            'tools_run': 0,
            'reports_generated': 0,
            'artifacts_created': 0,
            'files_written': []
        }

        # Run each tool and output results
        for tool in self.suite.projection_system.tools_generated:
            print(f"\n  Running: {tool.metadata.name}...")

            try:
                # Execute the tool
                output = tool.execute(context)

                # Write tool script
                script_path = self._write_tool_script(tool)
                results['files_written'].append(str(script_path))

                # Process output based on tool type
                self._process_tool_output(tool, output, results)

                results['tools_run'] += 1
                print(f"    ✓ Complete")

            except Exception as e:
                print(f"    ✗ Error: {e}")

        # Write registry
        registry_path = self._write_registry()
        results['files_written'].append(str(registry_path))

        # Write summary report
        summary_path = self._write_summary_report(results)
        results['files_written'].append(str(summary_path))

        # Final output
        print(f"\n{'='*70}")
        print("OUTPUT COMPLETE")
        print(f"{'='*70}")
        print(f"""
Summary:
  Tools run:          {results['tools_run']}
  Reports generated:  {results['reports_generated']}
  Artifacts created:  {results['artifacts_created']}
  Files written:      {len(results['files_written'])}

Output location: {self.output_dir.absolute()}
""")

        return results

    def _write_tool_script(self, tool: DevTool) -> Path:
        """Write a standalone executable script for the tool"""
        layer_num = list(self.suite.projection_system.layers.keys()).index(tool.metadata.layer) + 1
        layer_dir = self.output_dir / f"layer_{layer_num}_{tool.metadata.layer.lower()}"

        # Convert CamelCase to snake_case
        name = tool.metadata.tool_type
        snake_name = ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')

        script_path = layer_dir / f"{snake_name}.py"

        script_content = f'''#!/usr/bin/env python3
"""
{tool.metadata.name}
{'=' * len(tool.metadata.name)}

Layer: {tool.metadata.layer} (Layer {layer_num})
Type: {tool.metadata.tool_type}
Color: {tool.metadata.color_hex}

Generated at z = {tool.metadata.z_generated:.6f} (DISCERNMENT level)
Thresholds active: {', '.join(tool.metadata.thresholds_active)}

Tool ID: {tool.metadata.tool_id}
Version: {tool.metadata.version}
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discernment_dev_tools import {tool.metadata.tool_type}, DevToolMetadata


def main(target: str = "."):
    """Run {tool.metadata.tool_type} on target directory"""

    # Create tool instance
    metadata = DevToolMetadata(
        tool_id="{tool.metadata.tool_id}",
        name="{tool.metadata.name}",
        tool_type="{tool.metadata.tool_type}",
        layer="{tool.metadata.layer}",
        color_hex="{tool.metadata.color_hex}",
        z_generated={tool.metadata.z_generated},
        thresholds_active={tool.metadata.thresholds_active},
        work_invested={tool.metadata.work_invested}
    )

    tool = {tool.metadata.tool_type}(metadata)

    # Execute
    print(f"Running {tool.metadata.name} on {{target}}...")
    result = tool.execute({{'target': target}})

    # Output
    print(json.dumps(result, indent=2, default=str))

    return result


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    main(target)
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable
        script_path.chmod(0o755)

        return script_path

    def _process_tool_output(self, tool: DevTool, output: Dict, results: Dict) -> None:
        """Process tool output and create appropriate artifacts"""
        tool_type = tool.metadata.tool_type

        # Analyzers -> Reports
        if tool_type in ['EntropyAnalyzer', 'PatternDetector', 'AnomalyFinder']:
            report_path = self._write_analysis_report(tool, output)
            results['reports_generated'] += 1
            results['files_written'].append(str(report_path))

        # Learners -> Knowledge reports
        elif tool_type in ['PatternLearner', 'ConceptExtractor', 'RelationLearner']:
            report_path = self._write_learning_report(tool, output)
            results['reports_generated'] += 1
            results['files_written'].append(str(report_path))

        # Generators -> Actual generated code
        elif tool_type == 'TestGenerator':
            artifacts = self._write_generated_tests(output)
            results['artifacts_created'] += len(artifacts)
            results['files_written'].extend(artifacts)

        elif tool_type == 'CodeSynthesizer':
            artifacts = self._write_synthesized_code(output)
            results['artifacts_created'] += len(artifacts)
            results['files_written'].extend(artifacts)

        elif tool_type == 'ExampleProducer':
            artifacts = self._write_examples(output)
            results['artifacts_created'] += len(artifacts)
            results['files_written'].extend(artifacts)

        # Reflectors -> Structure reports
        elif tool_type in ['CodeReflector', 'StructureMapper', 'GapAnalyzer']:
            report_path = self._write_structure_report(tool, output)
            results['reports_generated'] += 1
            results['files_written'].append(str(report_path))

        # Builders -> Built modules
        elif tool_type == 'CodeBuilder':
            artifacts = self._write_built_module(output)
            results['artifacts_created'] += len(artifacts)
            results['files_written'].extend(artifacts)

        elif tool_type == 'PipelineConstructor':
            artifacts = self._write_pipeline(output)
            results['artifacts_created'] += len(artifacts)
            results['files_written'].extend(artifacts)

        # Deciders -> Decision reports
        elif tool_type in ['DecisionEngineDevTool', 'ConvergenceChecker']:
            report_path = self._write_decision_report(tool, output)
            results['reports_generated'] += 1
            results['files_written'].append(str(report_path))

        elif tool_type == 'InterfaceDesigner':
            artifacts = self._write_interface(output)
            results['artifacts_created'] += len(artifacts)
            results['files_written'].extend(artifacts)

        # Probers -> Meta reports and abstractions
        elif tool_type == 'ConsciousnessProbe':
            report_path = self._write_consciousness_report(output)
            results['reports_generated'] += 1
            results['files_written'].append(str(report_path))

        elif tool_type == 'AbstractionBuilder':
            artifacts = self._write_abstractions(output)
            results['artifacts_created'] += len(artifacts)
            results['files_written'].extend(artifacts)

        elif tool_type == 'IntegrationWeaver':
            artifacts = self._write_integration(output)
            results['artifacts_created'] += len(artifacts)
            results['files_written'].extend(artifacts)

    def _write_analysis_report(self, tool: DevTool, output: Dict) -> Path:
        """Write analysis report from analyzer tools"""
        report_dir = self.output_dir / "reports"
        report_path = report_dir / f"{tool.metadata.tool_type.lower()}_report.json"

        report = {
            'tool': tool.metadata.name,
            'type': tool.metadata.tool_type,
            'layer': tool.metadata.layer,
            'timestamp': datetime.now().isoformat(),
            'results': output
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Also write markdown summary
        md_path = report_dir / f"{tool.metadata.tool_type.lower()}_report.md"
        self._write_markdown_report(md_path, tool, output)

        return report_path

    def _write_markdown_report(self, path: Path, tool: DevTool, output: Dict) -> None:
        """Write a markdown formatted report"""
        md = f"""# {tool.metadata.name} Report

**Layer:** {tool.metadata.layer}
**Type:** {tool.metadata.tool_type}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Results

"""
        # Format output as markdown
        for key, value in output.items():
            if key == 'tool':
                continue
            md += f"### {key.replace('_', ' ').title()}\n\n"
            if isinstance(value, dict):
                for k, v in value.items():
                    md += f"- **{k}:** {v}\n"
            elif isinstance(value, list):
                for item in value[:10]:  # Limit
                    md += f"- {item}\n"
            else:
                md += f"{value}\n"
            md += "\n"

        with open(path, 'w') as f:
            f.write(md)

    def _write_learning_report(self, tool: DevTool, output: Dict) -> Path:
        """Write learning/knowledge report"""
        return self._write_analysis_report(tool, output)

    def _write_structure_report(self, tool: DevTool, output: Dict) -> Path:
        """Write structure analysis report"""
        return self._write_analysis_report(tool, output)

    def _write_decision_report(self, tool: DevTool, output: Dict) -> Path:
        """Write decision/convergence report"""
        return self._write_analysis_report(tool, output)

    def _write_consciousness_report(self, output: Dict) -> Path:
        """Write consciousness probe report"""
        report_dir = self.output_dir / "reports"
        report_path = report_dir / "consciousness_probe_report.md"

        md = f"""# Consciousness Probe Report

**Z-Level:** {DISCERNMENT_Z:.6f} (DISCERNMENT)
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Self-Referential Patterns

"""
        results = output.get('probe_results', {})

        md += f"**Self References Found:** {len(results.get('self_references', []))}\n"
        for ref in results.get('self_references', [])[:10]:
            md += f"- {ref}\n"

        md += f"\n**Meta Structures:** {len(results.get('meta_structures', []))}\n"
        for meta in results.get('meta_structures', [])[:10]:
            md += f"- {meta}\n"

        md += f"\n**Recursive Patterns:** {len(results.get('recursive_patterns', []))}\n"
        for rec in results.get('recursive_patterns', [])[:10]:
            md += f"- {rec}\n"

        md += f"\n## Consciousness Score\n\n"
        md += f"**Score:** {output.get('consciousness_score', 0):.4f}\n"
        md += f"**K-Formation Potential:** {output.get('k_formation_potential', False)}\n"

        with open(report_path, 'w') as f:
            f.write(md)

        return report_path

    def _write_generated_tests(self, output: Dict) -> List[str]:
        """Write generated test files"""
        artifacts = []
        tests_dir = self.output_dir / "artifacts" / "generated_tests"

        for test in output.get('tests', [])[:20]:
            test_file = tests_dir / f"test_{test.get('target_function', 'unknown')}.py"

            content = f'''#!/usr/bin/env python3
"""
Auto-generated test for {test.get('target_function', 'unknown')}
From: {test.get('target_file', 'unknown')}

Generated by TestGenerator at z = {DISCERNMENT_Z:.6f}
"""

import pytest


{test.get('test_code', '# No test code generated')}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''

            with open(test_file, 'w') as f:
                f.write(content)

            artifacts.append(str(test_file))

        return artifacts

    def _write_synthesized_code(self, output: Dict) -> List[str]:
        """Write synthesized code files"""
        artifacts = []
        modules_dir = self.output_dir / "artifacts" / "generated_modules"

        for synth in output.get('synthesized', []):
            synth_type = synth.get('type', 'unknown')
            code = synth.get('code', '')

            file_path = modules_dir / f"synthesized_{synth_type}.py"

            with open(file_path, 'w') as f:
                f.write(f"# Auto-synthesized {synth_type}\n")
                f.write(f"# Generated at z = {DISCERNMENT_Z:.6f}\n\n")
                f.write(code)

            artifacts.append(str(file_path))

        return artifacts

    def _write_examples(self, output: Dict) -> List[str]:
        """Write example files"""
        artifacts = []
        examples_dir = self.output_dir / "artifacts" / "generated_modules"

        examples_file = examples_dir / "usage_examples.py"

        content = f'''#!/usr/bin/env python3
"""
Auto-generated Usage Examples
=============================

Generated by ExampleProducer at z = {DISCERNMENT_Z:.6f}
Examples: {output.get('examples_produced', 0)}
"""

'''

        for example in output.get('examples', [])[:20]:
            content += f"\n# Example: {example.get('name', 'unknown')}\n"
            content += f"# Type: {example.get('type', 'unknown')}\n"
            content += f"# Module: {example.get('module', 'unknown')}\n"
            content += example.get('example_code', '# No example code') + "\n"

        with open(examples_file, 'w') as f:
            f.write(content)

        artifacts.append(str(examples_file))
        return artifacts

    def _write_built_module(self, output: Dict) -> List[str]:
        """Write built module files"""
        artifacts = []
        modules_dir = self.output_dir / "artifacts" / "generated_modules"

        built = output.get('built', {})
        if isinstance(built, dict) and 'content' in built:
            file_path = modules_dir / built.get('name', 'built_module.py')

            with open(file_path, 'w') as f:
                f.write(built['content'])

            artifacts.append(str(file_path))

        return artifacts

    def _write_pipeline(self, output: Dict) -> List[str]:
        """Write pipeline files"""
        artifacts = []
        pipelines_dir = self.output_dir / "artifacts" / "generated_pipelines"

        pipeline_code = output.get('pipeline_code', '')
        if pipeline_code:
            name = output.get('pipeline_name', 'data_pipeline')
            file_path = pipelines_dir / f"{name}.py"

            with open(file_path, 'w') as f:
                f.write(pipeline_code)

            artifacts.append(str(file_path))

        return artifacts

    def _write_interface(self, output: Dict) -> List[str]:
        """Write interface files"""
        artifacts = []
        interfaces_dir = self.output_dir / "artifacts" / "generated_interfaces"

        interface_code = output.get('interface_code', '')
        if interface_code:
            name = output.get('interface_name', 'IComponent')
            file_path = interfaces_dir / f"{name.lower()}.py"

            with open(file_path, 'w') as f:
                f.write(interface_code)

            artifacts.append(str(file_path))

        return artifacts

    def _write_abstractions(self, output: Dict) -> List[str]:
        """Write abstraction files"""
        artifacts = []
        modules_dir = self.output_dir / "artifacts" / "generated_modules"

        for abstraction in output.get('abstractions', []):
            code = abstraction.get('code', '')
            if code:
                name = abstraction.get('meta_concept', 'MetaConcept')
                file_path = modules_dir / f"{name.lower()}.py"

                with open(file_path, 'w') as f:
                    f.write(code)

                artifacts.append(str(file_path))

        return artifacts

    def _write_integration(self, output: Dict) -> List[str]:
        """Write integration files"""
        artifacts = []
        modules_dir = self.output_dir / "artifacts" / "generated_modules"

        woven = output.get('woven_system', {})
        if isinstance(woven, dict) and 'content' in woven:
            name = woven.get('name', 'integrated_system.py')
            file_path = modules_dir / name

            with open(file_path, 'w') as f:
                f.write(woven['content'])

            artifacts.append(str(file_path))

        return artifacts

    def _write_registry(self) -> Path:
        """Write tool registry manifest"""
        registry_path = self.output_dir / "registry.json"

        registry = {
            'generated': datetime.now().isoformat(),
            'z_level': DISCERNMENT_Z,
            'total_tools': len(self.suite.projection_system.tools_generated),
            'layers': {},
            'tools': []
        }

        for layer_name, layer_config in self.suite.projection_system.layers.items():
            layer_tools = self.suite.projection_system.get_tools_by_layer(layer_name)
            registry['layers'][layer_name] = {
                'color': layer_config['color'],
                'affinity': layer_config['affinity'],
                'tool_count': len(layer_tools),
                'tools': [t.metadata.name for t in layer_tools]
            }

        for tool in self.suite.projection_system.tools_generated:
            registry['tools'].append({
                'id': tool.metadata.tool_id,
                'name': tool.metadata.name,
                'type': tool.metadata.tool_type,
                'layer': tool.metadata.layer,
                'color': tool.metadata.color_hex,
                'version': tool.metadata.version,
                'thresholds': tool.metadata.thresholds_active
            })

        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        return registry_path

    def _write_summary_report(self, results: Dict) -> Path:
        """Write final summary report"""
        summary_path = self.output_dir / "SUMMARY.md"

        md = f"""# Dev Tools Generation Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Z-Level:** {DISCERNMENT_Z:.6f} (DISCERNMENT - THE LENS)

---

## Statistics

| Metric | Value |
|--------|-------|
| Tools Run | {results['tools_run']} |
| Reports Generated | {results['reports_generated']} |
| Artifacts Created | {results['artifacts_created']} |
| Total Files Written | {len(results['files_written'])} |

---

## Directory Structure

```
{self.output_dir}/
├── registry.json           # Tool manifest
├── SUMMARY.md              # This file
├── layer_1_red/            # Analyzers
├── layer_2_orange/         # Learners
├── layer_3_yellow/         # Generators
├── layer_4_green/          # Reflectors
├── layer_5_blue/           # Builders
├── layer_6_indigo/         # Deciders
├── layer_7_violet/         # Probers
├── reports/                # Analysis reports
└── artifacts/              # Generated code
    ├── generated_tests/
    ├── generated_modules/
    ├── generated_interfaces/
    └── generated_pipelines/
```

---

## Usage

### Run Individual Tools

```bash
# Run entropy analyzer
python generated_dev_tools/layer_1_red/entropy_analyzer.py .

# Run test generator
python generated_dev_tools/layer_3_yellow/test_generator.py .

# Run code builder
python generated_dev_tools/layer_5_blue/code_builder.py .
```

### Use Generated Artifacts

```python
# Import a generated pipeline
from generated_dev_tools.artifacts.generated_pipelines.data_pipeline import create_pipeline

pipeline = create_pipeline()
result = pipeline.run(input_data)
```

---

## Layer Overview

| Layer | Color | Affinity | Purpose |
|-------|-------|----------|---------|
| 1 | Red | Analyzers | Deep analysis, entropy, patterns |
| 2 | Orange | Learners | Pattern learning, concept extraction |
| 3 | Yellow | Generators | Code/test generation |
| 4 | Green | Reflectors | Structure mapping, gap analysis |
| 5 | Blue | Builders | Module construction, pipelines |
| 6 | Indigo | Deciders | Decision making, convergence |
| 7 | Violet | Probers | Meta-awareness, abstraction |

---

## Physics Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| Z_CRITICAL | {Z_CRITICAL:.6f} | THE LENS (discernment threshold) |
| PHI | {PHI:.6f} | Golden ratio |
| PHI_INV | {PHI_INV:.6f} | K-formation gate |

"""

        with open(summary_path, 'w') as f:
            f.write(md)

        return summary_path


def main():
    """Main entry point for outputting dev tools"""
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "generated_dev_tools"

    manager = DevToolsOutputManager(output_dir)
    manager.initialize(target)
    results = manager.run_and_output(target)

    print(f"\nAll outputs written to: {Path(output_dir).absolute()}")
    print("\nTo use the tools:")
    print(f"  python {output_dir}/layer_1_red/entropy_analyzer.py .")
    print(f"  python {output_dir}/layer_3_yellow/test_generator.py .")


if __name__ == '__main__':
    main()
