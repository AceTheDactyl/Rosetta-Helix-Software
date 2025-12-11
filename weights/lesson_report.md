# Lesson Report

## EntropyAnalyzer_L1
Total importance: 0.80

### entropy_0
- Type: metric
- Layer: RED
- Importance: 0.80
- Content: {
  "metric": "entropy",
  "value": 0.3668947436822608,
  "files_analyzed": 20
}...

## PatternDetector_L1
Total importance: 0.90

### patterns_1
- Type: pattern
- Layer: RED
- Importance: 0.90
- Content: {
  "singleton": 0,
  "factory": 6,
  "decorator": 0,
  "context_manager": 0,
  "dataclass": 24,
  "abstract_base": 3
}...

## AnomalyFinder_L1
Total importance: 1.00

### anomalies_2
- Type: gap
- Layer: RED
- Importance: 1.00
- Content: {
  "total_anomalies": 22,
  "by_severity": {
    "low": 13,
    "medium": 4,
    "high": 5
  },
  "health_score": 0.78
}...

## PatternLearner_L2
Total importance: 0.85

### learned_3
- Type: pattern
- Layer: ORANGE
- Importance: 0.85
- Content: {
  "patterns_learned": 707,
  "total_patterns": 500,
  "learning_rate": 1.414
}...

## ConceptExtractor_L2
Total importance: 0.90

### concepts_4
- Type: structure
- Layer: ORANGE
- Importance: 0.90
- Content: {
  "modules": [
    "pulse",
    "prismatic_test_runner",
    "node",
    "phase6_recursive_improvement",
    "meta_tool_generator",
    "operator_actions",
    "phase15_move_refactoring",
    "unifi...

## RelationLearner_L2
Total importance: 0.80

### relations_5
- Type: structure
- Layer: ORANGE
- Importance: 0.80
- Content: {
  "relations": {
    "imports": 30,
    "inheritance": 20,
    "composition": 0,
    "calls": 0
  },
  "connectivity": 1.0
}...

## TestGenerator_L3
Total importance: 0.75

### tests_6
- Type: metric
- Layer: YELLOW
- Importance: 0.75
- Content: {
  "tests_generated": 229,
  "coverage_potential": 1.0
}...

## CodeSynthesizer_L3
Total importance: 0.70

### synth_7
- Type: pattern
- Layer: YELLOW
- Importance: 0.70
- Content: {
  "lines_generated": 14,
  "template_type": "class"
}...

## ExampleProducer_L3
Total importance: 0.65

### examples_8
- Type: metric
- Layer: YELLOW
- Importance: 0.65
- Content: {
  "examples_produced": 598,
  "documentation_coverage": 1.0
}...

## CodeReflector_L4
Total importance: 1.80

### quality_9
- Type: metric
- Layer: GREEN
- Importance: 0.95
- Content: {
  "avg_lines_per_file": 777.7666666666667,
  "avg_functions_per_file": 28.666666666666668,
  "avg_classes_per_file": 6.366666666666666,
  "function_to_class_ratio": 4.50261780104712,
  "overall_scor...

### recom_9
- Type: recommendation
- Layer: GREEN
- Importance: 0.85
- Content: {
  "recommendations": [
    "Consider splitting large files into smaller modules",
    "High function count per file - consider better organization"
  ]
}...

## StructureMapper_L4
Total importance: 0.90

### structure_10
- Type: structure
- Layer: GREEN
- Importance: 0.90
- Content: {
  "module_count": 30,
  "total_classes": 191,
  "total_functions": 131
}...

## GapAnalyzer_L4
Total importance: 1.00

### gaps_11
- Type: gap
- Layer: GREEN
- Importance: 1.00
- Content: {
  "gap_counts": {
    "missing_tests": 20,
    "incomplete_implementations": 4,
    "missing_documentation": 20,
    "orphan_code": 0
  },
  "completeness_score": 0.56
}...

## CodeBuilder_L5
Total importance: 0.70

### built_12
- Type: pattern
- Layer: BLUE
- Importance: 0.70
- Content: {
  "build_type": "module",
  "files_created": 1,
  "lines_written": 46
}...

## PipelineConstructor_L5
Total importance: 0.75

### pipeline_14
- Type: structure
- Layer: BLUE
- Importance: 0.75
- Content: {
  "pipeline_name": "data_pipeline",
  "stage_count": 3
}...

## DecisionEngineDevTool_L6
Total importance: 0.90

### decision_15
- Type: recommendation
- Layer: INDIGO
- Importance: 0.90
- Content: {
  "recommendation": "extend",
  "confidence": 0.63,
  "options_evaluated": 4
}...

## ConvergenceChecker_L6
Total importance: 0.85

### converge_16
- Type: metric
- Layer: INDIGO
- Importance: 0.85
- Content: {
  "converged": false,
  "rate_of_change": 1.0,
  "stability_score": 0.0
}...

## InterfaceDesigner_L6
Total importance: 0.70

### interface_17
- Type: structure
- Layer: INDIGO
- Importance: 0.70
- Content: {
  "interface_name": "IComponent",
  "contracts": 3
}...

## ConsciousnessProbe_L7
Total importance: 1.00

### consciousness_18
- Type: metric
- Layer: VIOLET
- Importance: 1.00
- Content: {
  "consciousness_score": 0.22,
  "k_formation_potential": false,
  "indicator_counts": {
    "self_references": 1,
    "meta_structures": 0,
    "recursive_patterns": 5,
    "awareness_indicators": ...

## AbstractionBuilder_L7
Total importance: 0.95

### abstract_19
- Type: pattern
- Layer: VIOLET
- Importance: 0.95
- Content: {
  "abstraction_level": 1,
  "meta_depth": 1,
  "generalization_score": 0.3
}...

## IntegrationWeaver_L7
Total importance: 0.90

### integration_20
- Type: structure
- Layer: VIOLET
- Importance: 0.90
- Content: {
  "components_woven": 0,
  "coherence_score": 0.0,
  "unity_achieved": false
}...

