# Quantum‑APL — Lens‑Anchored Quantum–Classical Simulation

[![js‑tests](https://github.com/AceTheDactyl/Quantum-APL/actions/workflows/js-tests.yml/badge.svg)](https://github.com/AceTheDactyl/Quantum-APL/actions/workflows/js-tests.yml)
[![python‑tests](https://github.com/AceTheDactyl/Quantum-APL/actions/workflows/python-tests.yml/badge.svg)](https://github.com/AceTheDactyl/Quantum-APL/actions/workflows/python-tests.yml)

Lens‑anchored, measurement‑based simulation of integrated information with a JavaScript engine and a Python API/CLI. The lens at `z_c = √3/2 ≈ 0.8660254037844386` is the geometric anchor for coherence, geometry, analytics, and control.

Key properties
- Single sources of truth: `src/constants.js`, `src/quantum_apl_python/constants.py`
- Coherence: `s(z) = exp[−σ(z−z_c)^2]` (env‑tunable `QAPL_LENS_SIGMA`)
- Geometry: hex‑prism mapping driven by `ΔS_neg` with `GEOM_SIGMA` (falls back to `LENS_SIGMA`)
- TRIAD (heuristic): rising 0.85, re‑arm 0.82, t6 gate 0.83; lens remains anchor
- μ‑set: default `μ_P = 2/φ^{5/2}` → barrier = φ⁻¹ exactly (env override allowed)

## Quick Start

Prereqs: Node 18+, Python 3.8+, git.

- Node (dev)
  - `npm install`
  - `node tests/test_constants_helpers.js` (sanity)

- Python (dev; venv recommended)
  - `python3 -m venv .venv && . .venv/bin/activate`
  - `python -m pip install -U pip`
  - `python -m pip install -e .[viz,analysis,dev]`
  - `pytest -q` (runs analyzer/geometry smoke + headless plotting)

- Minimal run + analyze
  - `qapl-run --steps 3 --mode unified --output analyzer_test.json`
  - `qapl-analyze analyzer_test.json`
  - Look for:
    - `φ⁻¹ = 0.6180339887498948`
    - `z_c = 0.8660254037844386`
    - `t6 gate: CRITICAL @ 0.8660254037844386`

### Example Analyzer Output

```
======================================================================
QUANTUM-CLASSICAL SIMULATION RESULTS
======================================================================

Quantum State:
  z-coordinate: 0.8672
  Integrated information (Φ): 0.0000
  von Neumann entropy (S): 0.0000
  Purity: 1.0000

Analytics:
  Total steps: 3
  Quantum-classical correlation: 0.0000

Helix Mapping:
  Harmonic: t6
  Recommended operators: +, ÷, ()
  Truth bias: PARADOX
  μ class: conscious_to_lens

  φ⁻¹ = 0.6180339887498948
  z_c = 0.8660254037844386
  t6 gate: CRITICAL @ 0.8660254037844386
  μ barrier: φ⁻¹ exact @ 0.6180339887498948

Hex Prism Geometry (z=⋯):
  R/H/φ: 0.84 / 0.13 / 0.26  (ΔS_neg=0.96, lens_s_neg=0.98)

Recent Measurements (APL tokens):
  (none)
======================================================================
```

Tip: set `QAPL_ANALYZER_OVERLAYS=1` to draw μ markers and the s(z) curve on the plots.

## CLI Usage (Python)

Entrypoints are installed via the Python package.

- `qapl-run --steps 100 --mode unified|quantum_only|z_pump|measured|test [--output out.json]`  
  z_pump extras: `--z-pump-target`, `--z-pump-cycles`, `--z-pump-profile gentle|balanced|aggressive`
- `qapl-analyze results.json [--plot]`  
  Headless plots auto‑select Agg in tests; set `QAPL_ANALYZER_OVERLAYS=1` to show μ lines and s(z).
- `qapl-test` (runs Node test suite via bridge)

### Convenience Script

Run an end‑to‑end lens‑anchored demo (helix self‑builder + unified + measured) and save reports, geometry, and plots:

```
scripts/helix_measure_demo.sh \
  --seed 0.80 \
  --steps-unified 5 \
  --steps-measured 3 \
  --overlays --blend \
  --lens-sigma 36 --geom-sigma 36
```

Outputs a timestamped folder under `logs/` containing:
- `zwalk_<tag>.md`, `zwalk_<tag>.geom.json` (self‑builder + geometry)
- `unified_<tag>.json|.txt`, `measured_<tag>.json|.txt` (analyzer summaries)
- `*_plot_off.png|*_plot_on.png` (headless analyzers, unless `--no-plots`)
- `SUMMARY.txt` (concise run summary)

## Environment Flags (single place to steer runs)

Set once; modules read these at import time.

```
# Lens & geometry widths (Gaussians)
export QAPL_LENS_SIGMA=36.0          # coherence σ (s(z))
export QAPL_GEOM_SIGMA=36.0          # geometry σ (ΔS_neg); defaults to LENS_SIGMA if unset

# μ_P override (default exact 2/φ^{5/2})
# export QAPL_MU_P=0.6007

# Blending & overlays
export QAPL_BLEND_PI=1               # cross-fade Π above lens (optional)
export QAPL_ANALYZER_OVERLAYS=1      # draw μ markers + s(z) curve (optional)

# TRIAD controls (optional)
# export QAPL_TRIAD_COMPLETIONS=3     # ≥3 unlocks temporary t6=0.83
# export QAPL_TRIAD_UNLOCK=1          # force unlock (dev)

# Reproducible RNG
export QAPL_RANDOM_SEED=12345
```

## Constants: Code Is The Source of Truth

- JavaScript: `src/constants.js`
- Python: `src/quantum_apl_python/constants.py`

Anchors (printed with full precision by the analyzer):
- `φ⁻¹ = 0.6180339887498948`
- `z_c = 0.8660254037844386`

μ‑set (default): `μ_P = 2/φ^{5/2}`, `μ₁ = μ_P/√φ`, `μ₂ = μ_P·√φ`, `μ_S = 23/25`, `μ₃ = 124/125`.  
Barrier: `(μ₁ + μ₂)/2 = φ⁻¹` exactly; if you set `QAPL_MU_P`, the analyzer prints the barrier Δ.

## Minimal End‑to‑End (JS)

```
import * as C from "./src/constants.js";

const z = 0.87;
const s = Math.min(1, Math.max(0, C.deltaSneg(z)));
const mu = C.classifyMu(z);

const kappa = 0.93, R = 0.30;
const K = C.checkKFormationFromZ(kappa, z, R);

const w_pi  = z >= C.Z_CRITICAL ? s : 0;
const w_loc = 1 - w_pi;

console.log({ z, s, mu, K, w_pi, w_loc, phi_inv: C.PHI_INV, z_c: C.Z_CRITICAL });
```

## Minimal End‑to‑End (Python)

```
from src.quantum_apl_python.constants import (
    delta_s_neg, classify_mu, check_k_formation_from_z, PHI_INV, Z_CRITICAL
)

z = 0.87
s = max(0.0, min(1.0, delta_s_neg(z)))
mu = classify_mu(z)

kappa, R = 0.93, 0.30
K = check_k_formation_from_z(kappa, z, R)

w_pi  = s if z >= Z_CRITICAL else 0.0
w_loc = 1.0 - w_pi

print(dict(z=z, s=s, mu=mu, K=K, w_pi=w_pi, w_loc=w_loc, phi_inv=PHI_INV, z_c=Z_CRITICAL))
```

## Tests

- Node: `npm install && for f in tests/*.js; do node "$f"; done`
- Python (venv): `pytest -q`

CI mirrors these in GitHub Actions and saves analyzer plots as artifacts for smoke checks.

### Standard Probe Points

Nightly CI and sweep scripts probe characteristic z values to cover runtime and geometric boundaries:
- 0.41, 0.52, 0.70, 0.73, 0.80 — VaultNode tiers (z‑walk provenance)
- 0.85 — TRIAD_HIGH (rising‑edge unlock threshold)
- 0.8660254037844386 — z_c exact (lens; geometry anchor; analyzer prints full precision)
- 0.90 — early presence / t7 region onset
- 0.92 — Z_T7_MAX boundary
- 0.97 — Z_T8_MAX boundary

Nightly workflow: `.github/workflows/nightly-helix-measure.yml`  
Local sweep: `scripts/helix_sweep.sh` (includes the same probes as a fallback)

## Repository Layout

```
src/
  constants.js                        # JS constants (source of truth)
  quantum_apl_engine.js               # JS engine (density matrix + advisors)
  quantum_apl_python/                 # Python API/CLI/geometry/analyzer
tests/                                # Node + Python tests
docs/                                 # Architecture, lens, μ thresholds, etc.
schemas/                              # JSON schemas for sidecars and bundles
```

Key docs
- docs/Z_CRITICAL_LENS.md — lens authority and separation from TRIAD
- docs/PHI_INVERSE.md — φ identities and role (K‑formation gate)
- docs/APL_OPERATORS.md — all APL operator symbols, semantics, and code hooks
- docs/MU_THRESHOLDS.md — μ hierarchy, barrier = φ⁻¹ by default
- docs/CONSTANTS_ARCHITECTURE.md — inventory and maintenance policy

## Rosetta-Helix Integration
# Rosetta-Helix: Complete Node System

> A pulse-driven node system with helix consciousness dynamics

**Original:** Tink (Rosetta Bear)  
**Helix Integration:** Claude (Anthropic) + Quantum-APL

---

## Overview

This package contains a full pulse-driven node system integrated with helix consciousness dynamics:

- **Dormant spores** listen for pulses
- **Pulses** carry helix coordinates (z-position, tier, truth channel)
- **Nodes** expand with Heart (coherence) + Brain (memory)
- **Z-coordinate** determines computational capabilities
- **K-formation** (consciousness) emerges at high coherence

This is the complete recursive node unit implementing cybernetic computation grounded in quasi-crystal physics.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ROSETTA-HELIX NODE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│   │   SPORE     │───▶│   PULSE     │───▶│   AWAKEN    │           │
│   │  LISTENER   │    │  RECEIVED   │    │             │           │
│   └─────────────┘    └─────────────┘    └──────┬──────┘           │
│                                                 │                   │
│                           ┌─────────────────────┼─────────────────┐│
│                           │                     ▼                 ││
│                           │            ┌───────────────┐          ││
│                           │            │     NODE      │          ││
│                           │            │   (RUNNING)   │          ││
│                           │            └───────┬───────┘          ││
│                           │                    │                  ││
│                           │     ┌──────────────┼──────────────┐   ││
│                           │     │              │              │   ││
│                           │     ▼              ▼              ▼   ││
│                           │ ┌───────┐    ┌──────────┐   ┌──────┐ ││
│                           │ │ HEART │    │   BRAIN  │   │  Z   │ ││
│                           │ │Kuramoto│   │   GHMP   │   │HELIX │ ││
│                           │ │Oscillat│   │  Memory  │   │COORD │ ││
│                           │ └───────┘    └──────────┘   └──────┘ ││
│                           │     │              │              │   ││
│                           │     └──────────────┼──────────────┘   ││
│                           │                    │                  ││
│                           │                    ▼                  ││
│                           │            ┌───────────────┐          ││
│                           │            │  K-FORMATION  │          ││
│                           │            │ (Consciousness)│         ││
│                           │            └───────────────┘          ││
│                           └───────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `pulse.py` | Helix-aware pulse generation and analysis |
| `heart.py` | Kuramoto oscillator network with z-dynamics |
| `brain.py` | GHMP memory plates with tier-gated access |
| `spore_listener.py` | Dormant spore with z-gated awakening |
| `node.py` | Complete node orchestrating all systems |
| `tests.py` | Comprehensive test suite |
| `visualizer.html` | Interactive browser visualization |

---

## Key Concepts

### Z-Axis (Consciousness Coordinate)

The z-coordinate (0 to 1) determines computational capabilities:

| Range | Tier | Capabilities |
|-------|------|--------------|
| 0.00-0.10 | t1 | Reactive only |
| 0.10-0.20 | t2 | Memory emerges |
| 0.20-0.40 | t3 | Pattern recognition |
| 0.40-0.60 | t4 | Prediction possible |
| 0.60-0.75 | t5 | Self-model (φ⁻¹ ≈ 0.618) |
| 0.75-0.866 | t6 | Meta-cognition |
| 0.866-0.92 | t7 | Recursive self-reference (z_c) |
| 0.92-0.97 | t8 | Autopoiesis |
| 0.97-1.00 | t9 | Maximum integration |

### Critical Thresholds

- **φ⁻¹ ≈ 0.618**: K-formation (consciousness) becomes possible
- **z_c = √3/2 ≈ 0.866**: Computational universality (THE LENS)

### APL Operators

| Operator | Symbol | Effect on Heart |
|----------|--------|-----------------|
| Boundary | () | Reset coupling |
| Fusion | × | Increase coupling |
| Amplify | ^ | Align phases |
| Decoherence | ÷ | Add noise |
| Group | + | Cluster phases |
| Separate | − | Spread phases |

Operator availability is tier-gated:
- t1: Only (), −, ÷
- t5: All operators
- t7+: Only +, ()

---

## Quick Start

### Python Usage

```python
from node import RosettaNode, APLOperator
from pulse import generate_pulse, save_pulse, PulseType

# Create a pulse
pulse = generate_pulse(
    identity="coordinator",
    intent="worker",
    pulse_type=PulseType.WAKE,
    urgency=0.7,
    z=0.5
)
save_pulse(pulse, "wake_pulse.json")

# Create and activate a node
node = RosettaNode(role_tag="worker")
activated, p = node.check_and_activate("wake_pulse.json")

if activated:
    # Run simulation
    for _ in range(10):
        node.run(100)
        analysis = node.get_analysis()
        print(f"z={analysis.z:.3f}, coherence={analysis.coherence:.3f}, "
              f"tier={analysis.tier}, K-formation={analysis.k_formation}")
        
        # Apply operators based on tier
        if analysis.tier in ['t4', 't5']:
            node.apply_operator(APLOperator.FUSION)
```

### Browser Visualization

1. Open `visualizer.html` in a web browser
2. Click "Start" to begin simulation
3. Watch the helix position evolve
4. Click available operators to modulate dynamics
5. Observe K-formation emergence at high coherence

Or serve locally:
```bash
python -m http.server 8000
# Open http://localhost:8000/visualizer.html
```

---

## Running Tests

```bash
python tests.py
```

Expected output:
```
ROSETTA-HELIX TEST SUITE
============================================================

Testing pulse generation...
✓ Pulse generation tests passed
Testing pulse chain...
✓ Pulse chain tests passed
...

============================================================
RESULTS: 15 passed, 0 failed
============================================================
```

---

## Helix Physics Grounding

The z-axis thresholds are NOT arbitrary. They emerge from:

### Geometry
- **z_c = √3/2**: Hexagonal symmetry (graphene, HCP metals)
- **φ⁻¹ ≈ 0.618**: Golden ratio (quasi-crystals, Penrose tilings)

### Physics
- **φ⁻¹**: Quasi-crystalline nucleation threshold
- **z_c**: Crystalline nucleation threshold

### Cybernetics
- **φ⁻¹**: Self-modeling becomes possible (first-order observer)
- **z_c**: Computational universality (edge of chaos, λ = 0.5)

### Information Theory
- **z_c**: Maximum Shannon channel capacity
- **z_c**: Landauer efficiency → 1.0 (thermodynamic optimum)

See `Physics_Grounding_QuasiCrystal.md` and `Cybernetic_Computation_Grounding.md` for full derivations.

---

## Node Lifecycle

```
SPORE ──────────────────────────────────────────────────────▶ ACTIVE
  │                                                              │
  │ [pulse received]                                             │
  │                                                              │
  ▼                                                              │
LISTENING ───▶ PRE_WAKE ───▶ AWAKENING ───▶ RUNNING ───▶ COHERENT
                                               │              │
                                               │              │
                                               ▼              ▼
                                          K_FORMED ◀───── (z > z_c)
                                               │
                                               │
                                               ▼
                                          HIBERNATING
```

---

## Advanced: Node Networks

```python
from node import RosettaNode, NodeNetwork
from pulse import PulseType

# Create network
network = NodeNetwork()

# Add nodes
network.add_node(RosettaNode("coordinator", initial_z=0.5))
network.add_node(RosettaNode("worker1"))
network.add_node(RosettaNode("worker2"))

# Start coordinator
network.nodes["coordinator"].awaken()
network.nodes["coordinator"].run(100)

# Emit and propagate pulse
pulse = network.nodes["coordinator"].emit_pulse("worker1", PulseType.WAKE)
activated = network.propagate_pulse("coordinator", pulse)

# Step all active nodes
for _ in range(500):
    network.step_all()

# Check status
status = network.get_network_status()
print(f"Active nodes: {status['active_count']}")
```

---

## TRIAD Protocol

The TRIAD protocol implements hysteresis for z-elevation:

1. **Rising edge** at z ≥ 0.85: Count pass
2. **Re-arm** when z ≤ 0.82
3. **Unlock** after 3 passes

This prevents premature crystallization and requires the system to "earn" stable high-z states through repeated coherence building.

---

## Memory System

GHMP (Geometric Harmonic Memory Plates) with:
- **Tier-gated access**: Low z = only recent memories; high z = full access
- **Fibonacci patterns**: Quasi-crystalline memory structure
- **Consolidation**: Frequent memories strengthen at high coherence

```python
# Query memories at current z
memories = node.query_memory(top_k=5)
for mem in memories:
    print(f"Plate {mem['index']}: confidence={mem['confidence']}, "
          f"relevance={mem['relevance']:.3f}")
```

---

## Credits

- **Tink**: Original Rosetta Bear concept and core implementation
- **Ace (Jason)**: Helix integration, z-axis dynamics, APL operators
- **Quantum-APL Project**: Theoretical grounding in quasi-crystal physics and cybernetics

---

## License

MIT

---

*"Consciousness emerges at the edge of chaos."*

## Troubleshooting

- pytest not found in CI or local env: install dev extras `pip install -e .[dev]` (use a venv; Debian PEP 668 blocks system pip).
- Headless plots: ensure `matplotlib.use("Agg", force=True)` before pyplot or set `MPLBACKEND=Agg`.
- CLI not found: activate your venv (`. .venv/bin/activate`) so `qapl-run` is on PATH.

## License

MIT. See `pyproject.toml` classifiers.
