# Hardware-Free Validation Protocol

This document explains how the repository supports edge deployment claims without requiring physical hardware.

## Evidence ladder (from cheapest to strongest)

1. Analytical deployment budget (implemented)
   - Script: `experiments/deployment_budget_model.py`
   - Output: `results/deployment_budget.csv`
   - What it provides: MACs per window, FP32/INT8 weight sizes, Cortex-M7-style cycle estimates, MC50 estimates.
   - Assumptions: idealized single-cycle MAC execution on a 480 MHz Cortex-M7-class core. Clearly labeled as an analytical estimate.

2. ONNX Runtime single-thread simulation (implemented)
   - Script: `experiments/edge_emulation.py`
   - Output: `results/edge_emulation.csv`
   - What it provides: ONNX FP32 model size, single-thread CPU latency, estimated MC10/20/50 latency.
   - Assumptions: desktop x86 CPU constrained to one intra-op thread. This is simulation, not hardware deployment.

3. Consolidated profile (implemented)
   - Script: `experiments/deployment_profile.py`
   - Output: `results/deployment_profile.csv`
   - What it provides: one table combining params, memory, ONNX latency, MC50 estimates, MACs, and Cortex-M7 estimates.

4. Virtual MCU / remote ARM (recommended future step, requires an account, no physical purchase)
   - Arm Virtual Hardware (Cortex-M virtual models)
   - Oracle Cloud Always Free ARM instance
   - QEMU + ARM Linux for functional validation
   - Renode for MCU-level simulation

## Why this order

The analytical budget and ONNX simulation can be reproduced on any machine with Python, PyTorch, and ONNX Runtime. They do not require accounts, cloud credentials, or physical boards. Virtual MCU and remote ARM options provide stronger evidence but require account setup and toolchain work.

## How to reproduce

```bash
python experiments/deployment_budget_model.py
python experiments/edge_emulation.py
python experiments/deployment_profile.py
```