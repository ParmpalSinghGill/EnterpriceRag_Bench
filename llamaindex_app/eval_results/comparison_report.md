# Head-to-Head Benchmark: Native vs. LlamaIndex QueryEngine vs. LlamaIndex ReActAgent

- **Model**: `nvidia/nemotron-3.5-lightning-30b-a3b`
- **Total Evaluated Questions**: 10
- **Categories Evaluated**: 10
- **Top-K Context**: 7
- **Date**: 2026-10-14 20:58:28

## Overall Benchmark Comparison

| Pipeline | Correct / Total | Accuracy (%) | Avg Latency (s) | Relative Delta vs Native |
| :--- | :---: | :---: | :---: | :---: |
| **Native Hybrid Pipeline** | 10/10 | **100.0%** | 71.2s | Baseline |
| **LlamaIndex QueryEngine** | 9/10 | **90.0%** | 34.3s | **-10.0%** |
| **LlamaIndex ReAct Agent** | 8/10 | **80.0%** | 80.9s | **-20.0%** |

## Category-Level Breakdown

| Category | N | Native Acc (%) | LlamaIndex QE (%) | LlamaIndex Agent (%) | Drop / Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **basic** | 1 | 100.0% | 100.0% | 100.0% | +0.0% |
| **completeness** | 1 | 100.0% | 100.0% | 100.0% | +0.0% |
| **conflicting_info** | 1 | 100.0% | 100.0% | 100.0% | +0.0% |
| **constrained** | 1 | 100.0% | 100.0% | 100.0% | +0.0% |
| **high_level** | 1 | 100.0% | 100.0% | 100.0% | +0.0% |
| **info_not_found** | 1 | 100.0% | 100.0% | 0.0% | +0.0% |
| **intra_document_reasoning** | 1 | 100.0% | 100.0% | 100.0% | +0.0% |
| **miscellaneous** | 1 | 100.0% | 100.0% | 100.0% | +0.0% |
| **project_related** | 1 | 100.0% | 0.0% | 0.0% | -100.0% |
| **semantic** | 1 | 100.0% | 100.0% | 100.0% | +0.0% |

## Key Observations

1. **Accuracy Parity**: LlamaIndex QueryEngine achieved **90.0%** compared to Native's **100.0%**.
2. **Agent Reasoning**: LlamaIndex ReAct Agent achieved **80.0%**, executing autonomous multi-step retrieval workflows.
3. **Latency Profile**: QueryEngine averaged **34.3s** per question, while ReAct Agent averaged **80.9s** due to multi-turn tool interaction.
