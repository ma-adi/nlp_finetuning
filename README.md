# NLP Finetuning Experiments Report

## 1. Goal

Train an NLP-based model to convert XML inputs into a specific JSON schema, with an optional comparison to LLM‑based finetuning.

---

## 2. Experiments Conducted

### ❌ ML Indentation/Spacing Robustness
- **Setup**: Single format dataset with imperfect spacing (500 examples).
- **Result**: `39%` exact-match accuracy under formatting variation.

### ❌ ML Format Conversion Trials
- **Setup**: Three experiments using two training formats (1,000 examples each).
- **Test**: Unseen composite format.
- **Result**: `0.0%` exact-match accuracy on the held‑out format in all runs.

### ✅ SLM Evaluation (Preliminary)
- Full-scale test-dataset evaluation pending.
- On a small test set:
  - Shows **consistent generalization** for:
    - Value and length variations
    - Nested formatting structures

---

## 3. Results Summary

| Task                        | Training Examples         | ML Outcome | SLM Outcome | ML Notes                                     | SLM Notes                                          |
|----------------------------|---------------------------|------------|-------------|----------------------------------------------|----------------------------------------------------|
| Value & Length (short)     | 500–750                   | ❌ Fail     | ✅ Pass      | ❌ Fails on long lists when only short-trained | ✅ Generalizes to larger entries with different values |
| Value & Length (long)      | 1,000–2,000               | ✅ Pass     | ✅ Pass      | ✅ Trained on longer inputs                   | ✅ Generalizes to larger entries with different values |
| Format Conversion (nested) | ML: 15,000+ / SLM: 500–1,000 | ❌ Fail     | ✅ Pass      | ❌ Requires ≥ 15,000 varied examples          | ✅ Needs 500–1,000 examples per format for refinement |

> **Legend**: “SLM” refers to small‑LLM finetuning on **LLaMA 3.2 3B**.
> Link to experiments: https://docs.google.com/spreadsheets/d/1Vg8xPi4Ag4sMt__vSpLLViC2TWu9cTrJPGHCB1mrzDc/edit?usp=sharing
---

## 4. Experiments Overview

### 4.1 Value & Length Variation

**Objective**: Can the model generalize from short XML lists to longer ones?

**Training**:
```xml
<root><item>apple</item><item>banana</item></root>
```
➡️  
```json
{"root":{"item":["apple","banana"]}}
```

**Testing**:
```xml
<root><item>kiwi</item><item>mango</item><item>papaya</item></root>
```
➡️  
```json
{"root":{"item":["kiwi","mango","papaya"]}}
```

---

### 4.2 Format Conversion

**Objective**: Train on two simple formats and test on a nested variant (held-out).

**Format 1 (Flat List)**:
```xml
<root><value>foo</value>...</root>
```
➡️  
```json
{"root":{"value": [...]} }
```

**Format 2 (List of Objects)**:
```xml
<root><item><name>bar</name></item>...</root>
```
➡️  
```json
{"root":{"item":[{"name":"bar"},...]} }
```

**Held-out Test Format (Nested List)**:
```xml
<root><item><sub>a</sub><sub>b</sub></item></root>
```
➡️  
```json
{"root":{"item":[{"sub":["a","b"]}]}}
```

---

## 5. Performance & Cost Comparison

| Metric          | CodeT5+ 220M                        | LLaMA 3.2 3B (GGUF)                  |
|------------------|--------------------------------------|--------------------------------------|
| Parameters       | 220 million                         | 3 billion                            |
| Inference RAM    | ~1.2 GB (FF32, CPU-only)            | ~8 GB (CPU-only)                     |
| CPU Instance     | `e2-small` (1vCPU / 2GB RAM): $0.0337/hr | `e2-standard-2` (2vCPU / 8GB RAM): $0.067/hr |
| Recommended RAM  | ≥ 2 GB                              | ≥ 8 GB                               |

> Based on Google Cloud pricing ([cloudprice.net](https://cloudprice.net)) as of July 18, 2025.

---

## ✅ Additional Notes

- **SLM is converted to a CPU-friendly format** that uses relatively fewer resources (e.g., GGUF format).
- **Further quantization is possible**, and performance should be tested. This could make the hardware delta between CodeT5+ and SLM marginal.
