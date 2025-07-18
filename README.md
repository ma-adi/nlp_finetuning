# NLP Finetuning Experiments Report

## 1. Goal

Train an NLP-based model to convert XML inputs into a specific JSON schema, with an optional comparison to LLM‑based finetuning.

## 2. Experiments Overview

### 2.1 Value & Length Variation

- **Objective:** Can the model generalize from short XML lists to longer ones, and correctly capture varying field values?
- **Training Example:**

```xml
<root><item>apple</item><item>banana</item></root>
```

```json
{"root":{"item":["apple","banana"]}}
```

- **Testing Example:**

```xml
<root><item>kiwi</item><item>mango</item><item>papaya</item></root>
```

```json
{"root":{"item":["kiwi","mango","papaya"]}}
```

### 2.2 Format Conversion

- **Objective:** Train on two simple formats, then test on a nested variation (held‑out).
- **Formats Trained:**
  - **Format 1 (Flat List):** `<root><value>foo</value>...</root>` → `{"root":{"value":[...]}}`
  - **Format 2 (List of Objects):** `<root><item><name>bar</name></item>...</root>` → `{"root":{"item":[{"name":"bar"},...]}}`
- **Held‑out Test Format (Nested List):** `<root><item><sub>a</sub><sub>b</sub></item></root>` → `{"root":{"item":[{"sub":["a","b"]}]}}`

## 3. Results

| Task                             | Training Examples                     | ML Outcome | SLM Outcome | ML Notes                                      | SLM Notes                                               |
| -------------------------------- | ------------------------------------- | ---------- | ----------- | --------------------------------------------- | ------------------------------------------------------- |
| Value & Length (short lists)     | 500–750                               | ✅ Pass     | ✅ Pass      | ❌ Fails on long lists when only short-trained | Strong generalization on short inputs                   |
| Value & Length (long-focused)    | 1,000–2,000                           | ✅ Pass     | ✅ Pass      | Trained on longer inputs                      | Equally robust with refined LLM finetuning              |
| Format Conversion (nested lists) | ML: 15,000+ SLM: 500–1,000 per format | ❌ Fail     | ✅ Pass      | Requires ≳15,000 varied examples              | Refined finetuning; needs 500–1,000 examples per format |

> **Legend:** “SLM” refers to small‑LLM finetuning on Llama 3.2 3B.

## 4. Performance & Cost Comparison. Performance & Cost Comparison. Performance & Cost Comparison Performance & Cost Comparison. Performance & Cost Comparison Performance & Cost Comparison

| Metric            | CodeT5+ 220M                                                                                                                                 | Llama 3.2 3B (GGUF)                                                                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Params            | 220 million                                                                                                                                  | 3 billion                                                                                                                                             |
| Inference RAM     | \~1.2 GB (FF32, CPU-only)                                                                                                                    | \~8 GB (CPU-only)                                                                                                                                     |
| CPU‑only Instance | e2‑small (1 vCPU, 2 GB RAM) at \$0.0337 /hr ([cloudprice.net](https://cloudprice.net/gcp/compute/instances/e2-small?utm_source=chatgpt.com)) | e2‑standard‑2 (2 vCPU, 8 GB RAM) at \$0.067 /hr ([cloudprice.net](https://cloudprice.net/gcp/compute/instances/e2-standard-2?utm_source=chatgpt.com)) |
| Recommended RAM   | ≥ 2 GB                                                                                                                                       | ≥ 8 GB                                                                                                                                                |

*Costs based on Google Cloud (us‑central1).*\
CodeT5+ 220M on CPU requires \~1–2 GB RAM and can run on a small 2 GB instance; Llama 3.2 3B GGUF needs at least 8 GB RAM on a modest VM. Cloud (us‑central1).\*\
CodeT5+ can run on a low‑cost micro instance; Llama 3B requires a larger, more expensive VM.

---

*Report generated on July 18, 2025*

