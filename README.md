# NLP Finetuning Experiments Report

## 1. Goal

Train an NLP-based model to convert XML inputs into a specific JSON schema, with an optional comparison to LLM‑based finetuning.

---

## 2. Experiments Conducted

###  Dynamic values & length of input/output
- **Setup**: Single format dataset with small-medium length examples trained (500 examples).

###  Indentation/Spacing Deviation Handling
- **Setup**: Single format dataset with imperfect spacing (500 examples).

###  Format Understanding Ability
- **Setup**: Three experiments using two training formats (1,000 examples each).
- **Test Set**: Unseen composite format. (combination of concepts from format1 and format2)
- This tests whether the model has 'learnt' rules of format conversion.

---

## 3. Results Summary

| Task                        | Training Examples         | ML Outcome | SLM Outcome | ML Notes                                     | SLM Notes                                          |
|----------------------------|---------------------------|------------|-------------|----------------------------------------------|----------------------------------------------------|
| Changing value & length    | 500–750                   | ⚠️ Dataset dependent     | ✅ Pass      | Fails on long lists when only short-trained. Need trained examples of long lists. Has been fixed with rules| ✅ Generalizes to larger entries with different values |
| Indentation variation single format      | 500-750               | ✅ Pass     | ✅ Pass      | ✅ Passes on initial testing of changing \n and spaces                   | ✅ Passes on initial testing of changing \n and spaces |
| Simple Format Understanding (Object with list values) | ML: 750-1000 examples  | ✅ Pass     | ✅ Pass      | ✅ Pretraining can be done on format conversion fundamentals using curiculum learning and hard negatives to achieve this          | ✅ Needs smaller set of examples per format; finetuning technique refinement |
| Deep format Understanding (NestedList Experiment) | ML: 15,000+ / SLM: 500–1,000 (estimate) | ❌ Fail     | 🟡 Promising      | Requires 15,000 varied examples (estimate) for truly understanding the format rules          | ✅ Needs smaller set of examples per format; finetuning technique refinement |

> **Legend**: “SLM” refers to small‑LLM finetuning on **LLaMA 3.2 3B**.
> 
> **Link to experiments:** https://docs.google.com/spreadsheets/d/1Vg8xPi4Ag4sMt__vSpLLViC2TWu9cTrJPGHCB1mrzDc/edit?usp=sharing
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
### 4.2 Simple Format Conversion

**Objective**: Train on two simple formats and test on a nested variant (held-out).

**Format 1 (Flat List)**:
```xml
<outer>
  <lvals>word1</lvals>
  <lvals>word2</lvals>
  <lvals>word3</lvals>
</outer>
```
➡️  
```json
{
  "outer": {
    "lvals": [
      "word1",
      "word2",
      "word3"
    ]
  }
}
```

**Format 2 (List of Objects)**:
```xml
<container>
  <item>
    <name>Laptop</name>
  </item>
  <item>
    <name>Mouse</name>
  </item>
</container>
```
➡️  
```json
{
  "container": {
    "item": [
      {
        "name": "Laptop"
      },
      {
        "name": "Mouse"
      }
    ]
  }
}
```

**Held-out Test Format (Nested List)**:
```xml
<root>
  <entity>
   <name>Karthee</name>
   <name>Gautaman</name>
   <name>Adithya</name>
   <name>Dom</name>
  </entity>
</root>
```
➡️  
```json
{
  "root": {
    "entity": {
      "name": [
        "Karthee",
        "Gautaman",
        "Adithya",
        "Dom"
      ]
    }
  }
}
```
---
### 4.2 Simple Format Conversion

**Objective**: Train on two simple formats and test on a nested variant (held-out).

**Format 1 (Flat List)**:
```xml
<outer>
  <lvals>word1</lvals>
  <lvals>word2</lvals>
  <lvals>word3</lvals>
</outer>
```
➡️  
```json
{
  "outer": {
    "lvals": [
      "word1",
      "word2",
      "word3"
    ]
  }
}
```

**Format 2 (List of Objects)**:
```xml
<container>
  <item>
    <name>Laptop</name>
  </item>
  <item>
    <name>Mouse</name>
  </item>
</container>
```
➡️  
```json
{
  "container": {
    "item": [
      {
        "name": "Laptop"
      },
      {
        "name": "Mouse"
      }
    ]
  }
}
```

**Held-out Test Format (Nested List)**:
```xml
<wrapper>
  <element>
    <value>red</value>
    <value>green</value>
  </element>
  <element>
    <value>blue</value>
  </element>
</wrapper>
```
➡️  
```json
{
  "wrapper": {
    "element": [
      {
        "value": [
          "red",
          "green"
        ]
      },
      {
        "value": [
          "blue"
        ]
      }
    ]
  }
}
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
