# XML-to-JSON Conversion: Model Training Curriculum

This document outlines the structured curriculum used to train our NLP model on XML-to-JSON conversion. The curriculum is divided into two parts: the **Core Curriculum**, which teaches all fundamental rules, and the **Stress Tests**, which evaluate advanced reasoning on unseen composite structures.

### Core Curriculum (12 Formats)

These 12 formats form the foundational knowledge base for the model.

| # | Format ID | Purpose | Tags | Example XML |
|---|---|---|---|---|
| 1 | `SingleField` | Teaches basic key-value mapping. | `simple_kv`, `nested_object` | `<root><field>value</field></root>` |
| 2 | `AttributedEntity` | Teaches attribute handling in isolation. | `attribute_to_kv` | `<item id="123" status="active"/>` |
| 3 | `BasicEntity` | Teaches merging of attributes and children. | `simple_kv`, `attribute_to_kv`, `merged_object` | `<item id="123"><name>value</name></item>` |
| 4 | `PropertyList` | Teaches creating a list of simple values. | `list_of_values`, `nested_object` | `<items><item>A</item><item>B</item></items>` |
| 5 | `ListOfSimpleEntities` | Teaches creating a list of complex objects. | `list_of_objects`, `simple_kv`, `nested_object` | `<users><user><name>A</name></user><user><name>B</name></user></users>` |
| 6 | `MixedContentNode` | Teaches handling of text mixed with elements. | `mixed_content`, `attribute_to_kv`, `simple_kv` | `<p id="x">text <tag/> more text</p>` |
| 7 | `SpecialValues` | Teaches semantic interpretation (null, bool, types). | `type_casting`, `empty_element_to_null`, `self_closing_to_bool` | `<data><age>25</age><active/><notes></notes></data>` |
| 8 | `NamespacedObject` | Teaches handling of `prefix:tag` and `xmlns`. | `namespace_handling`, `attribute_to_kv` | `<doc xmlns:n="uri"><n:item/></doc>` |
| 9 | `HeterogeneousList` | Teaches lists with mixed content types. | `list_of_objects`, `list_of_values`, `mixed_content` | `<events><event>text</event><event><name>A</name></event></events>` |
| 10 | `CDataField` | Teaches treating CDATA as a literal string. | `cdata_section`, `simple_kv` | `<script><![CDATA[<p>code</p>]]></script>` |
| 11 | `ProcessingInstructionNode` | Teaches handling of `<? ... ?>` instructions. | `processing_instruction` | `<?xml-stylesheet ...?><doc/>` |
| 12 | `EntityReferenceField` | Teaches substitution of `&entity;` in text. | `simple_kv` | `<footer>Copyright © &year;</footer>` |

---

### Stress Tests (4 Formats)

These formats are held out from training. They test the model's ability to compose the foundational rules in novel and complex ways.

| # | Format ID | Purpose | Tags | Example XML |
|---|---|---|---|---|
| 1 | `ListOfLists` | Tests **pure recursion** by nesting a list of values inside a list of objects. | `list_of_objects`, `list_of_values`, `attribute_to_kv` | `<days><day name="Mon"><task>A</task><task>B</task></day>...</days>` |
| 2 | `DeeplyNested` | Tests **structural resilience** and context management over many levels. | `nested_object`, `simple_kv` | `<L1><L2><L3><L4>val</L4></L3></L2></L1>` |
| 3 | `ListOfNamespacedEntities` | Tests applying an advanced rule (`namespace`) within a common structure (`list`). | `list_of_objects`, `namespace_handling`, `attribute_to_kv` | `<feed xmlns:n="uri"><n:item/><n:item/></feed>` |
| 4 | `ComplexMixedContent` | Tests **robust sequential processing** of chaotic, mixed node types. | `mixed_content`, `namespace_handling`, `self_closing_to_bool` | `<log>text <n:tag/> more text <flag/></log>` |