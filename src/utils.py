"""
Module: xml_json_chunk_and_stitch.py

Provides:
  - split_xml_into_chunks(xml_str: str, max_tokens: int, *, model_name: str = "gpt-4") -> List[str]
      Splits any XML document into valid XML chunks under a token threshold, preserving
      namespaces and attributes. Token counting is based on the specified model.

  - deep_merge(*objs: Any) -> Any
      Deeply merges multiple JSON-like structures provided as separate args or a single
      list argument. Policies:
        * dict vs dict   -> merge keys recursively
        * list vs list   -> concatenate
        * list vs other  -> append other
        * other vs list  -> prepend other
        * scalar vs scalar -> wrap in list

  - stitch_json_fragments(fragments: List[Union[dict, str]]) -> dict
      Parses any JSON strings in the input list into dicts, then deep-merges all fragments.
      Returns a single merged dict.

Blueprint Logic:
1. **XML Chunking**
   - Parse once to get root tag, namespaces, attributes, and immediate children.
   - Serialize and tokenize each child, accumulating into chunks that stay under `max_tokens`.
   - Wrap each chunk’s children under a fresh root element, preserving context, and serialize.

2. **JSON Stitching**
   - Accept a list of JSON fragments (as dicts or JSON strings).
   - For string entries, parse with `json.loads`.
   - Use `deep_merge` to fold all fragments into one coherent JSON object.

Edge Cases:
- Handles input fragments as raw JSON strings or native dicts.
- Correctly merges arbitrary nesting, lists, and scalars.
- Preserves element order only if fragments include explicit ordering keys.

Dependencies: `tiktoken` for tokenization.
"""

import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import List, Any, Union
import tiktoken
import json


def split_xml_into_chunks(xml_str: str,
                           max_tokens: int,
                           *,
                           model_name: str = "gpt-4") -> List[str]:
    """
    Splits an XML document into multiple valid XML chunks under a token limit.
    Returns a list of serialized XML strings with declaration.
    """
    # Initialize tokenizer
    encoder = tiktoken.encoding_for_model(model_name)

    # Parse XML once
    parser = ET.XMLParser(encoding='utf-8')
    root = ET.fromstring(xml_str, parser=parser)

    # Extract root metadata
    nsmap = {k: v for k, v in root.attrib.items() if k.startswith('xmlns')}
    root_attrs = {k: v for k, v in root.attrib.items() if not k.startswith('xmlns')}
    tag = root.tag

    chunks, current_group, current_tokens = [], [], 0

    for child in list(root):
        serialized = ET.tostring(child, encoding='unicode')
        token_count = len(encoder.encode(serialized))

        if current_group and (current_tokens + token_count > max_tokens):
            chunks.append(current_group)
            current_group, current_tokens = [], 0

        current_group.append(child)
        current_tokens += token_count

    if current_group:
        chunks.append(current_group)

    # Wrap and serialize each chunk
    xml_chunks = []
    decl = '<?xml version="1.0" encoding="utf-8"?>\n'
    for group in chunks:
        new_root = ET.Element(tag, {**nsmap, **root_attrs})
        for elem in group:
            new_root.append(deepcopy(elem))
        xml_chunks.append(decl + ET.tostring(new_root, encoding='unicode'))

    return xml_chunks


def deep_merge_two(a: Any, b: Any) -> Any:
    """
    Merge two JSON-like structures.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        for key in set(a) | set(b):
            if key in a and key in b:
                out[key] = deep_merge_two(a[key], b[key])
            else:
                out[key] = a.get(key, b.get(key))
        return out

    if isinstance(a, list) and isinstance(b, list):
        return a + b
    if isinstance(a, list):
        return a + [b]
    if isinstance(b, list):
        return [a] + b

    return [a, b]


def deep_merge(*objs: Any) -> Any:
    """
    Deeply merge multiple JSON-like objects or lists.
    """
    # Support passing a single list of objects
    items = objs[0] if len(objs) == 1 and isinstance(objs[0], list) else list(objs)
    if not items:
        return {}
    merged = items[0]
    for item in items[1:]:
        merged = deep_merge_two(merged, item)
    return merged


def stitch_json_fragments(fragments: List[Union[dict, str]]) -> dict:
    """
    Convert JSON strings to dicts and merge all fragments into one dict.
    """
    parsed = []
    for frag in fragments:
        if isinstance(frag, str):
            parsed.append(json.loads(frag))
        else:
            parsed.append(frag)
    return deep_merge(parsed)


# Example CLI usage
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print('Usage: python xml_json_chunk_and_stitch.py <input.xml> <max_tokens> <model_name>')
        sys.exit(1)

    xml_str = open(sys.argv[1], 'r', encoding='utf-8').read()
    max_toks = int(sys.argv[2])
    model = sys.argv[3]

    parts = split_xml_into_chunks(xml_str, max_toks, model_name=model)
    for i, part in enumerate(parts, 1):
        print(f'--- XML Chunk {i} ---')
        print(part)
        print()

    stitched = stitch_json_fragments(parts)
    print('Stitched JSON:', json.dumps(stitched, indent=2))
