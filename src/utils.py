import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import List, Any, Union, Set
import tiktoken
import json
import re
import xml.etree.ElementTree as ET

# ==============================================================================
# REWRITTEN XML CHUNKING LOGIC
# ==============================================================================

def _get_element_token_count(element: ET.Element, encoder: tiktoken.Encoding) -> int:
    """Serializes and tokenizes a single XML element."""
    # Using 'unicode' ensures the output is a string, not bytes
    serialized = ET.tostring(element, encoding='unicode')
    return len(encoder.encode(serialized))


def _split_recursive(element: ET.Element, max_tokens: int, encoder: tiktoken.Encoding, chunks: List[str]):
    """
    Recursively splits an XML element into smaller, valid XML chunks.

    Args:
        element: The XML element to process.
        max_tokens: The maximum token limit for a chunk.
        encoder: The tiktoken encoder.
        chunks: A list to which serialized XML chunks will be appended.
    """
    # If the entire element is already under the limit, treat it as a single chunk.
    if _get_element_token_count(element, encoder) <= max_tokens:
        decl = '<?xml version="1.0" encoding="utf-8"?>\n'
        chunks.append(decl + ET.tostring(element, encoding='unicode'))
        return

    # If the element is too large, we must break it down by its children.
    # Create a "hollow shell" of the current element (tag and attributes only).
    shell = ET.Element(element.tag, element.attrib)
    
    # Calculate the token overhead of the shell (e.g., '<tag attr="val">...</tag>')
    # We do this by creating a temporary element with one tiny child.
    temp_shell_for_sizing = deepcopy(shell)
    temp_shell_for_sizing.append(ET.Element("a")) # Placeholder child
    shell_overhead = _get_element_token_count(temp_shell_for_sizing, encoder) - 1 # a is 1 token

    current_group = []
    current_group_tokens = 0

    def package_current_group():
        """Helper to package the current group of children into a valid XML chunk."""
        if not current_group:
            return
        
        new_root = deepcopy(shell)
        for child in current_group:
            new_root.append(deepcopy(child))
            
        decl = '<?xml version="1.0" encoding="utf-8"?>\n'
        chunks.append(decl + ET.tostring(new_root, encoding='unicode'))

    for child in element:
        child_token_count = _get_element_token_count(child, encoder)

        # Edge Case: A single child is too large to ever fit in a chunk with the shell.
        # We must recursively split this child.
        if shell_overhead + child_token_count > max_tokens:
            # First, package up any children we've already grouped.
            package_current_group()
            current_group, current_group_tokens = [], 0
            
            # Now, recursively call the function on the oversized child.
            # This will break it down further, adding its pieces to the final chunk list.
            _split_recursive(child, max_tokens, encoder, chunks)
            continue

        # If adding the next child to the current group would exceed the limit...
        if shell_overhead + current_group_tokens + child_token_count > max_tokens:
            # ...package the current group into a chunk...
            package_current_group()
            # ...and start a new group with the current child.
            current_group = [child]
            current_group_tokens = child_token_count
        else:
            # Otherwise, add the child to the current group.
            current_group.append(child)
            current_group_tokens += child_token_count

    # After the loop, package any remaining children in the last group.
    package_current_group()


def split_xml_into_chunks(xml_str: str,
                           max_tokens: int,
                           *,
                           model_name: str = "gpt-4") -> List[str]:
    """
    Splits an XML document recursively into multiple valid XML chunks under a token limit.
    This version descends into oversized elements to find smaller atomic units.
    """
    if max_tokens <= 50:
        raise ValueError("max_tokens must be greater than 50 to allow for basic XML structure.")
        
    encoder = tiktoken.encoding_for_model(model_name)
    
    try:
        # The parser must handle the possibility of malformed XML from the start.
        parser = ET.XMLParser(encoding='utf-8')
        root = ET.fromstring(xml_str, parser=parser)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML provided: {e}")

    xml_chunks = []
    _split_recursive(root, max_tokens, encoder, xml_chunks)
    
    # If the root element itself was too large and had no children, _split_recursive
    # would not add it. This is a fallback to ensure we don't return an empty list
    # for a single, oversized, childless element.
    if not xml_chunks and xml_str:
        print("Warning: The entire XML document is treated as a single chunk because it's oversized but cannot be subdivided.")
        decl = '<?xml version="1.0" encoding="utf-8"?>\n'
        xml_chunks.append(decl + ET.tostring(root, encoding='unicode'))

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


def repair_json_programmatically(s: str) -> Union[str, None]:
    """
    Attempts to repair a malformed JSON string using a series of rule-based steps.
    
    This is designed to fix common LLM output errors like XML-style attributes,
    un-nested objects, and truncated output.
    
    Returns:
        A potentially valid JSON string, or None if it's unrepairable.
    """
    if not isinstance(s, str):
        return None

    # For debugging: let's see the initial state
    # print(f"DEBUG: Initial malformed string: {s}")

    # Stage 1: Convert all XML-style attributes to JSON key-value pairs.
    # The key can contain hyphens. This adds a trailing comma we'll clean later.
    s = re.sub(r'([\w\-]+)\s*=\s*(".*?")', r'"\1": \2,', s)
    # print(f"DEBUG: After Stage 1 (Attribute Conversion): {s}")

    # Stage 2: Fix the un-nested object structure.
    # Look for a pattern of: `(unquoted-key) (space) ("quoted-key": ...)`
    # This is the signature of the error. We need to wrap this in an object.
    # We replace `unquoted-key "key":...` with `"unquoted-key": { "key":...`
    # The pattern looks for a word (can include hyphens), whitespace, then a quote.
    s = re.sub(r'(\w[\w\-]*)\s+(")', r'"\1": { \2', s)
    # print(f"DEBUG: After Stage 2 (Structural Correction): {s}")

    # Stage 3: Basic cleanup of common trailing garbage
    s = s.strip().rstrip('/>').strip()

    # Stage 4: Balance braces and brackets using a stack.
    # This will correctly close the new object we created in Stage 2.
    openers = []
    for char in s:
        if char in '{[':
            openers.append(char)
        elif char == '}':
            if openers and openers[-1] == '{':
                openers.pop()
        elif char == ']':
            if openers and openers[-1] == '[':
                openers.pop()

    # Close any remaining open structures in reverse order of opening
    for opener in reversed(openers):
        if opener == '{':
            s += '}'
        elif opener == '[':
            s += ']'
    # print(f"DEBUG: After Stage 4 (Balancing Brackets): {s}")

    # Stage 5: Clean up trailing commas that might exist before a closing brace/bracket.
    # This is crucial for the JSON to be valid.
    s = re.sub(r',\s*([}\]])', r'\1', s)
    # print(f"DEBUG: After Stage 5 (Comma Cleanup): {s}")

    # Final check: see if the result is valid JSON.
    try:
        json.loads(s)
        # print("DEBUG: Final string is valid JSON.")
        return s
    except json.JSONDecodeError as e:
        print(f"DEBUG: Final string is still invalid. Error: {e}")


# ==============================================================================
# UPDATED STITCHING FUNCTION
# ==============================================================================

def stitch_json_fragments(fragments: List[Union[dict, str]]) -> dict:
    """
    Convert JSON strings to dicts, attempt to repair malformed ones programmatically,
    and merge all fragments into one dict.
    """
    parsed = []
    for frag in fragments:
        if isinstance(frag, str):
            try:
                # First attempt to parse the fragment as is.
                parsed.append(json.loads(frag))
            except json.JSONDecodeError:
                # If it fails, log it and try to repair it.
                print(f"Warning: Could not parse a fragment. Attempting programmatic repair.")
                print(f"Malformed Fragment: {frag}")
                
                repaired_frag_str = repair_json_programmatically(frag)
                
                if repaired_frag_str:
                    try:
                        # The repair function already validated it, so this should pass.
                        parsed.append(json.loads(repaired_frag_str))
                        print("--- Repair Successful! ---")
                    except json.JSONDecodeError:
                        # This should be rare, but we handle it just in case.
                        print(f"Error: Repair attempt failed. Skipping.")
                        print(f"Repaired version that failed: {repaired_frag_str}")
                else:
                    # The repair function returned None, meaning it gave up.
                    print("Error: The repair utility could not fix the fragment. Skipping.")

        else: # If the fragment is already a dict
            parsed.append(frag)
            
    # The original deep_merge logic remains the same.
    # Assuming deep_merge function from your original code is available.
    return deep_merge(parsed)


def normalize_xml_indentation(xml_string: str) -> str:
    """
    Parses an XML string and re-serializes it with a standardized,
    pretty-printed indentation (2 spaces).

    This is useful for normalizing real-world XML to match the style
    of the generated training data.

    Args:
        xml_string: A string containing the raw XML data.

    Returns:
        A string containing the same XML data but with standardized indentation.
        Returns the original string if parsing fails.
    """
    try:
        # 1. Parse the raw XML string into an ElementTree object.
        #    The fromstring function is robust to whitespace issues.
        root = ET.fromstring(xml_string)

        # 2. Apply the standard indentation.
        #    This function modifies the tree in-place.
        ET.indent(root, space="  ")

        # 3. Re-serialize the tree back into a clean string.
        #    'unicode' encoding is equivalent to utf-8 for this purpose.
        normalized_string = ET.tostring(root, encoding='unicode')
        
        return normalized_string

    except ET.ParseError as e:
        print(f"Warning: Could not parse XML, returning original string. Error: {e}")
        return xml_string

# --- Example Usage ---

# Your real-world XML with 4-space indentation
real_world_xml = """
<root>
    <item>
        <name>Apple</name>
    </item>
</root>
"""

# Your training data's style (2-space indentation)
# This is what you want to match.
target_style_xml = """
<root>
  <item>
    <name>Apple</name>
  </item>
</root>
"""

# Normalize the real-world data
normalized_output = normalize_xml_indentation(real_world_xml)

print("--- Original Real-World XML ---")
print(real_world_xml)
print("\n--- Normalized XML (Matches Training Style) ---")
print(normalized_output)

# Verify that the normalized output matches the target style
# (Ignoring leading/trailing whitespace for the comparison)
assert normalized_output.strip() == target_style_xml.strip()
print("\nNormalization successful!")



def xml_parsing_with_ET(xml):
    root = ET.fromstring(xml)

    fragments = []
    for zone in root.findall(".//zone"):
        # Serialize this zone (with all nested elements) back to a string
        fragments.append(
            ET.tostring(zone, encoding="utf-8", xml_declaration=True).decode()
        )

    return fragments




#### XML splitter v2


import xml.etree.ElementTree as ET
from copy import deepcopy

def split_xml(
    root: ET.Element,
    max_tokens: int,
    encode_fn,                       # (str) -> int
    is_unit_of_interest=lambda e: True,
):
    """
    Yields well-formed XML strings, each ≤ max_tokens when run through encode_fn.
    Splits at any node where is_unit_of_interest(node) is True.
    """
    chunks = []
    _split_node(root, max_tokens, encode_fn, is_unit_of_interest, chunks)
    return chunks

def _get_token_count(elem: ET.Element, encode_fn) -> int:
    s = ET.tostring(elem, encoding="unicode")
    return encode_fn(s)

def _split_node(node, max_tokens, encode_fn, predicate, out_list):
    # If this node isn’t a “unit” we care about, just recurse into children
    if not predicate(node):
        for child in node:
            _split_node(child, max_tokens, encode_fn, predicate, out_list)
        return

    # If the entire subtree fits, emit and stop
    if _get_token_count(node, encode_fn) <= max_tokens:
        out_list.append(ET.tostring(node, encoding="unicode"))
        return

    # Otherwise we need to break it into pieces by grouping children
    # — but keep the parent’s tag & attributes as context “shell.”
    shell = ET.Element(node.tag, node.attrib)
    shell_overhead = _get_token_count(shell, encode_fn)

    group, group_tokens = [], 0
    def flush_group():
        if not group:
            return
        wrapper = deepcopy(shell)
        for child in group:
            wrapper.append(deepcopy(child))
        out_list.append(ET.tostring(wrapper, encoding="unicode"))

    for child in node:
        # if a child itself is too large, first flush whatever’s pending,
        # then recurse directly into that child (to split it further)
        child_tokens = _get_token_count(child, encode_fn)
        if shell_overhead + child_tokens > max_tokens:
            flush_group()
            group, group_tokens = [], 0
            _split_node(child, max_tokens, encode_fn, predicate, out_list)
            continue

        # if adding this child would overflow our current batch, flush then start new
        if shell_overhead + group_tokens + child_tokens > max_tokens:
            flush_group()
            group, group_tokens = [], 0

        group.append(child)
        group_tokens += child_tokens

    flush_group()



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



