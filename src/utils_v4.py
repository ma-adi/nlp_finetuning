import xml.etree.ElementTree as ET
import json
import uuid
from collections import defaultdict
import random
from io import BytesIO
import tiktoken

class XmlToJsonPipeline:
    """
    Orchestrates the conversion of an XML string to a JSON object by splitting
    the XML into manageable chunks, creating a structural blueprint, simulating
    an ML conversion, and then reconstructing the final JSON.
    """

    def __init__(self, inference_function, chunk_token_limit=25, child_batch_token_limit=50, attr_batch_token_limit=20):
        """
        Initializes the pipeline.

        Args:
            list_split_threshold (int): The number of identical sibling elements
                above which they will be batched into a single "list" chunk.
        """
        self.inference_function = inference_function if callable(inference_function) else self._default_simulator
        # State stores for the process
        self.blueprint = []
        self.chunks = {}
        self.root_node_id = None
        self.chunk_token_limit = chunk_token_limit
        self.child_batch_token_limit = child_batch_token_limit
        self.attr_batch_token_limit = attr_batch_token_limit
        # Initialize the tokenizer once for efficiency
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback for environments where tiktoken might not be installed
            print("Warning: tiktoken not found. Using character count as a fallback.")
            self.tokenizer = None

    def _get_token_count(self, text: str) -> int:
        """Calculates the number of tokens in a string, with a fallback to character length."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback if tiktoken is not available
            return len(text)

    def _is_simple_entity(self, element: ET.Element) -> bool:
        """
        Checks if an element is "simple," meaning all its direct children
        are leaf nodes (have no children of their own).
        """
        if not list(element): # If the element itself is a leaf, it's simple.
            return True
        # Check if any child has its own children.
        for child in element:
            if len(list(child)) > 0:
                return False # Found a grandchild, so this entity is not simple.
        return True

    def process(self, xml_string: str) -> dict:
        """
        Executes the full XML-to-JSON pipeline.

        Args:
            xml_string (str): The input XML document as a string.

        Returns:
            dict: The final, reconstructed JSON object.
        """
        print("--- Step 1: Splitting XML and Building Blueprint ---")
        self._split_and_build_blueprint(xml_string)
        print(f"Blueprint created with {len(self.blueprint)} nodes.")
        print(f"XML split into {len(self.chunks)} chunks for processing.\n")
        print(json.dumps(self.chunks, indent=2))
        print("--- Step 2: Running ML Model Inference ---")
        ml_results = self._run_inference()
        print("Inference complete.\n")
        print("ML results: ", ml_results)
        print("--- Step 3: Reconstructing JSON from Blueprint ---")
        final_json = self._reconstruct_from_blueprint(ml_results)
        print("Reconstruction complete.")
        
        return final_json


    def _hollow_out_element(self, element: ET.Element) -> str:
        """Creates a string representation of an element with its attributes but no children."""
        # Create a new element with the same tag and attributes to avoid modifying the original tree
        hollow_element = ET.Element(element.tag, element.attrib)
        # Return as a unicode string
        return ET.tostring(hollow_element, encoding='unicode').strip()

    def _create_entity_chunk(self, element: ET.Element, is_simple: bool) -> str:
        """
        Creates a chunk for an element.
        - If 'is_simple' is True, it includes the full children.
        - If 'is_simple' is False, it includes hollowed-out children.
        """
        # Create a deep copy to avoid modifying the original tree
        element_copy = ET.fromstring(ET.tostring(element))
        
        if not is_simple:
            # For complex entities, hollow out the children
            for child in element_copy:
                child.clear()
        
        # For simple entities, we do nothing, leaving the children intact.
        return ET.tostring(element_copy, encoding='unicode').strip()


    def _create_simple_entity_chunk(self, element: ET.Element) -> str:
        """Creates a chunk for a simple entity, which includes its children."""
        return ET.tostring(element, encoding='unicode').strip()
    
    def _register_namespaces(self, xml_string: str) -> dict:
        """
        Finds all xmlns declarations in the XML and registers them with ElementTree.
        This helps preserve original prefixes (e.g., 'doc:') when re-serializing.
        """
        source = BytesIO(xml_string.encode('utf-8'))
        namespaces = dict([
            node for _, node in ET.iterparse(source, events=['start-ns'])
        ])
        for prefix, uri in namespaces.items():
            ET.register_namespace(prefix, uri)
        return namespaces


    def _group_children_by_tag(self, element: ET.Element) -> dict:
        """Helper function to group direct children of an element by their tag name."""
        groups = defaultdict(list)
        for child in element:
            groups[child.tag].append(child)
        return groups

    # NEW HELPER FUNCTION - THE GATEKEEPER
    def _create_or_recurse_on_chunk(self, parent_id: str, base_name: str, batch_elements: list):
        """
        Vets a potential chunk. If it's small enough, creates it. 
        If it's too big, recurses on its contents instead.
        """
        wrapper = ET.Element("wrapper")
        wrapper.extend(batch_elements)
        xml_string = ET.tostring(wrapper, encoding='unicode')

        # THE VETTING STEP
        if self._get_token_count(xml_string) < self.child_batch_token_limit:
            # The batch is small enough. Create the final chunk.
            new_chunk_id = f"chunk_{base_name}"
            self.chunks[new_chunk_id] = xml_string
            self.blueprint.append({
                "node_id": new_chunk_id,
                "parent_id": parent_id,
                "tag_name": "wrapper",
                "chunk_id": new_chunk_id,
                "is_batch": True
            })
        else:
            # The batch is STILL too big. Do not create a chunk.
            # Instead, process each of its elements individually.
            print(f"-> Batch '{base_name}' was still too large. Recursing on its {len(batch_elements)} elements.")
            for element in batch_elements:
                self._traverse_and_chunk(element, parent_id=parent_id)


    # THE CORRECTED TRAVERSAL FUNCTION
    def _traverse_and_chunk(self, element: ET.Element, parent_id: str):
        """
        Builds the blueprint and uses the gatekeeper helper to ensure
        all chunks are properly vetted.
        """
        node_id = f"{element.tag}_{uuid.uuid4().hex[:8]}"
        if self.root_node_id is None:
            self.root_node_id = node_id

        blueprint_node = {
            "node_id": node_id,
            "parent_id": parent_id,
            "tag_name": element.tag,
            "attributes": dict(element.attrib),
            "chunk_id": None,
        }
        self.blueprint.append(blueprint_node)

        # --- Attribute splitting logic remains the same ---
        hollow_element_str = ET.tostring(ET.Element(element.tag, element.attrib), encoding='unicode')
        if self._get_token_count(hollow_element_str) > self.attr_batch_token_limit:
            blueprint_node["is_attribute_container"] = True
            # ... (rest of attribute splitting code is correct and unchanged)
            attributes = list(element.attrib.items())
            current_batch_attrs = []
            batch_num = 0
            for key, value in attributes:
                current_batch_attrs.append((key, value))
                temp_element = ET.Element(element.tag, dict(current_batch_attrs))
                temp_element_str = ET.tostring(temp_element, encoding='unicode')
                if self._get_token_count(temp_element_str) > self.attr_batch_token_limit and len(current_batch_attrs) > 1:
                    final_batch_attrs = current_batch_attrs[:-1]
                    sub_element = ET.Element(element.tag, dict(final_batch_attrs))
                    new_chunk_id = f"chunk_{element.tag}_attr_batch_{batch_num}"
                    self.chunks[new_chunk_id] = ET.tostring(sub_element, encoding='unicode')
                    self.blueprint.append({
                        "node_id": new_chunk_id, "parent_id": node_id,
                        "tag_name": element.tag, "chunk_id": new_chunk_id, "is_attribute_batch": True
                    })
                    batch_num += 1
                    current_batch_attrs = [(key, value)]
            if current_batch_attrs:
                sub_element = ET.Element(element.tag, dict(current_batch_attrs))
                new_chunk_id = f"chunk_{element.tag}_attr_batch_{batch_num}"
                self.chunks[new_chunk_id] = ET.tostring(sub_element, encoding='unicode')
                self.blueprint.append({
                    "node_id": new_chunk_id, "parent_id": node_id,
                    "tag_name": element.tag, "chunk_id": new_chunk_id, "is_attribute_batch": True
                })


        # --- Child splitting logic with the fix ---
        child_groups = self._group_children_by_tag(element)
        processed_children = set()

        for tag, children in child_groups.items():
            if not children: continue

            batches = []
            current_batch = []
            for child in children:
                current_batch.append(child)
                wrapper = ET.Element("wrapper")
                wrapper.extend(current_batch)
                if self._get_token_count(ET.tostring(wrapper, encoding='unicode')) > self.child_batch_token_limit and len(current_batch) > 1:
                    batches.append(current_batch[:-1])
                    current_batch = [child]
            if current_batch:
                batches.append(current_batch)

            if batches:
                # --- START: THE FIX IS HERE ---
                # If we are making any batches at all, this node is a container.
                # This is the crucial flag that prevents a hollow chunk from being made for the parent.
                blueprint_node["is_list_container"] = True
                # --- END: THE FIX ---

                for i, batch_elements in enumerate(batches):
                    base_name = f"{element.tag}_{tag}_batch_{i}"
                    self._create_or_recurse_on_chunk(node_id, base_name, batch_elements)

                for child in children:
                    processed_children.add(child)

        # --- Final check remains the same, but now works correctly ---
        remaining_children = [c for c in element if c not in processed_children]

        # A node is only a "final chunk" if it's NOT a container of any kind.
        if not remaining_children and not blueprint_node.get("is_attribute_container") and not blueprint_node.get("is_list_container"):
            blueprint_node["chunk_id"] = f"chunk_{node_id}"
            # --- FIX: Use the full element content, not the hollowed-out string ---
            # This correctly includes leaf children like <format> or text content.
            self.chunks[blueprint_node["chunk_id"]] = self._create_simple_entity_chunk(element) 
        else:
            # Otherwise, if it has remaining children that weren't batched, recurse on them.
            for child in remaining_children:
                self._traverse_and_chunk(child, parent_id=node_id)


    def _split_and_build_blueprint(self, xml_string: str):
        """
        Builds the blueprint and chunks in a single, robust pass.
        This orchestrates the new `_traverse_and_chunk` logic.
        """
        # Reset state
        self.blueprint = []
        self.chunks = {}
        self.root_node_id = None
        
        self._register_namespaces(xml_string)

        try:
            root_element = ET.fromstring(xml_string)
        except ET.ParseError as e:
            print(f"Error: Invalid XML provided. {e}")
            return

        # Start the single-pass traversal from the root
        self._traverse_and_chunk(root_element, parent_id=None)


    def _reconstruct_from_blueprint(self, ml_results: dict) -> dict:
        """
        Reconstructs the final JSON from the blueprint and ML results.
        This version is robustly designed to handle attribute and child batches correctly.
        """
        workspace = {}
        
        node_map = {n['node_id']: n for n in self.blueprint}
        children_map = defaultdict(list)
        for node in self.blueprint:
            if node['parent_id']:
                children_map[node['parent_id']].append(node)

        for node_info in reversed(self.blueprint):
            node_id = node_info['node_id']
            tag_name = node_info['tag_name']
            
            content_for_this_node = {}

            if node_info.get("chunk_id"):
                chunk_id = node_info['chunk_id']
                raw_ml_result = ml_results.get(chunk_id, "{}")
                
                try:
                    parsed_result = json.loads(raw_ml_result) if isinstance(raw_ml_result, str) else raw_ml_result
                except json.JSONDecodeError:
                    parsed_result = {"__error__": "Malformed JSON from ML", "raw_output": raw_ml_result}

                if isinstance(parsed_result, dict) and tag_name in parsed_result:
                    content_for_this_node = parsed_result[tag_name]
                else:
                    content_for_this_node = parsed_result
            
            else:
                content_for_this_node = {f"@{k}": v for k, v in node_info['attributes'].items()}
                
                for child_info in children_map.get(node_id, []):
                    child_id = child_info['node_id']
                    child_tag = child_info['tag_name']
                    child_content = workspace.get(child_id, {})

                    if child_info.get("is_attribute_batch"):
                        if isinstance(child_content, dict):
                            content_for_this_node.update(child_content)
                    
                    # --- START: THE FIX IS HERE ---
                    elif child_info.get("is_batch"):
                        # This child is a wrapper for a batch. Its content is a dictionary.
                        # e.g., {"zone": [...]} OR {"border-color": "...", "border-style": "..."}
                        if isinstance(child_content, dict):
                            for key, value in child_content.items():
                                # Check if the parent already has this key
                                if key not in content_for_this_node:
                                    # If not, just add it. The value is whatever the ML gave us.
                                    content_for_this_node[key] = value
                                else:
                                    # The key exists. We need to merge/append.
                                    # First, ensure the existing entry is a list.
                                    if not isinstance(content_for_this_node[key], list):
                                        content_for_this_node[key] = [content_for_this_node[key]]

                                    # Now, append/extend the new value(s).
                                    if isinstance(value, list):
                                        content_for_this_node[key].extend(value)
                                    else:
                                        content_for_this_node[key].append(value)
                    # --- END: THE FIX ---
                    
                    else:
                        # Default handling for a normal, nested child.
                        if child_tag not in content_for_this_node:
                            content_for_this_node[child_tag] = child_content
                        else:
                            # If key already exists, turn it into a list.
                            if not isinstance(content_for_this_node[child_tag], list):
                                content_for_this_node[child_tag] = [content_for_this_node[child_tag]]
                            content_for_this_node[child_tag].append(child_content)

            workspace[node_id] = content_for_this_node

        root_content = workspace.get(self.root_node_id, {})
        root_tag_name = node_map[self.root_node_id]['tag_name']
        return {root_tag_name: root_content}
    
# In your XmlToJsonPipeline class

    def _reconstruct_from_chunks(self, ml_results: dict) -> dict:
        """
        A simple stitcher that merges the results from the recursively split chunks.
        """
        final_json = {}

        def deep_merge(d1, d2):
            """Deeply merges d2 into d1."""
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    deep_merge(d1[k], v)
                elif k in d1 and isinstance(d1[k], list) and isinstance(v, list):
                    d1[k].extend(v)
                else:
                    d1[k] = v
            return d1

        for chunk_id, raw_ml_result in ml_results.items():
            try:
                parsed_result = json.loads(raw_ml_result) if isinstance(raw_ml_result, str) else raw_ml_result
                final_json = deep_merge(final_json, parsed_result)
            except json.JSONDecodeError:
                final_json[f"__error_{chunk_id}"] = {
                    "message": "Malformed JSON from ML",
                    "raw_output": raw_ml_result
                }
        return final_json


    def _is_terminal_complex_node(self, element: ET.Element) -> bool:
        """
        Checks if an element is a "terminal complex node".
        This is a node that may have children, but none of its children have children.
        These are the ideal "units of work" to send to the ML model.
        Example: <zone-style> containing multiple <format/> children.
        """
        if not list(element): # A simple leaf node is also a terminal node.
            return True
        # Check if any child has its own children (grandchildren).
        for child in element:
            if len(list(child)) > 0:
                return False # Found a grandchild, so this is a structural parent, not terminal.
        return True

    def _run_inference(self) -> dict:
        """Calls the provided inference function with the generated chunks."""
        if not self.chunks:
            print("Warning: No chunks were generated to run inference on.")
            return {}
        return self.inference_function(self.chunks)

    def _run_inference(self) -> dict:
        """Calls the provided inference function with the generated chunks."""
        if not self.chunks:
            print("Warning: No chunks were generated to run inference on.")
            return {}
        # The external function is responsible for its own logic (batching, etc.)
        return self.inference_function(self.chunks)

    def _default_simulator(self, chunks: dict) -> dict:
        """
        Simulates the ML model converting XML chunks to JSON objects.
        This function intentionally introduces errors to test the pipeline.

        Args:
            chunks (dict): A dictionary of {chunk_id: xml_string}.

        Returns:
            dict: A dictionary of {chunk_id: json_string_or_object}.
        """
        results = {}
        for chunk_id, xml_string in chunks.items():
            # Introduce a 10% chance of generating a malformed JSON
            if random.random() < 0.1:
                results[chunk_id] = '{"malformed": "json", "reason": "simulated error", }' # Extra comma
                continue

            try:
                root = ET.fromstring(xml_string)
                # Handle batched lists within a <wrapper>
                if root.tag == 'wrapper':
                    output = [dict(child.attrib) for child in root]
                # Handle single, hollowed-out elements
                else:
                    output = dict(root.attrib)
                
                # The ML model would output a string, which we then parse.
                # We'll just store the object directly for simplicity, but a real
                # implementation would have `json.dumps(output)`.
                results[chunk_id] = output

            except ET.ParseError:
                results[chunk_id] = '{"error": "invalid_xml_chunk"}'

        return results


# --- Main execution block ---
if __name__ == "__main__":
    # Your provided XML example
    input_xml = """
<dashboard _.fcp.AccessibleZoneTabOrder.true...enable-sort-zone-taborder='true' name='Area_context_filter'>
    <style />
    <size maxheight='800' maxwidth='1000' minheight='800' minwidth='1000' />
    <datasources>
        <datasource caption='Orders (Super_Store_Sales)' name='federated.01m8s430ttzqwp11ntkqx1t7bri8' />
    </datasources>
    <datasource-dependencies datasource='federated.01m8s430ttzqwp11ntkqx1t7bri8'>
        <column datatype='string' name='[Category]' role='dimension' type='nominal' />
        <column caption='Sub Category' datatype='string' name='[Sub_Category]' role='dimension' type='nominal' />
        <column-instance column='[Category]' derivation='None' name='[none:Category:nk]' pivot='key' type='nominal' />
        <column-instance column='[Sub_Category]' derivation='None' name='[none:Sub_Category:nk]' pivot='key' type='nominal' />
    </datasource-dependencies>
    <zones>
        <zone h='100000' id='4' type-v2='layout-basic' w='100000' x='0' y='0'>
            <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
            </zone-style>
        </zone>
    </zones>
    <simple-id uuid='{2D1B3BF2-337D-4CC5-8B7B-007CBBACE9BA}' />
</dashboard>
    """

    # Instantiate and run the pipeline
    # Set a low threshold to demonstrate both list and property logic
    pipeline = XmlToJsonPipeline(list_split_threshold=3)
    
    reconstructed_json = pipeline.process(input_xml)

    print("\n--- Final Reconstructed JSON ---")
    # Use json.dumps for pretty printing
    print(json.dumps(reconstructed_json, indent=2))