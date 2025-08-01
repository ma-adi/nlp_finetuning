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

    def __init__(self, inference_function, chunk_token_limit=25, child_batch_token_limit=100, attr_batch_token_limit=25):
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


    def _split_and_build_blueprint(self, xml_string: str):
        self.blueprint = []
        self.chunks = {}
        self.root_node_id = None

        # --- FIX 1: Register namespaces before processing ---
        self._register_namespaces(xml_string)
        # ---------------------------------------------------

        try:
            root_element = ET.fromstring(xml_string)
        except ET.ParseError as e:
            print(f"Error: Invalid XML provided. {e}")
            return
        
        ## Splitting based on nesting units (with chunkable logic.. still working on it)

        self._traverse_node(root_element, parent_id=None)

        ## Splitting based on length

        self._sub_split_large_chunks()


    def _is_chunkable_unit(self, element: ET.Element) -> bool:
        """
        Determines if an element is a self-contained "unit of work" for the ML model.
        A chunkable unit is an element whose children (if any) are all leaf nodes.
        This correctly identifies <style/>, <zone-style>, and <column> as units,
        while treating containers like <zones> and <datasource-dependencies> as
        structural parents that need to be recursed into.
        """
        # If the element has no children, it's a simple, chunkable leaf.
        if not list(element):
            return True

        # If it has children, check if any of THEM have children (i.e., grandchildren).
        for child in element:
            if len(list(child)) > 0:
                # Found a grandchild. This means the current element is a structural
                # container (e.g., <zones>), not a chunkable unit.
                return False

        # If we passed the check, it means all children are leaves.
        # This makes the current element a perfect, self-contained chunk.
        return True

    def _sub_split_large_chunks(self):
        """
        Second pass after initial chunking. Finds chunks that are too large based on
        a TOKEN LIMIT, then splits them based on whether they are "long" (too many
        children) or "wide" (too many attributes), using specialized thresholds for each.
        """
        original_chunks = list(self.chunks.items())

        for chunk_id, xml_string in original_chunks:
            # The "Gatekeeper" check remains the same
            if self._get_token_count(xml_string) <= self.chunk_token_limit:
                continue

            print(f"Chunk '{chunk_id}' is too large ({self._get_token_count(xml_string)} tokens). Analyzing for split...")
            
            try:
                root = ET.fromstring(xml_string)
                children = list(root)

                original_blueprint_node = next((n for n in self.blueprint if n.get("chunk_id") == chunk_id), None)
                if not original_blueprint_node:
                    continue

                if len(children) > 1:
                    # --- A) "LONG" ELEMENT: Use the child batch specialist threshold ---
                    print(f"-> Detected 'long' element. Splitting by children with limit: {self.child_batch_token_limit}...")
                    
                    del self.chunks[chunk_id]
                    original_blueprint_node["chunk_id"] = None
                    original_blueprint_node["is_list_container"] = True

                    current_batch = []
                    batch_num = 0
                    for child in children:
                        current_batch.append(child)
                        temp_wrapper = ET.Element("wrapper")
                        temp_wrapper.extend(current_batch)
                        batch_str = ET.tostring(temp_wrapper, encoding='unicode')

                        # --- CHANGE: Use the child-specific threshold ---
                        if self._get_token_count(batch_str) > self.child_batch_token_limit and len(current_batch) > 1:
                            final_batch_elements = current_batch[:-1]
                            # ... (rest of this block is unchanged)
                            wrapper = ET.Element("wrapper")
                            wrapper.extend(final_batch_elements)
                            new_chunk_id = f"chunk_{original_blueprint_node['tag_name']}_child_batch_{batch_num}"
                            self.chunks[new_chunk_id] = ET.tostring(wrapper, encoding='unicode')
                            self.blueprint.append({
                                "node_id": new_chunk_id, "parent_id": original_blueprint_node["node_id"],
                                "tag_name": "wrapper", "chunk_id": new_chunk_id, "is_batch": True
                            })
                            batch_num += 1
                            current_batch = [child]

                    if current_batch:
                        # ... (this block is unchanged)
                        wrapper = ET.Element("wrapper")
                        wrapper.extend(current_batch)
                        new_chunk_id = f"chunk_{original_blueprint_node['tag_name']}_child_batch_{batch_num}"
                        self.chunks[new_chunk_id] = ET.tostring(wrapper, encoding='unicode')
                        self.blueprint.append({
                            "node_id": new_chunk_id, "parent_id": original_blueprint_node["node_id"],
                            "tag_name": "wrapper", "chunk_id": new_chunk_id, "is_batch": True
                        })

                else:
                    # --- B) "WIDE" ELEMENT: Use the attribute batch specialist threshold ---
                    print(f"-> Detected 'wide' element. Splitting by attributes with limit: {self.attr_batch_token_limit}...")

                    del self.chunks[chunk_id]
                    original_blueprint_node["chunk_id"] = None
                    original_blueprint_node["is_attribute_container"] = True

                    attributes = list(root.attrib.items())
                    current_batch_attrs = []
                    batch_num = 0

                    for key, value in attributes:
                        current_batch_attrs.append((key, value))
                        temp_element = ET.Element(root.tag, dict(current_batch_attrs))
                        batch_str = ET.tostring(temp_element, encoding='unicode')

                        # --- CHANGE: Use the attribute-specific threshold ---
                        if self._get_token_count(batch_str) > self.attr_batch_token_limit and len(current_batch_attrs) > 1:
                            final_batch_attrs = current_batch_attrs[:-1]
                            # ... (rest of this block is unchanged)
                            sub_element = ET.Element(root.tag, dict(final_batch_attrs))
                            new_chunk_id = f"chunk_{original_blueprint_node['tag_name']}_attr_batch_{batch_num}"
                            self.chunks[new_chunk_id] = ET.tostring(sub_element, encoding='unicode')
                            self.blueprint.append({
                                "node_id": new_chunk_id, "parent_id": original_blueprint_node["node_id"],
                                "tag_name": root.tag, "chunk_id": new_chunk_id, "is_attribute_batch": True
                            })
                            batch_num += 1
                            current_batch_attrs = [(key, value)]

                    if current_batch_attrs:
                        # ... (this block is unchanged)
                        sub_element = ET.Element(root.tag, dict(current_batch_attrs))
                        new_chunk_id = f"chunk_{original_blueprint_node['tag_name']}_attr_batch_{batch_num}"
                        self.chunks[new_chunk_id] = ET.tostring(sub_element, encoding='unicode')
                        self.blueprint.append({
                            "node_id": new_chunk_id, "parent_id": original_blueprint_node["node_id"],
                            "tag_name": root.tag, "chunk_id": new_chunk_id, "is_attribute_batch": True
                        })
                    
                    if len(children) == 1:
                        self._traverse_node(children[0], parent_id=original_blueprint_node["node_id"])

            except ET.ParseError:
                print(f"Warning: Chunk '{chunk_id}' is too large but could not be parsed for splitting.")
                continue


    def _traverse_node(self, element: ET.Element, parent_id: str):
        """
        Traverses every node to build a complete blueprint.
        It creates chunks ONLY for "chunkable units" as defined by the new logic.
        """
        node_id = f"{element.tag}_{uuid.uuid4().hex[:8]}"
        if self.root_node_id is None:
            self.root_node_id = node_id

        is_chunkable = self._is_chunkable_unit(element)
        
        blueprint_node = {
            "node_id": node_id,
            "parent_id": parent_id,
            "tag_name": element.tag,
            "attributes": dict(element.attrib),
            "chunk_id": None,
        }

        if is_chunkable:
            # This is a unit of work. Create a chunk for the entire element.
            chunk_id = f"chunk_{node_id}"
            try:
                # Optional: pretty-print the XML for the model
                ET.indent(element, space="  ")
            except AttributeError:
                pass # ET.indent() not in Python < 3.9
            self.chunks[chunk_id] = ET.tostring(element, encoding='unicode')
            blueprint_node["chunk_id"] = chunk_id
            # Add the node to the blueprint, but DO NOT recurse. The ML model
            # handles this entire unit.
        
        self.blueprint.append(blueprint_node)

        if not is_chunkable:
            # This is a structural parent. We MUST recurse into its children.
            for child in element:
                self._traverse_node(child, parent_id=node_id)


    def _reconstruct_from_blueprint(self, ml_results: dict) -> dict:
        """
        Final corrected and resilient version.
        - Catches JSONDecodeError from malformed ML outputs.
        - Reports the error inline within the final JSON structure.
        - Assumes ML model returns a wrapped object: {"tag": ...}
        - Unwraps the object to store only the content in the workspace.
        - Re-wraps content when assembling the parent.
        """
        # This dictionary will store the UNWRAPPED CONTENT for each node_id.
        final_objects = {}
        
        node_map = {n['node_id']: n for n in self.blueprint}
        children_map = defaultdict(list)
        for node in self.blueprint:
            if node['parent_id']:
                children_map[node['parent_id']].append(node)

        for node_info in reversed(self.blueprint):
            node_id = node_info['node_id']
            tag_name = node_info['tag_name']
            
            content_obj = {}

            if node_info.get("chunk_id"):
                chunk_id = node_info['chunk_id']
                raw_ml_result = ml_results.get(chunk_id, {})
                
                parsed_result = None
                # --- START: RESILIENCE FIX ---
                if isinstance(raw_ml_result, str):
                    try:
                        parsed_result = json.loads(raw_ml_result)
                    except json.JSONDecodeError as e:
                        # If the JSON is malformed, create an error object instead of crashing.
                        # This allows the process to continue and reports the error in the output.
                        print(f"ERROR: Malformed JSON from ML for chunk '{chunk_id}'. Error: {e}")
                        parsed_result = {
                            "__error__": "Malformed JSON from ML model",
                            "raw_output": raw_ml_result
                        }
                else:
                    # It's already a Python object (from a simulator or other source)
                    parsed_result = raw_ml_result
                # --- END: RESILIENCE FIX ---

                # --- CHANGE FOR ATTRIBUTE BATCHES ---
                # If it's an attribute batch, the ML result might not be wrapped in the tag name.
                # The result is the attributes themselves.
                if node_info.get("is_attribute_batch"):
                    content_obj = parsed_result

                # UNWRAP the result from the model to get the pure content.
                elif isinstance(parsed_result, dict) and tag_name in parsed_result:
                    content_obj = parsed_result[tag_name]
                else:
                    # If it's an error object or in an unexpected format, use it as is.
                    # This ensures our error report is preserved.
                    content_obj = parsed_result
            else:
                # This is a structural node. Build its content from its children.
                content_obj = {f"@{k}": v for k, v in node_info['attributes'].items()}
                
                for child_info in children_map.get(node_id, []):
                    child_id = child_info['node_id']
                    child_tag = child_info['tag_name']
                    
                    # Get the UNWRAPPED content of the child from our workspace.
                    child_content = final_objects.get(child_id)

                    # --- NEW LOGIC FOR MERGING ATTRIBUTE BATCHES ---
                    if child_info.get("is_attribute_batch"):
                        # If the child was an attribute batch, merge its content
                        # directly into the parent's content object.
                        if isinstance(child_content, dict):
                            content_obj.update(child_content)
                        continue # Move to the next child
                    # --- END NEW LOGIC ---
                    
                    # Add the child's content to this parent.
                    if child_tag in content_obj:
                        if not isinstance(content_obj[child_tag], list):
                            content_obj[child_tag] = [content_obj[child_tag]]
                        content_obj[child_tag].append(child_content)
                    else:
                        is_list = len([c for c in children_map.get(node_id, []) if c['tag_name'] == child_tag]) > 1
                        content_obj[child_tag] = [child_content] if is_list else child_content
            
            # Store the UNWRAPPED content for this node.
            final_objects[node_id] = content_obj

        # The very last step: get the root's content and wrap it with the root tag.
        root_content = final_objects.get(self.root_node_id, {})
        root_tag_name = node_map[self.root_node_id]['tag_name']
        return {root_tag_name: root_content}

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