import xml.etree.ElementTree as ET
import json
import uuid
from collections import defaultdict
import random
from io import BytesIO

class XmlToJsonPipeline:
    """
    Orchestrates the conversion of an XML string to a JSON object by splitting
    the XML into manageable chunks, creating a structural blueprint, simulating
    an ML conversion, and then reconstructing the final JSON.
    """

    def __init__(self, inference_function, list_split_threshold=100):
        """
        Initializes the pipeline.

        Args:
            list_split_threshold (int): The number of identical sibling elements
                above which they will be batched into a single "list" chunk.
        """
        self.list_split_threshold = list_split_threshold
        self.inference_function = inference_function if callable(inference_function) else self._default_simulator
        # State stores for the process
        self.blueprint = []
        self.chunks = {}
        self.root_node_id = None

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

        print("--- Step 3: Reconstructing JSON from Blueprint ---")
        final_json = self._reconstruct_from_blueprint(ml_results)
        print("Reconstruction complete.")
        
        return final_json

    # Add these methods inside the XmlToJsonPipeline class

    # def _split_and_build_blueprint(self, xml_string: str):
    #     """
    #     Parses the XML and recursively traverses it to build the blueprint
    #     and generate chunks for the ML model.
    #     """
    #     # Clear previous state
    #     self.blueprint = []
    #     self.chunks = {}
    #     self.root_node_id = None

    #     # Use ElementTree for robust XML parsing
    #     try:
    #         root_element = ET.fromstring(xml_string)
    #     except ET.ParseError as e:
    #         print(f"Error: Invalid XML provided. {e}")
    #         return

        # Start the recursive traversal from the root
        # self._traverse_node(root_element, parent_id=None)

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

    # def _split_and_build_blueprint(self, xml_string: str):
    #     self.blueprint = []
    #     self.chunks = {}
    #     self.root_node_id = None
    #     try:
    #         root_element = ET.fromstring(xml_string)
    #     except ET.ParseError as e:
    #         print(f"Error: Invalid XML provided. {e}")
    #         return
    #     # Start the recursive traversal.
    #     self._traverse_node(root_element, parent_id=None)

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

    # def _split_and_build_blueprint(self, xml_string: str):
    #     self.blueprint = []
    #     self.chunks = {}
    #     self.root_node_id = None
    #     try:
    #         root_element = ET.fromstring(xml_string)
    #     except ET.ParseError as e:
    #         print(f"Error: Invalid XML provided. {e}")
    #         return
    #     self._traverse_node(root_element, parent_id=None)

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
        self._traverse_node(root_element, parent_id=None)

    # def _traverse_node(self, element: ET.Element, parent_id: str):
    #     """
    #     Corrected traversal. Creates ONE blueprint node per element and decides
    #     if it needs a chunk.
    #     """
    #     node_id = f"{element.tag}_{uuid.uuid4().hex[:8]}"
    #     if self.root_node_id is None:
    #         self.root_node_id = node_id

    #     is_simple = self._is_simple_entity(element)
        
    #     # Create a chunk for EVERY element. The chunk's content depends on simplicity.
    #     chunk_id = f"chunk_entity_{uuid.uuid4().hex[:8]}"
    #     self.chunks[chunk_id] = self._create_entity_chunk(element, is_simple)

    #     # Create ONE blueprint node for the current element.
    #     self.blueprint.append({
    #         "node_id": node_id,
    #         "parent_id": parent_id,
    #         "tag_name": element.tag,
    #         "chunk_id": chunk_id,
    #         "is_simple_entity": is_simple,
    #     })

    #     # Recurse ONLY if the entity is complex.
    #     # If it's simple, its children are handled by the parent's chunk.
    #     if not is_simple:
    #         for child in element:
    #             self._traverse_node(child, parent_id=node_id)

    def _traverse_node(self, element: ET.Element, parent_id: str):
        """
        Traverses every node to build a complete blueprint.
        It creates chunks ONLY for "terminal complex nodes".
        """
        node_id = f"{element.tag}_{uuid.uuid4().hex[:8]}"
        if self.root_node_id is None:
            self.root_node_id = node_id

        is_terminal = self._is_terminal_complex_node(element)
        
        blueprint_node = {
            "node_id": node_id,
            "parent_id": parent_id,
            "tag_name": element.tag,
            "attributes": dict(element.attrib),
            "chunk_id": None,
        }

        if is_terminal:
            # This is a unit of work for the ML model. Create a chunk.
            chunk_id = f"chunk_{node_id}"
            try:
                # This modifies the 'element' in-place
                ET.indent(element, space="  ")
            except AttributeError:
                # ET.indent() is not available in Python < 3.9. Silently pass.
                pass
            self.chunks[chunk_id] = ET.tostring(element, encoding='unicode')
            blueprint_node["chunk_id"] = chunk_id
            # We add the node to the blueprint, but we DO NOT recurse further,
            # as the ML model is responsible for processing this entire chunk.
        
        self.blueprint.append(blueprint_node)

        if not is_terminal:
            # This is a structural parent. We MUST recurse into its children.
            for child in element:
                self._traverse_node(child, parent_id=node_id)

    # --- CORRECTED RECONSTRUCTION LOGIC ---
    # def _reconstruct_from_blueprint(self, ml_results: dict) -> dict:
    #     """
    #     Correctly reconstructs the final JSON using a bottom-up assembly
    #     that respects the "wrapping" nature of complex (hollow) entities.
    #     """
    #     # Pass 1: Get the raw, unprocessed content for every node from its ML chunk.
    #     # This dictionary holds the "starting material" for each node.
    #     raw_content = {}
    #     for node_info in self.blueprint:
    #         node_id = node_info['node_id']
    #         chunk_id = node_info['chunk_id']
    #         ml_result = ml_results.get(chunk_id)

    #         content = None
    #         if ml_result is None:
    #             content = {"__error__": "ML result missing for chunk", "chunk_id": chunk_id}
    #         elif isinstance(ml_result, str):
    #             try:
    #                 parsed_json = json.loads(ml_result)
    #                 # The model's output is a dict with one key: the tag name.
    #                 # We want the inner object.
    #                 if isinstance(parsed_json, dict) and node_info['tag_name'] in parsed_json:
    #                     content = parsed_json[node_info['tag_name']]
    #                 else:
    #                     content = parsed_json
    #             except json.JSONDecodeError:
    #                 content = {"__error__": "Malformed JSON from ML model", "raw_output": ml_result}
    #         else:
    #             content = ml_result

    #         raw_content[node_id] = content

    #     # Pass 2: Bottom-up assembly.
    #     # We iterate through the blueprint in REVERSE. This ensures that by the time
    #     # we process a parent, all of its children have already been fully assembled.
    #     final_objects = {}
        
    #     # For efficient child lookup, create a map of parent_id -> [child_node_infos]
    #     children_map = defaultdict(list)
    #     for node in self.blueprint:
    #         if node['parent_id']:
    #             children_map[node['parent_id']].append(node)

    #     for node_info in reversed(self.blueprint):
    #         node_id = node_info['node_id']
            
    #         # Start with the raw content for the current node.
    #         # This might be a full object (if simple) or just attributes (if complex).
    #         current_obj = raw_content.get(node_id, {})

    #         # If the node is complex, its children were hollowed out. We need to fill them in.
    #         if not node_info['is_simple_entity']:
    #             # Get all children of the current node. We iterate in normal order
    #             # to preserve the original XML order.
    #             for child_info in children_map.get(node_id, []):
    #                 child_id = child_info['node_id']
    #                 child_tag_name = child_info['tag_name']
                    
    #                 # Get the FULLY ASSEMBLED child object from our final_objects workspace.
    #                 # Because we are iterating in reverse, this is guaranteed to exist.
    #                 child_final_obj = final_objects.get(child_id)

    #                 # Now, place the complete child object inside the parent object.
    #                 if child_tag_name in current_obj:
    #                     if isinstance(current_obj[child_tag_name], list):
    #                         current_obj[child_tag_name].append(child_final_obj)
    #                     else:
    #                         # Convert to list if a key collision happens
    #                         current_obj[child_tag_name] = [current_obj[child_tag_name], child_final_obj]
    #                 else:
    #                     # Check if it should be a list or a single property
    #                     if len([c for c in children_map.get(node_id, []) if c['tag_name'] == child_tag_name]) > 1:
    #                         current_obj[child_tag_name] = [child_final_obj]
    #                     else:
    #                         current_obj[child_tag_name] = child_final_obj
            
    #         # The current_obj is now fully assembled (it has its own content AND its
    #         # children's fully assembled content). Store it in our final workspace.
    #         final_objects[node_id] = current_obj

    #     # The final result is the object corresponding to the root node ID.
    #     root_object = final_objects.get(self.root_node_id, {})
    #     return {self.blueprint[0]['tag_name']: root_object}

    # --- REFACTORED RECONSTRUCTION LOGIC ---
    def _reconstruct_from_blueprint(self, ml_results: dict) -> dict:
        """
        Corrected reconstruction. It now correctly handles JSON strings returned
        from the ML model by parsing them into dictionaries before processing.
        """
        final_objects = {}
        
        children_map = defaultdict(list)
        for node in self.blueprint:
            if node['parent_id']:
                children_map[node['parent_id']].append(node)

        for node_info in reversed(self.blueprint):
            node_id = node_info['node_id']
            tag_name = node_info['tag_name']
            current_obj = None

            if node_info['chunk_id']:
                chunk_id = node_info['chunk_id']
                raw_ml_result = ml_results.get(chunk_id)
                
                parsed_result = None
                if raw_ml_result is None:
                    current_obj = {"__error__": "ML result missing for chunk", "chunk_id": chunk_id}
                else:
                    # This is the critical fix: handle both strings (from real model) and dicts (from simulator)
                    if isinstance(raw_ml_result, str):
                        try:
                            parsed_result = json.loads(raw_ml_result)
                        except json.JSONDecodeError:
                            current_obj = {"__error__": "Malformed JSON from ML model", "raw_output": raw_ml_result}
                    else:
                        # It's already a Python object (e.g., from the default simulator)
                        parsed_result = raw_ml_result

                # This block only runs if parsing was successful
                if current_obj is None:
                    if isinstance(parsed_result, dict) and tag_name in parsed_result:
                        current_obj = parsed_result[tag_name]
                    else:
                        current_obj = {"__error__": "ML result format mismatch or missing tag", "parsed_output": parsed_result}
            else:
                # This is a structural-only node. Start with its attributes.
                # current_obj = node_info['attributes'].copy()
                current_obj = {f"@{k}": v for k, v in node_info['attributes'].items()}

                
                child_nodes = children_map.get(node_id, [])
                for child_info in child_nodes:
                    child_id = child_info['node_id']
                    child_tag = child_info['tag_name']
                    child_obj = final_objects.get(child_id)

                    if child_tag in current_obj:
                        if not isinstance(current_obj[child_tag], list):
                            current_obj[child_tag] = [current_obj[child_tag]]
                        current_obj[child_tag].append(child_obj)
                    else:
                        is_list = len([c for c in child_nodes if c['tag_name'] == child_tag]) > 1
                        if is_list:
                            current_obj[child_tag] = [child_obj]
                        else:
                            current_obj[child_tag] = child_obj
            
            final_objects[node_id] = current_obj

        root_object = final_objects.get(self.root_node_id, {})
        root_tag_name = self.blueprint[0]['tag_name']
        return {root_tag_name: root_object}
    

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

        # Add this method inside the XmlToJsonPipeline class

    # def _reconstruct_from_blueprint(self, ml_results: dict) -> dict:
    #     """
    #     Reconstructs the final JSON object by assembling the ML results
    #     according to the instructions in the blueprint. (Corrected Version)

    #     Args:
    #         ml_results (dict): The dictionary of {chunk_id: json_output}.

    #     Returns:
    #         dict: The fully reconstructed JSON object.
    #     """
    #     reconstructed_objects = {}
        
    #     child_tags_of_parent = defaultdict(lambda: defaultdict(int))
    #     for node in self.blueprint:
    #         if node['parent_id']:
    #             child_tags_of_parent[node['parent_id']][node['tag_name']] += 1

    #     # Pass 1: Create the base object for every node in the blueprint
    #     # (This pass remains unchanged)
    #     for node_info in self.blueprint:
    #         node_id = node_info['node_id']
    #         chunk_id = node_info['chunk_id']
    #         ml_result = ml_results.get(chunk_id)

    #         content = None
    #         if ml_result is None:
    #             content = {"__error__": "ML result missing for chunk", "chunk_id": chunk_id}
    #         else:
    #             if isinstance(ml_result, str):
    #                 try:
    #                     ml_result = json.loads(ml_result)
    #                 except json.JSONDecodeError:
    #                     content = {"__error__": "Malformed JSON from ML model", "raw_output": ml_result}

    #             if content is None:
    #                 if node_info['index_in_chunk'] is not None:
    #                     if isinstance(ml_result, list) and node_info['index_in_chunk'] < len(ml_result):
    #                         content = ml_result[node_info['index_in_chunk']]
    #                     else:
    #                         content = {"__error__": "Index out of bounds in batched result"}
    #                 else:
    #                     content = ml_result
            
    #         reconstructed_objects[node_id] = content

    #     # Pass 2: Link the objects together into a tree
    #     final_tree = {}
    #     for node_info in self.blueprint:
    #         node_id = node_info['node_id']
    #         parent_id = node_info['parent_id']
    #         tag_name = node_info['tag_name']

    #         child_obj = reconstructed_objects.get(node_id, {})
            
    #         if parent_id is None:
    #             final_tree = {tag_name: child_obj}
    #         else:
    #             parent_obj = reconstructed_objects.get(parent_id)

    #             # --- THIS IS THE CORRECTED LOGIC ---
    #             # We can only attach children to a parent that is a dictionary.
    #             # An error object IS a dictionary, so this check allows attachment.
    #             if not isinstance(parent_obj, dict):
    #                 continue # Cannot attach to a non-dict (e.g., list, string, or None)

    #             hint = 'list_item' if child_tags_of_parent[parent_id][tag_name] > 1 else 'property'

    #             if hint == 'list_item':
    #                 if tag_name not in parent_obj:
    #                     parent_obj[tag_name] = []
    #                 if isinstance(parent_obj[tag_name], list):
    #                     parent_obj[tag_name].append(child_obj)
    #                 else:
    #                     parent_obj[tag_name] = [parent_obj[tag_name], child_obj]
    #             else:
    #                 parent_obj[tag_name] = child_obj
                    
    #     return final_tree
    


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