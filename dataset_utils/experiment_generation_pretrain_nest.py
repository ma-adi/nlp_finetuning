import json
import random
import string
import uuid
from lxml import etree as ET
from faker import Faker
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration & Setup ---

FAKER = Faker()

MAX_NESTING_DEPTH = 4 # Generate examples with nesting up to 3 levels deep

# --- Global Settings ---
N_EXAMPLES = 10000
DIFFICULTY_DISTRIBUTION = (0.3, 0.4, 0.3)
RANDOM_SEED = 42

# Options: 'pretty', 'random', 'none'
INDENTATION_MODE = 'none'
# --- NOTE: Set to False to test the bug and the fix.
FIXED_GRAMMAR_FOR_STRUCTURE = False
CURRICULUM_LEARNING = True

# --- Unified Tag Vocabulary ---
TAGS_VOCABULARY = [
    'simple_kv', 'attribute_to_kv', 'merged_object', 'list_of_values',
    'list_of_objects', 'nested_object', 'mixed_content', 'type_casting',
    'empty_element_to_null', 'namespace_handling',
    'cdata_section', 'processing_instruction'
]

# --- Experiment Definitions ---
EXPERIMENTS = {
    'master_curriculum': {
        'description': "A comprehensive curriculum covering 12 foundational XML rules with improved consistency.",
        'core_formats': [
            'SingleField', 'AttributedEntity', 'BasicEntity', 'PropertyList',
            'ListOfSimpleEntities', 
            # ### CHANGE 3: Replaced 'AdvancedMixedContent' with the more robust 'OrderedMixedContent'
            'OrderedMixedContent', 
            'SpecialValues', 'NamespacedObject', 'HeterogeneousList', 'CDataField',
            'EntityReferenceField','HintedEmptyNode', #'ProcessingInstructionNode',
            'AttributePromotedKeyValueList', 'ListOfAttributedEntities'
        ],
        'ap_formats': [ # Advanced Placement formats to be used as held-out tests
            'ListOfLists', 'DeeplyNested', 'ListOfNamespacedEntities'
        ]
        },
    'easy_master_curriculum': {
        'description': "More realistic compositional testing for master_curriculum -- same core formats",
        'core_formats': [
            'SingleField', 'AttributedEntity', 'BasicEntity', 'PropertyList',
            'ListOfSimpleEntities', 
            # ### CHANGE 3: Replaced 'AdvancedMixedContent' with the more robust 'OrderedMixedContent'
            'OrderedMixedContent', 
            'SpecialValues', 'NamespacedObject', 'HeterogeneousList', 'CDataField',
            'EntityReferenceField','HintedEmptyNode', #'ProcessingInstructionNode',
            'AttributePromotedKeyValueList', 'ListOfAttributedEntities'
        ],
        'ap_formats': ['NestedAttributedEntity', 'ListWithSpecialValues', 
                    'AttributedPropertyList', 'ListOfObjectsWithSpecialValues',
                    'NamespacedAttributes', 'ListOfMixedContentObjects']
    }
}

# --- Experiment Selection ---
EXPERIMENT_TO_RUN = 'easy_master_curriculum'
# HELD_OUT_TEST_FORMAT = 'ListOfLists'

# --- Dynamic Configuration based on Selection ---
config = EXPERIMENTS[EXPERIMENT_TO_RUN]
TRAINING_FORMATS = config['core_formats']
ALL_FORMATS = config['core_formats'] + config['ap_formats']
# TEST_FORMAT = HELD_OUT_TEST_FORMAT
# if TEST_FORMAT not in config['ap_formats']:
#     raise ValueError(f"Test format '{TEST_FORMAT}' is not a valid AP format for this experiment.")

# --- Difficulty Configuration ---
DIFFICULTY_CONFIG = {
    'easy': {'list_length':(1,4), 'sub_list_length':(1,2), 'nesting_depth':2, 'optional_prob':0.1},
    'medium': {'list_length':(1,9), 'sub_list_length':(1,4), 'nesting_depth':3, 'optional_prob':0.5},
    'hard': {'list_length':(2,20), 'sub_list_length':(2,8), 'nesting_depth':5, 'optional_prob':0.9}
}

# --- Base Class (with improved build step) ---
class FormatTemplate:
    def __init__(self, format_id: str, tags: List[str], **kwargs):
        self.format_id = format_id
        self.tags = tags
        self.structured_names = kwargs.pop('structured_names', False)
        self.TAG_VOCAB = [
            'data', 'item', 'record', 'field', 'value', 'entry', 'config', 'setting',
            'user', 'product', 'order', 'detail', 'property', 'attribute', 'id',
            'name', 'type', 'status', 'payload', 'wrapper', 'container', 'element',
            'group', 'flag', 'log', 'message', 'action', 'result', 'header', 'body'
        ]

    def _generate_name_map(self, keys: List[str]) -> Dict[str, str]:
        if self.structured_names:
            if len(keys) > len(self.TAG_VOCAB):
                raise ValueError(f"Not enough unique tags in vocabulary for '{self.format_id}'.")
            return {k: v for k, v in zip(keys, random.sample(self.TAG_VOCAB, len(keys)))}
        generated_names = set()
        while len(generated_names) < len(keys):
            prefix = random.choice(string.ascii_lowercase + '_')
            allowed_chars = string.ascii_lowercase + string.digits + '-._'
            num_parts = random.randint(1, 3)
            parts = [''.join(random.choices(allowed_chars, k=random.randint(3, 6))) for _ in range(num_parts)]
            name = prefix + random.choice(['-', '.', '_']).join(parts)
            generated_names.add(name[:20]) 

        return {k: v for k, v in zip(keys, list(generated_names))}

    def generate_component(self, difficulty: str = 'medium') -> Tuple[ET.Element, Dict[str, Any]]: #-> Tuple[ET.Element, Dict[str, Any]]:
        """
        This is the core generation logic that subclasses will implement.
        It should return the root XML element and the corresponding JSON dictionary.
        """
        raise NotImplementedError

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        """
        This method now calls the component generator and then builds the final
        output dict. This allows meta-generators to intercept the components.
        """
        xml_root, json_dict, format_id, tags = self.generate_component(difficulty)
    
        # Use the dynamically generated id and tags for the final output
        final_output = self._build_output(xml_root, json_dict)
        final_output['format_id'] = format_id
        final_output['tags'] = tags
        return final_output

    def _build_output(self, xml_root: ET.Element, json_dict: Dict, prolog: Optional[str] = None) -> Dict[str, Any]:
        xml_string = ""
        # ... (Indentation logic remains unchanged) ...
        if INDENTATION_MODE == 'pretty':
            try:
                ET.indent(xml_root, space="  ")
            except AttributeError:
                pass 
            xml_string = ET.tostring(xml_root, encoding='unicode')
            if prolog:
                xml_string = prolog + '\n' + xml_string
        elif INDENTATION_MODE == 'random':
            def build_chaotic_xml(element, level=0):
                indent = '\n' + ' ' * (level * random.randint(0, 4))
                parts = [f"<{element.tag}"]
                if element.attrib:
                    parts.append(" " + " ".join(f'{k}="{v}"' for k, v in element.attrib.items()))
                parts.append(">")
                if element.text and element.text.strip():
                    parts.append(element.text.strip())
                if len(element):
                    for child in element:
                        parts.append(build_chaotic_xml(child, level + 1))
                    parts.append('\n' + ' ' * (level * random.randint(0, 4)))
                parts.append(f"</{element.tag}>")
                return ''.join(parts).replace('\n', '\n' + ' ' * random.randint(0, 5))
            
            xml_string = build_chaotic_xml(xml_root).strip()
            if prolog:
                xml_string = prolog + '\n' + xml_string
        elif INDENTATION_MODE == 'none':
            xml_string = ET.tostring(xml_root, encoding='unicode', method='xml').strip().replace('\n', '').replace('  ', '')
            if prolog:
                xml_string = prolog  + xml_string
        else:
            raise ValueError(f"Invalid INDENTATION_MODE: '{INDENTATION_MODE}'. Must be 'pretty', 'random', or 'none'.")
        
        return {
            'question': xml_string,
            'answer': json.dumps(json_dict),
            # 'format_id': self.format_id,
            # 'tags': self.tags
        }

# --- Core Format Implementations (with changes) ---

class SingleField(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('SingleField', ['simple_kv', 'nested_object'], **kwargs)
    def generate_component(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'field'])
        root = ET.Element(nm['root'])
        field_el = ET.SubElement(root, nm['field'])
        # ### CHANGE 2: Ensure value can be numeric-like but is always a string
        field_val = str(FAKER.random_int(1000, 99999)) if random.random() < 0.3 else FAKER.word()
        field_el.text = field_val
        return root, {nm['root']: {nm['field']: field_val}}, self.format_id, self.tags

### LOSSLESS CHANGE ###
class AttributedEntity(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('AttributedEntity', ['attribute_to_kv'], **kwargs)
    def generate_component(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'attr1', 'attr2'])
        root = ET.Element(nm['root'], {nm['attr1']: FAKER.word(), nm['attr2']: str(FAKER.random_int(1, 100))})
        # Prefix all attribute keys with '@'
        json_data = {f"@{k}": v for k, v in root.attrib.items()}
        return root, {nm['root']: json_data}, self.format_id, self.tags

### LOSSLESS CHANGE ###
class BasicEntity(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('BasicEntity', ['simple_kv', 'attribute_to_kv', 'merged_object'], **kwargs)
    def generate_component(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'id_attr', 'field'])
        root = ET.Element(nm['root'], {nm['id_attr']: FAKER.uuid4()})
        f = ET.SubElement(root, nm['field']); f.text = FAKER.word()
        # Create a dictionary of prefixed attributes
        prefixed_attrs = {f"@{k}": v for k, v in root.attrib.items()}
        # Merge prefixed attributes with child element data
        json_data = {**prefixed_attrs, nm['field']: f.text}
        return root, {nm['root']: json_data}, self.format_id, self.tags

class PropertyList(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('PropertyList', ['list_of_values', 'nested_object'], **kwargs)
    def generate_component(self, difficulty='medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'item'])
        root = ET.Element(nm['root'])
        items = [FAKER.word() for _ in range(random.randint(*cfg['list_length']))]
        for it in items: ET.SubElement(root, nm['item']).text = it
        return root, {nm['root']: {nm['item']: items}}, self.format_id, self.tags


class ListOfSimpleEntities(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ListOfSimpleEntities', ['list_of_objects', 'simple_kv', 'nested_object'], **kwargs)
    def generate_component(self, difficulty='medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'item', 'field1', 'field2'])
        root = ET.Element(nm['root']); json_list = []
        for _ in range(random.randint(*cfg['list_length'])):
            item_el = ET.SubElement(root, nm['item'])
            # ### CHANGE 2: Values are kept as strings.
            f1_val, f2_val = FAKER.word(), str(FAKER.random_int(1, 500))
            ET.SubElement(item_el, nm['field1']).text = f1_val
            ET.SubElement(item_el, nm['field2']).text = f2_val
            json_list.append({nm['field1']: f1_val, nm['field2']: f2_val})
        return root, {nm['root']: {nm['item']: json_list}}, self.format_id, self.tags

        
### LOSSLESS CHANGE ###
class OrderedMixedContent(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('OrderedMixedContent', ['mixed_content', 'attribute_to_kv', 'simple_kv', 'nested_object'], **kwargs)
    def generate_component(self, difficulty: str = 'medium'):
        nm = self._generate_name_map(['root', 'child', 'attr'])
        root = ET.Element(nm['root'], {nm['attr']: FAKER.uuid4()})
        json_content_array = []
        # Prefix the root element's attributes
        json_data_root = {f"@{k}": v for k, v in root.attrib.items()}
        # ... rest of the logic is the same ...
        root.text = FAKER.sentence(nb_words=3) + " "; json_content_array.append(root.text.strip())
        child1_text = FAKER.word(); child1 = ET.SubElement(root, nm['child']); child1.text = child1_text; json_content_array.append({nm['child']: child1_text})
        child1.tail = " " + FAKER.sentence(nb_words=2) + " "; json_content_array.append(child1.tail.strip())
        if difficulty in ['medium', 'hard']:
            child2_text = str(FAKER.random_int(100, 500)); child2 = ET.SubElement(root, nm['child'] + '2'); child2.text = child2_text; json_content_array.append({nm['child'] + '2': child2_text})
            child2.tail = " " + FAKER.sentence(nb_words=4); json_content_array.append(child2.tail.strip())
        json_data_root['#content'] = json_content_array
        return root, {nm['root']: json_data_root}, self.format_id, self.tags
    

### REVISED CLASS - NO NAMESPACES, HANDLES AMBIGUITY ###

### REVISED CLASS - WITH MEANINGFUL DIFFICULTY USAGE ###

class HintedEmptyNode(FormatTemplate):
    """
    Teaches how to handle special "type" attributes on empty elements
    using a curriculum-based approach.
    - Easy: Only shows the special hint cases (type="object", etc.).
    - Medium: Introduces the boundary case where 'type' is a normal attribute.
    - Hard: Adds "distractor" attributes to test rule precedence.
    """
    def __init__(self, **kwargs):
        super().__init__('HintedEmptyNode', ['type_casting', 'attribute_to_kv'], **kwargs)
        self.hint_keywords = {
            'object': {},
            'list': [],
            'string': "",
            'boolean': False
        }

    def generate_component(self, difficulty: str = 'medium'):
        nm = self._generate_name_map(['root', 'field', 'sibling'])
        root = ET.Element(nm['root'])
        
        # --- Difficulty-based Logic ---
        is_hint_case = True
        if difficulty == 'easy':
            # Easy: Always demonstrate the core hint rule.
            is_hint_case = True
        elif difficulty == 'medium':
            # Medium: 50/50 chance to show the hint vs. the boundary case.
            is_hint_case = random.random() < 0.5
            ET.SubElement(root, nm['sibling']).text = FAKER.word() # Add context
        elif difficulty == 'hard':
            # Hard: Also a 50/50 chance, but we'll add distractors.
            is_hint_case = random.random() < 0.5
            ET.SubElement(root, nm['sibling']).text = FAKER.word() # Add context

        # --- Generate XML and JSON based on the decided case ---
        
        attrs = {}
        if is_hint_case:
            # --- CASE 1: Test the special hint ---
            chosen_hint = random.choice(list(self.hint_keywords.keys()))
            expected_value = self.hint_keywords[chosen_hint]
            attrs['type'] = chosen_hint

            if difficulty == 'hard':
                # Hard: Add a distractor attribute. The model must learn to ignore it.
                attrs['id'] = FAKER.uuid4()
            
            # Create the element with the generated attributes
            ET.SubElement(root, nm['field'], attrs)
            
            # The JSON output is the special value, ignoring any distractors
            json_data = {nm['field']: expected_value}

        else:
            # --- CASE 2: Test the boundary (a regular attribute) ---
            # This case only appears in 'medium' and 'hard' difficulties.
            regular_type_value = FAKER.word()
            while regular_type_value in self.hint_keywords:
                regular_type_value = FAKER.word()
            
            attrs['type'] = regular_type_value
            
            # Create the element with a normal 'type' attribute
            ET.SubElement(root, nm['field'], attrs)
            
            # The JSON output follows the standard attribute rule (@-prefix)
            json_data = {nm['field']: {'@type': regular_type_value}}

        # Add the sibling's data to the final JSON if it exists
        if root.find(nm['sibling']) is not None:
             json_data[nm['sibling']] = root.find(nm['sibling']).text

        return root, {nm['root']: json_data}, self.format_id, self.tags

### CHANGE 4: REVISED IMPLEMENTATION FOR SPECIAL VALUES ###
class SpecialValues(FormatTemplate):
    """
    Teaches consistent rules for special values:
    1. An empty element (<tag/> or <tag></tag>) becomes null.
    2. An element with text "true" or "false" becomes a boolean.
    """
    def __init__(self, **kwargs):
        # self_closing_to_bool is no longer accurate.
        super().__init__('SpecialValues', ['simple_kv', 'type_casting', 'empty_element_to_null'], **kwargs)

    def generate_component(self, difficulty: str = 'medium'):
        nm = self._generate_name_map(['root', 'is_admin', 'is_guest', 'notes', 'session_id'])
        root = ET.Element(nm['root'])
        json_data = {}

        # Rule 1: Empty element -> null
        ET.SubElement(root, nm['notes']) # Empty element
        json_data[nm['notes']] = None
        
        # Rule 2: Explicit "true"/"false" text -> boolean
        admin_el = ET.SubElement(root, nm['is_admin'])
        admin_el.text = 'true'
        json_data[nm['is_admin']] = True

        guest_el = ET.SubElement(root, nm['is_guest'])
        guest_el.text = 'false'
        json_data[nm['is_guest']] = False
        
        # Counter-example: An empty element that is NOT a boolean flag
        ET.SubElement(root, nm['session_id'])
        json_data[nm['session_id']] = None

        return root, {nm['root']: json_data}, self.format_id, self.tags

### LOSSLESS CHANGE v2 (Includes xmlns) ###
class NamespacedObject(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('NamespacedObject', ['namespace_handling', 'attribute_to_kv', 'simple_kv'], **kwargs)
    def generate_component(self, difficulty='medium'):
        nm = self._generate_name_map(['prefix', 'child', 'root'])
        ns_uri = FAKER.uri()
        nsmap = {nm['prefix']: ns_uri}
        root = ET.Element(nm['root'], nsmap=nsmap)
        child_tag = ET.QName(ns_uri, nm['child'])
        child_el = ET.SubElement(root, child_tag)
        child_el.text = FAKER.word()
        
        json_key = f'{nm["prefix"]}:{nm["child"]}'
        
        # Create a dictionary of prefixed xmlns attributes
        xmlns_attrs = {f"@xmlns:{k}": v for k, v in root.nsmap.items() if k is not None}
        
        # Merge the xmlns attributes with the child data
        json_data = {**xmlns_attrs, json_key: child_el.text}
        
        return root, {nm['root']: json_data}, self.format_id, self.tags

class HeterogeneousList(FormatTemplate):
    # ... (This class is already consistent, no changes needed) ...
    def __init__(self, **kwargs): super().__init__('HeterogeneousList', ['list_of_objects', 'list_of_values', 'mixed_content'], **kwargs)
    def generate_component(self, difficulty='medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'event', 'name', 'target'])
        root = ET.Element(nm['root']); json_list = []
        for _ in range(random.randint(*cfg['list_length'])):
            if random.random() > 0.5:
                val = FAKER.sentence(nb_words=3)
                ET.SubElement(root, nm['event']).text = val; json_list.append(val)
            else:
                item_el = ET.SubElement(root, nm['event'])
                n_val, t_val = FAKER.word(), FAKER.file_path()
                ET.SubElement(item_el, nm['name']).text = n_val
                ET.SubElement(item_el, nm['target']).text = t_val
                json_list.append({nm['name']: n_val, nm['target']: t_val})
        return root, {nm['root']: {nm['event']: json_list}}, self.format_id, self.tags

class CDataField(FormatTemplate):
    # ... (This class is already consistent, no changes needed) ...
    def __init__(self, **kwargs): super().__init__('CDataField', ['cdata_section', 'simple_kv'], **kwargs)
    def generate_component(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'script'])
        cdata_content = f'<p>Hello, {FAKER.name()}!</p><script>alert("XSS & fun");</script>'
        root = ET.Element(nm['root'])
        script_el = ET.SubElement(root, nm['script'])
        script_el.text = ET.CDATA(cdata_content)
        json_data = {nm['script']: cdata_content}
        return root, {nm['root']: json_data}, self.format_id, self.tags

### GENERALIZATION CHANGE ###
# class ProcessingInstructionNode(FormatTemplate):
#     def __init__(self, **kwargs):
#         super().__init__('ProcessingInstructionNode', ['processing_instruction'], **kwargs)
#         # A pool of realistic but varied PI targets
#         self.pi_targets = [
#             'xml-stylesheet', 'php', 'cocoon-process', 'robot-control', 'my-custom-parser'
#         ]

#     def generate_component(self, difficulty: str = 'medium'):
#         nm = self._generate_name_map(['root', 'field'])

#         # 1. Randomly select a target from the pool
#         pi_target = random.choice(self.pi_targets)

#         # 2. Generate varied data for the instruction
#         if random.random() > 0.5:
#             # Case A: Structured key-value data
#             key1, key2 = FAKER.word(), FAKER.word()
#             val1 = FAKER.file_name()
#             val2 = FAKER.word()
#             pi_data = f'{key1}="{val1}" {key2}="{val2}"'
#         else:
#             # Case B: Unstructured, simple string data
#             pi_data = FAKER.sentence(nb_words=4)

#         pi = ET.ProcessingInstruction(pi_target, pi_data)
#         prolog = ET.tostring(pi, encoding='unicode')

#         root = ET.Element(nm['root'])
#         ET.SubElement(root, nm['field']).text = 'content'

#         json_data = {
#             nm['field']: 'content',
#             '_processing_instructions': [{pi_target: pi_data}]
#         }
        
#         return root, {nm['root']: json_data}, prolog, self.format_id, self.tags
    

#     def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
#         """
#         Custom implementation of generate_example to handle the prolog string
#         returned by this class's generate_components method.
#         """
#         # Call this class's specific component generator
#         xml_root, json_dict, prolog, format_id, tags = self.generate_component(difficulty)
        
#         # Call the final build step, passing the prolog argument
#         final_output = self._build_output(xml_root, json_dict, prolog=prolog)
        
#         # Manually add the format_id and tags
#         final_output['format_id'] = format_id
#         final_output['tags'] = tags
        
#         return final_output

class EntityReferenceField(FormatTemplate):
    # ... (This class is already consistent, no changes needed) ...
    def __init__(self, **kwargs): super().__init__('EntityReferenceField', ['simple_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'footer'])
        company_name = FAKER.company().replace('&', 'and')
        year = FAKER.year()
        dtd = f'<!DOCTYPE {nm["root"]} [<!ENTITY company "{company_name}"><!ENTITY year "{year}">]>\n'
        xml_string_manual = f'<{nm["root"]}><{nm["footer"]}>Copyright © &year; &company;</{nm["footer"]}></{nm["root"]}>'
        json_text = f'Copyright © {year} {company_name}.'
        return {
            'question': dtd + xml_string_manual,
            'answer': json.dumps({nm['root']: {nm['footer']: json_text}}),
            'format_id': self.format_id, 'tags': self.tags
        }

### CHANGE 1: REVISED IMPLEMENTATION FOR CONDITIONAL ATTRIBUTE PROMOTION ###
### LOSSLESS CHANGE ###
class AttributePromotedKeyValueList(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('AttributePromotedKeyValueList', tags=['attribute_to_kv', 'merged_object', 'list_of_objects'], **kwargs)
    def generate_component(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        keys_to_map = ['root', 'item', 'key_attr', 'val_attr', 'extra_attr']
        nm = self._generate_name_map(keys_to_map)
        root = ET.Element(nm['root'])
        is_fallback_case = difficulty in ['medium', 'hard'] and random.random() < 0.4
        if is_fallback_case:
            # FALLBACK CASE: Must use '@' prefix
            json_payload = []
            for _ in range(random.randint(*cfg['list_length'])):
                attrs = {nm['key_attr']: FAKER.word(), nm['val_attr']: FAKER.word(), nm['extra_attr']: FAKER.word()}
                ET.SubElement(root, nm['item'], attrs)
                # Apply the prefix rule here
                prefixed_attrs = {f"@{k}": v for k, v in attrs.items()}
                json_payload.append(prefixed_attrs)
            final_json = {nm['root']: {nm['item']: json_payload}}
        else:
            # PROMOTION CASE: No prefix needed, as attributes are consumed
            json_payload = {}
            generated_keys = set()
            for _ in range(random.randint(*cfg['list_length'])):
                key = FAKER.word().replace(' ', '_'); value = str(FAKER.random_int(1, 500))
                ET.SubElement(root, nm['item'], {nm['key_attr']: key, nm['val_attr']: value})
                json_payload[key] = value
            final_json = {nm['root']: json_payload}
        return root, final_json, self.format_id, self.tags

class ListOfAttributedEntities(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ListOfAttributedEntities', tags=['list_of_objects', 'attribute_to_kv'], **kwargs)
    def generate_component(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'item', 'attr1', 'attr2'])
        root = ET.Element(nm['root']); json_list = []
        for _ in range(random.randint(*cfg['list_length'])):
            attrs = {nm['attr1']: FAKER.word(), nm['attr2']: FAKER.word()}
            if difficulty in ['medium', 'hard'] and random.random() > 0.5:
                attrs['optional_attr'] = str(FAKER.random_int(0,1))
            ET.SubElement(root, nm['item'], attrs)
            # Apply the prefix rule here
            prefixed_attrs = {f"@{k}": v for k, v in attrs.items()}
            json_list.append(prefixed_attrs)
        return root, {nm['root']: {nm['item']: json_list}}, self.format_id, self.tags


# --- AP / Held-Out Formats (already consistent) ---

### LOSSLESS CHANGE ###
class ListOfLists(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ListOfLists', ['list_of_objects','list_of_values','nested_object'], **kwargs)
    def generate_component(self, difficulty='medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root','day','task', 'name_attr'])
        root = ET.Element(nm['root']); out=[]
        for _ in range(random.randint(*cfg['list_length'])):
            day_name = FAKER.day_of_week()
            it = ET.SubElement(root, nm['day'], {nm['name_attr']: day_name})
            subs = [FAKER.bs() for _ in range(random.randint(*cfg['sub_list_length']))]
            for s in subs: ET.SubElement(it, nm['task']).text = s
            # Apply the prefix rule to the day's attributes
            day_data = {f"@{nm['name_attr']}": day_name, nm['task']: subs}
            out.append(day_data)
        return root, {nm['root']: {nm['day']: out}}, self.format_id, self.tags

class DeeplyNested(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('DeeplyNested', ['nested_object', 'simple_kv'], **kwargs)
    def generate_component(self, difficulty='medium'):
        depth = DIFFICULTY_CONFIG[difficulty]['nesting_depth']
        keys = [f'level{i+1}' for i in range(depth)] + ['field']
        nm = self._generate_name_map(keys)
        current_el = root = ET.Element(nm['level1'])
        for i in range(1, depth): current_el = ET.SubElement(current_el, nm[f'level{i+1}'])
        val = FAKER.word(); f = ET.SubElement(current_el, nm['field']); f.text = val
        current_dict = {nm['field']: val}
        for i in range(depth - 1, -1, -1): current_dict = {nm[f'level{i+1}']: current_dict}
        return root, current_dict, self.format_id, self.tags
    
### LOSSLESS CHANGE v2 (Includes xmlns) ###
class ListOfNamespacedEntities(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ListOfNamespacedEntities', ['list_of_objects', 'namespace_handling', 'attribute_to_kv'], **kwargs)
    def generate_component(self, difficulty='medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['prefix', 'item', 'id_attr', 'name', 'root'])
        ns_uri = FAKER.uri(); json_list = []
        nsmap = {nm['prefix']: ns_uri}
        root = ET.Element(nm['root'], nsmap=nsmap)
        item_qname = ET.QName(ns_uri, nm['item'])
        name_qname = ET.QName(ns_uri, nm['name'])
        
        for _ in range(random.randint(*cfg['list_length'])):
            item_id, item_name = FAKER.uuid4(), FAKER.word()
            item_el = ET.SubElement(root, item_qname, {nm['id_attr']: item_id})
            name_el = ET.SubElement(item_el, name_qname)
            name_el.text = item_name
            # Apply the prefix rule to the item's attributes
            json_item = {f"@{nm['id_attr']}": item_id, f'{nm["prefix"]}:{nm["name"]}': item_name}
            json_list.append(json_item)
        
        json_key = f'{nm["prefix"]}:{nm["item"]}'
        
        # Create a dictionary of prefixed xmlns attributes for the root
        xmlns_attrs = {f"@xmlns:{k}": v for k, v in root.nsmap.items() if k is not None}
        
        # The final JSON for the root object
        root_json_obj = {**xmlns_attrs, json_key: json_list}
        
        return root, {nm['root']: root_json_obj}, self.format_id, self.tags

###INTERMEDIATE FORMATS
### REVISED - AttributedPropertyList ###
class AttributedPropertyList(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('AttributedPropertyList', ['attribute_to_kv', 'list_of_values', 'merged_object'], **kwargs)

    def generate_component(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        # Added 'attr3' to the name map for the optional attribute
        nm = self._generate_name_map(['root', 'attr1', 'attr2', 'attr3', 'option'])
        
        # 1. Create root with base attributes
        root_attrs = {nm['attr1']: FAKER.word(), nm['attr2']: str(FAKER.random_int(1,100))}
        
        # --- Difficulty-based change ---
        # In 'medium' and 'hard', add an optional third attribute to test robustness
        if difficulty in ['medium', 'hard'] and random.random() < cfg['optional_prob']:
            root_attrs[nm['attr3']] = FAKER.boolean()

        root = ET.Element(nm['root'], {k: str(v) for k, v in root_attrs.items()})
        
        # 2. Add a list of simple child elements, with length controlled by difficulty
        options = [FAKER.word() for _ in range(random.randint(*cfg['list_length']))]
        for opt in options:
            ET.SubElement(root, nm['option']).text = opt
        
        # 3. Construct the JSON, ensuring all attributes (including optional ones) are captured
        json_data = {f"@{k}": str(v) for k, v in root_attrs.items()}
        json_data[nm['option']] = options
        
        return root, {nm['root']: json_data}, self.format_id, self.tags

### REVISED - ListWithSpecialValues ###
class ListWithSpecialValues(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('ListWithSpecialValues', ['list_of_values', 'type_casting'], **kwargs)

    def generate_component(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'value'])
        root = ET.Element(nm['root'])
        
        # --- Difficulty-based change ---
        # Easy difficulty only includes strings and nulls to teach the basic boundary.
        # Medium and Hard introduce booleans for more complex type casting.
        if difficulty == 'easy':
            possible_values = [
                (FAKER.word(), FAKER.word()),
                (None, None) # An empty element
            ]
        else: # medium and hard
            possible_values = [
                (FAKER.word(), FAKER.word()),
                ('true', True),
                (None, None),
                ('false', False)
            ]
        
        # Generate a list of mixed values with length controlled by difficulty
        json_list = []
        for _ in range(random.randint(*cfg['list_length'])):
            xml_val, json_val = random.choice(possible_values)
            ET.SubElement(root, nm['value']).text = xml_val
            json_list.append(json_val)

        return root, {nm['root']: {nm['value']: json_list}}, self.format_id, self.tags
    
### REVISED - NestedAttributedEntity ###
class NestedAttributedEntity(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('NestedAttributedEntity', ['nested_object', 'attribute_to_kv', 'merged_object'], **kwargs)

    def generate_component(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'child', 'attr1', 'attr2', 'grandchild', 'opt_attr'])
        root = ET.Element(nm['root'])
        
        # 1. Create the child element with base attributes
        child_attrs = {nm['attr1']: FAKER.word(), nm['attr2']: str(random.random() > 0.5).lower()}
        
        # --- Difficulty-based change ---
        # Add an optional "distractor" attribute in harder difficulties
        if difficulty in ['medium', 'hard'] and random.random() < cfg['optional_prob']:
            child_attrs[nm['opt_attr']] = FAKER.uuid4()
            
        child_el = ET.SubElement(root, nm['child'], child_attrs)
        
        # 2. Add a grandchild element to the child
        grandchild_text = FAKER.word()
        ET.SubElement(child_el, nm['grandchild']).text = grandchild_text
        
        # 3. Construct the nested JSON object, ensuring all attributes are included
        json_child_obj = {f"@{k}": v for k, v in child_attrs.items()}
        json_child_obj[nm['grandchild']] = grandchild_text
        
        json_data = {nm['child']: json_child_obj}
        
        return root, {nm['root']: json_data}, self.format_id, self.tags

### REVISED - ListOfObjectsWithSpecialValues ###
class ListOfObjectsWithSpecialValues(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('ListOfObjectsWithSpecialValues', ['list_of_objects', 'type_casting', 'empty_element_to_null'], **kwargs)

    def generate_component(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'item', 'name', 'flag', 'notes'])
        root = ET.Element(nm['root'])
        json_list = []

        # List length is controlled by difficulty
        for _ in range(random.randint(*cfg['list_length'])):
            item_el = ET.SubElement(root, nm['item'])
            
            # Setup data for one item
            name_val = FAKER.first_name()
            is_active_val = random.choice(['true', 'false'])
            
            # --- Difficulty-based change ---
            # The probability of 'notes' being null is now controlled by difficulty.
            # Easy: notes are almost always present.
            # Hard: notes are almost always null (more edge cases).
            notes_val = FAKER.sentence() if random.random() > cfg['optional_prob'] else None
            
            # Create XML
            ET.SubElement(item_el, nm['name']).text = name_val
            ET.SubElement(item_el, nm['flag']).text = is_active_val
            ET.SubElement(item_el, nm['notes']).text = notes_val
            
            # Create corresponding JSON
            json_list.append({
                nm['name']: name_val,
                nm['flag']: is_active_val == 'true',
                nm['notes']: notes_val
            })
        
        return root, {nm['root']: {nm['item']: json_list}}, self.format_id, self.tags
    
### REVISED - NamespacedAttributes ###
class NamespacedAttributes(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('NamespacedAttributes', ['attribute_to_kv', 'namespace_handling'], **kwargs)

    def generate_component(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'prefix1', 'prefix2', 'attr1', 'attr2', 'field', 'reg_attr'])
        ns_uri1 = FAKER.uri()
        ns_uri2 = FAKER.uri()
        
        # --- Difficulty-based change ---
        # The number of namespaces and the presence of distractor attributes are controlled.
        if difficulty == 'easy':
            # Easy: One namespace, one namespaced attribute.
            nsmap = {nm['prefix1']: ns_uri1}
            attrs = {ET.QName(ns_uri1, nm['attr1']): FAKER.uuid4()}
        elif difficulty == 'medium':
            # Medium: Two namespaces, two namespaced attributes.
            nsmap = {nm['prefix1']: ns_uri1, nm['prefix2']: ns_uri2}
            attrs = {
                ET.QName(ns_uri1, nm['attr1']): FAKER.uuid4(),
                ET.QName(ns_uri2, nm['attr2']): FAKER.word()
            }
        else: # hard
            # Hard: Two namespaces PLUS a regular, non-namespaced attribute as a distractor.
            nsmap = {nm['prefix1']: ns_uri1, nm['prefix2']: ns_uri2}
            attrs = {
                ET.QName(ns_uri1, nm['attr1']): FAKER.uuid4(),
                ET.QName(ns_uri2, nm['attr2']): FAKER.word(),
                nm['reg_attr']: FAKER.boolean() # The distractor
            }

        root = ET.Element(nm['root'], {k: str(v) for k,v in attrs.items()}, nsmap=nsmap)
        
        field_text = FAKER.sentence()
        ET.SubElement(root, nm['field']).text = field_text
        
        # Construct the JSON manually to match the generated XML for any difficulty
        json_data = {f"@{k}": str(v) for k, v in attrs.items()}
        for prefix, uri in nsmap.items():
            json_data[f"@xmlns:{prefix}"] = uri
        json_data[nm['field']] = field_text

        return root, {nm['root']: json_data}, self.format_id, self.tags

### REVISED - ListOfMixedContentObjects ###
class ListOfMixedContentObjects(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('ListOfMixedContentObjects', ['list_of_objects', 'mixed_content'], **kwargs)

    def generate_component(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'item', 'sub_item'])
        root = ET.Element(nm['root'])
        json_list = []

        # List length is controlled by difficulty
        for _ in range(random.randint(*cfg['list_length'])):
            # --- Difficulty-based change ---
            # 'Easy' difficulty only generates simple text items.
            # 'Medium' and 'Hard' have an increasing probability of generating complex, mixed-content items.
            is_complex = difficulty in ['medium', 'hard'] and random.random() < cfg['optional_prob']
            
            if not is_complex:
                # Generate a simple text item
                simple_text = FAKER.sentence()
                ET.SubElement(root, nm['item']).text = simple_text
                json_list.append(simple_text)
            else:
                # Generate a complex mixed-content item
                complex_item = ET.SubElement(root, nm['item'])
                complex_item.text = "Start text. "
                sub = ET.SubElement(complex_item, nm['sub_item'])
                sub.text = FAKER.word()
                sub.tail = " End text."
                
                json_list.append({
                    "#content": [
                        complex_item.text.strip(),
                        { nm['sub_item']: sub.text },
                        sub.tail.strip()
                    ]
                })
        
        return root, {nm['root']: {nm['item']: json_list}}, self.format_id, self.tags
    

### NEW CLASS ###
class NestedObjectWrapper(FormatTemplate):
    """
    A meta-generator that wraps the output of another generator in a new parent object.
    Teaches one level of object nesting.
    """
    def __init__(self, child_generators: List[FormatTemplate], **kwargs):
        # The tags for this wrapper itself
        self.base_tags = ['nested_object']
        
        # Collect all possible tags from children to create a complete potential tag set
        all_child_tags = set(tag for gen in child_generators for tag in gen.tags)

        combined_tags = set(self.base_tags) | all_child_tags

        # super().__init__('NestedObjectWrapper', list(self.base_tags | all_child_tags), **kwargs)
        super().__init__('NestedObjectWrapper', list(combined_tags), **kwargs)
        
        if not child_generators:
            raise ValueError("NestedObjectWrapper must be initialized with at least one child generator.")
        self.child_generators = child_generators

    def generate_component(self, difficulty: str = 'medium') -> Tuple[ET.Element, Dict[str, Any], str, List[str]]:
        # 1. Pick a random child format to nest
        child_generator = random.choice(self.child_generators)

        # 2. Generate the child's raw components
        child_xml, child_json, child_id, child_tags = child_generator.generate_components(difficulty)
        
        # 3. Create a new parent/wrapper
        nm = self._generate_name_map(['wrapper', 'sibling'])
        wrapper_xml = ET.Element(nm['wrapper'])
        wrapper_xml.append(child_xml) # Append the entire child element

        # Add a sibling node to make the structure less trivial
        sibling_text = FAKER.word()
        ET.SubElement(wrapper_xml, nm['sibling']).text = sibling_text

        # 4. Construct the new nested JSON
        # The child_json is like {'child_root': {...}}. We place this inside the wrapper.
        wrapper_json = {nm['wrapper']: {**child_json, nm['sibling']: sibling_text}}

        # 5. Dynamically generate the format_id and tags for this specific example
        final_format_id = f"Nested({child_id})"
        final_tags = sorted(list(set(self.base_tags) | set(child_tags)))

        return wrapper_xml, wrapper_json, final_format_id, final_tags

### NEW CLASS ###
class ListOfNestedObjectsWrapper(FormatTemplate):
    """
    A meta-generator that creates a list of complex objects, where each object
    is generated by a different child generator. Teaches heterogeneous lists of objects.
    """
    def __init__(self, child_generators: List[FormatTemplate], **kwargs):
        self.base_tags = ['list_of_objects']
        all_child_tags = set(tag for gen in child_generators for tag in gen.tags)

        combined_tags = set(self.base_tags) | all_child_tags

        super().__init__('ListOfNestedObjectsWrapper', list(combined_tags), **kwargs)

        
        if not child_generators:
            raise ValueError("ListOfNestedObjectsWrapper must be initialized with child generators.")
        self.child_generators = child_generators

    def generate_component(self, difficulty: str = 'medium') -> Tuple[ET.Element, Dict[str, Any], str, List[str]]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'item_container'])

        root_xml = ET.Element(nm['root'])
        item_container_xml = ET.SubElement(root_xml, nm['item_container'])
        
        json_list = []
        child_ids_in_list = set()

        # Generate a list of random child objects
        for _ in range(random.randint(*cfg['list_length'])):
            child_generator = random.choice(self.child_generators)
            child_xml, child_json, child_id, _ = child_generator.generate_components(difficulty)
            
            # The child's root tag becomes the item tag in the list
            item_container_xml.append(child_xml)
            # The JSON is a list of objects, preserving the child's root key
            json_list.append(child_json)
            child_ids_in_list.add(child_id)
        
        final_json = {nm['root']: {nm['item_container']: json_list}}
        
        # Create a representative format_id
        id_str = ",".join(sorted(list(child_ids_in_list)))
        final_format_id = f"ListOfNested({id_str})"
        # Tags for this format are its base tags plus all possible child tags
        final_tags = self.tags 

        return root_xml, final_json, final_format_id, final_tags
    

### NEW RECURSIVE GENERATOR CLASS ###
class CombinatorialGenerator:
    """
    Generates true combinatorial nested examples where a random format is nested
    inside a random child element of another random format.
    """
    def __init__(self, all_base_formats: Dict[str, FormatTemplate], use_structured_names: bool):
        self.use_structured_names = use_structured_names
        
        # --- STEP 1: Define all formats that are invalid for this combinatorial model ---
        # These formats will be excluded from all lists.
        excluded_formats = {
            'EntityReferenceField', # Does not implement .generate_component()
            'DeeplyNested'          # Its own complexity makes it a poor choice for nesting
        }

        # --- STEP 2: Create a clean dictionary of only valid, usable formats ---
        valid_formats = {
            name: gen for name, gen in all_base_formats.items()
            if name not in excluded_formats
        }

        # --- STEP 3: Create the list of formats that can be used as NESTED CONTENT ---
        # This can be any valid format. This list is used by the base case (depth=1).
        self.nestable_content_formats = list(valid_formats.values())

        # --- STEP 4: Create the list of formats that can act as WRAPPERS ---
        # This must be a SUBSET of valid_formats that have child elements to inject into.
        # This list is used by the recursive step (depth > 1).
        self.wrappable_formats = [
            gen for name, gen in valid_formats.items()
            if name not in [
                'AttributedEntity', # Has no child elements
                'CDataField',       # Has no child elements
                'SingleField'       # Edge case: only has one child, better to use more complex wrappers
            ]
        ]
        
        # A quick safety check
        if not self.wrappable_formats:
            raise ValueError("The list of wrappable_formats is empty. Check filtering logic.")

        # A dummy FormatTemplate instance just to get access to _build_output
        self._builder = FormatTemplate("builder", [])

    def generate_example(self, target_depth: int, difficulty: str = 'medium') -> Dict[str, Any]:
        """ Public-facing method to kick off recursive generation. """
        if target_depth < 1:
            raise ValueError("Target depth must be at least 1.")

        xml_root, json_dict, final_format_id, _ = self._generate_recursive_component(target_depth, difficulty)

        final_output = self._builder._build_output(xml_root, json_dict)
        final_output['format_id'] = final_format_id
        # We can calculate final tags later if needed, omitting for clarity here.
        final_output['tags'] = []
        return final_output

    def _generate_recursive_component(self, depth: int, difficulty: str):
        # --- BASE CASE ---
        # At the deepest level, just pick any format and return its components.
        if depth == 1:
            generator = random.choice(self.nestable_content_formats)
            # We only need the first 3 return values for the base case
            xml, json_obj, format_id, tags = generator.generate_component(difficulty)
            return xml, json_obj, format_id, tags

        # --- RECURSIVE STEP (depth > 1) ---
        
        # 1. Pick a random format to be the WRAPPER. It must be a format that has children.
        wrapper_generator = random.choice(self.wrappable_formats)
        
        # 2. Generate the wrapper's components *as a starting point*.
        wrapper_xml, wrapper_json, wrapper_id, wrapper_tags = wrapper_generator.generate_component(difficulty)
        
        # 3. Recursively call to get the CHILD content that we will inject.
        child_xml, child_json, child_id, _ = self._generate_recursive_component(depth - 1, difficulty)

        # 4. Find an "injection slot" in the wrapper's XML.
        potential_slots = list(wrapper_xml)
        if not potential_slots:
            # This should not happen due to our pre-filtering, but it's a good safeguard.
            # We'll just append the child, which is less ideal but won't crash.
            wrapper_xml.append(child_xml)
            # This JSON update is a simplification. A real implementation might need more logic.
            wrapper_json[list(wrapper_json.keys())[0]].update(child_json)

        else:
            # 5. This is the core logic: INJECT the child into the wrapper.
            injection_slot = random.choice(potential_slots)
            slot_tag_name = injection_slot.tag

            # Replace the old slot element with the new complex child_xml
            wrapper_xml.remove(injection_slot)
            wrapper_xml.append(child_xml)

            # Update the JSON to match. This is the trickiest part.
            # We assume a simple structure: {'root': {'slot_name': 'simple_value'}}
            # We need to change it to: {'root': {'child_root': {...}}}
            wrapper_root_key = list(wrapper_json.keys())[0]
            # Remove the old simple key-value from the wrapper's JSON
            if slot_tag_name in wrapper_json[wrapper_root_key]:
                 del wrapper_json[wrapper_root_key][slot_tag_name]
            # Add the new complex child JSON
            wrapper_json[wrapper_root_key].update(child_json)

        # 6. Create the final, descriptive format_id as you requested.
        final_format_id = f"NestedL{depth}({wrapper_id},{child_id})"

        return wrapper_xml, wrapper_json, final_format_id, wrapper_tags

# # --- Orchestration ---
# def generate_dataset(
#     total_examples: int,
#     formats_to_use: List[str],
#     distribution: Tuple[float, float, float] = (0.33,0.34,0.33),
#     random_seed: Optional[int] = None,
#     use_structured_names: bool = False,
#     curriculum: bool = False
# ) -> Dict[str, List[Dict[str, Any]]]:
#     if random_seed is not None:
#         random.seed(random_seed)
#         FAKER.seed_instance(random_seed)

#     all_generators = {
#         'SingleField': SingleField(structured_names=use_structured_names),
#         'AttributedEntity': AttributedEntity(structured_names=use_structured_names),
#         'BasicEntity': BasicEntity(structured_names=use_structured_names),
#         'PropertyList': PropertyList(structured_names=use_structured_names),
#         'HintedEmptyNode': HintedEmptyNode(structured_names = use_structured_names),
#         'ListOfSimpleEntities': ListOfSimpleEntities(structured_names=use_structured_names),
#         'OrderedMixedContent': OrderedMixedContent(structured_names=use_structured_names),
#         'SpecialValues': SpecialValues(structured_names=use_structured_names),
#         'NamespacedObject': NamespacedObject(structured_names=use_structured_names),
#         'HeterogeneousList': HeterogeneousList(structured_names=use_structured_names),
#         'CDataField': CDataField(structured_names=use_structured_names),
#         # 'ProcessingInstructionNode': ProcessingInstructionNode(structured_names=use_structured_names),
#         'EntityReferenceField': EntityReferenceField(structured_names=use_structured_names),
#         'AttributePromotedKeyValueList': AttributePromotedKeyValueList(structured_names=use_structured_names),
#         'ListOfAttributedEntities': ListOfAttributedEntities(structured_names=use_structured_names),
#         'ListOfLists': ListOfLists(structured_names=use_structured_names),
#         'DeeplyNested': DeeplyNested(structured_names=use_structured_names),
#         'ListOfNamespacedEntities': ListOfNamespacedEntities(structured_names=use_structured_names),
#         'NestedAttributedEntity': NestedAttributedEntity(structured_names = use_structured_names),
#         'ListWithSpecialValues': ListWithSpecialValues(structured_names = use_structured_names),
#         'AttributedPropertyList': AttributedPropertyList(structured_names = use_structured_names),
#         'ListOfObjectsWithSpecialValues': ListOfObjectsWithSpecialValues(structured_names = use_structured_names),
#         'NamespacedAttributes': NamespacedAttributes(structured_names = use_structured_names),
#         'ListOfMixedContentObjects': ListOfMixedContentObjects(structured_names = use_structured_names)
#     }

#     # Instantiate all base format generators
#     base_generators_map = {
#         'SingleField': SingleField(structured_names=use_structured_names),
#         'AttributedEntity': AttributedEntity(structured_names=use_structured_names),
#         'BasicEntity': BasicEntity(structured_names=use_structured_names),
#         'PropertyList': PropertyList(structured_names=use_structured_names),
#         'HintedEmptyNode': HintedEmptyNode(structured_names = use_structured_names),
#         'ListOfSimpleEntities': ListOfSimpleEntities(structured_names=use_structured_names),
#         'OrderedMixedContent': OrderedMixedContent(structured_names=use_structured_names),
#         'SpecialValues': SpecialValues(structured_names=use_structured_names),
#         'NamespacedObject': NamespacedObject(structured_names=use_structured_names),
#         'HeterogeneousList': HeterogeneousList(structured_names=use_structured_names),
#         'CDataField': CDataField(structured_names=use_structured_names),
#         # 'ProcessingInstructionNode': ProcessingInstructionNode(structured_names=use_structured_names),
#         'EntityReferenceField': EntityReferenceField(structured_names=use_structured_names),
#         'AttributePromotedKeyValueList': AttributePromotedKeyValueList(structured_names=use_structured_names),
#         'ListOfAttributedEntities': ListOfAttributedEntities(structured_names=use_structured_names),
#         'ListOfLists': ListOfLists(structured_names=use_structured_names),
#         'DeeplyNested': DeeplyNested(structured_names=use_structured_names),
#         'ListOfNamespacedEntities': ListOfNamespacedEntities(structured_names=use_structured_names),
#         'NestedAttributedEntity': NestedAttributedEntity(structured_names = use_structured_names),
#         'ListWithSpecialValues': ListWithSpecialValues(structured_names = use_structured_names),
#         'AttributedPropertyList': AttributedPropertyList(structured_names = use_structured_names),
#         'ListOfObjectsWithSpecialValues': ListOfObjectsWithSpecialValues(structured_names = use_structured_names),
#         'NamespacedAttributes': NamespacedAttributes(structured_names = use_structured_names),
#         'ListOfMixedContentObjects': ListOfMixedContentObjects(structured_names = use_structured_names)
#     }
    
#     # Create a list of base generators that are safe for nesting
#     nestable_base_generators = [
#         gen for name, gen in base_generators_map.items() 
#         if name not in ['EntityReferenceField', 'DeeplyNested'] # Avoid nesting these for simplicity
#     ]

#     # Instantiate the wrapper generators, feeding them the nestable base generators
#     wrapper_generators_map = {
#         'NestedObject': NestedObjectWrapper(nestable_base_generators, structured_names=use_structured_names),
#         'ListOfNestedObjects': ListOfNestedObjectsWrapper(nestable_base_generators, structured_names=use_structured_names)
#     }

#     # Combine all generators into a single dictionary for sampling
#     all_generators = {**base_generators_map, **wrapper_generators_map}

#     # Filter to only the generators specified for the current experiment
#     active_generators = {k: v for k, v in all_generators.items() if k in formats_to_use}
#     if not active_generators:
#         raise ValueError("No formats selected for generation. Check your experiment definition.")

#     counts = {
#         'easy': int(total_examples*distribution[0]),
#         'medium': int(total_examples*distribution[1]),
#         'hard': total_examples - int(total_examples*distribution[0]) - int(total_examples*distribution[1])
#     }
    
#     dataset=[]
#     order = ['easy','medium','hard'] if curriculum else random.sample(list(counts.keys()), len(counts))
#     active_format_names = list(active_generators.keys())

#     for diff in order:
#         for _ in range(counts[diff]):
#             fmt_name = random.choice(active_format_names)
#             dataset.append(active_generators[fmt_name].generate_example(difficulty=diff))
    
#     if not curriculum: random.shuffle(dataset)
#     return {'dataset': dataset}


# --- REWRITE the generate_dataset function ---

def generate_dataset(
    total_examples: int,
    formats_to_use: List[str], # We can simplify this now
    max_depth: int,
    distribution: Tuple[float, float, float] = (0.33,0.34,0.33),
    random_seed: Optional[int] = None,
    use_structured_names: bool = False,
    curriculum: bool = False
) -> Dict[str, List[Dict[str, Any]]]:

    if random_seed is not None:
        random.seed(random_seed)
        FAKER.seed_instance(random_seed)

    # 1. Instantiate ONLY the base format generators
    base_generators_map = {
        'SingleField': SingleField(structured_names=use_structured_names),
        'AttributedEntity': AttributedEntity(structured_names=use_structured_names),
        'BasicEntity': BasicEntity(structured_names=use_structured_names),
        'PropertyList': PropertyList(structured_names=use_structured_names),
        'HintedEmptyNode': HintedEmptyNode(structured_names = use_structured_names),
        'ListOfSimpleEntities': ListOfSimpleEntities(structured_names=use_structured_names),
        'OrderedMixedContent': OrderedMixedContent(structured_names=use_structured_names),
        'SpecialValues': SpecialValues(structured_names=use_structured_names),
        'NamespacedObject': NamespacedObject(structured_names=use_structured_names),
        'HeterogeneousList': HeterogeneousList(structured_names=use_structured_names),
        'CDataField': CDataField(structured_names=use_structured_names),
        # 'ProcessingInstructionNode': ProcessingInstructionNode(structured_names=use_structured_names),
        'EntityReferenceField': EntityReferenceField(structured_names=use_structured_names),
        'AttributePromotedKeyValueList': AttributePromotedKeyValueList(structured_names=use_structured_names),
        'ListOfAttributedEntities': ListOfAttributedEntities(structured_names=use_structured_names),
        'ListOfLists': ListOfLists(structured_names=use_structured_names),
        'DeeplyNested': DeeplyNested(structured_names=use_structured_names),
        'ListOfNamespacedEntities': ListOfNamespacedEntities(structured_names=use_structured_names),
        'NestedAttributedEntity': NestedAttributedEntity(structured_names = use_structured_names),
        'ListWithSpecialValues': ListWithSpecialValues(structured_names = use_structured_names),
        'AttributedPropertyList': AttributedPropertyList(structured_names = use_structured_names),
        'ListOfObjectsWithSpecialValues': ListOfObjectsWithSpecialValues(structured_names = use_structured_names),
        'NamespacedAttributes': NamespacedAttributes(structured_names = use_structured_names),
        'ListOfMixedContentObjects': ListOfMixedContentObjects(structured_names = use_structured_names)
    }
    # Filter out any special cases that can't be nested
    nestable_base_generators = [
        gen for name, gen in base_generators_map.items() 
        if name not in ['EntityReferenceField', 'DeeplyNested']
    ]

    # 2. Instantiate our single, powerful recursive generator
    combo_generator = CombinatorialGenerator(base_generators_map, use_structured_names)
    
    # 3. New generation loop based on nesting depth
    dataset = []
    examples_per_level = total_examples // max_depth

    print(f"Generating examples across {max_depth} nesting levels...")

    for depth in range(1, max_depth + 1):
        print(f"  - Generating {examples_per_level} examples for nesting level {depth}...")
        for _ in range(examples_per_level):
            # For depth 1, we just pick a base format directly
            if depth == 1:
                generator = random.choice(nestable_base_generators)
                example = generator.generate_example(difficulty='medium')
            # For depth > 1, we use our recursive generator
            else:
                example = combo_generator.generate_example(target_depth=depth, difficulty='medium')
            
            dataset.append(example)

    # --- THIS IS THE MISSING PART: Handle the remainder ---
    
    # 1. Calculate how many examples are still needed
    remaining_count = total_examples - len(dataset)
    
    if remaining_count > 0:
        print(f"  - Generating {remaining_count} remaining examples to meet total of {total_examples}...")
        
        # 2. Generate the remaining examples. It's common practice to add them
        #    to the most complex category to ensure enough hard examples exist.
        for _ in range(remaining_count):
            # We must use the combo_generator, as the target_depth will be at least 1.
            example = combo_generator.generate_example(target_depth=max_depth, difficulty='medium')
            dataset.append(example)

    print(f"\nTotal examples generated: {len(dataset)}")

    if not curriculum:
        random.shuffle(dataset)
        
    return {'dataset': dataset}

if __name__ == '__main__':
    print(f"--- Running Experiment: {EXPERIMENT_TO_RUN} ---")
    print(f"Description: {config['description']}")
    print(f"Training formats: {TRAINING_FORMATS}")
    # print(f"Held-out test format: [{TEST_FORMAT}]")
    print(f"Total examples: {N_EXAMPLES}")
    print(f"Curriculum learning: {CURRICULUM_LEARNING}")
    print(f"Using structured names: {FIXED_GRAMMAR_FOR_STRUCTURE}")
    print(f"Using random indentation: {INDENTATION_MODE}\n{'-'*50}")

    print(f"\n>>> Generating {N_EXAMPLES} training examples...")
    full_dataset = generate_dataset(
        total_examples=N_EXAMPLES,
        formats_to_use=ALL_FORMATS,
        distribution=DIFFICULTY_DISTRIBUTION,
        random_seed=RANDOM_SEED,
        use_structured_names=FIXED_GRAMMAR_FOR_STRUCTURE,
        curriculum=CURRICULUM_LEARNING,
        max_depth=MAX_NESTING_DEPTH
    )

    grammar_part = 'grammar' if FIXED_GRAMMAR_FOR_STRUCTURE else 'uuid'
    indent_part = INDENTATION_MODE + "_indent"
    filename = f'dataset_utils/nested_dataset_exp_{EXPERIMENT_TO_RUN}_{N_EXAMPLES}_{grammar_part}_{indent_part}_v2_emptylistdict.json'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=2)
    print(f"\nDataset successfully generated and saved to '{filename}'")


    print("---" * 25)
    print("--- VISUAL CONFIRMATION OF ALL FORMATS (2 examples each) ---")
    print("---" * 25)

    # Re-define the full list of generator classes here for the check
    # Note: Ensure all class definitions (SingleField, AttributedEntity, etc.) are complete in your script.
    all_generators_for_check = {
        'SingleField': SingleField(structured_names=False),
        'AttributedEntity': AttributedEntity(structured_names=True),
        'BasicEntity': BasicEntity(structured_names=True),
        'PropertyList': PropertyList(structured_names=True),
        'ListOfSimpleEntities': ListOfSimpleEntities(structured_names=True),
        'OrderedMixedContent': OrderedMixedContent(structured_names=True),
        'SpecialValues': SpecialValues(structured_names=True),
        'NamespacedObject': NamespacedObject(structured_names=False),
        'HeterogeneousList': HeterogeneousList(structured_names=True),
        'CDataField': CDataField(structured_names=True),
        # 'ProcessingInstructionNode': ProcessingInstructionNode(structured_names=True),
        'EntityReferenceField': EntityReferenceField(structured_names=True),
        'AttributePromotedKeyValueList': AttributePromotedKeyValueList(structured_names=True),
        'ListOfAttributedEntities': ListOfAttributedEntities(structured_names=True),
        'ListOfLists': ListOfLists(structured_names=True),
        'DeeplyNested': DeeplyNested(structured_names=True),
        'ListOfNamespacedEntities': ListOfNamespacedEntities(structured_names=False),
        'HintedEmptyNode': HintedEmptyNode(structured_names = False),
        'NestedAttributedEntity': NestedAttributedEntity(structured_names = False),
        'ListWithSpecialValues': ListWithSpecialValues(structured_names = False),
        'AttributedPropertyList': AttributedPropertyList(structured_names = False),
        'ListOfObjectsWithSpecialValues': ListOfObjectsWithSpecialValues(structured_names = False),
        'NamespacedAttributes': NamespacedAttributes(structured_names = False),
        'ListOfMixedContentObjects': ListOfMixedContentObjects(structured_names = False)

    }


    for name, generator in all_generators_for_check.items():
        print(f"\n{'='*20} Format: {name} {'='*20}")
        try:
            for i in range(2):
                print(f"--- Example {i+1} ---")
                example = generator.generate_example(difficulty='medium')
                
                print(">>> INPUT (XML):")
                print(example['question'])
                
                print("\n>>> OUTPUT (JSON):")
                # Parse and pretty-print the JSON string for readability
                parsed_json = json.loads(example['answer'])
                print(json.dumps(parsed_json, indent=2))
                print("-" * 50)
        except Exception as e:
            print(f"!!! ERROR generating example for {name}: {e}")

    print("\n--- VISUAL CONFIRMATION COMPLETE ---\n")