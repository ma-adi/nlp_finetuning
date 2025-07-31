import json
import random
import string
import uuid
from lxml import etree as ET
from faker import Faker
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration & Setup ---

FAKER = Faker()

# --- Global Settings ---
N_EXAMPLES = 3000
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
            'ListOfSimpleEntities','OrderedMixedContent', 
            'SpecialValues', 'NamespacedObject', 'HeterogeneousList', 'CDataField',
            'ProcessingInstructionNode', 'EntityReferenceField','HintedEmptyNode',
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
            'ListOfSimpleEntities','OrderedMixedContent', 
            'SpecialValues', 'NamespacedObject', 'HeterogeneousList', 'CDataField',
            'ProcessingInstructionNode', 'EntityReferenceField','HintedEmptyNode',
            'AttributePromotedKeyValueList', 'ListOfAttributedEntities'
        ],
        'ap_formats': ['NestedAttributedEntity', 'ListWithSpecialValues', 
                    'AttributedPropertyList', 'ListOfObjectsWithSpecialValues',
                    'NamespacedAttributes', 'ListOfMixedContentObjects']
    },
    'easy_master_curriculum_2': {
        'description': "More realistic compositional testing for master_curriculum -- same core formats",
        'core_formats': [
            'SingleField', 'AttributedEntity', 'BasicEntity', 'PropertyList',
            'ListOfSimpleEntities', 
            'OrderedMixedContent', 
            'SpecialValues', 'NamespacedObject', 'HeterogeneousList', 'CDataField',
            'ProcessingInstructionNode', 'EntityReferenceField','HintedEmptyNode',
            'AttributePromotedKeyValueList', 'ListOfAttributedEntities'
        ],
        'ap_formats': ['NestedAttributedEntity', 'ListWithSpecialValues', 
                    'AttributedPropertyList', 'ListOfObjectsWithSpecialValues',
                    'NamespacedAttributes', 'ListOfMixedContentObjects', 
                    'MixedPropertyGroup', 'AttributedObjectWithSpecialValues']
    },
    'easy_master_curriculum_3': {
        'description': "More realistic compositional testing for master_curriculum -- same core formats",
        'core_formats': [
            'SingleField', 'AttributedEntity', 'BasicEntity', 'PropertyList',
            'ListOfSimpleEntities', 
            'OrderedMixedContent', 
            'SpecialValues', 'NamespacedObject', 'HeterogeneousList', 'CDataField',
            'ProcessingInstructionNode', 'EntityReferenceField','HintedEmptyNode',
            'AttributePromotedKeyValueList', 'ListOfAttributedEntities', 
            'MixedPropertyGroup','AttributedPropertyList'
        ],
        'ap_formats': ['ListWithSpecialValues',
                    'NamespacedAttributes', 'ListOfMixedContentObjects', 
                    'AttributedObjectWithSpecialValues']
    },
'foundations_plus_composition': {
    'description': "A curriculum that teaches atomic rules first, then core compositional patterns, leaving novel compositions for the held-out test set.",
    'core_formats': [
        # --- Group A: Atomic Foundations ---
        'SingleField',                  # Simplest key-value
        'AttributedEntity',             # Basic attribute rule (@)
        'PropertyList',                 # Basic list of values
        'SpecialValues',                # Type casting (null, boolean)
        'HintedEmptyNode',              # Type casting from attributes
        'NamespacedObject',             # Basic namespace handling
        'OrderedMixedContent',          # The #content rule for mixed text/elements
        'CDataField',                   # CDATA rule
        'ProcessingInstructionNode',    # Processing Instruction rule
        'EntityReferenceField',         # Entity reference rule

        # --- Group B: Core Compositional Patterns ---
        'BasicEntity',                  # Teaches: Attributes + Child Elements
        'ListOfSimpleEntities',         # Teaches: List of Objects
        'ListOfAttributedEntities',     # Teaches: List of Attribute-only Objects
        'AttributedPropertyList',       # Teaches: Attributes + List of Values
        'MixedPropertyGroup',           # Teaches: Attributes + Mixed Single/List Children
        'AttributePromotedKeyValueList',# Teaches: Conditional attribute promotion rule
        'HeterogeneousList',            # Teaches: List of mixed strings and objects
        'NestedAttributedEntity'        # Teaches: Composition of nested objects and attributes
    ],
    'ap_formats': [
        # --- Held-out tests for generalization ---
        'ListWithSpecialValues',          # Tests: Can model apply type casting inside a list?
        'AttributedObjectWithSpecialValues', # Tests: Can model merge attributes and cast types on children in one object?
        'NamespacedAttributes',           # Tests: Can model handle namespaced attributes (not just elements)?
        'ListOfNamespacedEntities',       # Tests: Can model handle a list of namespaced objects?
        'ListOfMixedContentObjects',      # Tests: Can model handle lists containing complex #content objects? 
    ]
},
}

# --- Experiment Selection ---
EXPERIMENT_TO_RUN = 'easy_master_curriculum_3'
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
    'easy': {'list_length':(2,4), 'sub_list_length':(2,3), 'nesting_depth':2, 'optional_prob':0.1},
    'medium': {'list_length':(2,9), 'sub_list_length':(2,4), 'nesting_depth':3, 'optional_prob':0.5},
    'hard': {'list_length':(3,20), 'sub_list_length':(3,8), 'nesting_depth':5, 'optional_prob':0.9}
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

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        raise NotImplementedError

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
            'format_id': self.format_id,
            'tags': self.tags
        }

# --- Core Format Implementations (with changes) ---

class SingleField(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('SingleField', ['simple_kv', 'nested_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'field'])
        root = ET.Element(nm['root'])
        field_el = ET.SubElement(root, nm['field'])
        # ### CHANGE 2: Ensure value can be numeric-like but is always a string
        field_val = str(FAKER.random_int(1000, 99999)) if random.random() < 0.3 else FAKER.word()
        field_el.text = field_val
        return self._build_output(root, {nm['root']: {nm['field']: field_val}})

### LOSSLESS CHANGE ###
class AttributedEntity(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('AttributedEntity', ['attribute_to_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'attr1', 'attr2'])
        root = ET.Element(nm['root'], {nm['attr1']: FAKER.word(), nm['attr2']: str(FAKER.random_int(1, 100))})
        # Prefix all attribute keys with '@'
        json_data = {f"@{k}": v for k, v in root.attrib.items()}
        return self._build_output(root, {nm['root']: json_data})

### LOSSLESS CHANGE ###
class BasicEntity(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('BasicEntity', ['simple_kv', 'attribute_to_kv', 'merged_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'id_attr', 'field'])
        root = ET.Element(nm['root'], {nm['id_attr']: FAKER.uuid4()})
        f = ET.SubElement(root, nm['field']); f.text = FAKER.word()
        # Create a dictionary of prefixed attributes
        prefixed_attrs = {f"@{k}": v for k, v in root.attrib.items()}
        # Merge prefixed attributes with child element data
        json_data = {**prefixed_attrs, nm['field']: f.text}
        return self._build_output(root, {nm['root']: json_data})

class PropertyList(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('PropertyList', ['list_of_values', 'nested_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'item'])
        root = ET.Element(nm['root'])
        items = [FAKER.word() for _ in range(random.randint(*cfg['list_length']))]
        for it in items: ET.SubElement(root, nm['item']).text = it
        return self._build_output(root, {nm['root']: {nm['item']: items}})

class ListOfSimpleEntities(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ListOfSimpleEntities', ['list_of_objects', 'simple_kv', 'nested_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
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
        return self._build_output(root, {nm['root']: {nm['item']: json_list}})
        
### LOSSLESS CHANGE ###
class OrderedMixedContent(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('OrderedMixedContent', ['mixed_content', 'attribute_to_kv', 'simple_kv', 'nested_object'], **kwargs)
    def generate_example(self, difficulty: str = 'medium'):
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
        return self._build_output(root, {nm['root']: json_data_root})
    

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

    def generate_example(self, difficulty: str = 'medium'):
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

        return self._build_output(root, {nm['root']: json_data})

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

    def generate_example(self, difficulty: str = 'medium'):
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

        return self._build_output(root, {nm['root']: json_data})

### LOSSLESS CHANGE v2 (Includes xmlns) ###
class NamespacedObject(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('NamespacedObject', ['namespace_handling', 'attribute_to_kv', 'simple_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
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
        
        return self._build_output(root, {nm['root']: json_data})

class HeterogeneousList(FormatTemplate):
    # ... (This class is already consistent, no changes needed) ...
    def __init__(self, **kwargs): super().__init__('HeterogeneousList', ['list_of_objects', 'list_of_values', 'mixed_content'], **kwargs)
    def generate_example(self, difficulty='medium'):
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
        return self._build_output(root, {nm['root']: {nm['event']: json_list}})

class CDataField(FormatTemplate):
    # ... (This class is already consistent, no changes needed) ...
    def __init__(self, **kwargs): super().__init__('CDataField', ['cdata_section', 'simple_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'script'])
        cdata_content = f'<p>Hello, {FAKER.name()}!</p><script>alert("XSS & fun");</script>'
        root = ET.Element(nm['root'])
        script_el = ET.SubElement(root, nm['script'])
        script_el.text = ET.CDATA(cdata_content)
        json_data = {nm['script']: cdata_content}
        return self._build_output(root, {nm['root']: json_data})

### GENERALIZATION CHANGE ###
class ProcessingInstructionNode(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('ProcessingInstructionNode', ['processing_instruction'], **kwargs)
        # A pool of realistic but varied PI targets
        self.pi_targets = [
            'xml-stylesheet', 'php', 'cocoon-process', 'robot-control', 'my-custom-parser'
        ]

    def generate_example(self, difficulty: str = 'medium'):
        nm = self._generate_name_map(['root', 'field'])

        # 1. Randomly select a target from the pool
        pi_target = random.choice(self.pi_targets)

        # 2. Generate varied data for the instruction
        if random.random() > 0.5:
            # Case A: Structured key-value data
            key1, key2 = FAKER.word(), FAKER.word()
            val1 = FAKER.file_name()
            val2 = FAKER.word()
            pi_data = f'{key1}="{val1}" {key2}="{val2}"'
        else:
            # Case B: Unstructured, simple string data
            pi_data = FAKER.sentence(nb_words=4)

        pi = ET.ProcessingInstruction(pi_target, pi_data)
        prolog = ET.tostring(pi, encoding='unicode')

        root = ET.Element(nm['root'])
        ET.SubElement(root, nm['field']).text = 'content'

        json_data = {
            nm['field']: 'content',
            '_processing_instructions': [{pi_target: pi_data}]
        }
        
        return self._build_output(root, {nm['root']: json_data}, prolog=prolog)

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
    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
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
        return self._build_output(root, final_json)

class ListOfAttributedEntities(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ListOfAttributedEntities', tags=['list_of_objects', 'attribute_to_kv'], **kwargs)
    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
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
        return self._build_output(root, {nm['root']: {nm['item']: json_list}})


# --- AP / Held-Out Formats (already consistent) ---

### LOSSLESS CHANGE ###
class ListOfLists(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ListOfLists', ['list_of_objects','list_of_values','nested_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
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
        return self._build_output(root, {nm['root']: {nm['day']: out}})

class DeeplyNested(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('DeeplyNested', ['nested_object', 'simple_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        depth = DIFFICULTY_CONFIG[difficulty]['nesting_depth']
        keys = [f'level{i+1}' for i in range(depth)] + ['field']
        nm = self._generate_name_map(keys)
        current_el = root = ET.Element(nm['level1'])
        for i in range(1, depth): current_el = ET.SubElement(current_el, nm[f'level{i+1}'])
        val = FAKER.word(); f = ET.SubElement(current_el, nm['field']); f.text = val
        current_dict = {nm['field']: val}
        for i in range(depth - 1, -1, -1): current_dict = {nm[f'level{i+1}']: current_dict}
        return self._build_output(root, current_dict)
    
### LOSSLESS CHANGE v2 (Includes xmlns) ###
class ListOfNamespacedEntities(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ListOfNamespacedEntities', ['list_of_objects', 'namespace_handling', 'attribute_to_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
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
        
        return self._build_output(root, {nm['root']: root_json_obj})

###INTERMEDIATE FORMATS
### REVISED - AttributedPropertyList ###
class AttributedPropertyList(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('AttributedPropertyList', ['attribute_to_kv', 'list_of_values', 'merged_object'], **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
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
        
        return self._build_output(root, {nm['root']: json_data})

### REVISED - ListWithSpecialValues ###
class ListWithSpecialValues(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('ListWithSpecialValues', ['list_of_values', 'type_casting'], **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
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

        return self._build_output(root, {nm['root']: {nm['value']: json_list}})
    
### REVISED - NestedAttributedEntity ###
class NestedAttributedEntity(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('NestedAttributedEntity', ['nested_object', 'attribute_to_kv', 'merged_object'], **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
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
        
        return self._build_output(root, {nm['root']: json_data})

### REVISED - ListOfObjectsWithSpecialValues ###
class ListOfObjectsWithSpecialValues(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('ListOfObjectsWithSpecialValues', ['list_of_objects', 'type_casting', 'empty_element_to_null'], **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
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
        
        return self._build_output(root, {nm['root']: {nm['item']: json_list}})
    
### REVISED - NamespacedAttributes ###
class NamespacedAttributes(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('NamespacedAttributes', ['attribute_to_kv', 'namespace_handling'], **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
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

        return self._build_output(root, {nm['root']: json_data})

### REVISED - ListOfMixedContentObjects ###
class ListOfMixedContentObjects(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('ListOfMixedContentObjects', ['list_of_objects', 'mixed_content'], **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
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
        
        return self._build_output(root, {nm['root']: {nm['item']: json_list}})
    

class MixedPropertyGroup(FormatTemplate):
    """
    Tests a critical composite case: an element with attributes that contains
    a mix of a list of values (from repeated tags) and single key-value
    pairs (from unique tags). This is a combination of BasicEntity,
    PropertyList, and SingleField rules.
    """
    def __init__(self, **kwargs):
        super().__init__('MixedPropertyGroup', ['merged_object', 'list_of_values', 'simple_kv', 'attribute_to_kv'], **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        # Use more descriptive names for clarity
        nm = self._generate_name_map(['root', 'root_id_attr', 'list_item_tag', 'single_item_tag1', 'single_item_tag2'])

        # 1. Create root with an attribute
        root_attrs = {nm['root_id_attr']: FAKER.uuid4()}
        root = ET.Element(nm['root'], root_attrs)

        # 2. Construct the JSON data first (it's easier to reason about)
        json_data = {f"@{k}": v for k, v in root_attrs.items()}

        # 3. Add the list of values
        list_values = [FAKER.word() for _ in range(random.randint(2, 4))] # Always a list
        json_data[nm['list_item_tag']] = list_values
        for val in list_values:
            ET.SubElement(root, nm['list_item_tag']).text = val

        # 4. Add the single key-value pair
        single_val1 = str(FAKER.random_int(100, 200))
        json_data[nm['single_item_tag1']] = single_val1
        ET.SubElement(root, nm['single_item_tag1']).text = single_val1

        # 5. (Optional, for harder difficulties) Add a second single k-v pair
        if difficulty in ['medium', 'hard'] and random.random() < 0.5:
            single_val2 = FAKER.color_name()
            json_data[nm['single_item_tag2']] = single_val2
            ET.SubElement(root, nm['single_item_tag2']).text = single_val2

        return self._build_output(root, {nm['root']: json_data})
    

### NEW AP FORMAT - Attributed Object with Special Values ###
class AttributedObjectWithSpecialValues(FormatTemplate):
    """
    Tests the composition of attribute merging and special value casting on a
    single, non-list object.
    """
    def __init__(self, **kwargs):
        super().__init__('AttributedObjectWithSpecialValues', ['merged_object', 'type_casting', 'attribute_to_kv'], **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        nm = self._generate_name_map(['root', 'id_attr', 'enabled_flag', 'notes_field'])
        root = ET.Element(nm['root'], {nm['id_attr']: FAKER.uuid4()})

        # Add a boolean field
        ET.SubElement(root, nm['enabled_flag']).text = 'true'
        # Add a null field (empty element)
        ET.SubElement(root, nm['notes_field'])

        # Construct the JSON, merging attributes and casting values
        json_data = {
            f"@{nm['id_attr']}": root.get(nm['id_attr']),
            nm['enabled_flag']: True,
            nm['notes_field']: None
        }
        return self._build_output(root, {nm['root']: json_data})



# --- Orchestration ---
def generate_dataset(
    total_examples: int,
    formats_to_use: List[str],
    distribution: Tuple[float, float, float] = (0.33,0.34,0.33),
    random_seed: Optional[int] = None,
    use_structured_names: bool = False,
    curriculum: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    if random_seed is not None:
        random.seed(random_seed)
        FAKER.seed_instance(random_seed)



    all_generators = {
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
        'ProcessingInstructionNode': ProcessingInstructionNode(structured_names=use_structured_names),
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
        'ListOfMixedContentObjects': ListOfMixedContentObjects(structured_names = use_structured_names),
        'MixedPropertyGroup': MixedPropertyGroup(structured_names = use_structured_names),
        'AttributedObjectWithSpecialValues': AttributedObjectWithSpecialValues(structured_names = use_structured_names)
    }

    active_generators = {k: v for k, v in all_generators.items() if k in formats_to_use}
    if not active_generators:
        raise ValueError("No formats selected for generation. Check your experiment definition.")

    counts = {
        'easy': int(total_examples*distribution[0]),
        'medium': int(total_examples*distribution[1]),
        'hard': total_examples - int(total_examples*distribution[0]) - int(total_examples*distribution[1])
    }
    
    dataset=[]
    order = ['easy','medium','hard'] if curriculum else random.sample(list(counts.keys()), len(counts))
    active_format_names = list(active_generators.keys())

    for diff in order:
        for _ in range(counts[diff]):
            fmt_name = random.choice(active_format_names)
            dataset.append(active_generators[fmt_name].generate_example(difficulty=diff))
    
    if not curriculum: random.shuffle(dataset)
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
        curriculum=CURRICULUM_LEARNING
    )

    grammar_part = 'grammar' if FIXED_GRAMMAR_FOR_STRUCTURE else 'uuid'
    indent_part = INDENTATION_MODE + "_indent"
    filename = f'dataset_utils/tagged_dataset_exp_{EXPERIMENT_TO_RUN}_{N_EXAMPLES}_{grammar_part}_{indent_part}_v2_emptylistdict.json'
    
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
        'ProcessingInstructionNode': ProcessingInstructionNode(structured_names=True),
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
        'ListOfMixedContentObjects': ListOfMixedContentObjects(structured_names = False),
        'MixedPropertyGroup': MixedPropertyGroup(structured_names = False),
        'AttributedObjectWithSpecialValues': AttributedObjectWithSpecialValues(structured_names = False)


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