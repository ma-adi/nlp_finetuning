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
    'empty_element_to_null', 'self_closing_to_bool', 'namespace_handling',
    'cdata_section', 'processing_instruction'
]

# --- Experiment Definitions ---
EXPERIMENTS = {
    'master_curriculum': {
        'description': "A comprehensive curriculum covering 12 foundational XML rules.",
        'core_formats': [
            'SingleField', 'AttributedEntity', 'BasicEntity', 'PropertyList',
            'ListOfSimpleEntities', 'AdvancedMixedContent', 'SpecialValues',
            'NamespacedObject', 'HeterogeneousList', 'CDataField',
            'ProcessingInstructionNode', 'EntityReferenceField',
            ### New additions
            'AttributePromotedKeyValueList','ListOfAttributedEntities'
            ### New additions
        ],
        'ap_formats': [ # Advanced Placement formats to be used as held-out tests
            'ListOfLists', 'DeeplyNested', 'ListOfNamespacedEntities'
        ]
    }
}

# --- Experiment Selection ---
EXPERIMENT_TO_RUN = 'master_curriculum'
HELD_OUT_TEST_FORMAT = 'ListOfLists'

# --- Dynamic Configuration based on Selection ---
config = EXPERIMENTS[EXPERIMENT_TO_RUN]
TRAINING_FORMATS = config['core_formats']
ALL_FORMATS = config['core_formats'] + config['ap_formats']
TEST_FORMAT = HELD_OUT_TEST_FORMAT
if TEST_FORMAT not in config['ap_formats']:
    raise ValueError(f"Test format '{TEST_FORMAT}' is not a valid AP format for this experiment.")

# --- Difficulty Configuration ---
DIFFICULTY_CONFIG = {
    'easy': {'list_length':(0,4), 'sub_list_length':(0,2), 'nesting_depth':2, 'optional_prob':0.1},
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
        # --- FIX: Prepend a letter to guarantee a valid XML tag name.
        # prefix = random.choice(string.ascii_lowercase)
        # return {k: f"{prefix}{str(uuid.uuid4())[:8]}" for k in keys}
        # --- MODIFICATION ---
        # Generate more realistic, complex names with hyphens and dots, not just UUIDs.
        # This helps the model generalize to real-world names like 'datasource-dependencies'.
        generated_names = set()
        while len(generated_names) < len(keys):
            # A valid XML name must start with a letter or underscore.
            prefix = random.choice(string.ascii_lowercase + '_')
            # The rest can contain letters, digits, hyphens, dots, and underscores.
            allowed_chars = string.ascii_lowercase + string.digits + '-._'
            # Generate a name with 1 to 3 parts for complexity
            num_parts = random.randint(1, 3)
            parts = [''.join(random.choices(allowed_chars, k=random.randint(3, 6))) for _ in range(num_parts)]
            name = prefix + random.choice(['-', '.', '_']).join(parts)
            generated_names.add(name[:20]) # Cap length to avoid excessively long names

        return {k: v for k, v in zip(keys, list(generated_names))}

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        raise NotImplementedError

# In the FormatTemplate class:

    def _build_output(self, xml_root: ET.Element, json_dict: Dict, prolog: Optional[str] = None) -> Dict[str, Any]:
        xml_string = ""

        if INDENTATION_MODE == 'pretty':
            # OPTION 1: Clean, uniform indentation
            try:
                ET.indent(xml_root, space="  ")
            except AttributeError:
                pass # For older Python versions
            xml_string = ET.tostring(xml_root, encoding='unicode')
            if prolog:
                xml_string = prolog + '\n' + xml_string

        elif INDENTATION_MODE == 'random':
            # OPTION 2: Chaotic, randomized indentation
            def build_chaotic_xml(element, level=0):
                # Random spaces for hierarchical indent
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
                    # Random spaces for closing tag indent
                    parts.append('\n' + ' ' * (level * random.randint(0, 4)))
                parts.append(f"</{element.tag}>")
                # Add random spaces after every newline for extra chaos
                return ''.join(parts).replace('\n', '\n' + ' ' * random.randint(0, 5))
            
            xml_string = build_chaotic_xml(xml_root).strip()
            if prolog:
                xml_string = prolog + '\n' + xml_string

        elif INDENTATION_MODE == 'none':
            # OPTION 3: No indentation, all on one line
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

# --- The 12 Foundational "Core Curriculum" Formats (implementations unchanged) ---
# ... (All FormatTemplate subclasses from the previous answer are correct and go here) ...
# For brevity, I am omitting them here, but they should be pasted back in.
class SingleField(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('SingleField', ['simple_kv', 'nested_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'field'])
        root = ET.Element(nm['root']); f = ET.SubElement(root, nm['field']); f.text = FAKER.word()
        return self._build_output(root, {nm['root']: {nm['field']: f.text}})

class AttributedEntity(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('AttributedEntity', ['attribute_to_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'attr1', 'attr2'])
        root = ET.Element(nm['root'], {nm['attr1']: FAKER.word(), nm['attr2']: str(FAKER.random_int(1, 100))})
        return self._build_output(root, {nm['root']: dict(root.attrib)})

class BasicEntity(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('BasicEntity', ['simple_kv', 'attribute_to_kv', 'merged_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'id_attr', 'field'])
        root = ET.Element(nm['root'], {nm['id_attr']: FAKER.uuid4()})
        f = ET.SubElement(root, nm['field']); f.text = FAKER.word()
        json_data = {**root.attrib, nm['field']: f.text}
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
            f1_val, f2_val = FAKER.word(), FAKER.word()
            ET.SubElement(item_el, nm['field1']).text = f1_val
            ET.SubElement(item_el, nm['field2']).text = f2_val
            json_list.append({nm['field1']: f1_val, nm['field2']: f2_val})
        return self._build_output(root, {nm['root']: {nm['item']: json_list}})

class SimplifiedListOfComplexEntitiesFormat(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('SimplifiedListOfComplexEntities', ['list_of_objects', 'nested_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        min_len, max_len = DIFFICULTY_CONFIG[difficulty]['list_length']
        nm = self._generate_name_map(['root', 'item', 'name_field'])
        root = ET.Element(nm['root']); lst=[]
        for _ in range(random.randint(min_len, max_len)):
            it = ET.SubElement(root, nm['item'])
            txt = FAKER.name()
            sf = ET.SubElement(it, nm['name_field']); sf.text = txt
            lst.append({nm['name_field']: txt})
        return self._build_output(root, {nm['root']: {nm['item']: lst}})


class AdvancedMixedContent(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('AdvancedMixedContent', ['mixed_content', 'attribute_to_kv', 'simple_kv', 'namespace_handling'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'child', 'attr', '_value'])
        root = ET.Element(nm['root'], {nm['attr']: FAKER.uuid4()})
        root.text = FAKER.sentence(nb_words=3) + " "
        json_data = {nm['attr']: root.attrib[nm['attr']], nm['_value']: [root.text.strip()]}
        
        child1_text = FAKER.word()
        child1 = ET.SubElement(root, nm['child'])
        child1.text = child1_text
        child1.tail = " " + FAKER.sentence(nb_words=2) + " "
        json_data[nm['child']] = child1_text
        json_data[nm['_value']].append(child1.tail.strip())

        if difficulty in ['medium', 'hard']:
            child2_text = str(FAKER.random_int(100, 500))
            child2 = ET.SubElement(root, nm['child'] + '2')
            child2.text = child2_text
            child2.tail = " " + FAKER.sentence(nb_words=4)
            json_data[nm['child'] + '2'] = int(child2_text)
            json_data[nm['_value']].append(child2.tail.strip())

        return self._build_output(root, {nm['root']: json_data})

class SpecialValues(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('SpecialValues', ['simple_kv', 'type_casting', 'empty_element_to_null', 'self_closing_to_bool'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'age', 'is_admin', 'notes'])
        root = ET.Element(nm['root'])
        age_val = FAKER.random_int(18, 99)
        ET.SubElement(root, nm['age']).text = str(age_val)
        ET.SubElement(root, nm['is_admin']) 
        ET.SubElement(root, nm['notes']) 
        json_data = {nm['age']: age_val, nm['is_admin']: True, nm['notes']: None}
        return self._build_output(root, {nm['root']: json_data})

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
        json_data = {nm['root']: {json_key: child_el.text}}
        return self._build_output(root, json_data)

class HeterogeneousList(FormatTemplate):
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
    def __init__(self, **kwargs): super().__init__('CDataField', ['cdata_section', 'simple_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'script'])
        cdata_content = f'<p>Hello, {FAKER.name()}!</p><script>alert("XSS & fun");</script>'
        root = ET.Element(nm['root'])
        script_el = ET.SubElement(root, nm['script'])
        script_el.text = ET.CDATA(cdata_content)
        json_data = {nm['script']: cdata_content}
        return self._build_output(root, {nm['root']: json_data})

class ProcessingInstructionNode(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ProcessingInstructionNode', ['processing_instruction'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'field'])
        pi_target = 'xml-stylesheet'
        pi_data = f'href="{FAKER.file_name(extension="css")}" type="text/css"'
        pi = ET.ProcessingInstruction(pi_target, pi_data)
        prolog = ET.tostring(pi, encoding='unicode')

        root = ET.Element(nm['root'])
        ET.SubElement(root, nm['field']).text = 'content'
        json_data = {nm['field']: 'content', '_processing_instructions': [{pi_target: pi_data}]}
        return self._build_output(root, {nm['root']: json_data}, prolog=prolog)

class EntityReferenceField(FormatTemplate):
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
            out.append({nm['name_attr']: day_name, nm['task']: subs})
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
            json_list.append({nm['id_attr']: item_id, f'{nm["prefix"]}:{nm["name"]}': item_name})
        
        json_key = f'{nm["prefix"]}:{nm["item"]}'
        return self._build_output(root, {nm['root']: {json_key: json_list}})


class AttributePromotedKeyValueList(FormatTemplate):
    """
    Teaches the model to handle a list of elements where attributes of each element
    are promoted to become key-value pairs in the parent's JSON object.
    Example: <style><prop name="color" value="blue"/></style> -> {"style": {"color": "blue"}}
    This directly addresses the pattern seen in <zone-style> with <format> tags.
    """
    def __init__(self, **kwargs):
        super().__init__(
            'AttributePromotedKeyValueList',
            tags=['attribute_to_kv', 'merged_object'],
            **kwargs
        )

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        # Use more semantic names if structured_names is True for clarity
        keys_to_map = ['root', 'item', 'key_attr', 'val_attr']
        if self.structured_names:
            nm = {'root': 'style', 'item': 'property', 'key_attr': 'name', 'val_attr': 'value'}
        else:
            nm = self._generate_name_map(keys_to_map)

        root = ET.Element(nm['root'])
        json_payload = {}

        # Use a set to ensure we don't generate duplicate keys
        generated_keys = set()
        for _ in range(random.randint(*cfg['list_length'])):
            # Ensure the generated key is unique for this example
            key = FAKER.word().replace(' ', '_')
            while key in generated_keys:
                key = FAKER.word().replace(' ', '_')
            generated_keys.add(key)

            value = FAKER.word() if random.random() > 0.3 else str(FAKER.random_int(1, 500))
            ET.SubElement(root, nm['item'], {nm['key_attr']: key, nm['val_attr']: value})
            # The JSON value might be an int, so we handle that
            try:
                json_payload[key] = int(value)
            except ValueError:
                json_payload[key] = value

        return self._build_output(root, {nm['root']: json_payload})
    

class ListOfAttributedEntities(FormatTemplate):
    """
    Teaches the model to handle a list of sibling elements that contain only attributes.
    Example: <cols><col id="a"/><col id="b"/></cols> -> {"cols": {"col": [{"id":"a"}, {"id":"b"}]}}
    This addresses the pattern seen with multiple <column ... /> or <datasource ... /> tags.
    """
    def __init__(self, **kwargs):
        super().__init__(
            'ListOfAttributedEntities',
            tags=['list_of_objects', 'attribute_to_kv'],
            **kwargs
        )

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        cfg = DIFFICULTY_CONFIG[difficulty]
        keys_to_map = ['root', 'item', 'attr1', 'attr2']
        if self.structured_names:
            nm = {'root': 'dependencies', 'item': 'column', 'attr1': 'name', 'attr2': 'type'}
        else:
            nm = self._generate_name_map(keys_to_map)

        root = ET.Element(nm['root'])
        json_list = []

        for _ in range(random.randint(*cfg['list_length'])):
            attrs = {
                nm['attr1']: FAKER.word(),
                nm['attr2']: FAKER.word()
            }
            # Add an optional third attribute for medium/hard difficulties
            if difficulty in ['medium', 'hard'] and random.random() > 0.5:
                attrs['optional_attr'] = str(FAKER.random_int(0,1))

            ET.SubElement(root, nm['item'], attrs)
            json_list.append(attrs)

        return self._build_output(root, {nm['root']: {nm['item']: json_list}})


# --- Orchestration (unchanged) ---
# ... (The generate_dataset function and __main__ block are correct and go here) ...
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
        'ListOfSimpleEntities': ListOfSimpleEntities(structured_names=use_structured_names),
        'AdvancedMixedContent': AdvancedMixedContent(structured_names=use_structured_names),
        'SpecialValues': SpecialValues(structured_names=use_structured_names),
        'NamespacedObject': NamespacedObject(structured_names=use_structured_names),
        'HeterogeneousList': HeterogeneousList(structured_names=use_structured_names),
        'CDataField': CDataField(structured_names=use_structured_names),
        'ProcessingInstructionNode': ProcessingInstructionNode(structured_names=use_structured_names),
        'EntityReferenceField': EntityReferenceField(structured_names=use_structured_names),
        # --- ADD THESE NEW GENERATORS ---
        'AttributePromotedKeyValueList': AttributePromotedKeyValueList(structured_names=use_structured_names),
        'ListOfAttributedEntities': ListOfAttributedEntities(structured_names=use_structured_names),
        # --- (The rest of the generators remain) ---
        'ListOfLists': ListOfLists(structured_names=use_structured_names),
        'DeeplyNested': DeeplyNested(structured_names=use_structured_names),
        'ListOfNamespacedEntities': ListOfNamespacedEntities(structured_names=use_structured_names),
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
    print(f"Held-out test format: [{TEST_FORMAT}]")
    print(f"Total examples: {N_EXAMPLES}")
    print(f"Curriculum learning: {CURRICULUM_LEARNING}")
    print(f"Using structured names: {FIXED_GRAMMAR_FOR_STRUCTURE}")
    print(f"Using random indentation: {INDENTATION_MODE}\n{'-'*50}")
    
    # n_train = int(N_EXAMPLES * 0.8)
    # n_test = N_EXAMPLES - n_train

    print(f"\n>>> Generating {N_EXAMPLES} training examples...")
    full_dataset = generate_dataset(
        total_examples=N_EXAMPLES,
        formats_to_use=ALL_FORMATS,
        distribution=DIFFICULTY_DISTRIBUTION,
        random_seed=RANDOM_SEED,
        use_structured_names=FIXED_GRAMMAR_FOR_STRUCTURE,
        curriculum=CURRICULUM_LEARNING
    )

    # print(f"\n>>> Generating {n_test} test examples...")
    # test_dataset = generate_dataset(
    #     total_examples=n_test,
    #     formats_to_use=[TEST_FORMAT],
    #     distribution=DIFFICULTY_DISTRIBUTION,
    #     random_seed=RANDOM_SEED + 1,
    #     use_structured_names=FIXED_GRAMMAR_FOR_STRUCTURE,
    #     curriculum=False
    # )

    # full_dataset = {'dataset': train_dataset['dataset'] + test_dataset['dataset']}
    # random.shuffle(full_dataset['dataset'])

    grammar_part = 'grammar' if FIXED_GRAMMAR_FOR_STRUCTURE else 'uuid'
    indent_part = INDENTATION_MODE + "_indent"
    filename = f'dataset_utils/tagged_dataset_exp_{EXPERIMENT_TO_RUN}_{N_EXAMPLES}_{grammar_part}_{indent_part}_betternaming_newformats.json'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=2)
    print(f"\nDataset successfully generated and saved to '{filename}'")

    # print("\n--- Sample from Training Set ---")
    # sample_train = random.choice(full_dataset['dataset'])
    # print(json.dumps({'format_id': sample_train['format_id'], 'question': sample_train['question']}, indent=2))

    # print("\n--- Sample from Test Set ---")
    # sample_test = random.choice(test_dataset['dataset'])
    # print(json.dumps({'format_id': sample_test['format_id'], 'question': sample_test['question']}, indent=2))