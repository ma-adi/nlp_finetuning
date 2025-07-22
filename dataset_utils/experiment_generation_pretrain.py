import json
import random
import uuid
import xml.etree.ElementTree as ET
from faker import Faker
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration & Setup ---

FAKER = Faker()

# --- Global Settings ---
N_EXAMPLES = 3000
DIFFICULTY_DISTRIBUTION = (0.3, 0.4, 0.3)
RANDOM_SEED = 42
RANDOM_INDENT = False
FIXED_GRAMMAR_FOR_STRUCTURE = False
CURRICULUM_LEARNING = True

# --- Unified Tag Vocabulary ---
# The "DNA" of our XML structures.
TAGS_VOCABULARY = [
    'simple_kv', 'attribute_to_kv', 'merged_object', 'list_of_values',
    'list_of_objects', 'nested_object', 'mixed_content', 'type_casting',
    'empty_element_to_null', 'self_closing_to_bool', 'namespace_handling',
    'cdata_section', 'processing_instruction'
]

# --- Experiment Definitions ---
# This new structure defines the curriculum and the test.
EXPERIMENTS = {
    'master_curriculum': {
        'description': "A comprehensive curriculum covering 12 foundational XML rules.",
        'core_formats': [
            'SingleField', 'AttributedEntity', 'BasicEntity', 'PropertyList',
            'ListOfSimpleEntities', 'MixedContentNode', 'SpecialValues',
            'NamespacedObject', 'HeterogeneousList', 'CDataField',
            'ProcessingInstructionNode', 'EntityReferenceField'
        ],
        'ap_formats': [ # Advanced Placement formats to be used as held-out tests
            'ListOfLists', 'DeeplyNested', 'ListOfNamespacedEntities', 'ComplexMixedContent'
        ]
    },
    'nested_lists_with_dict': {
        'description': "Extends nested_lists_easy by adding pure multi-pair dictionary format.",
        'core_formats': ['PropertyList', 'SimplifiedListOfComplexEntities', 'ObjectWithList', 'Dictionary'],
        'ap_formats': 'ListOfLists'
    },
    'shape_combination': {
        'description': "Tests if the model can combine a simple KV field and a list into one object.",
        'core_formats': ['SingleField', 'SimpleList'],
        'ap_formats': 'MixedFields'
    },
    'comprehensive_mix': {
        'description': "A comprehensive mix of formats testing various XML structures, including attributes, type casting, and nested objects. Ported from the legacy generator.",
        'core_formats': ['BasicEntity', 'EmptyElement', 'MixedContent', 'PropertyList', 'DeeplyNested', 'NestedConfig', 'SimpleList'],
        'ap_formats': 'ListOfComplexEntities' # A great held-out format combining many features.
    }
}

# --- Experiment Selection ---
EXPERIMENT_TO_RUN = 'master_curriculum'
# CHOOSE YOUR HELD-OUT TEST FROM THE AP FORMATS:
HELD_OUT_TEST_FORMAT = 'ListOfLists'

# --- Dynamic Configuration based on Selection ---
config = EXPERIMENTS[EXPERIMENT_TO_RUN]
TRAINING_FORMATS = config['core_formats']
TEST_FORMAT = HELD_OUT_TEST_FORMAT
if TEST_FORMAT not in config['ap_formats']:
    raise ValueError(f"Test format '{TEST_FORMAT}' is not a valid AP format for this experiment.")

ALL_FOCUSED_FORMATS = TRAINING_FORMATS + [TEST_FORMAT]

# --- Difficulty Configuration (Unchanged) ---
DIFFICULTY_CONFIG = {
    'easy': {'list_length':(2,4), 'sub_list_length':(1,2), 'nesting_depth':2, 'optional_prob':0.1},
    'medium': {'list_length':(5,9), 'sub_list_length':(2,4), 'nesting_depth':3, 'optional_prob':0.5},
    'hard': {'list_length':(10,20), 'sub_list_length':(4,8), 'nesting_depth':5, 'optional_prob':0.9}
}

# --- Base Class (with improved name generator) ---
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
        return {k: str(uuid.uuid4())[:8] for k in keys}

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        raise NotImplementedError

    def _build_output(self, xml_string: str, json_dict: Dict) -> Dict[str, Any]:
        # Now takes a string to support complex XML like CDATA, PIs, etc.
        return {
            'question': xml_string,
            'answer': json.dumps(json_dict),
            'format_id': self.format_id,
            'tags': self.tags
        }

# --- The 12 Foundational "Core Curriculum" Formats ---

class SingleField(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('SingleField', ['simple_kv', 'nested_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'field'])
        root = ET.Element(nm['root']); f = ET.SubElement(root, nm['field']); f.text = FAKER.word()
        return self._build_output(ET.tostring(root, 'unicode'), {nm['root']: {nm['field']: f.text}})

class AttributedEntity(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('AttributedEntity', ['attribute_to_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'attr1', 'attr2'])
        root = ET.Element(nm['root'], {nm['attr1']: FAKER.word(), nm['attr2']: str(FAKER.random_int(1, 100))})
        return self._build_output(ET.tostring(root, 'unicode'), {nm['root']: dict(root.attrib)})

class BasicEntity(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('BasicEntity', ['simple_kv', 'attribute_to_kv', 'merged_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'id_attr', 'field'])
        root = ET.Element(nm['root'], {nm['id_attr']: FAKER.uuid4()})
        f = ET.SubElement(root, nm['field']); f.text = FAKER.word()
        json_data = {**root.attrib, nm['field']: f.text}
        return self._build_output(ET.tostring(root, 'unicode'), {nm['root']: json_data})

class PropertyList(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('PropertyList', ['list_of_values', 'nested_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'item'])
        root = ET.Element(nm['root'])
        items = [FAKER.word() for _ in range(random.randint(*cfg['list_length']))]
        for it in items: ET.SubElement(root, nm['item']).text = it
        return self._build_output(ET.tostring(root, 'unicode'), {nm['root']: {nm['item']: items}})

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
        return self._build_output(ET.tostring(root, 'unicode'), {nm['root']: {nm['item']: json_list}})

class MixedContentNode(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('MixedContentNode', ['mixed_content', 'attribute_to_kv', 'simple_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'id_attr', 'child_tag', '_value'])
        root = ET.Element(nm['root'], {nm['id_attr']: FAKER.uuid4()})
        root.text = FAKER.sentence(nb_words=4)
        child = ET.SubElement(root, nm['child_tag']); child.text = FAKER.word()
        root.tail = FAKER.sentence(nb_words=3) # Text after the child element
        json_data = {
            nm['id_attr']: root.attrib[nm['id_attr']],
            nm['_value']: root.text.strip(), # Using _value convention for mixed text
            nm['child_tag']: child.text
        }
        # Note: ElementTree doesn't easily support text after a child, so we simulate it.
        # A more robust XML library might be needed for perfect mixed content generation.
        # For now, we focus on text *before* children.
        xml_string = f'<{nm["root"]} {nm["id_attr"]}="{root.attrib[nm["id_attr"]]}">{root.text}<{nm["child_tag"]}>{child.text}</{nm["child_tag"]}></{nm["root"]}>'
        return self._build_output(xml_string, {nm['root']: json_data})

class SpecialValues(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('SpecialValues', ['simple_kv', 'type_casting', 'empty_element_to_null', 'self_closing_to_bool'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'age', 'is_admin', 'notes'])
        root = ET.Element(nm['root'])
        age_val = FAKER.random_int(18, 99)
        ET.SubElement(root, nm['age']).text = str(age_val)
        ET.SubElement(root, nm['is_admin']) # Self-closing for True
        ET.SubElement(root, nm['notes']).text = '' # Empty for Null
        json_data = {nm['age']: age_val, nm['is_admin']: True, nm['notes']: None}
        # Manually create self-closing tag for clarity
        xml_string = f'<{nm["root"]}><{nm["age"]}>{age_val}</{nm["age"]}><{nm["is_admin"]}/><{nm["notes"]}></{nm["notes"]}></{nm["root"]}>'
        return self._build_output(xml_string, {nm['root']: json_data})

class NamespacedObject(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('NamespacedObject', ['namespace_handling', 'attribute_to_kv', 'simple_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'prefix1', 'prefix2', 'child1', 'child2'])
        ns1, ns2 = FAKER.uri(), FAKER.uri()
        # JSON doesn't have namespaces, so we adopt a convention (e.g., prefix_tag)
        json_data = {f'{nm["prefix1"]}:{nm["child1"]}': FAKER.word()}
        # XML generation needs to be manual to handle namespaces correctly
        xml_string = f'<{nm["root"]} xmlns:{nm["prefix1"]}="{ns1}"><{nm["prefix1"]}:{nm["child1"]}>{json_data[f"{nm['prefix1']}:{nm['child1']}"]}</{nm["prefix1"]}:{nm["child1"]}></{nm["root"]}>'
        return self._build_output(xml_string, {nm['root']: json_data})

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
        return self._build_output(ET.tostring(root, 'unicode'), {nm['root']: {nm['event']: json_list}})

class CDataField(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('CDataField', ['cdata_section', 'simple_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'script'])
        cdata_content = f'<p>Hello, {FAKER.name()}!</p><script>alert("XSS & fun");</script>'
        json_data = {nm['script']: cdata_content}
        # Manual XML generation is required for CDATA
        xml_string = f'<{nm["root"]}><{nm["script"]}><![CDATA[{cdata_content}]]></{nm["script"]}></{nm["root"]}>'
        return self._build_output(xml_string, {nm['root']: json_data})

class ProcessingInstructionNode(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ProcessingInstructionNode', ['processing_instruction'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'field'])
        pi_target = 'xml-stylesheet'
        pi_data = f'href="{FAKER.file_name(extension="css")}" type="text/css"'
        json_data = {'_processing_instructions': [{pi_target: pi_data}]} # Convention
        # Manual XML generation required for PIs
        xml_string = f'<?{pi_target} {pi_data}?><{nm["root"]}><{nm["field"]}>content</{nm["field"]}></{nm["root"]}>'
        # We add the PI to the JSON but the model might learn to ignore it, which is also valid.
        # For this test, we'll assume it should be captured.
        json_root_data = {nm['field']: 'content', **json_data}
        return self._build_output(xml_string, {nm['root']: json_root_data})

class EntityReferenceField(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('EntityReferenceField', ['simple_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'footer'])
        company_name = FAKER.company()
        year = FAKER.year()
        # The XML contains the entity reference
        xml_text = f'Copyright © {year} {company_name}.'
        # The JSON contains the resolved text
        json_text = f'Copyright © {year} {company_name}.'
        # Manual XML generation required for DTD and entities
        dtd = f'<!DOCTYPE {nm["root"]} [<!ENTITY company "{company_name}"><!ENTITY year "{year}">]>'
        xml_string = f'{dtd}<{nm["root"]}><{nm["footer"]}>Copyright © &year; &company;</{nm["footer"]}></{nm["root"]}>'
        return self._build_output(xml_string, {nm['root']: {nm['footer']: json_text}})

# --- The 4 "Advanced Placement" (AP) Formats ---

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
        return self._build_output(ET.tostring(root, 'unicode'), {nm['root']: {nm['day']: out}})

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
        return self._build_output(ET.tostring(root, 'unicode'), current_dict)

class ListOfNamespacedEntities(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ListOfNamespacedEntities', ['list_of_objects', 'namespace_handling', 'attribute_to_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root', 'prefix', 'item', 'id_attr', 'name'])
        ns = FAKER.uri(); json_list = []
        xml_parts = [f'<{nm["root"]} xmlns:{nm["prefix"]}="{ns}">']
        for _ in range(random.randint(*cfg['list_length'])):
            item_id, item_name = FAKER.uuid4(), FAKER.word()
            json_list.append({nm['id_attr']: item_id, f'{nm["prefix"]}:{nm["name"]}': item_name})
            xml_parts.append(f'<{nm["prefix"]}:{nm["item"]} {nm["id_attr"]}="{item_id}"><{nm["prefix"]}:{nm["name"]}>{item_name}</{nm["prefix"]}:{nm["name"]}></{nm["prefix"]}:{nm["item"]}>')
        xml_parts.append(f'</{nm["root"]}>')
        return self._build_output("".join(xml_parts), {nm['root']: {f'{nm["prefix"]}:{nm["item"]}': json_list}})

class ComplexMixedContent(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ComplexMixedContent', ['mixed_content', 'namespace_handling', 'self_closing_to_bool'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root', 'prefix', 'action', 'alert', '_value'])
        ns = FAKER.uri()
        text1, text2 = FAKER.sentence(nb_words=3), FAKER.sentence(nb_words=2)
        action_text = FAKER.word()
        json_data = {
            nm['_value']: [text1.strip(), text2.strip()], # Convention for multiple text nodes
            f'{nm["prefix"]}:{nm["action"]}': action_text,
            f'{nm["prefix"]}:{nm["alert"]}': True
        }
        xml_string = f'<{nm["root"]} xmlns:{nm["prefix"]}="{ns}">{text1}<{nm["prefix"]}:{nm["action"]}>{action_text}</{nm["prefix"]}:{nm["action"]}>{text2}<{nm["prefix"]}:{nm["alert"]}/></{nm["root"]}>'
        return self._build_output(xml_string, {nm['root']: json_data})

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
        # Core Curriculum
        'SingleField': SingleField(structured_names=use_structured_names),
        'AttributedEntity': AttributedEntity(structured_names=use_structured_names),
        'BasicEntity': BasicEntity(structured_names=use_structured_names),
        'PropertyList': PropertyList(structured_names=use_structured_names),
        'ListOfSimpleEntities': ListOfSimpleEntities(structured_names=use_structured_names),
        'MixedContentNode': MixedContentNode(structured_names=use_structured_names),
        'SpecialValues': SpecialValues(structured_names=use_structured_names),
        'NamespacedObject': NamespacedObject(structured_names=use_structured_names),
        'HeterogeneousList': HeterogeneousList(structured_names=use_structured_names),
        'CDataField': CDataField(structured_names=use_structured_names),
        'ProcessingInstructionNode': ProcessingInstructionNode(structured_names=use_structured_names),
        'EntityReferenceField': EntityReferenceField(structured_names=use_structured_names),
        # AP Courses
        'ListOfLists': ListOfLists(structured_names=use_structured_names),
        'DeeplyNested': DeeplyNested(structured_names=use_structured_names),
        'ListOfNamespacedEntities': ListOfNamespacedEntities(structured_names=use_structured_names),
        'ComplexMixedContent': ComplexMixedContent(structured_names=use_structured_names),
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

# --- Main Execution ---
if __name__ == '__main__':
    print(f"--- Running Experiment: {EXPERIMENT_TO_RUN} ---")
    print(f"Description: {config['description']}")
    print(f"Training formats: {TRAINING_FORMATS}")
    print(f"Held-out test format: [{TEST_FORMAT}]")
    print(f"Total examples: {N_EXAMPLES}")
    print(f"Curriculum learning: {CURRICULUM_LEARNING}")
    print(f"Using structured names: {FIXED_GRAMMAR_FOR_STRUCTURE}")
    print(f"Using random indentation: {RANDOM_INDENT}\n{'-'*50}")
    
    # # --- Generate Training and Test Sets Separately ---
    # n_train = int(N_EXAMPLES * 0.8)
    # n_test = N_EXAMPLES - n_train

    # print(f"\n>>> Generating {n_train} training examples...")
    full_dataset = generate_dataset(
        total_examples=N_EXAMPLES,
        formats_to_use=ALL_FOCUSED_FORMATS,
        distribution=DIFFICULTY_DISTRIBUTION,
        random_seed=RANDOM_SEED,
        use_structured_names=FIXED_GRAMMAR_FOR_STRUCTURE,
        curriculum=CURRICULUM_LEARNING,
    )

    # print(f"\n>>> Generating {n_test} test examples...")
    # test_dataset = generate_dataset(
    #     total_examples=n_test,
    #     formats_to_use=[TEST_FORMAT],
    #     distribution=DIFFICULTY_DISTRIBUTION,
    #     random_seed=RANDOM_SEED + 1, # Use a different seed
    #     use_structured_names=FIXED_GRAMMAR_FOR_STRUCTURE,
    #     curriculum=False # No need for curriculum in test set
    # )

    # --- Combine and Save ---
    # full_dataset = {'dataset': full_dataset['dataset']}
    # random.shuffle(full_dataset['dataset'])

    grammar_part = 'grammar' if FIXED_GRAMMAR_FOR_STRUCTURE else 'uuid'
    indent_part = 'randomindent' if RANDOM_INDENT else 'prettyindent'
    filename = f'dataset_utils/tagged_dataset_{EXPERIMENT_TO_RUN}_test_{TEST_FORMAT}_{N_EXAMPLES}_{grammar_part}_{indent_part}.json'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=2)
    print(f"\nDataset successfully generated and saved to '{filename}'")

    # --- Verification ---
    print("\n--- Sample from Training Set ---")
    sample_train = random.choice(full_dataset['dataset'])
    print(json.dumps({'format_id': sample_train['format_id'], 'question': sample_train['question']}, indent=2))

    # print("\n--- Sample from Test Set ---")
    # sample_test = random.choice(test_dataset['dataset'])
    # print(json.dumps({'format_id': sample_test['format_id'], 'question': sample_test['question']}, indent=2))