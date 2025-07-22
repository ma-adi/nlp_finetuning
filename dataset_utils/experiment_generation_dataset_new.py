import json
import random
import uuid
import xml.etree.ElementTree as ET
from faker import Faker
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration & Setup ---

FAKER = Faker()

N_EXAMPLES = 750
DIFFICULTY_DISTRIBUTION = (0.3, 0.4, 0.3)  # Mix of easy, medium, hard
RANDOM_SEED = 42
RANDOM_INDENT = True  # Toggle to randomize XML indentation
FIXED_GRAMMAR_FOR_STRUCTURE = False # Use structured tag names instead of UUIDs
CURRICULUM_LEARNING = True

# --- Experiment Definitions ---
# Choose which experiment to run by setting this variable
EXPERIMENT_TO_RUN = 'comprehensive_mix'

EXPERIMENTS = {
    'nested_lists_with_dict': {
        'description': "Extends nested_lists_easy by adding pure multi-pair dictionary format.",
        'training_formats': ['PropertyList', 'SimplifiedListOfComplexEntities', 'ObjectWithList', 'Dictionary'],
        'test_format': 'ListOfLists'
    },
    'shape_combination': {
        'description': "Tests if the model can combine a simple KV field and a list into one object.",
        'training_formats': ['SingleField', 'SimpleList'],
        'test_format': 'MixedFields'
    },
    'comprehensive_mix': {
        'description': "A comprehensive mix of formats testing various XML structures, including attributes, type casting, and nested objects. Ported from the legacy generator.",
        'training_formats': ['BasicEntity', 'EmptyElement', 'MixedContent', 'PropertyList', 'DeeplyNested', 'NestedConfig', 'SimpleList'],
        'test_format': 'ListOfComplexEntities' # A great held-out format combining many features.
    }
}

TRAINING_FORMATS = EXPERIMENTS[EXPERIMENT_TO_RUN]['training_formats']
TEST_FORMAT = EXPERIMENTS[EXPERIMENT_TO_RUN]['test_format']
ALL_FOCUSED_FORMATS = TRAINING_FORMATS + [TEST_FORMAT]

# 2. UNIFIED & EXPANDED DIFFICULTY CONFIGURATION
DIFFICULTY_CONFIG = {
    'easy': {
        'list_length': (2, 4),
        'sub_list_length': (1, 2),
        'nesting_depth': 2,         # For formats like DeeplyNested
        'optional_prob': 0.1,       # For formats like NestedConfig
    },
    'medium': {
        'list_length': (5, 9),
        'sub_list_length': (2, 4),
        'nesting_depth': 3,
        'optional_prob': 0.5,
    },
    'hard': {
        'list_length': (10, 20),
        'sub_list_length': (4, 8),
        'nesting_depth': 5,
        'optional_prob': 0.9,
    }
}

# --- Base Class ---
class FormatTemplate:
    def __init__(self, format_id: str, tags: List[str], **kwargs):
        self.format_id = format_id
        self.tags = tags
        self.structured_names = kwargs.pop('structured_names', False)
        # Expanded vocabulary to accommodate all formats
        self.TAG_VOCAB = [
            'container', 'wrapper', 'item', 'element', 'property', 'field', 'name', 'value', 'id', 'data',
            'config', 'setting', 'user', 'product', 'details', 'entry', 'record', 'level', 'group', 'flag'
        ]

    def generate_name_map_structured(self, keys: List[str]) -> Dict[str, str]:
        if len(keys) > len(self.TAG_VOCAB):
            raise ValueError(f"Not enough unique tags in vocabulary for format '{self.format_id}'. Need {len(keys)}, have {len(self.TAG_VOCAB)}.")
        chosen = random.sample(self.TAG_VOCAB, len(keys))
        return {k: v for k, v in zip(keys, chosen)}

    def generate_name_map_random(self, keys: List[str]) -> Dict[str, str]:
        return {k: str(uuid.uuid4())[:8] for k in keys}

    def _generate_name_map(self, keys: List[str]) -> Dict[str, str]:
        return (self.generate_name_map_structured(keys)
                if self.structured_names else self.generate_name_map_random(keys))

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        raise NotImplementedError

    def _build_output(self, xml_root: ET.Element, json_dict: Dict) -> Dict[str, Any]:
        if not RANDOM_INDENT:
            try:
                ET.indent(xml_root, space="  ")
            except AttributeError:
                pass # For older Python versions
            xml_string = ET.tostring(xml_root, encoding='unicode')
        else:
            def build_chaotic_xml(element, level=0):
                indent = '\n' + ' ' * (level * random.randint(0,4))
                parts = [f"<{element.tag}"]
                if element.attrib:
                    parts.append(" " + " ".join(f'{k}="{v}"' for k, v in element.attrib.items()))
                parts.append(">")
                if element.text and element.text.strip():
                    parts.append(element.text.strip())
                if len(element):
                    for child in element:
                        parts.append(build_chaotic_xml(child, level + 1))
                    parts.append(indent)
                parts.append(f"</{element.tag}>")
                return ''.join(parts).replace('\n', '\n' + ' '*random.randint(0,5))
            xml_string = build_chaotic_xml(xml_root).strip()

        return {
            'question': xml_string,
            'answer': json.dumps(json_dict),
            'format_id': self.format_id,
            'tags': self.tags
        }

# --- Format Implementations (Original + Ported) ---

# --- Original Format Classes ---

class PropertyListFormat(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('PropertyList', ['list_of_values', 'nested_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        min_len, max_len = DIFFICULTY_CONFIG[difficulty]['list_length']
        items = [FAKER.word() for _ in range(random.randint(min_len, max_len))]
        nm = self._generate_name_map(['root', 'list_item'])
        root = ET.Element(nm['root'])
        for it in items:
            el = ET.SubElement(root, nm['list_item']); el.text = it
        return self._build_output(root, {nm['root']: {nm['list_item']: items}})

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

class ObjectWithListFormat(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ObjectWithList', ['nested_object', 'list_of_values'], **kwargs)
    def generate_example(self, difficulty='medium'):
        min_len, max_len = DIFFICULTY_CONFIG[difficulty]['list_length']
        nm = self._generate_name_map(['root', 'object_name', 'list_name'])
        root = ET.Element(nm['root']); obj=ET.SubElement(root, nm['object_name'])
        items = [FAKER.word() for _ in range(random.randint(min_len, max_len))]
        for it in items:
            si = ET.SubElement(obj, nm['list_name']); si.text = it
        return self._build_output(root, {nm['root']: {nm['object_name']: {nm['list_name']: items}}})

class ListOfListsFormat(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('ListOfLists', ['list_of_objects','list_of_values','nested_object'], **kwargs)
    def generate_example(self, difficulty='medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root','item','sub_item'])
        root = ET.Element(nm['root']); out=[]
        for _ in range(random.randint(*cfg['list_length'])):
            it = ET.SubElement(root, nm['item'])
            subs = [FAKER.word() for _ in range(random.randint(*cfg['sub_list_length']))]
            for s in subs:
                se = ET.SubElement(it, nm['sub_item']); se.text = s
            out.append({nm['sub_item']: subs})
        return self._build_output(root, {nm['root']: {nm['item']: out}})

class SingleFieldFormat(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('SingleField', ['simple_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        nm = self._generate_name_map(['root','field'])
        root = ET.Element(nm['root']); f=ET.SubElement(root, nm['field']); f.text=FAKER.catch_phrase()
        return self._build_output(root, {nm['root']: {nm['field']: f.text}})

class SimpleListFormat(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('SimpleList', ['list_of_values'], **kwargs)
    def generate_example(self, difficulty='medium'):
        min_len, max_len = DIFFICULTY_CONFIG[difficulty]['list_length']
        items = [FAKER.word() for _ in range(random.randint(min_len, max_len))]
        nm = self._generate_name_map(['root','item'])
        root = ET.Element(nm['root'])
        for it in items: ET.SubElement(root, nm['item']).text = it
        return self._build_output(root, {nm['root']: {nm['item']: items}})

class DictionaryFormat(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('Dictionary', ['multi_kv'], **kwargs)
    def generate_example(self, difficulty='medium'):
        min_len, max_len = DIFFICULTY_CONFIG[difficulty]['list_length']
        count = random.randint(min_len, max_len)
        keys = ['root'] + [f'field{i}' for i in range(1, count+1)]
        nm = self._generate_name_map(keys)
        root = ET.Element(nm['root']); data = {}
        for i in range(1, count+1):
            key_tag = nm[f'field{i}']; val = FAKER.word()
            el = ET.SubElement(root, key_tag); el.text = val; data[key_tag] = val
        return self._build_output(root, {nm['root']: data})

class MixedFieldsFormat(FormatTemplate):
    def __init__(self, **kwargs): super().__init__('MixedFields', ['simple_kv','list_of_values'], **kwargs)
    def generate_example(self, difficulty='medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        nm = self._generate_name_map(['root','field','item'])
        root = ET.Element(nm['root'])
        f=ET.SubElement(root, nm['field']); f.text=FAKER.catch_phrase()
        items = [FAKER.word() for _ in range(random.randint(*cfg['list_length']))]
        for it in items: ET.SubElement(root, nm['item']).text = it
        return self._build_output(root, {nm['root']: {nm['field']: f.text, nm['item']: items}})

# --- Ported and Refactored Format Classes ---

class BasicEntityFormat(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('BasicEntity', ['simple_kv', 'attribute_to_kv', 'numeric_casting'], **kwargs)
    def generate_example(self, difficulty: str = 'medium'):
        nm = self._generate_name_map(['root', 'id_attr', 'field1', 'field2'])
        root = ET.Element(nm['root'])
        root.set(nm['id_attr'], FAKER.uuid4())
        f1 = ET.SubElement(root, nm['field1']); f1.text = FAKER.word()
        age = FAKER.random_int(min=18, max=80); f2 = ET.SubElement(root, nm['field2']); f2.text = str(age)
        json_data = {nm['id_attr']: root.get(nm['id_attr']), nm['field1']: f1.text, nm['field2']: age}
        return self._build_output(root, {nm['root']: json_data})

class ListOfComplexEntitiesFormat(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('ListOfComplexEntities', ['list_of_objects', 'attribute_to_kv', 'boolean_casting', 'numeric_casting'], **kwargs)
    def generate_example(self, difficulty: str = 'medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        num_items = random.randint(*cfg['list_length'])
        nm = self._generate_name_map(['root', 'item', 'id_attr', 'flag_attr', 'field1', 'field2'])
        root = ET.Element(nm['root']); json_list = []
        for _ in range(num_items):
            is_active = FAKER.boolean()
            stock = FAKER.random_int(min=0, max=500)
            item_el = ET.SubElement(root, nm['item'])
            item_el.set(nm['id_attr'], FAKER.pystr(min_chars=8, max_chars=8))
            item_el.set(nm['flag_attr'], str(is_active).lower())
            f1 = ET.SubElement(item_el, nm['field1']); f1.text = FAKER.bs()
            f2 = ET.SubElement(item_el, nm['field2']); f2.text = str(stock)
            json_list.append({
                nm['id_attr']: item_el.get(nm['id_attr']),
                nm['flag_attr']: is_active,
                nm['field1']: f1.text,
                nm['field2']: stock
            })
        return self._build_output(root, {nm['root']: {nm['item']: json_list}})

class NestedConfigFormat(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('NestedConfig', ['nested_object', 'attribute_to_kv', 'empty_element_to_bool'], **kwargs)
    def generate_example(self, difficulty: str = 'medium'):
        cfg = DIFFICULTY_CONFIG[difficulty]
        include_optional = (random.random() < cfg['optional_prob'])
        nm = self._generate_name_map(['root', 'group1', 'group2', 'g1_attr', 'g2_field', 'opt_group', 'opt_flag'])
        root = ET.Element(nm['root'])
        g1 = ET.SubElement(root, nm['group1']); g1_active = FAKER.boolean(); g1.set(nm['g1_attr'], str(g1_active).lower())
        g2 = ET.SubElement(root, nm['group2']); g2f = ET.SubElement(g2, nm['g2_field']); g2f.text = FAKER.uri_path()
        json_data = {
            nm['group1']: {nm['g1_attr']: g1_active},
            nm['group2']: {nm['g2_field']: g2f.text}
        }
        if include_optional:
            opt = ET.SubElement(root, nm['opt_group']); ET.SubElement(opt, nm['opt_flag'])
            json_data[nm['opt_group']] = {nm['opt_flag']: True} # This rule (empty tag -> true) is a good test case
        return self._build_output(root, {nm['root']: json_data})

class DeeplyNestedFormat(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('DeeplyNested', ['nested_object', 'simple_kv', 'numeric_casting'], **kwargs)
    def generate_example(self, difficulty: str = 'medium'):
        depth = DIFFICULTY_CONFIG[difficulty]['nesting_depth']
        keys = [f'level{i+1}' for i in range(depth)] + ['field']
        nm = self._generate_name_map(keys)
        # Create XML
        current_el = root = ET.Element(nm['level1'])
        for i in range(1, depth): current_el = ET.SubElement(current_el, nm[f'level{i+1}'])
        val = FAKER.random_int(100, 999); f = ET.SubElement(current_el, nm['field']); f.text = str(val)
        # Create JSON
        current_dict = {nm['field']: val}
        for i in range(depth - 1, -1, -1): current_dict = {nm[f'level{i+1}']: current_dict}
        return self._build_output(root, current_dict)

class EmptyElementFormat(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('EmptyElement', ['simple_kv', 'empty_element_to_null', 'nested_object'], **kwargs)
    def generate_example(self, difficulty: str = 'medium'):
        nm = self._generate_name_map(['root', 'field1', 'field2_empty'])
        root = ET.Element(nm['root'])
        f1 = ET.SubElement(root, nm['field1']); f1.text = FAKER.job()
        f2 = ET.SubElement(root, nm['field2_empty']); f2.text = '' # Empty text
        json_data = {nm['field1']: f1.text, nm['field2_empty']: None}
        return self._build_output(root, {nm['root']: json_data})

class MixedContentFormat(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__('MixedContent', ['attribute_to_kv', 'simple_kv'], **kwargs)
    def generate_example(self, difficulty: str = 'medium'):
        nm = self._generate_name_map(['root', 'id_attr', 'field1'])
        root = ET.Element(nm['root'])
        root.set(nm['id_attr'], FAKER.license_plate())
        f1 = ET.SubElement(root, nm['field1']); f1.text = FAKER.color_name()
        json_data = {nm['id_attr']: root.get(nm['id_attr']), nm['field1']: f1.text}
        return self._build_output(root, {nm['root']: json_data})

# --- Orchestration ---
def generate_dataset(
    total_examples: int,
    distribution: Tuple[float, float, float] = (0.33,0.34,0.33),
    exclude_formats: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    use_structured_names: bool = False,
    curriculum: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    if random_seed is not None:
        random.seed(random_seed)
        FAKER.seed_instance(random_seed)

    all_generators = {
        # Original
        'PropertyList': PropertyListFormat(structured_names=use_structured_names),
        'SimplifiedListOfComplexEntities': SimplifiedListOfComplexEntitiesFormat(structured_names=use_structured_names),
        'ObjectWithList': ObjectWithListFormat(structured_names=use_structured_names),
        'ListOfLists': ListOfListsFormat(structured_names=use_structured_names),
        'SingleField': SingleFieldFormat(structured_names=use_structured_names),
        'SimpleList': SimpleListFormat(structured_names=use_structured_names),
        'Dictionary': DictionaryFormat(structured_names=use_structured_names),
        'MixedFields': MixedFieldsFormat(structured_names=use_structured_names),
        # Ported
        'BasicEntity': BasicEntityFormat(structured_names=use_structured_names),
        'ListOfComplexEntities': ListOfComplexEntitiesFormat(structured_names=use_structured_names),
        'NestedConfig': NestedConfigFormat(structured_names=use_structured_names),
        'DeeplyNested': DeeplyNestedFormat(structured_names=use_structured_names),
        'EmptyElement': EmptyElementFormat(structured_names=use_structured_names),
        'MixedContent': MixedContentFormat(structured_names=use_structured_names),
    }
    if exclude_formats:
        for fmt in exclude_formats: all_generators.pop(fmt, None)

    counts = {
        'easy': int(total_examples*distribution[0]),
        'medium': int(total_examples*distribution[1]),
        'hard': total_examples - int(total_examples*distribution[0]) - int(total_examples*distribution[1])
    }
    
    # Use the formats defined by the chosen experiment
    formats_for_experiment = [f for f in ALL_FOCUSED_FORMATS if f in all_generators]

    dataset=[]
    order = ['easy','medium','hard'] if curriculum else list(counts.keys())
    for diff in order:
        for _ in range(counts[diff]):
            fmt_name = random.choice(formats_for_experiment)
            dataset.append(all_generators[fmt_name].generate_example(difficulty=diff))
    
    if not curriculum: random.shuffle(dataset)
    return {'dataset': dataset}

# --- Main Execution ---
if __name__ == '__main__':
    print(f"--- Running Experiment: {EXPERIMENT_TO_RUN} ---")
    print(f"Description: {EXPERIMENTS[EXPERIMENT_TO_RUN]['description']}")
    print(f"Training formats: {TRAINING_FORMATS}")
    print(f"Held-out test format: [{TEST_FORMAT}]")
    print(f"Total examples: {N_EXAMPLES}")
    print(f"Curriculum learning: {CURRICULUM_LEARNING}")
    print(f"Using structured names: {FIXED_GRAMMAR_FOR_STRUCTURE}")
    print(f"Using random indentation: {RANDOM_INDENT}\n{'-'*50}")
    
    # Generate the dataset based on the chosen experiment
    full_dataset = generate_dataset(
        total_examples=N_EXAMPLES,
        distribution=DIFFICULTY_DISTRIBUTION,
        exclude_formats=None, # Filtering is now done by the experiment definition
        random_seed=RANDOM_SEED,
        use_structured_names=FIXED_GRAMMAR_FOR_STRUCTURE,
        curriculum=CURRICULUM_LEARNING
    )

    # Construct a descriptive filename
    grammar_part = 'grammar' if FIXED_GRAMMAR_FOR_STRUCTURE else 'uuid'
    indent_part = 'random_indent' if RANDOM_INDENT else 'pretty_indent'
    filename = f'dataset_exp_{EXPERIMENT_TO_RUN}_{N_EXAMPLES}_{grammar_part}_{indent_part}.json'
    
    # Save the dataset
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=2)
    print(f"Dataset successfully generated and saved to '{filename}'")

    # Print a random sample for verification
    print("\n--- Sample Generated Example ---")
    sample = random.choice(full_dataset['dataset'])
    print(json.dumps({
        'format_id': sample['format_id'],
        'tags': sample['tags'],
        'question': sample['question'],
        'answer': json.loads(sample['answer']) # pretty print the JSON answer
    }, indent=2))