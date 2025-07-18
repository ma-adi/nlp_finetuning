import json
import random
import uuid
import xml.etree.ElementTree as ET
from faker import Faker
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration & Setup ---

FAKER = Faker()

# 1. CLASSIFY FORMATS BY INTRINSIC DIFFICULTY
EASY_FORMATS = ['BasicEntity', 'EmptyElement', 'MixedContent']
MEDIUM_FORMATS = ['PropertyList', 'DeeplyNested']
HARD_FORMATS = ['ListOfComplexEntities', 'NestedConfig', 'ListOfLists']

# 2. CONFIGURABLE DIFFICULTY LIMITS
# This is the new, user-configurable section.
DIFFICULTY_CONFIG = {
    # 'easy': Examples with short lists and low chance of optional complex parts.
    'easy': {
        'list_length': (2, 4),      # (min, max) number of items in a list.
        'optional_prob': 0.1        # 10% chance of including an optional complex part.
    },
    # 'medium': Mid-range list lengths and a moderate chance of complexity.
    'medium': {
        'list_length': (5, 9),
        'optional_prob': 0.5        # 50% chance.
    },
    # 'hard': Long lists and a very high chance of including all complex parts.
    'hard': {
        'list_length': (10, 20),
        'optional_prob': 0.9        # 90% chance.
    }
}

# --- Base Class (No changes needed here) ---

class FormatTemplate:
    def __init__(self, format_id: str, tags: List[str]):
        self.format_id = format_id
        self.tags = tags

    def _generate_name_map(self, keys: List[str]) -> Dict[str, str]:
        return {key: str(uuid.uuid4())[:8] for key in keys}

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        raise NotImplementedError

    def _build_output(self, xml_root: ET.Element, json_dict: Dict) -> Dict[str, Any]:
        try:
            ET.indent(xml_root, space="  ")
        except AttributeError:
            pass
        xml_string = ET.tostring(xml_root, encoding='unicode')
        return {'question': xml_string, 'answer': json.dumps(json_dict), 'format_id': self.format_id, 'tags': self.tags}

# --- Format Definitions Using the New Config ---

class BasicEntityFormat(FormatTemplate):
    def __init__(self):
        super().__init__('BasicEntity', ['simple_kv', 'attribute_to_kv', 'numeric_casting'])

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        name_map = self._generate_name_map(['root', 'id_attr', 'field1', 'field2'])
        root_el = ET.Element(name_map['root'])
        root_el.set(name_map['id_attr'], FAKER.uuid4())
        field1_el = ET.SubElement(root_el, name_map['field1'])
        field1_el.text = FAKER.word()
        age_val = FAKER.random_int(min=18, max=80)
        field2_el = ET.SubElement(root_el, name_map['field2'])
        field2_el.text = str(age_val)
        json_dict = {name_map['root']: {name_map['id_attr']: root_el.get(name_map['id_attr']), name_map['field1']: field1_el.text, name_map['field2']: age_val}}
        return self._build_output(root_el, json_dict)

class ListOfComplexEntitiesFormat(FormatTemplate):
    def __init__(self):
        super().__init__('ListOfComplexEntities', ['list_of_objects', 'mixed_content', 'boolean_casting', 'numeric_casting'])

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        # Read list length directly from the central config
        config = DIFFICULTY_CONFIG[difficulty]
        min_len, max_len = config['list_length']
        num_items = random.randint(min_len, max_len)

        name_map = self._generate_name_map(['root', 'item', 'id_attr', 'flag_attr', 'field1', 'field2'])
        root_el = ET.Element(name_map['root'])
        json_list = []
        for _ in range(num_items):
            is_active = FAKER.boolean()
            stock_count = FAKER.random_int(min=0, max=500)
            item_el = ET.SubElement(root_el, name_map['item'])
            item_el.set(name_map['id_attr'], FAKER.pystr(min_chars=8, max_chars=8))
            item_el.set(name_map['flag_attr'], str(is_active).lower())
            field1_el = ET.SubElement(item_el, name_map['field1'])
            field1_el.text = FAKER.bs()
            field2_el = ET.SubElement(item_el, name_map['field2'])
            field2_el.text = str(stock_count)
            json_list.append({name_map['id_attr']: item_el.get(name_map['id_attr']), name_map['flag_attr']: is_active, name_map['field1']: field1_el.text, name_map['field2']: stock_count})
        json_dict = {name_map['root']: {name_map['item']: json_list}}
        return self._build_output(root_el, json_dict)

class NestedConfigFormat(FormatTemplate):
    def __init__(self):
        super().__init__('NestedConfig', ['nested_object', 'attribute_to_kv', 'self_closing_tag', 'boolean_casting'])

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        # Use probability from the central config to determine complexity
        prob_optional = DIFFICULTY_CONFIG[difficulty]['optional_prob']
        include_optional_section = (random.random() < prob_optional)

        name_map = self._generate_name_map(['root', 'group1', 'group2', 'g1_attr', 'g2_field', 'opt_group', 'opt_flag'])
        root_el = ET.Element(name_map['root'])
        g1_el = ET.SubElement(root_el, name_map['group1'])
        g1_el.set(name_map['g1_attr'], str(FAKER.boolean()).lower())
        g2_el = ET.SubElement(root_el, name_map['group2'])
        g2_field_el = ET.SubElement(g2_el, name_map['g2_field'])
        g2_field_el.text = FAKER.uri_path()
        json_dict = {name_map['root']: {name_map['group1']: {name_map['g1_attr']: g1_el.get(name_map['g1_attr']) == 'true'}, name_map['group2']: {name_map['g2_field']: g2_field_el.text}}}
        if include_optional_section:
            opt_el = ET.SubElement(root_el, name_map['opt_group'])
            ET.SubElement(opt_el, name_map['opt_flag'])
            json_dict[name_map['root']][name_map['opt_group']] = {name_map['opt_flag']: True}
        return self._build_output(root_el, json_dict)

class PropertyListFormat(FormatTemplate):
    def __init__(self):
        super().__init__('PropertyList', ['nested_object', 'list_of_values'])

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        # Use the central config to determine list length
        config = DIFFICULTY_CONFIG[difficulty]
        min_len, max_len = config['list_length']
        num_items = random.randint(min_len, max_len)

        name_map = self._generate_name_map(['root', 'list_item'])
        root_el = ET.Element(name_map['root'])
        
        items = [FAKER.word() for _ in range(num_items)]
        for item in items:
            item_el = ET.SubElement(root_el, name_map['list_item'])
            item_el.text = item
            
        json_dict = { name_map['root']: { name_map['list_item']: items } }
        return self._build_output(root_el, json_dict)

class DeeplyNestedFormat(FormatTemplate):
    def __init__(self):
        super().__init__('DeeplyNested', ['nested_object', 'simple_kv', 'numeric_casting'])

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        # For this format, we can vary the depth based on difficulty
        if difficulty == 'easy':
            depth = 2
        elif difficulty == 'hard':
            depth = 5
        else: # medium
            depth = 3

        keys = [f'level{i+1}' for i in range(depth)] + ['field']
        name_map = self._generate_name_map(keys)
        
        # Create the nested XML structure
        current_el = ET.Element(name_map['level1'])
        root_el = current_el
        for i in range(1, depth):
            current_el = ET.SubElement(current_el, name_map[f'level{i+1}'])
        
        field_el = ET.SubElement(current_el, name_map['field'])
        val = FAKER.random_int(min=100, max=999)
        field_el.text = str(val)
        
        # Create the nested JSON structure
        current_dict = {name_map['field']: val}
        for i in range(depth - 1, -1, -1):
            current_dict = {name_map[f'level{i+1}']: current_dict}
        
        json_dict = current_dict
        return self._build_output(root_el, json_dict)

# --- And here are the remaining missing classes for completeness ---

class EmptyElementFormat(FormatTemplate):
    def __init__(self):
        super().__init__('EmptyElement', ['simple_kv', 'empty_element', 'nested_object'])

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        name_map = self._generate_name_map(['root', 'field1', 'field2_empty'])
        root_el = ET.Element(name_map['root'])
        field1_el = ET.SubElement(root_el, name_map['field1'])
        field1_el.text = FAKER.job()
        field2_el = ET.SubElement(root_el, name_map['field2_empty'])
        field2_el.text = ''
        json_dict = {name_map['root']: {name_map['field1']: field1_el.text, name_map['field2_empty']: None}}
        return self._build_output(root_el, json_dict)

class MixedContentFormat(FormatTemplate):
    def __init__(self):
        super().__init__('MixedContent', ['mixed_content', 'attribute_to_kv', 'simple_kv'])

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        name_map = self._generate_name_map(['root', 'id_attr', 'field1'])
        root_el = ET.Element(name_map['root'])
        root_el.set(name_map['id_attr'], FAKER.license_plate())
        field1_el = ET.SubElement(root_el, name_map['field1'])
        field1_el.text = FAKER.color_name()
        json_dict = {name_map['root']: {name_map['id_attr']: root_el.get(name_map['id_attr']), name_map['field1']: field1_el.text}}
        return self._build_output(root_el, json_dict)

class ListOfListsFormat(FormatTemplate):
    def __init__(self):
        super().__init__('ListOfLists', ['list_of_objects', 'list_of_values', 'nested_object'])

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        config = DIFFICULTY_CONFIG[difficulty]
        min_len, max_len = config['list_length']
        num_items = random.randint(min_len, max_len)

        name_map = self._generate_name_map(['root', 'item', 'sub_item'])
        root_el = ET.Element(name_map['root'])
        json_list = []
        for _ in range(num_items):
            item_el = ET.SubElement(root_el, name_map['item'])
            num_sub_items = random.randint(2, 5)
            sub_items = [FAKER.word() for _ in range(num_sub_items)]
            for sub_item_text in sub_items:
                sub_item_el = ET.SubElement(item_el, name_map['sub_item'])
                sub_item_el.text = sub_item_text
            json_list.append({name_map['sub_item']: sub_items})
        json_dict = {name_map['root']: {name_map['item']: json_list}}
        return self._build_output(root_el, json_dict)

# --- Main Orchestration Function (No changes needed here) ---

def generate_dataset(
    total_examples: int,
    distribution: Tuple[float, float, float] = (0.33, 0.34, 0.33),
    exclude_formats: Optional[List[str]] = None,
    random_seed: Optional[int] = None
) -> Dict[str, List[Dict]]:
    if random_seed is not None:
        random.seed(random_seed)
        FAKER.seed_instance(random_seed)
        print(f"Using random seed: {random_seed}")

    all_generators = {
        'BasicEntity': BasicEntityFormat(),
        'EmptyElement': EmptyElementFormat(),
        'MixedContent': MixedContentFormat(),
        'PropertyList': PropertyListFormat(),
        'DeeplyNested': DeeplyNestedFormat(),
        'ListOfComplexEntities': ListOfComplexEntitiesFormat(),
        'NestedConfig': NestedConfigFormat(),
        'ListOfLists': ListOfListsFormat(),
    }

    if exclude_formats:
        all_generators = {k: v for k, v in all_generators.items() if k not in exclude_formats}
        print(f"Excluding formats: {exclude_formats}")

    available_easy = [f for f in EASY_FORMATS if f in all_generators]
    available_medium = [f for f in MEDIUM_FORMATS if f in all_generators]
    available_hard = [f for f in HARD_FORMATS if f in all_generators]
    difficulty_map = {'easy': available_easy, 'medium': available_medium, 'hard': available_hard}

    n_easy = int(total_examples * distribution[0])
    n_medium = int(total_examples * distribution[1])
    n_hard = total_examples - n_easy - n_medium
    counts_map = {'easy': n_easy, 'medium': n_medium, 'hard': n_hard}
    
    dataset = []
    print("\nStarting dataset generation...")
    for difficulty, num_examples in counts_map.items():
        available_formats = difficulty_map[difficulty]
        if not available_formats:
            print(f"Warning: No available formats for difficulty '{difficulty}'. Skipping {num_examples} examples.")
            continue
        print(f"Generating {num_examples} '{difficulty}' examples from formats: {available_formats}")
        for i in range(num_examples):
            format_id_to_use = available_formats[i % len(available_formats)]
            generator = all_generators[format_id_to_use]
            example = generator.generate_example(difficulty=difficulty)
            dataset.append(example)
            
    print("\nDataset generation complete.")
    random.shuffle(dataset)
    return {'dataset': dataset}

# --- Example Usage ---
if __name__ == '__main__':
    # === Configuration ===
    N_TOTAL_EXAMPLES = 750
    DIFFICULTY_DISTRIBUTION = (0.3, 0.4, 0.3)
    RANDOM_SEED = 42
    OUTPUT_FILENAME = f'tagged_dataset_{N_TOTAL_EXAMPLES}.json'
    
    # === Generation ===
    print(f"--- Generating a single, complete dataset with {N_TOTAL_EXAMPLES} examples ---")
    print(f"Using difficulty config: {json.dumps(DIFFICULTY_CONFIG, indent=2)}")
    
    full_dataset = generate_dataset(
        total_examples=N_TOTAL_EXAMPLES,
        distribution=DIFFICULTY_DISTRIBUTION,
        exclude_formats=None,
        random_seed=RANDOM_SEED
    )
    
    # === Save and Verify ===
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=2)
    print(f"\nFull dataset with {len(full_dataset['dataset'])} examples saved to '{OUTPUT_FILENAME}'")

    print("\n--- Sample Generated Example ---")
    sample_example = random.choice(full_dataset['dataset'])
    print(f"Format ID: {sample_example['format_id']}")
    print(f"Tags: {sample_example['tags']}")
    print("\nQuestion (XML):")
    print(sample_example['question'])
    print("\nAnswer (JSON):")
    print(json.dumps(json.loads(sample_example['answer']), indent=2))