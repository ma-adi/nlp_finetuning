import json
import random
import uuid
import xml.etree.ElementTree as ET
from faker import Faker
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration & Setup ---

FAKER = Faker()


N_EXAMPLES = 2000
# N_TEST_EXAMPLES = 100
DIFFICULTY_DISTRIBUTION = (0.3, 0.4, 0.3) # Mix of easy, medium, hard
RANDOM_SEED = 42

FIXED_GRAMMER_FOR_STRUCTURE = False

CURRICULUM_LEARNING = True

EXPERIMENT_TO_RUN = 'nested_lists_easy' 


EXPERIMENTS = {
    'nested_lists': {
        'description': "Tests if the model can learn to nest a list of values within a list of objects.",
        'training_formats': ['PropertyList', 'SimplifiedListOfComplexEntities'],
        'test_format': 'ListOfLists'
    },

    'nested_lists_easy': {
        'description': "Tests if the model can learn to nest a list of values within a list of objects.",
        'training_formats': ['PropertyList', 'SimplifiedListOfComplexEntities', 'ObjectWithList'],
        'test_format': 'ListOfLists'
    },
    'shape_combination': {
        'description': "Tests if the model can combine a simple KV field and a list into one object.",
        'training_formats': ['SingleField', 'SimpleList'],
        'test_format': 'MixedFields'
    }
}


TRAINING_FORMATS = EXPERIMENTS[EXPERIMENT_TO_RUN]['training_formats']

TEST_FORMAT = EXPERIMENTS[EXPERIMENT_TO_RUN]['test_format']

ALL_FOCUSED_FORMATS = TRAINING_FORMATS + [TEST_FORMAT]

EASY_FORMATS = ALL_FOCUSED_FORMATS
MEDIUM_FORMATS = ALL_FOCUSED_FORMATS
HARD_FORMATS = ALL_FOCUSED_FORMATS

# 2. CONFIGURABLE DIFFICULTY LIMITS (Unchanged)
DIFFICULTY_CONFIG = {
    'easy': {
        'list_length': (2, 4),
        'sub_list_length': (1, 2)  # <-- NEW: Keep inner lists very short for easy examples
    },
    'medium': {
        'list_length': (5, 9),
        'sub_list_length': (2, 4)  # <-- NEW: Mid-range inner lists
    },
    'hard': {
        'list_length': (10, 20),
        'sub_list_length': (4, 8)  # <-- NEW: Long inner lists for hard examples
    }
}

# --- Base Class (Unchanged) ---

class FormatTemplate:
    def __init__(self, format_id: str, tags: List[str], **kwargs):
        self.format_id = format_id
        self.tags = tags
        self.structured_names = kwargs.pop('structured_names', False)
        self.TAG_VOCAB = ['container','wrapper', 'item', 'element', 'property', 'field', 'name', 'value', 'id', 'data']

    def generate_name_map_structured(self, keys: List[str]) -> Dict[str, str]:
        # Sample without replacement from the vocabulary for this specific example
        # This ensures tags are different within one example but consistent across the dataset
        if len(keys) > len(self.TAG_VOCAB):
            raise ValueError("Not enough unique tags in vocabulary for this format.")
            
        chosen_tags = random.sample(self.TAG_VOCAB, len(keys))
        return {key: tag for key, tag in zip(keys, chosen_tags)}

    def generate_name_map_random(self, keys: List[str]) -> Dict[str, str]:
        return {key: str(uuid.uuid4())[:8] for key in keys}

    def _generate_name_map(self, keys: List[str]) -> Dict[str, str]:
        if self.structured_names:
            return self.generate_name_map_structured(keys)
        else:
            return self.generate_name_map_random(keys)


    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        raise NotImplementedError

    def _build_output(self, xml_root: ET.Element, json_dict: Dict) -> Dict[str, Any]:
        try:
            ET.indent(xml_root, space="  ")
        except AttributeError:
            pass
        xml_string = ET.tostring(xml_root, encoding='unicode')
        return {'question': xml_string, 'answer': json.dumps(json_dict), 'format_id': self.format_id, 'tags': self.tags}

# --- FOCUSED FORMAT DEFINITIONS ---

# TRAINING FORMAT 1: Teaches the concept of "list of values"
class PropertyListFormat(FormatTemplate):
    def __init__(self, **kwargs):
        super().__init__(format_id = 'PropertyList', tags=['list_of_values', 'nested_object'],  **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
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

# TRAINING FORMAT 2: Teaches the concept of "list of objects" (Simplified)
class SimplifiedListOfComplexEntitiesFormat(FormatTemplate):
    def __init__(self,  **kwargs):
        # Tags are simplified to reflect the core concept being taught
        super().__init__(format_id = 'SimplifiedListOfComplexEntities', tags = ['list_of_objects', 'nested_object'],  **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        config = DIFFICULTY_CONFIG[difficulty]
        min_len, max_len = config['list_length']
        num_items = random.randint(min_len, max_len)

        # Name map is simplified: no attributes, only one field per object
        name_map = self._generate_name_map(['root', 'item', 'name_field'])
        root_el = ET.Element(name_map['root'])
        json_list = []
        for _ in range(num_items):
            item_el = ET.SubElement(root_el, name_map['item'])
            
            # Each object now contains just one simple key-value pair
            field_text = FAKER.name()
            field_el = ET.SubElement(item_el, name_map['name_field'])
            field_el.text = field_text
            
            json_list.append({name_map['name_field']: field_text})
            
        json_dict = {name_map['root']: {name_map['item']: json_list}}
        return self._build_output(root_el, json_dict)
    

# A new format that explicitly teaches the target structure
class ObjectWithListFormat(FormatTemplate):
    def __init__(self,  **kwargs):
        super().__init__(format_id = 'ObjectWithList', tags = ['nested_object', 'list_of_values'],  **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        config = DIFFICULTY_CONFIG[difficulty]
        min_len, max_len = config['list_length']
        num_items = random.randint(min_len, max_len)

        # Use the NEW _generate_name_map
        name_map = self._generate_name_map(['root', 'object_name', 'list_name'])
        root_el = ET.Element(name_map['root'])
        
        # Create a single object that contains a list
        object_el = ET.SubElement(root_el, name_map['object_name'])
        items = [FAKER.word() for _ in range(num_items)]
        for item in items:
            item_el = ET.SubElement(object_el, name_map['list_name'])
            item_el.text = item
            
        # The JSON shows an object containing a key whose value is a list
        json_dict = {
            name_map['root']: {
                name_map['object_name']: {
                    name_map['list_name']: items
                }
            }
        }
        return self._build_output(root_el, json_dict)

# HELD-OUT TEST FORMAT: Tests the combination of the two learned concepts
class ListOfListsFormat(FormatTemplate):
    def __init__(self,  **kwargs):
        super().__init__(format_id = 'ListOfLists', tags = ['list_of_objects', 'list_of_values', 'nested_object'],  **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        config = DIFFICULTY_CONFIG[difficulty]
        min_len, max_len = config['list_length']
        # The outer list length is determined by the difficulty
        num_items = random.randint(min_len, max_len)

        min_inner, max_inner = config['sub_list_length']

        name_map = self._generate_name_map(['root', 'item', 'sub_item'])
        root_el = ET.Element(name_map['root'])
        json_list = []
        for _ in range(num_items):
            item_el = ET.SubElement(root_el, name_map['item'])
            # The inner list has a random length to add variability
            num_sub_items = random.randint(min_inner, max_inner)
            sub_items = [FAKER.word() for _ in range(num_sub_items)]
            for sub_item_text in sub_items:
                sub_item_el = ET.SubElement(item_el, name_map['sub_item'])
                sub_item_el.text = sub_item_text
            json_list.append({name_map['sub_item']: sub_items})
        json_dict = {name_map['root']: {name_map['item']: json_list}}
        return self._build_output(root_el, json_dict)
    
class SingleFieldFormat(FormatTemplate):
    def __init__(self,  **kwargs):
        super().__init__(format_id = 'SingleField', tags = ['simple_kv'],  **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        name_map = self._generate_name_map(['root', 'field'])
        root_el = ET.Element(name_map['root'])
        field_el = ET.SubElement(root_el, name_map['field'])
        field_el.text = FAKER.catch_phrase()
        
        json_dict = {name_map['root']: {name_map['field']: field_el.text}}
        return self._build_output(root_el, json_dict)

class SimpleListFormat(FormatTemplate):
    def __init__(self,  **kwargs):
        super().__init__(format_id = 'SimpleList', tags = ['list_of_values'], **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        config = DIFFICULTY_CONFIG[difficulty]
        min_len, max_len = config['list_length']
        num_items = random.randint(min_len, max_len)
        
        name_map = self._generate_name_map(['root', 'item'])
        root_el = ET.Element(name_map['root'])
        items = [FAKER.word() for _ in range(num_items)]
        for item_text in items:
            item_el = ET.SubElement(root_el, name_map['item'])
            item_el.text = item_text
            
        json_dict = {name_map['root']: {name_map['item']: items}}
        return self._build_output(root_el, json_dict)

class MixedFieldsFormat(FormatTemplate):
    def __init__(self,  **kwargs):
        super().__init__(format_id = 'MixedFields', tags = ['simple_kv', 'list_of_values'],  **kwargs)

    def generate_example(self, difficulty: str = 'medium') -> Dict[str, Any]:
        config = DIFFICULTY_CONFIG[difficulty]
        min_len, max_len = config['list_length']
        num_items = random.randint(min_len, max_len)
        
        name_map = self._generate_name_map(['root', 'field', 'item'])
        root_el = ET.Element(name_map['root'])
        
        # The simple field
        field_el = ET.SubElement(root_el, name_map['field'])
        field_el.text = FAKER.catch_phrase()
        
        # The list of values
        items = [FAKER.word() for _ in range(num_items)]
        for item_text in items:
            item_el = ET.SubElement(root_el, name_map['item'])
            item_el.text = item_text
            
        json_dict = {name_map['root']: {name_map['field']: field_el.text, name_map['item']: items}}
        return self._build_output(root_el, json_dict)

# --- Main Orchestration Function (Slightly modified to handle focused formats) ---

def generate_dataset(
    total_examples: int,
    distribution: Tuple[float, float, float] = (0.33, 0.34, 0.33),
    exclude_formats: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    use_structured_names: bool = False,
    curriculum: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate a mixed dataset of XML→JSON pairs, optionally ordered by difficulty.

    Args:
      total_examples: total number of examples
      distribution: fractions for (easy, medium, hard)
      exclude_formats: any format IDs to skip
      random_seed: for reproducibility
      use_structured_names: toggle fixed tag vocabulary
      curriculum: if True, outputs examples in easy→medium→hard blocks
    """
    if random_seed is not None:
        random.seed(random_seed)
        FAKER.seed_instance(random_seed)

    # Initialize generators with structured_names flag
    all_generators = {
        'PropertyList': PropertyListFormat(structured_names=use_structured_names),
        'SimplifiedListOfComplexEntities': SimplifiedListOfComplexEntitiesFormat(structured_names=use_structured_names),
        'ObjectWithList': ObjectWithListFormat(structured_names=use_structured_names),
        'ListOfLists': ListOfListsFormat(structured_names=use_structured_names),
        'SingleField': SingleFieldFormat(structured_names=use_structured_names),
        'SimpleList': SimpleListFormat(structured_names=use_structured_names),
        'MixedFields': MixedFieldsFormat(structured_names=use_structured_names)
    }
    if exclude_formats:
        for fmt in exclude_formats:
            all_generators.pop(fmt, None)

    # Prepare difficulty buckets
    counts = {
        'easy':   int(total_examples * distribution[0]),
        'medium': int(total_examples * distribution[1]),
        'hard':   total_examples - int(total_examples * distribution[0]) - int(total_examples * distribution[1])
    }
    buckets = {}
    for diff in counts:
        formats = [f for f in (TRAINING_FORMATS + [TEST_FORMAT]) if f in all_generators]
        buckets[diff] = formats

    dataset: List[Dict[str, Any]] = []

    # Generate examples
    order = ['easy', 'medium', 'hard'] if curriculum else list(counts.keys())
    for diff in order:
        print(f"Generating {counts[diff]} '{diff}' examples")
        examples = []
        for _ in range(counts[diff]):
            fmt = random.choice(buckets[diff])
            examples.append(all_generators[fmt].generate_example(difficulty=diff))
        if curriculum:
            random.shuffle(examples)
        dataset.extend(examples)

    if not curriculum:
        random.shuffle(dataset)

    return {'dataset': dataset}

# --- NEW Experimental Procedure ---
if __name__ == '__main__':
    # === Configuration for the Experiment ===
    
    print("--- SCALED-DOWN EXPERIMENT: 'LEARNING NESTED LISTS' ---")
    print(f"Training Formats: {TRAINING_FORMATS}")
    print(f"Held-out Test Format: ['{TEST_FORMAT}']")
    print("-" * 55)

    # === 1. Generate the TRAINING Dataset ===
    print(f"\n>>> STEP 1: Generating {N_EXAMPLES} training examples...")
    training_dataset = generate_dataset(
        total_examples=N_EXAMPLES,
        distribution=DIFFICULTY_DISTRIBUTION,
        exclude_formats=None, # Exclude the held-out test format
        random_seed=RANDOM_SEED,
        use_structured_names=FIXED_GRAMMER_FOR_STRUCTURE,
        curriculum=CURRICULUM_LEARNING
    )
    
    if FIXED_GRAMMER_FOR_STRUCTURE:
        fixed_grammer = 'grammar'
    else:
        fixed_grammer = 'uuid'
    # --- Save Training Dataset ---
    training_filename = f'/Users/maadi5/nlp_finetuning/dataset_utils/tagged_dataset_exp_{EXPERIMENT_TO_RUN}_{N_EXAMPLES}_{fixed_grammer}_curriculum.json'
    with open(training_filename, 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, indent=2)
    print(f"\nTraining dataset with {len(training_dataset['dataset'])} examples saved to '{training_filename}'")

    # # === 2. Generate the HELD-OUT TEST Dataset ===
    # print(f"\n>>> STEP 2: Generating {N_TEST_EXAMPLES} held-out test examples...")
    # # To generate *only* the test format, we exclude the training formats.
    # # We use a different seed to ensure the test set is genuinely different.
    # test_dataset = generate_dataset(
    #     total_examples=N_TEST_EXAMPLES,
    #     distribution=DIFFICULTY_DISTRIBUTION,
    #     exclude_formats=None, 
    #     random_seed=RANDOM_SEED + 1 # Use a different seed for the test set
    # )

    # # --- Save Test Dataset ---
    # test_filename = f'test_dataset_held_out_{N_TEST_EXAMPLES}.json'
    # with open(test_filename, 'w', encoding='utf-8') as f:
    #     json.dump(test_dataset, f, indent=2)
    # print(f"\nHeld-out test dataset with {len(test_dataset['dataset'])} examples saved to '{test_filename}'")

    # === 3. Verify by printing a sample from each dataset ===
    print("\n" + "-" * 55)
    print("--- Sample from TRAINING Dataset ---")
    sample_train = random.choice(training_dataset['dataset'])
    print(f"Format ID: {sample_train['format_id']}")
    print("Question (XML):\n" + sample_train['question'])
    print("Answer (JSON):\n" + json.dumps(json.loads(sample_train['answer']), indent=2))

    # print("\n--- Sample from HELD-OUT TEST Dataset ---")
    # sample_test = random.choice(test_dataset['dataset'])
    # print(f"Format ID: {sample_test['format_id']}")
    # print("Question (XML):\n" + sample_test['question'])
    # print("Answer (JSON):\n" + json.dumps(json.loads(sample_test['answer']), indent=2))