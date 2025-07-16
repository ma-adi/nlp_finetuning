import json
import random
import string
import uuid
from xml.sax.saxutils import escape

# --- Configuration ---
NUM_EXAMPLES = 750
TEST_COUNT_PER_TEMPLATE = 25
OUTPUT_FILE = 'input_dataset_v7.json'

# --- NEW: Configuration for Generated Data ---
MIN_LIST_LENGTH = 2         # The minimum number of items in any list
MAX_LIST_LENGTH = 15        # The maximum number of items in any list

MIN_KEY_LEN = 3             # Min character length for an XML tag/key
MAX_KEY_LEN = 6             # Max character length for an XML tag/key

MIN_VALUE_LEN = 3           # Min character length for a string value
MAX_VALUE_LEN = 8           # Max character length for a string value

# --- Helper Functions (Corrected) ---
def random_key():
    key_len = random.randint(MIN_KEY_LEN, MAX_KEY_LEN)
    return ''.join(random.choices(string.ascii_lowercase, k=1)) + \
           ''.join(random.choices(string.ascii_lowercase + string.digits, k=key_len - 1))

def random_value():
    r = random.random()
    if r < 0.7:
        val_len = random.randint(MIN_VALUE_LEN, MAX_VALUE_LEN)
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=val_len))
    elif r < 0.9:
        return random.randint(0, 10000)
    else:
        return random.choice([True, False])

# --- Centralized XML Builder (Unchanged) ---
def _get_xml_value(py_val):
    if isinstance(py_val, bool):
        return str(py_val).lower()
    if py_val is None:
        return ""
    return str(py_val)

def build_xml_from_json(tag, data, indent_level=0):
    indent = '  ' * indent_level
    if isinstance(data, dict):
        if '#item_name' in data and '#items' in data:
            item_tag = data['#item_name']
            items_list = data['#items']
            xml = f"{indent}<{tag}>\n"
            for item in items_list:
                xml += build_xml_from_json(item_tag, item, indent_level + 1)
            xml += f"{indent}</{tag}>\n"
            return xml

        attributes = ''
        children_xml = ''
        for key, value in data.items():
            if key.startswith('@'):
                attr_val_str = _get_xml_value(value)
                attributes += f' {key[1:]}="{escape(attr_val_str)}"'
            else:
                children_xml += build_xml_from_json(key, value, indent_level + 1)
        
        xml = f'{indent}<{tag}{attributes}'
        if not children_xml:
            return xml + ' />\n'
        xml += '>\n' + children_xml + indent + f'</{tag}>\n'
        return xml

    elif isinstance(data, list):
        xml = f"{indent}<{tag}>\n"
        for item in data:
            xml += build_xml_from_json('item', item, indent_level + 1)
        xml += f"{indent}</{tag}>\n"
        return xml
        
    else:
        xml_val = escape(_get_xml_value(data))
        return f'{indent}<{tag}>{xml_val}</{tag}>\n'

def generate_example(schema_obj):
    if len(schema_obj) != 1:
        raise ValueError("Schema must have a single root element key.")
    root_tag = list(schema_obj.keys())[0]
    root_data = schema_obj[root_tag]
    xml = build_xml_from_json(root_tag, root_data)
    return xml, schema_obj

# --- Generator Functions (Updated and Diversified) ---

# --- Easy ---
def gen_easy_flat():
    return {'root': {random_key(): random_value()}}

def gen_simple_list():
    return {'root': {'items': {
        '#item_name': 'item', 
        '#items': [random_value() for _ in range(random.randint(MIN_LIST_LENGTH, MAX_LIST_LENGTH))]
    }}}

# --- Medium ---
def gen_med_nested():
    return {'root': {random_key(): {random_key(): random_value()}}}

def gen_med_obj_array_varied_id():
    # **FIX**: No longer fixated on '@id'. It now uses a variety of identifiers.
    items = []
    for _ in range(random.randint(MIN_LIST_LENGTH, MAX_LIST_LENGTH)):
        identifier_key = random.choice(['@id', '@key', '@uuid', '@name'])
        item = {'name': random_key()}
        if identifier_key == '@uuid':
            item[identifier_key] = str(uuid.uuid4())
        else:
            item[identifier_key] = random_value()
        items.append(item)
        
    return {'root': {'users': {
        '#item_name': 'user',
        '#items': items
    }}}

# --- Hard ---
def gen_hard_deep_varied_id():
    # **FIX**: Diversified identifiers and structure.
    groups_list = []
    for _ in range(random.randint(1, 3)):
        num_users = random.randint(MIN_LIST_LENGTH, MAX_LIST_LENGTH)
        users_list = []
        for _ in range(num_users):
            identifier_key = random.choice(['@id', '@key', '@uuid'])
            user_obj = {'name': random_key()}
            user_obj[identifier_key] = str(uuid.uuid4()) if identifier_key == '@uuid' else random_value()
            users_list.append(user_obj)

        group_id_key = random.choice(['@id', '@name'])
        group_obj = {
            group_id_key: random_key(),
            'users': { '#item_name': 'user', '#items': users_list }
        }
        groups_list.append(group_obj)
        
    return {'root': {'groups': {
        '#item_name': 'group',
        '#items': groups_list
    }}}

# --- NEW Generators to Fix Model Confusion ---

def gen_flat_list_of_objects():
    """
    **NEW**: Teaches the model the pattern where the root element IS the list container.
    This directly addresses the <datasource> problem.
    e.g., <releations><releation .../><releation .../></releations>
    """
    item_tag = random_key()
    container_tag = item_tag + 's' # e.g., 'item' -> 'items'
    items = []
    for _ in range(random.randint(MIN_LIST_LENGTH, MAX_LIST_LENGTH)):
        items.append({
            '@name': f"project_id.{random_key()}",
            '@type': random.choice(['table', 'join', 'view']),
            '@status': random.choice(['active', 'inactive'])
        })
    return {container_tag: {
        '#item_name': item_tag,
        '#items': items
    }}

def gen_diverse_attributes():
    """
    **NEW**: Teaches the model to handle objects with multiple, varied attributes
    and not just a single ID.
    """
    return {'root': {
        random_key(): {
            '@status': random.choice(['ok', 'error', 'pending']),
            '@priority': random.randint(1, 5),
            '@enabled': random.choice([True, False]),
            '@name': random_key()
        }
    }}

# --- Test Templates (Unchanged, still valuable) ---
def gen_test_mixed_content():
    return {'root': {'file': { '@id': random.randint(100, 999), '@active': True, 'value': f"{random_key()}.txt"}}}
def gen_test_optional_list():
    data_obj = {'@id': random_value()}
    if random.random() > 0.5:
        data_obj['notes'] = {'#item_name': 'note', '#items': [random_value() for _ in range(random.randint(MIN_LIST_LENGTH, MAX_LIST_LENGTH))]}
    return {'root': {'data': data_obj}}
def gen_test_empty_object():
    return {'root': {}}
def gen_test_empty_tag_from_null():
    return {'root': {'config': None}}
def gen_test_self_closing_from_empty_dict():
    return {'root': {'meta': {}}}
def gen_test_boolean_attribute():
    return {'root': {'status': {'@active': random.choice([True, False])}}}

# --- Assemble Dataset ---
if __name__ == "__main__":
    # **UPDATED**: Added the new, more robust generators to the pool.
    generators = {
        'easy': [gen_easy_flat, gen_simple_list],
        'medium': [gen_med_nested, gen_med_obj_array_varied_id, gen_diverse_attributes],
        'hard': [gen_hard_deep_varied_id, gen_flat_list_of_objects],
    }
    
    test_generators = [
        gen_test_mixed_content,
        gen_test_optional_list,
        gen_test_empty_object,
        gen_test_empty_tag_from_null,
        gen_test_self_closing_from_empty_dict,
        gen_test_boolean_attribute
    ]

    dataset = []
    
    num_main_examples = NUM_EXAMPLES - (len(test_generators) * TEST_COUNT_PER_TEMPLATE)
    bucket_names = list(generators.keys())
    # Ensure at least one example per generator function
    num_per_bucket = {name: len(gens) for name, gens in generators.items()}
    total_gens = sum(num_per_bucket.values())
    if num_main_examples < total_gens:
        raise ValueError(f"NUM_EXAMPLES is too small. Need at least {total_gens + len(test_generators) * TEST_COUNT_PER_TEMPLATE}")

    # Generate examples more evenly across all generators
    all_gens = [gen for gens in generators.values() for gen in gens]
    for i in range(num_main_examples):
        schema = random.choice(all_gens)()
        xml, obj = generate_example(schema)
        dataset.append({'question': xml, 'answer': json.dumps(obj, ensure_ascii=False)})

    # Generate test examples
    for fn in test_generators:
        for _ in range(TEST_COUNT_PER_TEMPLATE):
            schema = fn()
            xml, obj = generate_example(schema)
            dataset.append({'question': xml, 'answer': json.dumps(obj, ensure_ascii=False)})

    random.shuffle(dataset)
    with open(OUTPUT_FILE, 'w', encoding='utf8') as f:
        json.dump({'dataset': dataset}, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(dataset)} examples for a robust dataset and saved to {OUTPUT_FILE}")