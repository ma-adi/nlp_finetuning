import json
import random
import string

# NEW: A variable to control the maximum size of generated examples
# This will affect the number of keys, list items, and nesting depth.
MAX_LENGTH = 20

# Helper functions
def random_key():
    return ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 6)))

def random_value():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(3, 8)))

# --- Templates for training formats (Updated with consistent rules) ---

# Rule being applied: Flat key-value pairs
def gen_easy_flat():
    keys = [random_key() for _ in range(random.randint(2, MAX_LENGTH))]
    xml = "<root>\n"
    obj = {}
    for k in keys:
        v = random_value()
        xml += f"  <{k}>{v}</{k}>\n"
        obj[k] = v
    xml += "</root>"
    return xml, obj

# Rule being applied: Repeating sibling tags become an array.
def gen_easy_list():
    count = random.randint(2, MAX_LENGTH)
    xml = "<root>\n"
    arr = []
    for _ in range(count):
        v = random_value()
        xml += f"  <item>{v}</item>\n"
        arr.append(v)
    xml += "</root>"
    return xml, {"item": arr}

# Medium complexity
def gen_med_nested():
    parent = random_key()
    children = [random_key() for _ in range(random.randint(2, MAX_LENGTH - 1))]
    xml = "<root>\n"
    xml += f"  <{parent}>\n"
    obj = {parent: {}}
    for c in children:
        v = random_value()
        xml += f"    <{c}>{v}</{c}>\n"
        obj[parent][c] = v
    xml += f"  </{parent}>\n</root>"
    return xml, obj

# Rule being applied: Repeating sibling tags become an array.
def gen_med_obj_array():
    count = random.randint(2, MAX_LENGTH)
    xml = "<root>\n"
    arr = []
    for _ in range(count):
        xml += "  <item>\n"
        obj = {}
        for prop in ("id", "value"):
            v = random_value()
            xml += f"    <{prop}>{v}</{prop}>\n"
            obj[prop] = v
        xml += "  </item>\n"
        arr.append(obj)
    xml += "</root>"
    return xml, {"item": arr}

# Hard complexity
# Rules: Attributes + text use '#text'. Repeating tags become an array.
def gen_hard_attr():
    xml = "<root>\n"
    pid = random_value()
    name = random_value()
    xml += f'  <person id="{pid}">{name}</person>\n'
    tags = [random_value() for _ in range(random.randint(2, MAX_LENGTH))]
    for t in tags:
        xml += f'  <tag value="{t}" />\n'
    xml += "</root>"
    json_obj = {
        "person": {"id": pid, "#text": name},
        "tag": [{"value": t} for t in tags]
    }
    return xml, json_obj

# Rule being applied: Repeating sibling tags become an array.
def gen_hard_deep():
    xml = "<root>\n"
    groups = []
    for _ in range(random.randint(1, MAX_LENGTH // 2)):
        xml += "    <group>\n"
        users = []
        for _ in range(random.randint(1, MAX_LENGTH // 2)):
            uname = random_value()
            xml += f"        <user><name>{uname}</name></user>\n"
            users.append({"name": uname})
        xml += "    </group>\n"
        groups.append({"user": users})
    xml += "</root>"
    return xml, {"group": groups}

# --- Test templates (hard) ---
def gen_test_singleton_array():
    v = random_value()
    xml = f"""<root>
  <item>{v}</item>
</root>"""
    return xml, {"item": [v]}

def gen_test_attr_order_variation():
    pid = random_value()
    name = random_value()
    xml = f"""<root>
  <person name=\"{name}\" id=\"{pid}\"/>
</root>"""
    return xml, {"person": {"name": name, "id": pid}}

def gen_test_cdata_entity():
    raw = random_value() + " & " + random_value()
    xml = f"""<root>
  <data><![CDATA[{raw}]]></data>
  <text>Test < & ></text>
</root>"""
    return xml, {"data": raw, "text": "Test < & >"}

def gen_test_optional_element():
    val = None
    xml = "<root>\n"
    if random.random() > 0.5:
        val = random_value()
        xml += f"  <optional>{val}</optional>\n"
    xml += "</root>"
    obj = {"optional": val} if val is not None else {}
    return xml, obj

def gen_test_nested_obj_array():
    parent = random_key()
    count = random.randint(2, MAX_LENGTH)
    arr = []
    xml = f"<root>\n  <{parent}>\n"
    for _ in range(count):
        idv = random_value()
        xml += f"    <item id=\"{idv}\">{idv}</item>\n"
        arr.append({"id": idv, "#text": idv})
    xml += f"  </{parent}>\n</root>"
    return xml, {parent: {"item": arr}}

def gen_test_deep_nested_attributes():
    group_id = random_value()
    count = random.randint(1, MAX_LENGTH - 1)
    users = []
    xml = f"<root>\n  <group id=\"{group_id}\">\n"
    for _ in range(count):
        uid = random_value()
        role = random_key()
        xml += f"    <user id=\"{uid}\" role=\"{role}\" />\n"
        users.append({"id": uid, "role": role})
    xml += "  </group>\n</root>"
    return xml, {"group": {"id": group_id, "user": users}}

# Main Generator Function
def generate_dataset():
    all_examples = []
    
    # List of all generator functions
    all_generators = [
        # Easy
        gen_easy_flat, gen_easy_list,
        # Medium
        gen_med_nested, gen_med_obj_array,
        # Hard
        gen_hard_attr, gen_hard_deep,
        # Specialized Test Cases
        gen_test_singleton_array,
        gen_test_attr_order_variation,
        gen_test_optional_element,
        gen_test_nested_obj_array,
        gen_test_deep_nested_attributes
    ]

    # Generate a set number of examples for each type to ensure variety
    num_examples_per_type = 25
    
    for gen_func in all_generators:
        for _ in range(num_examples_per_type):
            xml, obj = gen_func()
            all_examples.append({
                "question": xml,
                "answer": json.dumps(obj, ensure_ascii=False)
            })

    # Shuffle the combined list to mix the different types
    random.shuffle(all_examples)

    return {"dataset": all_examples}


if __name__ == "__main__":
    dataset = generate_dataset()
    
    # Save the simplified dataset to a new file
    output_filename = 'input_dataset_v1.5.json'
    with open(output_filename, 'w', encoding='utf8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Generated simplified dataset with {len(dataset['dataset'])} examples.")
    print(f"File saved as '{output_filename}'.")