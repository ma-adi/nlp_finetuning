import json
import random
import string

# --- Configuration ---
MAX_LIST_LENGTH = 20
MAX_COMPLEXITY_FACTOR = 7

# --- Helper Functions ---

def random_key():
    """Generates a random lowercase string to be used as an XML tag or JSON key."""
    return ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 6)))

def random_value():
    """Generates a random alphanumeric string to be used as a value."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(3, 8)))

# NEW: Helper function to generate list lengths skewed towards the maximum.
def get_skewed_random_length(min_val, max_val):
    """
    Generates a random integer between min_val and max_val.
    The distribution is triangular, heavily biased towards max_val, ensuring
    a healthy number of examples with long lists are generated.
    """
    return int(random.triangular(min_val, max_val, max_val))

# --- Template Generators (Logic Untouched, only random length generation is improved) ---

def gen_flat_object():
    # This generator does not create lists, so it remains unchanged.
    keys = {random_key() for _ in range(random.randint(2, 5))}
    xml = "<root>\n"
    obj = {}
    for k in keys:
        v = random_value()
        xml += f"  <{k}>{v}</{k}>\n"
        obj[k] = v
    xml += "</root>"
    return xml, obj

def gen_simple_list():
    """Generates a list of simple string values, using the skewed length generator."""
    # UPDATED to use the new skewed random function
    count = get_skewed_random_length(2, MAX_LIST_LENGTH)
    list_name = random_key()
    item_name = random_key()
    xml = f"<root>\n  <{list_name}>\n"
    arr = []
    for _ in range(count):
        v = random_value()
        xml += f"    <{item_name}>{v}</{item_name}>\n"
        arr.append(v)
    xml += f"  </{list_name}>\n</root>"
    return xml, {list_name: arr}

def gen_singleton_list():
    # This generator is for a single item, so it remains unchanged.
    list_name = random_key()
    item_name = random_key()
    v = random_value()
    xml = f"<root>\n  <{list_name}>\n    <{item_name}>{v}</{item_name}>\n  </{list_name}>\n</root>"
    return xml, {list_name: [v]}

def gen_nested_object():
    # This generator does not create lists, so it remains unchanged.
    parent = random_key()
    children = {random_key() for _ in range(random.randint(2, 4))}
    xml = f"<root>\n  <{parent}>\n"
    obj = {parent: {}}
    for c in children:
        v = random_value()
        xml += f"    <{c}>{v}</{c}>\n"
        obj[parent][c] = v
    xml += f"  </{parent}>\n</root>"
    return xml, obj

def gen_list_of_objects():
    """Generates a list of objects, using the skewed length generator."""
    # UPDATED to use the new skewed random function
    count = get_skewed_random_length(2, MAX_LIST_LENGTH)
    xml = "<root>\n  <items>\n"
    arr = []
    keys = {random_key() for _ in range(random.randint(2, 3))}
    for _ in range(count):
        xml += "    <item>\n"
        item_obj = {}
        for prop in keys:
            v = random_value()
            xml += f"      <{prop}>{v}</{prop}>\n"
            item_obj[prop] = v
        xml += "    </item>\n"
        arr.append(item_obj)
    xml += "  </items>\n</root>"
    return xml, {"items": arr}

def gen_mixed_siblings_with_list():
    """Generates an object with mixed simple children and a list, using skewed length."""
    meta_val = random_value()
    xml = f"<root>\n  <data>\n    <timestamp>{meta_val}</timestamp>\n"
    points = []
    # UPDATED to use the new skewed random function
    for _ in range(get_skewed_random_length(3, MAX_LIST_LENGTH)):
        p = str(random.randint(1, 100))
        xml += f"    <point>{p}</point>\n"
        points.append(p)
    xml += "  </data>\n</root>"
    return xml, {"data": {"timestamp": meta_val, "point": points}}

# --- All other generators remain exactly as they were in the previous version ---
def gen_attributes_and_children():
    pid = random_value()
    name = random_value()
    city = random_value()
    xml = f'<root>\n  <person id="{pid}">\n    <name>{name}</name>\n    <city>{city}</city>\n  </person>\n</root>'
    return xml, {"person": {"id": pid, "name": name, "city": city}}

def gen_attributes_and_text():
    sku = random_value()
    name = " ".join([random_value(), random_value()])
    xml = f'<root>\n  <product sku="{sku}">{name}</product>\n</root>'
    return xml, {"product": {"sku": sku, "#text": name}}

def gen_self_closing_with_attributes():
    xml = f'<root>\n  <link type="external" href="http://{random_value()}.com" />\n</root>'
    href_val = xml.split('href="')[1].split('"')[0]
    return xml, {"link": {"type": "external", "href": href_val}}

def gen_deep_nesting_mixed():
    org_id = random_value()
    dep_name = random_key()
    xml = f'<root>\n  <organization id="{org_id}">\n    <department name="{dep_name}">\n      <teams>\n'
    teams = []
    for i in range(random.randint(2, 4)):
        lead_name = random_value()
        team_obj = {"lead": lead_name, "member": []}
        xml += f'        <team lead="{lead_name}">\n'
        for j in range(random.randint(2, 5)):
            member_name = random_value()
            team_obj["member"].append(member_name)
            xml += f'          <member>{member_name}</member>\n'
        xml += '        </team>\n'
        teams.append(team_obj)
    xml += '      </teams>\n    </department>\n  </organization>\n</root>'
    obj = {"organization": {"id": org_id, "department": {"name": dep_name, "teams": {"team": teams}}}}
    return xml, obj

def gen_hard_deep_and_wide_safely():
    xml = "<root>\n  <reports>\n"
    reports_list = []
    for _ in range(get_skewed_random_length(2, MAX_COMPLEXITY_FACTOR)):
        report_id = random_value()
        report_obj = {"id": report_id, "section": []}
        xml += f'    <report id="{report_id}">\n'
        for _ in range(get_skewed_random_length(2, MAX_COMPLEXITY_FACTOR - 1)):
            section_title = random_key()
            section_obj = {"title": section_title, "entry": []}
            xml += f'      <section title="{section_title}">\n'
            for _ in range(get_skewed_random_length(1, MAX_COMPLEXITY_FACTOR - 2)):
                entry_val = random_value()
                section_obj["entry"].append(entry_val)
                xml += f'        <entry>{entry_val}</entry>\n'
            xml += '      </section>\n'
            report_obj["section"].append(section_obj)
        xml += '    </report>\n'
        reports_list.append(report_obj)
    xml += "  </reports>\n</root>"
    return xml, {"reports": {"report": reports_list}}

def generate_dataset(num_examples=300):
    all_generators = [
        gen_flat_object, gen_simple_list, gen_singleton_list, gen_nested_object,
        gen_list_of_objects, gen_attributes_and_children, gen_attributes_and_text,
        gen_self_closing_with_attributes, gen_mixed_siblings_with_list,
        gen_deep_nesting_mixed, gen_hard_deep_and_wide_safely,
    ]
    dataset_entries = []
    for _ in range(num_examples):
        generator_func = random.choice(all_generators)
        xml_question, py_obj_answer = generator_func()
        entry = {"question": xml_question, "answer": json.dumps(py_obj_answer, ensure_ascii=False)}
        dataset_entries.append(entry)
    return {"dataset": dataset_entries}

if __name__ == "__main__":
    dataset = generate_dataset(num_examples=750)
    output_filename = 'xml_to_json_dataset.json'
    with open(output_filename, 'w', encoding='utf8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Generated dataset with {len(dataset['dataset'])} examples.")
    print(f"Saved to '{output_filename}'.")