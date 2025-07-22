import random
import json

class DatasetGenerator:
    """
    Generates a dataset of examples for converting an XML employee list to JSON.

    Each example consists of an 'input' XML string and a corresponding 'output' JSON string.
    The generator introduces randomness in employee names, list sizes, and XML formatting
    to create a robust dataset for training.
    """

    FIRST_NAMES = [
        "Ashley", "Daniel", "Emily", "Jordan", "Rachel", "Tyler", "Nina", "Liam",
        "Olivia", "Noah", "Emma", "Oliver", "Ava", "Elijah", "Charlotte", "William",
        "Sophia", "James", "Amelia", "Benjamin", "Isabella", "Lucas", "Mia", "Henry",
        "Evelyn", "Alexander", "Harper", "Michael", "Abigail", "Ethan", "Madison"
    ]
    LAST_NAMES = [
        "Carter", "Rivera", "Brooks", "Bennett", "Flores", "Jameson", "Patel", "Smith",
        "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
        "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
        "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"
    ]

    def _generate_full_name(self) -> str:
        """Generates a random full name from the predefined lists."""
        first = random.choice(self.FIRST_NAMES)
        last = random.choice(self.LAST_NAMES)
        return f"{first} {last}"

    def _create_xml_input(self, employee_names: list[str]) -> str:
        """
        Creates a randomized XML string with imperfect/inconsistent indentation.

        This method randomizes indentation, newlines, and block structures to produce
        diverse and messy (but valid) XML. This forces a model to learn the tag
        hierarchy rather than relying on consistent formatting.
        """
        employee_blocks = []
        for name in employee_names:
            # Randomly choose a structure for this employee block
            style = random.choice(['compact', 'expanded', 'mixed'])
            
            # 1. Random indentation for the entire <employee> block
            base_indent = " " * random.randint(0, 8)
            
            # 2. Random newlines and internal indentation based on style
            if style == 'compact':
                # e.g., "    <employee><fullName>Nina Patel</fullName></employee>"
                block = f"{base_indent}<employee><fullName>{name}</fullName></employee>"
            elif style == 'expanded':
                # e.g., "  <employee>
                #            <fullName>Nina Patel</fullName>
                #   </employee>"
                # Note the inconsistent internal indentation
                internal_indent = " " * random.randint(1, 12)
                end_indent = " " * random.randint(0, 8)
                block = (
                    f"{base_indent}<employee>\n"
                    f"{base_indent}{internal_indent}<fullName>{name}</fullName>\n"
                    f"{end_indent}</employee>"
                )
            else: # 'mixed' style
                # e.g., " <employee><fullName>Nina Patel</fullName>
                # </employee>" (newline before closing tag with random indent)
                end_indent = " " * random.randint(0, 8)
                block = f"{base_indent}<employee><fullName>{name}</fullName>\n{end_indent}</employee>"

            employee_blocks.append(block)
        
        # 3. Join employee blocks with a random number of newlines
        body = ("\n" * random.randint(0, 2)).join(employee_blocks)
        
        # 4. Add random newlines/spacing around the main body
        start_sep = "\n" if random.random() > 0.3 else ""
        end_sep = "\n" if random.random() > 0.3 else ""
        final_indent = " " * random.randint(0, 4) # trailing whitespace

        return f"<employeeList>{start_sep}{body}{end_sep}{final_indent}</employeeList>"

    def _create_json_output(self, employee_names: list[str]) -> str:
        """Creates a compact JSON string from a list of names, matching the target format."""
        employee_list_of_dicts = [{"fullName": name} for name in employee_names]
        output_dict = {
            "employeeList": {
                "employee": employee_list_of_dicts
            }
        }
        # Use separators=(',', ':') to create a compact JSON string without whitespace
        return json.dumps(output_dict, separators=(',', ':'))

    def _create_example(self, num_employees: int) -> dict:
        """
        Generates a single input/output example pair using scoped randomness.

        It first generates the list of names, then uses that same list to create the
        corresponding XML and JSON, ensuring they are perfectly aligned.
        """
        # 1. Generate unique random names for this example
        names = set()
        while len(names) < num_employees:
            names.add(self._generate_full_name())
        employee_names = list(names)
        random.shuffle(employee_names) # Avoid any implicit ordering

        # 2. Create the XML input from the list of names
        xml_input = self._create_xml_input(employee_names)
        
        # 3. Create the JSON output from the same list of names
        json_output = self._create_json_output(employee_names)
        
        return {
            "input": xml_input,
            "output": json_output
        }

    def generate_dataset(self, num_examples: int) -> dict:
        """
        Generates a complete dataset of n examples.

        Args:
            num_examples: The total number of examples to generate in the dataset.

        Returns:
            A dictionary adhering to the specified schema:
            {'dataset': [{'input': <xml_string>, 'output': <json_string>}, ...]}
        """
        if num_examples <= 0:
            return {'dataset': []}
            
        # Define size categories for small, medium, and large lists
        size_ranges = {
            'small': (1, 3),
            'medium': (4, 8),
            'large': (9, 15)
        }
        categories = list(size_ranges.keys())
        
        examples = []
        for i in range(num_examples):
            # Cycle through categories to ensure a roughly equal distribution
            category = categories[i % len(categories)]
            min_size, max_size = size_ranges[category]
            num_employees = random.randint(min_size, max_size)
            
            example = self._create_example(num_employees)
            examples.append(example)
            
        return {'dataset': examples}

# --- Main execution block to demonstrate the generator ---
if __name__ == "__main__":
    # Instantiate the generator
    generator = DatasetGenerator()

    # Generate a dataset with 4 examples
    num_examples_to_generate = 500
    generated_dataset = generator.generate_dataset(num_examples_to_generate)

    # Print the generated dataset in a readable format
    json.dump(generated_dataset, open('dynamic_intendation.json', 'w', encoding='utf8'), ensure_ascii=False)