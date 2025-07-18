# client.py
import requests
import json

# API server URL
API_URL = "http://localhost:8000/generate"

def inference_model(prompt: str, max_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.95, stop: list[str] = None):
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop if stop is not None else []
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to the API server. Is it running?"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    

def to_instruct_inference(context, question):
    instr = question
    # build the single-text prompt+response
    prompt = (
        "### Instruction:\n" + context + "\n"
        + (f"### Context:\n{instr}\n" if instr else False)
    )
    return prompt

if __name__ == "__main__":

    context_message = '''Your goal is to help users with format transformation (from XML to JSON). 
    The specific goal in question is to convert an input XML into sqsh-internal JSON format.
    Think through the XML structure breakdown and how it reconstructs back into the sqsh-internal schema.
    Critical note: Think only about the actual schema/structure while doing the reconstruction. Once you do that, then relay the actual content (text,etc) appropriately.'''


    input_xml = \
'''
<employeeList>
  <employee>
    <fullName>Ashley Carter</fullName>
  </employee>
  <employee>
    <fullName>Daniel Rivera</fullName>
  </employee>
  <employee>
    <fullName>Emily Brooks</fullName>
  </employee>
  <employee>
    <fullName>Jordan Bennett</fullName>
  </employee>
  <employee>
    <fullName>Rachel Flores</fullName>
  </employee>
  <employee>
    <fullName>Tyler Jameson</fullName>
  </employee>
  <employee>
    <fullName>Nina Patel</fullName>
  </employee>
  <employee>
    <fullName>Nina abc</fullName>
  </employee>
  <employee>
    <fullName>Nina def</fullName>
  </employee>
  <employee>
    <fullName>Nina ghi</fullName>
  </employee>
  <employee>
    <fullName>Nina jkl</fullName>
  </employee>
  <employee>
    <fullName>Nina mno</fullName>
  </employee>
  <employee>
    <fullName>Nina pqr</fullName>
  </employee>
  <employee>
    <fullName>Nina stu</fullName>
  </employee>
  <employee>
    <fullName>Nina vwx</fullName>
  </employee>
  <employee>
    <fullName>Nina yz1</fullName>
  </employee>
  <employee>
    <fullName>Nina 123</fullName>
  </employee>
  <employee>
    <fullName>Nina 456</fullName>
  </employee>
  <employee>
    <fullName>Nina 789</fullName>
  </employee>
  <employee>
    <fullName>Nina 101112</fullName>
  </employee>
  <employee>
    <fullName>Nina 131415</fullName>
  </employee>
  <employee>
    <fullName>Nina 161718</fullName>
  </employee>
  <employee>
    <fullName>Nina 192021</fullName>
  </employee>
  <employee>
    <fullName>Nina 222324</fullName>
  </employee>
  <employee>
    <fullName>Nina 252627</fullName>
  </employee>
  <employee>
    <fullName>Nina 282930</fullName>
  </employee>
  <employee>
    <fullName>Chris Walters</fullName>
  </employee>
  <employee>
    <fullName>Sophia Martinez</fullName>
  </employee>
  <employee>
    <fullName>Michael Chen</fullName>
  </employee>
  <employee>
    <fullName>Linda Gomez</fullName>
  </employee>
  <employee>
    <fullName>Kevin O'Neil</fullName>
  </employee>
  <employee>
    <fullName>Jessica Nguyen</fullName>
  </employee>
  <employee>
    <fullName>David Kim</fullName>
  </employee>
</employeeList>
'''

    input_xml = 'Hi'
    # '''
    # <6a1017b2>
    # <abc>
    #     <f9beb562>prevent</f9beb562>
    #     <f9beb562>lot</f9beb562>
    #     <f9beb562>at</f9beb562>
    #     <f9beb562>bring</f9beb562>
    # </abc>
    # <def>
    #     <list_cat2>think</list_cat2>
    #     <list_cat2>west</list_cat2>
    #     <list_cat2>money</list_cat2>
    # </def>
    # <ghi>
    #     <list_cat3>picture</list_cat3>
    #     <list_cat3>north</list_cat3>
    #     <list_cat3>admit</list_cat3>
    #     <list_cat3>project</list_cat3>
    # </ghi>
    # </6a1017b2>
    # '''

    user_prompt = to_instruct_inference(context= context_message, question = input_xml)


    response_data = inference_model(
        prompt=user_prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        stop=["\nUser:", "<|im_end|>"] # Example stop sequences
    )

    if "generated_text" in response_data:
        print("\nGenerated Text:")
        print(response_data["generated_text"].strip())
    else:
        print("\nError during inference:")
        print(response_data.get("error", "Unknown error"))