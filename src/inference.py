from model_setup import NLPCoder
import json

def run_inference(input_text):

    model_inference = NLPCoder(model_identifier='/Users/maadi5/NLP_code_gen/format_conversion3_1_ft_weights',
                            load_fine_tuned=True)
    output = model_inference.infer(input_text)

    # token_count_1 = model_inference.count_tokens(input_text)
    # tokens_1 = model_inference.get_tokens(input_text)
    
    # print(f"\nInput Text: '{input_text}'")
    # print(f"Number of Tokens: {token_count_1}")
    # print(f"Tokens: {tokens_1}")

    return output
    



if __name__ == "__main__":

    input_text= \
    '''<?xml version="1.0" encoding="utf-8"?>
<datasource>
    <releation type="table" name="project_id.adit" table="users" />
    <releation type="table" name="project_id.another" table="orders" />
    <releation type="table" name="project_id.ad" table="orders" />
    <releation type="table" name="project_id.kt" table="orders" />
    <releation type="table" name="project_id.john" table="orders" />
</datasource>
'''

    output = run_inference(input_text)

    # parsed = json.loads(output)
    print(output)