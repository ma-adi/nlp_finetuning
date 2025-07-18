from model_setup import NLPCoder
import json

def run_inference(input_text):

    model_inference = NLPCoder(model_identifier='/Users/maadi5/nlp_finetuning/format_conversion_tagged_weights_exp1',
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
  <employee>
    <fullName>Alicia Reed</fullName>
  </employee>
  <employee>
    <fullName>Tommy Blake</fullName>
  </employee>
  <employee>
    <fullName>Monica Allen</fullName>
  </employee>
</employeeList>
'''

    output = run_inference(input_text)

    # parsed = json.loads(output)
    print(output)