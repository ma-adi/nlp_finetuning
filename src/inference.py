# inference_client.py
import requests
import json

def run_inference(input_text, server_url="http://127.0.0.1:5000/infer"):
    """
    Sends the input text to the local inference server and returns the output.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {'input_text': input_text}

    print(f"Sending inference request to {server_url}...")
    try:
        response = requests.post(server_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        result = response.json()
        if "output" in result:
            return result["output"]
        elif "error" in result:
            print(f"Server returned an error: {result['error']}")
            return None
        else:
            print(f"Unexpected response from server: {result}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the server at {server_url}.")
        print("Please ensure 'app_server.py' is running.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        return None

if __name__ == "__main__":

#     input_text = \
# '''
# <keyword_list>
#   <keyword>report</keyword>
#   <keyword>company</keyword>
#   <keyword>service</keyword>
#   <keyword>system</keyword>
#   <keyword>data</keyword>
#   <keyword>analysis</keyword>
#   <keyword>market</keyword>
#   <keyword>customer</keyword>
#   <keyword>project</keyword>
#   <keyword>team</keyword>
#   <keyword>customer</keyword>
#   <keyword>project</keyword>
#   <keyword>team</keyword>
# </keyword_list>
# '''


#     input_text = \
# '''
# <root>
#   <entity>
#    <name>Karthee</name>
#   </entity>
#   <entity>
#    <name>Magic</name>
#   </entity>
#   <entity>
#    <name>Gautaman</name>
#   </entity>
# </root>
# '''

#     input_text = \
# '''
# <root>
#   <entity>
#    <name>Karthee</name>
#   </entity>
#   <entity>
#    <name>Magic</name>
#   </entity>
#   <entity>
#    <name>Gautaman</name>
#   </entity>
#   <entity>
#    <name>Dom</name>
#   </entity>
#   <entity>
#    <name>Sriram</name>
#   </entity>
#   <entity>
#    <name>Gopal</name>
#   </entity>
#   <entity>
#    <name>Vishnu</name>
#   </entity>
#   <entity>
#    <name>Arun</name>
#   </entity>
#   <entity>
#    <name>Bala</name>
#   </entity>
#   <entity>
#    <name>Chandru</name>
#   </entity>
#   <entity>
#    <name>Deepak</name>
#   </entity>
#   <entity>
#    <name>Elan</name>
#   </entity>
#   <entity>
#    <name>Farook</name>
#   </entity>
#   <entity>
#    <name>Guhan</name>
#   </entity>
#   <entity>
#    <name>Harish</name>
#   </entity>
#   <entity>
#    <name>Imran</name>
#   </entity>
#   <entity>
#    <name>Jay</name>
#   </entity>
#   <entity>
#    <name>Kavin</name>
#   </entity>
#   <entity>
#    <name>Lakshman</name>
#   </entity>
#   <entity>
#    <name>Manoj</name>
#   </entity>
#   <entity>
#    <name>Naveen</name>
#   </entity>
#   <entity>
#    <name>Om</name>
#   </entity>
#   <entity>
#    <name>Pranav</name>
#   </entity>
#   <entity>
#    <name>Quadir</name>
#   </entity>
#   <entity>
#    <name>Ravi</name>
#   </entity>
#   <entity>
#    <name>Surya</name>
#   </entity>
#   <entity>
#    <name>Thiru</name>
#   </entity>
#   <entity>
#    <name>Uday</name>
#   </entity>
#   <entity>
#    <name>Vicky</name>
#   </entity>
#   <entity>
#    <name>Waseem</name>
#   </entity>
#   <entity>
#    <name>Xavier</name>
#   </entity>
#   <entity>
#    <name>Yash</name>
#   </entity>
#   <entity>
#    <name>Zakir</name>
#   </entity>
#   <entity>
#    <name>Ashwin</name>
#   </entity>
#   <entity>
#    <name>Bharath</name>
#   </entity>
#   <entity>
#    <name>Cyril</name>
#   </entity>
#   <entity>
#    <name>Dinesh</name>
#   </entity>
#   <entity>
#    <name>Eshan</name>
#   </entity>
#   <entity>
#    <name>Franklin</name>
#   </entity>
#   <entity>
#    <name>Ganesh</name>
#   </entity>
# </root>
# '''

#     input_text = \
# '''
# <root>
#   <entity>
#    <name>Karthee</name>
#   </entity>
#   <entity>
#    <name>Magic</name>
#   </entity>
#   <entity>
#    <name>Gautaman</name>
#   </entity>
#   <entity>
#    <name>Dom</name>
#   </entity>
#   <entity>
#    <name>Sriram</name>
#   </entity>
#   <entity>
#    <name>Gopal</name>
#   </entity>
#   <entity>
#    <name>Vishnu</name>
#   </entity>
#   <entity>
#    <name>Arun</name>
#   </entity>
#   <entity>
#    <name>Bala</name>
#   </entity>
#   <entity>
#    <name>Chandru</name>
#   </entity>
#   <entity>
#    <name>Deepak</name>
#   </entity>
#   <entity>
#    <name>Elan</name>
#   </entity>
#   <entity>
#    <name>Farook</name>
#   </entity>
# </root>
# '''

#     input_text = \
# '''
# <root>
#   <entity>
#    <name>Karthee</name>
#    <name>Gautaman</name>
#    <name>Adithya</name>
#    <name>Dom</name>
#   </entity>
# </root>
# '''

    input_text=\
'''
<dashboard _.fcp.AccessibleZoneTabOrder.true...enable-sort-zone-taborder='true' name='Area_context_filter'>
    <style />
    <size maxheight='800' maxwidth='1000' minheight='800' minwidth='1000' />
    <datasources>
    <datasource caption='Orders (Super_Store_Sales)' name='federated.01m8s430ttzqwp11ntkqx1t7bri8' />
    </datasources>
    <datasource-dependencies datasource='federated.01m8s430ttzqwp11ntkqx1t7bri8'>
    <column datatype='string' name='[Category]' role='dimension' type='nominal' />
    <column caption='Sub Category' datatype='string' name='[Sub_Category]' role='dimension' type='nominal' />
    <column-instance column='[Category]' derivation='None' name='[none:Category:nk]' pivot='key' type='nominal' />
    <column-instance column='[Sub_Category]' derivation='None' name='[none:Sub_Category:nk]' pivot='key' type='nominal' />
    </datasource-dependencies>
    <zones>
    <zone h='100000' id='4' type-v2='layout-basic' w='100000' x='0' y='0'>
        <zone h='98000' id='7' param='horz' type-v2='layout-flow' w='98400' x='800' y='1000'>
        <zone h='98000' id='5' type-v2='layout-basic' w='82400' x='800' y='1000'>
            <zone h='98000' id='3' name='Simple_area_context_filter' w='82400' x='800' y='1000'>
            <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
            </zone-style>
            </zone>
        </zone>
        <zone fixed-size='160' h='98000' id='6' is-fixed='true' param='vert' type-v2='layout-flow' w='16000' x='83200' y='1000'>
            <zone h='56250' id='8' name='Simple_area_context_filter' param='[federated.01m8s430ttzqwp11ntkqx1t7bri8].[none:Sub_Category:nk]' type-v2='filter' w='16000' x='83200' y='1000'>
            <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
            </zone-style>
            </zone>
            <zone h='16000' id='9' name='Simple_area_context_filter' param='[federated.01m8s430ttzqwp11ntkqx1t7bri8].[none:Category:nk]' type-v2='filter' w='16000' x='83200' y='57250'>
            <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
            </zone-style>
            </zone>
        </zone>
        </zone>
        <zone-style>
        <format attr='border-color' value='#000000' />
        <format attr='border-style' value='none' />
        <format attr='border-width' value='0' />
        <format attr='margin' value='8' />
        </zone-style>
    </zone>
    </zones>
    <devicelayouts>
    <devicelayout auto-generated='true' name='Phone'>
        <size maxheight='700' minheight='700' sizing-mode='vscroll' />
        <zones>
        <zone h='100000' id='11' type-v2='layout-basic' w='100000' x='0' y='0'>
            <zone h='98000' id='10' param='vert' type-v2='layout-flow' w='98400' x='800' y='1000'>
            <zone h='56250' id='8' mode='checkdropdown' name='Simple_area_context_filter' param='[federated.01m8s430ttzqwp11ntkqx1t7bri8].[none:Sub_Category:nk]' type-v2='filter' w='16000' x='83200' y='1000'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone h='16000' id='9' mode='checkdropdown' name='Simple_area_context_filter' param='[federated.01m8s430ttzqwp11ntkqx1t7bri8].[none:Category:nk]' type-v2='filter' w='16000' x='83200' y='57250'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone fixed-size='280' h='98000' id='3' is-fixed='true' name='Simple_area_context_filter' w='82400' x='800' y='1000'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            </zone>
            <zone-style>
            <format attr='border-color' value='#000000' />
            <format attr='border-style' value='none' />
            <format attr='border-width' value='0' />
            <format attr='margin' value='8' />
            </zone-style>
        </zone>
        </zones>
    </devicelayout>
    </devicelayouts>
    <simple-id uuid='{2D1B3BF2-337D-4CC5-8B7B-007CBBACE9BA}' />
</dashboard>
'''



    output = run_inference(input_text)

    if output is not None:
        print("\n--- Inference Output ---")
        print(output)
        # If the output is JSON, you might want to parse and pretty print it
        try:
            # parsed_output = json.loads(output)
            print("\n--- Parsed JSON Output (Pretty Printed) ---")
            print(json.dumps(output, indent=2))
        except json.JSONDecodeError:
            print("\nOutput is not valid JSON, printing as-is.")