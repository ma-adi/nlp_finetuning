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
#   <keyword>customer</keyword>
#   <keyword>project</keyword>
#   <keyword>team</keyword>
#   <keyword>customer</keyword>
#   <keyword>project</keyword>
#   <keyword>team</keyword>
#   <keyword>customer</keyword>
#   <keyword>project</keyword>
#   <keyword>team</keyword>
#   <keyword>customer</keyword>
#   <keyword>project</keyword>
#   <keyword>team</keyword>
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
#    <name>Domnic</name>
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

#     input_text=\
# '''
# <dashboard _.fcp.AccessibleZoneTabOrder.true...enable-sort-zone-taborder='true' name='Area_context_filter'>
#     <style />
#     <size maxheight='800' maxwidth='1000' minheight='800' minwidth='1000' />
#     <datasources>
#     <datasource caption='Orders (Super_Store_Sales)' name='federated.01m8s430ttzqwp11ntkqx1t7bri8' />
#     </datasources>
#     <datasource-dependencies datasource='federated.01m8s430ttzqwp11ntkqx1t7bri8'>
#     <column datatype='string' name='[Category]' role='dimension' type='nominal' />
#     <column caption='Sub Category' datatype='string' name='[Sub_Category]' role='dimension' type='nominal' />
#     <column-instance column='[Category]' derivation='None' name='[none:Category:nk]' pivot='key' type='nominal' />
#     <column-instance column='[Sub_Category]' derivation='None' name='[none:Sub_Category:nk]' pivot='key' type='nominal' />
#     </datasource-dependencies>
#     <zones>
#     <zone h='100000' id='4' type-v2='layout-basic' w='100000' x='0' y='0'>
#         <zone h='98000' id='7' param='horz' type-v2='layout-flow' w='98400' x='800' y='1000'>
#         <zone h='98000' id='5' type-v2='layout-basic' w='82400' x='800' y='1000'>
#             <zone h='98000' id='3' name='Simple_area_context_filter' w='82400' x='800' y='1000'>
#             <zone-style>
#                 <format attr='border-color' value='#000000' />
#                 <format attr='border-style' value='none' />
#                 <format attr='border-width' value='0' />
#                 <format attr='margin' value='4' />
#             </zone-style>
#             </zone>
#         </zone>
#         <zone fixed-size='160' h='98000' id='6' is-fixed='true' param='vert' type-v2='layout-flow' w='16000' x='83200' y='1000'>
#             <zone h='56250' id='8' name='Simple_area_context_filter' param='[federated.01m8s430ttzqwp11ntkqx1t7bri8].[none:Sub_Category:nk]' type-v2='filter' w='16000' x='83200' y='1000'>
#             <zone-style>
#                 <format attr='border-color' value='#000000' />
#                 <format attr='border-style' value='none' />
#                 <format attr='border-width' value='0' />
#                 <format attr='margin' value='4' />
#             </zone-style>
#             </zone>
#             <zone h='16000' id='9' name='Simple_area_context_filter' param='[federated.01m8s430ttzqwp11ntkqx1t7bri8].[none:Category:nk]' type-v2='filter' w='16000' x='83200' y='57250'>
#             <zone-style>
#                 <format attr='border-color' value='#000000' />
#                 <format attr='border-style' value='none' />
#                 <format attr='border-width' value='0' />
#                 <format attr='margin' value='4' />
#             </zone-style>
#             </zone>
#         </zone>
#         </zone>
#         <zone-style>
#         <format attr='border-color' value='#000000' />
#         <format attr='border-style' value='none' />
#         <format attr='border-width' value='0' />
#         <format attr='margin' value='8' />
#         </zone-style>
#     </zone>
#     </zones>
#     <devicelayouts>
#     <devicelayout auto-generated='true' name='Phone'>
#         <size maxheight='700' minheight='700' sizing-mode='vscroll' />
#         <zones>
#         <zone h='100000' id='11' type-v2='layout-basic' w='100000' x='0' y='0'>
#             <zone h='98000' id='10' param='vert' type-v2='layout-flow' w='98400' x='800' y='1000'>
#             <zone h='56250' id='8' mode='checkdropdown' name='Simple_area_context_filter' param='[federated.01m8s430ttzqwp11ntkqx1t7bri8].[none:Sub_Category:nk]' type-v2='filter' w='16000' x='83200' y='1000'>
#                 <zone-style>
#                 <format attr='border-color' value='#000000' />
#                 <format attr='border-style' value='none' />
#                 <format attr='border-width' value='0' />
#                 <format attr='margin' value='4' />
#                 <format attr='padding' value='0' />
#                 </zone-style>
#             </zone>
#             <zone h='16000' id='9' mode='checkdropdown' name='Simple_area_context_filter' param='[federated.01m8s430ttzqwp11ntkqx1t7bri8].[none:Category:nk]' type-v2='filter' w='16000' x='83200' y='57250'>
#                 <zone-style>
#                 <format attr='border-color' value='#000000' />
#                 <format attr='border-style' value='none' />
#                 <format attr='border-width' value='0' />
#                 <format attr='margin' value='4' />
#                 <format attr='padding' value='0' />
#                 </zone-style>
#             </zone>
#             <zone fixed-size='280' h='98000' id='3' is-fixed='true' name='Simple_area_context_filter' w='82400' x='800' y='1000'>
#                 <zone-style>
#                 <format attr='border-color' value='#000000' />
#                 <format attr='border-style' value='none' />
#                 <format attr='border-width' value='0' />
#                 <format attr='margin' value='4' />
#                 <format attr='padding' value='0' />
#                 </zone-style>
#             </zone>
#             </zone>
#             <zone-style>
#             <format attr='border-color' value='#000000' />
#             <format attr='border-style' value='none' />
#             <format attr='border-width' value='0' />
#             <format attr='margin' value='8' />
#             </zone-style>
#         </zone>
#         </zones>
#     </devicelayout>
#     </devicelayouts>
#     <simple-id uuid='{2D1B3BF2-337D-4CC5-8B7B-007CBBACE9BA}' />
# </dashboard>
# '''

    input_text = \
'''
<dashboard _.fcp.AccessibleZoneTabOrder.true...enable-sort-zone-taborder='true' name='Area_with_distribution_band'>
    <style />
    <size maxheight='800' maxwidth='1000' minheight='800' minwidth='1000' />
    <zones>
    <zone h='100000' id='4' type-v2='layout-basic' w='100000' x='0' y='0'>
        <zone h='49000' id='3' name='Area with distribution band (table)' w='49200' x='800' y='1000'>
        <zone-style>
            <format attr='border-color' value='#000000' />
            <format attr='border-style' value='none' />
            <format attr='border-width' value='0' />
            <format attr='margin' value='4' />
        </zone-style>
        </zone>
        <zone h='49000' id='5' name='Area with distribution band (pane)' w='49200' x='50000' y='1000'>
        <zone-style>
            <format attr='border-color' value='#000000' />
            <format attr='border-style' value='none' />
            <format attr='border-width' value='0' />
            <format attr='margin' value='4' />
        </zone-style>
        </zone>
        <zone h='49000' id='6' name='Area with distribution band (cell)' w='49200' x='800' y='50000'>
        <zone-style>
            <format attr='border-color' value='#000000' />
            <format attr='border-style' value='none' />
            <format attr='border-width' value='0' />
            <format attr='margin' value='4' />
        </zone-style>
        </zone>
        <zone h='49000' id='7' name='Continuous area with distribution band (cell)' w='49200' x='50000' y='50000'>
        <zone-style>
            <format attr='border-color' value='#000000' />
            <format attr='border-style' value='none' />
            <format attr='border-width' value='0' />
            <format attr='margin' value='4' />
        </zone-style>
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
        <size maxheight='1200' minheight='1200' sizing-mode='vscroll' />
        <zones>
        <zone h='100000' id='25' type-v2='layout-basic' w='100000' x='0' y='0'>
            <zone h='98000' id='24' param='vert' type-v2='layout-flow' w='98400' x='800' y='1000'>
            <zone fixed-size='280' h='49000' id='3' is-fixed='true' name='Area with distribution band (table)' w='49200' x='800' y='1000'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone fixed-size='280' h='49000' id='5' is-fixed='true' name='Area with distribution band (pane)' w='49200' x='50000' y='1000'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone fixed-size='280' h='49000' id='6' is-fixed='true' name='Area with distribution band (cell)' w='49200' x='800' y='50000'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone fixed-size='280' h='49000' id='7' is-fixed='true' name='Continuous area with distribution band (cell)' w='49200' x='50000' y='50000'>
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
    <simple-id uuid='{492BFFBC-C43D-4D7E-A41C-FEA973E2DA1E}' />
</dashboard>
'''

#     input_text = \
# '''
# <zone>
#     <zone_val>hi</zone_val>
#     <zone_val>there</zone_val>xw
#     <zone_val_2>val<zone_val_2>xw
#     <zone_val_3>val<zone_val_3>
# </zone>
# '''
#     input_text = \
# '''
# <?robot-control command="scan_area" speed="medium"?>
# <zone>
#     <zone_val></zone_val>
# </zone>
# '''
#     input_text = \
# '''
# <style>
#   <property name="color" value="blue"/>
#   <property name="font-size" value="12px"/>
#   <property name="color" value="red"/>
# </style>
# '''

    input_text = \
'''
<style id="magic">
  <value>hello</value>
  <abc>adi</abc>
  <abc>magic</abc>
  <abc/>
</style>
'''

#     input_text = \
# '''
# <permissions>
#   <permission>read</permission>
#   <permission/>
#   <permission>write</permission>
# </permissions>
# '''

    input_text = \
'''
<dashboard enable-sort-zone-taborder='true' name='Date_parameter_dashboard'>
    <style />
    <size maxheight='800' maxwidth='1000' minheight='800' minwidth='1000' />
    <datasources>
    <datasource name='Parameters' />
    </datasources>
    <datasource-dependencies datasource='Parameters'>
    <column caption='Date Parameter' datatype='date' datatype-customized='true' name='[Boolean Parameter (copy)_85286952441368584]' param-domain-type='list' role='measure' type='quantitative' value='#2014-01-09#'>
        <calculation class='tableau' formula='#2014-01-09#' />
        <members>
        <member value='#2025-07-23#' />
        <member value='#2014-01-09#' />
        <member value='#2015-09-07#' />
        <member value='#2017-03-20#' />
        </members>
    </column>
    <column caption='Date Parameter 1' datatype='date' datatype-customized='true' default-format='L' name='[Date Parameter (copy)_85286952507097118]' param-domain-type='list' role='measure' type='quantitative' value='#2014-01-09#'>
        <calculation class='tableau' formula='#2014-01-09#' />
        <members>
        <member value='#2025-07-23#' />
        <member value='#2014-01-09#' />
        <member value='#2015-09-07#' />
        <member value='#2017-03-20#' />
        </members>
    </column>
    <column caption='Date Parameter 2' datatype='date' datatype-customized='true' default-format='*m/d/yyyy' name='[Date Parameter 1 (copy)_85286952508256287]' param-domain-type='list' role='measure' type='quantitative' value='#2014-01-09#'>
        <calculation class='tableau' formula='#2014-01-09#' />
        <members>
        <member value='#2025-07-23#' />
        <member value='#2014-01-09#' />
        <member value='#2015-09-07#' />
        <member value='#2017-03-20#' />
        </members>
    </column>
    <column caption='Date Parameter 4' datatype='date' datatype-customized='true' default-format='*yyyy/mm/dd' name='[Date Parameter 1 (copy)_85286952511352870]' param-domain-type='list' role='measure' type='quantitative' value='#2014-01-09#'>
        <calculation class='tableau' formula='#2014-01-09#' />
        <members>
        <member value='#2025-07-23#' />
        <member value='#2014-01-09#' />
        <member value='#2015-09-07#' />
        <member value='#2017-03-20#' />
        </members>
    </column>
    </datasource-dependencies>
    <zones>
    <zone h='100000' id='4' type-v2='layout-basic' w='100000' x='0' y='0'>
        <zone h='98000' id='7' param='horz' type-v2='layout-flow' w='98400' x='800' y='1000'>
        <zone h='98000' id='5' type-v2='layout-basic' w='82400' x='800' y='1000' />
        <zone fixed-size='160' h='98000' id='6' is-fixed='true' param='vert' type-v2='layout-flow' w='16000' x='83200' y='1000'>
            <zone h='7000' id='8' mode='compact' param='[Parameters].[Boolean Parameter (copy)_85286952441368584]' type-v2='paramctrl' w='16000' x='83200' y='1000'>
            <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
            </zone-style>
            </zone>
            <zone h='7000' id='10' mode='compact' param='[Parameters].[Date Parameter (copy)_85286952507097118]' type-v2='paramctrl' w='16000' x='83200' y='8000'>
            <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
            </zone-style>
            </zone>
            <zone h='7000' id='12' mode='compact' param='[Parameters].[Date Parameter 1 (copy)_85286952508256287]' type-v2='paramctrl' w='16000' x='83200' y='15000'>
            <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
            </zone-style>
            </zone>
            <zone h='7000' id='14' mode='compact' param='[Parameters].[Date Parameter 1 (copy)_85286952511352870]' type-v2='paramctrl' w='16000' x='83200' y='22000'>
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
    <zone h='45250' id='3' name='Date_parameter' w='35800' x='3200' y='3500' />
    <zone h='41000' id='9' name='Date_parameter_standard_long_date' w='38600' x='40900' y='4250' />
    <zone h='44000' id='11' name='Date_parameter_mm/dd/yyyy' w='36800' x='2200' y='52375' />
    <zone h='42375' id='13' name='Date_parameter_custom' w='40000' x='40700' y='50625' />
    </zones>
    <devicelayouts>
    <devicelayout auto-generated='true' name='Phone'>
        <size maxheight='1450' minheight='1450' sizing-mode='vscroll' />
        <zones>
        <zone h='100000' id='20' type-v2='layout-basic' w='100000' x='0' y='0'>
            <zone h='98000' id='19' param='vert' type-v2='layout-flow' w='98400' x='800' y='1000'>
            <zone h='7000' id='8' mode='compact' param='[Parameters].[Boolean Parameter (copy)_85286952441368584]' type-v2='paramctrl' w='16000' x='83200' y='1000'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone fixed-size='280' h='45250' id='3' is-fixed='true' name='Date_parameter' w='35800' x='3200' y='3500'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone fixed-size='280' h='41000' id='9' is-fixed='true' name='Date_parameter_standard_long_date' w='38600' x='40900' y='4250'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone h='7000' id='10' mode='compact' param='[Parameters].[Date Parameter (copy)_85286952507097118]' type-v2='paramctrl' w='16000' x='83200' y='8000'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone h='7000' id='12' mode='compact' param='[Parameters].[Date Parameter 1 (copy)_85286952508256287]' type-v2='paramctrl' w='16000' x='83200' y='15000'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone h='7000' id='14' mode='compact' param='[Parameters].[Date Parameter 1 (copy)_85286952511352870]' type-v2='paramctrl' w='16000' x='83200' y='22000'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone fixed-size='280' h='42375' id='13' is-fixed='true' name='Date_parameter_custom' w='40000' x='40700' y='50625'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone fixed-size='280' h='44000' id='11' is-fixed='true' name='Date_parameter_mm/dd/yyyy' w='36800' x='2200' y='52375'>
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
    <simple-id uuid='{7945DE4A-A13E-42B4-B374-526456C5FD44}' />
</dashboard>
'''


    input_text = \
'''
<dashboard enable-sort-zone-taborder='true' name='Sets_With_Calculated_Fields'>
    <style />
    <size maxheight='800' maxwidth='1000' minheight='800' minwidth='1000' />
    <datasources>
    <datasource caption='Orders (Super_Store_Sales)' name='federated.0xbpdl609snqhq1doy2k70u4yenb' />
    </datasources>
    <datasource-dependencies datasource='federated.0xbpdl609snqhq1doy2k70u4yenb'>
    <column caption='Top Product' datatype='boolean' name='[Calculation_1122803716932669440]' role='dimension' type='nominal'>
        <calculation class='tableau' formula='IF [Category] = &quot;Technology&quot; THEN TRUE &#13;&#10;ELSE FALSE&#13;&#10;END' />
    </column>
    <column-instance column='[Calculation_1122803716932669440]' derivation='None' name='[none:Calculation_1122803716932669440:nk]' pivot='key' type='nominal' />
    </datasource-dependencies>
    <zones>
    <zone h='100000' id='4' type-v2='layout-basic' w='100000' x='0' y='0'>
        <zone h='98000' id='9' param='horz' type-v2='layout-flow' w='98400' x='800' y='1000'>
        <zone h='98000' id='7' type-v2='layout-basic' w='82400' x='800' y='1000' />
        <zone fixed-size='160' h='98000' id='8' is-fixed='true' param='vert' type-v2='layout-flow' w='16000' x='83200' y='1000'>
            <zone h='13125' id='10' name='Created_Set_From_Calculated_Field' param='[federated.0xbpdl609snqhq1doy2k70u4yenb].[none:Calculation_1122803716932669440:nk]' type-v2='filter' w='16000' x='83200' y='1000'>
            <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
            </zone-style>
            </zone>
            <zone h='13125' id='11' type-v2='empty' w='16000' x='83200' y='14125'>
            <zone h='13125' id='11' name='Created_Set_From_Calculated_Field' param='[federated.0xbpdl609snqhq1doy2k70u4yenb].[Top Product Set]' type-v2='setMembership' w='16000' x='83200' y='14125'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                </zone-style>
            </zone>
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
    <zone h='33250' id='3' name='Set_CalculatedField_Output_String' w='33300' x='3300' y='4125' />
    <zone h='33250' id='5' name='Set_CalculatedField_Output_Boolean' w='33300' x='42500' y='2375' />
    <zone h='33250' id='6' name='Created_Set_From_Calculated_Field' w='33300' x='4200' y='43625' />
    </zones>
    <devicelayouts>
    <devicelayout auto-generated='true' name='Phone'>
        <size maxheight='950' minheight='950' sizing-mode='vscroll' />
        <zones>
        <zone h='100000' id='15' type-v2='layout-basic' w='100000' x='0' y='0'>
            <zone h='84000' id='14' param='vert' type-v2='layout-flow' w='84000' x='8000' y='8000'>
            <zone fixed-size='266' h='33250' id='5' is-fixed='true' name='Set_CalculatedField_Output_Boolean' w='33300' x='42500' y='2375'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone fixed-size='266' h='33250' id='3' is-fixed='true' name='Set_CalculatedField_Output_String' w='33300' x='3300' y='4125'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone h='13125' id='10' mode='checkdropdown' name='Created_Set_From_Calculated_Field' param='[federated.0xbpdl609snqhq1doy2k70u4yenb].[none:Calculation_1122803716932669440:nk]' type-v2='filter' w='16000' x='83200' y='1000'>
                <zone-style>
                <format attr='border-color' value='#000000' />
                <format attr='border-style' value='none' />
                <format attr='border-width' value='0' />
                <format attr='margin' value='4' />
                <format attr='padding' value='0' />
                </zone-style>
            </zone>
            <zone h='13125' id='11' type-v2='empty' w='16000' x='83200' y='14125'>
                <zone h='13125' id='11' name='Created_Set_From_Calculated_Field' param='[federated.0xbpdl609snqhq1doy2k70u4yenb].[Top Product Set]' type-v2='setMembership' w='16000' x='83200' y='14125'>
                <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='none' />
                    <format attr='border-width' value='0' />
                    <format attr='margin' value='4' />
                    <format attr='padding' value='0' />
                </zone-style>
                </zone>
            </zone>
            <zone fixed-size='266' h='33250' id='6' is-fixed='true' name='Created_Set_From_Calculated_Field' w='33300' x='4200' y='43625'>
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
    <simple-id uuid='{DFC91EFC-2CB9-47DA-A940-7825DD8D4C13}' />
</dashboard>
'''

#     input_text = \
# '''
# <settings>
#   <param name="timeout" value="5000"/>
#   <param name="retries" value="3"/>
#   <param name="user" value="admin"/>
# </settings>
# '''

#     input_text = \
# '''
# <system-snapshot report_id="snap-9f8e7d6c" generated_at="2023-11-01T18:00:00Z">
#     <metadata>
#         <source-system>Mainframe-A</source-system>
#         <data-owner>Finance Dept</data-owner>
#     </metadata>
#     <server-health id="prod-db-01" status="online" cpu_load_percent="78.5" memory_used_gb="212.3" memory_total_gb="256.0" disk_io_reads_ps="15023" disk_io_writes_ps="8109" network_packets_in_pm="9876543" network_packets_out_pm="7654321" uptime_seconds="12345678" active_connections="432" db_version="PostgreSQL 15.1" os_version="RHEL 8.6" kernel_version="4.18.0-372.19.1.el8_6.x86_64" last_maintenance_utc="2023-10-15T04:00:00Z" is_primary_node="true"/>
#     <transaction-log batch_id="batch-abc-123">
#         <tx id="TXN001" amount="100.50" currency="USD" status="completed" />
#         <tx id="TXN002" amount="50.00" currency="EUR" status="completed" />
#         <tx id="TXN003" amount="25.75" currency="USD" status="pending" />
#         <tx id="TXN004" amount="1200.00" currency="JPY" status="completed" />
#         <tx id="TXN005" amount="88.20" currency="USD" status="completed" />
#         <tx id="TXN006" amount="300.00" currency="GBP" status="failed" />
#         <tx id="TXN007" amount="45.50" currency="USD" status="completed" />
#         <tx id="TXN008" amount="99.99" currency="EUR" status="completed" />
#         <tx id="TXN009" amount="10.00" currency="USD" status="completed" />
#         <tx id="TXN010" amount="5000.00" currency="JPY" status="pending" />
#         <tx id="TXN011" amount="75.00" currency="GBP" status="completed" />
#         <tx id="TXN012" amount="150.25" currency="USD" status="completed" />
#     </transaction-log>
# </system-snapshot>
# '''

#     input_text = \
# '''
# <wrapper>
#   <entry level="info" timestamp="2023-10-27T10:00:01Z" message="User logged in" />
#   <entry level="info" timestamp="2023-10-27T10:00:02Z" message="Data loaded" />
#   <entry level="info" timestamp="2023-10-27T10:00:03Z" message="Filter applied" />
# </wrapper>
# '''

#     input_text = \
# '''
# <wrapper>
#   <entry level="warn" timestamp="2023-10-27T10:00:04Z" message="Cache miss" />
#   <entry level="info" timestamp="2023-10-27T10:00:05Z" message="Calculation started" />
#   <entry level="info" timestamp="2023-10-27T10:00:06Z" message="Calculation finished" />
# </wrapper>
# '''

#     input_text = \
# '''
# <style/>
# '''

#     input_text = \
# '''
# <data-segment segment_id="seg-123" region="us-east-1" start_time="2023-11-01T12:00:00Z" end_time="2023-11-01T13:00:00Z" data_quality_score="0.98" source_db="primary_analytics_db" is_realtime="false">
#     <record id="rec_a" value="101" status="ok" />
#     <record id="rec_b" value="102" status="ok" />
#     <record id="rec_c" value="103" status="error" />
#     <record id="rec_d" value="104" status="ok" />
#     <record id="rec_e" value="105" status="ok" />
#     <record id="rec_f" value="106" status="ok" />
#     <record id="rec_g" value="106" status="ok" />
#     <record id="rec_h" value="106" status="ok" />
#     <record id="rec_i" value="106" status="ok" />
#     <record id="rec_j" value="106" status="ok" />
#     <record id="rec_k" value="106" status="ok" />
#     <record id="rec_l" value="106" status="ok" />
#     <record id="rec_m" value="106" status="ok" />
#     <record id="rec_n" value="106" status="ok" />
#     <record id="rec_o" value="106" status="ok" />
# </data-segment>
# '''

#     input_text = \
# '''
# <wrapper><record id="rec_a" value="101" status="ok" />
#   <record id="rec_b" value="102" status="ok" />
#   <record id="rec_c" value="103" status="error" />
#   <record id="rec_d" value="104" status="ok" />
#   <record id="rec_e" value="105" status="ok" />
#   </wrapper>'''

#     input_text = \
# '''
# <wrapper><calculation class=\"tableau\" formula=\"IF [Category] = &quot;Technology&quot; THEN TRUE &#13;&#10;ELSE FALSE&#13;&#10;END\" /></wrapper>'''


#     input_text = \
# '''
# <zones>
#     <zone id="1" name="Parent">
#         <zone id="1a" name="Child"/>
#     </zone>
#     <zone id="2" name="Sibling"/>
# </zones>
# '''


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