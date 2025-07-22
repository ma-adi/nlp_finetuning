# app_server.py
from flask import Flask, request, jsonify
from model_setup_new import NLPCoder # Assuming model_setup.py is in the same directory or accessible
import os
from utils import  split_xml_into_chunks, stitch_json_fragments
import json

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Minimum level to log
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("app.log"),       # Log to a file
        logging.StreamHandler()               # Log to the console
    ]
)

# Create a logger
logger = logging.getLogger(__name__)

app = Flask(__name__)

THRESHOLD_FOR_SPLIT = 100
MAX_BATCH_SIZE = 3

# Global variable to hold the loaded model
model_inference = None

def load_model():
    """
    Loads the NLPCoder model once when the server starts.
    """
    global model_inference
    if model_inference is None:
        print("Loading NLPCoder model... This will happen only once.")
        # Ensure the model path is correct for the environment where this server runs
        model_path = '/Users/maadi5/nlp_finetuning/format_conversion_tagged_weights_exp1'
        
        # Basic check if the path exists (optional, but good for debugging)
        if not os.path.exists(model_path):
            print(f"Warning: Model path '{model_path}' does not exist. Please verify.")
            # You might want to raise an error or handle this more robustly
            # For now, we'll proceed, assuming NLPCoder handles missing paths gracefully.

        try:
            model_inference = NLPCoder(
                model_identifier=model_path,
                load_fine_tuned=True
            )
            print("NLPCoder model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Depending on the error, you might want to exit or disable inference
            model_inference = None # Ensure it's None if loading failed

@app.route('/infer', methods=['POST'])
def infer_endpoint():
    """
    Endpoint for performing inference.
    Expects a JSON payload with 'input_text'.
    """
    if model_inference is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    data = request.get_json()
    if not data or 'input_text' not in data:
        return jsonify({"error": "Invalid request. 'input_text' is required."}), 400

    input_text = data['input_text']

    split_xml = split_xml_into_chunks(xml_str=input_text, max_tokens=THRESHOLD_FOR_SPLIT)

    if len(split_xml)>1:

        try:
            logging.info(f"Input split into {len(split_xml)} parts..")
            # logging.info(f"{json.dumps(split_xml, indent=2)}")

            groups = []
            group = []
            count = 0
            for val in split_xml:
                if count% MAX_BATCH_SIZE == 0 and count != 0:
                    groups.append(group)
                    group = []
                group.append(val)
                count += 1

            # logging.info(f"groups: {groups}")
            outputs_to_stitch = []
            for idx, g in enumerate(groups):
                logging.info(f'Running inference on batch {idx+1}. Batch size: {len(g)}')
                outputs = model_inference.infer_batch(g)
                outputs_to_stitch.extend(outputs)

            logging.info(f"Model outputs: ")
            logging.info(f"{json.dumps(outputs_to_stitch, indent=2)}")

            logging.info(f"Stitching model outputs...")
            output = stitch_json_fragments(fragments=outputs_to_stitch)

            return jsonify({"output": output})
        except Exception as e:
            print(f"Error during inference: {e}")
            return jsonify({"error": f"An error occurred during inference: {str(e)}"}), 500

    else:
    
        try:
            output = model_inference.infer(input_text)
            return jsonify({"output": output})
        except Exception as e:
            print(f"Error during inference: {e}")
            return jsonify({"error": f"An error occurred during inference: {str(e)}"}), 500

if __name__ == '__main__':
    # Load the model before starting the Flask application
    load_model()

    app.run(host='127.0.0.1', port=5000)

    ##Debug:

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

#     split_xml = split_xml_into_chunks(xml_str=input_text, max_tokens=THRESHOLD_FOR_SPLIT)
#     print(split_xml)

#     json.dump(split_xml, open('split_xml.json', 'w', encoding='utf8'), ensure_ascii=False)

#     for idx , val in enumerate(split_xml):
#         output = model_inference.infer(val)

#         print(f"input_{idx}: ", val)
#         print(f"output_{idx}: ", output)





