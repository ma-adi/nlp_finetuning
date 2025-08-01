# app_server.py
from flask import Flask, request, jsonify
from model_setup_new import NLPCoder # Assuming model_setup.py is in the same directory or accessible
import os
from utils import  split_xml_into_chunks, stitch_json_fragments, split_xml
# from utils_v2 import create_xml_blueprint, stitch_json_from_blueprint
import json

import logging
import traceback

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

THRESHOLD_FOR_SPLIT = 101
MAX_BATCH_SIZE = 1

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
        # model_path = '/Users/maadi5/nlp_finetuning/master_curriculum_3000_weights_hint0.3_bestmodel_fixed'
        model_path = '/Users/maadi5/nlp_finetuning/easy_master_curriculum_3_3000_weights_allformatstrain_bestmodel_fixed_noneindent_v2_emptylistdict'

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
            logging.info("MODEL LOADING FAILED")
            logging.info(traceback.format_exc())
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

    # # split_chunks = split_xml(root=input_text,max_tokens=THRESHOLD_FOR_SPLIT)#(xml_str=input_text, max_tokens=THRESHOLD_FOR_SPLIT)
    # blueprint, split_chunks = create_xml_blueprint(xml_string=input_text, max_tokens=THRESHOLD_FOR_SPLIT)

    # # blueprint, split_xml = create_xml_blueprint(xml_text=input_text, max_tokens=THRESHOLD_FOR_SPLIT)
    # if len(split_chunks)>1:

    #     try:
    #         logging.info(f"Input split into {len(split_chunks)} parts..")
    #         # logging.info(f"{json.dumps(split_xml, indent=2)}")

    #         groups = []
    #         group = {}
    #         count = 0
    #         for key, val in split_chunks.items():
    #             if count% MAX_BATCH_SIZE == 0 and count != 0:
    #                 groups.append(group)
    #                 group = {}
    #             group[key] = val
    #             count += 1

    #             if count == len(split_chunks):
    #                 groups.append(group)

    #         # logging.info(f"groups: {groups}")
    #         outputs_to_stitch = {}
    #         for idx, g in enumerate(groups):
    #             logging.info(f'Running inference on batch {idx+1}. Batch size: {len(g)}')
    #             keys = list(g.keys())
    #             values_list = list(g.values())
    #             outputs = model_inference.infer_batch(values_list)
    #             output_dict = {}
    #             for ind, k in enumerate(keys):
    #                 output_dict[k] = outputs[ind]
                    
    #             outputs_to_stitch.update(output_dict)

    #         logging.info(f"Model outputs: ")
    #         logging.info(f"{json.dumps(outputs_to_stitch, indent=2)}")

    #         logging.info(f"Stitching model outputs...")
    #         # output = stitch_json_fragments(fragments=outputs_to_stitch)
    #         output = stitch_json_from_blueprint(blueprint=blueprint, processed_json_chunks= outputs_to_stitch)

    #         return jsonify({"output": output})
    #     except Exception as e:
    #         print(f"Error during inference: {e}")
    #         return jsonify({"error": f"An error occurred during inference: {str(e)}"}), 500

    # else:
    
    try:
        logging.info("GOING TO MODEL...")
        output = model_inference.infer(input_text)
        logging.info(f"output: {output}")
        return jsonify({"output": output})
    except Exception as e:
        logging.info(f"Error during inference: {e}")
        logging.info(traceback.format_exc())
        return jsonify({"error": f"An error occurred during inference: {str(e)}"}), 500

if __name__ == '__main__':
    # Load the model before starting the Flask application
    load_model()

    app.run(host='127.0.0.1', port=5000)



