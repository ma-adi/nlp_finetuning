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
            logging.info(f"{json.dumps(split_xml, indent=2)}")

            outputs = model_inference.infer_batch(split_xml)

            logging.info(f"Model outputs: ")
            logging.info(f"{json.dumps(outputs, indent=2)}")

            logging.info(f"Stitching model outputs...")
            output = stitch_json_fragments(fragments=outputs)

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
    # Run the Flask app
    # You can change the host and port if needed.
    # host='0.0.0.0' makes it accessible from other machines on the network.
    # debug=True allows for automatic reloading on code changes and provides a debugger.
    app.run(host='127.0.0.1', port=5000)
