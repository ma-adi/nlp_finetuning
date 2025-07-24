from flask import Flask, request, jsonify
from model_setup_new import NLPCoder
import os
import json
import logging

# Import our new, flexible pipeline class
from utils_v4 import XmlToJsonPipeline


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
MAX_BATCH_SIZE = 12

model_inference = None

def load_model():
    """
    Loads the NLPCoder model once when the server starts.
    """
    global model_inference
    if model_inference is None:
        print("Loading NLPCoder model... This will happen only once.")
        # Ensure the model path is correct for the environment where this server runs
        model_path = '/Users/maadi5/nlp_finetuning/master_curriculum_3000_weights_hint0.3_bestmodel_fixed'
        
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

def real_inference_adapter(chunks_to_process: dict) -> dict:
    """
    An adapter function that matches the signature required by XmlToJsonPipeline.
    It takes a dictionary of chunks, handles batching, calls the real model,
    and returns a dictionary of results.
    """
    if not chunks_to_process:
        return {}

    logging.info(f"Preparing to run inference on {len(chunks_to_process)} chunks.")
    
    # Your existing batching logic
    groups = []
    group = {}
    count = 0
    for key, val in chunks_to_process.items():
        if count % MAX_BATCH_SIZE == 0 and count != 0:
            groups.append(group)
            group = {}
        group[key] = val
        count += 1
    if group: # Add the last group if it's not empty
        groups.append(group)

    outputs_to_stitch = {}
    for idx, g in enumerate(groups):
        logging.info(f'Running inference on batch {idx+1}. Batch size: {len(g)}')
        keys = list(g.keys())
        values_list = list(g.values())
        
        # This is the actual call to your model
        outputs = model_inference.infer_batch(values_list)
        
        # The model returns a list of JSON strings. We map them back to their chunk_ids.
        for i, key in enumerate(keys):
            outputs_to_stitch[key] = outputs[i]
            
    logging.info(f"Model inference complete for all batches.")
    return outputs_to_stitch


@app.route('/infer', methods=['POST'])
def infer_endpoint():
    """
    Endpoint for performing inference using the new pipeline.
    """
    if model_inference is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    data = request.get_json()
    if not data or 'input_text' not in data:
        return jsonify({"error": "Invalid request. 'input_text' is required."}), 400

    input_text = data['input_text']

    try:
        # 1. Instantiate the pipeline, injecting our real inference function
        pipeline = XmlToJsonPipeline(
            inference_function=real_inference_adapter,
            list_split_threshold=THRESHOLD_FOR_SPLIT
        )

        # 2. Run the entire process with one simple call
        final_output = pipeline.process(input_text)

        return jsonify({"output": final_output})

    except Exception as e:
        logging.error(f"An error occurred during the pipeline process: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during inference: {str(e)}"}), 500
    

if __name__ == '__main__':
    # Load the model before starting the Flask application
    load_model()

    app.run(host='127.0.0.1', port=5000)
