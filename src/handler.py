# handler.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ts.torch_handler.base_handler import BaseHandler
import os
import json
import logging
from typing import List

# Import your NLPCoder class from the local file
from NLPCoder_infer import NLPCoder

logger = logging.getLogger(__name__)

class NLPCoderHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.nlpcoder_instance = None # To hold an instance of your NLPCoder class
        self.model_name = "t5p" # Default, will be overridden by model_identifier if provided

    def initialize(self, context):
        """
        Initializes the model, tokenizer, and sets up the device.
        This method is called once when the model is loaded.
        """
        properties = context.system_properties
        # Determine device (CPU or GPU)
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(self.map_location)
        logger.info(f"Using device: {self.device}")

        # Path to model artifacts (where model_weights/ is in the MAR file)
        model_dir = properties.get("model_dir")
        
        # In a MAR file, your model artifacts are typically in the root of the MAR.
        # So, the path to `pytorch_model.bin`, `config.json` etc., is `model_dir`.
        self.model_path = model_dir
        
        # Load the NLPCoder class and then load the model/tokenizer
        try:
            # Instantiate your NLPCoder class
            # The 'model_identifier' passed to NLPCoder's __init__ will be the path
            # within the MAR file where the actual model config/weights are located.
            # In our case, it's the model_dir itself.
            self.nlpcoder_instance = NLPCoder(model_identifier=self.model_path)
            self.nlpcoder_instance.load_model_and_tokenizer(self.model_path, self.device)
            self.initialized = True
            logger.info("NLPCoder model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing NLPCoder: {e}")
            raise e

    def preprocess(self, data):
        """
        Preprocesses the incoming requests.
        `data` here is a list of individual requests that TorchServe has batched.
        Each element in `data` is from a separate client request.
        """
        input_texts = []
        for row in data:
            # Each 'row' is one individual client's request payload
            # Extract the actual input string from the row
            if isinstance(row.get("data"), bytes):
                input_data = row.get("data").decode('utf-8')
            elif isinstance(row.get("body"), bytes):
                input_data = row.get("body").decode('utf-8')
            else:
                input_data = row.get("data") or row.get("body")

            # This part assumes a client sends ONLY a single string, or a list IF it's designed
            # for a multi-input single request. If you strictly want *individual* strings per client request,
            # you might simplify this part and raise an error if `input_data` is a list.
            if isinstance(input_data, str):
                input_texts.append(input_data)
            # You might want to remove or be very careful with the `elif isinstance(input_data, list):`
            # part if you want clients to *always* send individual strings and rely on TorchServe batching.
            # If a client sends `{"data": ["text1", "text2"]}`, your current handler would extend `input_texts`
            # with these, effectively treating it as a client-side batch inside TorchServe's dynamic batch.
            # This is generally not ideal.
            # For optimal dynamic batching, clients should send one string per request.
            # Example for strictly single inputs per client:
            # else:
            #     raise ValueError(f"Unsupported input format: {type(input_data)}. Expected a single string.")

        if not input_texts:
            raise ValueError("No input text found in the request.")

        return input_texts

    def inference(self, preprocessed_data):
        """
        Performs inference using the loaded NLPCoder instance.
        """
        if not self.initialized:
            raise RuntimeError("Model is not initialized.")

        # preprocessed_data will be a list of strings from `preprocess`
        try:
            # Use infer_batch for efficiency
            predictions = self.nlpcoder_instance.infer_batch(preprocessed_data)
            return predictions
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise e

    def postprocess(self, inference_output):
        """
        Post-processes the output from inference.
        Returns the predictions as a list of dictionaries (JSON serializable).
        """
        # inference_output is already a list of strings from infer_batch
        # We'll wrap each prediction in a dictionary for JSON response format
        return [{"prediction": text} for text in inference_output]