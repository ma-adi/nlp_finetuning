import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ts.torch_handler.base_handler import BaseHandler
import logging

logger = logging.getLogger(__name__)

class CodeT5pHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.initialized = False
        self.context = None # Store context for later use if needed

    def initialize(self, context):
        """
        Initializes the model and tokenizer.
        """
        properties = context.system_properties
        self.model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        # The tokenizer files are typically copied to the model_dir by torch-model-archiver
        logger.info(f"Loading tokenizer from: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)

        # Load model
        logger.info(f"Loading model from: {self.model_dir}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

        self.initialized = True
        logger.info("Model and Tokenizer initialized successfully.")

    def preprocess(self, data):
        """
        Preprocesses the input data for inference.
        TorchServe typically receives data as a list of dictionaries.
        Each dictionary might have 'body' or 'data' key containing the actual input.
        """
        input_texts = []
        for row in data:
            # Assuming input is in 'data' field of the request body, decode if bytes
            if isinstance(row.get("data"), bytes):
                input_texts.append(row.get("data").decode('utf-8'))
            elif isinstance(row.get("body"), bytes):
                input_texts.append(row.get("body").decode('utf-8'))
            else:
                input_texts.append(str(row.get("data") or row.get("body"))) # Handle other types or missing keys

        logger.info(f"Received input texts: {input_texts}")

        # Tokenize inputs
        inputs = self.tokenizer(
            input_texts,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='longest'
        ).to(self.device)
        return inputs

    def inference(self, inputs):
        """
        Performs inference on the preprocessed data.
        """
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=512, # Adjust as needed
                num_beams=1,        # Adjust as needed
                # early_stopping=True
            )
        return outputs

    def postprocess(self, outputs):
        """
        Postprocesses the model output for the response.
        """
        decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.info(f"Decoded predictions: {decoded_preds}")
        return decoded_preds