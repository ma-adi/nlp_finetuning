# NLPCoder_model.py
import json
import torch
from datasets import Dataset # Keep for potential local testing, but not strictly needed for inference
from typing import Sequence, Tuple, List, Dict, Optional
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import os

class NLPCoder:
    def __init__(self, model_name_or_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # If model_name_or_path is a directory, load from there
        if os.path.isdir(model_name_or_path):
            print(f"Loading model and tokenizer from local directory: {model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        else:
            # Fallback to downloading if not a local path (less common in production)
            print(f"Loading model and tokenizer from Hugging Face Hub: {model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)

        self.model.eval() # Set model to evaluation mode

    # Keep only the inference methods
    def infer(self, input_text: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer not loaded. Call initialize() first.")
        
        # Ensure model is in evaluation mode
        self.model.eval()

        # Place tensors on the correct device (TorchServe handler will set this up)
        inputs = self.tokenizer(
            input_text, return_tensors='pt', max_length=512,
            truncation=False
        ).to(self.model.device) # self.model.device will be set by TorchServe

        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'], max_new_tokens=512,
                num_beams=3, early_stopping=False
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def infer_batch(self, input_texts: List[str],
                    max_length: int = 256,
                    num_beams: int = 1) -> List[str]:
        """
        Performs generation on a batch of input strings.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer not loaded. Call initialize() first.")

        # Ensure model is in evaluation mode
        self.model.eval()

        # tokenize all inputs at once (padding to the longest in the batch)
        encodings = self.tokenizer(
            input_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.model.device) # self.model.device will be set by TorchServe

        with torch.no_grad():
            batch_outputs = self.model.generate(
                encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                max_new_tokens=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        # decode each sample in the batch
        return [
            self.tokenizer.decode(t, skip_special_tokens=True)
            for t in batch_outputs
        ]

    # New method to load model and tokenizer
    def load_model_and_tokenizer(self, model_path: str, device: torch.device):
        """
        Loads the fine-tuned model and tokenizer from a given path.
        """
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loading model from {model_path} to device {device}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.model.eval() # Ensure it's in evaluation mode after loading
        print("Model and tokenizer loaded successfully.")