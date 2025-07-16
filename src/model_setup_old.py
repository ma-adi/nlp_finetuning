import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np  # Make sure this is at the top of your file

exact_match = evaluate.load("exact_match")

def compute_metrics(eval_preds, tokenizer):
    """
    eval_preds.predictions: np.ndarray shape (batch_size, seq_len)
    eval_preds.label_ids:   np.ndarray shape (batch_size, seq_len)
    """
    pred_ids, label_ids = eval_preds.predictions, eval_preds.label_ids
    pad_id   = tokenizer.pad_token_id
    vocab_sz = tokenizer.vocab_size

    # 1) Mask and clamp labels
    label_ids = np.where(label_ids != -100, label_ids, pad_id)
    label_ids = np.clip(label_ids, 0, vocab_sz - 1)

    # 2) Clamp preds
    pred_ids = np.clip(pred_ids, 0, vocab_sz - 1)

    # # 3) Decode
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    label_ids_for_decoding = np.where(label_ids != -100, label_ids, tokenizer.eos_token_id)
    decoded_labels = tokenizer.batch_decode(label_ids_for_decoding, skip_special_tokens=True)
    
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 4) Compute aggregate exact‑match
    em = exact_match.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )["exact_match"]

    print(f"\n>>> Final exact_match = {em:.4f}\n")
    return {"exact_match": em}

class NLPCoder:
    """A class for fine-tuning and performing inference with NLP code-to-code models."""

    def __init__(self, model_identifier: str, dataset_path: str = None, output_dir: str = None, use_context_as_prefix: bool = True, load_fine_tuned: bool = False):
        self.model_identifier = model_identifier # Can be a HF model name or a local path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.use_context_as_prefix = use_context_as_prefix

        if load_fine_tuned:
            print(f"Loading fine-tuned model from: {self.model_identifier}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_identifier)
        else:
            print(f"Loading base model: {self.model_identifier}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_identifier)

        self.model.eval() # Set model to evaluation mode initially

    def _load_and_prepare_data(self):
        """Loads and prepares data for training."""
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for data loading.")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        train_data = data.get('train', [])
        test_data = data.get('test', []) # Use the 'test' key from the dataset

        hf_train_dataset = Dataset.from_list(train_data)
        hf_test_dataset = Dataset.from_list(test_data)

        def preprocess_function(examples):
            inputs = examples['question']
            targets = examples['answer']
            contexts = examples['context']

            if self.use_context_as_prefix:
                inputs = [f"{ctx} {q}" for ctx, q in zip(contexts, inputs)]

            return {'input_text': inputs, 'target_text': targets}

        processed_train_dataset = hf_train_dataset.map(preprocess_function, batched=True)
        processed_test_dataset = hf_test_dataset.map(preprocess_function, batched=True)

        return processed_train_dataset, processed_test_dataset

    def _tokenize_function(self, examples):
        """Tokenizes inputs and targets for the model."""
        model_inputs = self.tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length") # Reverted max_length to 512
        labels = self.tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length") # Reverted max_length to 512
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def fine_tune(self):
        """Fine-times the model."""
        if self.dataset_path is None or self.output_dir is None:
            raise ValueError("dataset_path and output_dir must be provided for fine-tuning.")

        train_dataset, eval_dataset = self._load_and_prepare_data()

        tokenized_train_dataset = train_dataset.map(self._tokenize_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(self._tokenize_function, batched=True)

        # Reduce the size of the evaluation dataset used during training to save memory
        # The full eval_dataset will still be used for the independent evaluation after training.
        reduced_eval_dataset_for_training = tokenized_eval_dataset.select(range(min(40, len(tokenized_eval_dataset))))

        self.model.train() # Re-set model to training mode before fine-tuning

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=4,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
            predict_with_generate = True,
            generation_max_length=512,         # ← or generation_max_new_tokens=N
            generation_num_beams=5,      # allow up to 64 new tokens  # keep going until true EOS
            metric_for_best_model="exact_match",
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=reduced_eval_dataset_for_training, # Use the reduced eval dataset for training
            tokenizer=self.tokenizer,
            compute_metrics=lambda p: compute_metrics(p, self.tokenizer) # Pass tokenizer to metrics function
        )

        trainer.train()

        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"Model and tokenizer saved to {self.output_dir}")
        self.model.eval() # Set back to eval mode after fine-tuning

    def infer(self, input_text: str):
        """Performs inference with the fine-tuned model."""
        # Model and tokenizer are already loaded in __init__

        if self.use_context_as_prefix:
            context = "Convert the following XML to sqsh-internal format"
            input_text = f"{context} {input_text}"

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], max_new_tokens=512, num_beams=5, early_stopping=True)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate(self, tokenized_test_dataset: Dataset):
        """Evaluates the model on a given tokenized test dataset using Trainer.predict and reports exact match accuracy."""
        print("\n--- Starting Optimized Evaluation ---")
        self.model.eval() # Ensure model is in evaluation mode

        # Create a temporary Trainer instance for prediction
        # We need to set a dummy output_dir for the Trainer, even if we're just predicting
        temp_training_args = Seq2SeqTrainingArguments(
            output_dir="./tmp_eval_output", # Temporary directory for eval logs/results
            report_to="none",
            generation_num_beams=5,
            generation_max_length=512,
            predict_with_generate=True,
            per_device_eval_batch_size=2,
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=temp_training_args,
            tokenizer=self.tokenizer,
            compute_metrics=lambda p: compute_metrics(p, self.tokenizer) # Pass tokenizer to metrics function
        )

        predictions_output = trainer.predict(test_dataset=tokenized_test_dataset)
        metrics = predictions_output.metrics

        print(f"Evaluation Complete. Metrics: {metrics}")
        print("---------------------------")
        return metrics

if __name__ == "__main__":
    # Example usage for training and inference
    model_base_name = "Salesforce/codet5-base"
    dataset_file_path = "simple_transform_dataset_shrunk.json"
    output_model_directory = f"./fine_tuned_models/{model_base_name.replace('/', '_')}"
    use_context_toggle = True # Set to False to disable using context as a prefix

    print("--- Training Example ---")
    print(f"Initializing NLPCoder for training with model: {model_base_name}")
    nlp_coder_train = NLPCoder(
        model_identifier=model_base_name,
        dataset_path=dataset_file_path,
        output_dir=output_model_directory,
        use_context_as_prefix=use_context_toggle,
        load_fine_tuned=False # Initialize for training (load base model)
    )

    print(f"Starting fine-tuning for {model_base_name}...")
    nlp_coder_train.fine_tune()

    print("\n--- Inference Example (using the freshly trained model) ---")
    xml_input_train_infer = "<root><item>example_item_1</item></root>"
    generated_json_train_infer = nlp_coder_train.infer(xml_input_train_infer)
    print(f"XML Input: {xml_input_train_infer}")
    print(f"Generated JSON: {generated_json_train_infer}")

    # Automatically call evaluation after training, using the test split from the loaded data
    print("\n--- Evaluation after Training ---")
    _, test_dataset_for_evaluation_raw = nlp_coder_train._load_and_prepare_data() # Load data again to get the test split
    # Tokenize the test dataset for evaluation before passing to evaluate method
    tokenized_test_dataset_for_evaluation = test_dataset_for_evaluation_raw.map(nlp_coder_train._tokenize_function, batched=True, remove_columns=test_dataset_for_evaluation_raw.column_names)
    nlp_coder_train.evaluate(tokenized_test_dataset_for_evaluation)

    print("\n--- Inference Example (loading a previously fine-tuned model) ---")
    # This demonstrates loading the model independently for inference
    fine_tuned_model_path = output_model_directory # Path where the model was saved

    print(f"Initializing NLPCoder for inference from path: {fine_tuned_model_path}")
    nlp_coder_infer = NLPCoder(
        model_identifier=fine_tuned_model_path,
        use_context_as_prefix=use_context_toggle, # IMPORTANT: Match this with how the model was trained!
        load_fine_tuned=True # Initialize for inference (load fine-tuned model)
    )

    xml_input_separate_infer = "<root><data>another_test</data><id>456</id></root>"
    generated_json_separate_infer = nlp_coder_infer.infer(xml_input_separate_infer)
    print(f"XML Input: {xml_input_separate_infer}")
    print(f"Generated JSON: {generated_json_separate_infer}")

    # Demonstrate independent evaluation as well
    print("\n--- Independent Evaluation (of loaded fine-tuned model) ---")
    # Load and tokenize data for this independent evaluation instance
    _, independent_test_dataset_raw = nlp_coder_infer._load_and_prepare_data() 
    tokenized_independent_test_dataset = independent_test_dataset_raw.map(nlp_coder_infer._tokenize_function, batched=True, remove_columns=independent_test_dataset_raw.column_names)
    nlp_coder_infer.evaluate(tokenized_independent_test_dataset)