import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.model_selection import train_test_split

import evaluate
import numpy as np  # Make sure this is at the top of your file

exact_match = evaluate.load("exact_match")

def compute_metrics(eval_preds, tokenizer):
    pred_ids, label_ids = eval_preds.predictions, eval_preds.label_ids
    pad, vocab = tokenizer.pad_token_id, tokenizer.vocab_size

    # 1) Mask and clamp labels back to pad for decoding
    #    (we need a real token for `.batch_decode`)
    label_ids_for_decoding = np.where(label_ids != -100, label_ids, pad)
    label_ids_for_decoding = np.clip(label_ids_for_decoding, 0, vocab-1)

    # 2) Clamp preds into valid range
    pred_ids = np.clip(pred_ids, 0, vocab-1)

    # 3) Decode both
    decoded_preds  = tokenizer.batch_decode(pred_ids,
                                            skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids_for_decoding,
                                            skip_special_tokens=True)

    # 4) Compute exact match
    em = exact_match.compute(predictions=decoded_preds,
                             references=decoded_labels)["exact_match"]
    return {"exact_match": em}

class NLPCoder:
    """A class for fine-tuning and performing inference with NLP code-to-code models."""

    def __init__(self, model_identifier: str, dataset_path: str = None, output_dir: str = None, load_fine_tuned: bool = False):
        self.model_identifier = model_identifier # Can be a HF model name or a local path
        self.dataset_path = dataset_path
        self.output_dir = output_dir

        if load_fine_tuned:
            print(f"Loading fine-tuned model from: {self.model_identifier}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_identifier)
        else:
            print(f"Loading base model: {self.model_identifier}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_identifier)

        self.model.eval() # Set model to evaluation mode initially

    # def _load_and_split_raw_data(self,
    #                              raw_dataset_path: str,
    #                              test_size: float = 0.2,
    #                              train_size: float = None,
    #                              random_state: int = 42):
    #     """
    #     Loads a raw JSON of the form:
    #         { "dataset": [ {"question":…, "answer":…, "complexity":…}, … ] }
    #     Splits into train/test.
    #     If any example has a 'complexity' key, splits stratified by that; otherwise random.
    #     Returns two lists of dicts: train_data, test_data.
    #     """
    #     with open(raw_dataset_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #     all_examples = data.get("dataset", [])
    #     if not all_examples:
    #         raise ValueError(f"No examples found in `{raw_dataset_path}` under key 'dataset'")

    #     # decide stratification
    #     if all("complexity" in ex for ex in all_examples):
    #         strata = [ex["complexity"] for ex in all_examples]
    #     else:
    #         strata = None

    #     train_data, test_data = train_test_split(
    #         all_examples,
    #         test_size=test_size,
    #         train_size=train_size,
    #         stratify=strata,
    #         random_state=random_state
    #     )
    #     return train_data, test_data

    def _load_and_split_raw_data(self,
                                raw_dataset_path: str,
                                test_size: float = 0.2,
                                train_size: float = None,
                                random_state: int = 42):
        """
        Loads a raw JSON of the form:
            { "dataset": [ {"question":…, "answer":…, "complexity":…}, … ] }
        Splits into train/test.
        If 'complexity' is missing, it's generated based on I/O length percentiles.
        The split is always stratified by complexity.
        Returns two lists of dicts: train_data, test_data.
        """
        with open(raw_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_examples = data.get("dataset", [])
        if not all_examples:
            raise ValueError(f"No examples found in `{raw_dataset_path}` under key 'dataset'")

        # Generate complexity based on length if not present
        if not all("complexity" in ex for ex in all_examples):
            # 1. Determine the max length for each example
            max_lengths = [max(len(ex.get("question", "")), len(ex.get("answer", ""))) for ex in all_examples]

            # 2. Calculate 25th and 75th percentiles of these lengths
            p25, p75 = np.percentile(max_lengths, [25, 75])

            # 3. Assign complexity to each example dictionary
            for ex, length in zip(all_examples, max_lengths):
                if length < p25:
                    ex["complexity"] = 0  # Shorter examples
                elif length <= p75:
                    ex["complexity"] = 1  # Medium examples
                else:
                    ex["complexity"] = 2  # Longer examples

        # Stratify by the (now guaranteed) complexity key
        strata = [ex["complexity"] for ex in all_examples]

        train_data, test_data = train_test_split(
            all_examples,
            test_size=test_size,
            train_size=train_size,
            stratify=strata,
            random_state=random_state
        )
        return train_data, test_data

    def _load_and_prepare_data(self):
        """Loads and prepares data for training."""
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for data loading.")

        train_data, test_data = self._load_and_split_raw_data(
            raw_dataset_path=self.dataset_path,
            test_size=0.2,        # or pass these values in via arguments
            random_state=42
            )
        hf_train_dataset = Dataset.from_list(train_data)
        hf_test_dataset = Dataset.from_list(test_data)

        def preprocess_function(examples):
            inputs = examples['question']
            targets = examples['answer']
            return {'input_text': inputs, 'target_text': targets}

        processed_train_dataset = hf_train_dataset.map(preprocess_function, batched=True)
        processed_test_dataset = hf_test_dataset.map(preprocess_function, batched=True)

        return processed_train_dataset, processed_test_dataset

    def _tokenize_function(self, examples):
        """Tokenizes inputs and targets for the model."""
        model_inputs = self.tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length") # Reverted max_length to 512
        labels = self.tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length") # Reverted max_length to 512
        # replace pad_token_id with -100 so it doesn’t count in loss
        labels_ids = [
            [(tok if tok != self.tokenizer.pad_token_id else -100)
            for tok in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = labels_ids
        
        # model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def fine_tune(self, batch_size, train_epochs):
        """Fine-times the model."""
        if self.dataset_path is None or self.output_dir is None:
            raise ValueError("dataset_path and output_dir must be provided for fine-tuning.")

        train_dataset, eval_dataset = self._load_and_prepare_data()

        tokenized_train_dataset = train_dataset.map(self._tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        tokenized_eval_dataset = eval_dataset.map(self._tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

        # Reduce the size of the evaluation dataset used during training to save memory
        # The full eval_dataset will still be used for the independent evaluation after training.
        reduced_eval_dataset_for_training = tokenized_eval_dataset.select(range(min(40, len(tokenized_eval_dataset))))

        self.model.train() # Re-set model to training mode before fine-tuning

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
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
            eval_dataset=tokenized_eval_dataset, # Use the reduced eval dataset for training
            tokenizer=self.tokenizer,
            compute_metrics=lambda p: compute_metrics(p, self.tokenizer) # Pass tokenizer to metrics function
        )

        trainer.train()

        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"Model and tokenizer saved to {self.output_dir}")
        self.model.eval() # Set back to eval mode after fine-tuning

        print("\n--- Performing Final Evaluation on Full Test Set ---")
        metrics = trainer.evaluate(eval_dataset=tokenized_eval_dataset)
        
        # The trainer.evaluate method already logs the metrics, but we can print them again for clarity
        print(f"Final evaluation on hold-out set: {metrics}")
        print("-------------------------------------------------")

        print(f"Model and tokenizer saved to {self.output_dir}")
        return metrics

    def infer(self, input_text: str):
        """Performs inference with the fine-tuned model."""
        # Model and tokenizer are already loaded in __init__

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], max_new_tokens=1024, num_beams=10, early_stopping=False)

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

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given string using the model's tokenizer.
        """
        # The tokenizer returns a dictionary, we need the 'input_ids'
        tokenized_output = self.tokenizer(text)
        return len(tokenized_output['input_ids'])

    # --- You can also add a method to SEE the tokens ---
    def get_tokens(self, text: str) -> list[str]:
        """
        Returns the list of token strings for a given text.
        Useful for debugging and understanding how text is split.
        """
        return self.tokenizer.tokenize(text)

if __name__ == "__main__":
    # Example usage for training and inference
    model_base_name = "Salesforce/codet5-base"

    nlp_class = NLPCoder(model_identifier=model_base_name, 
                         dataset_path='/Users/maadi5/NLP_code_gen/xml_to_json_dataset.json')
    
    train, test = nlp_class._load_and_split_raw_data(raw_dataset_path='/Users/maadi5/NLP_code_gen/synthetic_conversion_dataset.json')
    json.dump(test, open('test_data.json', 'w', encoding='utf8'), ensure_ascii=False)
    json.dump(train, open('train_data.json', 'w', encoding='utf8'), ensure_ascii=False)