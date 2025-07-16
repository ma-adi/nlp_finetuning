import json
import torch
from datasets import Dataset
from typing import Sequence, Tuple, List, Dict
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np  # Ensure at top
from collections import Counter
from collections import Counter, defaultdict

import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

exact_match = evaluate.load("exact_match")

def compute_metrics(eval_preds, tokenizer):
    # Handle tuple outputs
    preds = eval_preds.predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    labels = eval_preds.label_ids
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    # 1) Mask and clamp labels for decoding
    labels_for_decoding = np.where(labels != -100, labels, pad_id)
    labels_for_decoding = np.clip(labels_for_decoding, 0, vocab_size - 1)

    # 2) Clamp predictions to valid range
    preds = np.clip(preds, 0, vocab_size - 1)

    # 3) Decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_for_decoding, skip_special_tokens=True)

    # 4) Exact match
    em = exact_match.compute(predictions=decoded_preds, references=decoded_labels)["exact_match"]
    return {"exact_match": em}

class NLPCoder:
    def __init__(
        self, model_identifier: str, 
        dataset_path: str = None,
        output_dir: str = None, use_context_as_prefix: bool = False,
        load_fine_tuned: bool = False,
        test_unseen_formats: bool = True,
        holdout_format_ids: Optional[List[str]] = None
    ):
        self.model_identifier = model_identifier
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.use_context_as_prefix = use_context_as_prefix
        self.holdout_format_ids = holdout_format_ids

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_identifier)
        self.model.eval()
        self.test_unseen_formats = test_unseen_formats


    def _load_and_split_raw_data(
        self,
        raw_dataset_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        n_complexity_bins: int = 3,
    ) -> None:
        
        print("--- Starting Data Preparation and Splitting ---")

        holdout_format_ids = self.holdout_format_ids
        # 1. Load and Pre-process Data
        with open(raw_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_examples = data.get("dataset", [])
        if not all_examples:
            raise ValueError(f"No examples found in `{raw_dataset_path}` under key 'dataset'")

        # 2. Calculate Dynamic Complexity
        print(f"Calculating dynamic complexity using {n_complexity_bins} bins...")
        lengths = [max(len(ex.get("question", "")), len(ex.get("answer", ""))) for ex in all_examples]
        # Create bin edges from percentiles (e.g., [p0, p33, p66, p100])
        bin_edges = np.percentile(lengths, np.linspace(0, 100, n_complexity_bins + 1))
        for ex, length in zip(all_examples, lengths):
            # np.digitize finds which bin a value belongs to. Bins are 1-indexed, so we subtract 1.
            ex["complexity"] = np.digitize(length, bin_edges, right=False) - 1
            # Ensure the last element goes into the last bin
            if ex["complexity"] == n_complexity_bins:
                ex["complexity"] -= 1
        
        print("Complexity calculation complete.")

        # 3. Pre-calculate statistics for selection logic
        total_examples = len(all_examples)
        format_stats = defaultdict(lambda: {'count': 0, 'tags': set(), 'complexities': set()})
        all_tags = set()
        for ex in all_examples:
            fid = ex['format_id']
            format_stats[fid]['count'] += 1
            format_stats[fid]['tags'].update(ex.get('tags', []))
            format_stats[fid]['complexities'].add(ex['complexity'])
            all_tags.update(ex.get('tags', []))

        for fid in format_stats:
            format_stats[fid]['proportion'] = format_stats[fid]['count'] / total_examples

        # 4. Select Unseen Holdout Format(s)
        if holdout_format_ids is not None:
            print(f"\nUser has provided manual holdout formats: {holdout_format_ids}")
            print("Skipping automatic selection.")
        else:
            print("\nStarting automatic multi-format holdout selection...")
            all_fids = list(format_stats.keys())
            all_combinations = []
            # Generate all non-empty, proper subsets of formats
            for r in range(1, len(all_fids)):
                all_combinations.extend(combinations(all_fids, r))

            scored_combinations = []
            for combo in all_combinations:
                combo_tags = set().union(*(format_stats[fid]['tags'] for fid in combo))
                combo_complexities = set().union(*(format_stats[fid]['complexities'] for fid in combo))
                combo_proportion = sum(format_stats[fid]['proportion'] for fid in combo)

                # Find tags in the "training" portion
                train_fids = set(all_fids) - set(combo)
                train_tags = set().union(*(format_stats[fid]['tags'] for fid in train_fids))
                
                # --- Scoring Function ---
                # 1. Tag Leakage Penalty (high cost)
                leaked_tags = combo_tags - train_tags
                leakage_penalty = len(leaked_tags) * 1.0  # A large penalty for each leaked tag

                # 2. Size Mismatch Score (primary objective)
                size_score = abs(combo_proportion - test_size)

                # 3. Diversity Penalty (minor factor)
                diversity_penalty = (n_complexity_bins - len(combo_complexities)) * 0.01

                final_score = leakage_penalty + size_score + diversity_penalty
                scored_combinations.append((combo, final_score, leaked_tags))

            if not scored_combinations:
                raise RuntimeError("Could not generate any format combinations for holdout.")

            # Select the combination with the lowest score
            scored_combinations.sort(key=lambda x: x[1])
            best_combo, best_score, leaked_tags = scored_combinations[0]
            holdout_format_ids = list(best_combo)

            print("Automatic selection complete.")
            print(f"  > Best holdout combination: {holdout_format_ids}")
            best_prop = sum(format_stats[fid]['proportion'] for fid in holdout_format_ids)
            print(f"  > Covers {best_prop:.2%} of the dataset (target was {test_size:.2%})")
            if leaked_tags:
                print(f"  > WARNING: This selection results in tag leakage. Tags in test but not train: {leaked_tags}")
            else:
                print("  > Success: All tags in the holdout set are covered by the training set.")

        # 5. Partition the Data
        unseen_test_data = [ex for ex in all_examples if ex['format_id'] in holdout_format_ids]
        remaining_data = [ex for ex in all_examples if ex['format_id'] not in holdout_format_ids]
        print(f"\nPartitioned {len(unseen_test_data)} examples for the unseen holdout set.")

        # 6. Perform Stratified Split on Remainder
        # Adjust split ratio to meet the master test_size as closely as possible
        n_target_test = int(round(test_size * total_examples))
        n_needed_from_rem = max(0, n_target_test - len(unseen_test_data))
        
        test_size_adj = n_needed_from_rem / len(remaining_data) if remaining_data else 0

        
        train_rem, test_rem = [], []
        if remaining_data and test_size_adj > 0:
            print(f"Splitting remaining {len(remaining_data)} examples...")

            tags_are_present = 'tags' in remaining_data[0]
            if tags_are_present:
                print("  > Applying composite stratification using 'complexity' and 'tags'.")
                strata = [
                    f"{ex['complexity']}_{'-'.join(sorted(ex.get('tags', [])))}"
                    for ex in remaining_data
                ]
            else:
                print("  > Applying stratification using 'complexity' only.")
                strata = [ex['complexity'] for ex in remaining_data]

            try:
                train_rem, test_rem = train_test_split(
                    remaining_data,
                    test_size=test_size_adj,
                    stratify=strata,
                    random_state=random_state
                )
            except ValueError:
                print("  > WARNING: Stratified split failed (likely due to small stratum size). Falling back to non-stratified split.")
                train_rem, test_rem = train_test_split(
                    remaining_data,
                    test_size=test_size_adj,
                    random_state=random_state
                )
        else:
            train_rem = remaining_data
            print("No additional examples needed from remaining data for the test set.")

        # 7. Combine and Finalize
        train_data = train_rem
        test_data = test_rem + unseen_test_data

        # 8. Report and Save
        print("\n--- Final Split Summary ---")
        print(f"Total examples: {total_examples}")
        print(f"Train set: {len(train_data)} examples ({len(train_data)/total_examples:.2%})")
        print(f"Test set:  {len(test_data)} examples ({len(test_data)/total_examples:.2%})")
        
        print("\nComplexity Distribution:")
        train_counts = Counter(ex['complexity'] for ex in train_data)
        test_counts = Counter(ex['complexity'] for ex in test_data)
        print(f"  {'Bin':<5} | {'Train':<10} | {'Test':<10}")
        print(f"  {'-'*5} | {'-'*10} | {'-'*10}")
        for i in range(n_complexity_bins):
            print(f"  {i:<5} | {train_counts.get(i, 0):<10} | {test_counts.get(i, 0):<10}")

        return train_data, test_data


    

    def _load_and_prepare_data(self):
        if not self.dataset_path:
            raise ValueError("dataset_path required")
        train, test = self._load_and_split_raw_data(self.dataset_path)
        train_ds = Dataset.from_list(train)
        test_ds = Dataset.from_list(test)

        def preprocess(ex):
            return {"input_text": ex['question'], "target_text": ex['answer']}

        return (
            train_ds.map(preprocess, batched=True),
            test_ds.map(preprocess, batched=True)
        )

    def _tokenize_function(self, examples):
        inputs = self.tokenizer(
            examples['input_text'], max_length=512,
            truncation=True, padding='longest'
        )
        labels = self.tokenizer(
            examples['target_text'], max_length=512,
            truncation=True, padding='longest'
        )
        labels_ids = [
            [(t if t != self.tokenizer.pad_token_id else -100)
             for t in seq]
            for seq in labels['input_ids']
        ]
        inputs['labels'] = labels_ids
        return inputs

    def fine_tune(self, train_epochs: int, batch_size: int):
        torch.manual_seed(42)
        np.random.seed(42)

        train_ds, eval_ds = self._load_and_prepare_data()
        tok_train = train_ds.map(
            self._tokenize_function, batched=True,
            remove_columns=train_ds.column_names
        )
        tok_eval = eval_ds.map(
            self._tokenize_function, batched=True,
            remove_columns=eval_ds.column_names
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )

        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy='epoch', save_strategy='epoch',
            logging_dir='./logs', logging_steps=50,
            load_best_model_at_end=True, report_to='none',
            predict_with_generate=True,
            generation_max_length=512,
            generation_num_beams=5,
            metric_for_best_model='exact_match'
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=tok_train,
            eval_dataset=tok_eval,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, self.tokenizer)
        )

        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # post-hoc evaluation
        metrics = trainer.evaluate(eval_dataset=tok_eval)
        self.model.eval()
        return metrics

    def infer(self, input_text: str) -> str:
        inputs = self.tokenizer(
            input_text, return_tensors='pt', max_length=1024,
            truncation=True
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'], max_new_tokens=1024,
                num_beams=5, early_stopping=False
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate(self, tok_test_ds: Dataset):
        self.model.eval()
        args = Seq2SeqTrainingArguments(
            output_dir='./tmp_eval', report_to='none',
            predict_with_generate=True,
            per_device_eval_batch_size=2,
            generation_num_beams=5,
            generation_max_length=512
        )
        trainer = Seq2SeqTrainer(
            model=self.model, args=args,
            tokenizer=self.tokenizer,
            compute_metrics=lambda p: compute_metrics(p, self.tokenizer)
        )
        preds = trainer.predict(tok_test_ds)
        print(f"Optimized evaluation EM: {preds.metrics['exact_match']}")
        return preds.metrics


if __name__ == "__main__":
    # Example usage for training and inference
    model_base_name = "Salesforce/codet5-base"

    nlp_class = NLPCoder(model_identifier=model_base_name, 
                         dataset_path='/Users/maadi5/nlp_finetuning/tagged_dataset_v1.json')
    
    train, test = nlp_class._load_and_split_raw_data(raw_dataset_path='/Users/maadi5/nlp_finetuning/tagged_dataset_v1.json')
    json.dump(test, open('test_data.json', 'w', encoding='utf8'), ensure_ascii=False)
    json.dump(train, open('train_data.json', 'w', encoding='utf8'), ensure_ascii=False)