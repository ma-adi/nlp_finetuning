import json
import torch
from datasets import Dataset
from typing import Sequence, Tuple, List, Dict, Optional
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np  # Ensure at top
from collections import Counter
from collections import Counter, defaultdict
from itertools import combinations

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
        holdout_format_ids: Optional[List[str]] = None,
        unseen_hint_proportion: float = 0.0,
    ):
        self.model_identifier = model_identifier
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.use_context_as_prefix = use_context_as_prefix
        self.holdout_format_ids = holdout_format_ids
        self.unseen_hint_proportion = unseen_hint_proportion

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_identifier, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_identifier, trust_remote_code=True)
        self.model.eval()
        self.test_unseen_formats = test_unseen_formats


    def _load_and_split_raw_data(
        self,
        raw_dataset_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        n_complexity_bins: int = 3,
    ) -> tuple[list, list]:
        
        print("--- Starting Data Preparation and Splitting ---")

        # 1. Load and Pre-process Data
        with open(raw_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_examples = data.get("dataset", [])
        if not all_examples:
            raise ValueError(f"No examples found in `{raw_dataset_path}` under key 'dataset'")

        # 2. Calculate Dynamic Complexity (Common for both paths)
        print(f"Calculating dynamic complexity using {n_complexity_bins} bins...")
        lengths = [max(len(ex.get("question", "")), len(ex.get("answer", ""))) for ex in all_examples]
        bin_edges = np.percentile(lengths, np.linspace(0, 100, n_complexity_bins + 1))
        # Ensure bin edges are unique to avoid issues with digitize
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2: # Handle case where all lengths are the same
            bin_edges = np.array([min(lengths), max(lengths) + 1])
            n_complexity_bins = 1

        for ex, length in zip(all_examples, lengths):
            # np.digitize finds which bin a value belongs to. Bins are 1-indexed.
            complexity_bin = np.digitize(length, bin_edges, right=False) - 1
            # Clamp the value to be within the valid bin range [0, n_complexity_bins - 1]
            ex["complexity"] = max(0, min(complexity_bin, n_complexity_bins - 1))
        
        print("Complexity calculation complete.")
        
        # --- NEW: Check for `format_id` and choose splitting strategy ---
        has_format_id = 'format_id' in all_examples[0]

        if has_format_id:
            # --- PATH A: Original logic for format_id-based holdout splitting ---
            print("\nFound 'format_id'. Proceeding with unseen format holdout strategy.")
            
            unseen_hint_proportion = self.unseen_hint_proportion
            if not 0.0 <= unseen_hint_proportion < 1.0:
                raise ValueError("unseen_hint_proportion must be between 0.0 and 1.0")
            holdout_format_ids = self.holdout_format_ids

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
                for r in range(1, len(all_fids)):
                    all_combinations.extend(combinations(all_fids, r))

                scored_combinations = []
                for combo in all_combinations:
                    combo_tags = set().union(*(format_stats[fid]['tags'] for fid in combo))
                    combo_complexities = set().union(*(format_stats[fid]['complexities'] for fid in combo))
                    combo_proportion = sum(format_stats[fid]['proportion'] for fid in combo)
                    train_fids = set(all_fids) - set(combo)
                    train_tags = set().union(*(format_stats[fid]['tags'] for fid in train_fids))
                    
                    leaked_tags = combo_tags - train_tags
                    leakage_penalty = len(leaked_tags) * 1.0
                    size_score = abs(combo_proportion - test_size)
                    diversity_penalty = (n_complexity_bins - len(combo_complexities)) * 0.01
                    final_score = leakage_penalty + size_score + diversity_penalty
                    scored_combinations.append((combo, final_score, leaked_tags))

                if not scored_combinations:
                    raise RuntimeError("Could not generate any format combinations for holdout.")

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

            # 5.5 Handle "Hint" Leakage
            if unseen_hint_proportion > 0.0 and unseen_test_data:
                print(f"\nAttempting to leak a 'hint' of {unseen_hint_proportion:.1%} from the holdout set...")
                hint_candidates = sorted([ex for ex in unseen_test_data if ex['complexity'] == 0], key=lambda x: id(x))
                if not hint_candidates:
                    print("  > WARNING: No low-complexity (bin 0) examples found in holdout set. Cannot leak a hint.")
                else:
                    n_to_leak = int(round(unseen_hint_proportion * len(unseen_test_data)))
                    n_to_leak = min(n_to_leak, len(hint_candidates))
                    hint_examples_for_train = hint_candidates[:n_to_leak]
                    leaked_ids = {id(ex) for ex in hint_examples_for_train}
                    unseen_test_data = [ex for ex in unseen_test_data if id(ex) not in leaked_ids]
                    remaining_data.extend(hint_examples_for_train) # Add hints to the pool to be split
                    print(f"  > Leaked {len(hint_examples_for_train)} low-complexity examples into the training pool.")
                    print(f"  > Holdout set size reduced to {len(unseen_test_data)}.")

            # 6. Perform Stratified Split on Remainder
            n_target_test = int(round(test_size * len(all_examples)))
            n_needed_from_rem = max(0, n_target_test - len(unseen_test_data))
            
            test_size_adj = n_needed_from_rem / len(remaining_data) if remaining_data else 0
            train_rem, test_rem = [], []
            if remaining_data and test_size_adj > 0 and test_size_adj < 1.0:
                print(f"Splitting remaining {len(remaining_data)} examples...")
                strata = self._get_strata(remaining_data)
                try:
                    train_rem, test_rem = train_test_split(
                        remaining_data, test_size=test_size_adj, stratify=strata, random_state=random_state
                    )
                except ValueError:
                    print("  > WARNING: Stratified split failed. Falling back to non-stratified split.")
                    train_rem, test_rem = train_test_split(
                        remaining_data, test_size=test_size_adj, random_state=random_state
                    )
            elif test_size_adj >= 1.0:
                test_rem = remaining_data
            else:
                train_rem = remaining_data
                print("No additional examples needed from remaining data for the test set.")
            
            # 7. Combine and Finalize
            train_data = train_rem
            test_data = test_rem + unseen_test_data

        else:
            # --- PATH B: New fallback logic for data without format_id ---
            print("\n'format_id' not found. Falling back to standard stratified split based on complexity.")
            total_examples = len(all_examples)
            train_data, test_data = [], []

            if test_size > 0 and test_size < 1.0:
                strata = self._get_strata(all_examples)
                try:
                    train_data, test_data = train_test_split(
                        all_examples,
                        test_size=test_size,
                        stratify=strata,
                        random_state=random_state
                    )
                except ValueError:
                    print("  > WARNING: Stratified split failed (likely due to small stratum size). Falling back to non-stratified split.")
                    train_data, test_data = train_test_split(
                        all_examples,
                        test_size=test_size,
                        random_state=random_state
                    )
            elif test_size >= 1.0:
                test_data = all_examples
            else: # test_size <= 0
                train_data = all_examples

        # 8. Report and Save (Common to both paths)
        total_examples = len(all_examples)
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

    # Helper method to determine stratification strategy
    def _get_strata(self, data: list) -> list:
        if not data:
            return []
        
        tags_are_present = 'tags' in data[0] and data[0]['tags']
        if tags_are_present:
            print("  > Applying composite stratification using 'complexity' and 'tags'.")
            return [
                f"{ex['complexity']}_{'-'.join(sorted(ex.get('tags', [])))}"
                for ex in data
            ]
        else:
            print("  > Applying stratification using 'complexity' only.")
            return [ex['complexity'] for ex in data]


    

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
            eval_strategy='epoch',          # Keep evaluating every epoch
            save_strategy='epoch',          # Keep saving every epoch
            logging_dir='./logs', logging_steps=50,
            load_best_model_at_end=True,    # This is key!
            report_to='none',
            
            # --- THE CRITICAL CHANGES ---
            predict_with_generate=False,    # <<<<<<< 1. DO NOT generate during training evaluations
            metric_for_best_model='loss',   # <<<<<<< 2. Use eval loss to find the best model
            greater_is_better=False,        # <<<<<<< 3. For loss, lower is better
            # ----------------------------

            save_total_limit=1
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=tok_train,
            eval_dataset=tok_eval,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            # compute_metrics=lambda p: compute_metrics(p, self.tokenizer)
        )

        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # post-hoc evaluation
        metrics = self.evaluate(tok_test_ds=tok_eval)
        self.model.eval()
        return metrics

    def infer(self, input_text: str) -> str:
        inputs = self.tokenizer(
            input_text, return_tensors='pt', max_length=512,
            truncation=True
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'], max_new_tokens=256,
                num_beams=1, early_stopping=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def infer_batch(self, input_texts: List[str], 
                    max_length: int = 256,
                    num_beams: int = 1) -> List[str]:
        """
        Performs generation on a batch of input strings.
        """
        # tokenize all inputs at once (padding to the longest in the batch)
        encodings = self.tokenizer(
            input_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.model.device)

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


    def evaluate(self, tok_test_ds: Dataset):
        self.model.eval()
        args = Seq2SeqTrainingArguments(
            output_dir='./tmp_eval', report_to='none',
            predict_with_generate=True,
            per_device_eval_batch_size=2,
            generation_num_beams=3,
            generation_max_length=512,
            # batch_eval_metrics='exact_match'
        )
        trainer = Seq2SeqTrainer(
            model=self.model, args=args,
            tokenizer=self.tokenizer,
            compute_metrics=lambda p: compute_metrics(p, self.tokenizer)
        )
        preds = trainer.predict(tok_test_ds)
        print(f"Optimized evaluation EM: {preds.metrics}")
        return preds.metrics