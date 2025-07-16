import json
import torch
from datasets import Dataset
from src.model_setup import NLPCoder # Import your NLPCoder class

def run_debug_evaluation():
    # Configuration - ensure these match your fine_tune setup
    model_base_name = "Salesforce/codet5-base"
    dataset_file_path = "simple_transform_dataset_shrunk.json"
    output_model_directory = f"./fine_tuned_models/{model_base_name.replace('/', '_')}"
    print("loading weights from: ", output_model_directory)
    use_context_toggle = False # IMPORTANT: Match how the model was trained!

    # 1. Initialize NLPCoder to load the fine-tuned model
    print(f"Initializing NLPCoder for evaluation from path: {output_model_directory}")
    nlp_coder = NLPCoder(
        model_identifier=output_model_directory,
        dataset_path=dataset_file_path, # Added dataset_path here
        use_context_as_prefix=use_context_toggle,
        load_fine_tuned=True # Load the already fine-tuned model
    )

    # 2. Load and tokenize the full test dataset for evaluation
    print("Loading and preparing full test dataset for evaluation...")
    _, independent_test_dataset_raw = nlp_coder._load_and_prepare_data() # Use NLPCoder's data loading
    tokenized_independent_test_dataset = independent_test_dataset_raw.map(
        nlp_coder._tokenize_function,
        batched=True,
        remove_columns=independent_test_dataset_raw.column_names
    )

    # 3. Reduce the size of the test dataset for standalone evaluation to avoid OOM
    # The exact_match metric is sensitive to the full dataset, but this allows debugging
    # without crashing. You can adjust this number based on your memory capacity.
    # eval_subset_size = 50 # Evaluate on the first 50 examples
    # tokenized_test_subset = tokenized_independent_test_dataset.select(range(min(eval_subset_size, len(tokenized_independent_test_dataset))))

    # 4. Run the evaluation using the NLPCoder's evaluate method
    print(f"Starting independent evaluation with debug prints on {len(tokenized_independent_test_dataset)} examples...")
    metrics = nlp_coder.evaluate(tokenized_independent_test_dataset) # Pass the reduced subset
    print(f"Final evaluation metrics: {metrics}")

if __name__ == "__main__":
    run_debug_evaluation() 