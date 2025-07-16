from model_setup import NLPCoder

def train_and_check(dataset_path: str, model_save_dir: str, threshold: float = 0.8, 
                    model_name: str = "Salesforce/codet5-base", model_base_dir = None, train_epochs=4, batch_size=1) -> bool:
    """
    Wrapper that fine-tunes and evaluates the model. Returns True if exact_match < threshold.
    """
    if model_base_dir is None:
        model_base_dir = model_name
    coder = NLPCoder(model_identifier=model_base_dir,
                     dataset_path=dataset_path,
                     output_dir=model_save_dir,
                     load_fine_tuned=False)

    results = coder.fine_tune(train_epochs = train_epochs, batch_size = batch_size)

    print(results)

    return results['eval_exact_match']> threshold


if __name__ == "__main__":

    train_and_check(dataset_path='/Users/maadi5/NLP_code_gen/input_dataset_v1.5.json', 
                    model_save_dir='/Users/maadi5/NLP_code_gen/format_conversion1_5',
                    # model_base_dir='/Users/maadi5/NLP_code_gen/format_conversion_weights',
                    train_epochs=5,
                    batch_size=1,
                    threshold=0.9)