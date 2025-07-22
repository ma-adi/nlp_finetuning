from model_setup_new import NLPCoder

def train_and_check(dataset_path: str, model_save_dir: str, threshold: float = 0.8, 
                    model_name: str = "Salesforce/codet5p-220m", model_base_dir = None, train_epochs=4, batch_size=1,
                    heldout_format_ids = None, unseen_hint_proportion: float = 0.0, epoch_weights = None) -> bool:
    """
    Wrapper that fine-tunes and evaluates the model. Returns True if exact_match < threshold.
    """
    if model_base_dir is None:
        model_base_dir = model_name
    coder = NLPCoder(model_identifier=model_base_dir,
                     dataset_path=dataset_path,
                     output_dir=model_save_dir,
                     load_fine_tuned=False,
                     holdout_format_ids=heldout_format_ids,
                     unseen_hint_proportion=unseen_hint_proportion)

    results = coder.fine_tune(train_epochs = train_epochs, batch_size = batch_size) #,
                              #epoch_weights = epoch_weights)

    print(results)

    return results['eval_exact_match']> threshold


if __name__ == "__main__":

    train_and_check(dataset_path='/Users/maadi5/nlp_finetuning/dataset_utils/tagged_dataset_exp_master_curriculum_3000_uuid_randomindent.json', 
                    model_save_dir='/Users/maadi5/nlp_finetuning/master_curriculum_3000_random_indent_weights_hint0.3_curriculum',
                    # model_base_dir='/Users/maadi5/NLP_code_gen/format_conversion_weights',
                    train_epochs=16,
                    batch_size=1,
                    threshold=0.9,
                    heldout_format_ids=['ListOfLists', 'DeeplyNested', 'ListOfNamespacedEntities'],
                    unseen_hint_proportion = 0.3)#,
                    # epoch_weights=[5, 6, 5])