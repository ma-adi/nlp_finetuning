from model_setup_new import NLPCoder
import json

train_dataset = '/Users/maadi5/nlp_finetuning/dataset_utils/tagged_dataset_exp_easy_master_curriculum_3_2400_uuid_none_indent_v2_emptylistdict.json'

test_dataset = '/Users/maadi5/nlp_finetuning/dataset_utils/tagged_dataset_exp_easy_master_curriculum_3_800_uuid_none_indent_v2_emptylistdict.json'

model_path = '/Users/maadi5/nlp_finetuning/easy_master_curriculum_3_3000_weights_allformatstrain_bestmodel_fixed_noneindent_v2_emptylistdict'

model_inference = NLPCoder(
    model_identifier=model_path,
    load_fine_tuned=True
)

_ , test_ds= model_inference._load_and_prepare_data(train_json=train_dataset, test_json=test_dataset)

mismatches = model_inference.eval_mismatch(test_dataset = test_ds, batch_size = 16, num_beams = 1)

json.dump(mismatches, open('mismatched_predictions_allformatstrain.json', 'w', encoding='utf8'), ensure_ascii=False)