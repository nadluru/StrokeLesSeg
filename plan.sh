export nnUNet_raw_data_base='atlasv2/raw/'
export nnUNet_preprocessed='atlasv2/preprocessed'
export RESULTS_FOLDER='atlasv2/results/'

nnUNet_plan_and_preprocess -t TASK_ID --verify_dataset_integrity
