export nnUNet_raw_data_base='atlasv2/raw/'
export nnUNet_preprocessed='/opt/algorithm/preprocessed'
export RESULTS_FOLDER='atlasv2/results/'

nnUNet_train 3d_fullres TRAINER TASK FOLD --npz
