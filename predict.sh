#!/usr/bin/env bash
# SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

export nnUNet_raw_data_base='atlasv2/raw'
export nnUNet_preprocessed='preprocessed'
export RESULTS_FOLDER='atlasv2/results/'
export CUDA_VISIBLE_DEVICES=7

NNUNet_FOLDER="nnUNet/nnunet/"
NNUNet_DATA_FOLDER="atlasv2/raw/nnUNet_raw_data/Task100_ATLAS_v2/imagesTs"
TASK100_NAME="Task100_ATLAS_v2"
TASK103_NAME="Task103_ATLAS_v2_Self_Training"
THREADS=16

mkdir -p predictions

mkdir -p predictions/t100_def_fold0 predictions/t100_def_fold1 predictions/t100_def_fold2 predictions/t100_def_fold3 predictions/t100_def_fold4
mkdir -p predictions/t100_dtk_fold0 predictions/t100_dtk_fold1 predictions/t100_dtk_fold2 predictions/t100_dtk_fold3 predictions/t100_dtk_fold4
mkdir -p predictions/t100_res_fold0 predictions/t100_res_fold1 predictions/t100_res_fold2 predictions/t100_res_fold3 predictions/t100_res_fold4
# mkdir -p predictions/t100_slf_fold0 predictions/t100_slf_fold1 predictions/t100_slf_fold2 predictions/t100_slf_fold3 predictions/t100_slf_fold4

mkdir -p predictions/ensemble

# default model
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o predictions/t100_def_fold0 -tr nnUNetTrainerV2_MSCSA_Depth_1_SGD --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 0 -z 
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o predictions/t100_def_fold1 -tr nnUNetTrainerV2_MSCSA_Depth_1_SGD --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 1 -z 
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o predictions/t100_def_fold2 -tr nnUNetTrainerV2_MSCSA_Depth_1_SGD --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 2 -z 
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o predictions/t100_def_fold3 -tr nnUNetTrainerV2_MSCSA_Depth_1_SGD --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 3 -z 
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o predictions/t100_def_fold4 -tr nnUNetTrainerV2_MSCSA_Depth_1_SGD --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 4 -z 

# dtk10 model
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o predictions/t100_dtk_fold0 -tr nnUNetTrainerV2_800epochs_Loss_DiceTopK10_MSCSA_Depth_1_SGD --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 0 -z 
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o predictions/t100_dtk_fold1 -tr nnUNetTrainerV2_800epochs_Loss_DiceTopK10_MSCSA_Depth_1_SGD --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 1 -z 
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o predictions/t100_dtk_fold2 -tr nnUNetTrainerV2_800epochs_Loss_DiceTopK10_MSCSA_Depth_1_SGD --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 2 -z 
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o predictions/t100_dtk_fold3 -tr nnUNetTrainerV2_800epochs_Loss_DiceTopK10_MSCSA_Depth_1_SGD --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 3 -z 
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o predictions/t100_dtk_fold4 -tr nnUNetTrainerV2_800epochs_Loss_DiceTopK10_MSCSA_Depth_1_SGD --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 4 -z 

# res u-net model
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_res_fold0 -tr nnUNetTrainerV2_ResencUNet_MSCSA_Depth_1_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 0 -z
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_res_fold1 -tr nnUNetTrainerV2_ResencUNet_MSCSA_Depth_1_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 1 -z
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_res_fold2 -tr nnUNetTrainerV2_ResencUNet_MSCSA_Depth_1_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 2 -z
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_res_fold3 -tr nnUNetTrainerV2_ResencUNet_MSCSA_Depth_1_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 3 -z
python $NNUNet_FOLDER"inference/predict_simple.py" -i $NNUNet_DATA_FOLDER -o t100_res_fold4 -tr nnUNetTrainerV2_ResencUNet_MSCSA_Depth_1_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $TASK100_NAME -m 3d_fullres -f 4 -z

python ensemble_predictions.py --npz -t $THREADS -o predictions/ensemble \
    -f t153_def_fold0 t153_def_fold1 t153_def_fold2 t153_def_fold3 t153_def_fold4 \
    t153_foc_fold0 t153_foc_fold1 t153_foc_fold2 t153_foc_fold3 t153_foc_fold4 \
    t153_res_fold0 t153_res_fold1 t153_res_fold2 t153_res_fold3 t153_res_fold4 \
    t173_def_fold0 t173_def_fold1 t173_def_fold2 t173_def_fold3 t173_def_fold4

mkdir -p ensemble_ms/predictions_157
python ensemble_predictions_ms.py --npz -t $THREADS -o ensemble_ms/predictions_157 \
    -f predictions/t100_def_fold0 predictions/t100_def_fold1 predictions/t100_def_fold2 predictions/t100_def_fold3 predictions/t100_def_fold4 \
    predictions/t100_dtk_fold0 predictions/t100_dtk_fold1 predictions/t100_dtk_fold2 predictions/t100_dtk_fold3 predictions/t100_dtk_fold4 \
    predictions/t100_res_fold0 predictions/t100_res_fold1 predictions/t100_res_fold2 predictions/t100_res_fold3 predictions/t100_res_fold4

