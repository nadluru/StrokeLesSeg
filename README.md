# Stroke Lesion Segmentation on Anatomical Tracings of Lesions After Stroke Segmentation (ATLAS) V2.0 Dataset
This repository is based on the [submission](https://github.com/King-HAW/ATLAS-R2-Docker-Submission) by Team CTRL in the 2022 MICCAI ATLAS Challenge.

This repository includes the implementation of the following papers:
Segmenting Small Stroke Lesions with Novel Labeling Strategies
Stroke Lesion Segmentation using Multi-Stage Cross-Scale Attention

## Docker Environment

The code is designed to run in a Docker environment. To get started, create the Docker image using the following command:

```
bash build.sh
```

## Environment Variables

Our code customizes several default environment variables from nnUNet. Ensure to set the following environment variables in each bash script you create:

```
export nnUNet_raw_data_base='atlasv2/raw/'
export nnUNet_preprocessed='/opt/algorithm/preprocessed'
export RESULTS_FOLDER='atlasv2/results/'
```

## Dataset Preparation

### Default Dataset

To create the default dataset from the unzipped ATLAS v2.0 dataset, use the following command. Prior to running it, update the input path in `convert.py` as necessary:

```
python convert.py
```

### Multi-Size Labeling (MSL) Dataset

After the default dataset is created. The MSL dataset can be generated using the following command:

```
cp -r atlasv2/raw/nnUNet_raw_data/Task100_ATLAS_v2/ atlasv2/raw/nnUNet_raw_data/Task104_ATLAS_v2_Multilabel/
cp -r atlasv2/raw/nnUNet_raw_data/Task104_ATLAS_v2_Multilabel/labelsTr/ atlasv2/raw/nnUNet_raw_data/Task104_ATLAS_v2_Multilabel/labelsTr_2/
python multilabel.py
```

### Distance-Based Labeling (DBL) Dataset

To generate the DBL dataset after creating the default dataset, use the following commands:

```
cp -r atlasv2/raw/nnUNet_raw_data/Task100_ATLAS_v2/ atlasv2/raw/nnUNet_raw_data/Task110_ATLAS_v2_TwoDistance/
cp -r atlasv2/raw/nnUNet_raw_data/Task110_ATLAS_v2_TwoDistance/labelsTr/ atlasv2/raw/nnUNet_raw_data/Task110_ATLAS_v2_TwoDistance/labelsTr_2/
python distance_transform.py
```


## Training

### Experiment Planning

Before starting training, some pre-processing is required. Please run the following command for experiment planning:

```
bash plan.sh
```

Ensure to update the `TASK_ID` in `plan.sh`. The `TASK_ID` specifies the dataset to be preprocessed. Specifically, `100`, `104`, and `110` correspond to the default, MSL, and DBL datasets, respectively. Additionally, for preprocessing for Res U-Net schemes, update the last command in `plan.sh` with the following:

```
nnUNet_plan_and_preprocess -t TASK_ID --verify_dataset_integrity -pl3d ExperimentPlanner3DFabiansResUNet_v21 -pl2d None
```

After completing experiment planning, please copy the split configuration of the size-balanced 5-fold cross-validation to the preprocessed dataset folder using the following command:

```
cp splits_final.pkl /opt/algorithm/preprocessed/TASK/
```

The candidates of the 'TASK' folder are listed below:

#### Task candidates:

Dataset  | Task ID | Task
---- | ----- | ----- 
Default  | 100 | `Task100_ATLAS_v2` 
MSL  | 104 | `Task104_ATLAS_v2_Multilabel` 
DBL  | 110 | `Task110_ATLAS_v2_TwoDistance` 


### Model Training

For model training, run the following command:

```
bash train.sh
```

Ensure to update the `TRAINER`, `TASK`, and `Fold` in `train.sh`. The `Fold` can be 0 to 5 for each run of 5-fold cross-validation. The candidates for trainers are listed below:

#### Trainer candidates for Baselines and MSCSA models:

 Scheme  | Baseline Trainer  | MSCSA Trainer
 ---- | ----- | ------ 
 Default  | `nnUNetTrainerV2` | `nnUNetTrainerV2_MSCSA_Depth_1_SGD`
 Focal  | `nnUNetTrainerV2_Focal` | `nnUNetTrainerV2_MSCSA_Depth_1_Focal`
 DTK10  | `nnUNetTrainerV2_800epochs_Loss_DiceTopK10` | `nnUNetTrainerV2_800epochs_Loss_DiceTopK10_MSCSA_Depth_1_SGD`
 Res U\-Net | `nnUNetTrainerV2_ResencUNet_DA3` | `nnUNetTrainerV2_ResencUNet_MSCSA_Depth_1_DA3`

Additionally, for Res U-Net schemes, update the last command in `train.sh` with the following:

```
nnUNet_train 3d_fullres TRAINER TASK 0 --npz -p nnUNetPlans_FabiansResUNet_v2.1
```

###



## Citation
If the code is useful for your research, please consider citing our paper:
```bibtex

```

## Acknowledgments
We thank the Applied Computer Vision Lab (ACVL) for developing and maintaining [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), and challenge organization team for releasing [ATLAS R2.0 Dataset](https://atlas.grand-challenge.org/ATLAS/).
