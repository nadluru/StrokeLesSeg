# Stroke Lesion Segmentation on Anatomical Tracings of Lesions After Stroke Segmentation (ATLAS) V2.0 Dataset
This repository is based on the [submission](https://github.com/King-HAW/ATLAS-R2-Docker-Submission) by Team CTRL in the 2022 MICCAI ATLAS Challenge.

This repository includes the implementation of the following papers:

**[MLCN 2024] Segmenting Small Stroke Lesions with Novel Labeling Strategies (MSLDBL) [[paper](https://link.springer.com/chapter/10.1007/978-3-031-78761-4_11)] [[arxiv](https://arxiv.org/abs/2408.02929)]**

**[ISBI 2025] Stroke Lesion Segmentation using Multi-Stage Cross-Scale Attention (MSCSA) [paper] [[arxiv](https://arxiv.org/abs/2501.15423)]**

## Docker Environment

The code is designed to run in a Docker environment. To get started, create the Docker image using the following command:

```
bash build.sh
```

## Environment Variables

Our code customizes several default environment variables from nnUNet. Ensure to set the following environment variables in each bash script you create:

```
export nnUNet_raw_data_base='atlasv2/raw/'
export nnUNet_preprocessed='atlasv2/preprocessed'
export RESULTS_FOLDER='atlasv2/results/'
```

## Dataset Preparation

### Default Dataset

To create the default dataset from the unzipped ATLAS v2.0 dataset, use the following command in the project root directory. Prior to running it, update the input path in `convert.py` as necessary:

```
mkdir -p atlasv2/raw/nnUNet_raw_data/
python convert.py
```

### Multi-Size Labeling (MSL) Dataset

After the default dataset is created. The MSL dataset can be generated using the following command in the project root directory:

```
cp -r atlasv2/raw/nnUNet_raw_data/Task100_ATLAS_v2/ atlasv2/raw/nnUNet_raw_data/Task104_ATLAS_v2_Multilabel/
cp -r atlasv2/raw/nnUNet_raw_data/Task104_ATLAS_v2_Multilabel/labelsTr/ atlasv2/raw/nnUNet_raw_data/Task104_ATLAS_v2_Multilabel/labelsTr_2/
python multilabel.py
```

### Distance-Based Labeling (DBL) Dataset

To generate the DBL dataset after creating the default dataset, use the following commands in the project root directory:

```
cp -r atlasv2/raw/nnUNet_raw_data/Task100_ATLAS_v2/ atlasv2/raw/nnUNet_raw_data/Task110_ATLAS_v2_TwoDistance/
cp -r atlasv2/raw/nnUNet_raw_data/Task110_ATLAS_v2_TwoDistance/labelsTr/ atlasv2/raw/nnUNet_raw_data/Task110_ATLAS_v2_TwoDistance/labelsTr_2/
python distance_transform.py
```


## Training

### Experiment Planning

Before starting training, some pre-processing is required. Please run the following command for experiment planning in the project root directory:

```
bash plan.sh
```

Ensure to update the `TASK_ID` in `plan.sh`. The `TASK_ID` specifies the dataset to be preprocessed. Specifically, `100`, `104`, and `110` correspond to the default, MSL, and DBL datasets, respectively. Additionally, for preprocessing for Res U-Net schemes, update the last command in `plan.sh` with the following:

```
nnUNet_plan_and_preprocess -t TASK_ID --verify_dataset_integrity -pl3d ExperimentPlanner3DFabiansResUNet_v21 -pl2d None
```

After completing experiment planning, please copy the split configuration of the size-balanced 5-fold cross-validation to the preprocessed dataset folder using the following command in the project root directory:

```
cp splits_final.pkl atlasv2/preprocessed/TASK/
```

The candidates of the 'TASK' folder are listed below:

#### Task candidates:

Dataset  | Task ID | Task Folder
---- | ----- | ----- 
Default  | 100 | `Task100_ATLAS_v2` 
MSL  | 104 | `Task104_ATLAS_v2_Multilabel` 
DBL  | 110 | `Task110_ATLAS_v2_TwoDistance` 


### Model Training

For model training, run the following command  in the project root directory:

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

### Self-Training Scheme

#### Generate Predictions and Create Self-Training Dataset:

For the experiments in the MSCSA paper, an additional Self-Training scheme is conducted using 300 hidden MRIs and pseudo-masks generated from the MSCSA models of Default, DTK10, and Res U-Net schemes. To generate the Self-Training Dataset, follow these steps in the project root directory after completing the 5-fold cross-validation training of the Default, DTK10, and Res U-Net schemes on the default dataset:

```
bash predict.sh
cp -r atlasv2/raw/nnUNet_raw_data/Task100_ATLAS_v2/ atlasv2/raw/nnUNet_raw_data/Task103_ATLAS_v2_Self_Training/
cp predictions/ensemble/*.gz atlasv2/raw/nnUNet_raw_data/Task103_ATLAS_v2_Self_Training/labelsTr/
cp atlasv2/raw/nnUNet_raw_data/Task103_ATLAS_v2_Self_Training/imagesTs/*.gz atlasv2/raw/nnUNet_raw_data/Task103_ATLAS_v2_Self_Training/imagesTr/
```

This will prepare the dataset for the Self-Training scheme with a Task ID of 103 and a Task Folder of Task103_ATLAS_v2_Self_Training.

#### Experiment Planning:

Next, perform the experiment planning for the Self-Training dataset. Since the split configuration for the size-balanced 5-fold cross-validation will be different for this scheme, update it using the following command in the project root directory:

```
cp splits_final_2.pkl atlasv2/preprocessed/Task103_ATLAS_v2_Self_Training/splits_final.pkl
```

#### Training the Self-Training Scheme:

Finally, run the Self-Training scheme by executing `bash train.sh` again and setting `nnUNetTrainerV2` as the trainer.

## Ensemble and Post-processing for MSLDBL

### Collect Predictions:

First, gather all predictions into the `results` folder using the following command in the project root directory:

```
bash cp.sh
```

### Generate Pure MSL and DBL Ensemble Results:

Generate ensemble results for pure MSL and DBL predictions using the command below:

```
bash ensemble_multilabel.py
```

### Install Required Packages:

Before running the final ensemble and post-processing steps, install the `parallel` package:

```
apt-get update; apt-get install -y parallel
```

### Run Post-processing

To perform post-processing and generate final evaluation results for pure MSL, pure DBL, pure MSL ensemble, and pure DBL ensemble, use:

```
bash run_postprocessing_2.sh
```

### Final Ensemble and Post-processing

The final ensemble, post-processing, and evaluation results are generated by:

```
bash run_postprocessing.sh
```

## Post-processing for MSCSA

### Collect MSCSA Predictions

Gather all predictions into the `results_mscsa folder` using the following command:

```
bash cp_mscsa.sh
```

### Run MSCSA Evaluation

Execute the evaluation script for MSCSA results:

```
python run_parallel.py -f results_mscsa
```


## Results Gathering

### Gather Results for Entire Dataset:

To compile results for the entire dataset, run:

```
python gather_postprocessing.py -f FOLDER
```

Here, `FOLDER` specifies the directory containing the `.txt` evaluation results for each method.

### Gather Results for Small Lesion Subset:

To compile results for all schemes on the small lesion subset, use:

```
python gather_small.py -f FOLDER
```


## Citation
If the code is useful for your research, please consider citing our paper:
```bibtex
@inproceedings{shang2025segmenting,
  title={Segmenting small stroke lesions with novel labeling strategies},
  author={Shang, Liang and Lou, Zhengyang and Alexander, Andrew L and Prabhakaran, Vivek and Sethares, William A and Nair, Veena A and Adluru, Nagesh},
  booktitle={International Workshop on Machine Learning in Clinical Neuroimaging},
  pages={113--122},
  year={2025},
  organization={Springer}
}
@article{shang2025stroke,
  title={Stroke Lesion Segmentation using Multi-Stage Cross-Scale Attention},
  author={Shang, Liang and Sethares, William A and Adluru, Anusha and Alexander, Andrew L and Prabhakaran, Vivek and Nair, Veena A and Adluru, Nagesh},
  journal={arXiv preprint arXiv:2501.15423},
  year={2025}
}
```

## Acknowledgments
We thank the Applied Computer Vision Lab (ACVL) for developing and maintaining [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), challenge organization team for releasing [ATLAS R2.0 Dataset](https://atlas.grand-challenge.org/ATLAS/), and Team CTRL for releasing their code for [Docker submission](https://github.com/King-HAW/ATLAS-R2-Docker-Submission).
