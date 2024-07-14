folders="0 1 2 3 4"
tasks=('Task100_ATLAS_v2')
outputs=('results_mscsa/t100')

mkdir -p 'results_mscsa'

for ((num=0;num<${#tasks[*]};num++))
do

    for folder in $folders
    do
        mkdir ${outputs[num]}'_def'_fold_$folder
        mkdir ${outputs[num]}'_dtk'_fold_$folder
        mkdir ${outputs[num]}'_res'_fold_$folder
        cp ../atlasv2/results/nnUNet/3d_fullres/${tasks[num]}/nnUNetTrainerV2_MSCSA_Depth_1_SGD__nnUNetPlansv2.1/fold_$folder/validation_raw/* ${outputs[num]}'_def'_fold_$folder
        cp ../atlasv2/results/nnUNet/3d_fullres/${tasks[num]}/nnUNetTrainerV2_800epochs_Loss_DiceTopK10_MSCSA_Depth_1_SGD__nnUNetPlansv2.1/fold_$folder/validation_raw/* ${outputs[num]}'_dtk'_fold_$folder
        cp ../atlasv2/results/nnUNet/3d_fullres/${tasks[num]}/nnUNetTrainerV2_ResencUNet_MSCSA_Depth_1_DA3__nnUNetPlans_FabiansResUNet_v2.1/fold_$folder/validation_raw/* ${outputs[num]}'_res'_fold_$folder
    done

done

tasks=('Task103_ATLAS_v2_Self_Training')
outputs=('results_mscsa/t100')

for ((num=0;num<${#tasks[*]};num++))
do

    for folder in $folders
    do
        mkdir ${outputs[num]}'_slf'_fold_$folder
        cp ../atlasv2/results/nnUNet/3d_fullres/${tasks[num]}/nnUNetTrainerV2_MSCSA_Depth_1_SGD__nnUNetPlansv2.1/fold_$folder/validation_raw/* ${outputs[num]}'_slf'_fold_$folder
    done

done
