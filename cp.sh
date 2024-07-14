folders="0 1 2 3 4"
tasks=('Task104_ATLAS_v2_Multilabel' 'Task110_ATLAS_v2_TwoDistance')
outputs=('results/t104' 'results/t110')

mkdir -p 'results'

# for ((num=0;num<${#tasks[*]};num++))
for ((num=0;num<${#tasks[*]};num++))
do

    for folder in $folders
    do
        mkdir ${outputs[num]}'_def'_fold_$folder
        mkdir ${outputs[num]}'_foc'_fold_$folder
        cp atlasv2/results/nnUNet/3d_fullres/${tasks[num]}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_$folder/validation_raw/* ${outputs[num]}'_def'_fold_$folder
        cp atlasv2/results/nnUNet/3d_fullres/${tasks[num]}/nnUNetTrainerV2_Focal__nnUNetPlansv2.1/fold_$folder/validation_raw/* ${outputs[num]}'_foc'_fold_$folder
    done

done
