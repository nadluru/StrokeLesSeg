THREADS=5
folders="0 1 2 3 4"


for folder in $folders
do
    folder_name='results/ensemble_t104_def+foc'
    mkdir -p $folder_name'_fold_'$folder
    python ensemble_predictions_multilabel.py --npz -t $THREADS -o $folder_name'_fold_'$folder \
        -f results/t104_def_fold_$folder \
        results/t104_foc_fold_$folder
done

for folder in $folders
do
    folder_name='results/ensemble_t110_def+foc'
    mkdir -p $folder_name'_fold_'$folder
    python ensemble_predictions_multilabel.py --npz -t $THREADS -o $folder_name'_fold_'$folder \
        -f results/t110_def_fold_$folder \
        results/t110_foc_fold_$folder
done
