THREADS=3
ref='ensemble_t110_def+foc'
# method='ensemble_t104_def+foc2'
method='ensemble_t104_def+foc'

python postprocessing_parallel.py \
    -f $method'_fold_'$5 \
    -r $ref'_fold_'$5 \
    -p $2 -v $3 -x $4 -m $1 -t $THREADS
                
folder_name='results/postprocessing_'$3'_'$2'_'$1'_'$4
file_name=$method'_'$ref'_fold_'$5
mkdir -p '/src/workspace/ATLAS3D/results/postprocessing'
mv $folder_name'_'$file_name '/src/workspace/ATLAS3D/results/postprocessing'