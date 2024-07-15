THREADS=3

python postprocessing_parallel_single.py \
    -f $5'_fold_'$3 \
    -p $1 -v $2 -m $4 -t $THREADS
                
folder_name='results/postprocessing_'$2'_'$1'_'$4'_'$5'_fold_'$3
mkdir -p 'results/postprocessing_single'
mv $folder_name 'results/postprocessing_single'