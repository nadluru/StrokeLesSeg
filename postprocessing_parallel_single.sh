THREADS=3

python postprocessing_parallel_single_ms.py \
    -f $5'_fold_'$3 \
    -p $1 -v $2 -m $4 -t $THREADS
                
folder_name='test/postprocessing_'$2'_'$1'_'$5'_'$3
file_name=$method'_'$ref'_fold_'$4
mkdir -p '/src/workspace/ATLAS3D/results/postprocessing_single'
mv $folder_name'_'$file_name '/src/workspace/ATLAS3D/results/postprocessing_single'