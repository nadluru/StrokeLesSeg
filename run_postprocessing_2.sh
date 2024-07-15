parallel --jobs 3 -k --bar \
    bash postprocessing_parallel_single.sh \
    ::: 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9\
    ::: 1000 \
    ::: 0 1 2 3 4 \
    ::: max \
    ::: ensemble_t104_def+foc ensemble_t110_def+foc \
    t104_def t104_foc t110_def t110_foc

python run_parallel.py -f results/postprocessing_single