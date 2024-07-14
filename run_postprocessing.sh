parallel --jobs 3 -k --bar \
    bash postprocessing_parallel.sh \
    ::: max \
    ::: 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 \
    ::: 1000 \
    ::: 1.0 0.9 0.8 0.7 0.6 0.5 0.0 \
    ::: 0 1 2 3 4

python run_parallel.py -f results/postprocessing

