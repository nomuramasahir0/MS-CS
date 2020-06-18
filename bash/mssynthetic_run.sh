#!/bin/bash -exu

# trap signal
trap "exit" INT

B=50
NUM_DATA=1000
SOURCE_MU_BOUNDS=(1.0 2.0 3.0 4.0 5.0)
DATASET=MSSynthetic1D
END_POINT=api.py
SEED_END=3000

for SOURCE_MU_BOUND in ${SOURCE_MU_BOUNDS[@]}; do
    for seed in $(seq 1 100 ${SEED_END}); do
        for est in naive upper unbiased vr
        do
            python api.py ${DATASET} --seed=${seed} \
            --estimator-type=${est} \
            --optimizer=GPUCB \
            --B=${B} \
            --num_data=${NUM_DATA} \
            --num_sources=2 \
            --target_mu_bound=1.0 \
            --source_mu_bound=${SOURCE_MU_BOUND} \
            --tuning_path=${SOURCE_MU_BOUND}
        done
    done
done
